"""
nnU-Net inference and post-processing.

- 5-fold ensemble of the frozen model (mean of the 5 fold probabilities, nnU-Net default test-time mirroring on all three axes),
  input = the z-score SPIR crop only.
- Binarisation: SN probability >= 0.5.
- Post-processing per side (CIT168 midline hemisphere map): keep voxels inside the atlas outer limit; keep 26-connected pieces
  touching the atlas SNc anchor (fallback: largest piece); fill holes in-plane.
- Output on the full 320 x 320 x 50 SPIR grid with the SPIR header: 0 background, 1 left SN, 2 right SN.
- Label-free checks: both sides present, contrast inside/shell > 1.0, L/R volume ratio 0.69-1.27, largest piece >= 0.90 per side.

Outputs:
  sn_masks/<subject_id>_sn_mask.nii.gz                     FINAL segmentation per patient on the SPIR grid
  nnunet/predictions/                                      raw nnU-Net output (probabilities .npz, argmax .nii.gz)
  nnunet/postprocessed/crop|spir/<subject>_nnunet_lr.nii.gz, nnunet/prob_crop/<subject>_sn_prob.nii.gz
  nnunet/segmentation_metrics.csv, nnunet/nm_measures.csv, nnunet/README.txt
"""
import os
import shutil

import numpy as np
import pandas as pd
from scipy import ndimage

from .nnunet_common import CropData, model_complete, orient_prob, results_dir, run_cmd, setup_environment
from .utils import S6, S26, mi, postprocess_network, save_nii, write_text, log


def metrics_free(L, d, raw_pred=None):
    """numbers that need no mask (usable on every subject)"""
    S = L > 0; v = d["vvox"]; r = {}
    r["vol_mm3"] = float(S.sum() * v)
    for side, nm in ((1, "left"), (2, "right")):
        s = L == side; r[f"vol_{nm}_mm3"] = float(s.sum() * v)
        lab, n = ndimage.label(s, structure=S26); r[f"n_pieces_{nm}"] = int(n)
        r[f"largest_share_{nm}"] = float(np.bincount(lab.ravel())[1:].max() / s.sum()) if n else np.nan
        ks = [k for k in range(s.shape[2]) if s[..., k].any()]
        r[f"slices_{nm}"] = len(ks)
        r[f"slices_multi_piece_{nm}"] = int(sum(ndimage.label(s[..., k])[1] > 1 for k in ks))
    r["lr_ratio"] = r["vol_left_mm3"] / r["vol_right_mm3"] if r["vol_right_mm3"] > 0 else np.nan
    if S.any():
        shell = ndimage.binary_dilation(S, structure=S6) & ~S & (d["hemi"] > 0)
        r["inside_median_ratio"] = float(np.median(d["rr"][S]))
        r["contrast"] = r["inside_median_ratio"] / float(np.median(d["rr"][shell])) if shell.any() else np.nan
        r["share_snc"] = float((S & d["snc25"]).sum() / S.sum())
        r["share_outside_sn_atlas"] = float((S & ~d["snc25"] & ~d["snr25"]).sum() / S.sum())
    else:
        r.update(inside_median_ratio=np.nan, contrast=np.nan, share_snc=np.nan, share_outside_sn_atlas=np.nan)
    if raw_pred is not None:
        r["raw_vol_mm3"] = float(raw_pred.sum() * v)
        r["raw_outside_outer_limit_mm3"] = float((raw_pred & ~d["reg"]).sum() * v)
        r["removed_by_postprocess_mm3"] = float((raw_pred & ~S).sum() * v)
    return r


def crus_lr(d):
    """median raw SPIR in the left crus / right crus reference ROI (image brightness left vs right)"""
    a, b = d["raw"][d["crus"] == 1], d["raw"][d["crus"] == 2]
    return float(np.median(a) / np.median(b)) if len(a) and len(b) else np.nan


def nm_measures(L, d):
    """NM signal per side in the segmentation: raw SPIR intensities and crus ratio (CNR = ratio - 1)."""
    out = {}
    for side, nm in ((1, "left"), (2, "right"), (None, "both")):
        s = (L > 0) if side is None else (L == side)
        if s.any():
            out.update({f"{nm}_vol_mm3": float(s.sum() * d["vvox"]), f"{nm}_raw_mean": float(d["raw"][s].mean()),
                        f"{nm}_raw_median": float(np.median(d["raw"][s])), f"{nm}_ratio_mean": float(d["rr"][s].mean()),
                        f"{nm}_ratio_median": float(np.median(d["rr"][s])), f"{nm}_cnr_mean": float(d["rr"][s].mean() - 1)})
        else:
            out.update({f"{nm}_vol_mm3": 0.0})
    return out


def label_free_flags(mt, G):
    lo, hi = G["lr_ratio_range"]
    return [f for f, bad in (("side empty", not (mt["vol_left_mm3"] > 0 and mt["vol_right_mm3"] > 0)),
                             ("contrast", not (mt["contrast"] > G["contrast_min_exclusive"])),
                             ("L/R ratio", not (lo <= mt["lr_ratio"] <= hi)),
                             ("fragmented", not (min(mt["largest_share_left"], mt["largest_share_right"]) >= G["largest_piece_min"])))
            if bad]


def device_flag(ctx):
    if ctx.device in ("cpu", "cuda", "mps"):
        return ctx.device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def predict(ctx, CD, sids, out_dir):
    """5-fold ensemble prediction (z-score crop input only); returns {sid: SN probability (x, y, z)}"""
    ncfg = ctx.cfg["nnunet"]
    assert model_complete(ncfg), (f"trained model {ncfg['dataset_name']} ({ncfg['trainer']}__{ncfg['plans']}__{ncfg['configuration']}, "
                                  f"folds {ncfg['folds']}, {ncfg['checkpoint']}) not found in {os.environ['nnUNet_results']}")
    inp = ctx.paths.work / "pred_in"
    inp.mkdir(parents=True, exist_ok=True)
    for s in sids:
        dst = inp / f"{s}_0000.nii.gz"
        if not dst.exists():
            shutil.copy2(CD.crop_path(s, "zscore", "spir_zscore"), dst)
    todo = [s for s in sids if not (out_dir / f"{s}.npz").exists()]
    if todo:
        out_dir.mkdir(parents=True, exist_ok=True)
        dev = device_flag(ctx)
        cmd = ["nnUNetv2_predict", "-i", str(inp), "-o", str(out_dir), "-d", str(ncfg["dataset_id"]), "-c", ncfg["configuration"],
               "-tr", ncfg["trainer"], "-p", ncfg["plans"], "-f", *[str(k) for k in ncfg["folds"]], "-chk", ncfg["checkpoint"],
               "--save_probabilities"]
        if dev != "cuda":
            cmd += ["-device", dev]
        run_cmd(cmd, out_dir / "predict.log")
    return {s: orient_prob(out_dir / f"{s}.npz", out_dir / f"{s}.nii.gz")[1] for s in sids}


def run(ctx):
    P, ncfg = ctx.paths, ctx.cfg["nnunet"]
    G = ctx.cfg["segmentation_rules"]["gate"]
    thr = float(ncfg["probability_threshold"])
    setup_environment(ctx)
    log(f"model: {results_dir(ncfg)}")
    CD = CropData(ctx)
    ALL = CD.ALL[CD.ALL.include_inference.astype(bool)]
    if ncfg["inference_subjects"] == "test":
        ALL = ALL[ALL.role == "test"]
    SIDS = list(ALL.index)
    skipped = sorted(set(CD.ALL.index) - set(SIDS))
    log(f"inference on {len(SIDS)} subjects" + (f" (not included: {skipped})" if skipped else ""))
    OUT = P.nnunet
    PT = predict(ctx, CD, SIDS, OUT / "predictions")
    suffix = ctx.cfg["output"]["mask_suffix"]
    rows, nmrows = [], []
    for s in SIDS:
        d = CD.load_subject(s)
        raw_pred = PT[s] >= thr
        L = postprocess_network(raw_pred, d)
        save_nii(L, d["caff"], OUT / "postprocessed" / "crop" / f"{s}_nnunet_lr.nii.gz", np.uint8)
        full = np.zeros(d["fshape"], np.uint8); full[d["i0"]:d["i0"] + L.shape[0], d["j0"]:d["j0"] + L.shape[1], :] = L
        save_nii(full, d["faff"], OUT / "postprocessed" / "spir" / f"{s}_nnunet_lr.nii.gz", np.uint8)
        save_nii(full, d["faff"], P.sn_masks / f"{s}{suffix}", np.uint8)                               # FINAL OUTPUT
        save_nii(PT[s], d["caff"], OUT / "prob_crop" / f"{s}_sn_prob.nii.gz", np.float32)
        mt = {**metrics_free(L, d, raw_pred), "crus_lr": crus_lr(d)}
        flags = label_free_flags(mt, G)
        rows.append(dict(subject_id=s, group=ALL.loc[s, "group"], role=ALL.loc[s, "role"], scale_group=ALL.loc[s, "scale_group"], **mt,
                         label_free_gate_pass=len(flags) == 0, flags="; ".join(flags)))
        nmrows.append(dict(subject_id=s, group=ALL.loc[s, "group"], role=ALL.loc[s, "role"], scale_group=ALL.loc[s, "scale_group"],
                           **nm_measures(L, d)))
    TM = pd.DataFrame(rows); TM.to_csv(OUT / "segmentation_metrics.csv", index=False)
    pd.DataFrame(nmrows).to_csv(OUT / "nm_measures.csv", index=False)
    log(f"{ncfg['dataset_name']} {ncfg['configuration']} on {len(SIDS)} subjects | volume {mi(TM.vol_mm3, '{:.0f}')} mm3 | "
        f"L/R {mi(TM.lr_ratio)} | contrast {mi(TM.contrast, '{:.3f}')} | label-free gate {int(TM.label_free_gate_pass.sum())}/{len(TM)}")
    if (~TM.label_free_gate_pass).any():
        log(TM[~TM.label_free_gate_pass][["subject_id", "flags"]].to_string(index=False))
    write_text(OUT / "README.txt",
               f"Inference, model {ncfg['dataset_name']} {ncfg['configuration']} ({ncfg['trainer']}, 5-fold ensemble)\n"
               f"../sn_masks/<subject>{suffix} : FINAL mask, full 320 x 320 x 50 SPIR grid (open on the original T1W_3D_SPIR.nii); 0 / 1 left / 2 right\n"
               "postprocessed/spir/ : the same; postprocessed/crop/ : same on the 112 x 80 x 50 crop grid; prob_crop/ : SN probability (crop grid)\n"
               "post-processing: per side inside the atlas outer limit, pieces touching the atlas SNc anchor, holes filled in-plane\n"
               "segmentation_metrics.csv : label-free numbers and flags; nm_measures.csv : raw SPIR and crus-ratio values per side\n"
               "Note: development subjects that were in the training set are segmented by models that saw them in training.\n")
