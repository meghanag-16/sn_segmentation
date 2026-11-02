"""
nnU-Net helpers shared by training and inference: environment folders, model lookup, command runner,
subject loading on the crop grid, and the probability orientation check.
"""
import glob
import json
import os
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from .utils import full_affine, load_nii, outer_limit, read_subjects, snc_anchor, vox_size, log


def setup_environment(ctx):
    """nnU-Net reads its folders from environment variables. Raw and preprocessed data are temporary;
    nnUNet_results is the model folder (given by the user, or inside the output folder when the model is trained here)."""
    P = ctx.paths
    raw = P.work / "nnUNet_raw"
    pre = P.work / "nnUNet_preprocessed"
    results = Path(ctx.model_dir) if ctx.model_dir else P.nnunet / "nnUNet_results"
    for d in (raw, pre, results):
        d.mkdir(parents=True, exist_ok=True)
    os.environ["nnUNet_raw"] = str(raw)
    os.environ["nnUNet_preprocessed"] = str(pre)
    os.environ["nnUNet_results"] = str(results)
    os.environ["nnUNet_compile"] = "f"                 # no torch.compile (slow start, not needed for this size)
    return raw, pre, results


def results_dir(ncfg):
    """folder <nnUNet_results>/<dataset>/<trainer>__<plans>__<configuration>"""
    hits = glob.glob(os.path.join(os.environ["nnUNet_results"], f"Dataset{int(ncfg['dataset_id']):03d}_*",
                                  f"{ncfg['trainer']}__{ncfg['plans']}__{ncfg['configuration']}"))
    return Path(hits[0]) if len(hits) == 1 else None


def fold_done(ncfg, k):
    rd = results_dir(ncfg)
    return rd is not None and (rd / f"fold_{k}" / ncfg["checkpoint"]).exists()


def model_complete(ncfg):
    return all(fold_done(ncfg, k) for k in ncfg["folds"])


def run_cmd(cmd, log_path, env=None):
    """runs an nnU-Net command, output to a log file; prints the end of the log if it fails"""
    os.makedirs(os.path.dirname(str(log_path)), exist_ok=True)
    log("$ " + " ".join(cmd) + f" | log: {log_path}")
    with open(log_path, "a") as lf:
        r = subprocess.run(cmd, env=dict(env or os.environ, PYTHONUNBUFFERED="1"), stdout=lf, stderr=subprocess.STDOUT, text=True)
    if r.returncode:
        print(open(log_path, errors="ignore").read()[-4000:])
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd)}")


def orient_prob(npz_path, seg_path):
    """nnU-Net stores probabilities as (C, z, y, x); returns (C, x, y, z) matching nibabel, checked against nnU-Net's own seg."""
    z = np.load(npz_path)
    P = z["probabilities"] if "probabilities" in z else z[list(z.keys())[0]]
    seg = np.asarray(nib.load(str(seg_path)).dataobj)
    for cand in (P.transpose(0, 3, 2, 1), P):
        if cand.shape[1:] == seg.shape and (np.argmax(cand, 0) == seg).mean() > 0.999:
            return cand.astype(np.float32)
    raise RuntimeError(f"probability orientation could not be matched to {seg_path}")


class CropData:
    """per-subject data on the crop grid (no manual mask is read here)"""

    def __init__(self, ctx):
        P = ctx.paths
        self.P_CROP = P.cropped
        self.rules = ctx.cfg["segmentation_rules"]
        subj = read_subjects(P)
        qc = pd.read_csv(P.cropped / "qc_table.csv")
        cw = pd.read_csv(P.cropped / "crop_windows.csv")
        self.CHANNELS = json.load(open(P.atlas / "atlas_registration_decisions.json"))["outputs"]["channels"]
        self.iSNc, self.iSNr, self.iRN = self.CHANNELS.index("SNc"), self.CHANNELS.index("SNr"), self.CHANNELS.index("RN")
        self.ALL = (subj[["subject_id", "group", "role", "fold"]]
                    .merge(qc[["subject_id", "scale_group", "include_inference", "include_training"]], on="subject_id", how="inner")
                    .merge(cw[["subject_id", "i0", "j0", "nx", "ny", "nz"]], on="subject_id", how="left")).set_index("subject_id")

    def crop_path(self, sid, key, name):
        return self.P_CROP / "crops" / key / f"{sid}_{name}.nii.gz"

    def load_subject(self, sid):
        r = self.ALL.loc[sid]
        rr, caff = load_nii(self.crop_path(sid, "refratio", "spir_refratio"))
        d = dict(sid=sid, rr=rr, caff=caff, raw=load_nii(self.crop_path(sid, "raw", "spir_raw"))[0],
                 atl=load_nii(self.crop_path(sid, "atlas", "atlas"))[0], hemi=load_nii(self.crop_path(sid, "hemi", "hemi"), np.uint8)[0],
                 zpath=self.crop_path(sid, "zscore", "spir_zscore"), i0=int(r.i0), j0=int(r.j0), fshape=(int(r.nx), int(r.ny), int(r.nz)),
                 crus=load_nii(self.crop_path(sid, "ref_roi", "ref_crus"), np.uint8)[0])        # crus cerebri reference, 1 left / 2 right
        d["zs"] = vox_size(caff); d["vvox"] = float(np.prod(d["zs"])); d["faff"] = full_affine(caff, d["i0"], d["j0"])
        d["reg"] = outer_limit(d["atl"], d["zs"], self.rules, self.iSNc, self.iSNr, self.iRN)
        d["anc"] = snc_anchor(d["atl"], d["hemi"], self.rules, self.iSNc)
        d["snc25"], d["snr25"] = d["atl"][..., self.iSNc] >= 0.25, d["atl"][..., self.iSNr] >= 0.25
        return d
