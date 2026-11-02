"""
GMM masks and refined training labels (development subjects with a manual mask only).

1. GMM mask: per subject, a 3-component Gaussian mixture on the crus-ratio image inside the atlas outer limit;
   bright class posterior >= 0.5; per side, pieces (26-connected) touching the atlas SNc anchor; holes filled in-plane.
   No manual mask is used.
2. Refined mask (training label): the GMM mask inside (manual mask AND atlas outer limit); on each slice and side, new
   voxels are added only in the narrow corridor between two separate pieces that are at most 2.0 mm apart ("bridging").
   The outer outline of a piece is never grown and bays are not filled; pieces not connected to the SNc anchor are dropped.
   The manual mask is only an upper bound here (it is too generous to be a target itself).

Outputs (gmm_refined/):
  seg_crop/gmm/<subject>_gmm_lr.nii.gz, seg_crop/gmm_refined/<subject>_gmm_refined_lr.nii.gz   crop grid, 0 / 1 left / 2 right
  seg_spir/gmm/..., seg_spir/gmm_refined/...                                                   full SPIR grid, SPIR header
  gmm_details.csv, bridging_counts.csv, metrics.csv, summary.csv, gmm_refinement_settings.json, README.txt
  qc_pages/qc_<n>_<subject>.png   image | manual mask | GMM | refined (bridge voxels in magenta)
"""
import datetime
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.mixture import GaussianMixture

from .config import sha_settings
from .utils import (S6, S26, auto_gate, fill_inplane, keep_anchor_pieces, load_nii, mi, outer_limit, postprocess_gmm,
                    read_subjects, snc_anchor, vox_size, write_text, log)

S4 = ndimage.generate_binary_structure(2, 1)


def bridge_slice(s, A, zs2, G, tol):
    lab, n = ndimage.label(s, structure=S4)
    if n < 2 or G <= 0:
        return s, 0
    D = [ndimage.distance_transform_edt(lab != i, sampling=zs2) for i in range(1, n + 1)]   # distance to piece i (mm)
    add = np.zeros_like(s); nb = 0
    for i in range(n):
        for j in range(i + 1, n):
            dij = float(D[j][lab == i + 1].min())
            if dij <= G:
                add |= (D[i] + D[j] <= dij + tol) & A & ~s
                nb += 1
    return s | add, nb


def joined(L, d, G, tol):
    """bridging inside (manual mask AND outer limit), per side and slice; then holes filled in-plane and anchor pieces kept"""
    out = np.zeros(L.shape, np.uint8); nbridge = 0; fb = False
    zs2 = d["zs"][:2]
    for side in (1, 2):
        A = d["mb"] & d["reg"] & (d["hemi"] == side)
        s = (L == side) & A
        if not s.any():
            continue
        for k in range(s.shape[2]):
            if s[..., k].any():
                s[..., k], nb = bridge_slice(s[..., k], A[..., k], zs2, G, tol); nbridge += nb
        s = fill_inplane(s) & A
        s, f = keep_anchor_pieces(s, d["anc"] == side); fb |= f
        out[s] = side
    return out, nbridge, fb


def metrics(L, d, core_percentile):
    S = L > 0; M = d["mb"]; v = d["vvox"]; r = {}
    r["vol_mm3"] = float(S.sum() * v); r["share_of_mask_kept"] = float((S & M).sum() / M.sum())
    for side, nm in ((1, "left"), (2, "right")):
        s = L == side; r[f"vol_{nm}_mm3"] = float(s.sum() * v)
        lab, n = ndimage.label(s, structure=S26); r[f"n_pieces_{nm}"] = int(n)
        r[f"largest_share_{nm}"] = float(np.bincount(lab.ravel())[1:].max() / s.sum()) if n else np.nan
    r["lr_ratio"] = r["vol_left_mm3"] / r["vol_right_mm3"] if r["vol_right_mm3"] > 0 else np.nan
    SZ = S & d["Z"]
    r["containment"] = float((SZ & M).sum() / SZ.sum()) if SZ.any() else np.nan
    r["core_coverage"] = float((S & d["core"]).sum() / d["core"].sum())
    c, k_ = r["containment"], r["core_coverage"]
    r["score_hm"] = float(2 * c * k_ / (c + k_)) if (np.isfinite(c) and c + k_ > 0) else np.nan
    if S.any():
        shell = ndimage.binary_dilation(S, structure=S6) & ~S & (d["hemi"] > 0)
        r["inside_median_ratio"] = float(np.median(d["rr"][S]))
        r["contrast"] = r["inside_median_ratio"] / float(np.median(d["rr"][shell])) if shell.any() else np.nan
        r["share_snc"] = float((S & d["snc25"]).sum() / S.sum())
        r["share_outside_sn"] = float((S & ~d["snc25"] & ~d["snr25"]).sum() / S.sum())
    else:
        r.update(inside_median_ratio=np.nan, contrast=np.nan, share_snc=np.nan, share_outside_sn=np.nan)
    kk = np.arange(S.shape[2]); r["above_mask_mm3"] = float(S[..., kk > d["kmax"]].sum() * v)
    multi = empty = n = 0                      # fragmentation on labelled slices (side halves where the mask has >= 10 voxels)
    for side in (1, 2):
        for k in range(d["kmin"], d["kmax"] + 1):
            if (d["mlr"][..., k] == side).sum() < 10:
                continue
            n += 1; npc = ndimage.label(L[..., k] == side)[1]; multi += npc > 1; empty += npc == 0
    r["slice_sides_multi_piece"] = multi / n if n else np.nan; r["slice_sides_empty"] = int(empty)
    return r


def run(ctx):
    P, cfg = ctx.paths, ctx.cfg
    rules, gcfg = cfg["segmentation_rules"], cfg["gmm_refinement"]
    OUT = P.gmm
    FIG = OUT / "qc_pages"
    FIG.mkdir(parents=True, exist_ok=True)
    P_CROP = P.cropped
    subj = read_subjects(P)
    qc = pd.read_csv(P_CROP / "qc_table.csv")
    cw = pd.read_csv(P_CROP / "crop_windows.csv")
    CHANNELS = json.load(open(P.atlas / "atlas_registration_decisions.json"))["outputs"]["channels"]
    D = (subj[subj.role == "dev"][["subject_id", "group", "fold"]].merge(qc[["subject_id", "scale_group", "include_training"]], on="subject_id", how="inner")
         .merge(cw[["subject_id", "i0", "j0"]], on="subject_id", how="left"))
    D = D[D.include_training.astype(bool)].reset_index(drop=True)
    if D.empty:
        log("no development subject with a usable manual mask: GMM label refinement skipped (inference only)")
        write_text(OUT / "README.txt", "No development subject with a usable manual mask in this dataset: no labels were made.")
        return
    D["fold"] = D.fold.fillna(-1).astype(int)
    iSNc, iSNr, iRN = CHANNELS.index("SNc"), CHANNELS.index("SNr"), CHANNELS.index("RN")
    GM, G, CP = rules["gmm"], rules["gate"], int(rules["core_percentile"])
    GAP, TOL = float(gcfg["bridge_gap_mm"]), float(gcfg["bridge_tol_mm"])
    SETTINGS = {
        "gmm": {"from_config": GM, "outer_limit": rules["search_region"], "snc_anchor": rules["snc_anchor"],
                "postprocess": "per side: pieces (26-connected) touching the SNc anchor; holes filled in-plane; inside the outer limit"},
        "bridging": {
            "allowed_region": "per side: manual mask AND atlas outer limit",
            "where": "each slice separately (the fragments are in-plane cross-sections; each side is already one piece in 3D)",
            "pieces": "4-connected 2D pieces of the GMM mask inside the allowed region",
            "rule": "for every pair of pieces i, j on a slice with nearest-voxel distance d_ij <= G: add voxels v in the allowed region, "
                    "not already in the GMM mask, with dist(v, i) + dist(v, j) <= d_ij + tol (a narrow corridor between the closest points)",
            "tol_mm": TOL, "G_mm": GAP, "reason": gcfg["reason"],
            "after": ["fill holes in-plane (interior only, outline unchanged), AND allowed region",
                      "keep 26-connected pieces touching the SNc anchor of that side (if none: largest piece); drop the rest"]},
        "never": ["test subjects", "Dice vs masks for any choice", "group differences for any choice"],
    }
    SET_HASH = sha_settings(SETTINGS)
    json.dump(dict(settings=SETTINGS, settings_sha256=SET_HASH), open(OUT / "gmm_refinement_settings.json", "w"), indent=2, default=str)

    def crop(sid, key, name, dtype=np.float32):
        return load_nii(P_CROP / "crops" / key / f"{sid}_{name}.nii.gz", dtype)

    def load_subject(r):
        sid = r.subject_id
        rr, caff = crop(sid, "refratio", "spir_refratio")
        atl = crop(sid, "atlas", "atlas")[0]; hemi = crop(sid, "hemi", "hemi", np.uint8)[0]
        mb = crop(sid, "mask_bin", "mask_bin", np.uint8)[0] > 0; mlr = crop(sid, "mask_lr", "mask_lr", np.uint8)[0]
        mfull, faff = load_nii(P_CROP / "masks_full" / f"{sid}_sn_mask_bin.nii.gz", np.uint8)
        back = np.zeros(mfull.shape, bool); back[r.i0:r.i0 + mb.shape[0], r.j0:r.j0 + mb.shape[1], :] = mb
        assert (back == (mfull > 0)).all(), f"{sid}: crop does not restore onto the full SPIR grid"
        zs = vox_size(caff); km = np.flatnonzero(mb.any(axis=(0, 1)))
        Z = np.zeros(mb.shape, bool); Z[..., km.min():km.max() + 1] = True
        snc25, snr25 = atl[..., iSNc] >= 0.25, atl[..., iSNr] >= 0.25
        return dict(rr=rr, atl=atl, hemi=hemi, mb=mb, mlr=mlr, caff=caff, faff=faff, fshape=mfull.shape, i0=int(r.i0), j0=int(r.j0),
                    zs=zs, vvox=float(np.prod(zs)), Z=Z, kmin=int(km.min()), kmax=int(km.max()),
                    reg=outer_limit(atl, zs, rules, iSNc, iSNr, iRN), anc=snc_anchor(atl, hemi, rules, iSNc), snc25=snc25, snr25=snr25,
                    core=mb & Z & (rr >= np.percentile(rr[mb], CP)), fold=int(r.fold))

    def save_label(L, sid, method, d):
        for grid in ("crop", "spir"):
            if grid == "spir":
                arr = np.zeros(d["fshape"], np.uint8); arr[d["i0"]:d["i0"] + L.shape[0], d["j0"]:d["j0"] + L.shape[1], :] = L; aff = d["faff"]
            else:
                arr, aff = L, d["caff"]
            od = OUT / f"seg_{grid}" / method; od.mkdir(parents=True, exist_ok=True)
            im = nib.Nifti1Image(arr.astype(np.uint8), aff); im.set_sform(aff, 1); im.set_qform(aff, 1)
            nib.save(im, str(od / f"{sid}_{method}_lr.nii.gz"))

    t0 = time.time(); DATA = {r.subject_id: load_subject(r) for r in D.itertuples()}
    log(f"{len(DATA)} subjects loaded ({time.time() - t0:.0f} s); crop -> SPIR restore exact for all")

    # GMM mask on every development subject
    GM_RS = int(np.random.default_rng([ctx.seed, int(GM["seed_tail"])]).integers(2**31 - 1))
    SEG, INFO = {s: {} for s in DATA}, []
    for sid, d in DATA.items():
        x = d["rr"][d["reg"]].reshape(-1, 1).astype(np.float64)
        gm = GaussianMixture(n_components=int(GM["n_components"]), covariance_type=GM["covariance"], random_state=GM_RS).fit(x)
        b = int(np.argmax(gm.means_.ravel()))
        post = np.zeros(d["mb"].shape, np.float32); post[d["reg"]] = gm.predict_proba(x)[:, b]
        SEG[sid]["gmm"] = postprocess_gmm(post >= float(GM["posterior_min"]), d)
        INFO.append(dict(subject_id=sid, gmm_converged=bool(gm.converged_), gmm_means=str(np.round(gm.means_.ravel(), 4).tolist())))
        save_label(SEG[sid]["gmm"], sid, "gmm", d)
    pd.DataFrame(INFO).to_csv(OUT / "gmm_details.csv", index=False)
    log(f"GMM masks saved for {len(SEG)} subjects | GMM converged for all: {all(i['gmm_converged'] for i in INFO)}")

    # bridging: no bridging (reference for the QC display) and the chosen gap
    JN, FB = [], []
    for sid, d in DATA.items():
        L0, _, _ = joined(SEG[sid]["gmm"], d, 0.0, TOL)
        L, nb, fb = joined(SEG[sid]["gmm"], d, GAP, TOL)
        SEG[sid]["no_bridge"], SEG[sid]["gmm_refined"] = L0, L
        added = int(((L > 0) & ~(L0 > 0)).sum())
        JN.append(dict(subject_id=sid, gap_mm=GAP, n_bridges=nb, added_vox=added, added_mm3=added * d["vvox"]))
        if fb:
            FB.append(sid)
        assert not ((L > 0) & ~(d["mb"] & d["reg"])).any(), f"{sid}: voxel outside (manual mask AND outer limit)"
        save_label(L, sid, "gmm_refined", d)
    JN = pd.DataFrame(JN); JN.to_csv(OUT / "bridging_counts.csv", index=False)
    log(f"refined masks saved (bridging gap {GAP} mm) | median added {JN.added_mm3.median():.1f} mm3 | "
        f"sides with no piece touching the SNc anchor (largest kept): {FB or 'none'}")

    # metrics and gate (descriptive record of the labels)
    rows = []
    for sid, d in DATA.items():
        r0 = D.set_index("subject_id").loc[sid]
        for m in ("gmm", "gmm_refined"):
            mt = metrics(SEG[sid][m], d, CP); ok, why = auto_gate(mt, G)
            rows.append(dict(subject_id=sid, group=r0.group, fold=int(r0.fold), scale_group=r0.scale_group,
                             mask_type=m, **mt, auto_gate_pass=ok, auto_gate_fail_reasons=why))
    MET = pd.DataFrame(rows)
    MET.to_csv(OUT / "metrics.csv", index=False)
    SUM = {}
    for m in ("gmm", "gmm_refined"):
        x = MET[MET.mask_type == m]
        SUM[m] = {"volume mm3": mi(x.vol_mm3, "{:.0f}"), "share of manual mask kept": mi(x.share_of_mask_kept),
                  "bright-core coverage": mi(x.core_coverage), "L/R volume ratio": mi(x.lr_ratio),
                  "inside median crus ratio": mi(x.inside_median_ratio, "{:.3f}"), "contrast inside/shell": mi(x.contrast, "{:.3f}"),
                  "share in atlas SNc": mi(x.share_snc), "slice-sides with >1 piece": mi(x.slice_sides_multi_piece),
                  "empty slice-sides (total)": int(x.slice_sides_empty.sum()),
                  "auto gate pass": f"{int(x.auto_gate_pass.sum())}/{len(x)}"}
    SUMT = pd.DataFrame(SUM); SUMT.to_csv(OUT / "summary.csv")
    log(SUMT.to_string())

    # QC pages: image | manual mask | GMM | refined
    cols = [("image", None), ("manual mask", "MASK"), ("GMM", "gmm"), (f"refined ({GAP:g} mm bridges)", "gmm_refined")]
    for pn, (sid, d) in enumerate(DATA.items(), 1):
        raw = crop(sid, "raw", "spir_raw")[0]
        m = d["reg"].any(axis=2) | d["mb"].any(axis=2)
        ii, jj = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
        bi = slice(max(ii.min() - 6, 0), min(ii.max() + 7, m.shape[0])); bj = slice(max(jj.min() - 6, 0), min(jj.max() + 7, m.shape[1]))
        v_ = raw[d["reg"]]; vmin, vmax = float(np.percentile(v_, 1)), float(np.percentile(v_, 99.5))
        ks = list(range(max(d["kmin"] - 1, 0), min(d["kmax"] + 3, raw.shape[2])))
        w, h = bi.stop - bi.start, bj.stop - bj.start; pw = 2.3; ph = pw * h / w
        fig, axs = plt.subplots(len(ks), len(cols), figsize=(pw * len(cols), ph * len(ks) + 0.9), squeeze=False)
        for r_i, k in enumerate(ks):
            base = SEG[sid]["no_bridge"][bi, bj, k]
            for c_i, (title, key) in enumerate(cols):
                ax = axs[r_i, c_i]; ax.set_xticks([]); ax.set_yticks([])
                ax.imshow(raw[bi, bj, k].T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest")
                if key is not None:
                    L = (d["mlr"] if key == "MASK" else SEG[sid][key])[bi, bj, k]
                    for side, col in ((1, "orange"), (2, "cyan")):
                        s = L == side
                        if s.any():
                            ax.imshow(np.ma.masked_where(~s.T, s.T), cmap=ListedColormap([col]), origin="lower", alpha=0.45,
                                      vmin=0, vmax=1, interpolation="nearest")
                    if key == "gmm_refined":                         # bridge voxels in solid magenta
                        br = (L > 0) & ~(base > 0)
                        if br.any():
                            ax.imshow(np.ma.masked_where(~br.T, br.T), cmap=ListedColormap(["magenta"]), origin="lower",
                                      vmin=0, vmax=1, interpolation="nearest")
                    if key != "MASK" and d["mb"][bi, bj, k].any():
                        ax.contour(d["mb"][bi, bj, k].T.astype(np.uint8), [0.5], colors="red", linewidths=0.7)
                if r_i == 0:
                    ax.set_title(title, fontsize=8)
            tag = "labelled" if d["kmin"] <= k <= d["kmax"] else ("above" if k > d["kmax"] else "below")
            axs[r_i, 0].set_ylabel(f"sl {k} {tag}", fontsize=7)
        fig.legend(handles=[Line2D([], [], color="red", lw=1.2, label="manual mask outline"),
                            Line2D([], [], color="orange", lw=6, alpha=0.5, label="left"),
                            Line2D([], [], color="cyan", lw=6, alpha=0.5, label="right"),
                            Line2D([], [], color="magenta", lw=6, label="voxels added by bridging")],
                   loc="lower center", ncol=4, fontsize=8, frameon=False)
        fig.suptitle(f"{pn}/{len(DATA)}  {sid} (fold {d['fold']}, {D.set_index('subject_id').loc[sid, 'group']})  |  raw SPIR {vmin:.0f}-{vmax:.0f}",
                     fontsize=9)
        plt.tight_layout(rect=(0, 0.02, 1, 0.98))
        fig.savefig(FIG / f"qc_{pn:02d}_{sid}.png", dpi=110)
        plt.close(fig)

    json.dump(dict(gap_mm=GAP, reason=gcfg["reason"], date=datetime.date.today().isoformat(), settings_sha256=SET_HASH,
                   summary=SUMT["gmm_refined"].to_dict()), open(OUT / "gmm_refinement_choice.json", "w"), indent=2, default=str)
    write_text(OUT / "README.txt",
               "GMM masks and refined training labels\n\n"
               "seg_spir/<type>/<subject>_<type>_lr.nii.gz : full 320 x 320 x 50 SPIR grid, SPIR header (open on the original T1W_3D_SPIR.nii)\n"
               "seg_crop/<type>/<subject>_<type>_lr.nii.gz : 112 x 80 x 50 crop grid, same world coordinates\n"
               "labels: 0 background, 1 left, 2 right\n"
               "types:\n"
               "  gmm          per-subject Gaussian mixture (bright class), pieces touching the SNc anchor, inside the atlas outer limit (no manual mask used)\n"
               f"  gmm_refined  gmm inside (manual mask AND atlas outer limit), fragments on each slice joined by bridging only (gap <= {GAP:g} mm),\n"
               "               holes filled, pieces connected to the SNc anchor kept; outlines not grown. Needs the manual mask: development subjects only.\n")
