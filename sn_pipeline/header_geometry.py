"""
Header geometry on the native SPIR grid: orientation codes, header consistency, slab tilt per subject, and a
test that the SPIR array index order matches its header. Nothing is registered or resampled for later use.

Outputs (audit/header_geometry/):
  geometry.csv, tilt_covariates.csv, geometry_summary.txt
  index_order_check.csv, index_order_summary.txt, figs/<subject>_index_order.png (largest, median, smallest pitch)
  header_geometry_report.txt
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.orientations import aff2axcodes
from nibabel.processing import resample_from_to

from .subject_inventory import data_root_of
from .utils import nmi, read_subjects, write_text, log


def angles(aff):
    # definitions (right-hand rule, RAS world): n = slice (k) axis; pitch = rotation of n about the L-R axis,
    # roll = rotation of n about the A-P axis, inplane = angle of the i axis in the axial plane
    M = np.asarray(aff, float)[:3, :3]
    R = M / np.linalg.norm(M, axis=0)
    s = 1.0 if R[2, 2] >= 0 else -1.0
    n = R[:, 2] * s
    return dict(pitch_deg=float(np.degrees(np.arctan2(-n[1], n[2]))),
                roll_deg=float(np.degrees(np.arctan2(n[0], n[2]))),
                inplane_deg=float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
                total_tilt_deg=float(np.degrees(np.arccos(min(1.0, abs(R[2, 2]))))),
                orthonormal_err=float(np.abs(R.T @ R - np.eye(3)).max()),
                slice_axis_points_superior=bool(s > 0))


def stats(x):
    return f"mean {x.mean():.2f}, sd {x.std():.2f}, min {x.min():.2f}, max {x.max():.2f}"


def orientation_and_tilt(subj, root, override, json_fields_csv, out):
    """orientation codes, header consistency, slab tilt per subject (headers only)"""
    rows = []
    for r in subj.itertuples():
        row = dict(subject_id=r.subject_id, group=r.group, role=r.role)
        imgs = {}
        for kind in ("spir", "t1w", "mask"):
            p = getattr(r, f"{kind}_path")
            if not p:
                continue
            im = nib.load(str(root / p))
            imgs[kind] = im
            h = im.header
            qc, sc = int(h["qform_code"]), int(h["sform_code"])
            row[f"{kind}_axcodes"] = "".join(aff2axcodes(im.affine))
            row[f"{kind}_qform_code"], row[f"{kind}_sform_code"] = qc, sc
            row[f"{kind}_qform_sform_maxdiff"] = float(np.abs(im.get_qform() - im.get_sform()).max()) if qc > 0 and sc > 0 else np.nan
            row[f"{kind}_det_sign"] = int(np.sign(np.linalg.det(im.affine[:3, :3])))
        row.update({f"spir_{k}": v for k, v in angles(imgs["spir"].affine).items()})
        t = angles(imgs["t1w"].affine)
        row.update(t1w_total_tilt_deg=t["total_tilt_deg"], t1w_pitch_deg=t["pitch_deg"], t1w_orthonormal_err=t["orthonormal_err"])
        if "mask" in imgs:
            d = imgs["mask"].affine - imgs["spir"].affine
            row["mask_affine_maxdiff"] = float(np.abs(d).max())
            row["mask_translation_diff_mm"] = str(np.round(d[:3, 3], 2).tolist())
        rows.append(row)

    geo = pd.DataFrame(rows)
    geo.to_csv(out / "geometry.csv", index=False)
    geo[["subject_id", "group", "role", "spir_pitch_deg", "spir_roll_deg", "spir_inplane_deg", "spir_total_tilt_deg"]].to_csv(
        out / "tilt_covariates.csv", index=False)

    L = ["ORIENTATION AND HEADER CONSISTENCY (headers only)", "=" * 60]
    for kind in ("spir", "t1w", "mask"):
        c = f"{kind}_axcodes"
        if c not in geo:
            continue
        L.append(f"{kind} orientation codes: {geo[c].value_counts(dropna=True).to_dict()}")
        L.append(f"{kind} qform/sform codes: {geo.groupby([f'{kind}_qform_code', f'{kind}_sform_code']).size().to_dict()}; "
                 f"max qform-sform difference {geo[f'{kind}_qform_sform_maxdiff'].max():.2e}; "
                 f"determinant signs {geo[f'{kind}_det_sign'].value_counts(dropna=True).to_dict()}")
    L.append(f"max deviation from an orthonormal rotation: SPIR {geo.spir_orthonormal_err.max():.2e}, T1w {geo.t1w_orthonormal_err.max():.2e}")
    L.append(f"SPIR slice axis points superior in all subjects: {bool(geo.spir_slice_axis_points_superior.all())}")
    L += ["", "SPIR slab angles (degrees; definitions in the code):",
          f"  pitch (about L-R): {stats(geo.spir_pitch_deg)}",
          f"  roll  (about A-P): {stats(geo.spir_roll_deg)}",
          f"  in-plane         : {stats(geo.spir_inplane_deg)}",
          f"  total tilt       : {stats(geo.spir_total_tilt_deg)}",
          f"  T1w total tilt   : {stats(geo.t1w_total_tilt_deg)}",
          "", "pitch by group (median, min, max):",
          geo.groupby("group").spir_pitch_deg.agg(["median", "min", "max"]).round(2).to_string(),
          "", f"subjects with |roll| or |in-plane| > 2 deg: "
          f"{geo.loc[(geo.spir_roll_deg.abs() > 2) | (geo.spir_inplane_deg.abs() > 2), 'subject_id'].tolist()}",
          "", "5 largest total tilts:",
          geo.nlargest(5, "spir_total_tilt_deg")[["subject_id", "spir_pitch_deg", "spir_total_tilt_deg"]].round(2).to_string(index=False)]
    mm = set(geo.loc[geo.get("mask_affine_maxdiff", pd.Series(dtype=float)) > 1e-3, "subject_id"]) if "mask_affine_maxdiff" in geo else set()
    L += ["", f"masks whose header differs from SPIR: {sorted(mm)}", f"equals the mask-header-override list: {mm == override}"]
    try:
        jf = pd.read_csv(json_fields_csv)
        jf = jf[jf["file"] == "spir"][["subject_id", "derived_slab_tilt_deg"]].dropna()
        m = geo.merge(jf, on="subject_id")
        L += ["", f"cross-check with the audit derived_slab_tilt_deg ({len(m)} subjects): "
              f"corr with total tilt {np.corrcoef(m.derived_slab_tilt_deg, m.spir_total_tilt_deg)[0, 1]:.4f}, "
              f"max |difference| {np.abs(m.derived_slab_tilt_deg - m.spir_total_tilt_deg).max():.3f} deg"]
    except Exception as e:
        L += ["", f"tilt cross-check with the audit skipped ({type(e).__name__})"]
    txt = "\n".join(L)
    write_text(out / "geometry_summary.txt", txt)
    log(txt)
    return geo, mm


def t1_in_spir_grid(r, spir_img, root):
    t1 = nib.load(str(root / r.t1w_path))
    return resample_from_to(t1, (spir_img.shape, spir_img.affine), order=1, cval=0.0).get_fdata(dtype=np.float32)


OPP = {"R": "L", "L": "R", "A": "P", "P": "A", "S": "I", "I": "S"}


def _labels(ax, hcode, vcode):
    kw = dict(transform=ax.transAxes, color="yellow", fontsize=13, fontweight="bold", ha="center", va="center")
    ax.text(0.03, 0.5, OPP[hcode], **kw); ax.text(0.97, 0.5, hcode, **kw)
    ax.text(0.5, 0.03, OPP[vcode], **kw); ax.text(0.5, 0.97, vcode, **kw)


def index_order(subj, root, geo, out):
    """Carry the T1w into the SPIR grid using the two headers only (no registration). If the SPIR header is right,
    the unflipped SPIR should share the most information with it; flipping an axis should lower it.
    Caveats: head motion between the two scans is not corrected; the left-right test is weak because the head is
    nearly symmetric, so a wrong L-R flag cannot be excluded by this test alone."""
    rows = []
    for r in subj.itertuples():
        si = nib.load(str(root / r.spir_path))
        spir = si.get_fdata(dtype=np.float32)
        t1r = t1_in_spir_grid(r, si, root)
        valid = t1r > 0.05 * t1r.max()
        sub = (slice(None, None, 2), slice(None, None, 2), slice(None))
        v = valid[sub]
        b = t1r[sub][v]
        res = {"none": nmi(spir[sub][v], b)}
        for ax, name in ((0, "flip_i"), (1, "flip_j"), (2, "flip_k")):
            res[name] = nmi(np.flip(spir, axis=ax)[sub][v], b)
        best = max(res, key=res.get)
        rows.append(dict(subject_id=r.subject_id, group=r.group, role=r.role, t1w_coverage_frac=float(valid.mean()),
                         **{f"nmi_{k}": v_ for k, v_ in res.items()}, best=best,
                         margin_i=res["none"] - res["flip_i"], margin_j=res["none"] - res["flip_j"], margin_k=res["none"] - res["flip_k"]))
        del spir, t1r
    io = pd.DataFrame(rows)
    io.to_csv(out / "index_order_check.csv", index=False)
    L = ["INDEX ORDER CHECK (header-only T1w -> SPIR, normalised mutual information)", "=" * 60,
         f"subjects where the unflipped SPIR matches best: {int((io.best == 'none').sum())} of {len(io)}",
         f"margin vs flipping j (A-P): min {io.margin_j.min():.4f}, median {io.margin_j.median():.4f}",
         f"margin vs flipping k (slice): min {io.margin_k.min():.4f}, median {io.margin_k.median():.4f}",
         f"margin vs flipping i (L-R, weak test): min {io.margin_i.min():.4f}, median {io.margin_i.median():.4f}",
         f"T1w coverage of the SPIR slab (fraction of voxels): min {io.t1w_coverage_frac.min():.3f}"]
    bad = io[io.best != "none"]
    if len(bad):
        L += ["", "subjects where a flipped SPIR matches better (inspect these):",
              bad[["subject_id", "best", "margin_i", "margin_j", "margin_k"]].round(4).to_string(index=False)]
    txt = "\n".join(L)
    write_text(out / "index_order_summary.txt", txt)
    log(txt)

    # visual check: largest pitch, median pitch, smallest pitch
    figs = out / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    g = geo.sort_values("spir_pitch_deg").reset_index(drop=True)
    fig_subjects = list(dict.fromkeys([g.subject_id.iloc[-1], g.subject_id.iloc[len(g) // 2], g.subject_id.iloc[0]]))
    for sid in fig_subjects:
        r = subj.loc[subj.subject_id == sid].iloc[0]
        si = nib.load(str(root / r.spir_path))
        spir = si.get_fdata(dtype=np.float32)
        t1r = t1_in_spir_grid(r, si, root)
        m = nib.load(str(root / r.mask_path)).get_fdata(dtype=np.float32) != 0 if r.mask_path else None
        zi, zj, zk = si.header.get_zooms()[:3]
        ci, cj, ck = (np.round(np.argwhere(m).mean(0)).astype(int) if m is not None and m.any() else np.array(spir.shape) // 2)
        codes = aff2axcodes(si.affine)
        views = [("axial index k=%d" % ck, lambda a: a[:, :, ck].T, codes[0], codes[1], zj / zi),
                 ("index j=%d" % cj, lambda a: a[:, cj, :].T, codes[0], codes[2], zk / zi),
                 ("index i=%d" % ci, lambda a: a[ci, :, :].T, codes[1], codes[2], zk / zj)]
        fig, axs = plt.subplots(2, 3, figsize=(16, 9))
        for row_i, (vol, name) in enumerate(((spir, "SPIR"), (t1r, "T1w via headers only"))):
            lo, hi = np.percentile(vol[vol > 0], [1, 99.5]) if (vol > 0).any() else (0, 1)
            for col, (title, cut, hc, vc, asp) in enumerate(views):
                ax = axs[row_i, col]
                ax.imshow(cut(vol), cmap="gray", origin="lower", vmin=lo, vmax=hi, aspect=asp)
                if m is not None:
                    ax.contour(cut(m.astype(np.uint8)), levels=[0.5], colors="red", linewidths=0.8)
                _labels(ax, hc, vc)
                ax.set_title(f"{name}, {title}", fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
        pitch = float(geo.loc[geo.subject_id == sid, "spir_pitch_deg"].iloc[0])
        fig.suptitle(f"{sid}  (pitch {pitch:.1f} deg; letters = direction from the header; red = binary mask)", fontsize=12)
        plt.tight_layout()
        fig.savefig(figs / f"{sid}_index_order.png", dpi=110)
        plt.close(fig)
    return io


def run(ctx):
    P = ctx.paths
    out = P.header_geometry
    out.mkdir(parents=True, exist_ok=True)
    root = data_root_of(ctx)
    subj = read_subjects(P)
    override = set(ctx.cfg["dataset"]["mask_header_override"]) & set(subj.subject_id)
    geo, mm = orientation_and_tilt(subj, root, override, P.audit / "json_fields.csv", out)
    io = index_order(subj, root, geo, out)

    decisions = {
        "working_grid": "native T1W_3D_SPIR grid (320 x 320 x 50, 0.5625 x 0.5625 x 1 mm); SPIR and binary masks are never resampled",
        "how_other_data_enter_spir_space": "T1w and atlas/template are carried into SPIR space by transforms (registration stages)",
        "tilt_handling": "per-subject pitch/roll/in-plane/total tilt recorded in tilt_covariates.csv; reporting covariate",
        "mask_header_rule": "masks in the mask-header-override list use the SPIR header",
    }
    json.dump(decisions, open(out / "header_geometry_decisions.json", "w"), indent=2)
    checks = {
        "one orientation code for all SPIR": geo.spir_axcodes.nunique() == 1,
        "one orientation code for all T1w": geo.t1w_axcodes.nunique() == 1,
        "SPIR affines are rotations (err < 1e-3)": geo.spir_orthonormal_err.max() < 1e-3,
        "mask header mismatches equal the override list": mm == override,
        "unflipped SPIR best in all subjects (index order)": bool((io.best == "none").all()),
    }
    L = ["HEADER GEOMETRY STATUS", "=" * 60] + [f"{'PASS' if ok else 'OPEN/FAIL'}: {k}" for k, ok in checks.items()]
    L += ["", "Look at figs/*_index_order.png: the letters must agree with the anatomy (nose = A, top of head = S), the SPIR and",
          "the header-aligned T1w must show the same anatomy at the same place, and the red mask must sit on the midbrain."]
    write_text(out / "header_geometry_report.txt", "\n".join(L))
    log("\n".join(L))
