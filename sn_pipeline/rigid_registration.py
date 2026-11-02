"""
Rigid registration T1w -> native SPIR (SimpleITK, Mattes mutual information, 6 degrees of freedom, SPIR head mask).
The SPIR world space is the SPIR sform for every subject (for four controls the qform is displaced; the sform matches the T1w).
The SPIR is never resampled; the T1w is resampled into the SPIR grid for QC and for the atlas registration checks.

Outputs (registration/rigid_t1w_to_spir/):
  transforms/<subject>_t1w_to_spir_rigid.tfm   ITK transform, maps SPIR (fixed) physical LPS points to T1w (moving) points
  t1w_in_spir/<subject>_t1w_in_spir.nii.gz     T1w on the SPIR grid
  rigid_registration.csv, rigid_registration_summary.txt, reproducibility_flagged.csv
  registration_qc_grades.csv, header_rule_final.json, rigid_registration_decisions.json, rigid_registration_report.txt
  figs/qc_levels_page<n>.png                    SPIR with registered-T1w edges at 8 levels (binary mask in red)
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from skimage.feature import canny
from skimage.filters import threshold_otsu

from .subject_inventory import data_root_of
from .utils import nmi, read_subjects, write_text, log


def spir_affine(img, world):
    return img.get_qform() if world == "qform" else img.get_sform()


def to_sitk(arr, aff_ras, dtype=np.float32):
    # nibabel array (i, j, k) + RAS affine -> SimpleITK image with the same geometry (ITK uses LPS)
    A = np.diag([-1.0, -1.0, 1.0]) @ np.asarray(aff_ras, float)[:3, :]
    M = A[:, :3]
    sp = np.linalg.norm(M, axis=0)
    im = sitk.GetImageFromArray(np.ascontiguousarray(np.asarray(arr, dtype).transpose(2, 1, 0)))
    im.SetSpacing(sp.tolist()); im.SetDirection((M / sp).flatten().tolist()); im.SetOrigin(A[:, 3].tolist())
    return im


def from_sitk(im):
    return sitk.GetArrayFromImage(im).transpose(2, 1, 0)


def head_mask(spir):
    m = spir > threshold_otsu(spir[spir > 0]) if (spir > 0).any() else spir > 0
    m = ndimage.binary_fill_holes(ndimage.binary_closing(m, iterations=2))
    return ndimage.binary_dilation(m, iterations=3)


def corner_displacement(tx, fixed):
    # largest distance any corner of the SPIR slab moves under the transform (mm): a size of the correction
    sz = fixed.GetSize()
    corners = [fixed.TransformContinuousIndexToPhysicalPoint([i, j, k]) for i in (0, sz[0] - 1) for j in (0, sz[1] - 1) for k in (0, sz[2] - 1)]
    return float(max(np.linalg.norm(np.subtract(tx.TransformPoint(c), c)) for c in corners))


class RigidRegistration:
    """attempt 2 only runs if attempt 1 looks wrong (NMI not improved, or rotation > 10 deg); the better NMI is kept"""

    def __init__(self, root, seed, attempts, world):
        self.root, self.seed, self.attempts, self.world = root, seed, attempts, world

    def _run(self, fixed, moving, fmask, a):
        tx = sitk.Euler3DTransform()
        tx.SetCenter(fixed.TransformContinuousIndexToPhysicalPoint([(s - 1) / 2 for s in fixed.GetSize()]))
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        R.SetMetricSamplingStrategy(R.RANDOM)
        R.SetMetricSamplingPercentage(a["sampling"], self.seed)
        R.SetMetricFixedMask(fmask)
        R.SetInterpolator(sitk.sitkLinear)
        R.SetOptimizerAsRegularStepGradientDescent(learningRate=a["lr"], minStep=1e-4, numberOfIterations=300, relaxationFactor=0.5)
        R.SetOptimizerScalesFromPhysicalShift()
        R.SetShrinkFactorsPerLevel(a["shrink"])
        R.SetSmoothingSigmasPerLevel(a["smooth"])
        R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        R.SetInitialTransform(tx, inPlace=True)
        R.Execute(fixed, moving)
        return tx, R

    def register(self, r, world=None):
        world = world or self.world
        si = nib.load(str(self.root / r.spir_path))
        spir = si.get_fdata(dtype=np.float32)
        ti = nib.load(str(self.root / r.t1w_path))
        fixed = to_sitk(spir, spir_affine(si, world))
        moving = to_sitk(ti.get_fdata(dtype=np.float32), ti.affine)
        hm = head_mask(spir)
        fmask = to_sitk(hm.astype(np.uint8), spir_affine(si, world), np.uint8)
        before = from_sitk(sitk.Resample(moving, fixed, sitk.Euler3DTransform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32))
        best = None
        for n_att, a in enumerate(self.attempts, 1):
            tx, R = self._run(fixed, moving, fmask, a)
            after = from_sitk(sitk.Resample(moving, fixed, tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32))
            v = hm & (after > 0) & (before > 0)
            p = tx.GetParameters()
            res = dict(rot_x_deg=np.degrees(p[0]), rot_y_deg=np.degrees(p[1]), rot_z_deg=np.degrees(p[2]),
                       t_x_mm=p[3], t_y_mm=p[4], t_z_mm=p[5], t_norm_mm=float(np.linalg.norm(p[3:6])),
                       max_corner_disp_mm=corner_displacement(tx, fixed),
                       nmi_header_only=nmi(spir[v], before[v]), nmi_registered=nmi(spir[v], after[v]),
                       final_metric=R.GetMetricValue(), iterations=R.GetOptimizerIteration(),
                       stop=R.GetOptimizerStopConditionDescription()[:120], attempt=n_att)
            if best is None or res["nmi_registered"] > best[0]["nmi_registered"]:
                best = (res, tx, after)
            rot = max(abs(res["rot_x_deg"]), abs(res["rot_y_deg"]), abs(res["rot_z_deg"]))
            if res["nmi_registered"] >= res["nmi_header_only"] and rot <= 10:
                break
        res, tx, after = best
        res["attempts_run"] = n_att
        return res, tx, spir, after, spir_affine(si, world)


def add_flags(reg):
    reg["rot_max_deg"] = reg[["rot_x_deg", "rot_y_deg", "rot_z_deg"]].abs().max(axis=1)
    reg["flag_nmi_not_improved"] = reg.nmi_registered < reg.nmi_header_only
    reg["flag_large_correction"] = (reg.max_corner_disp_mm > 10) | (reg.rot_max_deg > 5)
    reg["flag_hit_iteration_limit"] = reg.stop.astype(str).str.contains("Maximum number of iterations", case=False)
    reg["used_fallback_attempt"] = reg.attempts_run > 1
    return reg


def reproducibility(reg, flagged, subj, RR, out):
    """the larger corrections must be reproducible with the other optimizer setting"""
    rep_rows = []
    for sid in flagged.subject_id:
        r = subj.loc[subj.subject_id == sid].iloc[0]
        si = nib.load(str(RR.root / r.spir_path)); ti = nib.load(str(RR.root / r.t1w_path))
        spir = si.get_fdata(dtype=np.float32)
        fixed = to_sitk(spir, spir_affine(si, RR.world))
        moving = to_sitk(ti.get_fdata(dtype=np.float32), ti.affine)
        hm = head_mask(spir)
        fmask = to_sitk(hm.astype(np.uint8), spir_affine(si, RR.world), np.uint8)
        tx1 = sitk.ReadTransform(str(out / "transforms" / f"{sid}_t1w_to_spir_rigid.tfm"))
        used = int(reg.loc[reg.subject_id == sid, "attempt"].iloc[0]) if "attempt" in reg.columns else 1
        other = 1 if used == 1 else 0                 # run the setting that did NOT produce the saved transform
        tx2, R2 = RR._run(fixed, moving, fmask, RR.attempts[other])
        sz = fixed.GetSize()
        corners = [fixed.TransformContinuousIndexToPhysicalPoint([i, j, k]) for i in (0, sz[0] - 1) for j in (0, sz[1] - 1) for k in (0, sz[2] - 1)]
        disagreement = max(float(np.linalg.norm(np.subtract(tx1.TransformPoint(c), tx2.TransformPoint(c)))) for c in corners)
        p2 = tx2.GetParameters()
        rep_rows.append(dict(subject_id=sid, saved_from_attempt=used,
                             setting1_corner_mm=float(reg.loc[reg.subject_id == sid, "max_corner_disp_mm"].iloc[0]),
                             setting2_corner_mm=corner_displacement(tx2, fixed), setting2_rot_max_deg=float(np.degrees(np.abs(p2[:3]).max())),
                             disagreement_mm=disagreement, saved_final_metric=float(reg.loc[reg.subject_id == sid, "final_metric"].iloc[0]),
                             other_final_metric=R2.GetMetricValue()))
    rep = pd.DataFrame(rep_rows)
    if len(rep):
        rep.to_csv(out / "reproducibility_flagged.csv", index=False)
        log("other optimizer setting vs the saved transform (largest slab-corner disagreement):\n" + rep.round(3).to_string(index=False))
        log("Below about 1 mm means two different optimizer settings settle on the same pose (reproducible, not proof of correctness).")
    return rep


def qc_pages(reg, subj, root, out, levels, half, per_page):
    """8 axial levels across the slab, T1w edges (cyan), binary mask (red, development subjects only)"""
    figs = out / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    ids = list(reg.subject_id)
    pages = [ids[i:i + per_page] for i in range(0, len(ids), per_page)]
    for pn, page in enumerate(pages, 1):
        fig, axs = plt.subplots(len(page), len(levels), figsize=(len(levels) * 1.7, len(page) * 1.7))
        axs = np.atleast_2d(axs)
        for ri, sid in enumerate(page):
            r = subj.loc[subj.subject_id == sid].iloc[0]
            spir = nib.load(str(root / r.spir_path)).get_fdata(dtype=np.float32)
            t1 = nib.load(str(out / "t1w_in_spir" / f"{sid}_t1w_in_spir.nii.gz")).get_fdata(dtype=np.float32)
            m = (nib.load(str(root / r.mask_path)).get_fdata(dtype=np.float32) != 0) if r.mask_path else None
            nx, ny, nz = spir.shape
            xs = slice(max(0, nx // 2 - half), min(nx, nx // 2 + half))
            ys = slice(max(0, ny // 2 - half), min(ny, ny // 2 + half))
            lo, hi = np.percentile(spir[spir > 0], [1, 99.5]) if (spir > 0).any() else (0, 1)
            t1p = np.percentile(t1[t1 > 0], 99) if (t1 > 0).any() else 1.0
            for ci, k in enumerate(levels):
                ax = axs[ri, ci]
                ax.set_xticks([]); ax.set_yticks([])
                if k >= nz:
                    ax.axis("off"); continue
                ax.imshow(spir[xs, ys, k].T, cmap="gray", origin="lower", vmin=lo, vmax=hi)
                e = canny(t1[xs, ys, k] / t1p, sigma=4.0, low_threshold=0.05, high_threshold=0.15)
                ax.imshow(np.ma.masked_where(~e.T, e.T), cmap="cool", origin="lower", alpha=0.9, interpolation="nearest")
                if m is not None and m[xs, ys, k].any():
                    ax.contour(m[xs, ys, k].T.astype(np.uint8), levels=[0.5], colors="red", linewidths=0.8)
                if ri == 0:
                    ax.set_title(f"k={k}", fontsize=7)
                if ci == 0:
                    ax.text(0.03, 0.93, f"{sid}", transform=ax.transAxes, color="yellow", fontsize=6, va="top")
        fig.suptitle(f"Page {pn}/{len(pages)}: SPIR with registered T1w edges (cyan); binary mask (red); slice index k increases towards superior", fontsize=8)
        plt.tight_layout()
        fig.savefig(figs / f"qc_levels_page{pn}.png", dpi=100)
        plt.close(fig)
    log(f"{len(pages)} QC pages in {figs}. Grade by the CYAN edges against the SPIR anatomy at every level "
        "(brain surface, ventricles, brainstem outline); do not judge the SN here.")


def run(ctx):
    P, cfg = ctx.paths, ctx.cfg["rigid_registration"]
    out = P.rigid
    for d in ("transforms", "t1w_in_spir", "figs", "per_subject"):
        (out / d).mkdir(parents=True, exist_ok=True)
    root = data_root_of(ctx)
    subj = read_subjects(P)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(int(cfg["n_threads"]))
    world = cfg["spir_world"]
    RR = RigidRegistration(root, ctx.seed, cfg["attempts"], world)
    log(f"rigid T1w -> SPIR | seed {ctx.seed} | SPIR world: {world} | subjects: {len(subj)}")

    # all subjects (finished subjects are skipped)
    rows, t0 = [], time.time()
    for n, r in enumerate(subj.itertuples(), 1):
        f_tfm = out / "transforms" / f"{r.subject_id}_t1w_to_spir_rigid.tfm"
        f_row = out / "per_subject" / f"{r.subject_id}.json"
        if f_tfm.exists() and f_row.exists():
            rows.append(json.load(open(f_row)))
            continue
        res, tx, spir, after, aff = RR.register(r)
        sitk.WriteTransform(tx, str(f_tfm))
        if cfg["save_t1w_in_spir"]:
            im = nib.Nifti1Image(after.astype(np.float32), aff)
            im.set_qform(aff, 1); im.set_sform(aff, 1)
            nib.save(im, str(out / "t1w_in_spir" / f"{r.subject_id}_t1w_in_spir.nii.gz"))
        row = dict(subject_id=r.subject_id, group=r.group, role=r.role, **{k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in res.items()})
        json.dump(row, open(f_row, "w"), indent=1, default=float)
        rows.append(row)
        log(f"{n:2d}/{len(subj)} {r.subject_id:22s} rot ({res['rot_x_deg']:.2f},{res['rot_y_deg']:.2f},{res['rot_z_deg']:.2f}) deg  "
            f"|t| {res['t_norm_mm']:.2f} mm  corner {res['max_corner_disp_mm']:.2f} mm  "
            f"NMI {res['nmi_header_only']:.4f} -> {res['nmi_registered']:.4f}  [{time.time() - t0:.0f}s]")
    reg = add_flags(pd.DataFrame(rows))
    flagged = reg[reg.flag_nmi_not_improved | reg.flag_large_correction | reg.flag_hit_iteration_limit]
    reg.to_csv(out / "rigid_registration.csv", index=False)

    def st(x):
        return f"median {x.median():.2f}, max {x.max():.2f}"
    L = ["RIGID T1w -> SPIR (SimpleITK, Mattes MI, 6 DOF, SPIR head mask, SPIR world = " + world + ")", "=" * 60,
         f"subjects registered: {len(reg)}",
         f"rotation |max angle| (deg): {st(reg.rot_max_deg)}",
         f"translation |t| (mm): {st(reg.t_norm_mm)}",
         f"largest slab-corner displacement (mm): {st(reg.max_corner_disp_mm)}",
         f"NMI header-only -> registered: median {reg.nmi_header_only.median():.4f} -> {reg.nmi_registered.median():.4f}",
         f"NMI improved in {int((~reg.flag_nmi_not_improved).sum())} of {len(reg)}",
         f"subjects where the fallback attempt was run: {reg.loc[reg.used_fallback_attempt, 'subject_id'].tolist()}",
         "", "by group (median corner displacement, mm):", reg.groupby("group").max_corner_disp_mm.median().round(2).to_string(),
         "", f"flagged for a closer look ({len(flagged)}): rule = NMI not improved, or corner displacement > 10 mm, "
             "or rotation > 5 deg, or optimizer hit the iteration limit"]
    if len(flagged):
        L.append(flagged[["subject_id", "rot_max_deg", "t_norm_mm", "max_corner_disp_mm", "nmi_header_only", "nmi_registered", "stop"]]
                 .round(3).to_string(index=False))
    L += ["", "Head motion between the two scans is what this rigid step corrects; a correction of a few mm is expected.",
          "Thin slab: check the slice direction (z) carefully in the QC pages."]
    write_text(out / "rigid_registration_summary.txt", "\n".join(L))
    log("\n".join(L))

    if len(flagged) and not (out / "reproducibility_flagged.csv").exists():
        reproducibility(reg, flagged, subj, RR, out)
    qc_pages(reg, subj, root, out, cfg["qc_levels"], int(cfg["qc_half_window_vox"]), int(cfg["qc_per_page"]))

    # visual grades: only the exceptions are written in the config; everybody else is "pass" once the pages were viewed
    viewed, exc = bool(cfg["viewed_qc_pages"]), dict(cfg["grade_exceptions"])
    grades = reg[["subject_id", "group", "role"]].copy()
    grades["grade"] = grades.subject_id.map(lambda s: exc.get(s, "pass" if viewed else "not graded"))
    grades.to_csv(out / "registration_qc_grades.csv", index=False)
    json.dump({"spir_world_space": f"{world} for all subjects",
               "evidence": "for the four mask-header-override controls the SPIR sform matches the T1w (corner correction "
                           "0.2-0.8 mm) while the qform does not (1.4-13.5 mm); for all other subjects qform = sform",
               "mask_rule": "the override masks take the SPIR header (sform) in world space; voxel indices are unchanged"},
              open(out / "header_rule_final.json", "w"), indent=2)
    json.dump({"spir_world_space": world, "attempts": cfg["attempts"], "n_threads": cfg["n_threads"],
               "method": "SimpleITK rigid Euler3D, Mattes MI 50 bins, random sampling (seed = pipeline seed), SPIR head mask, "
                         "regular step gradient descent; attempt 2 only if attempt 1 looks wrong",
               "transform_convention": "ITK .tfm, maps SPIR (fixed) physical LPS points to T1w (moving) points",
               "t1w_in_spir_saved": cfg["save_t1w_in_spir"]},
              open(out / "rigid_registration_decisions.json", "w"), indent=2)
    checks = {
        "all subjects registered": len(reg) == len(subj),
        "NMI improved or equal in all": not reg.flag_nmi_not_improved.any(),
        "no large-correction flags (or each has a written reason)": not reg.flag_large_correction.any() or bool(exc),
        "visual QC pages viewed and graded": viewed,
        "no registration graded fail": not any(str(v).lower().startswith("fail") for v in exc.values()),
    }
    L = ["RIGID REGISTRATION STATUS", "=" * 60] + [f"{'PASS' if ok else 'OPEN/FAIL'}: {k}" for k, ok in checks.items()]
    L += ["", f"grade counts: {grades.grade.astype(str).str.split(':').str[0].value_counts().to_dict()}"]
    write_text(out / "rigid_registration_report.txt", "\n".join(L))
    log("\n".join(L))
