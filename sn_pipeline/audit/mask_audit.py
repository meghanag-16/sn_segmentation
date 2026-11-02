"""
mask_audit.py - dataset audit: structure of the manual SN masks.

Structural audit of the SN masks (not a quality judgement).
Per labelled subject: values, volume, connected components, left/right position,
bounding box, contact with the slab edge, and simple intensity statistics of the
SPIR image inside the mask.

Outputs: mask_audit.csv, mask_audit_group_summary.csv, mask_flags.csv,
         mask_summary.txt, mask_volume_by_group.png, mask_audit.log
This audit does not need the geometry audit; it is independent.
"""
from collections import Counter

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from . import audit_common as C

STRUCT26 = np.ones((3, 3, 3), dtype=bool)


def audit_one(s, flags):
    sid = s["subject_id"]
    img, arr, squeezed = C.load_nifti(s["mask"])
    row_excluded = s["mask_excluded_reason"]
    if sid in C.MASK_HEADER_OVERRIDE and s["spir"] is not None:
        # decision: SPIR header is authoritative for these (checked visually)
        simg0 = nib.load(str(s["spir"]))
        if tuple(simg0.shape[:3]) == tuple(arr.shape[:3]):
            img = nib.Nifti1Image(np.asarray(arr), simg0.affine)
            flags.append((sid, "INFO", "mask header replaced by the SPIR header (visually confirmed alignment)"))
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    row = {"subject_id": sid, "group": s["group"], "excluded_from_development": bool(row_excluded),
           "exclusion_reason": row_excluded}
    if arr.ndim != 3:
        flags.append((sid, "ERROR", f"mask is {arr.ndim}D"))
        return row
    vals = np.unique(arr[np.isfinite(arr)]) if arr.dtype.kind == "f" else np.unique(arr)
    row["mask_unique_values"] = str(vals[:10].tolist()) + (" ..." if vals.size > 10 else "")
    row["mask_n_unique_values"] = int(vals.size)
    binm = (arr != 0) & np.isfinite(arr)
    nz_vals = arr[binm].astype(np.float64)
    row["mask_value_min_nonzero"] = round(float(nz_vals.min()), 5) if nz_vals.size else np.nan
    row["mask_value_max"] = round(float(nz_vals.max()), 5) if nz_vals.size else np.nan
    row["mask_n_negative_voxels"] = int((nz_vals < 0).sum())
    if (nz_vals < 0).any():
        flags.append((sid, "WARN", f"mask has {(nz_vals < 0).sum()} negative-valued voxels (min {nz_vals.min():.4f}); masks of other subjects have none"))
    if vals.size > 2 and not np.isin(vals, [0, 1]).all():
        flags.append((sid, "WARN", "mask is not a 0/1 label image (continuous non-zero values); membership is taken as value != 0"))
    vox_vol = float(np.prod(zooms))
    n = int(binm.sum())
    row["mask_voxels"] = n
    row["mask_volume_mm3"] = round(n * vox_vol, 3)
    if n == 0:
        flags.append((sid, "ERROR", "mask is empty"))
        return row

    # connected components (26-connectivity)
    lab, ncomp = ndi.label(binm, structure=STRUCT26)
    sizes = np.bincount(lab.ravel())[1:]
    order = np.argsort(sizes)[::-1]
    sizes_sorted = sizes[order]
    row["n_components"] = int(ncomp)
    row["largest_component_voxels"] = int(sizes_sorted[0])
    row["second_component_voxels"] = int(sizes_sorted[1]) if ncomp > 1 else 0
    row["third_component_voxels"] = int(sizes_sorted[2]) if ncomp > 2 else 0
    row["n_tiny_components_lt10vox"] = int((sizes < 10).sum())
    row["largest_component_fraction"] = round(float(sizes_sorted[0]) / n, 4)
    if ncomp == 1:
        flags.append((sid, "INFO", "mask has 1 connected component (left and right merged, or only one side drawn)"))
    if ncomp > 2:
        flags.append((sid, "WARN", f"mask has {ncomp} connected components; sizes (voxels) {sizes_sorted[:6].tolist()}"))

    # world coordinates (RAS+: +x is right)
    idx = np.argwhere(binm)
    world = nib.affines.apply_affine(img.affine, idx)
    row["world_extent_x_mm"] = round(float(np.ptp(world[:, 0])), 2)
    row["world_extent_y_mm"] = round(float(np.ptp(world[:, 1])), 2)
    row["world_extent_z_mm"] = round(float(np.ptp(world[:, 2])), 2)
    comp_cent = []
    for k in order[:2]:
        cidx = np.argwhere(lab == (k + 1))
        comp_cent.append(nib.affines.apply_affine(img.affine, cidx).mean(axis=0))
    if len(comp_cent) == 2:
        dx = comp_cent[0][0] - comp_cent[1][0]
        row["comp1_minus_comp2_x_mm"] = round(float(dx), 2)
        row["two_largest_separated_in_x_gt5mm"] = bool(abs(dx) > 5)
        right, left = (0, 1) if dx > 0 else (1, 0)
        row["right_component_voxels"] = int(sizes_sorted[right])
        row["left_component_voxels"] = int(sizes_sorted[left])
        row["left_right_voxel_ratio"] = round(float(sizes_sorted[left]) / float(sizes_sorted[right]), 3)
        row["left_right_note"] = "relative (right = larger world x of the two largest components)"
        if abs(dx) <= 5:
            flags.append((sid, "WARN", "two largest components are less than 5 mm apart in x; left/right assignment unreliable"))

    # bounding box and slab-edge contact
    bb_min, bb_max = idx.min(axis=0), idx.max(axis=0)
    row["bbox_min_vox"] = str(bb_min.tolist())
    row["bbox_max_vox"] = str(bb_max.tolist())
    ax = C.slab_axis(arr.shape, zooms)
    row["slab_axis_index"] = ax
    row["slab_slices_with_mask"] = int(np.unique(idx[:, ax]).size)
    row["slab_first_last_mask_slice"] = f"{int(bb_min[ax])}-{int(bb_max[ax])} of {arr.shape[ax]}"
    touches_slab = bool(bb_min[ax] == 0 or bb_max[ax] == arr.shape[ax] - 1)
    row["touches_slab_edge"] = touches_slab
    if touches_slab:
        flags.append((sid, "WARN", "mask touches the first or last slice of the slab (the SN may be cut by the slab)"))
    inplane = [a for a in range(3) if a != ax]
    touch_ip = any(bb_min[a] == 0 or bb_max[a] == arr.shape[a] - 1 for a in inplane)
    row["touches_inplane_border"] = bool(touch_ip)
    if touch_ip:
        flags.append((sid, "WARN", "mask touches the in-plane border of the image"))

    # intensity statistics on SPIR (only if geometry agrees)
    if s["spir"] is None:
        return row
    simg, sarr, _ = C.load_nifti(s["spir"])
    if sarr.shape != arr.shape:
        flags.append((sid, "ERROR", f"mask shape {arr.shape} != SPIR shape {sarr.shape}; intensity statistics skipped"))
        return row
    if not np.allclose(simg.affine, img.affine, atol=1e-3):
        flags.append((sid, "ERROR", "mask affine != SPIR affine; intensity statistics skipped"))
        return row
    sv = sarr.astype(np.float64)
    ok = np.isfinite(sv)
    thr = C.otsu_threshold(sv[ok & (sv > 0)]) if (ok & (sv > 0)).any() else 0.0
    fg = ok & (sv > thr)
    fg_med = float(np.median(sv[fg])) if fg.any() else np.nan
    inside = sv[binm & ok]
    row["spir_otsu_threshold"] = round(thr, 4)
    row["spir_foreground_median"] = round(fg_med, 4)
    row["inside_mean"] = round(float(inside.mean()), 4)
    row["inside_sd"] = round(float(inside.std()), 4)
    for q in (1, 5, 50, 95, 99):
        row[f"inside_p{q}"] = round(float(np.percentile(inside, q)), 4)
    row["inside_min"] = round(float(inside.min()), 4)
    row["inside_max"] = round(float(inside.max()), 4)
    row["inside_mean_over_fg_median"] = round(float(inside.mean() / fg_med), 4) if fg_med else np.nan
    row["inside_p1_over_fg_median"] = round(float(np.percentile(inside, 1) / fg_med), 4) if fg_med else np.nan
    row["fraction_mask_voxels_below_otsu"] = round(float((inside <= thr).mean()), 4)
    row["mask_voxels_where_spir_is_zero"] = int((inside == 0).sum())
    if row["fraction_mask_voxels_below_otsu"] > 0.2:
        flags.append((sid, "WARN", f"{row['fraction_mask_voxels_below_otsu']:.0%} of mask voxels are below the image's Otsu threshold (dark voxels inside mask)"))
    if row["mask_voxels_where_spir_is_zero"] > 0:
        flags.append((sid, "INFO", f"{row['mask_voxels_where_spir_is_zero']} mask voxels lie where SPIR intensity is exactly 0"))
    return row


def run(data_root, out_dir, limit=0):
    args = C.make_args(data_root, out_dir, limit)
    log = C.setup_logging(args.out_dir, "mask_audit")
    subjects, notes = C.discover_subjects(args.data_root, args.limit, unzip_dir=args.out_dir / "_unzipped")
    labelled = [s for s in subjects if s["mask"] is not None]   # includes excluded masks (still audited)
    log.info("%d subject folders, %d with a mask file", len(subjects), len(labelled))
    flags = [("(discovery)", "WARN", n) for n in notes]

    rows = []
    for i, s in enumerate(labelled, 1):
        log.info("[%d/%d] %s (%s)", i, len(labelled), s["subject_id"], s["group"])
        r = C.run_safely(log, s["subject_id"], audit_one, s, flags)
        if r is None:
            flags.append((s["subject_id"], "ERROR", "mask audit crashed for this subject (see log)"))
            r = {"subject_id": s["subject_id"], "group": s["group"]}
        rows.append(r)

    df_all = pd.DataFrame(rows)
    df = df_all[~df_all["excluded_from_development"].fillna(False).astype(bool)].copy() if "excluded_from_development" in df_all else df_all
    n_excl = len(df_all) - len(df)

    # volume outliers across all labelled subjects (robust z, descriptive)
    if "mask_volume_mm3" in df and df["mask_volume_mm3"].notna().sum() >= 5:
        v = df["mask_volume_mm3"].astype(float)
        med = v.median()
        mad = (v - med).abs().median() * 1.4826
        if mad > 0:
            df["volume_robust_z"] = ((v - med) / mad).round(2)
            for _, r in df[df["volume_robust_z"].abs() > 3.5].iterrows():
                flags.append((r["subject_id"], "WARN", f"mask volume {r['mask_volume_mm3']} mm3 is an outlier (robust z {r['volume_robust_z']})"))
    df_all.to_csv(args.out_dir / "mask_audit.csv", index=False)   # all rows, excluded ones marked

    # group summary (descriptive only)
    num_cols = [c for c in ["mask_voxels", "mask_volume_mm3", "n_components", "left_right_voxel_ratio",
                            "slab_slices_with_mask", "inside_mean", "inside_p1", "inside_min",
                            "inside_mean_over_fg_median", "inside_p1_over_fg_median",
                            "fraction_mask_voxels_below_otsu"] if c in df.columns]
    if num_cols and "group" in df:
        gs = df.groupby("group")[num_cols].agg(["count", "mean", "std", "min", "median", "max"]).round(3)
        gs.columns = ["_".join(c) for c in gs.columns]
        gs.to_csv(args.out_dir / "mask_audit_group_summary.csv")
    pd.DataFrame(flags, columns=["subject_id", "severity", "message"]).to_csv(
        args.out_dir / "mask_flags.csv", index=False)

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if "mask_volume_mm3" in df and df["mask_volume_mm3"].notna().any():
            groups = sorted(df["group"].unique())
            fig, ax = plt.subplots(figsize=(6, 4))
            data = [df.loc[df["group"] == g, "mask_volume_mm3"].dropna().values for g in groups]
            try:
                ax.boxplot(data, tick_labels=groups, showmeans=True)
            except TypeError:  # older matplotlib
                ax.boxplot(data, labels=groups, showmeans=True)
            for k, d in enumerate(data, 1):
                ax.scatter(np.random.default_rng(0).normal(k, 0.04, len(d)), d, s=12, alpha=0.6)
            ax.set_ylabel("mask volume (mm^3)")
            ax.set_title("Mask volume by group (descriptive)")
            fig.tight_layout()
            fig.savefig(args.out_dir / "mask_volume_by_group.png", dpi=150)
            plt.close(fig)
    except Exception as e:  # noqa
        log.warning("Figure skipped: %s", e)

    # text summary
    lines = ["MASK AUDIT SUMMARY", "=" * 60,
             f"Masks audited: {len(df_all)}; excluded from development: {n_excl}; summarised below: {len(df)}",
             f"By group: {dict(Counter(df['group']))}", ""]
    if "n_components" in df:
        lines.append("Connected components per mask: " + str(dict(Counter(df["n_components"].dropna().astype(int)))))
    if "touches_slab_edge" in df:
        lines.append(f"Masks touching the slab edge: {int(df['touches_slab_edge'].fillna(False).sum())}")
    if "mask_n_unique_values" in df:
        lines.append("Distinct values per mask: " + str(dict(Counter(df["mask_n_unique_values"].dropna().astype(int)))))
    if "mask_volume_mm3" in df:
        lines.append("Mask volume mm3 by group (mean, sd): " +
                     str(df.groupby("group")["mask_volume_mm3"].agg(["mean", "std"]).round(1).to_dict("index")))
    # consistency of the lower intensity bound inside masks, raw vs relative
    if "inside_p1" in df and df["inside_p1"].notna().sum() >= 5:
        def cv(x):
            x = x.dropna().astype(float)
            return float(x.std() / x.mean()) if len(x) > 1 and x.mean() else float("nan")
        lines += ["",
                  "Spread of the lowest intensities inside masks across subjects (coefficient of variation):",
                  f"  raw 1st percentile inside mask: {cv(df['inside_p1']):.3f}",
                  f"  1st percentile / foreground median: {cv(df['inside_p1_over_fg_median']):.3f}",
                  "  (Descriptive only. A much smaller value in one of the two rows may hint at how the masks were",
                  "   produced, but this does not prove which rule was used. Ask the person who made them.)"]
    sev = Counter(f[1] for f in flags)
    lines += ["", f"FLAGS: {dict(sev)}  (details in mask_flags.csv)"]
    for f in flags[:40]:
        lines.append(f"  [{f[1]}] {f[0]}: {f[2]}")
    if len(flags) > 40:
        lines.append(f"  ... {len(flags) - 40} more in mask_flags.csv")
    lines += ["", "Left/right is only relative here (no midline is defined). Nothing was modified or excluded."]
    text = "\n".join(lines)
    (args.out_dir / "mask_summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)

