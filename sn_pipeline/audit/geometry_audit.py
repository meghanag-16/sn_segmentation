"""
geometry_audit.py - dataset audit: file inventory and image geometry.

For every subject: manifest (what files exist), image geometry (shape, voxel size,
orientation, affine, intensity statistics) for SPIR, T1w and mask, whitelisted
JSON header fields, mask-vs-SPIR geometry agreement, and a list of flags.

Outputs (in the output folder):
  manifest.csv, geometry_audit.csv, json_fields.csv, json_consistency.csv,
  geometry_flags.csv, geometry_summary.txt, geometry_audit.log
"""
from collections import Counter

import nibabel as nib
import numpy as np
import pandas as pd

from . import audit_common as C


def describe_image(path, kind):
    img, arr, squeezed = C.load_nifti(path)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    a = arr.astype(np.float64, copy=False) if arr.dtype.kind in "fiu" else arr
    finite = np.isfinite(a)
    n_bad = int(a.size - finite.sum())
    vals = a[finite]
    nz = vals[vals != 0]
    row = {
        f"{kind}_file": path.name,
        f"{kind}_ndim": arr.ndim,
        f"{kind}_squeezed_4d": squeezed,
        f"{kind}_shape": str(tuple(int(s) for s in arr.shape)),
        f"{kind}_zooms_mm": str(tuple(round(z, 5) for z in zooms)),
        f"{kind}_dtype": str(arr.dtype),
        f"{kind}_axcodes": "".join(nib.aff2axcodes(img.affine)),
        f"{kind}_nan_inf_count": n_bad,
        f"{kind}_min": float(vals.min()) if vals.size else np.nan,
        f"{kind}_max": float(vals.max()) if vals.size else np.nan,
        f"{kind}_mean": float(vals.mean()) if vals.size else np.nan,
        f"{kind}_p99": float(np.percentile(vals, 99)) if vals.size else np.nan,
        f"{kind}_nonzero_fraction": float(nz.size / vals.size) if vals.size else np.nan,
        f"{kind}_unique_values": int(np.unique(vals).size) if kind == "mask" else np.nan,
        f"{kind}_affine": np.array2string(img.affine.round(5), separator=",").replace("\n", ""),
        f"{kind}_qform_code": int(img.header["qform_code"]),
        f"{kind}_sform_code": int(img.header["sform_code"]),
    }
    return img, arr, row, zooms


def expected_check(kind, shape, zooms, flags, sid):
    exp = C.EXPECTED.get(kind)
    if exp is None:
        return
    shape = tuple(int(s) for s in shape[:3])
    if shape != exp["shape"]:
        if sorted(shape) == sorted(exp["shape"]):
            flags.append((sid, "INFO", f"{kind} shape {shape} has the expected dimensions in a different axis order than {exp['shape']}"))
        else:
            flags.append((sid, "WARN", f"{kind} shape {shape} differs from expected {exp['shape']}"))
    zr = tuple(round(z, 4) for z in zooms)
    ze = tuple(round(z, 4) for z in exp["zooms"])
    if zr != ze and sorted(zr) != sorted(ze):
        flags.append((sid, "WARN", f"{kind} voxel size {zr} differs from expected {ze}"))


def run(data_root, out_dir, limit=0):
    args = C.make_args(data_root, out_dir, limit)
    log = C.setup_logging(args.out_dir, "geometry_audit")
    subjects, notes = C.discover_subjects(args.data_root, args.limit, unzip_dir=args.out_dir / "_unzipped")
    log.info("Found %d subject folders under %s", len(subjects), args.data_root)
    for n in notes:
        log.warning(n)

    pd.DataFrame(C.manifest_rows(subjects)).to_csv(args.out_dir / "manifest.csv", index=False)

    flags = []
    for n in notes:
        flags.append(("(discovery)", "WARN", n))

    # counts check (labelled = has a mask and the mask is not excluded)
    grp = Counter(s["group"] for s in subjects if s["labelled"])
    n_test = sum(1 for s in subjects if s["is_test"])
    for g_, n in C.EXPECTED_COUNTS.items():
        if grp.get(g_, 0) != n:
            flags.append(("(counts)", "WARN", f"labelled '{g_}' subjects: found {grp.get(g_, 0)}, expected {n}"))
    if n_test != C.EXPECTED_TEST:
        flags.append(("(counts)", "WARN", f"test subjects (folder 'testing'): found {n_test}, expected {C.EXPECTED_TEST}"))
    other = {g_: n for g_, n in grp.items() if g_ not in C.EXPECTED_COUNTS}
    if other:
        flags.append(("(counts)", "WARN", f"labelled subjects in unexpected groups: {other}"))
    for s in subjects:
        if s["mask_excluded_reason"]:
            flags.append((s["subject_id"], "INFO", "mask excluded from development: " + s["mask_excluded_reason"]))
        if s["is_test"] and s["mask"] is not None:
            flags.append((s["subject_id"], "WARN", "test subject has a mask file"))
    # duplicate ids across groups
    ids = [s["subject_id"] for s in subjects]
    for sid, n in Counter(ids).items():
        if n > 1:
            flags.append((sid, "WARN", "subject id appears more than once"))

    geo_rows, json_rows = [], []
    for i, s in enumerate(subjects, 1):
        sid = s["subject_id"]
        log.info("[%d/%d] %s (%s)", i, len(subjects), sid, s["group"])
        row = {"subject_id": sid, "group": s["group"], "labelled": s["labelled"]}
        if s["file_notes"]:
            flags.append((sid, "INFO", "files: " + s["file_notes"]))
        imgs = {}

        for kind, key in (("spir", "spir"), ("t1w", "t1w"), ("mask", "mask")):
            p = s[key]
            if p is None:
                if kind != "mask":
                    flags.append((sid, "ERROR", f"{kind} file missing"))
                elif s["labelled"]:
                    flags.append((sid, "ERROR", "mask file missing"))
                continue
            res = C.run_safely(log, f"{sid}:{kind}", describe_image, p, kind)
            if res is None:
                flags.append((sid, "ERROR", f"{kind} file could not be read: {p.name}"))
                continue
            img, arr, r, zooms = res
            row.update(r)
            imgs[kind] = (img, arr, zooms)
            if r[f"{kind}_nan_inf_count"]:
                flags.append((sid, "ERROR", f"{kind} contains {r[f'{kind}_nan_inf_count']} NaN/inf voxels"))
            if arr.ndim != 3:
                flags.append((sid, "ERROR", f"{kind} is {arr.ndim}D, expected 3D"))
            if r[f"{kind}_squeezed_4d"]:
                flags.append((sid, "INFO", f"{kind} was 4D with a single volume; squeezed to 3D"))
            if kind in ("spir", "t1w"):
                expected_check(kind, arr.shape, zooms, flags, sid)
                if r[f"{kind}_max"] == r[f"{kind}_min"]:
                    flags.append((sid, "ERROR", f"{kind} image is constant"))

        # mask vs SPIR agreement
        if "mask" in imgs and "spir" in imgs:
            mi, ma, mz = imgs["mask"]
            si, sa, sz = imgs["spir"]
            same_shape = tuple(ma.shape) == tuple(sa.shape)
            same_aff = bool(np.allclose(mi.affine, si.affine, atol=1e-3))
            row["mask_same_shape_as_spir"] = same_shape
            row["mask_same_affine_as_spir"] = same_aff
            if not same_shape:
                flags.append((sid, "ERROR", f"mask shape {ma.shape} != SPIR shape {sa.shape}"))
            elif not same_aff:
                if sid in C.MASK_HEADER_OVERRIDE:
                    flags.append((sid, "INFO", "mask header differs from SPIR; checked visually and aligned by voxel index; SPIR header is used"))
                else:
                    flags.append((sid, "ERROR", "mask and SPIR have the same shape but different affine"))
            if row.get("mask_unique_values", 0) > 2:
                flags.append((sid, "INFO", f"mask has {row['mask_unique_values']} distinct values (not strictly binary)"))

        # JSON
        js_spir = C.load_json_path(s["spir_json"])
        js_t1 = C.load_json_path(s["t1w_json"])
        if s["spir"] and js_spir is None:
            flags.append((sid, "WARN", "SPIR json not found or unreadable"))
        if s["t1w"] and js_t1 is None and s["t1w"] is not None:
            flags.append((sid, "WARN", "T1w json not found or unreadable"))
        if s["json_notes"]:
            flags.append((sid, "INFO", "json: " + s["json_notes"]))
        for kind, js in (("spir", js_spir), ("t1w", js_t1)):
            if js is None:
                continue
            jr = {"subject_id": sid, "group": s["group"], "file": kind}
            jr.update(C.json_subset(js))
            tilt = C.tilt_from_iop(js.get("ImageOrientationPatientDICOM"))
            jr["derived_slab_tilt_deg"] = round(tilt, 3) if tilt is not None else None
            json_rows.append(jr)
            if kind == "spir":
                row["json_spir_slab_tilt_deg"] = jr["derived_slab_tilt_deg"]
        gap = C.acq_gap_minutes(js_spir, js_t1)
        row["acq_gap_spir_minus_t1w_min"] = gap

        # array shape vs header fields (recorded, not interpreted)
        if js_spir and "spir" in imgs:
            row["json_spir_slice_thickness"] = js_spir.get("SliceThickness")
            row["json_spir_slice_spacing"] = js_spir.get("SpacingBetweenSlices")
            row["json_spir_oop_encoding_steps"] = js_spir.get("PhaseEncodingStepsOutOfPlane")
            row["json_spir_acq_matrix_pe"] = js_spir.get("AcquisitionMatrixPE")
            row["json_spir_recon_matrix_pe"] = js_spir.get("ReconMatrixPE")
            row["json_spir_freq_steps"] = js_spir.get("FrequencyEncodingSteps")
            row["json_spir_philips_rescale_slope"] = js_spir.get("PhilipsRescaleSlope")
            row["json_spir_philips_scale_slope"] = js_spir.get("PhilipsScaleSlope")

        geo_rows.append(row)

    geo = pd.DataFrame(geo_rows)
    geo.to_csv(args.out_dir / "geometry_audit.csv", index=False)
    js_df = pd.DataFrame(json_rows)
    js_df.to_csv(args.out_dir / "json_fields.csv", index=False)

    # Protocol consistency across subjects (per file type)
    cons_rows = []
    if not js_df.empty:
        for kind, sub in js_df.groupby("file"):
            for col in [c for c in sub.columns if c not in ("subject_id", "group", "file")]:
                vc = sub[col].astype(str).value_counts()
                cons_rows.append({
                    "file": kind, "field": col, "n_subjects": int(sub[col].notna().sum()),
                    "n_unique_values": int(vc.size),
                    "values_and_counts": "; ".join(f"{k} x{v}" for k, v in vc.head(8).items()),
                })
                if vc.size > 1 and col not in ("PhilipsRescaleSlope", "PhilipsScaleSlope"):
                    flags.append(("(protocol)", "WARN",
                                  f"{kind} JSON field {col} differs across subjects: "
                                  + "; ".join(f"{k} x{v}" for k, v in vc.head(5).items())))
    pd.DataFrame(cons_rows).to_csv(args.out_dir / "json_consistency.csv", index=False)

    pd.DataFrame(flags, columns=["subject_id", "severity", "message"]).to_csv(
        args.out_dir / "geometry_flags.csv", index=False)

    # Text summary
    lines = ["GEOMETRY AUDIT SUMMARY", "=" * 60,
             f"Data root: {args.data_root}",
             f"Subject folders found: {len(subjects)}",
             f"  labelled (mask present and not excluded) by group: {dict(grp)}",
             f"  test subjects: {n_test}",
             f"  masks excluded: {[s['subject_id'] for s in subjects if s['mask_excluded_reason']]}", ""]
    for kind in ("spir", "t1w", "mask"):
        for col, title in ((f"{kind}_shape", "shape"), (f"{kind}_zooms_mm", "voxel size"),
                           (f"{kind}_axcodes", "orientation"), (f"{kind}_dtype", "dtype")):
            if col in geo:
                vc = geo[col].value_counts()
                lines.append(f"{kind} {title}: " + "; ".join(f"{k} x{v}" for k, v in vc.items()))
        lines.append("")
    if "mask_same_shape_as_spir" in geo:
        lines.append(f"Masks with same shape as SPIR: {int(geo['mask_same_shape_as_spir'].sum())} of {geo['mask_same_shape_as_spir'].notna().sum()}")
        lines.append(f"Masks with same affine as SPIR: {int(geo['mask_same_affine_as_spir'].sum())} of {geo['mask_same_affine_as_spir'].notna().sum()}")
    if "json_spir_slab_tilt_deg" in geo:
        lines.append("SPIR slab tilt from JSON (deg): " + str(geo["json_spir_slab_tilt_deg"].describe().round(3).to_dict()))
    if "acq_gap_spir_minus_t1w_min" in geo:
        lines.append("Acquisition gap SPIR minus T1w (minutes): " + str(geo["acq_gap_spir_minus_t1w_min"].describe().round(2).to_dict()))
    if "spir_p99" in geo:
        cols = [c for c in ("spir_p99", "json_spir_philips_rescale_slope", "json_spir_philips_scale_slope") if c in geo]
        gm = geo.groupby("group")[cols].median().round(4)
        lines += ["", "SPIR raw intensity scale by group (median per group; raw intensities are not comparable between subjects):",
                  gm.to_string()]
    with_json = int(geo["json_spir_slab_tilt_deg"].notna().sum()) if "json_spir_slab_tilt_deg" in geo else 0
    lines += ["", f"SPIR json found and read for {with_json} of {len(geo)} subjects"]
    sev = Counter(f[1] for f in flags)
    lines += ["", f"FLAGS: {dict(sev)}  (details in geometry_flags.csv)"]
    for f in flags[:40]:
        lines.append(f"  [{f[1]}] {f[0]}: {f[2]}")
    if len(flags) > 40:
        lines.append(f"  ... {len(flags) - 40} more in geometry_flags.csv")
    lines += ["", "Nothing was modified, reoriented or excluded. Flags are for you to review."]
    text = "\n".join(lines)
    (args.out_dir / "geometry_summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)

