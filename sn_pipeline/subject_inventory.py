"""
Subject table: finds every subject folder, assigns group / role / fold, checks the NIfTI headers and
builds the intensity-scale (PhilipsScaleSlope) table.

Outputs (audit/subject_table/):
  subjects.csv            one row per subject: subject_id, group, role (dev / test / excluded), fold, relative file paths
  setup_report.txt        every check with PASS / FAIL
  env_info.json, requirements_lock.txt   software and hardware of this run
  scale_slope_table.csv, scale_slope_summary.txt
"""
import hashlib
import importlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from .utils import Report, natural_key, write_text, log


# ----------------------------------------------------------------------------------------------------------------
# environment record
# ----------------------------------------------------------------------------------------------------------------
PACKAGES = [("numpy", "numpy"), ("scipy", "scipy"), ("pandas", "pandas"), ("nibabel", "nibabel"), ("SimpleITK", "SimpleITK"),
            ("ANTsPy (antspyx)", "ants"), ("scikit-image", "skimage"), ("scikit-learn", "sklearn"), ("matplotlib", "matplotlib"),
            ("torch", "torch"), ("nnunetv2", "nnunetv2")]


def record_environment(ctx, out_dir):
    pk = {}
    for label, mod in PACKAGES:
        try:
            m = importlib.import_module(mod)
            pk[label] = getattr(m, "__version__", "installed (no __version__)")
        except Exception as e:
            pk[label] = f"NOT AVAILABLE ({type(e).__name__})"
    gpu = {"nvidia_smi": None, "torch_cuda_available": None}
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=20)
            gpu["nvidia_smi"] = r.stdout.strip() or r.stderr.strip()
        except Exception:
            pass
    try:
        import torch
        gpu["torch_cuda_available"] = bool(torch.cuda.is_available())
    except ImportError:
        pass
    mem_gb = None
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                mem_gb = round(int(line.split()[1]) / 1024 / 1024, 1)
                break
    except OSError:
        pass
    env = {"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "pipeline_seed": ctx.seed,
           "python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count(),
           "memory_gb": mem_gb, "disk_free_gb_output": round(shutil.disk_usage(str(out_dir)).free / 1e9, 1),
           "gpu": gpu, "packages": pk, "config_sha256": ctx.config_hash()}
    json.dump(env, open(out_dir / "env_info.json", "w"), indent=2)
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=180).stdout
        open(out_dir / "requirements_lock.txt", "w").write(freeze)
    except Exception:
        pass
    missing = [k for k, v in pk.items() if v.startswith("NOT AVAILABLE")]
    log("not importable in this environment: " + (", ".join(missing) if missing else "none"))
    return env


# ----------------------------------------------------------------------------------------------------------------
# subject table
# ----------------------------------------------------------------------------------------------------------------
def find_data_root(explicit, group_rules):
    """The given folder if it holds a first-level group folder; otherwise the shallowest folder below it that does."""
    firsts = {p[0].lower() for p, _ in group_rules}
    queue = [(Path(explicit), 0)]
    while queue:
        d, depth = queue.pop(0)
        try:
            kids = sorted(c for c in d.iterdir() if c.is_dir())
        except OSError:
            continue
        if firsts & {c.name.lower() for c in kids}:
            return d
        if depth < 4:
            queue.extend((c, depth + 1) for c in kids)
    raise FileNotFoundError(f"Data root not found under {explicit}: no folder contains one of the group folders {sorted(firsts)}.")


def discover(root, file_stems, accepted_ext):
    wanted = {stem + ext: kind for kind, stem in file_stems.items() for ext in accepted_ext}
    found = defaultdict(lambda: defaultdict(list))
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort(key=natural_key)
        for fn in filenames:
            if fn in wanted:
                found[Path(dirpath)][wanted[fn]].append(fn)
    return found


def classify(rel_parts, group_rules):
    low = [p.lower() for p in rel_parts]
    hits = []
    for prefix, group in group_rules:
        n = len(prefix)
        if len(low) > n and low[:n] == [p.lower() for p in prefix]:
            hits.append((group, rel_parts[n], "/".join(rel_parts[n + 1:])))
    return hits[0] if len(hits) == 1 else (None if not hits else "ambiguous")


def build_table(root, rep, ds):
    found = discover(root, ds["file_stems"], ds["accepted_ext"])
    rows = []
    for d in sorted(found, key=lambda p: natural_key(str(p))):
        rel_parts = list(d.relative_to(root).parts)
        cls, files, where = classify(rel_parts, ds["group_rules"]), found[d], "/".join(rel_parts) or "."
        if cls is None:
            rep.error(f"folder not matched by any group rule: {where}"); continue
        if cls == "ambiguous":
            rep.error(f"folder matched by more than one group rule: {where}"); continue
        group, folder_name, subfolder = cls
        sid = ds["subject_id_format"].format(group=group, folder_name=folder_name)
        multi = {k: v for k, v in files.items() if len(v) > 1}
        if multi:
            rep.error(f"{sid}: more than one file for {multi}"); continue
        rel = lambda k: ("/".join(rel_parts + [files[k][0]]) if k in files else "")
        rows.append({"subject_id": sid, "group": group, "folder_name": folder_name, "subfolder": subfolder,
                     "spir_path": rel("spir"), "t1w_path": rel("t1w"), "mask_path": rel("mask")})
    df = pd.DataFrame(rows)
    if df.empty:
        rep.error("no subject folders found"); return df
    dup = df.subject_id[df.subject_id.duplicated()].unique().tolist()
    rep.check(not dup, f"subject_ids are unique (duplicates: {dup})")
    pat = re.compile(ds["folder_name_pattern"])
    odd = df.loc[~df.folder_name.apply(lambda s: bool(pat.match(s))), "folder_name"].tolist()
    if odd: rep.warn(f"folder names that do not look like SCZ-27M / HC-23M (possible patient names): {odd}")
    else:   rep.info("all folder names look like SCZ-27M / HC-23M (no obvious patient names)")
    for kind in ("spir", "t1w"):
        miss = df.loc[df[f"{kind}_path"] == "", "subject_id"].tolist()
        rep.check(not miss, f"every folder has a {kind} file (missing: {miss})")
    excluded = set(ds["excluded_masks"])
    df["role"] = df.apply(lambda r: "test" if r.group in ds["test_groups"] else ("excluded" if r.subject_id in excluded else "dev"), axis=1)
    miss_mask = df.loc[df.role.isin(["dev", "excluded"]) & (df.mask_path == ""), "subject_id"].tolist()
    rep.check(not miss_mask, f"every dev/excluded subject has a mask (missing: {miss_mask})")
    extra = df.loc[(df.role == "test") & (df.mask_path != ""), "subject_id"].tolist()
    if extra: rep.warn(f"test subjects that have a mask file (test set should have none): {extra}")
    df["mask_header_override"] = df.subject_id.isin(ds["mask_header_override"])
    return df


def check_counts(df, rep, ds):
    exp_n, exp_dev, exp_test = ds["expected_n_folders"], ds["expected_dev_counts"], ds["expected_test_counts"]
    rep.check(len(df) == exp_n, f"{len(df)} subject folders (expected {exp_n})")
    dev = df[df.role == "dev"].groupby("group").size().to_dict()
    test = df[df.role == "test"].groupby("group").size().to_dict()
    rep.check(dev == exp_dev, f"dev counts {dev} (expected {exp_dev})")
    rep.check(test == exp_test, f"test counts {test} (expected {exp_test})")
    ex = sorted(df.loc[df.role == "excluded", "subject_id"])
    rep.check(ex == sorted(ds["excluded_masks"]), f"excluded subjects {ex} (expected {sorted(ds['excluded_masks'])})")
    gone = sorted(set(ds["mask_header_override"]) - set(df.subject_id))
    rep.check(not gone, f"all mask-header-override subjects exist in the table (missing: {gone})")


def merge_folds(df, folds_csv, test_csv, rep, n_folds):
    if folds_csv is None or not Path(folds_csv).exists():
        rep.warn("no folds file (no development subjects with masks): fold left empty")
        df["fold"] = pd.array([pd.NA] * len(df), dtype="Int64")
        return df
    f = pd.read_csv(folds_csv)
    need = {"subject_id", "group", "fold"}
    if not need <= set(f.columns):
        rep.error(f"folds file lacks columns {need - set(f.columns)}"); df["fold"] = pd.NA; return df
    f = f[["subject_id", "group", "fold"]].rename(columns={"group": "group_folds", "fold": "fold_folds"})
    rep.check(not f.subject_id.duplicated().any(), "each subject appears once in the folds file")
    dev_ids, fold_ids = set(df.loc[df.role == "dev", "subject_id"]), set(f.subject_id)
    rep.check(dev_ids == fold_ids, f"dev subjects in the table equal subjects in the folds file (only in table: {sorted(dev_ids - fold_ids)}; only in folds: {sorted(fold_ids - dev_ids)})")
    rep.check(set(f.fold_folds.unique()) == set(range(n_folds)), f"fold values are exactly 0..{n_folds - 1}")
    leak = sorted(fold_ids & set(df.loc[df.role.isin(["test", "excluded"]), "subject_id"]))
    rep.check(not leak, f"no test or excluded subject is in the folds (found: {leak})")
    m = df.merge(f, on="subject_id", how="left")
    bad = m[m.fold_folds.notna() & (m.group != m.group_folds)].subject_id.tolist()
    rep.check(not bad, f"group in table equals group in folds file (differences: {bad})")
    df["fold"] = m["fold_folds"].astype("Int64").values
    if (df.role == "dev").any():
        tab = df[df.role == "dev"].pivot_table(index="group", columns="fold", values="subject_id", aggfunc="count", margins=True)
        rep.info("subjects per group and fold:\n" + tab.to_string())
    if test_csv is not None and Path(test_csv).exists():
        t = pd.read_csv(test_csv)
        t_ids, tab_ids = set(t.subject_id), set(df.loc[df.role == "test", "subject_id"])
        rep.check(t_ids == tab_ids, f"test subjects in the table equal test_subjects.csv (only in table: {sorted(tab_ids - t_ids)}; only in file: {sorted(t_ids - tab_ids)})")
    return df


def verify_headers(df, root, rep, ds):
    exp_shape = {k: tuple(v) for k, v in ds["expected_shape"].items()}
    exp_zoom = {k: tuple(v) for k, v in ds["expected_zooms"].items()}
    affine_diff, bad_shape, bad_zoom, unreadable = set(), [], [], []
    for r in df.itertuples():
        imgs = {}
        for kind in ("spir", "t1w", "mask"):
            p = getattr(r, f"{kind}_path")
            if p:
                try: imgs[kind] = nib.load(str(root / p))
                except Exception as e: unreadable.append(f"{r.subject_id}/{kind}: {e}")
        for kind, im in imgs.items():
            if tuple(im.shape) != exp_shape[kind]: bad_shape.append(f"{r.subject_id}/{kind}: {tuple(im.shape)}")
            z = tuple(float(x) for x in im.header.get_zooms()[:3])
            if not np.allclose(z, exp_zoom[kind], atol=ds["zoom_tolerance"]): bad_zoom.append(f"{r.subject_id}/{kind}: {z}")
        if "spir" in imgs and "mask" in imgs and not np.allclose(imgs["spir"].affine, imgs["mask"].affine, atol=ds["affine_tolerance"]):
            affine_diff.add(r.subject_id)
    rep.check(not unreadable, f"all files readable (problems: {unreadable})")
    rep.check(not bad_shape, f"all shapes as expected (problems: {bad_shape})")
    rep.check(not bad_zoom, f"all voxel sizes as expected (problems: {bad_zoom})")
    ov = set(ds["mask_header_override"]) & set(df.subject_id)
    ok = affine_diff == ov
    (rep.info if ok else rep.warn)(("PASS: " if ok else "CHECK: ") + f"masks whose header differs from SPIR = {sorted(affine_diff)} (expected exactly {sorted(ov)})")
    rep.info("headers only were read; voxel data were not loaded")


# ----------------------------------------------------------------------------------------------------------------
# intensity-scale (PhilipsScaleSlope) table
# ----------------------------------------------------------------------------------------------------------------
def slope_from_json(row, root):
    """JSON next to the SPIR whose name contains 'spir' (if several, the ones not starting with 'sub-')."""
    folder = (root / row.spir_path).parent
    js = [p for p in folder.glob("*.json") if "spir" in p.name.lower()]
    if len(js) > 1:
        js = [p for p in js if not p.name.lower().startswith("sub-")] or js
    if len(js) != 1:
        return np.nan
    try:
        return float(json.load(open(js[0])).get("PhilipsScaleSlope", np.nan))
    except Exception:
        return np.nan


def scale_slope_table(subj, root, json_fields_csv, test_groups, out_dir):
    slopes = {r.subject_id: slope_from_json(r, root) for r in subj.itertuples()}
    source = {s: ("json_next_to_spir" if np.isfinite(v) else "") for s, v in slopes.items()}
    n_missing = sum(1 for v in slopes.values() if not np.isfinite(v))
    source_note = "JSON next to each SPIR (name contains 'spir')"
    if n_missing:
        # JSON file names are not uniform: fall back to the audit's json_fields.csv, where the SPIR JSON is found by CONTENT
        try:
            jf = pd.read_csv(json_fields_csv)
            jf = jf[jf["file"] == "spir"].set_index("subject_id")["PhilipsScaleSlope"]
            for s in [k for k, v in slopes.items() if not np.isfinite(v)]:
                if s in jf.index and np.isfinite(float(jf[s])):
                    slopes[s] = float(jf[s]); source[s] = "audit_json_fields"
            source_note = f"JSON next to SPIR; {n_missing} looked up in the audit json_fields.csv (SPIR JSON identified by content)"
        except Exception as e:
            source_note = f"JSON next to SPIR; {n_missing} subjects have no slope and json_fields.csv was not usable ({type(e).__name__})"

    rows = []
    for r in subj.itertuples():
        a = nib.load(str(root / r.spir_path)).get_fdata(dtype=np.float32)
        p99 = float(np.percentile(a, 99))
        s = slopes[r.subject_id]
        rows.append(dict(subject_id=r.subject_id, group=r.group, role=r.role, philips_scale_slope=s,
                         max_raw=float(a.max()), p99_raw=p99, p99_scaled=p99 * s if np.isfinite(s) else np.nan,
                         slope_source=source[r.subject_id]))
        del a
    sl = pd.DataFrame(rows)
    sl["set"] = np.where(sl.group.isin(test_groups), "test", "dev")
    sl.to_csv(out_dir / "scale_slope_table.csv", index=False)

    def cv(x):
        x = x.dropna()
        return float(x.std(ddof=1) / x.mean()) if len(x) > 1 else np.nan
    L = ["SCALE-SLOPE TABLE", "=" * 60, f"slope source: {source_note}",
         f"subjects with a slope: {int(sl.philips_scale_slope.notna().sum())} of {len(sl)}",
         "p99 = 99th percentile over ALL voxels of the volume as read by nibabel",
         f"CV (sd/mean, ddof=1) of p99 across all subjects: raw {cv(sl.p99_raw):.3f}, after multiplying by slope {cv(sl.p99_scaled):.3f}",
         f"CV of max intensity (raw): {cv(sl.max_raw):.3f}", "", "median p99 by group:",
         sl.groupby("group")[["p99_raw", "p99_scaled", "philips_scale_slope"]].median().round(4).to_string(),
         "", "median p99 by set (dev vs test):", sl.groupby("set")[["p99_raw", "p99_scaled"]].median().round(4).to_string(),
         "", "Only the numbers are reproduced here. Why the slope removes the group differences is not established."]
    write_text(out_dir / "scale_slope_summary.txt", "\n".join(L))
    return sl


# ----------------------------------------------------------------------------------------------------------------
# stage entry point
# ----------------------------------------------------------------------------------------------------------------
def run(ctx):
    P, ds = ctx.paths, ctx.cfg["dataset"]
    out = P.subject_table
    out.mkdir(parents=True, exist_ok=True)
    record_environment(ctx, out)

    rep = Report()
    rep.info(f"config sha256 {ctx.config_hash()[:16]}")
    root = find_data_root(ctx.data_root, ds["group_rules"])
    folds_csv = Path(ds["folds_file"]) if ds.get("folds_file") else P.splits / "folds.csv"
    test_csv = Path(ds["test_subjects_file"]) if ds.get("test_subjects_file") else P.splits / "test_subjects.csv"
    rep.info(f"data root: {root}"); rep.info(f"folds file: {folds_csv}"); rep.info(f"test file: {test_csv}")

    df = build_table(root, rep, ds)
    if df.empty:
        write_text(out / "setup_report.txt", "\n".join(rep.lines))
        raise RuntimeError("no subject folders found: see " + str(out / "setup_report.txt"))
    check_counts(df, rep, ds)
    df = merge_folds(df, folds_csv, test_csv, rep, int(ds["n_folds"]))
    verify_headers(df, root, rep, ds)
    cols = ["subject_id", "group", "role", "fold", "folder_name", "subfolder", "spir_path", "t1w_path", "mask_path", "mask_header_override"]
    _rank = {sid: i for i, sid in enumerate(sorted(df.subject_id, key=natural_key))}   # natural sort by subject_id
    df["_ord"] = df.subject_id.map(_rank)
    df = df.sort_values(["role", "group", "_ord"]).drop(columns="_ord").reset_index(drop=True)
    if ctx.limit:
        df = df.head(ctx.limit)
        rep.warn(f"subject limit {ctx.limit} in use (quick test run)")
    df[cols].to_csv(P.subjects_csv(), index=False)   # relative paths only
    rep.info(f"written: {P.subjects_csv()} ({len(df)} rows)")
    rep.info("subjects.csv sha256: " + hashlib.sha256(open(P.subjects_csv(), "rb").read()).hexdigest())
    status = "ALL CHECKS PASSED" if not rep.errors else f"{len(rep.errors)} CHECK(S) FAILED"
    text = "\n".join(["SUBJECT TABLE REPORT", "=" * 60, *rep.lines, "", f"RESULT: {status}; warnings: {len(rep.warnings)}",
                      "(count checks compare with the expected dataset sizes in the config; for another cohort they only inform)"])
    write_text(out / "setup_report.txt", text)
    log(text)

    subj = pd.read_csv(P.subjects_csv()).fillna({"mask_path": "", "t1w_path": ""})
    scale_slope_table(subj, root, P.audit / "json_fields.csv", ds["test_groups"], out)
    json.dump({"data_root": str(root)}, open(out / "data_root.json", "w"), indent=2)
    return root


def data_root_of(ctx):
    """data root found by the subject-table stage (the folder that holds the group folders)"""
    f = ctx.paths.subject_table / "data_root.json"
    if f.exists():
        return Path(json.load(open(f))["data_root"])
    return find_data_root(ctx.data_root, ctx.cfg["dataset"]["group_rules"])
