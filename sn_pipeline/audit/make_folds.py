"""
make_folds.py - dataset audit: fixed cross-validation folds for the development set.

What it does
  - Reads manifest.csv (written by the geometry audit).
  - Development set = subjects that have a mask, whose mask is not excluded, and that are not test subjects.
  - Randomly assigns every development subject to one of N folds (default 5), BY SUBJECT,
    keeping the group mix (control / chronic / acute) as even as possible in every fold.
  - The random choice uses a fixed seed (723635, drawn once at random), so the same
    dataset always gives the same folds.
  - Writes folds.csv, test_subjects.csv and folds_summary.txt.
  - Never overwrites an existing folds.csv unless force=True, so the split stays frozen.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def as_bool(series):
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def run(manifest, out_dir, seed=723635, n_folds=5, force=False):
    manifest, out_dir = Path(manifest), Path(out_dir)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    out_dir.mkdir(parents=True, exist_ok=True)
    folds_path = out_dir / "folds.csv"
    if folds_path.exists() and not force:
        print(f"{folds_path} already exists and is kept (the split is meant to stay frozen).")
        return folds_path

    m = pd.read_csv(manifest)
    need = {"subject_id", "group", "has_mask", "is_test"}
    missing = need - set(m.columns)
    if missing:
        raise RuntimeError(f"manifest.csv is missing columns: {sorted(missing)}. Re-run the geometry audit.")
    excl = m["mask_excluded_reason"].fillna("").astype(str).str.strip() if "mask_excluded_reason" in m else pd.Series([""] * len(m))

    has_mask = as_bool(m["has_mask"])
    is_test = as_bool(m["is_test"])
    dev = m[has_mask & ~is_test & (excl == "")].copy()
    test = m[is_test].copy()
    excluded = m[has_mask & (excl != "")].copy()

    test_out = test[["subject_id", "group"] + [c for c in ("folder",) if c in test.columns]]
    if dev.empty:
        print("No development subjects (no subject with a usable mask): no folds written.")
        test_out.to_csv(out_dir / "test_subjects.csv", index=False)
        return None
    if dev["subject_id"].duplicated().any():
        raise RuntimeError("Duplicate subject_id values among development subjects; fix before splitting.")

    rng = np.random.default_rng(seed)
    assigned = []
    counter = 0  # runs across groups so fold sizes differ by at most 1
    for grp in sorted(dev["group"].unique()):
        ids = sorted(dev.loc[dev["group"] == grp, "subject_id"])
        order = rng.permutation(len(ids))
        for k in order:
            assigned.append((ids[k], grp, counter % n_folds))
            counter += 1
    folds = pd.DataFrame(assigned, columns=["subject_id", "group", "fold"]).sort_values(["fold", "group", "subject_id"])
    folds = folds.merge(dev[["subject_id"] + [c for c in ("folder",) if c in dev.columns]], on="subject_id", how="left")
    folds["seed"] = seed
    folds.to_csv(folds_path, index=False)
    test_out.to_csv(out_dir / "test_subjects.csv", index=False)

    # ---- sanity checks
    problems = []
    if folds["subject_id"].duplicated().any():
        problems.append("a subject appears in more than one fold")
    if set(folds["subject_id"]) & set(test["subject_id"]):
        problems.append("a test subject appears in the development folds")
    if set(folds["subject_id"]) & set(excluded["subject_id"]):
        problems.append("a subject with an excluded mask appears in the folds")
    if len(folds) != len(dev):
        problems.append("fold table size differs from the development set size")

    tab = pd.crosstab(folds["group"], folds["fold"])
    tab["total"] = tab.sum(axis=1)
    tab.loc["all groups"] = tab.sum(axis=0)
    lines = ["FOLD FILE SUMMARY", "=" * 60,
             f"Manifest: {manifest}",
             f"Seed: {seed}    Folds: {n_folds}",
             f"Development subjects (in folds): {len(folds)}",
             f"Test subjects (never in folds): {len(test)}   -> test_subjects.csv",
             f"Excluded masks (not in folds): {excluded['subject_id'].tolist()}",
             "", "Subjects per group and fold:", tab.to_string(), "",
             "Checks: " + ("ALL PASSED (each subject in exactly one fold; no test or excluded subject in folds)"
                           if not problems else "PROBLEMS: " + "; ".join(problems)),
             "", "Rules: folds are by subject, never by slice. The test subjects are not used for training,",
             "tuning, model selection or fitting of normalisation statistics or priors."]
    text = "\n".join(lines)
    (out_dir / "folds_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nWritten to: {out_dir}")
    if problems:
        raise RuntimeError("fold file problems: " + "; ".join(problems))
    return folds_path
