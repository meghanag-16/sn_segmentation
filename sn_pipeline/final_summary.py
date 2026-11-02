"""
Final summary: one table with a row per subject (inclusion, registration strategy and grade, QC reasons, output file,
volumes, label-free flags, NM measures) and a short run report.

Outputs (final_summary/): final_summary.csv, run_report.txt, config_used.json
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd

from .utils import mi, read_subjects, write_text, log


def run(ctx, timings=None):
    P = ctx.paths
    out = P.summary
    out.mkdir(parents=True, exist_ok=True)
    subj = read_subjects(P)[["subject_id", "group", "role", "fold"]]
    t = subj.copy()
    f_atl = P.atlas / "atlas_registration.csv"
    if f_atl.exists():
        a = pd.read_csv(f_atl)
        t = t.merge(a[[c for c in ("subject_id", "reg_strategy", "grade") if c in a]].rename(columns={"grade": "atlas_reg_grade"}),
                    on="subject_id", how="left")
    f_qc = P.cropped / "qc_table.csv"
    if f_qc.exists():
        q = pd.read_csv(f_qc)
        t = t.merge(q[["subject_id", "scale_group", "include_inference", "include_training", "qc_reasons"]], on="subject_id", how="left")
    f_m = P.nnunet / "segmentation_metrics.csv"
    if f_m.exists():
        m = pd.read_csv(f_m)
        keep = ["subject_id", "vol_mm3", "vol_left_mm3", "vol_right_mm3", "lr_ratio", "contrast", "crus_lr", "label_free_gate_pass", "flags"]
        t = t.merge(m[[c for c in keep if c in m]], on="subject_id", how="left")
    f_nm = P.nnunet / "nm_measures.csv"
    if f_nm.exists():
        nm = pd.read_csv(f_nm)
        keep = [c for c in nm.columns if c.endswith(("_ratio_median", "_cnr_mean", "_raw_median"))]
        t = t.merge(nm[["subject_id"] + keep], on="subject_id", how="left")
    suffix = ctx.cfg["output"]["mask_suffix"]
    t["sn_mask_file"] = t.subject_id.map(lambda s: str(P.sn_masks / f"{s}{suffix}") if (P.sn_masks / f"{s}{suffix}").exists() else "")
    t.to_csv(out / "final_summary.csv", index=False)
    json.dump({"config_sha256": ctx.config_hash(), "config": ctx.cfg}, open(out / "config_used.json", "w"), indent=2)

    n_out = int((t.sn_mask_file != "").sum())
    L = ["SN SEGMENTATION PIPELINE - RUN REPORT", "=" * 60,
         f"finished: {datetime.now().isoformat(timespec='seconds')}",
         f"dataset: {ctx.data_root}", f"atlas: {ctx.atlas_dir}", f"output: {P.root}",
         f"model folder: {ctx.model_dir if ctx.model_dir else P.nnunet / 'nnUNet_results'}",
         f"config sha256: {ctx.config_hash()}", "",
         f"subjects found: {len(t)} | roles: {t.role.value_counts().to_dict()}",
         f"segmentations written: {n_out} -> {P.sn_masks}"]
    if "include_inference" in t:
        excl = t.loc[~t.include_inference.fillna(False).astype(bool), "subject_id"].tolist()
        L.append(f"not segmented (registration fail / crop cut / manual exclusion / not processed): {excl or 'none'}")
    if "reg_strategy" in t:
        L.append(f"atlas registration strategy: {t.reg_strategy.fillna('n/a').value_counts().to_dict()}")
    if f_m.exists():
        L += ["", "segmentations, median [IQR]:",
              f"  volume mm3 {mi(t.vol_mm3, '{:.0f}')} | L/R {mi(t.lr_ratio)} | contrast inside/shell {mi(t.contrast, '{:.3f}')}",
              f"  label-free gate passed: {int(t.label_free_gate_pass.fillna(False).astype(bool).sum())}/{int(t.label_free_gate_pass.notna().sum())}"]
        bad = t[t.label_free_gate_pass.notna() & ~t.label_free_gate_pass.fillna(True).astype(bool)]
        if len(bad):
            L.append("  flagged (look at their visual QC pages): " + "; ".join(f"{r.subject_id} ({r.flags})" for r in bad.itertuples()))
    if timings:
        L += ["", "stage times (min): " + ", ".join(f"{k} {v / 60:.1f}" for k, v in timings.items())]
    L += ["", "Where to look:",
          "  audit/                    dataset audit, subject table, header geometry, folds",
          "  registration/             rigid and atlas registration, QC pages (figs/)",
          "  cropped/                  crops, normalised images, QC table and pages",
          "  gmm_refined/              GMM masks and refined training labels (subjects with a manual mask)",
          "  nnunet/                   predictions, post-processed segmentations, metrics",
          "  sn_masks/                 FINAL masks on the SPIR grid (0 background, 1 left, 2 right)",
          "  visual_qc/                one QC page per segmented subject",
          "  final_summary/            this report and final_summary.csv"]
    write_text(out / "run_report.txt", "\n".join(L))
    log("\n".join(L))
