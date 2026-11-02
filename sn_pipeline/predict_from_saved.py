"""
Prediction only, from registration and crop outputs that already exist (no audit, no registration, no cropping again).

Accepts the outputs of the saved notebooks (data-tree, registration, crop-mask-prep, nnU-Net results) or of an earlier
run of this pipeline. The needed files are found by name below SAVED_DIR (or given one by one), copied into the new
output folder in the layout of this pipeline, and then only these stages run:
    nnunet_inference -> visual_qc -> final_summary

Needed from the earlier run (found by file name, so the folders can sit anywhere below SAVED_DIR):
  subjects.csv                                          subject table (data-tree output, or audit/subject_table/)
  qc_table_1_8.csv or qc_table.csv  + crop_windows.csv   crop output (with the crops/ folder next to them)
  phase_1_3b_decisions.json or atlas_registration_decisions.json   only for the list of atlas channels
  atlas_registration_1_3b.csv or atlas_registration.csv            optional (registration strategy and grade in the summary)
  nnUNet_results/Dataset721_chealdev49r1/...             the trained model (or give --model-dir)

Command line:
  python -m sn_pipeline.predict_from_saved --saved-dir <folder> --output <folder> [--model-dir <folder>]
                                           [--subjects test|all] [--device auto|cpu|cuda]
"""
import argparse
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import final_summary, nnunet_inference, visual_qc
from .config import Context, load_config, set_global_seed
from .pipeline import _Tee
from .utils import log

# kinds of crops the prediction, post-processing and QC read: (folder, file-name part)
CROP_KINDS = [("zscore", "spir_zscore"), ("refratio", "spir_refratio"), ("raw", "spir_raw"), ("atlas", "atlas"),
              ("hemi", "hemi"), ("ref_roi", "ref_crus")]


def find_file(base, names, required=True):
    """the single file below base whose name is one of names (folders named 'retry' are ignored)"""
    hits = []
    for dp, ds, fs in os.walk(base):
        ds[:] = [d for d in ds if d != "retry"]
        hits += [Path(dp) / f for f in fs if f in names]
    if len(hits) == 1:
        return hits[0]
    if not hits and not required:
        return None
    raise FileNotFoundError(f"{' or '.join(names)}: found {len(hits)} below {base}: {[str(h) for h in hits]}. "
                            "Give the file with its own option instead.")


def find_model_dir(base, dataset_name):
    """an nnUNet_results folder below base that contains the trained model folder"""
    hits = sorted({Path(dp) for dp, ds, _ in os.walk(base) if dataset_name in ds})
    if len(hits) != 1:
        raise FileNotFoundError(f"folder {dataset_name}: found in {len(hits)} places below {base}: {[str(h) for h in hits]}. "
                                "Give MODEL_DIR (the folder that contains it).")
    return hits[0]


def import_saved(ctx, saved_dir, subjects_csv=None, qc_csv=None, atlas_json=None, which="test"):
    """copies the needed tables and crops into the new output folder; returns the subjects to segment"""
    P = ctx.paths
    f_subj = Path(subjects_csv) if subjects_csv else find_file(saved_dir, ["subjects.csv"])
    f_qc = Path(qc_csv) if qc_csv else find_file(saved_dir, ["qc_table_1_8.csv", "qc_table.csv"])
    f_cw = f_qc.parent / "crop_windows.csv"
    crops = f_qc.parent / "crops"
    f_atl = Path(atlas_json) if atlas_json else find_file(saved_dir, ["phase_1_3b_decisions.json", "atlas_registration_decisions.json"])
    f_reg = find_file(f_atl.parent, ["atlas_registration_1_3b.csv", "atlas_registration.csv"], required=False)
    for f in (f_subj, f_qc, f_cw, f_atl):
        if not f.exists():
            raise FileNotFoundError(f"not found: {f}")
    if not crops.is_dir():
        raise FileNotFoundError(f"crops folder not found next to {f_qc.name}: {crops}")
    log(f"subject table : {f_subj}\ncrop QC table : {f_qc}\ncrop windows  : {f_cw}\ncrops         : {crops}\n"
        f"atlas channels: {f_atl}\natlas table   : {f_reg or 'not found (optional)'}")

    shutil.copy2(f_subj, P.subjects_csv())
    shutil.copy2(f_qc, P.cropped / "qc_table.csv")
    shutil.copy2(f_cw, P.cropped / "crop_windows.csv")
    shutil.copy2(f_atl, P.atlas / "atlas_registration_decisions.json")
    if f_reg is not None:
        shutil.copy2(f_reg, P.atlas / "atlas_registration.csv")

    subj = pd.read_csv(P.subjects_csv())
    qc = pd.read_csv(P.cropped / "qc_table.csv")
    need = {"subject_id", "scale_group", "include_inference", "include_training"}
    assert need <= set(qc.columns), f"crop QC table lacks columns {need - set(qc.columns)}"
    sel = subj.merge(qc[["subject_id", "include_inference"]], on="subject_id")
    sel = sel[sel.include_inference.astype(bool)]
    if which == "test":
        sel = sel[sel.role == "test"]
    sids = list(sel.subject_id)
    assert sids, f"no subject to segment (subjects: {which})"

    missing = []
    for sid in sids:
        for kind, name in CROP_KINDS:
            src = crops / kind / f"{sid}_{name}.nii.gz"
            if not src.exists():
                missing.append(str(src)); continue
            dst = P.cropped / "crops" / kind / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)
    assert not missing, f"missing crop files ({len(missing)}): {missing[:5]} ..."
    log(f"{len(sids)} subjects to segment ({which}); crops copied into {P.cropped / 'crops'}")
    return sids


def main(argv=None):
    p = argparse.ArgumentParser(prog="sn_pipeline.predict_from_saved",
                                description="nnU-Net prediction from existing registration and crop outputs")
    p.add_argument("--saved-dir", required=True, help="folder that holds the earlier outputs (searched by file name)")
    p.add_argument("--output", required=True)
    p.add_argument("--model-dir", default=None, help="nnU-Net results folder with Dataset721_...; empty = search below --saved-dir")
    p.add_argument("--subjects", default="test", choices=["test", "all"])
    p.add_argument("--subjects-csv", default=None)
    p.add_argument("--qc-table", default=None, help="qc_table_1_8.csv / qc_table.csv (crop_windows.csv and crops/ must be next to it)")
    p.add_argument("--atlas-decisions", default=None, help="phase_1_3b_decisions.json / atlas_registration_decisions.json")
    p.add_argument("--config", default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    a = p.parse_args(argv)
    for name in ("model_dir", "config", "subjects_csv", "qc_table", "atlas_decisions"):   # empty strings mean "not given"
        if getattr(a, name) is not None and not str(getattr(a, name)).strip():
            setattr(a, name, None)

    cfg = load_config(a.config)
    cfg["nnunet"]["inference_subjects"] = a.subjects
    saved = Path(a.saved_dir)
    if not saved.is_dir():
        print(f"folder not found: {saved}"); return 1
    model_dir = Path(a.model_dir) if a.model_dir else None
    ctx = Context(cfg, saved, saved, a.output, model_dir=model_dir, device=a.device)
    ctx.paths.make()
    log_file = ctx.paths.logs / f"prediction_{datetime.now():%Y%m%d_%H%M%S}.log"
    fh = open(log_file, "w", encoding="utf-8")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(old_out, fh), _Tee(old_err, fh)
    try:
        if ctx.model_dir is None:
            ctx.model_dir = find_model_dir(saved, cfg["nnunet"]["dataset_name"])
        log(f"saved outputs {saved}\nmodel folder  {ctx.model_dir}\noutput        {ctx.paths.root}\nlog           {log_file}")
        set_global_seed(ctx.seed)
        timings = {}
        t0 = time.time()
        import_saved(ctx, saved, a.subjects_csv, a.qc_table, a.atlas_decisions, a.subjects)
        timings["import"] = time.time() - t0
        for name, fn in (("nnunet_inference", nnunet_inference.run), ("visual_qc", visual_qc.run)):
            log(f"\n===== {name} =====")
            t0 = time.time()
            fn(ctx)
            timings[name] = time.time() - t0
        log("\n===== final_summary =====")
        final_summary.run(ctx, timings)
        if not cfg["output"]["keep_work_files"]:
            shutil.rmtree(ctx.paths.work, ignore_errors=True)
        log(f"\nDone. Final masks: {ctx.paths.sn_masks}")
        return 0
    except Exception:
        traceback.print_exc()
        log(f"\nStopped. Full log: {log_file}")
        return 1
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        fh.close()


if __name__ == "__main__":
    sys.exit(main())
