"""
End-to-end pipeline: runs the stages in order, skips stages that finished in an earlier run (resume),
and writes everything into one output folder.

Stage order (name used with --stages):
  audit               dataset audit: geometry, masks, slice overlap, duplicate scans, cross-validation folds
  subject_table       subjects.csv (group / role / fold), header checks, intensity-scale table
  header_geometry     orientation, slab tilt, index-order check
  rigid_registration  T1w -> SPIR rigid (SimpleITK)
  atlas_registration  CIT168 -> T1w (ANTs SyN) -> SPIR, retries, QC
  crop                crop, z-score and crus-ratio normalisation, masks, QC table
  gmm_refinement      GMM masks and refined training labels (only subjects with a manual mask)
  nnunet_training     only if the trained model is not found
  nnunet_inference    5-fold ensemble, post-processing, final masks on the SPIR grid
  visual_qc           one QC page per segmented subject
  final_summary       final_summary.csv and run_report.txt
"""
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from . import (atlas_registration, crop_and_normalise, final_summary, gmm_label_refinement, header_geometry, nnunet_inference,
               nnunet_training, rigid_registration, subject_inventory, visual_qc)
from .audit import audit_common, duplicate_scan_check, geometry_audit, make_folds, mask_audit, slice_overlap_check
from .config import Context, load_config, set_global_seed
from .utils import log


def stage_audit(ctx):
    audit_common.apply_settings(ctx.cfg)
    root = subject_inventory.find_data_root(ctx.data_root, ctx.cfg["dataset"]["group_rules"])
    out = ctx.paths.audit
    geometry_audit.run(root, out, ctx.limit)
    mask_audit.run(root, out, ctx.limit)
    slice_overlap_check.run(root, out, ctx.limit)
    duplicate_scan_check.run(root, out, ctx.limit)
    ds = ctx.cfg["dataset"]
    if ds.get("folds_file"):
        log(f"folds file given in the config: {ds['folds_file']} (not regenerated)")
    else:
        make_folds.run(out / "manifest.csv", ctx.paths.splits, seed=int(ctx.cfg["seeds"]["fold_seed"]), n_folds=int(ds["n_folds"]))


STAGES = [
    ("audit", stage_audit),
    ("subject_table", subject_inventory.run),
    ("header_geometry", header_geometry.run),
    ("rigid_registration", rigid_registration.run),
    ("atlas_registration", atlas_registration.run),
    ("crop", crop_and_normalise.run),
    ("gmm_refinement", gmm_label_refinement.run),
    ("nnunet_training", nnunet_training.run),
    ("nnunet_inference", nnunet_inference.run),
    ("visual_qc", visual_qc.run),
    ("final_summary", None),          # handled below (needs the stage times)
]
STAGE_NAMES = [s for s, _ in STAGES]


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def default_output(dataset):
    """no output folder given: a folder next to the dataset folder (the dataset itself is never written to)"""
    d = Path(dataset).resolve()
    return d.parent / f"{d.name}_sn_pipeline_output"


def run_pipeline(dataset, atlas, output=None, model_dir=None, config=None, stages=None, force=(), device="auto", limit=0):
    cfg = load_config(config)
    output = Path(output) if output else default_output(dataset)
    ctx = Context(cfg, dataset, atlas, output, model_dir=model_dir, device=device, limit=limit)
    ctx.paths.make()
    log_file = ctx.paths.logs / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
    fh = open(log_file, "w", encoding="utf-8")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(old_out, fh), _Tee(old_err, fh)
    timings = {}
    try:
        for p, what in ((ctx.data_root, "dataset"), (ctx.atlas_dir, "atlas")):
            if not Path(p).is_dir():
                raise FileNotFoundError(f"{what} folder not found: {p}")
        set_global_seed(ctx.seed)
        selected = list(stages) if stages else STAGE_NAMES
        unknown = [s for s in selected + list(force) if s not in STAGE_NAMES]
        if unknown:
            raise ValueError(f"unknown stage(s) {unknown}; stages are {STAGE_NAMES}")
        log(f"dataset {ctx.data_root}\natlas   {ctx.atlas_dir}\noutput  {ctx.paths.root}\nmodel   {ctx.model_dir or '(train or look in the output folder)'}\n"
            f"stages  {selected}\nlog     {log_file}")
        for name, fn in STAGES:
            if name not in selected:
                continue
            marker = ctx.paths.logs / f"{name}.done"
            if marker.exists() and name not in force and name != "final_summary":
                log(f"\n===== {name}: finished in an earlier run, skipped (use --force {name} to redo) =====")
                continue
            log(f"\n===== {name} =====")
            t0 = time.time()
            set_global_seed(ctx.seed)
            if name == "final_summary":
                final_summary.run(ctx, timings)
            else:
                fn(ctx)
            timings[name] = time.time() - t0
            marker.write_text(datetime.now().isoformat(timespec="seconds"))
        if not cfg["output"]["keep_work_files"] and "nnunet_inference" in selected:
            shutil.rmtree(ctx.paths.work, ignore_errors=True)
        log(f"\nDone. Final masks: {ctx.paths.sn_masks}")
        return 0
    except Exception:
        traceback.print_exc()
        log(f"\nPipeline stopped. Full log: {log_file}")
        return 1
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        fh.close()
