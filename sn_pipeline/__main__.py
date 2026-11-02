"""
Command line:  python -m sn_pipeline --dataset <folder> --atlas <folder> [--output <folder>] [--model-dir <folder>]

  --dataset    folder that holds the group folders (Healthy_controls, Schiz, testing) or a folder above it
  --atlas      folder with CIT168_T1w_head_700um.nii, prob_atlas_bilateral.nii, prob_atlas.nii, labels.txt
  --output     where results are written (default: <dataset folder name>_sn_pipeline_output next to the dataset folder)
  --model-dir  nnU-Net results folder that contains the trained model folder (Dataset721_...); if the model is not there
               it is trained (GPU needed) into this folder, or into <output>/nnunet/nnUNet_results when not given
  --config     settings file (default: config/pipeline_config.json)
  --stages     run only these stages (names in sn_pipeline/pipeline.py)
  --force      redo these stages even if they finished in an earlier run
  --device     auto | cuda | cpu | mps (nnU-Net inference device)
  --limit      only the first N subjects (quick test)
"""
import argparse
import sys

from .pipeline import STAGE_NAMES, run_pipeline


def main(argv=None):
    p = argparse.ArgumentParser(prog="sn_pipeline", description="Substantia nigra segmentation on T1W_3D_SPIR (end-to-end pipeline)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--atlas", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--stages", nargs="*", default=None, choices=STAGE_NAMES)
    p.add_argument("--force", nargs="*", default=[], choices=STAGE_NAMES)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args(argv)
    for name in ("output", "model_dir", "config"):          # empty strings from the launcher mean "not given"
        if getattr(a, name) is not None and not str(getattr(a, name)).strip():
            setattr(a, name, None)
    return run_pipeline(a.dataset, a.atlas, a.output, a.model_dir, a.config, a.stages, a.force, a.device, a.limit)


if __name__ == "__main__":
    sys.exit(main())
