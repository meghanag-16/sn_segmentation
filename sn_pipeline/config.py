"""
Configuration and output-folder layout shared by every pipeline stage.

The frozen settings live in config/pipeline_config.json. This module loads them, fixes the random seeds,
and defines where each stage writes its results inside the output folder.
"""
import copy
import hashlib
import json
import os
import random
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PACKAGE_ROOT / "config" / "pipeline_config.json"


def load_config(path=None):
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.pop("_about", None)
    return cfg


def sha_settings(d):
    """sha256 of json.dumps(settings, sort_keys=True): the hash rule used for every settings record."""
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def set_global_seed(seed):
    """Fixes python, numpy and (if present) torch random generators."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


class RunPaths:
    """Folder layout of one pipeline run. Only the important intermediate results are kept."""

    def __init__(self, output_root):
        self.root = Path(output_root)
        self.audit = self.root / "audit"                         # dataset audit, splits, subject table, header geometry
        self.splits = self.audit / "splits"
        self.subject_table = self.audit / "subject_table"
        self.header_geometry = self.audit / "header_geometry"
        self.registration = self.root / "registration"
        self.rigid = self.registration / "rigid_t1w_to_spir"     # T1w -> SPIR rigid transforms and QC
        self.atlas = self.registration / "atlas_in_spir"         # CIT168 atlas carried into SPIR space and QC
        self.cropped = self.root / "cropped"                     # crops, normalised images, masks, QC table
        self.gmm = self.root / "gmm_refined"                     # GMM masks and refined training labels
        self.nnunet = self.root / "nnunet"                       # labels, predictions, post-processed segmentations
        self.sn_masks = self.root / "sn_masks"                   # FINAL OUTPUT: one mask per patient on the SPIR grid
        self.visual_qc = self.root / "visual_qc"
        self.summary = self.root / "final_summary"
        self.work = self.root / "_work"                          # temporary files (nnU-Net raw / preprocessed); removed at the end
        self.logs = self.root / "logs"

    def subjects_csv(self):
        return self.subject_table / "subjects.csv"

    def make(self):
        for p in (self.root, self.audit, self.splits, self.subject_table, self.header_geometry, self.rigid, self.atlas,
                  self.cropped, self.gmm, self.nnunet, self.sn_masks, self.visual_qc, self.summary, self.work, self.logs):
            p.mkdir(parents=True, exist_ok=True)


class Context:
    """Everything a stage needs: settings, input paths and output layout."""

    def __init__(self, cfg, data_root, atlas_dir, output_root, model_dir=None, device="auto", limit=0):
        self.cfg = copy.deepcopy(cfg)
        self.data_root = Path(data_root)
        self.atlas_dir = Path(atlas_dir)
        self.paths = RunPaths(output_root)
        self.model_dir = Path(model_dir) if model_dir else None
        self.device = device
        self.limit = int(limit or 0)
        self.seed = int(cfg["seeds"]["pipeline_seed"])

    def config_hash(self):
        return sha_settings(self.cfg)
