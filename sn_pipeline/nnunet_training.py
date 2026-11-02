"""
nnU-Net training on the refined GMM labels. Runs ONLY when the trained model is not found in the model folder.

- Network input: ONE channel, the z-score SPIR crop (112 x 80 x 50). Nothing else.
- Training target per voxel: 1 = SN (refined GMM mask), 0 = background, 2 = "ignore" (no loss). Ignore =
  (a) candidate bright tissue: GMM mask minus refined mask (the GMM bright class inside the atlas outer limit that the
      refinement removed), and
  (b) the atlas outer limit on the 2 slices above the top refined-mask slice.
- The manual masks are NEVER read here (no input, no target, no ignore region).
- The atlas is NEVER a network input; it is used only for location (outer limit, SNc anchor).
- Training subjects: development subjects with labels, minus the under-segmented ("weak") subjects listed in the config.
- 5 folds from the frozen folds file, trainer nnUNetTrainer_100epochs (100 epochs, checked in its source), one fold per GPU in parallel.
  Finished folds are skipped; unfinished folds continue from their last checkpoint.

Outputs: nnunet/labels/<dataset>/<subject>.nii.gz, nnunet/labels/<dataset>_label_table.csv, nnunet/settings/<dataset>_settings.json,
         nnunet/logs/*.log, and the trained model in <nnUNet_results>/<dataset>/.
"""
import datetime
import inspect
import json
import os
import re
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

from .config import sha_settings
from .nnunet_common import CropData, fold_done, model_complete, results_dir, run_cmd, setup_environment
from .utils import load_nii, save_nii, log


def check_trainer(trainer):
    import nnunetv2
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    _tr = recursive_find_python_class(os.path.join(nnunetv2.__path__[0], "training", "nnUNetTrainer"), trainer,
                                      "nnunetv2.training.nnUNetTrainer")
    assert _tr is not None, f"{trainer} not found in this nnunetv2 installation"
    assert re.search(r"num_epochs\s*=\s*100\b", inspect.getsource(_tr)), f"{trainer} does not set 100 epochs"
    log(f"trainer {trainer}: 100 epochs (checked in its source)")


def run(ctx):
    ncfg = ctx.cfg["nnunet"]
    raw_root, pre_root, res_root = setup_environment(ctx)
    if model_complete(ncfg):
        log(f"trained model found in {res_root}: training skipped")
        return
    if not ncfg["train_if_model_missing"]:
        raise RuntimeError(f"no trained model {ncfg['dataset_name']} in {res_root} and training is switched off")
    import torch
    n_gpu = torch.cuda.device_count()
    assert n_gpu >= 1, ("no GPU: nnU-Net training needs a CUDA GPU. Give the folder of the trained model (MODEL_DIR) "
                        "or run the training on a GPU machine.")
    check_trainer(ncfg["trainer"])
    P = ctx.paths
    OUT = P.nnunet
    DID, DNAME, CFG_NAME, TRAINER = int(ncfg["dataset_id"]), ncfg["dataset_name"], ncfg["configuration"], ncfg["trainer"]
    N_ABOVE = int(ncfg["ignore_slices_above"])
    CD = CropData(ctx)
    D = CD.ALL[(CD.ALL.role == "dev") & CD.ALL.include_training.astype(bool)].copy()
    have = [s for s in D.index if (P.gmm / "seg_crop" / "gmm_refined" / f"{s}_gmm_refined_lr.nii.gz").exists()]
    weak = set(ncfg["weak_subjects"])
    TRAIN = [s for s in have if s not in weak]
    assert TRAIN, "no training subject with refined GMM labels"
    D["fold"] = D.fold.astype(int)
    log(f"training subjects: {len(TRAIN)} (weak subjects left out: {sorted(weak & set(have))}) | per fold: "
        f"{D.loc[TRAIN].fold.value_counts().sort_index().to_dict()}")

    # ---- labels (manual masks are not read)
    DATA = {}
    for s in TRAIN:
        d = CD.load_subject(s)
        fin, _ = load_nii(P.gmm / "seg_crop" / "gmm_refined" / f"{s}_gmm_refined_lr.nii.gz", np.uint8)
        gmm, _ = load_nii(P.gmm / "seg_crop" / "gmm" / f"{s}_gmm_lr.nii.gz", np.uint8)
        _, faff_saved = load_nii(P.gmm / "seg_spir" / "gmm_refined" / f"{s}_gmm_refined_lr.nii.gz", np.uint8)
        assert np.allclose(faff_saved, d["faff"], atol=1e-4), f"{s}: computed full-grid affine differs from the saved one"
        d["fin"], d["gmm"] = fin, gmm
        SN = fin > 0
        ks = np.flatnonzero(SN.any(axis=(0, 1))); d["ktop"] = int(ks.max()); d["kbot"] = int(ks.min())
        d["cand"] = (gmm > 0) & ~SN & d["reg"]                       # (a) candidate bright tissue
        above = np.zeros(SN.shape, bool); above[..., d["ktop"] + 1: d["ktop"] + 1 + N_ABOVE] = True
        d["above"] = above & d["reg"] & ~SN                                # (b) slices above the top refined-mask slice
        DATA[s] = d

    def base_label(d):
        y = np.zeros(d["fin"].shape, np.uint8)
        y[d["cand"] | d["above"]] = 2
        y[d["fin"] > 0] = 1
        return y

    SETTINGS = {"dataset": DNAME, "training_subjects": sorted(TRAIN),
                "input": "one channel: z-score SPIR crop (nnU-Net channel name 'zscore'); atlas and masks are not inputs",
                "labels": {"background": 0, "SN": 1, "ignore": 2}, "SN": "refined GMM mask (2.0 mm bridging)",
                "ignore": f"(a) GMM mask minus refined mask inside the atlas outer limit; (b) atlas outer limit on {N_ABOVE} slices above the "
                          "top refined-mask slice",
                "manual_masks": "not used for labels or ignore region", "trainer": TRAINER, "configs": [CFG_NAME],
                "folds": "frozen folds file", "segmentation_rules": ctx.cfg["segmentation_rules"]}
    SET_HASH = sha_settings(SETTINGS)
    sp = OUT / "settings" / f"{DNAME}_settings.json"; sp.parent.mkdir(parents=True, exist_ok=True)
    if sp.exists():
        old = json.load(open(sp))
        assert old["settings_sha256"] == SET_HASH, f"{DNAME} was started with different settings ({sp}); use a new dataset id"
    else:
        json.dump(dict(settings=SETTINGS, settings_sha256=SET_HASH, date=datetime.date.today().isoformat()), open(sp, "w"), indent=2, default=str)

    RAW = raw_root / DNAME
    for sub in ("imagesTr", "labelsTr"):
        (RAW / sub).mkdir(parents=True, exist_ok=True)
    rows = []
    for s, d in DATA.items():
        y = base_label(d)
        shutil.copy2(d["zpath"], RAW / "imagesTr" / f"{s}_0000.nii.gz")
        save_nii(y, d["caff"], RAW / "labelsTr" / f"{s}.nii.gz", np.uint8)
        save_nii(y, d["caff"], OUT / "labels" / DNAME / f"{s}.nii.gz", np.uint8)   # kept with the output
        rows.append(dict(subject_id=s, fold=int(D.loc[s, "fold"]), group=D.loc[s, "group"], scale_group=D.loc[s, "scale_group"],
                         sn_mm3=float((y == 1).sum() * d["vvox"]), refined_mm3=float((d["fin"] > 0).sum() * d["vvox"]),
                         ignore_mm3=float((y == 2).sum() * d["vvox"]), ignore_candidate_mm3=float((d["cand"] & (y == 2)).sum() * d["vvox"]),
                         ignore_above_mm3=float((d["above"] & (y == 2)).sum() * d["vvox"])))
        assert ((y == 1) | ~(d["fin"] > 0)).all(), f"{s}: a refined-mask voxel is not SN"
    LT = pd.DataFrame(rows).sort_values(["fold", "subject_id"]); LT.to_csv(OUT / "labels" / f"{DNAME}_label_table.csv", index=False)
    json.dump({"channel_names": {"0": "zscore"}, "labels": {"background": 0, "SN": 1, "ignore": 2}, "numTraining": len(DATA),
               "file_ending": ".nii.gz", "name": DNAME, "description": "SN (NM-bright) on SPIR crops; " + SETTINGS["SN"]},
              open(RAW / "dataset.json", "w"), indent=2)
    log(f"{DNAME}: {len(DATA)} cases written\n" + LT[["sn_mm3", "refined_mm3", "ignore_mm3"]].describe().loc[["50%", "25%", "75%"]].round(1).to_string())

    # ---- plan + preprocess, and the frozen folds as splits_final.json
    PRE = pre_root / DNAME
    if not (PRE / "nnUNetPlans.json").exists() or not (PRE / f"nnUNetPlans_{CFG_NAME}").exists():
        run_cmd(["nnUNetv2_plan_and_preprocess", "-d", str(DID), "-c", CFG_NAME, "--verify_dataset_integrity"],
                OUT / "logs" / f"{DNAME}_plan_preprocess.log")
    splits = []
    for k in ncfg["folds"]:
        val = sorted(s for s in TRAIN if int(D.loc[s, "fold"]) == k)
        splits.append({"train": sorted(s for s in TRAIN if s not in val), "val": val})
    assert all(len(sp_["val"]) > 0 for sp_ in splits), "a fold has no validation subject"
    json.dump(splits, open(PRE / "splits_final.json", "w"), indent=2)
    plans = json.load(open(PRE / "nnUNetPlans.json"))
    pc = plans["configurations"][CFG_NAME]
    log(f"{CFG_NAME}: patch {pc.get('patch_size')} | batch {pc.get('batch_size')} | spacing {pc.get('spacing')} | "
        f"normalisation {pc.get('normalization_schemes')}")

    # ---- training: one fold per GPU in parallel; finished folds skipped; unfinished folds continued (--c)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    jobs = []
    for k in ncfg["folds"]:
        rd = results_dir(ncfg)
        if fold_done(ncfg, k):
            log(f"{CFG_NAME} fold {k}: finished earlier, skipped"); continue
        if rd is not None and (rd / f"fold_{k}" / "checkpoint_latest.pth").exists():
            extra = ["--c", "--npz"]                         # continue from last checkpoint
        else:
            extra = ["--npz"]
        jobs.append((k, ["nnUNetv2_train", str(DID), CFG_NAME, str(k), "-tr", TRAINER, *extra]))
    n_da = str(max(1, (os.cpu_count() or 2) // max(1, min(n_gpu, len(jobs) or 1))))
    log(f"{len(jobs)} training jobs on {n_gpu} GPU(s); data-augmentation workers per job: {n_da}")
    running, t_start = {}, time.time()
    queue = list(jobs)
    while queue or running:
        for g in range(n_gpu):
            if g not in running and queue:
                k, cmd = queue.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), nnUNet_n_proc_DA=n_da, PYTHONUNBUFFERED="1")
                lp = OUT / "logs" / f"{DNAME}_{CFG_NAME}_fold{k}.log"
                log(f"[{time.strftime('%H:%M:%S')}] GPU {g}: {' '.join(cmd)}")
                running[g] = (k, subprocess.Popen(cmd, env=env, stdout=open(lp, "a"), stderr=subprocess.STDOUT), lp)
        time.sleep(120)
        for g in list(running):
            k, p, lp = running[g]
            txt = open(lp, errors="ignore").read()
            ep = re.findall(r"Epoch (\d+)\s*\n", txt); et = re.findall(r"Epoch time: ([\d.]+) s", txt)
            pdice = re.findall(r"Pseudo dice \[(.*?)\]", txt)
            log(f"  [{(time.time() - t_start) / 60:5.0f} min] fold {k}: epoch {ep[-1] if ep else '-'} / 100 | "
                f"epoch time {et[-1] if et else '-'} s | pseudo dice {pdice[-1] if pdice else '-'}")
            if p.poll() is not None:
                if p.returncode:
                    print(txt[-4000:]); raise RuntimeError(f"fold {k} failed, log {lp}")
                log(f"  fold {k}: done"); del running[g]
    log(f"folds finished: {[k for k in ncfg['folds'] if fold_done(ncfg, k)]}")
    assert model_complete(ncfg), "not all folds finished"
