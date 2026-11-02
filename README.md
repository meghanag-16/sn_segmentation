# SN segmentation pipeline (T1W_3D_SPIR, neuromelanin-bright substantia nigra)

End-to-end, automated segmentation of the left and right substantia nigra (the neuromelanin-bright part) in the
native T1W_3D_SPIR space. The saved notebooks of the project (folder `final_code`) are chained here into one small
Python package with one launcher. The processing logic, the settings and the decisions are the frozen ones from the
notebooks and logs; only the structure (modules, folders, file names) changed.

**Input:** a dataset folder and the CIT168 atlas folder.
**Output:** one mask per patient on the SPIR grid, `sn_masks/<subject_id>_sn_mask.nii.gz`
(0 = background, 1 = left SN, 2 = right SN, same 320 x 320 x 50 grid and header as the patient's `T1W_3D_SPIR.nii`).

---

## 1. Folder contents

```
refactored_code/
  run_pipeline.bat            launcher for Windows AND Linux (paths are set here)
  run_prediction_only.bat     prediction only, from registration / crop outputs that already exist (section 3b)
  requirements.txt            Python packages
  config/pipeline_config.json all frozen settings (one place)
  sn_pipeline/                the package
    __main__.py               command line (python -m sn_pipeline ...)
    pipeline.py               runs the stages in order, resume, logging
    config.py                 settings loader, seeds, output-folder layout
    utils.py                  shared helpers (NIfTI I/O, grid arithmetic, atlas limit, SNc anchor, post-processing rules)
    audit/                    dataset audit
      audit_common.py         subject discovery, JSON identification by content, shared helpers
      geometry_audit.py       file inventory, shapes, voxel sizes, orientation, JSON header fields
      mask_audit.py           structure of the manual masks
      slice_overlap_check.py  neighbour correlation along the slab axis
      duplicate_scan_check.py same scan under two names?
      make_folds.py           frozen 5-fold split of the development subjects (seed 723635)
    subject_inventory.py      subjects.csv (group / role / fold), header checks, PhilipsScaleSlope table
    header_geometry.py        orientation, slab tilt, index-order check
    rigid_registration.py     T1w -> SPIR rigid registration (SimpleITK)
    atlas_registration.py     CIT168 template -> T1w (ANTs SyN) -> SPIR, chain test, retries, QC
    crop_and_normalise.py     crop, z-score and crus-ratio normalisation, binary / left-right masks, QC table
    gmm_label_refinement.py   GMM masks and refined training labels (only subjects with a manual mask)
    nnunet_common.py          nnU-Net folders, model lookup, subject loading on the crop grid
    nnunet_training.py        nnU-Net training (only when no trained model is found)
    nnunet_inference.py       5-fold ensemble, post-processing, final masks, label-free checks, NM measures
    visual_qc.py              one QC page per segmented subject
    final_summary.py          final_summary.csv and run_report.txt
    predict_from_saved.py     prediction only from existing outputs (used by run_prediction_only.bat)
```

---

## 2. Pipeline architecture

```
 dataset folder + CIT168 atlas folder
            |
 [1] audit ................ geometry / mask / slice-overlap / duplicate audits, folds.csv        -> audit/
 [2] subject_table ........ subjects.csv (subject_id = <group>_<folder>), header checks,        -> audit/subject_table/
                            PhilipsScaleSlope table (intensity-scale group)
 [3] header_geometry ...... orientation codes, slab tilt, index-order check                      -> audit/header_geometry/
 [4] rigid_registration ... T1w -> SPIR rigid (SPIR never resampled)                             -> registration/rigid_t1w_to_spir/
 [5] atlas_registration ... CIT168 template -> T1w (SyN), chained with [4] into SPIR space;       -> registration/atlas_in_spir/
                            label-free retry rule (head_init start-up); QC pages
 [6] crop ................. 112 x 80 x 50 crop at the atlas SN centre; z-score (network input);   -> cropped/
                            crus-cerebri ratio (measurements); hemisphere map; masks; QC table
 [7] gmm_refinement ....... GMM masks + refined labels (manual mask used as upper bound only)    -> gmm_refined/
 [8] nnunet_training ...... ONLY if the trained model is missing (GPU)                           -> nnunet/ + model folder
 [9] nnunet_inference ..... 5-fold ensemble, p >= 0.5, post-processing, full SPIR grid           -> nnunet/, sn_masks/  (FINAL)
[10] visual_qc ............ one page per segmented subject                                       -> visual_qc/
[11] final_summary ........ one row per subject + run report                                     -> final_summary/
```

### What each stage does (frozen settings)

1. **audit** - reads every subject folder (never writes into it). Subject folders are found by file name
   (`T1W_3D_SPIR`, `T1W`, `sn_mask`, `.nii` or `.nii.gz`), with a fall-back by image shape. JSON files are recognised
   by their **content** (ProtocolName / SeriesDescription contains SPIR, or T1FFE + fat suppression, or flip angle 20
   and TR 25.4 ms = SPIR; T1TFE or flip angle 9 = T1w), so their names do not matter. Writes `manifest.csv`,
   `geometry_audit.csv`, `json_fields.csv`, `mask_audit.csv`, `slice_overlap_per_subject.csv`, summaries and flags.
   Then `splits/folds.csv`: development subjects (mask present, not excluded, not test) in 5 folds, seed 723635,
   group mix balanced; an existing folds.csv is never overwritten.
2. **subject_table** - `subjects.csv` with `subject_id = <group>_<folder name>` (the same folder name can be a
   different person in another group), role `dev` / `test` / `excluded`, fold, relative paths; header checks
   (shape, voxel size, mask header vs SPIR header); PhilipsScaleSlope per subject: first from the JSON next to the
   SPIR whose name contains "spir" (if several, the one not starting with "sub-"), otherwise from the audit's
   content-identified SPIR JSON (`json_fields.csv`). Scale group: slope > 0.01 = high_slope (covariate only).
3. **header_geometry** - orientation codes, qform/sform consistency, pitch / roll / in-plane / total tilt
   (`tilt_covariates.csv`), and a test that the SPIR index order matches its header (normalised mutual information
   with the header-aligned T1w, unflipped vs flipped).
4. **rigid_registration** - SimpleITK Euler 3D, Mattes MI (50 bins), random sampling with the pipeline seed 595026,
   SPIR head mask, SPIR world = **sform** for all subjects; a second optimizer setting runs only when the first
   looks wrong; the better NMI is kept. Flags: NMI not improved, corner displacement > 10 mm, rotation > 5 deg,
   iteration limit; flagged subjects get a reproducibility check with the other setting.
5. **atlas_registration** - `ants.registration(type_of_transform="SyN")` of the CIT168 T1w head template
   (resampled to 1 mm) to the subject T1w; the chain [SPIR->T1w rigid, warp, affine] puts the atlas probability
   maps (SNc, SNr, RN, PBP, VTA, left SNc/SNr) on the SPIR grid in one interpolation, plus an index map (CIT168 voxel
   index of every SPIR voxel). The chain order is verified on the subject with the largest head motion. Label-free
   retry rule: midbrain template-T1w correlation < median - 3 robust SD, or SNc in slab < 0.90, or SNc at the slab
   edge -> re-register with `head_init` (upper-head centroid start-up, head-masked metric) and keep it only if the
   slab correlation is better by 0.02. Masks are used for QC numbers only.
6. **crop** - one crop size for all (112 x 80 voxels in-plane, all 50 slices; restore:
   `full[i0:i0+112, j0:j0+80, :] = crop`), placed at the registered atlas SNc+SNr centre (label-free).
   Network input = z-score of the raw crop (mean / SD of crop voxels > 5 % of the crop p99, between their 1st and
   99th percentile; not clipped). Measurement scale = ratio to the crus cerebri reference ROI (CIT168 white matter
   1-3 mm in front of the SN, carried by the index map, CSF-dark voxels removed, 1-voxel erosion with fall-back).
   Hemisphere map from the CIT168 midline (1 = left, 2 = right). Masks: binary (value != 0), SPIR header, split
   left/right by the midline. QC table with inclusion: inference is excluded only for a failed registration, a crop
   that cuts the SN region, or a manual reason; training also needs a non-excluded development mask.
   N4 and denoising are off.
7. **gmm_refinement** (subjects with a manual mask only) - per subject a 3-component Gaussian mixture on the crus
   ratio inside the atlas outer limit (CIT168 SNc+SNr p >= 0.25, dilated 1 mm, minus red nucleus p >= 0.5);
   brightest class posterior >= 0.5; per side, pieces touching the SNc anchor (SNc p >= 0.5, fall-back 0.25);
   holes filled in-plane -> `gmm`. Refined label: `gmm` inside (manual mask AND outer limit), fragments on a slice
   joined only through corridors between pieces <= 2.0 mm apart, outlines never grown -> `gmm_refined`.
8. **nnunet_training** - skipped when the trained model is found. Otherwise nnU-Net (3d_fullres,
   nnUNetTrainer_100epochs, nnUNetPlans) on the development subjects minus the 7 under-segmented ones listed in the
   config; labels 0 background, 1 SN (= `gmm_refined`), 2 ignore (= `gmm` minus `gmm_refined` inside the outer limit,
   plus the outer limit on the 2 slices above the top label slice); input = z-score crop only; folds from folds.csv.
   Manual masks and the atlas are never network inputs.
9. **nnunet_inference** - `nnUNetv2_predict` with the 5 folds, probabilities averaged (default mirroring),
   SN probability >= 0.5; post-processing per side: inside the atlas outer limit, 26-connected pieces touching the
   SNc anchor (fall-back: largest piece), holes filled in-plane; pasted back on the full SPIR grid. Label-free checks:
   both sides present, contrast inside/shell > 1.0, L/R volume ratio 0.69-1.27, largest piece >= 0.90 per side.
   NM measures per side (raw SPIR, crus ratio, CNR = ratio - 1).
10. **visual_qc** - image | probability | post-processed, every slice with segmentation (+2 above / below).
11. **final_summary** - `final_summary.csv` (one row per subject: role, registration strategy and grade, QC reasons,
    inclusion, volumes, L/R, flags, NM measures, path of the mask) and `run_report.txt`.

---

## 3. How to run

### First time
1. Install Python 3.10 - 3.13 (Windows: python.org, tick "Add python to PATH").
2. Open `run_pipeline.bat` in a text editor and set the paths in the part for your system:
   - `DATASET_DIR` - the folder that holds `Healthy_controls`, `Schiz`, `testing` (or a folder above it)
   - `ATLAS_DIR`   - the CIT168 folder (`CIT168_T1w_head_700um.nii`, `prob_atlas_bilateral.nii`, `prob_atlas.nii`, `labels.txt`)
   - `OUTPUT_DIR`  - optional; empty = a new folder `<dataset folder name>_sn_pipeline_output` next to the dataset folder
   - `MODEL_DIR`   - the nnU-Net results folder that contains `Dataset721_chealdev49r1` (the frozen trained model,
     i.e. `Dataset721_chealdev49r1/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/fold_0 ... fold_4/checkpoint_final.pth`
     plus the `plans.json` / `dataset.json` nnU-Net saved next to the folds). Empty = train a new model (CUDA GPU needed).
   - `DEVICE`      - `auto` (GPU if present), `cpu` or `cuda`
3. Run it:
   - **Windows:** double-click `run_pipeline.bat` (or type `run_pipeline.bat` in Command Prompt in this folder)
   - **Linux:** `bash run_pipeline.bat`

   The first run creates a virtual environment `.venv` in this folder and installs `requirements.txt`
   (internet needed once). Later runs reuse it.

### Options (added after the launcher name, or with `python -m sn_pipeline`)
```
--stages audit subject_table ...   run only these stages
--force crop gmm_refinement        redo these stages even if they finished before
--limit 3                          quick test on the first 3 subjects (use a separate OUTPUT_DIR for this)
--config my_settings.json          another settings file
--device cpu                       inference device
```
Direct call without the launcher:
`python -m sn_pipeline --dataset <data> --atlas <atlas> [--output <out>] [--model-dir <model>]`

### Resume
Every stage writes `logs/<stage>.done` when it finishes; a new run skips finished stages. Inside the registration
stages, finished subjects are skipped as well. Use `--force <stage>` to redo a stage (later stages that depend on it
should be forced too). Each run writes a full log to `logs/pipeline_<date_time>.log`.

### Time and hardware
Registration and cropping run on the CPU. The atlas registration is the slow step (on Kaggle with 4 threads it took about
2.5 h for 76 subjects). nnU-Net inference runs on CPU or GPU (GPU is much faster). Training needs a CUDA GPU
(about 5 h for 5 folds on 2 x T4).

---

## 3b. Prediction only (registration and crops already done)

When the registration and crop outputs exist already (from the saved notebooks or from an earlier run of this
pipeline), use `run_prediction_only.bat` instead. It skips audit, registration, cropping, GMM labels and training,
and runs only: nnU-Net prediction -> post-processing -> visual QC -> final summary.

Put the earlier outputs in one folder (any sub-folder structure; files are found by name) and set `SAVED_DIR` to it:

| needed file / folder | from the saved notebooks | from an earlier run of this pipeline |
|---|---|---|
| subject table | `subjects.csv` (data-tree output) | `audit/subject_table/subjects.csv` |
| crop QC table + crop windows + crops | `qc_table_1_8.csv`, `crop_windows.csv`, `crops/` (crop-mask-prep output, same folder) | `cropped/qc_table.csv`, `cropped/crop_windows.csv`, `cropped/crops/` |
| atlas channel list | `phase_1_3b_decisions.json` (registration output) | `registration/atlas_in_spir/atlas_registration_decisions.json` |
| optional: registration table | `atlas_registration_1_3b.csv` | `registration/atlas_in_spir/atlas_registration.csv` |
| trained model | `nnUNet_results/Dataset721_chealdev49r1/...` (training output) | the model folder |

Only these crop kinds are read, for the subjects being segmented: `zscore`, `refratio`, `raw`, `atlas`, `hemi`,
`ref_roi`. The large registration files (4D atlas maps, templates, transforms) are not needed.

Settings in `run_prediction_only.bat`: `SAVED_DIR`, `OUTPUT_DIR`, `MODEL_DIR` (empty = search for the
`Dataset721_chealdev49r1` folder below `SAVED_DIR`), `SUBJECTS` (`test` = only test subjects, `all` = every subject
included for inference), `DEVICE`. Run it like the main launcher (double-click on Windows, `bash run_prediction_only.bat`
on Linux). If a file is found more than once below `SAVED_DIR`, the run stops and names the copies; then give it
directly (`--subjects-csv`, `--qc-table`, `--atlas-decisions` after the launcher name).
The output folder has the same layout as a full run (`sn_masks/`, `nnunet/`, `visual_qc/`, `final_summary/`).

---

## 4. Rules for running

**Dataset layout** (group folders are matched case-insensitively; set in `config/pipeline_config.json`, `group_rules`):
```
<dataset>/Healthy_controls/<ID>/...           -> group control
<dataset>/Schiz/acute/<ID>/...                -> group acute
<dataset>/Schiz/chronic/<ID>/...              -> group chronic
<dataset>/testing/HC/<ID>/...                 -> group test_HC
<dataset>/testing/SCZ/<ID>/...                -> group test_SCZ
```
The folder right after the group prefix is the subject folder; the scans can sit directly in it or in a subfolder
(for example `Neuromelanin`). A folder that matches no rule is reported as an error and not processed.

**File names inside each subject folder** (exact, case-sensitive for the subject table):
- `T1W_3D_SPIR.nii` - required
- `T1W.nii` - required (only to place the atlas in SPIR space)
- `sn_mask.nii` - only if present (development subjects); used as a loose upper bound for training labels and for
  QC numbers, never as a network input or for any test subject
- `.nii.gz` is accepted as well; exactly one file of each kind per folder.
- JSON files: any name. The SPIR JSON is found by content (see stage 1) or by "spir" in its name; it is only needed
  for the PhilipsScaleSlope (a covariate). A missing slope gives scale group "unknown" and does not stop the run.

**Subject IDs and roles**: `subject_id = <group>_<folder name>` (for example `chronic_SCZ-27M`, `test_HC_HC-23M`).
Subjects in the test groups are role `test` and are never used for training, fitting or choices. Masks listed in
`excluded_masks` (chronic_SCZ-60M) are role `excluded` (image still segmented, mask not used). Masks listed in
`mask_header_override` take the SPIR header.

**Never written to**: the dataset folder, the atlas folder and the model folder given by the user (except when a new
model is trained into an empty model folder). The saved notebooks in `final_code` are not used or changed.

**Privacy**: the audit extracts any zip archives found in subject folders into `audit/_unzipped`; keep that folder
(and the outputs) out of any public upload. Printed and saved tables contain subject IDs only, no names.

**Expected values**: the counts and shapes in the config (76 folders, 320 x 320 x 50 SPIR, 192 x 256 x 256 T1w) are
the ones of the pilot dataset. For another cohort the count checks show FAIL in `setup_report.txt` but the run
continues; the crop size (112 x 80) and the trained model stay fixed.

**Manual visual grades**: the QC pages of the registration and crop stages are meant to be looked at. Grades are
taken from the config (`viewed_qc_pages`, `grade_exceptions`); a subject whose grade text starts with `fail` is
excluded from the later stages. For a new cohort, look at the pages and add exceptions, then `--force` the stage.

---

## 5. Output folder

```
<output>/
  audit/                     audit tables and summaries, splits/folds.csv, subject_table/subjects.csv, header_geometry/
  registration/
    rigid_t1w_to_spir/       transforms (.tfm), t1w_in_spir, tables, figs/qc_levels_page*.png
    atlas_in_spir/           atlas_in_spir (4D maps + index maps), template_in_spir, transforms, retry/, tables, figs/
  cropped/                   crops/{raw,zscore,refratio,atlas,hemi,ref_roi,mask_bin,mask_lr}, masks_full, qc_table.csv, figs/
  gmm_refined/               seg_crop|seg_spir/{gmm,gmm_refined}, metrics.csv, summary.csv, qc_pages/
  nnunet/                    predictions/, postprocessed/{crop,spir}, prob_crop/, segmentation_metrics.csv, nm_measures.csv
                             (labels/, settings/, logs/, nnUNet_results/ only when the model was trained here)
  sn_masks/                  <subject_id>_sn_mask.nii.gz   <- FINAL OUTPUT
  visual_qc/                 qc_<n>_<subject_id>.png
  final_summary/             final_summary.csv, run_report.txt, config_used.json
  logs/                      pipeline_<date_time>.log, <stage>.done
```
Temporary nnU-Net files (`_work/`) are removed at the end (`keep_work_files` in the config).

---

## 6. What changed compared with the saved notebooks

Kept unchanged: every processing rule, threshold, seed, file format and the frozen model specification
(see `config/pipeline_config.json`). Changed only to chain the notebooks into one pipeline:

- Kaggle input searching, notebook-to-notebook hash checks and "resume from an attached earlier output" were replaced
  by fixed folders inside one output folder and one settings file; `config_used.json` records its sha256.
- The SPIR world space is the sform from the start for every subject (the notebooks first used the qform and then
  re-registered the four affected controls with the sform; the final state is the same).
- The crop size is the frozen 112 x 80 (the size computed from the data is still reported; `size_mode: "fit"`
  restores the computed size, which then no longer matches the trained model).
- The atlas retry uses the automatic label-free rule with the `head_init` start-up only (the `initializer` start-up
  was dropped in the notebooks because it failed at every start point).
- Only the chosen bridging gap (2.0 mm) is computed; the GMM QC pages show image | mask | GMM | refined.
  Names: `M4_gmm` -> `gmm`, `M4_final` -> `gmm_refined`.
- Hard count assertions (76 subjects, 56 development, 19 test) became report checks, a missing scale slope gives
  "unknown" instead of stopping, and a crop that cuts the SN region excludes that subject from inference
  (the notebook stopped; `stop_if_crop_cuts: true` restores that).
- Inference runs on every subject included for inference (`inference_subjects: "all"`; `"test"` = test subjects only)
  and passes `-device cpu` to nnU-Net when there is no GPU. Development subjects that were in the training set are
  segmented by models that saw them in training.

Left out on request (evaluation / analysis only, no effect on the masks): the checks of the `norm_` files, the
qform-vs-sform investigation, the label-analysis notebook (labelling rule, looseness, containment, population prior,
10-subject subset), the baseline configuration figures and the classical baselines (atlas, threshold, region growing,
random walker, graph cut) with mask cleaning, the out-of-fold evaluation against the manual masks, the second training
round with self-correction (not chosen), the left/right brightness diagnostic and the patient-vs-control comparison.

Names that look like numbering but are kept on purpose: `Dataset721_chealdev49r1` (the folder name of the trained
model; nnU-Net finds the model by it), `CIT168` (the atlas name) and scanner header field names such as
`PhaseEncodingSteps` in the audit tables.

---

## 7. Known limitations (from the development logs)

- There is no voxel-accurate reference: "correct" means the label-free checks plus visual review.
- The training labels come from single-rater, region-grown masks via GMM + bridging; the model inherits their
  left/right pattern. Left/right volume follows image brightness (no N4 bias correction).
- Segmentation and NM measures use the same image; never rank or conclude on contrast alone.
- Single scanner and protocol (Philips Ingenia CX 3T); other scanners or protocols are untested.
