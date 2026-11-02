"""
Shared helpers of the dataset audit (geometry audit, mask audit, slice-overlap check, duplicate scan check).

Reads data only (never writes into the data folder). All outputs go to the audit output folder.
Expected values and recorded decisions are taken from config/pipeline_config.json by apply_settings().
"""
import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np

# ----------------------------------------------------------------------------
# SETTINGS (filled from the pipeline configuration by apply_settings)
# ----------------------------------------------------------------------------
# Names of the three files inside each subject's folder (compared case-insensitively).
SPIR_NAME = "t1w_3d_spir"
T1W_NAME = "t1w"
MASK_NAME = "sn_mask"
NII_EXTS = (".nii", ".nii.gz")

# Expected values. They are only used to raise flags; nothing is changed or corrected automatically.
EXPECTED = {
    "spir": {"shape": (320, 320, 50), "zooms": (0.5625, 0.5625, 1.0)},
    "t1w": {"shape": (192, 256, 256), "zooms": (1.0, 1.0, 1.0)},
}
EXPECTED_COUNTS = {"control": 20, "chronic": 19, "acute": 17}
EXPECTED_TEST = 19

# Recorded decisions:
# 1. masks excluded from development (the subject's images are still audited).
EXCLUDED_MASKS = {}
# 2. masks whose header position differs from their SPIR but which sit on the correct SPIR region by voxel
#    index (checked visually). The SPIR header is used for them.
MASK_HEADER_OVERRIDE = set()


def apply_settings(cfg):
    """Copies the dataset settings of the pipeline configuration into this module."""
    global SPIR_NAME, T1W_NAME, MASK_NAME, NII_EXTS, EXPECTED, EXPECTED_COUNTS, EXPECTED_TEST
    global EXCLUDED_MASKS, MASK_HEADER_OVERRIDE
    ds = cfg["dataset"]
    SPIR_NAME = ds["file_stems"]["spir"].lower()
    T1W_NAME = ds["file_stems"]["t1w"].lower()
    MASK_NAME = ds["file_stems"]["mask"].lower()
    NII_EXTS = tuple(ds["accepted_ext"])
    EXPECTED = {k: {"shape": tuple(ds["expected_shape"][k]), "zooms": tuple(ds["expected_zooms"][k])} for k in ("spir", "t1w")}
    EXPECTED_COUNTS = dict(ds["expected_dev_counts"])
    EXPECTED_TEST = int(sum(ds["expected_test_counts"].values()))
    EXCLUDED_MASKS = dict(ds["excluded_masks"])
    MASK_HEADER_OVERRIDE = set(ds["mask_header_override"])

# JSON fields that are copied into the audit tables. Fields that identify the
# site, scanner or patient (address, serial number, station name, procedure
# description, clock times) are deliberately NOT in this list.
JSON_FIELDS = [
    "Modality", "MagneticFieldStrength", "Manufacturer", "ManufacturersModelName",
    "SoftwareVersions", "MRAcquisitionType", "SeriesDescription", "ProtocolName",
    "ScanningSequence", "SequenceVariant", "ScanOptions", "PulseSequenceName",
    "ImageType", "SliceThickness", "SpacingBetweenSlices", "NumberOfAverages",
    "EchoTime", "RepetitionTime", "MTState", "FlipAngle", "CoilString",
    "PercentSampling", "PartialFourierEnabled", "PartialFourierDirection",
    "PhaseEncodingStepsNoPartialFourier", "FrequencyEncodingSteps",
    "PhaseEncodingSteps", "PhaseEncodingStepsOutOfPlane", "EchoTrainLength",
    "AcquisitionMatrixPE", "ReconMatrixPE", "ParallelReductionFactorInPlane",
    "ParallelReductionFactorOutOfPlane", "ParallelAcquisitionTechnique",
    "PhilipsRescaleSlope", "PhilipsRescaleIntercept", "PhilipsScaleSlope",
    "UsePhilipsFloatNotDisplayScaling", "AcquisitionDuration", "PixelBandwidth",
    "ImageOrientationPatientDICOM", "InPlanePhaseEncodingDirectionDICOM",
    "ConversionSoftware", "ConversionSoftwareVersion",
]


# ----------------------------------------------------------------------------
# arguments / logging
# ----------------------------------------------------------------------------
def make_args(data_root, out_dir, limit=0):
    """data root, output folder (created) and an optional subject limit for a quick test"""
    from types import SimpleNamespace
    args = SimpleNamespace(data_root=Path(data_root), out_dir=Path(out_dir), limit=int(limit or 0))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return args


def setup_logging(out_dir, name):
    log_path = Path(out_dir) / f"{name}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return logging.getLogger(name)


# ----------------------------------------------------------------------------
# Subject discovery
# ----------------------------------------------------------------------------
def _find_file(folder, stem_lower):
    """Case-insensitive search for <stem>.nii or <stem>.nii.gz in folder."""
    hits = []
    for f in folder.iterdir():
        if not f.is_file():
            continue
        n = f.name.lower()
        for ext in NII_EXTS:
            if n == stem_lower + ext:
                hits.append(f)
    return sorted(hits)


def _find_json(nii_path):
    if nii_path is None:
        return None
    name = nii_path.name
    for ext in (".nii.gz", ".nii"):
        if name.lower().endswith(ext):
            base = name[: -len(ext)]
            cand = nii_path.with_name(base + ".json")
            return cand if cand.exists() else None
    return None


def infer_group(path, data_root=None):
    """Group from the folder path RELATIVE to the data root (so words in the parent
    path cannot confuse it). Test patients live under a 'testing' folder."""
    path = Path(path)
    try:
        rel = path.relative_to(data_root) if data_root is not None else path
    except ValueError:
        rel = path
    parts = [p.lower() for p in rel.parts]
    if "testing" in parts:
        if any(p in ("hc", "healthy", "healthy_controls", "control", "controls") for p in parts):
            return "test_HC"
        if any(p in ("scz", "schiz", "schizophrenia") for p in parts):
            return "test_SCZ"
        return "test_unknown"
    joined = "/".join(parts)
    if "healthy" in joined or "control" in joined:
        return "control"
    if "chronic" in joined:
        return "chronic"
    if "acute" in joined:
        return "acute"
    if "schiz" in joined or "scz" in joined:
        return "schiz_unspecified"
    return "unknown"


def _read_json_file(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def classify_json(js):
    """Return 'spir', 't1w' or None from the CONTENT of a json header (names are not reliable)."""
    import re
    text = (str(js.get("ProtocolName", "")) + " " + str(js.get("SeriesDescription", ""))).lower()
    pulse = str(js.get("PulseSequenceName", "")).upper()
    opts = [o.strip().upper() for o in re.split(r"[\\/]", str(js.get("ScanOptions", "")))]
    fa = js.get("FlipAngle")
    tr = js.get("RepetitionTime")
    if "spir" in text:
        return "spir"
    if pulse == "T1FFE" and "FS" in opts:
        return "spir"
    try:
        if fa is not None and abs(float(fa) - 20) < 0.5 and tr is not None and abs(float(tr) - 0.0254) < 0.002:
            return "spir"
    except Exception:
        pass
    if pulse == "T1TFE":
        return "t1w"
    try:
        if fa is not None and abs(float(fa) - 9) < 0.5:
            return "t1w"
    except Exception:
        pass
    if "t1w" in text.replace(" ", ""):
        return "t1w"
    return None


def find_jsons(folder, spir_path, t1w_path, extra_jsons=()):
    """
    JSON file names are not uniform, so identify them by CONTENT (see classify_json).
    Returns (spir_json_path, t1w_json_path, notes).
    """
    notes = []
    folder = Path(folder)
    jsons = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".json"] + list(extra_jsons))
    cand = {"spir": [], "t1w": []}
    unreadable, unclassified = [], []
    for p in jsons:
        js = _read_json_file(p)
        if js is None:
            unreadable.append(p.name)
            continue
        k = classify_json(js)
        if k is None:
            unclassified.append(p.name)
        else:
            cand[k].append(p)
    spir_j = cand["spir"][0] if cand["spir"] else None
    t1_j = cand["t1w"][0] if cand["t1w"] else None
    for k, lst in cand.items():
        if len(lst) > 1:
            notes.append(f"{len(lst)} {k}-like json files in folder, using {lst[0].name}")
    if spir_j is None and spir_path is not None:
        notes.append("no SPIR json identified in folder")
    if t1_j is None and t1w_path is not None:
        notes.append("no T1w json identified in folder")
    if unreadable:
        notes.append(f"unreadable json files: {unreadable}")
    if unclassified:
        notes.append(f"json files that match neither SPIR nor T1w: {unclassified}")
    return spir_j, t1_j, notes


def _is_nii(p):
    n = p.name.lower()
    return p.is_file() and (n.endswith(".nii") or n.endswith(".nii.gz"))


def _extract_zips(folder, key, unzip_dir, notes):
    """Extract .nii/.nii.gz/.json members of any .zip in folder into unzip_dir/<key>/ (never into the data folder)."""
    import zipfile
    out = []
    if unzip_dir is None:
        return out
    for z in sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".zip"):
        dest = Path(unzip_dir) / key
        try:
            with zipfile.ZipFile(z) as zf:
                for member in zf.namelist():
                    base = Path(member).name
                    if not base or not base.lower().endswith((".nii", ".nii.gz", ".json")):
                        continue
                    target = dest / base
                    if not target.exists():
                        dest.mkdir(parents=True, exist_ok=True)
                        with zf.open(member) as src, open(target, "wb") as dst:
                            dst.write(src.read())
                    out.append(target)
            notes.append(f"extracted from {z.name} into {dest}")
        except Exception as e:  # noqa
            notes.append(f"could not read zip {z.name}: {e}")
    return out


def _shape_of(p):
    try:
        return tuple(int(x) for x in nib.load(str(p)).shape[:3])
    except Exception:
        return None


def discover_subjects(data_root, limit=0, unzip_dir=None):
    """
    A subject folder is any folder called 'Neuromelanin' or containing T1W_3D_SPIR.nii(.gz).
    Files are found by exact name first (T1W_3D_SPIR, T1W, sn_mask; .nii or .nii.gz). If that fails,
    by image SHAPE (320x320x50 = SPIR, 192x256x256 = T1w) and, for the mask, by 'mask' in the name.
    Zip archives are extracted into unzip_dir (never into the data folder).
    subject_id is <group>_<folder name> because the same folder name can occur in different groups
    for different people.
    """
    data_root = Path(data_root)
    notes = []
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    folders = set()
    for p in data_root.rglob("*"):
        if p.is_dir() and p.name.lower() == "neuromelanin":
            folders.add(p)
        elif _is_nii(p) and any(p.name.lower() == SPIR_NAME + e for e in NII_EXTS):
            folders.add(p.parent)
    folders = sorted(folders)

    subjects = []
    seen_ids = {}
    for folder in folders:
        raw_id = folder.parent.name if folder.name.lower() == "neuromelanin" else folder.name
        group = infer_group(folder, data_root)
        sid = f"{group}_{raw_id}"
        if sid in seen_ids:
            notes.append(f"Subject key '{sid}' occurs twice ({folder} and {seen_ids[sid]})")
            sid = f"{sid}__dup{len(seen_ids)}"
        seen_ids[sid] = folder
        fnotes = []

        extra = _extract_zips(folder, sid, unzip_dir, fnotes)
        extra_nii = [p for p in extra if _is_nii(p)]
        extra_json = [p for p in extra if p.suffix.lower() == ".json"]
        all_nii = sorted([p for p in folder.iterdir() if _is_nii(p)]) + extra_nii

        def exact(stem):
            return [p for p in all_nii if any(p.name.lower() == stem + e for e in NII_EXTS)]
        spir, t1, mask = exact(SPIR_NAME), exact(T1W_NAME), exact(MASK_NAME)
        used = set(spir + t1 + mask)
        rest = [p for p in all_nii if p not in used]
        if not spir or not t1:
            for p in rest:
                sh = _shape_of(p)
                if sh is None:
                    continue
                if not spir and sorted(sh) == sorted(EXPECTED["spir"]["shape"]):
                    spir.append(p); used.add(p)
                    fnotes.append(f"SPIR identified by shape {sh}: {p.name}")
                elif not t1 and sorted(sh) == sorted(EXPECTED["t1w"]["shape"]):
                    t1.append(p); used.add(p)
                    fnotes.append(f"T1w identified by shape {sh}: {p.name}")
        if not mask:
            cands = [p for p in rest if p not in used and "mask" in p.name.lower()]
            if cands:
                mask.append(cands[0])
                fnotes.append(f"mask identified by name: {cands[0].name}")
        left = [p.name for p in all_nii if p not in set(spir[:1] + t1[:1] + mask[:1])]
        if left:
            fnotes.append(f"other NIfTI files in folder, not used: {left}")
        for label, lst in (("SPIR", spir), ("T1w", t1), ("mask", mask)):
            if len(lst) > 1:
                fnotes.append(f"{len(lst)} {label} files found, using {lst[0].name}")

        spir_json, t1_json, jnotes = find_jsons(folder, spir[0] if spir else None, t1[0] if t1 else None, extra_json)
        excluded = EXCLUDED_MASKS.get(sid)
        has_mask = bool(mask)
        subjects.append({
            "subject_id": sid,
            "folder_name": raw_id,
            "group": group,
            "folder": str(folder),
            "spir": spir[0] if spir else None,
            "t1w": t1[0] if t1 else None,
            "mask": mask[0] if mask else None,
            "spir_json": spir_json,
            "t1w_json": t1_json,
            "json_notes": "; ".join(jnotes),
            "file_notes": "; ".join(fnotes),
            "mask_excluded_reason": excluded or "",
            "labelled": bool(has_mask and not excluded),
            "is_test": group.startswith("test_"),
        })
    subjects.sort(key=lambda s: (s["group"], s["subject_id"]))
    if limit:
        subjects = subjects[:limit]
    return subjects, notes


def manifest_rows(subjects):
    rows = []
    for s in subjects:
        rows.append({
            "subject_id": s["subject_id"], "folder_name": s["folder_name"],
            "group": s["group"], "labelled": s["labelled"],
            "has_spir": s["spir"] is not None, "has_t1w": s["t1w"] is not None,
            "has_mask": s["mask"] is not None,
            "has_spir_json": s["spir_json"] is not None,
            "has_t1w_json": s["t1w_json"] is not None,
            "spir_json_name": s["spir_json"].name if s["spir_json"] else "",
            "t1w_json_name": s["t1w_json"].name if s["t1w_json"] else "",
            "spir_file": s["spir"].name if s["spir"] else "",
            "t1w_file": s["t1w"].name if s["t1w"] else "",
            "mask_file": s["mask"].name if s["mask"] else "",
            "mask_excluded_reason": s["mask_excluded_reason"],
            "is_test": s["is_test"],
            "file_notes": s["file_notes"],
            "json_notes": s["json_notes"],
            "folder": s["folder"],
        })
    return rows


# ----------------------------------------------------------------------------
# Loading / small utilities
# ----------------------------------------------------------------------------
def load_nifti(path):
    """Returns (nibabel image, numpy array). 4D arrays with a singleton last dim are squeezed."""
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)
    squeezed = False
    if arr.ndim == 4 and arr.shape[3] == 1:
        arr = arr[..., 0]
        squeezed = True
    return img, arr, squeezed


def load_json_path(json_path):
    """Read a json file by its own path (None if missing or unreadable)."""
    if json_path is None:
        return None
    return _read_json_file(json_path)


def json_subset(js):
    """Only whitelisted, non-identifying fields. Lists become strings."""
    out = {}
    if js is None:
        return out
    for k in JSON_FIELDS:
        if k in js:
            v = js[k]
            if isinstance(v, (list, tuple)):
                v = str(list(v))
            out[k] = v
    return out


def parse_clock_seconds(s):
    """'14:09:44.098000' or '13:51:8.081000' -> seconds since midnight (float)."""
    try:
        h, m, sec = str(s).split(":")
        return int(h) * 3600 + int(m) * 60 + float(sec)
    except Exception:
        return None


def acq_gap_minutes(js_spir, js_t1):
    if not js_spir or not js_t1:
        return None
    a = parse_clock_seconds(js_spir.get("AcquisitionTime"))
    b = parse_clock_seconds(js_t1.get("AcquisitionTime"))
    if a is None or b is None:
        return None
    return round((a - b) / 60.0, 2)


def tilt_from_iop(iop):
    """Angle (deg) between the slice normal and the nearest scanner axis, from
    ImageOrientationPatientDICOM [rx, ry, rz, cx, cy, cz]."""
    try:
        v = np.array(iop, dtype=float)
        r, c = v[:3], v[3:6]
        n = np.cross(r, c)
        n = n / np.linalg.norm(n)
        best = np.max(np.abs(n))
        return float(np.degrees(np.arccos(min(1.0, best))))
    except Exception:
        return None


def slab_axis(shape, zooms):
    """Index of the through-slab (slice) axis: the axis with the largest voxel size;
    if all voxel sizes are equal, the axis with the fewest voxels."""
    z = np.asarray(zooms[:3], dtype=float)
    if np.ptp(z) > 1e-3:
        return int(np.argmax(z))
    return int(np.argmin(shape[:3]))


def otsu_threshold(values, nbins=256):
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    lo, hi = np.percentile(v, [0.5, 99.5])
    if hi <= lo:
        return float(lo)
    hist, edges = np.histogram(np.clip(v, lo, hi), bins=nbins, range=(lo, hi))
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    s0 = np.cumsum(hist * centers)
    m0 = s0 / np.maximum(w0, 1e-12)
    m1 = (s0[-1] - s0) / np.maximum(w1, 1e-12)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[int(np.argmax(between))])


def run_safely(log, subject_id, fn, *a, **kw):
    """Run fn; on any error log the traceback and return None so the loop continues."""
    try:
        return fn(*a, **kw)
    except Exception as e:  # noqa
        log.error("Subject %s failed: %s", subject_id, e)
        log.debug(traceback.format_exc())
        log.error(traceback.format_exc())
        return None
