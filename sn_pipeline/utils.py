"""
Small helpers shared by several stages: NIfTI input/output, grid arithmetic, label-image rules
(atlas outer limit, SNc anchor, post-processing) and summary formatting.
"""
import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

S6 = ndimage.generate_binary_structure(3, 1)
S26 = np.ones((3, 3, 3), bool)


# ----------------------------------------------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------------------------------------------
class Report:
    """Collects INFO / WARN / ERROR lines for a text report."""

    def __init__(self):
        self.lines, self.errors, self.warnings = [], [], []

    def info(self, m):
        self.lines.append("[INFO]  " + m)

    def warn(self, m):
        self.warnings.append(m)
        self.lines.append("[WARN]  " + m)

    def error(self, m):
        self.errors.append(m)
        self.lines.append("[ERROR] " + m)

    def check(self, ok, what):
        (self.info if ok else self.error)(("PASS: " if ok else "FAIL: ") + what)


def log(msg):
    print(msg, flush=True)


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def mi(s, fmt="{:.2f}"):
    """median [IQR] as text"""
    s = pd.Series(s, dtype=float).dropna()
    return f"{fmt.format(s.median())} [{fmt.format(s.quantile(.25))}-{fmt.format(s.quantile(.75))}]" if len(s) else "n/a"


def jsonable(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, float)):
            out[k] = None if not np.isfinite(v) else float(v)
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.bool_,)):
            out[k] = bool(v)
        else:
            out[k] = v
    return out


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else np.nan


def write_text(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")


# ----------------------------------------------------------------------------------------------------------------
# NIfTI and grids
# ----------------------------------------------------------------------------------------------------------------
def vox_size(aff):
    return np.sqrt((np.asarray(aff)[:3, :3] ** 2).sum(0))


def load_nii(path, dtype=np.float32):
    im = nib.load(str(path))
    return np.asarray(im.dataobj).astype(dtype), im.affine


def save_nii(arr, aff, path, dtype=np.float32):
    """Saves with qform = sform = aff (code 1), creating the folder if needed."""
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    im = nib.Nifti1Image(np.asarray(arr).astype(dtype), aff)
    im.set_qform(aff, 1)
    im.set_sform(aff, 1)
    nib.save(im, str(path))


def crop_affine(aff, i0, j0):
    """affine of a crop whose voxel (0, 0, 0) is full-grid voxel (i0, j0, 0)"""
    T = np.eye(4)
    T[0, 3], T[1, 3] = i0, j0
    return aff @ T


def full_affine(caff, i0, j0):
    """inverse of crop_affine: crop voxel (0,0,0) = full-grid voxel (i0, j0, 0)"""
    f = np.array(caff, float).copy()
    f[:3, 3] = f[:3, 3] - f[:3, :3] @ np.array([i0, j0, 0], float)
    return f


def find_file(folder, pattern):
    rx = re.compile(pattern)
    hits = sorted(f for f in os.listdir(folder) if rx.match(f))
    if len(hits) != 1:
        raise FileNotFoundError(f"{pattern} in {folder}: {hits}")
    return Path(folder) / hits[0]


def read_atlas_labels(path):
    """labels file of the CIT168 atlas: '<volume index> <name> ...' per line"""
    labels = {}
    for line in open(path, encoding="utf-8", errors="ignore"):
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            labels[parts[1]] = int(parts[0])
    return labels


def nmi(a, b, bins=64):
    """normalised mutual information (H(a) + H(b)) / H(a, b)"""
    h, _, _ = np.histogram2d(a, b, bins=bins)
    p = h / h.sum()
    ent = lambda q: -np.sum(q[q > 0] * np.log(q[q > 0]))
    return (ent(p.sum(1)) + ent(p.sum(0))) / ent(p)


# ----------------------------------------------------------------------------------------------------------------
# label rules on the crop grid (atlas outer limit, SNc anchor, post-processing)
# ----------------------------------------------------------------------------------------------------------------
def outer_limit(atl, zs, rules, iSNc, iSNr, iRN):
    """CIT168 SNc+SNr p >= p_min, dilated by dilate_mm, minus red nucleus p >= exclude p_min"""
    SR = rules["search_region"]
    sn = (atl[..., iSNc] + atl[..., iSNr]) >= SR["p_min"]
    dmm = float(SR["dilate_mm"])
    reg = (ndimage.distance_transform_edt(~sn, sampling=zs) <= dmm) if dmm > 0 else sn.copy()
    return reg & ~(atl[..., iRN] >= SR["exclude"]["p_min"])


def snc_anchor(atl, hemi, rules, iSNc):
    """per side: CIT168 SNc p >= p_min (fallback p >= fallback_p_min); labels 1 left / 2 right"""
    AN = rules["snc_anchor"]
    a = np.zeros(hemi.shape, np.uint8)
    for side in (1, 2):
        s = (atl[..., iSNc] >= AN["p_min"]) & (hemi == side)
        if not s.any():
            s = (atl[..., iSNc] >= AN["fallback_p_min"]) & (hemi == side)
        a[s] = side
    return a


def fill_inplane(m):
    out = m.copy()
    for k in range(m.shape[2]):
        if m[..., k].any():
            out[..., k] = ndimage.binary_fill_holes(m[..., k])
    return out


def keep_anchor_pieces(s, anc_side):
    lab, n = ndimage.label(s, structure=S26)
    if n == 0:
        return s, False
    ids = np.unique(lab[anc_side & s])
    ids = ids[ids > 0]
    if len(ids):
        return np.isin(lab, ids), False
    return lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1), True     # fallback: largest piece


def postprocess_gmm(S, d):
    """GMM rule: per side, inside the outer limit, 26-connected pieces touching the SNc anchor (no fallback),
    holes filled in-plane. Labels 0 / 1 left / 2 right."""
    out = np.zeros(S.shape, np.uint8)
    for side in (1, 2):
        s = S & (d["hemi"] == side) & d["reg"]
        if not s.any():
            continue
        lab, n = ndimage.label(s, structure=S26)
        ids = np.unique(lab[(d["anc"] == side) & s])
        ids = ids[ids > 0]
        keep = fill_inplane(np.isin(lab, ids)) & d["reg"] & (d["hemi"] == side)
        out[keep] = side
    return out


def postprocess_network(S, d):
    """Network rule: per side, inside the atlas outer limit, 26-connected pieces touching the SNc anchor
    (fallback: largest piece), holes filled in-plane. Labels 0 / 1 left / 2 right."""
    out = np.zeros(S.shape, np.uint8)
    for side in (1, 2):
        s = S & (d["hemi"] == side) & d["reg"]
        if not s.any():
            continue
        lab, n = ndimage.label(s, structure=S26)
        ids = np.unique(lab[(d["anc"] == side) & s])
        ids = ids[ids > 0]
        keep = np.isin(lab, ids) if len(ids) else lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
        out[fill_inplane(keep) & d["reg"] & (d["hemi"] == side)] = side
    return out


def auto_gate(r, G):
    """pre-set gate: both sides, containment, core coverage, contrast, L/R ratio, largest piece"""
    why = []
    if not (r["vol_left_mm3"] > 0 and r["vol_right_mm3"] > 0): why.append("side empty")
    if not (r["containment"] >= G["containment_min"]): why.append("containment")
    if not (r["core_coverage"] >= G["core_coverage_min"]): why.append("core coverage")
    if not (r["contrast"] > G["contrast_min_exclusive"]): why.append("contrast")
    lo, hi = G["lr_ratio_range"]
    if not (lo <= r["lr_ratio"] <= hi): why.append("L/R ratio")
    if not (min(r["largest_share_left"], r["largest_share_right"]) >= G["largest_piece_min"]): why.append("fragmented")
    return len(why) == 0, "; ".join(why)


def read_subjects(paths):
    subj = pd.read_csv(paths.subjects_csv()).fillna({"mask_path": "", "t1w_path": ""})
    return subj
