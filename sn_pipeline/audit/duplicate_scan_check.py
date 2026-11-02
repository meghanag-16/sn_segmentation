"""
duplicate_scan_check.py - cheap safeguard against the same scan appearing under two names.

Because some folder names repeat across groups (for example SCZ-27M in 'acute' and 'chronic'),
this script checks, using only the image data, whether two different subject entries are in
fact the same scan:
  1. identical files (SHA-256 of the T1W_3D_SPIR file)
  2. near-identical images (correlation of the fine-detail part of a 2x downsampled SPIR volume > 0.95;
     the smooth head shape is removed first, so two different heads do not match just because
     both are head-shaped. Re-scaled copies of one scan are still caught.)
Outputs: duplicate_scan_check.csv, duplicate_scan_check.txt, duplicate_scan_check.log
Nothing is changed. Different people with the same folder name will NOT be reported unless
their images really match.
"""
import hashlib
from itertools import combinations

import numpy as np
import pandas as pd

from . import audit_common as C


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def small_copy(path):
    from scipy import ndimage as ndi
    img, arr, _ = C.load_nifti(path)
    a = np.nan_to_num(arr.astype(np.float32))
    a = a[: (a.shape[0] // 2) * 2, : (a.shape[1] // 2) * 2]
    a = a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2, a.shape[2]).mean(axis=(1, 3))
    a = a - ndi.uniform_filter(a, size=5)   # keep fine detail only
    return a.ravel()


def run(data_root, out_dir, limit=0):
    args = C.make_args(data_root, out_dir, limit)
    log = C.setup_logging(args.out_dir, "duplicate_scan_check")
    subjects, _ = C.discover_subjects(args.data_root, args.limit, unzip_dir=args.out_dir / "_unzipped")
    subjects = [s for s in subjects if s["spir"] is not None]
    hashes, smalls = {}, {}
    for i, s in enumerate(subjects, 1):
        log.info("[%d/%d] %s", i, len(subjects), s["subject_id"])
        h = C.run_safely(log, s["subject_id"], sha256, s["spir"])
        v = C.run_safely(log, s["subject_id"], small_copy, s["spir"])
        if h:
            hashes[s["subject_id"]] = h
        if v is not None:
            smalls[s["subject_id"]] = v
    rows = []
    for a, b in combinations(sorted(smalls), 2):
        va, vb = smalls[a], smalls[b]
        if va.shape != vb.shape:
            continue
        same_hash = hashes.get(a) is not None and hashes.get(a) == hashes.get(b)
        c = float(np.corrcoef(va, vb)[0, 1]) if va.std() > 0 and vb.std() > 0 else np.nan
        if same_hash or (np.isfinite(c) and c > 0.95):
            rows.append({"subject_a": a, "subject_b": b, "identical_file": same_hash, "downsampled_correlation": round(c, 5)})
    df = pd.DataFrame(rows, columns=["subject_a", "subject_b", "identical_file", "downsampled_correlation"])
    df.to_csv(args.out_dir / "duplicate_scan_check.csv", index=False)
    text = ["DUPLICATE SCAN CHECK", "=" * 60, f"SPIR volumes compared: {len(smalls)}",
            f"Pairs flagged (identical file or fine-detail correlation > 0.95): {len(df)}"]
    for _, r in df.iterrows():
        text.append(f"  {r.subject_a}  <->  {r.subject_b}  identical_file={r.identical_file}  corr={r.downsampled_correlation}")
    if df.empty:
        text.append("No two entries look like the same scan.")
    out = "\n".join(text)
    (args.out_dir / "duplicate_scan_check.txt").write_text(out, encoding="utf-8")
    print("\n" + out)

