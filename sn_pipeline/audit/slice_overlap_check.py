"""
slice_overlap_check.py - dataset audit: neighbour correlation along the slab axis.

Empirical look at how correlated neighbouring samples are along the slab (slice)
axis compared with the two in-plane axes, on the T1W_3D_SPIR volume. This is
descriptive. It cannot prove whether slices overlap or were interpolated; it
measures what the images look like.

Measurements per subject (SPIR only):
  1. noise_corr_lagK_<axis>   : Pearson correlation of background (air) voxels
                                separated by K voxels along each axis (K = 1..4)
  2. signal_corr_lagK_<axis>  : same, inside the head (anatomy dominates)
  3. highfreq_power_frac_<axis>: fraction of non-DC power in the upper half of the
                                spatial-frequency band along each axis
  4. grad_p90_per_voxel/_per_mm_<axis>: 90th percentile of |first difference|
                                inside the head, relative to the 99th intensity percentile
Axis names: slab = through-slab axis, ip_a / ip_b = the two in-plane axes.

Outputs: slice_overlap_per_subject.csv, slice_overlap_summary.txt,
         slice_overlap_noise_corr.png, slice_overlap_signal_corr.png, slice_overlap.log
"""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from . import audit_common as C

LAGS = (1, 2, 3, 4)


def lagged_corr(vol, valid, axis, lag):
    """Pearson correlation between vol and vol shifted by `lag` along `axis`,
    using only pairs where both voxels are valid."""
    n = vol.shape[axis]
    if n <= lag:
        return np.nan
    sl_a = [slice(None)] * 3
    sl_b = [slice(None)] * 3
    sl_a[axis] = slice(0, n - lag)
    sl_b[axis] = slice(lag, n)
    a, b = vol[tuple(sl_a)], vol[tuple(sl_b)]
    m = valid[tuple(sl_a)] & valid[tuple(sl_b)]
    if m.sum() < 1000:
        return np.nan
    a, b = a[m].astype(np.float64), b[m].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    if den == 0:
        return np.nan
    return float((a * b).sum() / den)


def highfreq_fraction(vol, axis):
    v = vol.astype(np.float64)
    v = v - v.mean()
    n = v.shape[axis]
    shape = [1, 1, 1]
    shape[axis] = n
    win = np.hanning(n).reshape(shape)
    ps = np.abs(np.fft.rfft(v * win, axis=axis)) ** 2
    other = tuple(a for a in range(3) if a != axis)
    prof = ps.mean(axis=other)  # mean power per frequency
    prof = prof[1:]  # drop DC
    if prof.sum() == 0:
        return np.nan
    half = len(prof) // 2
    return float(prof[half:].sum() / prof.sum())


def analyse(s, flags):
    sid = s["subject_id"]
    img, arr, _ = C.load_nifti(s["spir"])
    if arr.ndim != 3:
        flags.append((sid, "ERROR", f"SPIR is {arr.ndim}D"))
        return None
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    vol = arr.astype(np.float32)
    valid_all = np.isfinite(vol)
    vol = np.where(valid_all, vol, 0)
    ax_slab = C.slab_axis(vol.shape, zooms)
    ip = [a for a in range(3) if a != ax_slab]
    names = {ax_slab: "slab", ip[0]: "ip_a", ip[1]: "ip_b"}

    pos = vol[vol > 0]
    if pos.size < 1000:
        flags.append((sid, "ERROR", "SPIR has almost no positive voxels"))
        return None
    thr = C.otsu_threshold(pos)
    fg = vol > thr
    # background: away from the head by 6 voxels in all directions
    near_head = ndi.binary_dilation(fg, iterations=6)
    bg = ~near_head
    bg_vals = vol[bg]
    row = {"subject_id": sid, "group": s["group"], "slab_axis_index": ax_slab,
           "shape": str(vol.shape), "zooms_mm": str(tuple(round(z, 4) for z in zooms)),
           "otsu_threshold": round(float(thr), 4),
           "foreground_voxels": int(fg.sum()), "background_voxels": int(bg.sum()),
           "background_mean": round(float(bg_vals.mean()), 5) if bg_vals.size else np.nan,
           "background_sd": round(float(bg_vals.std()), 5) if bg_vals.size else np.nan}
    bg_ok = bg_vals.size > 5000 and float(bg_vals.std()) > 0
    if not bg_ok:
        flags.append((sid, "WARN", "background region is too small or has zero variance (e.g. noise-free background); noise correlation not computed"))

    # head interior (eroded) for signal correlation
    head = ndi.binary_erosion(fg, iterations=2)

    for ax in range(3):
        nm = names[ax]
        for lag in LAGS:
            row[f"noise_corr_lag{lag}_{nm}"] = round(lagged_corr(vol, bg, ax, lag), 4) if bg_ok else np.nan
            row[f"signal_corr_lag{lag}_{nm}"] = round(lagged_corr(vol, head, ax, lag), 4)
        row[f"highfreq_power_frac_{nm}"] = round(highfreq_fraction(vol, ax), 5)
        d = np.abs(np.diff(vol, axis=ax))
        # valid difference pairs: both voxels inside the head
        sa = [slice(None)] * 3
        sb = [slice(None)] * 3
        sa[ax] = slice(0, vol.shape[ax] - 1)
        sb[ax] = slice(1, vol.shape[ax])
        pm = head[tuple(sa)] & head[tuple(sb)]
        p99 = float(np.percentile(vol[head], 99)) if head.any() else np.nan
        if pm.sum() > 1000 and p99 > 0:
            g = float(np.percentile(d[pm], 90) / p99)
            row[f"grad_p90_per_voxel_{nm}"] = round(g, 5)
            row[f"grad_p90_per_mm_{nm}"] = round(g / zooms[ax], 5)
        else:
            row[f"grad_p90_per_voxel_{nm}"] = np.nan
            row[f"grad_p90_per_mm_{nm}"] = np.nan

    if bg_ok:
        l1 = [row.get(f"noise_corr_lag1_{n}") for n in ("ip_a", "ip_b", "slab")]
        l4 = [row.get(f"noise_corr_lag4_{n}") for n in ("ip_a", "ip_b", "slab")]
        if np.isfinite(l1 + l4).all() and min(l4) > 0.5:
            row["background_not_noise_like"] = True
    # ratios of slab to in-plane (convenience)
    for kind in ("noise_corr", "signal_corr"):
        for lag in (1, 2):
            a = row.get(f"{kind}_lag{lag}_slab")
            b1, b2 = row.get(f"{kind}_lag{lag}_ip_a"), row.get(f"{kind}_lag{lag}_ip_b")
            if a is not None and b1 is not None and b2 is not None and np.isfinite([a, b1, b2]).all():
                row[f"{kind}_lag{lag}_slab_minus_mean_inplane"] = round(a - (b1 + b2) / 2, 4)
    return row


def run(data_root, out_dir, limit=0):
    args = C.make_args(data_root, out_dir, limit)
    log = C.setup_logging(args.out_dir, "slice_overlap")
    subjects, notes = C.discover_subjects(args.data_root, args.limit, unzip_dir=args.out_dir / "_unzipped")
    subjects = [s for s in subjects if s["spir"] is not None]
    log.info("%d subjects with a SPIR file", len(subjects))
    flags = [("(discovery)", "WARN", n) for n in notes]

    rows = []
    for i, s in enumerate(subjects, 1):
        log.info("[%d/%d] %s (%s)", i, len(subjects), s["subject_id"], s["group"])
        r = C.run_safely(log, s["subject_id"], analyse, s, flags)
        if r is None:
            flags.append((s["subject_id"], "ERROR", "slice-overlap analysis failed for this subject (see log)"))
            continue
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "slice_overlap_per_subject.csv", index=False)
    pd.DataFrame(flags, columns=["subject_id", "severity", "message"]).to_csv(
        args.out_dir / "slice_overlap_flags.csv", index=False)

    axes = ["ip_a", "ip_b", "slab"]
    # figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for kind, fname, title in (("noise_corr", "slice_overlap_noise_corr.png", "Background (noise) correlation vs lag"),
                                   ("signal_corr", "slice_overlap_signal_corr.png", "Signal correlation inside head vs lag")):
            if df.empty:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            for nm, mk in zip(axes, ("o-", "s-", "^-")):
                m = [df[f"{kind}_lag{l}_{nm}"].astype(float).mean() for l in LAGS]
                sd = [df[f"{kind}_lag{l}_{nm}"].astype(float).std() for l in LAGS]
                ax.errorbar(LAGS, m, yerr=sd, fmt=mk, capsize=3, label=nm)
            ax.set_xlabel("lag (voxels)")
            ax.set_ylabel("Pearson correlation")
            ax.set_xticks(LAGS)
            ax.set_title(title + f" (mean +/- sd, n={len(df)})")
            ax.legend()
            fig.tight_layout()
            fig.savefig(args.out_dir / fname, dpi=150)
            plt.close(fig)
    except Exception as e:  # noqa
        log.warning("Figure skipped: %s", e)

    # summary text
    L = ["SLICE-OVERLAP EMPIRICAL CHECK SUMMARY", "=" * 60,
         f"Subjects analysed: {len(df)} of {len(subjects)}", ""]

    def mstat(col):
        if col not in df:
            return "n/a"
        x = df[col].astype(float).dropna()
        return f"{x.mean():.4f} (sd {x.std():.4f}, n={len(x)})" if len(x) else "n/a"
    for kind, label in (("noise_corr", "Background (air) noise correlation"),
                        ("signal_corr", "Signal correlation inside head")):
        L.append(label + ", mean across subjects:")
        for lag in LAGS:
            L.append(f"  lag {lag}: " + " | ".join(f"{nm}={mstat(f'{kind}_lag{lag}_{nm}')}" for nm in axes))
        L.append("")
    L.append("High-frequency power fraction (upper half band, cycles per voxel):")
    L.append("  " + " | ".join(f"{nm}={mstat(f'highfreq_power_frac_{nm}')}" for nm in axes))
    L.append("Gradient (90th pct |difference| / 99th pct intensity), per voxel and per mm:")
    L.append("  per voxel: " + " | ".join(f"{nm}={mstat(f'grad_p90_per_voxel_{nm}')}" for nm in axes))
    L.append("  per mm:    " + " | ".join(f"{nm}={mstat(f'grad_p90_per_mm_{nm}')}" for nm in axes))
    if "background_not_noise_like" in df:
        nb = int(df["background_not_noise_like"].fillna(False).sum())
        if nb > len(df) / 2:
            L += ["", f"NOTE: in {nb} of {len(df)} subjects the 'background' region is still strongly correlated at lag 4 (>0.5)",
                  "in every direction. That is not what pure noise looks like (smooth shading, ghosting, or earlier processing",
                  "can cause it), so the noise-correlation rows above do NOT separate slab from in-plane and should not be",
                  "used to argue about slice overlap."]
    L += ["", "HOW TO READ THIS (and what it cannot tell you)",
          "- ip_a and ip_b are the two in-plane axes; slab is the through-slab axis.",
          "- If slab-axis noise correlation at lag 1 (and 2) is clearly higher than the in-plane values, neighbouring",
          "  slices are not independent samples. That is consistent with overlap or with interpolation along slab;",
          "  these measurements cannot separate the two.",
          "- In-plane noise can also be correlated (SENSE, partial Fourier, zero-filling to a finer grid),",
          "  and magnitude-image noise is not Gaussian, so compare slab with in-plane; do not compare with zero.",
          "- Signal correlation is dominated by smooth anatomy and is only a rough guide.",
          "- Do not claim a mechanism in the paper from these numbers alone. Quote the scanner header",
          "  text as recorded, and describe these measurements as what they are."]
    if len(flags):
        L += ["", f"FLAGS ({len(flags)}), see slice_overlap_flags.csv:"] + [f"  [{f[1]}] {f[0]}: {f[2]}" for f in flags[:30]]
    text = "\n".join(L)
    (args.out_dir / "slice_overlap_summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)

