"""
Visual QC of the final segmentations: one page per subject, every slice that contains the segmentation (+2 above and below),
columns image | nnU-Net probability | nnU-Net post-processed (left orange, right cyan).

Outputs: visual_qc/qc_<n>_<subject>.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

from .nnunet_common import CropData
from .utils import load_nii, postprocess_network, log


def run(ctx):
    P, ncfg = ctx.paths, ctx.cfg["nnunet"]
    thr = float(ncfg["probability_threshold"])
    dpi = int(ctx.cfg["output"]["qc_pages_dpi"])
    FIG = P.visual_qc
    FIG.mkdir(parents=True, exist_ok=True)
    CD = CropData(ctx)
    TM = pd.read_csv(P.nnunet / "segmentation_metrics.csv")
    SIDS = list(TM.subject_id)
    for pn, s in enumerate(SIDS, 1):
        d = CD.load_subject(s)
        p = load_nii(P.nnunet / "prob_crop" / f"{s}_sn_prob.nii.gz")[0]
        L = postprocess_network(p >= thr, d)
        m = d["reg"].any(axis=2); ii, jj = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
        bi = slice(max(ii.min() - 6, 0), min(ii.max() + 7, m.shape[0])); bj = slice(max(jj.min() - 6, 0), min(jj.max() + 7, m.shape[1]))
        v_ = d["raw"][d["reg"]]; vmin, vmax = float(np.percentile(v_, 1)), float(np.percentile(v_, 99.5))
        occ = np.flatnonzero(L.any(axis=(0, 1))) if L.any() else np.flatnonzero(d["reg"].any(axis=(0, 1)))
        ks = list(range(max(occ.min() - 2, 0), min(occ.max() + 3, d["raw"].shape[2])))
        w, h = bi.stop - bi.start, bj.stop - bj.start; pw = 2.6; ph = pw * h / w
        fig, axs = plt.subplots(len(ks), 3, figsize=(pw * 3, ph * len(ks) + 0.8), squeeze=False)
        for r_i, k in enumerate(ks):
            for c_i, title in enumerate(["image", "nnU-Net probability", "nnU-Net (post-processed)"]):
                ax = axs[r_i, c_i]; ax.set_xticks([]); ax.set_yticks([])
                ax.imshow(d["raw"][bi, bj, k].T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest")
                if c_i == 1:
                    ax.imshow(np.ma.masked_where(p[bi, bj, k].T < 0.05, p[bi, bj, k].T), cmap="hot", origin="lower", vmin=0, vmax=1,
                              alpha=0.6, interpolation="nearest")
                if c_i == 2:
                    for side, col in ((1, "orange"), (2, "cyan")):
                        s_ = L[bi, bj, k] == side
                        if s_.any():
                            ax.imshow(np.ma.masked_where(~s_.T, s_.T), cmap=ListedColormap([col]), origin="lower", alpha=0.45,
                                      vmin=0, vmax=1, interpolation="nearest")
                if r_i == 0:
                    ax.set_title(title, fontsize=8)
            axs[r_i, 0].set_ylabel(f"sl {k}", fontsize=7)
        row = TM.set_index("subject_id").loc[s]
        flag = "" if bool(row.label_free_gate_pass) else f" | FLAG: {row.flags}"
        fig.suptitle(f"{pn}/{len(SIDS)} {s} ({CD.ALL.loc[s, 'group']}, {CD.ALL.loc[s, 'scale_group']}) | {ncfg['dataset_name']} "
                     f"{ncfg['configuration']} | raw SPIR {vmin:.0f}-{vmax:.0f}{flag}", fontsize=9)
        plt.tight_layout(rect=(0, 0.02, 1, 0.98)); fig.savefig(FIG / f"qc_{pn:02d}_{s}.png", dpi=dpi)
        plt.close(fig)
    log(f"{len(SIDS)} QC pages in {FIG}")
