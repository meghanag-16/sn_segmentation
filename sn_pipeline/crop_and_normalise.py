"""
Crop, normalisation and mask preparation on the native SPIR grid (nothing is resampled).
All mapping from template space uses the per-subject index map written by the atlas registration (no ANTs needed).

| step          | what it does |
|---------------|--------------|
| crop          | one fixed in-plane window for everybody (112 x 80 voxels, all 50 slices), placed per subject at the registered atlas SN centre (label-free, so it works on test subjects) |
| normalisation | two outputs per subject from the raw SPIR only, chosen by purpose: (a) trimmed z-score inside the crop = network input; (b) ratio to a crus cerebri reference region (located through the atlas, never through the mask) = GMM labels, NM measures |
| N4 / denoise  | off (an optional N4 crop can be switched on for an ablation) |
| masks         | binary (value != 0, uint8), SPIR header for all; left/right split by the template midline (via the index map) |
| QC            | one table with flags, inclusion for training / inference and reasons; visual QC pages; status report; restore check |

Outputs (cropped/):
  crops/{raw,zscore,refratio,atlas,hemi,ref_roi,mask_bin,mask_lr}/<subject>_*.nii.gz   crop grid, world coordinates kept
  masks_full/<subject>_sn_mask_bin.nii.gz, <subject>_sn_mask_lr.nii.gz               full SPIR grid
  template_space/crus_cerebri_ref_cit168.nii.gz
  crop_windows.csv, crop_window.json, per_subject_crop.csv, normalisation_selection.json, mask_preparation_summary.txt
  qc_table.csv (inclusion for training / inference), crop_decisions.json, crop_report.txt, restore_check.csv
  figs/template_crus_cerebri.png, figs/crop_window.png, figs/qc_crop_page<n>.png
"""
import json
import math
import os
import re
import shutil
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.filters import threshold_multiotsu

from .subject_inventory import data_root_of
from .utils import (cohen_d, crop_affine, find_file, jsonable, read_atlas_labels, read_subjects, save_nii, vox_size,
                    write_text, log)


class CropStage:
    def __init__(self, ctx):
        self.ctx, self.cfg = ctx, ctx.cfg["crop_and_normalise"]
        P = ctx.paths
        self.OUT = P.cropped
        self.SUB = {k: P.cropped.joinpath(*v) for k, v in {
            "raw": ("crops", "raw"), "zscore": ("crops", "zscore"), "refratio": ("crops", "refratio"), "atlas": ("crops", "atlas"),
            "hemi": ("crops", "hemi"), "ref_roi": ("crops", "ref_roi"), "mask_bin": ("crops", "mask_bin"), "mask_lr": ("crops", "mask_lr"),
            "n4_raw": ("crops", "n4_raw"), "masks_full": ("masks_full",), "per_subject": ("per_subject",), "figs": ("figs",),
            "template": ("template_space",)}.items()}
        for k, d in self.SUB.items():
            if k != "n4_raw" or self.cfg["run_n4"]:
                d.mkdir(parents=True, exist_ok=True)
        self.root = data_root_of(ctx)
        self.subj = read_subjects(P)
        self.P_ATLAS = P.atlas
        self.excluded_masks = list(ctx.cfg["dataset"]["excluded_masks"])
        self.override = list(ctx.cfg["dataset"]["mask_header_override"])
        self.RNG = np.random.default_rng(ctx.seed)
        self._check_inputs()
        self._load_atlas()

    # ------------------------------------------------------------------------------------------------ inputs
    def _check_inputs(self):
        atlas_table = pd.read_csv(self.P_ATLAS / "atlas_registration.csv")
        assert "grade" in atlas_table, "the atlas registration has no grades yet"
        ungraded = atlas_table.subject_id[atlas_table.grade.astype(str).str.startswith("not graded")].tolist()
        assert not ungraded, f"atlas registration subjects not graded: {ungraded}"
        self.REG_FAIL = atlas_table.subject_id[atlas_table.grade.astype(str).str.lower().str.startswith("fail")].tolist()
        self.atlas_table = atlas_table
        dec_atlas = json.load(open(self.P_ATLAS / "atlas_registration_decisions.json"))
        self.CHANNELS = dec_atlas["outputs"]["channels"]
        assert "SNc" in self.CHANNELS and "SNr" in self.CHANNELS, self.CHANNELS
        # subjects without registration output (rigid graded fail) cannot be cropped
        done = set(atlas_table.subject_id)
        self.subj = self.subj[self.subj.subject_id.isin(done)].reset_index(drop=True)
        missing = [s for s in self.subj.subject_id for f in (f"{s}_cit168_in_spir.nii.gz", f"{s}_cit168_index_in_spir.nii.gz")
                   if not (self.P_ATLAS / "atlas_in_spir" / f).exists()]
        assert not missing, f"missing atlas registration files: {missing[:6]} ..."
        log(f"atlas registration: grades {atlas_table.grade.astype(str).str.split(':').str[0].value_counts().to_dict()} | failed: {self.REG_FAIL}")
        # intensity-scale group for ALL subjects, from the slope table
        st = pd.read_csv(self.ctx.paths.subject_table / "scale_slope_table.csv")
        st["philips_scale_slope"] = pd.to_numeric(st.philips_scale_slope, errors="coerce")
        no_slope = sorted(set(self.subj.subject_id) - set(st.subject_id[st.philips_scale_slope.notna()]))
        if no_slope:
            log(f"WARNING: no PhilipsScaleSlope for {no_slope}: their scale group is 'unknown' (covariate only)")
        self.SLOPE = dict(zip(st.subject_id, st.philips_scale_slope))
        thr = self.cfg["high_slope_min"]
        self.SCALE_GROUP = {s: ("unknown" if not np.isfinite(self.SLOPE.get(s, np.nan)) else ("high_slope" if self.SLOPE[s] > thr else "low_slope"))
                            for s in self.subj.subject_id}
        self.TEST_LIKE = sorted(s for s in self.subj.subject_id[self.subj.role == "dev"] if self.SCALE_GROUP[s] == "high_slope")
        _sg = self.subj.assign(scale_group=self.subj.subject_id.map(self.SCALE_GROUP))
        log("intensity-scale group by diagnosis group:\n" + pd.crosstab(_sg.group, _sg.scale_group).to_string())

    def _load_atlas(self):
        F = self.ctx.cfg["atlas_registration"]["files"]
        ATL = self.ctx.atlas_dir
        self.ATL = ATL
        self.F_TEMPLATE = find_file(ATL, F["template"])
        self.F_PROB = find_file(ATL, F["prob_bilateral"])
        self.F_LABELS = find_file(ATL, F["labels"])
        self.F_LEFT_PATTERN = F["prob_left"]
        self.labels = read_atlas_labels(self.F_LABELS)
        self.T_img, self.P_img = nib.load(str(self.F_TEMPLATE)), nib.load(str(self.F_PROB))
        assert sorted(self.labels.values()) == list(range(self.P_img.shape[3])), "label numbers are not the 0-based volume index"
        assert np.allclose(self.T_img.affine, self.P_img.affine, atol=1e-4), "template and atlas grids differ"

    # ------------------------------------------------------------------------------------------------ helpers
    def load_spir(self, r):
        im = nib.load(str(self.root / r.spir_path))
        return im.get_fdata(dtype=np.float32), im.get_sform()     # SPIR world space = sform

    def load_mask(self, r):
        return nib.load(str(self.root / r.mask_path)).get_fdata(dtype=np.float32) != 0   # binary only; values discarded

    def load_atlas(self, sid):
        a = np.asarray(nib.load(str(self.P_ATLAS / "atlas_in_spir" / f"{sid}_cit168_in_spir.nii.gz")).dataobj, dtype=np.float32)
        return {name: a[..., i] for i, name in enumerate(self.CHANNELS)}

    def load_index(self, sid):
        """index map: CIT168 voxel index + 1 for every SPIR voxel (0 = outside the template)."""
        a = np.asarray(nib.load(str(self.P_ATLAS / "atlas_in_spir" / f"{sid}_cit168_index_in_spir.nii.gz")).dataobj, dtype=np.float32)
        return a, (a > 0).all(axis=-1)

    @staticmethod
    def tmpl_lookup(idx, valid, vol):
        """nearest-neighbour value of a template-space volume at every SPIR voxel (0 outside the template)"""
        ijk = np.rint(idx - 1.0).astype(np.int64)
        for ax in range(3):
            np.clip(ijk[..., ax], 0, vol.shape[ax] - 1, out=ijk[..., ax])
        out = vol[ijk[..., 0], ijk[..., 1], ijk[..., 2]]
        return np.where(valid, out, 0)

    def tmpl_world_x(self, idx, valid):
        """RAS x (mm) of the template position of every SPIR voxel"""
        A = self.T_img.affine
        x = (idx - 1.0) @ A[0, :3] + A[0, 3]
        return np.where(valid, x, np.nan)

    # ------------------------------------------------------------------------------------------------ template space
    def template_space(self):
        """midline and crus cerebri reference region (built once, label-free).
        Crus cerebri (cerebral peduncle) = white matter lying in FRONT of (anterior/anterolateral to) the SN. Rule, in CIT168 space:
        distance from the atlas SN (SNc+SNr p>=0.25) inside ref_band_mm; the nearest SN voxel lies BEHIND it (direction anterior);
        template T1w intensity in the white-matter class (3-class Otsu near the midbrain); not part of any CIT168 structure (p>0.05);
        within the SN's superior-inferior range; largest piece per hemisphere. Masks are not used."""
        T_img, P_img, labels, SUB = self.T_img, self.P_img, self.labels, self.SUB
        band = self.cfg["ref_band_mm"]
        A = T_img.affine; ZT = vox_size(A)
        ch = lambda name: np.asarray(P_img.dataobj[..., labels[name]], dtype=np.float32)
        sn_t = (ch("SNc") + ch("SNr")) >= 0.25
        any_struct = np.zeros(T_img.shape[:3], bool)
        for name in labels:
            any_struct |= ch(name) > 0.05
        rn = ch("RN")
        ijk_rn = np.argwhere(rn > 0.01)
        X_MID = float(np.average(ijk_rn @ A[0, :3] + A[0, 3], weights=rn[rn > 0.01]))    # bilateral RN is centred on the midline
        log(f"template midline (RAS x of the bilateral RN centre): {X_MID:.2f} mm")
        # work in a box around the SN (+25 mm) to keep memory small
        pad = np.ceil(25 / ZT).astype(int)
        lo = np.maximum(np.argwhere(sn_t).min(0) - pad, 0); hi = np.minimum(np.argwhere(sn_t).max(0) + pad + 1, sn_t.shape)
        bx = tuple(slice(a, b) for a, b in zip(lo, hi))
        snb, exb = sn_t[bx], any_struct[bx]
        tmpl = np.asarray(T_img.dataobj, dtype=np.float32)[bx]
        dist, inds = ndimage.distance_transform_edt(~snb, sampling=ZT, return_indices=True)
        vox = np.indices(snb.shape)
        dw = np.tensordot(A[:3, :3], (vox - inds).astype(np.float32), axes=1)       # vector nearest-SN-voxel -> voxel, RAS mm
        dn = np.linalg.norm(dw, axis=0) + 1e-6
        anterior = (dw[1] / dn) > 0.3
        wm_thr = float(threshold_multiotsu(tmpl[tmpl > 0], classes=3)[-1])
        world = np.tensordot(A[:3, :3], (vox + lo[:, None, None, None]).astype(np.float32), axes=1) + A[:3, 3, None, None, None]
        zs = world[2][snb]
        cand = (dist >= band[0]) & (dist <= band[1]) & anterior & (tmpl > wm_thr) & ~exb \
               & (world[2] >= zs.min()) & (world[2] <= zs.max())
        CRUS_T = np.zeros(T_img.shape[:3], np.uint8)        # 1 = left, 2 = right crus cerebri
        vol_t = float(np.prod(ZT))
        crus_info = {}
        for lab, side in ((1, world[0] < X_MID), (2, world[0] >= X_MID)):
            cc, n = ndimage.label(cand & side)
            if n == 0:
                raise RuntimeError(f"no crus cerebri candidate on side {lab}")
            big = cc == (np.argmax(np.bincount(cc.ravel())[1:]) + 1)
            CRUS_T[bx][big] = lab
            crus_info["left" if lab == 1 else "right"] = round(float(big.sum() * vol_t), 1)
        log(f"white-matter threshold {wm_thr:.1f} | crus cerebri ROI volume in template (mm3): {crus_info}")
        assert min(crus_info.values()) >= 20, "crus cerebri ROI is very small"
        save_nii(CRUS_T, A, SUB["template"] / "crus_cerebri_ref_cit168.nii.gz", np.uint8)
        # hemisphere naming check in template space: the left-hemisphere atlas (prob_atlas) must sit at x < X_MID
        f_left = [f for f in os.listdir(self.ATL) if re.match(self.F_LEFT_PATTERN, f)]
        if f_left:
            lsn = np.asarray(nib.load(str(self.ATL / f_left[0])).dataobj[..., labels["SNc"]], dtype=np.float32)
            ijk_l = np.argwhere(lsn > 0.01)
            frac_left = float(lsn[lsn > 0.01][(ijk_l @ A[0, :3] + A[0, 3]) < X_MID].sum() / lsn[lsn > 0.01].sum())
            log(f"left-hemisphere SNc map lying at x < midline: {frac_left:.3f} (should be close to 1)")
            assert frac_left > 0.95, "hemisphere naming check failed"
            del lsn
        # QC figure: template slices at the SN level with SN (green), crus cerebri ROI (cyan / blue), other structures (yellow)
        ks = np.unique(np.argwhere(snb)[:, 2])
        vlo, vhi = np.percentile(tmpl[tmpl > 0], [1, 99.5])
        show = ks[np.linspace(0, len(ks) - 1, 5).round().astype(int)]
        fig, axs = plt.subplots(1, 5, figsize=(16, 3.6))
        for ax, k in zip(axs, show):
            ax.imshow(tmpl[..., k].T, cmap="gray", origin="lower", vmin=vlo, vmax=vhi)
            ax.contour(snb[..., k].T.astype(np.uint8), [0.5], colors="lime", linewidths=0.7)
            ax.contour((exb & ~snb)[..., k].T.astype(np.uint8), [0.5], colors="yellow", linewidths=0.4)
            cr = CRUS_T[bx][..., k]
            ax.contour((cr == 1).T.astype(np.uint8), [0.5], colors="cyan", linewidths=0.9)
            ax.contour((cr == 2).T.astype(np.uint8), [0.5], colors="deepskyblue", linewidths=0.9)
            ax.set_title(f"template slice {k + lo[2]}", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("CIT168 template: green = SN (p>=0.25), cyan/blue = crus cerebri reference ROI (left/right), yellow = other atlas structures. "
                     "The ROI should lie in the white-matter band in front of the SN.", fontsize=8)
        plt.tight_layout(); fig.savefig(SUB["figs"] / "template_crus_cerebri.png", dpi=110); plt.close(fig)
        self.X_MID, self.CRUS_T = X_MID, CRUS_T

    # ------------------------------------------------------------------------------------------------ crop window
    def crop_window(self):
        """One size for all; size frozen (or, with size_mode 'fit', sized from DEVELOPMENT subjects only: masks + atlas extent
        + margin); position per subject = registered atlas SN centre (label-free); all slices."""
        subj, SUB = self.subj, self.SUB
        rows = []
        for r in subj.itertuples():
            atl = self.load_atlas(r.subject_id)
            sn = np.clip(atl["SNc"] + atl["SNr"], 0, None)
            ci, cj, ck = ndimage.center_of_mass(sn)
            uni = np.zeros(sn.shape, bool)
            for c in ("SNc", "SNr", "RN", "PBP", "VTA"):
                if c in atl:
                    uni |= atl[c] > 0.05
            ii, jj, kk = np.nonzero(uni)
            row = dict(subject_id=r.subject_id, group=r.group, role=r.role, ci=ci, cj=cj, ck=ck, nx=sn.shape[0], ny=sn.shape[1], nz=sn.shape[2],
                       atl_i_min=ii.min(), atl_i_max=ii.max(), atl_j_min=jj.min(), atl_j_max=jj.max(), atl_k_min=kk.min(), atl_k_max=kk.max())
            if r.role == "dev" and r.mask_path:
                mi, mj, mk = np.nonzero(self.load_mask(r))
                row.update(mask_i_min=mi.min(), mask_i_max=mi.max(), mask_j_min=mj.min(), mask_j_max=mj.max(), mask_k_min=mk.min(), mask_k_max=mk.max())
            rows.append(row)
        cw = pd.DataFrame(rows)
        for c in ("mask_i_min", "mask_i_max", "mask_j_min", "mask_j_max", "mask_k_min", "mask_k_max"):
            if c not in cw:
                cw[c] = np.nan
        dev = cw[cw.role == "dev"]
        ZS = vox_size(self.load_spir(subj.iloc[0])[1])
        MARGIN_VOX = {ax: int(np.ceil(self.cfg["margin_mm"] / ZS[n])) for n, ax in enumerate(("i", "j"))}
        need = {}
        for ax in ("i", "j"):
            src = dev if len(dev) else cw
            c = src[f"c{ax}"]
            parts = [c - src[f"atl_{ax}_min"], src[f"atl_{ax}_max"] - c]
            if len(dev):
                parts += [c - dev[f"mask_{ax}_min"], dev[f"mask_{ax}_max"] - c]
            need[ax] = float(np.nanmax(np.concatenate([np.asarray(p, float) for p in parts])))
        SIZE = {}
        for ax in ("i", "j"):
            half = int(np.ceil(need[ax])) + MARGIN_VOX[ax]
            SIZE[ax] = int(min(np.ceil(2 * half / self.cfg["round_to_multiple"]) * self.cfg["round_to_multiple"], cw[f"n{'x' if ax == 'i' else 'y'}"].min()))
        fitted = dict(SIZE)
        if self.cfg["size_mode"] == "frozen":
            SIZE = {"i": int(self.cfg["size_voxels"][0]), "j": int(self.cfg["size_voxels"][1])}
        log(f"largest extent from the SN centre (voxels): i {need['i']:.1f}, j {need['j']:.1f} | margin {MARGIN_VOX} | "
            f"size from this data {fitted['i']} x {fitted['j']} | crop size used ({self.cfg['size_mode']}) {SIZE['i']} x {SIZE['j']} x all slices "
            f"({SIZE['i'] * ZS[0]:.1f} x {SIZE['j'] * ZS[1]:.1f} mm)")
        for ax, n in (("i", "nx"), ("j", "ny")):
            start = np.rint(cw[f"c{ax}"]).astype(int) - SIZE[ax] // 2
            cw[f"{ax}0"] = np.clip(start, 0, cw[n] - SIZE[ax])
            cw[f"{ax}_shift"] = cw[f"{ax}0"] - start
            cw[f"atl_margin_{ax}"] = np.minimum(cw[f"atl_{ax}_min"] - cw[f"{ax}0"], cw[f"{ax}0"] + SIZE[ax] - 1 - cw[f"atl_{ax}_max"])
            cw[f"mask_margin_{ax}"] = np.minimum(cw[f"mask_{ax}_min"] - cw[f"{ax}0"], cw[f"{ax}0"] + SIZE[ax] - 1 - cw[f"mask_{ax}_max"])
        cw["atl_margin_min"] = cw[["atl_margin_i", "atl_margin_j"]].min(axis=1)
        cw["mask_margin_min"] = cw[["mask_margin_i", "mask_margin_j"]].min(axis=1)
        cw["crop_shifted"] = (cw.i_shift != 0) | (cw.j_shift != 0)
        min_ok = min(MARGIN_VOX.values()) // 2
        cw["flag_crop_tight"] = (cw.atl_margin_min < min_ok) | (cw.mask_margin_min < min_ok)
        cw["flag_crop_cuts"] = (cw.atl_margin_min < 0) | (cw.mask_margin_min < 0)
        cw.to_csv(self.OUT / "crop_windows.csv", index=False)
        CROP = dict(size_i=SIZE["i"], size_j=SIZE["j"], z="all slices", size_mode=self.cfg["size_mode"], size_fitted_from_this_data=fitted,
                    margin_mm=self.cfg["margin_mm"], margin_vox=MARGIN_VOX, multiple=self.cfg["round_to_multiple"],
                    sized_from="development subjects: binary mask extent and atlas SN/RN/PBP/VTA extent (p>0.05) around the atlas SN centre "
                               "(frozen at 112 x 80 for the trained model)",
                    placed_by="registered CIT168 SNc+SNr centre of mass (label-free, same for test subjects)",
                    restore="full-grid voxel [i0:i0+size_i, j0:j0+size_j, :] = crop; i0/j0 per subject in crop_windows.csv")
        json.dump(CROP, open(self.OUT / "crop_window.json", "w"), indent=2)
        log(f"atlas margin inside the crop (voxels): min {cw.atl_margin_min.min()}, median {cw.atl_margin_min.median():.0f} | "
            f"development mask margin: min {cw.mask_margin_min.min():.0f} | shifted to stay inside the image: {int(cw.crop_shifted.sum())} | "
            f"tight: {cw.subject_id[cw.flag_crop_tight].tolist()} | CUT: {cw.subject_id[cw.flag_crop_cuts].tolist()}")
        if self.cfg["stop_if_crop_cuts"]:
            assert not cw.flag_crop_cuts.any(), "the crop cuts the atlas region or a mask for some subjects"
        elif cw.flag_crop_cuts.any():
            log("WARNING: the crop cuts the atlas region or a mask for " + str(cw.subject_id[cw.flag_crop_cuts].tolist())
                + " -> these subjects are excluded from inference by the QC rule")
        if len(dev) and dev.mask_i_min.notna().any():
            fig, ax = plt.subplots(figsize=(5.5, 5))
            for _, q in dev.iterrows():
                ax.add_patch(plt.Rectangle(((q.mask_i_min - q.ci) * ZS[0], (q.mask_j_min - q.cj) * ZS[1]), (q.mask_i_max - q.mask_i_min) * ZS[0],
                                           (q.mask_j_max - q.mask_j_min) * ZS[1], fill=False, color="red", lw=0.4, alpha=0.6))
            ax.add_patch(plt.Rectangle((-SIZE["i"] / 2 * ZS[0], -SIZE["j"] / 2 * ZS[1]), SIZE["i"] * ZS[0], SIZE["j"] * ZS[1], fill=False, color="k", lw=1.5))
            ax.set_xlim(-45, 45); ax.set_ylim(-45, 45); ax.set_aspect("equal")
            ax.set_xlabel("i offset from atlas SN centre (mm)"); ax.set_ylabel("j offset (mm)")
            ax.set_title("red: bounding box of each development mask; black: crop window", fontsize=8)
            plt.tight_layout(); fig.savefig(SUB["figs"] / "crop_window.png", dpi=110); plt.close(fig)
        self.cw, self.SIZE, self.CROP = cw, SIZE, CROP

    # ------------------------------------------------------------------------------------------------ per subject
    def bootstrap_cv_median(self, v, n=200):
        if len(v) < 5:
            return np.nan
        meds = np.median(self.RNG.choice(v, size=(n, len(v)), replace=True), axis=1)
        return float(meds.std() / abs(np.median(v))) if np.median(v) != 0 else np.nan

    @staticmethod
    def n4_correct(arr, aff):
        import SimpleITK as sitk
        img = sitk.GetImageFromArray(np.ascontiguousarray(arr.transpose(2, 1, 0)))
        img.SetSpacing([float(x) for x in vox_size(aff)])
        msk = sitk.OtsuThreshold(img, 0, 1, 200)
        small = sitk.Shrink(img, [2, 2, 1]); small_m = sitk.Shrink(msk, [2, 2, 1])
        f = sitk.N4BiasFieldCorrectionImageFilter()
        f.Execute(small, small_m)
        log_bias = f.GetLogBiasFieldAsImage(img)
        return (arr / np.exp(sitk.GetArrayFromImage(log_bias).transpose(2, 1, 0))).astype(np.float32)

    def per_subject(self):
        """crop, both normalisations, optional N4, binary + left/right masks, QC numbers"""
        cw, SIZE, SUB, X_MID, CRUS_T = self.cw, self.SIZE, self.SUB, self.X_MID, self.CRUS_T
        REF_ERODE_VOX, REF_MIN_VOX = int(self.cfg["ref_erode_vox"]), int(self.cfg["ref_min_vox"])
        t0 = time.time(); rows = []
        for n, r in enumerate(self.subj.itertuples(), 1):
            sid = r.subject_id
            q = cw.loc[cw.subject_id == sid].iloc[0]
            i0, j0, si, sj = int(q.i0), int(q.j0), SIZE["i"], SIZE["j"]
            box = (slice(i0, i0 + si), slice(j0, j0 + sj), slice(None))
            spir, aff = self.load_spir(r)
            caff = crop_affine(aff, i0, j0)
            zs = vox_size(aff); vvox = float(np.prod(zs))
            atl = self.load_atlas(sid)
            idx, valid = self.load_index(sid)
            # hemisphere map (1 = left, 2 = right, 0 = outside template) from the template midline
            xw = self.tmpl_world_x(idx, valid)
            hemi = np.where(valid, np.where(xw >= X_MID, 2, 1), 0).astype(np.uint8)
            # reference region (crus cerebri) carried from template space; voxels with any SN probability removed for safety
            sn_any = (atl["SNc"] + atl["SNr"]) > 0.05
            ref_t = (self.tmpl_lookup(idx, valid, CRUS_T) > 0) & ~sn_any
            # refinement (label-free, from this subject's own image): slices with atlas SN only, no CSF-dark voxels,
            # 1-voxel in-plane erosion, largest piece per side; falls back step by step if too little is left
            sn_slices = ((atl["SNc"] + atl["SNr"]) >= 0.25).any(axis=(0, 1))
            crop_v = spir[i0:i0 + SIZE["i"], j0:j0 + SIZE["j"]][valid[i0:i0 + SIZE["i"], j0:j0 + SIZE["j"]]]
            csf_thr = float(threshold_multiotsu(crop_v[crop_v > 0], classes=3)[0])
            ref1 = ref_t & sn_slices[None, None, :] & (spir > csf_thr)
            ref2 = ndimage.binary_erosion(ref1, structure=np.ones((3, 3, 1), bool), iterations=REF_ERODE_VOX)
            ref = np.zeros(spir.shape, bool)
            for lab in (1, 2):
                cc, ncc = ndimage.label(ref2 & (hemi == lab))
                if ncc:
                    ref |= cc == (np.argmax(np.bincount(cc.ravel())[1:]) + 1)
            ref_step = "refined"
            if ref.sum() < REF_MIN_VOX:
                ref, ref_step = ref1, "no erosion (too few voxels)"
            if ref.sum() < REF_MIN_VOX:
                ref, ref_step = ref_t, "template ROI only (too few voxels)"
            ref_vals = spir[ref]
            ref_med = float(np.median(ref_vals)) if ref_vals.size else np.nan
            # "midbrain tissue" for QC: crop voxels in the template, outside the dilated SN/PBP/VTA and outside the reference
            near_sn = np.zeros(spir.shape, bool)
            for c in ("SNc", "SNr", "PBP", "VTA"):
                if c in atl:
                    near_sn |= atl[c] > 0.05
            near_sn = ndimage.binary_dilation(near_sn, iterations=3)
            crop = spir[box]
            p99c = float(np.percentile(crop, 99))
            tissue = (valid & ~near_sn & ~ref_t)[box] & (crop > 0.05 * p99c)
            s1, s2 = float(np.median(crop[tissue])), float(np.percentile(crop[tissue], 90))
            # (a) trimmed z-score inside the crop (statistics from voxels between the 1st and 99th percentile; image NOT clipped)
            fg = crop[crop > 0.05 * p99c]
            lo_, hi_ = np.percentile(fg, [1, 99])
            core = fg[(fg >= lo_) & (fg <= hi_)]
            mu, sd = float(core.mean()), float(core.std())
            z = (crop - mu) / sd
            # (b) ratio to the reference median (contrast ratio; CNR = ratio - 1)
            rr = crop / ref_med if np.isfinite(ref_med) and ref_med > 0 else np.full(crop.shape, np.nan, np.float32)
            # background noise proxy (corners of the full image)
            cs = 16
            corners = np.concatenate([spir[:cs, :cs].ravel(), spir[-cs:, :cs].ravel(), spir[:cs, -cs:].ravel(), spir[-cs:, -cs:].ravel()])
            bg_sd = float(corners.std())
            row = dict(subject_id=sid, group=r.group, role=r.role, test_like_slope=sid in self.TEST_LIKE, scale_group=self.SCALE_GROUP[sid],
                       philips_scale_slope=self.SLOPE.get(sid, np.nan), i0=i0, j0=j0, ref_n_template=int(ref_t.sum()), ref_refine_step=ref_step,
                       csf_threshold_raw=csf_thr, ref_n_vox=int(ref.sum()), ref_vol_mm3=float(ref.sum() * vvox), ref_median_raw=ref_med,
                       ref_boot_cv=self.bootstrap_cv_median(ref_vals), ref_left_n=int((ref & (hemi == 1)).sum()), ref_right_n=int((ref & (hemi == 2)).sum()),
                       tissue_median_raw=s1, tissue_p90_raw=s2, crop_p99_raw=p99c, z_mu=mu, z_sd=sd,
                       tissue_median_z=(s1 - mu) / sd, tissue_median_ratio=s1 / ref_med if ref_med else np.nan,
                       ref_median_z=(ref_med - mu) / sd, bg_sd=bg_sd, snr_proxy=s1 / bg_sd if bg_sd > 0 else np.nan,
                       hemi_valid_frac_crop=float((hemi[box] > 0).mean()))
            # atlas left-hemisphere maps (if written) must fall in hemisphere 1
            if "SNc_left" in atl and atl["SNc_left"].sum() > 0:
                row["snc_left_in_hemi1"] = float(atl["SNc_left"][hemi == 1].sum() / atl["SNc_left"].sum())
            # saves: crops on the SPIR grid (crop affine keeps world positions)
            save_nii(crop, caff, SUB["raw"] / f"{sid}_spir_raw.nii.gz")
            save_nii(z, caff, SUB["zscore"] / f"{sid}_spir_zscore.nii.gz")
            save_nii(rr, caff, SUB["refratio"] / f"{sid}_spir_refratio.nii.gz")
            save_nii(np.stack([atl[c][box] for c in self.CHANNELS], axis=-1), caff, SUB["atlas"] / f"{sid}_atlas.nii.gz")
            save_nii(hemi[box], caff, SUB["hemi"] / f"{sid}_hemi.nii.gz", np.uint8)
            save_nii((ref.astype(np.uint8) * hemi)[box], caff, SUB["ref_roi"] / f"{sid}_ref_crus.nii.gz", np.uint8)
            if self.cfg["run_n4"]:
                save_nii(self.n4_correct(spir, aff)[box], caff, SUB["n4_raw"] / f"{sid}_spir_n4_raw.nii.gz")
            # masks: binary, SPIR header for every subject, values only as metadata
            if r.mask_path:
                m = self.load_mask(r)
                lr = (m * hemi).astype(np.uint8)
                cc, ncc = ndimage.label(m, structure=np.ones((3, 3, 3)))
                sizes = np.bincount(cc.ravel())[1:]
                purity, joined = [], False
                for lab in np.argsort(sizes)[::-1][:2] + 1:
                    hh = hemi[cc == lab]; hh = hh[hh > 0]
                    if hh.size:
                        f1 = float((hh == 1).mean()); purity.append(max(f1, 1 - f1))
                        joined |= min(f1, 1 - f1) > 0.05
                two = np.argsort(sizes)[::-1][:2] + 1 if ncc >= 2 else []
                sides = [int(np.bincount(hemi[cc == lab][hemi[cc == lab] > 0]).argmax()) if (hemi[cc == lab] > 0).any() else 0 for lab in two]
                row.update(has_mask=True, mask_excluded=sid in self.excluded_masks, mask_n_vox=int(m.sum()), mask_vol_mm3=float(m.sum() * vvox),
                           mask_vol_left_mm3=float((lr == 1).sum() * vvox), mask_vol_right_mm3=float((lr == 2).sum() * vvox),
                           mask_vox_outside_template=int((m & (hemi == 0)).sum()), mask_vox_outside_crop=int(m.sum() - m[box].sum()),
                           mask_n_components=int(ncc), mask_joined_lr=bool(joined), mask_min_component_purity=min(purity) if purity else np.nan,
                           mask_two_largest_same_side=bool(len(sides) == 2 and sides[0] == sides[1]),
                           mask_raw_min=float(spir[m].min()), mask_raw_max=float(spir[m].max()),   # METADATA only, never an input
                           mask_ratio_min=float(spir[m].min() / ref_med) if ref_med else np.nan)
                save_nii(m, aff, SUB["masks_full"] / f"{sid}_sn_mask_bin.nii.gz", np.uint8)
                save_nii(lr, aff, SUB["masks_full"] / f"{sid}_sn_mask_lr.nii.gz", np.uint8)
                save_nii(m[box], caff, SUB["mask_bin"] / f"{sid}_mask_bin.nii.gz", np.uint8)
                save_nii(lr[box], caff, SUB["mask_lr"] / f"{sid}_mask_lr.nii.gz", np.uint8)
            else:
                row.update(has_mask=False, mask_excluded=False)
            row = jsonable(row)
            json.dump(row, open(SUB["per_subject"] / f"{sid}.json", "w"), indent=1)
            rows.append(row)
            if n % 10 == 0 or n == len(self.subj):
                log(f"{n}/{len(self.subj)} done, {time.time() - t0:.0f} s, free disk {shutil.disk_usage(str(self.OUT)).free / 1e9:.1f} GB")
        ps = pd.DataFrame(rows)
        ps.to_csv(self.OUT / "per_subject_crop.csv", index=False)
        log(f"reference ROI voxels: min {ps.ref_n_vox.min()} median {int(ps.ref_n_vox.median())} | "
            f"bootstrap CV of the reference median: max {round(ps.ref_boot_cv.max(), 4)}")
        self.ps = ps

    # ------------------------------------------------------------------------------------------------ summaries
    def normalisation_record(self):
        """normalisation chosen by PURPOSE (network input: z-score; measurements: crus ratio). The table is descriptive only:
        it shows how much of the scale-group difference each normalisation leaves."""
        ps = self.ps
        devp = ps[ps.role == "dev"].copy()
        tl = devp.scale_group == "high_slope"
        desc = []
        for label, a, b in (("high vs low slope (development)", devp[tl], devp[~tl]), ("test vs development", ps[ps.role == "test"], devp)):
            desc.append(dict(comparison=label, n=f"{len(a)} vs {len(b)}",
                             d_raw_tissue=cohen_d(np.log(a.tissue_median_raw), np.log(b.tissue_median_raw)),
                             d_zscore_tissue=cohen_d(a.tissue_median_z, b.tissue_median_z),
                             d_zscore_crus=cohen_d(a.ref_median_z, b.ref_median_z),
                             d_refratio_tissue=cohen_d(a.tissue_median_ratio, b.tissue_median_ratio)))
        desc = pd.DataFrame(desc)
        log("descriptive differences after normalisation (Cohen's d; 0 = no difference):\n" + desc.round(2).to_string(index=False))
        SEL = dict(network_input=self.cfg["norm_network"], measurements=self.cfg["norm_measure"],
                   reason="chosen by purpose. Neither normalisation removes the difference between the two intensity-scale groups "
                          "(it is partly contrast, not only scale), so the scale group is kept as a covariate and training uses both groups.",
                   scale_group_rule=f"PhilipsScaleSlope > {self.cfg['high_slope_min']} -> high_slope",
                   descriptive=desc.to_dict("records"),
                   candidates={"zscore": "(x - mean) / sd, statistics from crop voxels > 5% of crop p99 and between their 1st and 99th percentile; "
                                         "image not clipped",
                               "refratio": "x / median(crus cerebri reference ROI); CNR = ratio - 1"})
        json.dump(SEL, open(self.OUT / "normalisation_selection.json", "w"), indent=2, default=str)

    def mask_checks(self):
        """left/right split by the template midline, joined masks, mask metadata"""
        ps = self.ps
        if "has_mask" not in ps or not ps.has_mask.astype(bool).any():
            write_text(self.OUT / "mask_preparation_summary.txt", "no manual masks in this dataset")
            return
        mk = ps[ps.has_mask.astype(bool)].copy()
        devm = mk[mk.role == "dev"]
        L = [f"masks processed: {len(mk)} (development {len(devm)}; excluded mask(s): {mk.subject_id[mk.mask_excluded.astype(bool)].tolist()})",
             f"mask voxels outside the crop (must be 0 for development): max {int(devm.mask_vox_outside_crop.max()) if len(devm) else 0}",
             f"mask voxels outside the template (no hemisphere): max {int(mk.mask_vox_outside_template.max())}"]
        if "snc_left_in_hemi1" in ps:
            L.append(f"atlas left SNc inside hemisphere 1 (left) in SPIR space: min {ps.snc_left_in_hemi1.min():.3f}, median {ps.snc_left_in_hemi1.median():.3f}")
        if len(devm):
            joined = devm.subject_id[devm.mask_joined_lr.astype(bool)].tolist()
            L += [f"masks whose largest component crosses the midline (split by the midline rule): {joined}",
                  f"two largest components on the SAME side (would mean a midline problem): {devm.subject_id[devm.mask_two_largest_same_side.astype(bool)].tolist()}",
                  f"component purity (share of a component on one side), masks not joined: min "
                  f"{devm.loc[~devm.mask_joined_lr.astype(bool), 'mask_min_component_purity'].min():.3f}",
                  f"left/right volume ratio of development masks (left / right): median {(devm.mask_vol_left_mm3 / devm.mask_vol_right_mm3).median():.2f}, "
                  f"range {(devm.mask_vol_left_mm3 / devm.mask_vol_right_mm3).min():.2f} - {(devm.mask_vol_left_mm3 / devm.mask_vol_right_mm3).max():.2f}",
                  f"mask lower bound as ratio to crus (metadata): median {devm.mask_ratio_min.median():.3f}, "
                  f"range {devm.mask_ratio_min.min():.3f} - {devm.mask_ratio_min.max():.3f}"]
        txt = "\n".join(L)
        log(txt)
        write_text(self.OUT / "mask_preparation_summary.txt", txt)

    def qc_pages(self):
        """one row per subject: crop at the atlas SN centre slice and +-3 slices; raw crop with mask (red), atlas SNc p=0.5
        (green), crus reference ROI (cyan), midline (white dashed); network input image; left/right mask (blue / orange)"""
        cw, SUB, CHANNELS = self.cw, self.SUB, self.CHANNELS
        NORM = self.cfg["norm_network"]
        ids = list(self.subj.subject_id); PER_PAGE = 10
        pages = [ids[i:i + PER_PAGE] for i in range(0, len(ids), PER_PAGE)]
        for pn, page in enumerate(pages, 1):
            fig, axs = plt.subplots(len(page), 5, figsize=(13, 2.5 * len(page)))
            axs = np.atleast_2d(axs)
            for ri, sid in enumerate(page):
                q = cw.loc[cw.subject_id == sid].iloc[0]
                raw = nib.load(str(SUB["raw"] / f"{sid}_spir_raw.nii.gz")).get_fdata(dtype=np.float32)
                nrm = nib.load(str(SUB[NORM] / f"{sid}_spir_{NORM}.nii.gz")).get_fdata(dtype=np.float32)
                atc = nib.load(str(SUB["atlas"] / f"{sid}_atlas.nii.gz")).get_fdata(dtype=np.float32)[..., CHANNELS.index("SNc")]
                hm = nib.load(str(SUB["hemi"] / f"{sid}_hemi.nii.gz")).get_fdata().astype(np.uint8)
                rf = nib.load(str(SUB["ref_roi"] / f"{sid}_ref_crus.nii.gz")).get_fdata() > 0
                fm = SUB["mask_lr"] / f"{sid}_mask_lr.nii.gz"
                mlr = nib.load(str(fm)).get_fdata().astype(np.uint8) if fm.exists() else None
                k = int(np.clip(round(q.ck), 0, raw.shape[2] - 1))
                lo, hi = np.percentile(raw[..., max(0, k - 3):k + 4], [1, 99.7])
                for c_i, kk in enumerate((k - 3, k, k + 3)):
                    ax = axs[ri, c_i]; ax.set_xticks([]); ax.set_yticks([])
                    if not 0 <= kk < raw.shape[2]:
                        ax.set_facecolor("black"); continue
                    ax.imshow(raw[..., kk].T, cmap="gray", origin="lower", vmin=lo, vmax=hi)
                    if (atc[..., kk] > 0.5).any():
                        ax.contour(atc[..., kk].T, [0.5], colors="lime", linewidths=0.6)
                    if rf[..., kk].any():
                        ax.contour(rf[..., kk].T.astype(np.uint8), [0.5], colors="cyan", linewidths=0.6)
                    if mlr is not None and mlr[..., kk].any():
                        ax.contour((mlr[..., kk] > 0).T.astype(np.uint8), [0.5], colors="red", linewidths=0.6)
                    ax.contour((hm[..., kk] == 2).T.astype(np.uint8), [0.5], colors="white", linewidths=0.5, linestyles="dashed")
                    if ri == 0:
                        ax.set_title(f"raw crop, slice centre {kk - k:+d}", fontsize=7)
                    if c_i == 0:
                        ax.text(0.02, 0.95, sid, transform=ax.transAxes, color="yellow", fontsize=6, va="top")
                ax = axs[ri, 3]; ax.set_xticks([]); ax.set_yticks([])
                vlo, vhi = np.nanpercentile(nrm[..., k], [1, 99.7])
                ax.imshow(nrm[..., k].T, cmap="gray", origin="lower", vmin=vlo, vmax=vhi)
                if ri == 0:
                    ax.set_title(f"{NORM} (network input), centre slice", fontsize=7)
                ax = axs[ri, 4]; ax.set_xticks([]); ax.set_yticks([])
                ax.imshow(raw[..., k].T, cmap="gray", origin="lower", vmin=lo, vmax=hi)
                if mlr is not None:
                    ov = np.zeros(mlr.shape[:2] + (4,)); ov[mlr[..., k] == 1] = (0.2, 0.4, 1, 0.55); ov[mlr[..., k] == 2] = (1, 0.55, 0, 0.55)
                    ax.imshow(np.transpose(ov, (1, 0, 2)), origin="lower")
                ax.contour((hm[..., k] == 2).T.astype(np.uint8), [0.5], colors="white", linewidths=0.5, linestyles="dashed")
                if ri == 0:
                    ax.set_title("mask left (blue) / right (orange)", fontsize=7)
            fig.suptitle(f"crop QC page {pn}/{len(pages)}: red = binary mask, green = atlas SNc p=0.5, cyan = crus reference ROI, "
                         "white dashed = template midline", fontsize=8, y=0.995)
            plt.tight_layout(rect=(0, 0, 1, 0.985))
            fig.savefig(SUB["figs"] / f"qc_crop_page{pn}.png", dpi=90); plt.close(fig)

    def qc_table(self):
        """QC table, inclusion, decisions and status"""
        ps, cw, atlas_table, cfg = self.ps, self.cw, self.atlas_table, self.cfg
        REF_MIN_VOX, REF_MAX_BOOT_CV = int(cfg["ref_min_vox"]), float(cfg["ref_max_boot_cv"])
        MANUAL_EXCLUDE, MANUAL_NOTES = dict(cfg["manual_exclude"]), dict(cfg["manual_notes"])
        q = ps.merge(cw[["subject_id", "ci", "cj", "ck", "atl_margin_min", "mask_margin_min", "crop_shifted", "flag_crop_tight", "flag_crop_cuts"]],
                     on="subject_id").merge(atlas_table[["subject_id", "grade"] + (["reg_strategy"] if "reg_strategy" in atlas_table else [])]
                                            .rename(columns={"grade": "reg_grade"}), on="subject_id")
        asbool = lambda s: s.map(lambda v: bool(v) if isinstance(v, (bool, np.bool_)) else False)

        def robust_low(s):
            s = s.astype(float); med = s.median(); mad = (s - med).abs().median() * 1.4826
            return s < med - 3 * mad
        q["flag_reg_fail"] = q.subject_id.isin(self.REG_FAIL)
        q["flag_ref_unreliable"] = (q.ref_n_vox < REF_MIN_VOX) | (q.ref_boot_cv > REF_MAX_BOOT_CV) | (q.ref_left_n < REF_MIN_VOX // 4) | (q.ref_right_n < REF_MIN_VOX // 4)
        q["flag_low_snr_proxy"] = robust_low(q.snr_proxy)
        q["flag_mask_midline"] = asbool(q["mask_two_largest_same_side"]) if "mask_two_largest_same_side" in q else False
        q["flag_mask_outside_crop"] = (pd.to_numeric(q["mask_vox_outside_crop"], errors="coerce").fillna(0) > 0) if "mask_vox_outside_crop" in q else False
        q["flag_ref_fallback"] = q.ref_refine_step != "refined"

        def reasons(x):
            out = []
            if x.flag_reg_fail: out.append("registration graded fail")
            if x.flag_crop_cuts: out.append("crop cuts atlas region or mask")
            if x.flag_crop_tight: out.append("crop margin < half the planned margin (review)")
            if x.flag_ref_unreliable: out.append("reference ROI small or unstable (review; affects ratio measures)")
            if x.flag_ref_fallback: out.append(f"reference refinement fell back: {x.ref_refine_step} (review)")
            if x.flag_low_snr_proxy: out.append("low background-noise SNR proxy (review)")
            if x.flag_mask_midline: out.append("both mask components on one side of the midline (review)")
            if x.flag_mask_outside_crop: out.append("mask voxels outside the crop")
            if bool(x.mask_excluded): out.append("mask excluded in the audit (outlier mask)")
            if x.subject_id in MANUAL_EXCLUDE: out.append("manual: " + MANUAL_EXCLUDE[x.subject_id])
            return "; ".join(out)
        q["qc_reasons"] = q.apply(reasons, axis=1)
        man_scope = {s: v.split(":")[0].strip().lower() for s, v in MANUAL_EXCLUDE.items()}
        blocks_inf = q.flag_reg_fail | q.flag_crop_cuts | q.subject_id.map(lambda s: man_scope.get(s) in ("inference", "both"))
        q["include_inference"] = ~blocks_inf
        q["include_training"] = q.include_inference & (q.role == "dev") & q.has_mask.astype(bool) & ~q.mask_excluded.astype(bool) \
                                & ~q.flag_mask_outside_crop & ~q.subject_id.map(lambda s: man_scope.get(s) in ("training", "both"))
        q["notes"] = q.subject_id.map(lambda s: MANUAL_NOTES.get(s, ""))
        q.to_csv(self.OUT / "qc_table.csv", index=False)     # includes scale_group and philips_scale_slope (covariate)

        ov = atlas_table[atlas_table.subject_id.isin(self.override)][["subject_id", "qc_centroid_dist_mm"]] if "qc_centroid_dist_mm" in atlas_table else pd.DataFrame(columns=["subject_id", "qc_centroid_dist_mm"])
        decisions = {
            "crop": self.CROP,
            "normalisation": {"network_input": cfg["norm_network"], "measurements": cfg["norm_measure"], "selection_file": "normalisation_selection.json",
                              "reference_region": f"crus cerebri: CIT168 white matter {cfg['ref_band_mm'][0]}-{cfg['ref_band_mm'][1]} mm anterior to the atlas SN "
                              "(p>=0.25), outside all CIT168 structures, within the SN superior-inferior range, largest piece per side; carried to SPIR by "
                              "the index map; then per subject: SN-probability voxels removed, only slices with atlas SN, voxels below the subject's "
                              f"CSF/tissue threshold removed, {cfg['ref_erode_vox']}-voxel in-plane erosion, largest piece per side (fallback without "
                              "erosion when too few voxels remain). Never located through the masks.",
                              "scale_group": f"PhilipsScaleSlope > {cfg['high_slope_min']} = high_slope; covariate"},
            "n4_denoising": {"n4": "N4 raw crops saved" if cfg["run_n4"] else "off", "denoising": "off"},
            "masks": {"binary": "value != 0, uint8, SPIR sform header for all subjects", "values": "never used; raw SPIR min/max inside the "
                      "mask kept as metadata only", "midline_rule": f"{cfg['midline_rule']}: hemisphere = side of the CIT168 midline (x = {self.X_MID:.2f} mm "
                      "RAS in template space, centre of the bilateral red nucleus) for every SPIR voxel via the index map; 1 = left, 2 = right",
                      "excluded_masks": self.excluded_masks},
            "qc": {"rules": "inference excluded only for: registration fail, crop cutting the SN region, or a manual reason. Training additionally "
                            "needs a development mask that is not excluded. Other flags are for review only.",
                   "manual_exclusions": MANUAL_EXCLUDE, "viewed_qc_pages": cfg["viewed_qc_pages"]},
            "override_masks_centroid_distance_mm": jsonable(dict(zip(ov.subject_id, ov.qc_centroid_dist_mm))),
            "fitting_rules_followed": "crop size frozen (sized from development subjects only); normalisation per subject from its own image; "
                                      "no diagnosis groups and no test data used for any choice",
        }
        json.dump(decisions, open(self.OUT / "crop_decisions.json", "w"), indent=2, default=str)
        checks = {
            "crop contains the atlas SN region for all subjects and every development mask": not q.flag_crop_cuts.any() and not q.flag_mask_outside_crop.any(),
            "reference ROI reliable for all subjects": not q.flag_ref_unreliable.any(),
            "hemisphere naming consistent (atlas left SNc in hemisphere 1)": bool(ps.get("snc_left_in_hemi1", pd.Series([1.0])).min() > 0.95),
            "no mask with both components on one side": not q.flag_mask_midline.any(),
            "override masks within 3 mm of the atlas SNc": bool((ov.qc_centroid_dist_mm.astype(float) < 3).all()),
            "QC pages viewed": bool(cfg["viewed_qc_pages"]),
        }
        L = ["CROP AND NORMALISATION STATUS", "=" * 60] + [f"{'PASS' if ok else 'OPEN/FAIL'}: {k}" for k, ok in checks.items()]
        L += ["", f"crop {self.SIZE['i']} x {self.SIZE['j']} x all slices | network input: {cfg['norm_network']} | measurements: {cfg['norm_measure']}",
              f"reference ROI voxels after refinement: min {int(q.ref_n_vox.min())}, median {int(q.ref_n_vox.median())}; fallbacks: "
              f"{q.subject_id[q.flag_ref_fallback].tolist()}",
              f"included for inference: {int(q.include_inference.sum())}/{len(q)} | for training: {int(q.include_training.sum())} development subjects",
              "subjects with flags or notes:"]
        fl = q[q.qc_reasons != ""][["subject_id", "qc_reasons", "include_inference", "include_training"]]
        L.append(fl.to_string(index=False) if len(fl) else "  none")
        write_text(self.OUT / "crop_report.txt", "\n".join(L))
        log("\n".join(L))

    def restore_check(self):
        """puts every crop back into the full SPIR grid and checks it is identical to the original (images, masks, world coordinates)"""
        SUB, cw, SIZE = self.SUB, self.cw, self.SIZE
        res = []
        for r in self.subj.itertuples():
            sid = r.subject_id
            spir, aff = self.load_spir(r)
            ci = nib.load(str(SUB["raw"] / f"{sid}_spir_raw.nii.gz"))
            q = cw.loc[cw.subject_id == sid].iloc[0]
            back, aff_back = uncrop(ci.get_fdata(dtype=np.float32), q, aff)
            box = (slice(int(q.i0), int(q.i0) + SIZE["i"]), slice(int(q.j0), int(q.j0) + SIZE["j"]), slice(None))
            world_ok = np.allclose(ci.affine, crop_affine(aff, int(q.i0), int(q.j0)), atol=1e-5) and np.allclose(aff_back, aff, atol=1e-5)
            row = dict(subject_id=sid, shape_ok=back.shape == spir.shape, image_identical_in_crop=bool(np.array_equal(back[box], spir[box])),
                       world_coordinates_ok=bool(world_ok))
            if r.mask_path:
                mb = nib.load(str(SUB["mask_bin"] / f"{sid}_mask_bin.nii.gz")).get_fdata().astype(np.uint8)
                mfull, _ = uncrop(mb, q, aff)
                row["mask_identical_full_grid"] = bool(np.array_equal(mfull.astype(bool), self.load_mask(r)))
            res.append(row)
        res = pd.DataFrame(res)
        res.to_csv(self.OUT / "restore_check.csv", index=False)
        ok_cols = [c for c in res.columns if c != "subject_id"]
        log("restore check passed / checked: " + str({c: f"{int(res[c].dropna().astype(bool).sum())}/{int(res[c].notna().sum())}" for c in ok_cols}))
        bad = res[~res[ok_cols].fillna(True).astype(bool).all(axis=1)].subject_id.tolist()
        assert not bad, f"restore check failed for {bad}: see restore_check.csv"


def uncrop(crop_arr, q, aff, fill=0):
    """crop on the crop window -> array on the full SPIR grid + the full SPIR affine (sform)"""
    full = np.full((int(q.nx), int(q.ny), int(q.nz)) + crop_arr.shape[3:], fill, dtype=crop_arr.dtype)
    full[int(q.i0):int(q.i0) + crop_arr.shape[0], int(q.j0):int(q.j0) + crop_arr.shape[1]] = crop_arr
    return full, aff


def run(ctx):
    S = CropStage(ctx)
    S.template_space()
    S.crop_window()
    S.per_subject()
    S.normalisation_record()
    S.mask_checks()
    S.qc_pages()
    S.qc_table()
    S.restore_check()
