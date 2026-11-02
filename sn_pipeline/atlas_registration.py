"""
CIT168 atlas -> subject T1w -> native SPIR space.

1. Registers the CIT168 T1w head template (700 um) to each subject's T1w (ANTs SyN: affine + nonlinear).
2. Chains that with the rigid SPIR -> T1w transform and resamples the CIT168 probability maps straight onto each
   subject's native SPIR grid (one interpolation; the SPIR itself is never resampled).
3. Checks the transform chain, measures registration quality, retries subjects that look wrong with a head-masked
   start-up (label-free rule), and makes QC pages. Manual masks are used only for QC numbers, never for any decision.

Outputs (registration/atlas_in_spir/):
  atlas_in_spir/<subject>_cit168_in_spir.nii.gz        4D probabilities on the SPIR grid (channels in 'channels')
  atlas_in_spir/<subject>_cit168_index_in_spir.nii.gz  CIT168 voxel index + 1 for every SPIR voxel (0 = outside the template)
  template_in_spir/<subject>_cit168_t1w_in_spir.nii.gz CIT168 T1w template on the SPIR grid (QC)
  transforms/  rigid transform in ANTs format, template -> T1w affine, chain description per subject
  per_subject/<subject>.json, atlas_registration.csv, atlas_registration_summary.txt, retry_results.csv, retry/<subject>/
  chain_test.json, atlas_registration_decisions.json, atlas_registration_report.txt, figs/qc_atlas_page<n>.png
"""
import json
import os
import re
import shutil
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from .subject_inventory import data_root_of
from .utils import find_file, nmi, read_atlas_labels, read_subjects, vox_size, write_text, log

STRATEGIES = ("default", "head_init", "initializer")
SUBDIRS = ("atlas_in_spir", "template_in_spir", "per_subject", "transforms")


def ants_from_nib(ants, arr, aff_ras):
    A = np.diag([-1.0, -1.0, 1.0]) @ np.asarray(aff_ras, float)[:3, :]
    M = A[:, :3]
    sp = np.linalg.norm(M, axis=0)
    return ants.from_numpy(np.asarray(arr, np.float32), origin=tuple(A[:, 3]), spacing=tuple(sp), direction=M / sp)


def upper_head(arr, aff, depth_mm):
    """head mask from the vertex down to depth_mm (RAS world z), its centroid (RAS mm), and whether the head
    reaches the top of the field of view (then the vertex is cut and the centroid is less reliable)."""
    a = ndimage.gaussian_filter(np.asarray(arr, np.float32), 1.0)
    thr = 0.1 * np.percentile(a[a > 0], 99)
    lab, n = ndimage.label(a > thr)
    if n == 0:
        raise RuntimeError("no head found in image")
    m = ndimage.binary_fill_holes(lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1))
    ijk = np.argwhere(m)
    z = ijk @ aff[2, :3] + aff[2, 3]
    keep = z > z.max() - depth_mm
    out = np.zeros(m.shape, bool); out[tuple(ijk[keep].T)] = True
    xyz = ijk[keep] @ aff[:3, :3].T + aff[:3, 3]
    corners = np.array([[i, j, k] for i in (0, m.shape[0] - 1) for j in (0, m.shape[1] - 1) for k in (0, m.shape[2] - 1)])
    z_fov_top = float((corners @ aff[2, :3] + aff[2, 3]).max())
    return out, xyz.mean(0), bool(z.max() > z_fov_top - 2.0)


def slab_corr(tmpl, t1s):
    v = (tmpl > 0) & (t1s > 0)
    return float(np.corrcoef(tmpl[v], t1s[v])[0, 1]) if v.sum() > 1000 else np.nan


class AtlasRegistration:
    def __init__(self, ctx):
        acfg = ctx.cfg["atlas_registration"]
        self.acfg = acfg
        # thread count is set before ANTs / ITK are imported
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(acfg["n_threads"])
        os.environ["ANTS_RANDOM_SEED"] = str(ctx.seed)      # ANTs reads this for its random sampling
        import ants
        import SimpleITK as sitk
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(int(acfg["n_threads"]))
        self.ants, self.sitk = ants, sitk
        # antspyx has no random_seed argument; ants.registration adds --random-seed when ants.config._random_seed is set
        ants.config._random_seed = ctx.seed
        self.seed = ctx.seed
        self.P = ctx.paths
        self.out = ctx.paths.atlas
        self.p_rigid = ctx.paths.rigid
        self.root = data_root_of(ctx)
        self.subj = read_subjects(ctx.paths)
        for d in SUBDIRS + ("figs",):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        self._check_rigid(ctx)
        self._load_atlas(ctx)
        self._prepare_template()

    # ---------------------------------------------------------------------------------------------------- inputs
    def _check_rigid(self, ctx):
        P_RIGID = self.p_rigid
        rule = json.load(open(P_RIGID / "header_rule_final.json"))
        assert rule["spir_world_space"].startswith("sform"), "rigid registration did not use the sform rule"
        rr = pd.read_csv(P_RIGID / "rigid_registration.csv")
        assert set(rr.subject_id) == set(self.subj.subject_id), "rigid_registration.csv does not cover all subjects"
        override = [s for s in ctx.cfg["dataset"]["mask_header_override"] if s in set(self.subj.subject_id)]
        bad = rr[rr.subject_id.isin(override) & (rr.max_corner_disp_mm > 2)]
        assert bad.empty, f"override subjects have qform-like rigid transforms: {bad.subject_id.tolist()}"
        missing_tfm = [s for s in self.subj.subject_id if not (P_RIGID / "transforms" / f"{s}_t1w_to_spir_rigid.tfm").exists()]
        assert not missing_tfm, f"missing rigid transforms: {missing_tfm}"
        self.rr, self.override = rr, override
        g = pd.read_csv(P_RIGID / "registration_qc_grades.csv")
        fails = g[g.grade.astype(str).str.lower().str.startswith("fail")].subject_id.tolist()
        log(f"rigid grades: {g.grade.astype(str).str.split(':').str[0].value_counts().to_dict()} | failed: {fails}")
        if (g.grade == "not graded").any():
            log("WARNING: some rigid registrations are not graded yet")
        self.exclude = fails

    def _load_atlas(self, ctx):
        ants = self.ants
        F = self.acfg["files"]
        ATL = ctx.atlas_dir
        self.F_TEMPLATE = find_file(ATL, F["template"])
        self.F_PROB = find_file(ATL, F["prob_bilateral"])
        self.F_LEFT = find_file(ATL, F["prob_left"])
        self.F_LABELS = find_file(ATL, F["labels"])
        labels = read_atlas_labels(self.F_LABELS)
        log(f"atlas labels: {labels}")
        T_img, P_img, L_img = nib.load(str(self.F_TEMPLATE)), nib.load(str(self.F_PROB)), nib.load(str(self.F_LEFT))
        assert P_img.shape[3] == len(labels), "number of atlas volumes differs from the label list"
        assert np.allclose(T_img.affine, P_img.affine, atol=1e-4) and np.allclose(L_img.affine, P_img.affine, atol=1e-4), "template/atlas grids differ"
        for need in ("SNc", "SNr", "RN"):
            assert need in labels, f"label {need} not in labels file"
        # the number in front of each label is used directly as the volume index of the 4D atlas -> must be 0 .. n-1
        n_vol = P_img.shape[3]
        lab_vals = sorted(labels.values())
        if lab_vals != list(range(n_vol)):
            raise ValueError(f"label numbers in {self.F_LABELS.name} are {lab_vals}, the atlas has {n_vol} volumes "
                             f"(index 0..{n_vol - 1}). The numbers are not the 0-based volume index.")

        # anatomical check that the chosen volumes are the right structures: SNc and SNr lie lateral to RN
        def _xyz_w(img, vol):
            a = np.asarray(img.dataobj[..., vol], dtype=np.float32)
            nz = np.nonzero(a > 0.01)
            if len(nz[0]) == 0:
                raise ValueError(f"atlas volume {vol} is empty")
            ijk1 = np.vstack([*nz, np.ones(len(nz[0]))]).astype(np.float64)
            return img.affine[:3] @ ijk1, a[nz].astype(np.float64)
        _c = {nm: _xyz_w(P_img, labels[nm]) for nm in ("SNc", "SNr", "RN")}
        _x_mid = float(np.average(_c["RN"][0][0], weights=_c["RN"][1]))       # bilateral RN is centred on the midline
        LAT = {nm: float(np.average(np.abs(xyz[0] - _x_mid), weights=w)) for nm, (xyz, w) in _c.items()}
        log("label check: mean distance from midline (mm): " + str({k: round(v, 1) for k, v in LAT.items()}))
        assert LAT["SNc"] > LAT["RN"] and LAT["SNr"] > LAT["RN"], "SNc/SNr are not lateral to RN: the label-to-volume mapping looks wrong."
        del _c
        CHANNELS = [("SNc", self.F_PROB, labels["SNc"]), ("SNr", self.F_PROB, labels["SNr"]), ("RN", self.F_PROB, labels["RN"])]
        for opt in ("PBP", "VTA"):
            if opt in labels:
                CHANNELS.append((opt, self.F_PROB, labels[opt]))
        # is prob_atlas the LEFT hemisphere? (checked on the data)
        xs = np.arange(P_img.shape[0])
        wx = (P_img.affine @ np.c_[xs, 0 * xs, 0 * xs, 1 + 0 * xs].T)[0]
        lsn = np.asarray(L_img.dataobj[..., labels["SNc"]], dtype=np.float32)
        left_frac = float(lsn[wx < 0].sum() / max(lsn.sum(), 1e-9))
        self.LEFT_OK = left_frac > 0.95
        log(f"prob_atlas SNc mass on the RAS-left side: {left_frac:.3f} -> left-hemisphere maps {'used' if self.LEFT_OK else 'NOT used'}")
        if self.LEFT_OK:
            CHANNELS += [("SNc_left", self.F_LEFT, labels["SNc"]), ("SNr_left", self.F_LEFT, labels["SNr"])]
        log("channels written to SPIR space: " + str([c[0] for c in CHANNELS]))
        self.labels, self.CHANNELS = labels, CHANNELS
        self.T_img, self.P_img, self.L_img = T_img, P_img, L_img

    def _prepare_template(self):
        ants, T_img = self.ants, self.T_img
        # self-test: the nibabel -> ANTs conversion equals ANTs' own reader for a T1w (qform = sform there)
        r0 = self.subj.iloc[0]
        a_own = ants_from_nib(ants, nib.load(str(self.root / r0.t1w_path)).get_fdata(dtype=np.float32), nib.load(str(self.root / r0.t1w_path)).affine)
        a_ref = ants.image_read(str(self.root / r0.t1w_path))
        assert np.allclose(a_own.origin, a_ref.origin, atol=1e-3) and np.allclose(a_own.direction, a_ref.direction, atol=1e-5) \
               and np.allclose(a_own.spacing, a_ref.spacing, atol=1e-5), "nibabel->ANTs geometry conversion disagrees with ANTs reader"
        log("geometry conversion self-test passed")
        sp = float(self.acfg["template_reg_spacing_mm"])
        self.TEMPLATE = ants_from_nib(ants, T_img.get_fdata(dtype=np.float32), T_img.affine)
        self.TEMPLATE_REG = ants.resample_image(self.TEMPLATE, (sp,) * 3, use_voxels=False, interp_type=0)
        self.ATLAS_CH = {}
        for name, f, idx in self.CHANNELS:
            img = self.P_img if f == self.F_PROB else self.L_img
            self.ATLAS_CH[name] = ants_from_nib(ants, np.asarray(img.dataobj[..., idx], dtype=np.float32), img.affine)
        VOX_ATLAS = float(np.prod(T_img.header.get_zooms()[:3]))
        self.ATLAS_MASS = {k: float(v.numpy().sum() * VOX_ATLAS) for k, v in self.ATLAS_CH.items()}   # mm3 of probability
        # upper-head centroid of the template (every 2nd voxel is enough for a centroid)
        _tsub = np.asarray(T_img.dataobj, dtype=np.float32)[::2, ::2, ::2]
        _, self.TEMPLATE_HEAD_CENTROID, _t_top = upper_head(_tsub, T_img.affine @ np.diag([2.0, 2.0, 2.0, 1.0]), self.acfg["head_depth_mm"])
        del _tsub
        # template voxel-index images (+1, so 0 means "outside the template"); carried into SPIR space they give, for every
        # SPIR voxel, its position in the CIT168 template. Any template-space region can later be mapped without re-registering.
        _ii, _jj, _kk = np.indices(T_img.shape[:3], dtype=np.float32)
        self.IDX_IMGS = [ants_from_nib(ants, a + 1.0, T_img.affine) for a in (_ii, _jj, _kk)]
        del _ii, _jj, _kk
        # an all-ones template image; after warping, values < 1 mark SPIR voxels whose interpolation mixed in "outside template"
        self.INSIDE_IMG = ants_from_nib(ants, np.ones(T_img.shape[:3], np.float32), T_img.affine)

    # ---------------------------------------------------------------------------------------------------- per subject
    def spir_ref(self, r):
        im = nib.load(str(self.root / r.spir_path))
        aff = im.get_sform()           # SPIR world space = sform
        arr = im.get_fdata(dtype=np.float32)
        return ants_from_nib(self.ants, arr, aff), aff, arr

    def rigid_to_ants(self, sid, base=None):
        # ITK Euler transform (SPIR point -> T1w point, LPS mm) -> ANTs affine file; verified on test points
        ants, sitk = self.ants, self.sitk
        t = sitk.ReadTransform(str(self.p_rigid / "transforms" / f"{sid}_t1w_to_spir_rigid.tfm")).Downcast()
        if isinstance(t, sitk.CompositeTransform):
            t = t.GetNthTransform(0).Downcast()
        M = np.array(t.GetMatrix()).reshape(3, 3)
        c = np.array(t.GetCenter()); tr = np.array(t.GetTranslation())
        # stored with centre 0 and the full offset, so nothing depends on how the centre is written to file
        atx = ants.create_ants_transform(transform_type="AffineTransform", precision="float", dimension=3,
                                         matrix=M, center=np.zeros(3), translation=c + tr - M @ c)
        out = os.path.join(str(base or self.out), "transforms", f"{sid}_spir2t1w_rigid_ants.mat")
        ants.write_transform(atx, out)
        back = ants.read_transform(out)                     # test the FILE, as apply_transforms will read it
        pts = [c + d for d in ([0, 0, 0], [80, 0, 0], [0, 80, 0], [0, 0, 30], [-70, 60, -25])]
        err = max(np.linalg.norm(np.subtract(t.TransformPoint(tuple(p)), ants.apply_ants_transform_to_point(back, tuple(p)))) for p in pts)
        assert err < 1e-2, f"{sid}: rigid conversion error {err} mm"
        return out

    def slab_corr_from_files(self, sid, base=None):
        tm = nib.load(os.path.join(str(base or self.out), "template_in_spir", f"{sid}_cit168_t1w_in_spir.nii.gz")).get_fdata(dtype=np.float32)
        ts = nib.load(str(self.p_rigid / "t1w_in_spir" / f"{sid}_t1w_in_spir.nii.gz")).get_fdata(dtype=np.float32)
        return slab_corr(tm, ts)

    def process_subject(self, r, overwrite=False, keep_warp=False, strategy="default", base=None):
        """strategy: 'default' or 'head_init' (retry). base: output folder (default the stage folder)."""
        ants, acfg = self.ants, self.acfg
        assert strategy in STRATEGIES, strategy
        base = str(base or self.out)
        for d in SUBDIRS:
            os.makedirs(os.path.join(base, d), exist_ok=True)
        sid = r.subject_id
        f_atlas = os.path.join(base, "atlas_in_spir", f"{sid}_cit168_in_spir.nii.gz")
        f_row = os.path.join(base, "per_subject", f"{sid}.json")
        if not overwrite and os.path.exists(f_atlas) and os.path.exists(f_row):
            return json.load(open(f_row))
        free = shutil.disk_usage(str(self.out)).free / 1e9
        if free < acfg["min_free_gb"]:
            raise RuntimeError(f"only {free:.1f} GB free in the output folder")
        t0 = time.time()
        t1n = nib.load(str(self.root / r.t1w_path))
        t1arr = t1n.get_fdata(dtype=np.float32)
        t1 = ants_from_nib(ants, t1arr, t1n.affine)
        ref, aff, spir = self.spir_ref(r)
        rigid = self.rigid_to_ants(sid, base)
        tdir = os.path.join(base, "transforms")
        init_note, head_top = "ANTsPy default start (centre-of-mass alignment), whole image", None
        depth = acfg["head_depth_mm"]
        with tempfile.TemporaryDirectory() as td:
            kw = {}
            if strategy != "default":
                hm, c_subj, head_top = upper_head(t1arr, t1n.affine, depth)
                mask_img = ants_from_nib(ants, hm.astype(np.float32), t1n.affine)
                kw.update(mask=mask_img, mask_all_stages=True)
                f_init = os.path.join(td, "init.mat")
                if strategy == "head_init":
                    # ANTs initial transform maps fixed (subject) points to moving (template) points: p_template = p_subject + t
                    t_lps = np.diag([-1.0, -1.0, 1.0]) @ (self.TEMPLATE_HEAD_CENTROID - c_subj)
                    ants.write_transform(ants.create_ants_transform(transform_type="AffineTransform", precision="float", dimension=3,
                                                                    matrix=np.eye(3), center=np.zeros(3), translation=t_lps), f_init)
                    init_note = f"translation aligning upper-head centroids ({depth:.0f} mm from vertex); metric masked to that head region"
                else:
                    g = (4.0, 4.0, 4.0)
                    f_init = ants.affine_initializer(ants.resample_image(t1, g, False, 0), ants.resample_image(self.TEMPLATE_REG, g, False, 0),
                                                     search_factor=20, radian_fraction=0.1, use_principal_axis=False,
                                                     local_search_iterations=10, mask=ants.resample_image(mask_img, g, False, 1), txfn=f_init)
                    init_note = f"ants.affine_initializer at 4 mm (search_factor 20, radian_fraction 0.1); metric masked to upper head ({depth:.0f} mm)"
                kw["initial_transform"] = f_init
            reg = ants.registration(fixed=t1, moving=self.TEMPLATE_REG, type_of_transform=acfg["reg_type"],
                                    outprefix=os.path.join(td, "r_"), verbose=False, **kw)
            fwd_tmp = list(reg["fwdtransforms"])
            # ORDER (tested on real data): the SPIR->T1w rigid must be FIRST in the list, followed by ANTs' own [warp, affine]
            # (with a start-up transform ANTs folds it into its affine, so the list has the same form)
            chain_tmp = [rigid] + fwd_tmp
            maps = []
            for name, _, _ in self.CHANNELS:
                w = ants.apply_transforms(fixed=ref, moving=self.ATLAS_CH[name], transformlist=chain_tmp, interpolator="linear").numpy()
                maps.append(np.clip(w, 0, 1).astype(np.float32))
            maps = np.stack(maps, axis=-1)
            tmpl = ants.apply_transforms(fixed=ref, moving=self.TEMPLATE, transformlist=chain_tmp, interpolator="linear").numpy().astype(np.float32)
            idx = np.stack([ants.apply_transforms(fixed=ref, moving=im, transformlist=chain_tmp, interpolator="linear").numpy()
                            for im in self.IDX_IMGS], axis=-1).astype(np.float32)
            inside = ants.apply_transforms(fixed=ref, moving=self.INSIDE_IMG, transformlist=chain_tmp, interpolator="linear").numpy()
            idx[inside < 0.999] = 0.0                     # "0 = outside" also holds at the template border
            # atlas mass in the subject's FULL T1w grid (whole head, no slab cut) = denominator of the in-slab fraction
            t1_vox = float(np.prod(vox_size(t1n.affine)))
            mass_t1w = {}
            for name in ("SNc", "SNr"):
                w1 = ants.apply_transforms(fixed=t1, moving=self.ATLAS_CH[name], transformlist=fwd_tmp, interpolator="linear").numpy()
                mass_t1w[name] = float(np.clip(w1, 0, 1).sum() * t1_vox)
            saved = []
            for i, f in enumerate(fwd_tmp):
                is_warp = f.endswith(".nii.gz")
                if is_warp and not (keep_warp or acfg["save_warps"]):
                    saved.append(None); continue
                dst = os.path.join(tdir, f"{sid}_cit2t1w_fwd{i}{'.nii.gz' if is_warp else os.path.splitext(f)[1]}")
                shutil.copy(f, dst); saved.append(dst)
        out_files = ((maps, f_atlas), (tmpl, os.path.join(base, "template_in_spir", f"{sid}_cit168_t1w_in_spir.nii.gz")),
                     (idx, os.path.join(base, "atlas_in_spir", f"{sid}_cit168_index_in_spir.nii.gz")))
        for arr, fpath in out_files:
            im = nib.Nifti1Image(arr, aff); im.set_qform(aff, 1); im.set_sform(aff, 1); nib.save(im, fpath)
        fwd = saved

        # ---- metrics (QC only) ----
        ch = {name: maps[..., i] for i, (name, _, _) in enumerate(self.CHANNELS)}
        zooms = vox_size(aff); vox = float(np.prod(zooms))
        sn = ch["SNc"]
        cen = np.array(ndimage.center_of_mass(sn)) if sn.sum() > 0 else np.full(3, np.nan)
        faces = (sn[0], sn[-1], sn[:, 0], sn[:, -1], sn[..., 0], sn[..., -1])
        t1s = nib.load(str(self.p_rigid / "t1w_in_spir" / f"{sid}_t1w_in_spir.nii.gz")).get_fdata(dtype=np.float32)
        row = dict(subject_id=sid, group=r.group, role=r.role, seconds=round(time.time() - t0, 1),
                   reg_strategy=strategy, ants_random_seed=ants.config._random_seed, head_reaches_fov_top=head_top,
                   snc_centroid_i=cen[0], snc_centroid_j=cen[1], snc_centroid_k=cen[2],
                   snc_in_slab_frac=float(sn.sum() * vox / mass_t1w["SNc"]) if mass_t1w["SNc"] > 0 else np.nan,
                   snr_in_slab_frac=float(ch["SNr"].sum() * vox / mass_t1w["SNr"]) if mass_t1w["SNr"] > 0 else np.nan,
                   snc_mass_t1w_mm3=mass_t1w["SNc"], snr_mass_t1w_mm3=mass_t1w["SNr"],
                   snc_mass_ratio_vs_template=float(sn.sum() * vox / self.ATLAS_MASS["SNc"]),
                   snc_touches_slab_edge=bool(any((f > acfg["slab_edge_p"]).any() for f in faces)),
                   snc_vol_p25_mm3=float((sn > 0.25).sum() * vox),
                   slab_corr_template_t1w=slab_corr(tmpl, t1s))
        union = (ch["SNc"] + ch["SNr"] + ch["RN"]) > 0.1
        if union.any():
            lo = np.maximum(np.argwhere(union).min(0) - np.ceil(15 / zooms).astype(int), 0)
            hi = np.minimum(np.argwhere(union).max(0) + np.ceil(15 / zooms).astype(int) + 1, spir.shape)
            box = tuple(slice(a, b) for a, b in zip(lo, hi))
            v = (tmpl[box] > 0) & (t1s[box] > 0)
            row["template_vs_t1w_corr_midbrain"] = float(np.corrcoef(tmpl[box][v], t1s[box][v])[0, 1]) if v.sum() > 100 else np.nan
            row["template_vs_t1w_nmi_midbrain"] = float(nmi(tmpl[box][v], t1s[box][v])) if v.sum() > 100 else np.nan
        if r.mask_path:
            m = nib.load(str(self.root / r.mask_path)).get_fdata(dtype=np.float32) != 0
            mc = np.array(ndimage.center_of_mass(m))
            atl = (ch["SNc"] + ch["SNr"]) >= 0.25
            wide = ndimage.binary_dilation((ch["SNc"] + ch["SNr"] + ch.get("PBP", 0) + ch.get("VTA", 0)) > 0.05, iterations=2)
            row.update(mask_centroid_k=mc[2], qc_centroid_dist_mm=float(np.linalg.norm((cen - mc) * zooms)),
                       qc_dice_mask_vs_atlas_sn=float(2 * (m & atl).sum() / (m.sum() + atl.sum())),
                       qc_mask_frac_inside_atlas_region=float((m & wide).sum() / m.sum()))
        json.dump({k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()}, open(f_row, "w"), indent=1, default=float)
        json.dump({"fwd_cit2t1w": [os.path.basename(f) if f else "not saved" for f in fwd], "spir2t1w_rigid": os.path.basename(rigid),
                   "chain_order": "[spir2t1w_rigid] + fwd_cit2t1w  (ants.apply_transforms(fixed=SPIR with sform, moving=CIT168 image))",
                   "index_map": f"{sid}_cit168_index_in_spir.nii.gz: CIT168 voxel index + 1 (i, j, k) for every SPIR voxel; "
                                "0 = outside template or touching its border",
                   "registration_strategy": strategy, "start_up": init_note},
                  open(os.path.join(base, "transforms", f"{sid}_chain.json"), "w"), indent=1)
        return row

    # ---------------------------------------------------------------------------------------------------- chain test
    def chain_test(self):
        """The subject with the LARGEST rigid head-motion correction is used, because a wrong transform order looks almost
        right when the rigid transform is close to identity. The second test subject is a mask-header-override control."""
        ants, sitk, out = self.ants, self.sitk, self.out
        F_CHAIN = out / "chain_test.json"
        if F_CHAIN.exists():
            ct = json.load(open(F_CHAIN))
            if ct.get("all_passed"):
                return ct
            F_CHAIN.unlink()                              # a stale failed result must never count as a pass
        rr = self.rr[~self.rr.subject_id.isin(self.exclude)]
        big = rr.sort_values("max_corner_disp_mm", ascending=False).subject_id.iloc[0]
        pilot_ids = list(dict.fromkeys([big] + self.override[:1]))
        thr = float(self.acfg["chain_test_threshold"])
        results = []
        for sid in pilot_ids:
            r = self.subj.loc[self.subj.subject_id == sid].iloc[0]
            row = self.process_subject(r, overwrite=True, keep_warp=True)
            # (a) one-step chain into SPIR vs (b) template -> T1w grid (ANTs), then T1w -> SPIR with the SimpleITK rigid
            chainf = json.load(open(out / "transforms" / f"{sid}_chain.json"))
            fwd = [str(out / "transforms" / f) for f in chainf["fwd_cit2t1w"]]
            t1n = nib.load(str(self.root / r.t1w_path))
            t1 = ants_from_nib(ants, t1n.get_fdata(dtype=np.float32), t1n.affine)
            in_t1 = ants.apply_transforms(fixed=t1, moving=self.TEMPLATE, transformlist=fwd, interpolator="linear").numpy()
            A = np.diag([-1.0, -1.0, 1.0]) @ t1n.affine[:3, :]
            sp = np.linalg.norm(A[:, :3], axis=0)
            s_in = sitk.GetImageFromArray(np.ascontiguousarray(in_t1.astype(np.float32).transpose(2, 1, 0)))
            s_in.SetSpacing(sp.tolist()); s_in.SetDirection((A[:, :3] / sp).flatten().tolist()); s_in.SetOrigin(A[:, 3].tolist())
            _, aff, spir = self.spir_ref(r)
            B = np.diag([-1.0, -1.0, 1.0]) @ aff[:3, :]
            spb = np.linalg.norm(B[:, :3], axis=0)
            s_ref = sitk.Image([int(x) for x in spir.shape], sitk.sitkFloat32)
            s_ref.SetSpacing(spb.tolist()); s_ref.SetDirection((B[:, :3] / spb).flatten().tolist()); s_ref.SetOrigin(B[:, 3].tolist())
            rig = sitk.ReadTransform(str(self.p_rigid / "transforms" / f"{sid}_t1w_to_spir_rigid.tfm"))
            two_step = sitk.GetArrayFromImage(sitk.Resample(s_in, s_ref, rig, sitk.sitkLinear, 0.0)).transpose(2, 1, 0)
            one_step = nib.load(str(out / "template_in_spir" / f"{sid}_cit168_t1w_in_spir.nii.gz")).get_fdata(dtype=np.float32)
            v = (one_step > 0) & (two_step > 0)
            c = float(np.corrcoef(one_step[v], two_step[v])[0, 1])
            ref, _, _ = self.spir_ref(r)
            wrong = ants.apply_transforms(fixed=ref, moving=self.TEMPLATE, transformlist=fwd + [str(out / "transforms" / chainf["spir2t1w_rigid"])],
                                          interpolator="linear").numpy()
            vw = (wrong > 0) & (two_step > 0)
            cw = float(np.corrcoef(wrong[vw], two_step[vw])[0, 1])
            disp = float(self.rr.loc[self.rr.subject_id == sid, "max_corner_disp_mm"].iloc[0])
            log(f"chain test {sid}: one-step vs two-step correlation = {c:.4f} -> {'PASS' if c > thr else 'FAIL'} "
                f"(reversed order gives {cw:.4f}; corner motion of this subject {disp:.1f} mm)")
            results.append(dict(subject_id=sid, corr_one_vs_two_step=c, corr_reversed_order=cw, max_corner_disp_mm=disp, passed=bool(c > thr)))
            if not self.acfg["save_warps"]:
                for f in fwd:
                    if f.endswith(".nii.gz") and os.path.exists(f):
                        os.remove(f)
            if c <= thr:
                break
        all_passed = len(results) == len(pilot_ids) and all(p["passed"] for p in results)
        ct = {"threshold": thr, "all_passed": all_passed, "subjects": results}
        json.dump(ct, open(F_CHAIN, "w"), indent=1)
        assert all_passed, "transform chain test failed: the atlas would land in the wrong place; see chain_test.json"
        return ct

    # ---------------------------------------------------------------------------------------------------- retry
    def _rows(self):
        d = self.out / "per_subject"
        return pd.DataFrame([json.load(open(d / f)) for f in sorted(os.listdir(d)) if f.endswith(".json")])

    def _files_of(self, sid, base):
        out = []
        for d in SUBDIRS:
            p = os.path.join(base, d)
            if os.path.isdir(p):
                out += [os.path.join(d, f) for f in os.listdir(p) if f.startswith(sid + "_") or f == sid + ".json"]
        return out

    def _copy_files(self, sid, src_base, dst_base):
        for rel in self._files_of(sid, src_base):
            os.makedirs(os.path.dirname(os.path.join(dst_base, rel)), exist_ok=True)
            shutil.copy2(os.path.join(src_base, rel), os.path.join(dst_base, rel))

    def retry(self):
        """Retries registrations that look wrong and keeps the best strategy per subject.
        Label-free rule, identical for development and test subjects (the masks decide nothing here):
          midbrain template-T1w correlation missing or < median - 3 robust SD, or SNc in slab < 0.90, or SNc at the slab edge.
        For each: run head_init and compare with the original by the template-T1w correlation over the whole SPIR slab;
        a retry replaces the original only if it is better by retry_keep_margin."""
        strategies = tuple(self.acfg["retry_strategies"])
        margin = float(self.acfg["retry_keep_margin"])
        OUT = str(self.out)
        _fin = lambda v: float(v) if v is not None and np.isfinite(float(v)) else -np.inf
        rows = self._rows()
        c = rows.template_vs_t1w_corr_midbrain.astype(float)
        med, mad = c.median(), (c - c.median()).abs().median()
        rule = c.isna() | (c < med - 3 * 1.4826 * mad) | (rows.snc_in_slab_frac.astype(float) < 0.9) | rows.snc_touches_slab_edge.astype(bool)
        _rd = rows["retry_done"] if "retry_done" in rows else pd.Series(None, index=rows.index)
        already = set(rows.subject_id[_rd.map(lambda v: v is True)])
        todo = [s for s in rows.loc[rule, "subject_id"] if s not in already]
        log(f"retry rule selects {int(rule.sum())} subjects; already retried: {sorted(already)}; to do now: {todo}")
        results = []
        for n, sid in enumerate(todo, 1):
            t_s = time.time()
            r = self.subj.loc[self.subj.subject_id == sid].iloc[0]
            rb = os.path.join(OUT, "retry", sid)
            dflt = os.path.join(rb, "default")
            if not os.path.isdir(os.path.join(dflt, "per_subject")):
                self._copy_files(sid, OUT, dflt)                         # keep the original result
            before = json.load(open(os.path.join(dflt, "per_subject", f"{sid}.json")))
            score = {"default": self.slab_corr_from_files(sid, dflt)}
            for s in strategies:
                sb = os.path.join(rb, s)
                f_row = os.path.join(sb, "per_subject", f"{sid}.json")
                if os.path.exists(f_row) and json.load(open(f_row)).get("reg_strategy") == s:
                    rw = json.load(open(f_row))                        # finished earlier (resume, no new registration)
                else:
                    rw = self.process_subject(r, overwrite=True, strategy=s, base=sb)
                score[s] = rw.get("slab_corr_template_t1w")
            best = max(strategies, key=lambda s: _fin(score[s]))
            chosen = best if _fin(score[best]) >= _fin(score["default"]) + margin else "default"
            src_dir = dflt if chosen == "default" else os.path.join(rb, chosen)
            for rel in self._files_of(sid, OUT):                         # replace this subject's files in the main folders
                os.remove(os.path.join(OUT, rel))
            self._copy_files(sid, src_dir, OUT)
            f_main = os.path.join(OUT, "per_subject", f"{sid}.json")
            row = json.load(open(f_main))
            row.update(reg_strategy=chosen, retry_done=True,
                       retry_slab_corr={k: (round(_fin(v), 4) if np.isfinite(_fin(v)) else None) for k, v in score.items()})
            if chosen == "default":
                row["slab_corr_template_t1w"] = score["default"] if np.isfinite(_fin(score["default"])) else None
            json.dump(row, open(f_main, "w"), indent=1, default=float)
            results.append(dict(subject_id=sid, chosen=chosen, **{f"slab_corr_{k}": score[k] for k in score},
                                midbrain_corr_before=before.get("template_vs_t1w_corr_midbrain"), midbrain_corr_after=row.get("template_vs_t1w_corr_midbrain"),
                                snc_in_slab_before=before.get("snc_in_slab_frac"), snc_in_slab_after=row.get("snc_in_slab_frac"),
                                snc_centroid_k_after=row.get("snc_centroid_k"),
                                qc_centroid_dist_before=before.get("qc_centroid_dist_mm"), qc_centroid_dist_after=row.get("qc_centroid_dist_mm")))
            log(f"{n}/{len(todo)} {sid:22s} slab corr " + ", ".join(f"{k} {_fin(v):.3f}" for k, v in score.items())
                + f" -> keep {chosen} | {(time.time() - t_s) / 60:.1f} min")
        f_res = os.path.join(OUT, "retry_results.csv")
        if results:
            new = pd.DataFrame(results)
            if os.path.exists(f_res):
                prev = pd.read_csv(f_res)
                new = pd.concat([prev[~prev.subject_id.isin(new.subject_id)], new], ignore_index=True)
            new.to_csv(f_res, index=False)
        if os.path.exists(f_res):
            log("retry results (mask distance columns are for information only; they decided nothing):\n"
                + pd.read_csv(f_res).round(3).to_string(index=False))

    # ---------------------------------------------------------------------------------------------------- summary + QC
    def summary(self):
        reg_table = self._rows()
        reg_table["reg_strategy"] = reg_table["reg_strategy"].map(lambda v: v if isinstance(v, str) else "default") if "reg_strategy" in reg_table else "default"
        reg_table["retry_done"] = reg_table["retry_done"].map(lambda v: v is True) if "retry_done" in reg_table else False
        c = reg_table.template_vs_t1w_corr_midbrain.astype(float)
        med, mad = c.median(), (c - c.median()).abs().median()
        reg_table["flag_low_similarity"] = c < med - 3 * 1.4826 * mad
        reg_table["flag_sn_partly_outside_slab"] = (reg_table.snc_in_slab_frac.astype(float) < 0.9) | reg_table.snc_touches_slab_edge.astype(bool)
        dev = reg_table[reg_table.role == "dev"].copy()
        if "qc_centroid_dist_mm" in dev:
            dev["flag_qc_far_from_mask"] = dev.qc_centroid_dist_mm > 5
        else:
            dev["flag_qc_far_from_mask"] = False
        flag_ids = sorted(set(reg_table.loc[reg_table.flag_low_similarity | reg_table.flag_sn_partly_outside_slab, "subject_id"]) |
                          set(dev.loc[dev.flag_qc_far_from_mask, "subject_id"]))
        _pk = dev[["snc_centroid_k", "mask_centroid_k"]].astype(float).dropna() if "mask_centroid_k" in dev else pd.DataFrame()
        r_k = float(np.corrcoef(_pk.snc_centroid_k, _pk.mask_centroid_k)[0, 1]) if len(_pk) > 2 else np.nan
        fr = reg_table.snc_in_slab_frac.astype(float)
        L = ["CIT168 -> T1w -> SPIR", "=" * 60,
             f"subjects done: {len(reg_table)} (excluded because the rigid registration was graded fail: {self.exclude})",
             f"template vs subject T1w correlation near the midbrain: median {med:.3f}, min {c.min():.3f} (flag rule: < median - 3 robust SD)",
             f"SNc probability mass inside the SPIR slab (relative to the subject's full T1w): median {fr.median():.2f}, "
             f"min {fr.min():.2f}, max {fr.max():.2f} (flag < 0.90; values near 1 expected)",
             f"SNc reaching the edge of the SPIR volume (p > {self.acfg['slab_edge_p']}): {int(reg_table.snc_touches_slab_edge.astype(bool).sum())} subjects (flagged)",
             f"SNc mass relative to the template (shows brain-size scaling, not used for flags): "
             f"median {reg_table.snc_mass_ratio_vs_template.astype(float).median():.2f}",
             f"atlas SNc volume at p>0.25 in SPIR space (mm3): median {reg_table.snc_vol_p25_mm3.median():.0f}",
             "", "LABEL-FREE CONSISTENCY: atlas SNc centroid in SPIR voxel indices, all subjects",
             f"  i (L-R): mean {reg_table.snc_centroid_i.mean():.1f}, sd {reg_table.snc_centroid_i.std():.1f}",
             f"  j (A-P): mean {reg_table.snc_centroid_j.mean():.1f}, sd {reg_table.snc_centroid_j.std():.1f}",
             f"  k (slice): min {reg_table.snc_centroid_k.min():.1f}, max {reg_table.snc_centroid_k.max():.1f}, sd {reg_table.snc_centroid_k.std():.1f}"]
        if len(dev) and "qc_centroid_dist_mm" in dev:
            L += ["", "QC AGAINST THE MANUAL MASKS (development subjects only; used for checking registration, never for fitting)",
                  f"  correlation of atlas SNc slice position with mask slice position: r = {r_k:.3f} (n = {len(_pk)})",
                  f"  centroid distance atlas SNc vs mask (mm): median {dev.qc_centroid_dist_mm.median():.2f}, max {dev.qc_centroid_dist_mm.max():.2f} (flag > 5 mm)",
                  f"  Dice mask vs atlas SNc+SNr (p>=0.25): median {dev.qc_dice_mask_vs_atlas_sn.median():.3f}",
                  f"  fraction of mask inside the dilated atlas SN/PBP/VTA region: median {dev.qc_mask_frac_inside_atlas_region.median():.3f}, "
                  f"min {dev.qc_mask_frac_inside_atlas_region.min():.3f}"]
        L += ["", f"retried: {int(reg_table.retry_done.sum())} subjects; strategy in use: {reg_table.reg_strategy.value_counts().to_dict()}",
              "", f"flagged subjects ({len(flag_ids)}): {flag_ids}"]
        if flag_ids:
            cols = ["subject_id", "reg_strategy", "template_vs_t1w_corr_midbrain", "snc_in_slab_frac", "snc_touches_slab_edge", "snc_centroid_k"]
            t = reg_table[reg_table.subject_id.isin(flag_ids)][cols]
            if "qc_centroid_dist_mm" in dev:
                t = t.merge(dev[["subject_id", "qc_centroid_dist_mm"]], how="left")
            L.append(t.round(3).to_string(index=False))
        write_text(self.out / "atlas_registration_summary.txt", "\n".join(L))
        log("\n".join(L))
        self.reg_table, self.flag_ids = reg_table, flag_ids
        return reg_table

    def qc_pages(self):
        """SPIR around the atlas SNc level, atlas SNc p=0.5 (green), SNr p=0.5 (yellow), binary mask (red, development only),
        and a checkerboard of the warped template vs the subject T1w at the centre level."""
        acfg, reg_table, out = self.acfg, self.reg_table, self.out
        OFFS, HALF, PER_PAGE = acfg["qc_offsets"], int(acfg["qc_half_window_vox"]), int(acfg["qc_per_page"])
        names = [c[0] for c in self.CHANNELS]

        def draw_pages(ids, prefix, what):
            pages = [ids[i:i + PER_PAGE] for i in range(0, len(ids), PER_PAGE)]
            for pn, page in enumerate(pages, 1):
                fig, axs = plt.subplots(len(page), len(OFFS) + 1, figsize=(2.0 * (len(OFFS) + 1), 2.0 * len(page)))
                axs = np.atleast_2d(axs)
                for ri, sid in enumerate(page):
                    r = self.subj.loc[self.subj.subject_id == sid].iloc[0]
                    q = reg_table.loc[reg_table.subject_id == sid].iloc[0]
                    spir = nib.load(str(self.root / r.spir_path)).get_fdata(dtype=np.float32)
                    mp = nib.load(str(out / "atlas_in_spir" / f"{sid}_cit168_in_spir.nii.gz")).get_fdata(dtype=np.float32)
                    snc, snr = mp[..., names.index("SNc")], mp[..., names.index("SNr")]
                    m = (nib.load(str(self.root / r.mask_path)).get_fdata(dtype=np.float32) != 0) if r.mask_path else None
                    cvals = [q.snc_centroid_i, q.snc_centroid_j, q.snc_centroid_k]
                    no_atlas = any(v is None or not np.isfinite(float(v)) for v in cvals)
                    if no_atlas:
                        ci, cj, ck = spir.shape[0] // 2, spir.shape[1] // 2, spir.shape[2] // 2
                    else:
                        ci, cj, ck = [int(round(float(x))) for x in cvals]
                    ck = int(np.clip(ck, 0, spir.shape[2] - 1))
                    xs = slice(max(0, ci - HALF), min(spir.shape[0], ci + HALF)); ys = slice(max(0, cj - HALF), min(spir.shape[1], cj + HALF))
                    crop = spir[xs, ys, max(0, ck - 4):ck + 5]
                    lo, hi = np.percentile(crop[crop > 0], [1, 99.5]) if (crop > 0).any() else (0, 1)
                    label = sid + ("" if q.reg_strategy == "default" else f" [{q.reg_strategy}]") + (" ATLAS SNc OUTSIDE SLAB" if no_atlas else "")
                    for c_i, o in enumerate(OFFS):
                        ax = axs[ri, c_i]; ax.set_xticks([]); ax.set_yticks([])
                        k = ck + o
                        if 0 <= k < spir.shape[2]:
                            ax.imshow(spir[xs, ys, k].T, cmap="gray", origin="lower", vmin=lo, vmax=hi)
                            for arr, col in ((snc, "lime"), (snr, "yellow")):
                                if (arr[xs, ys, k] > 0.5).any():
                                    ax.contour(arr[xs, ys, k].T, levels=[0.5], colors=col, linewidths=0.7)
                            if m is not None and m[xs, ys, k].any():
                                ax.contour(m[xs, ys, k].T.astype(np.uint8), levels=[0.5], colors="red", linewidths=0.7)
                        else:
                            ax.set_facecolor("black"); ax.text(0.5, 0.5, "slice outside volume", color="w", fontsize=6,
                                                               ha="center", transform=ax.transAxes)
                        if ri == 0:
                            ax.set_title(f"k = centre {o:+d}", fontsize=7)
                        if c_i == 0:
                            ax.text(0.03, 0.95, label, transform=ax.transAxes, color="red" if no_atlas else "cyan", fontsize=6, va="top")
                    ax = axs[ri, -1]; ax.set_xticks([]); ax.set_yticks([])
                    tm = nib.load(str(out / "template_in_spir" / f"{sid}_cit168_t1w_in_spir.nii.gz")).get_fdata(dtype=np.float32)[xs, ys, ck]
                    ts = nib.load(str(self.p_rigid / "t1w_in_spir" / f"{sid}_t1w_in_spir.nii.gz")).get_fdata(dtype=np.float32)[xs, ys, ck]
                    nrm = lambda a: (a - np.percentile(a, 2)) / (np.percentile(a, 98) - np.percentile(a, 2) + 1e-6)
                    yy, xx = np.indices(tm.shape)
                    ax.imshow(np.where(((yy // 15) + (xx // 15)) % 2 == 0, nrm(tm), nrm(ts)).T, cmap="gray", origin="lower", vmin=0, vmax=1)
                    if ri == 0:
                        ax.set_title("template | T1w checker", fontsize=7)
                fig.suptitle(f"atlas {what} page {pn}/{len(pages)}: green = atlas SNc p=0.5, yellow = atlas SNr p=0.5, red = binary mask; "
                             "last column: edges should continue across the squares", fontsize=8, y=0.995)
                plt.tight_layout(rect=(0, 0, 1, 0.985))
                fig.savefig(out / "figs" / f"{prefix}{pn}.png", dpi=100)
                plt.close(fig)

        order = list(self.subj.subject_id)
        all_ids = sorted(reg_table.subject_id, key=order.index)
        review_ids = sorted(set(self.flag_ids) | set(reg_table.subject_id[reg_table.retry_done]), key=order.index)
        draw_pages(all_ids, "qc_atlas_page", "all subjects")
        if review_ids:
            draw_pages(review_ids, "qc_atlas_review_page", "REVIEW (flagged or retried)")

    def grades_and_report(self):
        acfg, reg_table = self.acfg, self.reg_table
        viewed, exc = bool(acfg["viewed_qc_pages"]), dict(acfg["grade_exceptions"])
        # a grade is written for every subject that is still flagged, plus notes worth keeping; text before ":" is the grade
        reg_table["grade"] = reg_table.subject_id.map(lambda s: exc.get(s, "pass" if viewed else "not graded"))
        reg_table.to_csv(self.out / "atlas_registration.csv", index=False)
        flagged_now = sorted(reg_table.loc[reg_table.flag_low_similarity | reg_table.flag_sn_partly_outside_slab, "subject_id"])
        ungraded_flags = [s for s in flagged_now if s not in exc]
        ct_f = self.out / "chain_test.json"
        _ct = json.load(open(ct_f)) if ct_f.exists() else {}
        decisions = {
            "antspyx": self.ants.__version__,
            "atlas": "CIT168 subcortical atlas, native 700 um space; template CIT168_T1w_head_700um",
            "atlas_files": [self.F_TEMPLATE.name, self.F_PROB.name, self.F_LEFT.name, self.F_LABELS.name],
            "labels_used": {c[0]: c[2] for c in self.CHANNELS}, "left_hemisphere_maps": self.LEFT_OK,
            "registration": f"ants.registration(fixed=subject T1w 1 mm, moving=CIT168 T1w head resampled to {acfg['template_reg_spacing_mm']} mm, "
                            f"type_of_transform='{acfg['reg_type']}'), {acfg['n_threads']} threads; seed {self.seed}",
            "chain": "ants.apply_transforms(fixed=SPIR grid (sform), moving=CIT168 image, transformlist=[SPIR->T1w rigid, cit2t1w warp, "
                     "cit2t1w affine]), linear; order verified by the chain test; with a start-up transform (head_init) ANTs folds it "
                     "into the affine, so the chain is the same",
            "index_map": "per subject *_cit168_index_in_spir.nii.gz: CIT168 voxel index + 1 for every SPIR voxel (0 = outside)",
            "warps_saved": acfg["save_warps"],
            "outputs": {"atlas_in_spir": "4D float32, channels in the order of 'channels'", "channels": [c[0] for c in self.CHANNELS],
                        "template_in_spir": "CIT168 T1w head template in SPIR grid (QC)"},
            "use_of_masks": "QC only (centroid distance, Dice, coverage). Nothing is fitted on masks and no retry decision used them.",
            "reproducibility_note": "ANTs runs multi-threaded with random sampling; results can differ slightly between runs even with a fixed seed",
            "retry": {"rule": "label-free: midbrain template-T1w correlation missing or < median - 3 robust SD, or SNc in slab < 0.90, "
                              "or SNc at slab edge (same rule for development and test subjects)",
                      "method": f"head_init = translation aligning the upper-head centroids of template and subject, similarity metric limited "
                                f"to the head from the vertex down to {acfg['head_depth_mm']:.0f} mm, all stages; the result with the higher "
                                f"template-T1w correlation over the SPIR slab is kept (head_init must be better by {acfg['retry_keep_margin']})",
                      "results_file": "retry_results.csv; originals kept in retry/<subject>/default/",
                      "strategy_in_use": reg_table.reg_strategy.value_counts().to_dict()},
            "atlas_vs_nm_contrast_note": "CIT168 boundaries are not neuromelanin-based: atlas SNr lies largely outside the NM-bright region "
                                         "and the SNc p=0.5 core is smaller than it. The atlas is used for localisation, cropping, midline "
                                         "and limits, not as the SN segmentation.",
            "grades": {"viewed_pages": viewed, "exceptions": exc},
            "chain_test": _ct if _ct else "not run",
        }
        json.dump(decisions, open(self.out / "atlas_registration_decisions.json", "w"), indent=2, default=str)
        checks = {
            "all included subjects processed": len(reg_table) == len(self.subj) - len(self.exclude),
            "transform chain test passed": bool(_ct.get("all_passed")),
            "every flagged subject has a written grade": not ungraded_flags,
            "SN inside the slab for all subjects": not reg_table.flag_sn_partly_outside_slab.any(),
            "every automatically flagged subject was retried": bool(reg_table.loc[reg_table.flag_low_similarity | reg_table.flag_sn_partly_outside_slab, "retry_done"].all()),
            "QC pages viewed and graded": viewed,
            "no subject graded fail": not any(str(v).lower().startswith("fail") for v in exc.values()),
        }
        L = ["ATLAS REGISTRATION STATUS", "=" * 60] + [f"{'PASS' if ok else 'OPEN/FAIL'}: {k}" for k, ok in checks.items()]
        if ungraded_flags:
            L.append(f"   flagged but without a written grade (look at the review pages): {ungraded_flags}")
        L += ["", f"flagged now: {flagged_now}", f"strategy in use: {reg_table.reg_strategy.value_counts().to_dict()}",
              f"grades: {reg_table.grade.astype(str).str.split(':').str[0].value_counts().to_dict()}"]
        write_text(self.out / "atlas_registration_report.txt", "\n".join(L))
        log("\n".join(L))


def run(ctx):
    A = AtlasRegistration(ctx)
    if ctx.cfg["atlas_registration"]["run_chain_test"]:
        A.chain_test()
    todo = [r for r in A.subj.itertuples() if r.subject_id not in A.exclude]
    t_start = time.time()
    for n, r in enumerate(todo, 1):
        row = A.process_subject(r)
        log(f"{n:2d}/{len(todo)} {r.subject_id:22s} {row['seconds']:6.1f}s | SNc in slab {row['snc_in_slab_frac'] if row['snc_in_slab_frac'] is not None else float('nan'):.2f} | "
            f"corr {row.get('template_vs_t1w_corr_midbrain') or float('nan'):.3f} | elapsed {(time.time() - t_start) / 60:.0f} min")
    A.retry()
    A.summary()
    A.qc_pages()
    A.grades_and_report()
