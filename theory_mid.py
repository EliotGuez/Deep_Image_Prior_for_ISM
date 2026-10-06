from __future__ import annotations

import numpy as np
import torch
from s2ism import s2ism as s2

def _to_numpy(x): # convert torch to numpy
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)

def center_crop_hw(x, size): # crop in the center
    if size is None or size <= 0:
        return x
    h, w = x.shape[:2]
    if size > min(h, w):
        raise ValueError(f"crop_size={size} exceeds data size {h}x{w}")
    y0, x0 = (h - size) // 2, (w - size) // 2
    return x[y0:y0 + size, x0:x0 + size]

def extract_mid_image(res, mid_iter, z_idx, n_planes, spatial_shape):
    r = np.asarray(res)

    if r.ndim == 4:
        r = r[-1 if mid_iter is None else mid_iter]

    if r.ndim == 3:
        plane_axis = None
        for axis in range(3):
            other = tuple(s for i, s in enumerate(r.shape) if i != axis)
            if r.shape[axis] == n_planes and other == tuple(spatial_shape):
                plane_axis = axis
                break
        if plane_axis is None:
            raise ValueError(f"Cannot identify the MID plane axis in shape {r.shape}")
        r = np.take(r, z_idx, axis=plane_axis)

    if r.shape != tuple(spatial_shape):
        raise ValueError(f"MID image has shape {r.shape}, expected {spatial_shape}")

    return r.astype(np.float32)


def load_pth_dataset(path_data):
    bundle = torch.load(path_data, map_location="cpu", weights_only=False)

    meas = np.squeeze(_to_numpy(bundle["measurment"]).astype(np.float32))
    if meas.shape[0] == 25:
        dset = np.moveaxis(meas, 0, -1)
    elif meas.shape[-1] == 25:
        dset = meas
    else:
        raise ValueError(f"Cannot find 25 detector channels in measurement shape {meas.shape}")

    psf = _to_numpy(bundle["PSF"]).astype(np.float32)
    if psf.ndim != 4 or psf.shape[0] != 25:
        raise ValueError(f"Expected PSF shape (25, Nz, Kh, Kw), got {psf.shape}")
    psf_ism = np.moveaxis(psf, 0, -1)  # (Nz, Kh, Kw, 25)

    meta = bundle.get("metadati")
    if meta is None or not hasattr(meta, "dx"):
        raise ValueError("The .pth file must contain metadati.dx")

    print("[pth] measurement:", dset.shape)
    print("[pth] PSF stack:", psf_ism.shape)
    print("[pth] pxsizex:", float(meta.dx) * 1e3, "nm")

    return dset, psf_ism, meta

# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def run_theory_and_mid(path_data,out_npz,crop_size=512,num_iter=15,z_in_idx=1,mid_planes="all"):
    dset, psf_ism, meta = load_pth_dataset(path_data)
    pxsizex_nm = float(meta.dx) * 1e3

    if not 0 <= z_in_idx < psf_ism.shape[0]:
        raise ValueError(f"z_in_idx={z_in_idx}, but the PSF stack has {psf_ism.shape[0]} plane(s)")

    dset = center_crop_hw(dset, crop_size)

    if mid_planes == "all":
        psf_mid = psf_ism
        mid_z_idx = z_in_idx
    else:
        psf_mid = psf_ism[z_in_idx:z_in_idx + 1]
        mid_z_idx = 0

    print(f"[MID] data={dset.shape}, PSFs={psf_mid.shape}, iterations={num_iter}")
    
    res, *_ = s2.max_likelihood_reconstruction(
        dset,
        psf_mid,
        stop="fixed",
        max_iter=num_iter,
        rep_to_save="all",
    )
    res = _to_numpy(res).astype(np.float32)
    print("[MID] result:", res.shape)

    np.savez_compressed(out_npz, dset=dset, psf_ism=psf_ism, mid_all=res, 
        pxsizex_nm=pxsizex_nm, z_in_idx=int(z_in_idx), mid_z_idx=int(mid_z_idx),
        mid_planes=mid_planes, mid_n_planes=int(psf_mid.shape[0]))
        
    print(f"[theory] saved {out_npz}")
