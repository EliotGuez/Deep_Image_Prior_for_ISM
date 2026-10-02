from __future__ import annotations
import numpy as np
import torch
from s2ism import s2ism as s2

def _to_numpy(x): # convert torch to numpy
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def to_dset(data, n_ch=25): # return the dataset as (Nx, Ny, Ch)
    d = np.squeeze(_to_numpy(data))
    if d.ndim != 3 or d.shape[-1] != n_ch:
        raise ValueError(f"Expected the squeezed dataset to be (Nx, Ny, {n_ch}), got {d.shape}. ")
    return d.astype(np.float32)

def center_crop_hw(x, size): # crop in the center
    if size is None or size <= 0:
        return x
    h, w = x.shape[:2]
    if size > min(h, w):
        raise ValueError(f"crop_size={size} exceeds data size {h}x{w}")
    y0, x0 = (h - size) // 2, (w - size) // 2
    return x[y0:y0 + size, x0:x0 + size]


def extract_mid_image(res, mid_iter, z_idx, n_planes, spatial_shape): # print the MID image at the given iteration and z-plane
    r = np.asarray(res)
    if r.ndim == 4:
        r = r[-1 if mid_iter is None else mid_iter]
    if r.ndim == 3:
        plane_axis = None
        for a in range(3):
            rest = tuple(s for i, s in enumerate(r.shape) if i != a)
            if rest == tuple(spatial_shape) and r.shape[a] == n_planes:
                plane_axis = a
        if plane_axis is None:
            raise ValueError(f"Cannot locate the plane axis in MID result of shape {r.shape} ")
        r = np.take(r, z_idx, axis=plane_axis)
    if r.shape != tuple(spatial_shape):
        raise ValueError(f"MID image has shape {r.shape}, expected {spatial_shape}.")
    return r.astype(np.float32)

def load_pth_dataset(path_data: str):
    bundle = torch.load(path_data, map_location="cpu", weights_only=False)

    print("[pth] keys:", list(bundle.keys()))

    meas = np.squeeze(_to_numpy(bundle["measurment"]).astype(np.float32))
    if meas.ndim != 3:
        raise ValueError(f"Expected measurement with 3 dimensions after squeeze, got {meas.shape}")

    if meas.shape[0] == 25:        # (25, H, W) -> (H, W, 25)
        dset = np.moveaxis(meas, 0, -1)
    elif meas.shape[-1] == 25:         # already (H, W, 25)
        dset = meas
    else:
        raise ValueError(f"Cannot find 25 detector channels in measurement shape {meas.shape}")


    psf = _to_numpy(bundle["PSF"]).astype(np.float32)
    if psf.ndim != 4:
        raise ValueError(f"Expected PSF with 4 dimensions, got {psf.shape}")

    if psf.shape[0] == 25:        # (25, Nz, Kh, Kw) -> (Nz, Kh, Kw, 25)
        psf_ism = np.moveaxis(psf, 0, -1)        
    else:
        raise ValueError(f"Cannot find 25 detector channels in PSF shape {psf.shape}")


    meta = bundle.get("metadati", None)
    if meta is None or not hasattr(meta, "dx"):
        raise ValueError("The .pth file must contain metadati.dx")

    pxsizex_nm = float(meta.dx) * 1e3

    print("[pth] measurement:", dset.shape)
    print("[pth] PSF stack:", psf_ism.shape)
    print("[pth] pxsizex:", pxsizex_nm, "nm")

    return dset, psf_ism, meta

# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def run_theory_and_mid(path_data, out_npz, crop_size=256, na=1.49, exwl=639.0, emwl=665.0, mask_sampl=101, num_iter=15, z_in_idx=1, show_graph = False):
    dset, psf_ism, meta = load_pth_dataset(path_data)
    pxsizex_nm = float(meta.dx) * 1e3
    print(f"[theory] dset {dset.shape}, pxsizex = {pxsizex_nm:.3f} nm")

    if psf_ism.ndim != 4:
        raise ValueError(f"Expected psf_ism as (Nz, Kh, Kw, 25), got {psf_ism.shape}")

    if psf_ism.shape[-1] != 25:
        raise ValueError(f"Expected 25 detector channels, got {psf_ism.shape}")

    if not (0 <= z_in_idx < psf_ism.shape[0]):
        raise ValueError(f"z_in_idx={z_in_idx}, but stored PSF stack only has {psf_ism.shape[0]} axial plane(s)")

    print(f"[theory] dset {dset.shape}, psf_ism {psf_ism.shape}, pxsizex={pxsizex_nm:.3f} nm")


    # ---- MID on the SAME crop that the DIP methods will see ---------------
    dset_c = center_crop_hw(dset, crop_size)
    print(f"[MID] running {num_iter} iterations on {dset_c.shape}")
    res, *_ = s2.max_likelihood_reconstruction(dset_c, psf_ism, stop="fixed", max_iter=num_iter, rep_to_save="all")
    res = _to_numpy(res).astype(np.float32)
    print(f"[MID] result array shape = {res.shape}")

    np.savez_compressed(out_npz, dset=dset_c, psf_ism=psf_ism, mid_all=res, pxsizex_nm=pxsizex_nm, z_in_idx=int(z_in_idx),na=na, exwl=exwl, emwl=emwl) 
    print(f"[theory] saved {out_npz}")
