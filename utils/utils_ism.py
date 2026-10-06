from __future__ import annotations

import math

import numpy as np
import torch
from torchvision.transforms.functional import rotate as tv_rotate

try:
    import ISM.simulation.PSF_sim as psf_sim
    import ISM.simulation.detector as ism_detector
except ModuleNotFoundError:
    import brighteyes_ism.simulation.PSF_sim as psf_sim
    import brighteyes_ism.simulation.detector as ism_detector


from s2ism.psf_estimator import GridFinder

FIXED_MIRRORING = -1
FIXED_ROTATION_DEG = -76.30
FIXED_M = 450.0


def build_real_grid(kh, pxsizex_nm, magnification=None, rotation_deg=None,
    mirroring=None, estimate_grid=False, dset=None, exwl=639.0, emwl=665.0, na=1.49):
    
    grid = psf_sim.GridParameters()
    grid.N = 5
    grid.Nx = int(kh)
    grid.Nz = 1
    grid.pxsizex = float(pxsizex_nm)

    # Fixed defaults
    grid.M = FIXED_M
    grid.rotation = math.radians(FIXED_ROTATION_DEG)
    grid.mirroring = FIXED_MIRRORING

    if estimate_grid:
        if dset is None:
            raise ValueError("dset is required when estimate_grid=True")

        grid_est = GridFinder(grid)
        grid_est.estimate(np.asarray(dset), float(exwl), float(emwl), float(na))

        print(f"[GridFinder] M={grid_est.M:.2f}, rotation={np.degrees(grid_est.rotation):.2f} deg, mirroring={grid_est.mirroring}")

        grid.M = float(grid_est.M)
        grid.rotation = float(grid_est.rotation)
        grid.mirroring = int(grid_est.mirroring)

    if magnification is not None:
        grid.M = float(magnification)
    if rotation_deg is not None:
        grid.rotation = math.radians(float(rotation_deg))
    if mirroring is not None:
        grid.mirroring = int(mirroring)

    return grid

def make_detector_masks(grid, device="cpu"):
    coords = ism_detector.det_coords(grid.N, grid.geometry)
    coords = torch.as_tensor(coords, dtype=torch.float32)
    coords = torch.round(coords * grid.pxpitch / grid.M / grid.pxsizex).long()

    detector = ism_detector.pinhole_array(coords, int(grid.Nx), grid.M,
        grid.pxsizex, grid.pxdim, grid.pinhole_shape, torch.device(device))

    if grid.mirroring == -1:
        detector = detector.reshape(grid.Nx, grid.Nx, grid.N, grid.N)
        detector = detector.flip(-1)
        detector = detector.reshape(grid.Nx, grid.Nx, grid.N ** 2)

    rotation_deg = math.degrees(grid.rotation)
    if rotation_deg != 0:
        detector = detector.movedim(-1, 0)
        detector = tv_rotate(detector, rotation_deg)
        detector = detector.movedim(0, -1)

    detector[detector < 1e-2] = 0
    return detector.to(device=device, dtype=torch.float32)
    
    
def barycentres_xy(stack):
    if stack.ndim == 4:
        stack = stack[:, 0]

    stack = stack.detach().float()
    _, h, w = stack.shape

    yy, xx = torch.meshgrid(
        torch.arange(h, device=stack.device, dtype=stack.dtype),
        torch.arange(w, device=stack.device, dtype=stack.dtype),
        indexing="ij",
    )

    mass = stack.sum((-2, -1)).clamp_min(1e-12)
    bx = (stack * xx).sum((-2, -1)) / mass - (w - 1) / 2
    by = (stack * yy).sum((-2, -1)) / mass - (h - 1) / 2

    return torch.stack((bx, by), dim=1).cpu().numpy()

def orientation_diagnostic(grid, pinholes, ref_psf):
    """Compare detector-mask centroids with the stored theoretical PSFs."""
    p = barycentres_xy(pinholes.permute(2, 0, 1))
    q = barycentres_xy(ref_psf)

    p -= p.mean(0)
    q -= q.mean(0)

    scale = np.sum(p * q) / max(np.sum(p * p), 1e-12)
    residual = q - scale * p
    rmse = np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))
    r2 = 1 - np.sum(residual ** 2) / max(np.sum(q ** 2), 1e-12)

    return {
        "M": float(grid.M),
        "rotation_deg": float(math.degrees(grid.rotation)),
        "mirroring": int(grid.mirroring),
        "scale": float(scale),
        "rmse_px": float(rmse),
        "r2": float(r2),
    }

    
    
# def build_real_grid( kh, pxsizex_nm, magnification=None, rotation_deg=None,
#     mirroring=None, estimate_grid=False, dset=None, exwl=639.0, emwl=665.0, na=1.49):    
    # grid = ism.GridParameters()
    # grid.N = 5
    # grid.Nx = kh
    # grid.Nz = 1
    # grid.pxsizex = float(pxsizex_nm)
    # grid_est = GridFinder(grid)
    # grid_est.estimate(np.asarray(dset), float(exwl), float(emwl), float(na))
    # print(
    #     "[GridFinder] "
    #     f"M={grid_est.M}, "
    #     f"rotation={np.degrees(grid_est.rotation):.2f} deg, "
    #     f"mirroring={grid_est.mirroring}, "
    #     f"shift={grid_est.shift}"
    # )
    # grid.mirroring = int(FIXED_MIRRORING)
    # grid.rotation = float(math.radians(FIXED_ROTATION_DEG))
    # # grid.M = float(grid_est.M) # the M estimated is 511
    # grid.M = 450.0

    # calibration = evaluate_fixed_orientation(grid, ref_psf)
    # calibration["M"] = float(grid.M)

    # calibration["gridfinder_rotation_deg"] = float(np.degrees(grid_est.rotation))
    # calibration["gridfinder_mirroring"] = int(grid_est.mirroring)
    # expected = exwl / (exwl + emwl)
    # if abs(calibration["scale"] - expected) > 0.1:
    #     print(f"[WARNING] calibration scale {calibration['scale']:.3f} vs expected ~{expected:.3f}: "
    #           "M or the geometry is probably wrong.")
    # return grid, calibration


# def center_crop(x, size):
#     h, w = x.shape[-2:]
#     if size >= h and size >= w:
#         return x
#     y0 = max((h - size) // 2, 0)
#     x0 = max((w - size) // 2, 0)
#     return x[..., y0:y0 + min(size, h), x0:x0 + min(size, w)]


# def load_real_data(path, z_idx, crop_size, device):

#     data = torch.load(path, map_location="cpu", weights_only=False)
#     meas = data["measurment"].float()
#     if meas.dim() == 3:
#         meas = meas.unsqueeze(1)
#     original_shape = tuple(meas.shape)
#     meas = center_crop(meas, crop_size)

#     psf = data["PSF"].float()
#     if psf.dim() == 3:
#         psf = psf.unsqueeze(1)
#     if not (0 <= z_idx < psf.shape[1]):
#         raise ValueError(f"psf_z_idx={z_idx}, but PSF has {psf.shape[1]} plane(s).")
#     ref_psf = psf[:, z_idx:z_idx + 1]
#     fingerprint = ref_psf.flatten(1).sum(1).view(-1, 1, 1, 1)
#     ref_psf_norm = ref_psf / (fingerprint + EPS)

#     meta = data.get("metadati", None)
#     if meta is None or not hasattr(meta, "dx"):
#         raise ValueError("metadati.dx is required to build the real detector grid.")
#     pxsizex_nm = float(meta.dx) * 1e3

#     return {
#         "meas": meas.to(device),
#         "ref_psf": ref_psf_norm.to(device),
#         "metadata": meta,
#         "pxsizex_nm": pxsizex_nm,
#         "original_shape": original_shape,
#         "optics": {k: data.get(k, None) for k in ("exwl", "emwl", "na")},
#     }


# def barycentres_xy(stack):
#     if stack.dim() == 4:
#         stack = stack[:, 0]
#     stack = stack.detach().float()
#     _, h, w = stack.shape
#     yy, xx = torch.meshgrid(
#         torch.arange(h, device=stack.device, dtype=stack.dtype),
#         torch.arange(w, device=stack.device, dtype=stack.dtype),
#         indexing="ij",
#     )
#     mass = stack.sum(dim=(-2, -1)).clamp_min(EPS)
#     bx = (stack * xx).sum(dim=(-2, -1)) / mass - (w - 1) / 2
#     by = (stack * yy).sum(dim=(-2, -1)) / mass - (h - 1) / 2
#     return torch.stack((bx, by), dim=1).cpu().numpy()


# def make_detector_masks(grid, mirroring=1, rotation_deg=0.0, device="cpu"):
#     n = int(grid.N)
#     coords = ism_detector.det_coords(grid.N, grid.geometry)
#     coords = coords * grid.pxpitch
#     coords = torch.round(coords / grid.M / grid.pxsizex).to(torch.int)

#     detector = ism_detector.pinhole_array(coords, int(grid.Nx), grid.M, grid.pxsizex, grid.pxdim, grid.pinhole_shape, torch.device(device))

#     if int(mirroring) == -1:
#         if isinstance(n, tuple):
#             nx, ny = n
#         else:
#             nx = ny = n
#         nch = detector.shape[-1]
#         detector = detector.reshape(int(grid.Nx), int(grid.Nx), nx, ny)
#         detector = torch.flip(detector, dims=(-1,))
#         detector = detector.reshape(int(grid.Nx), int(grid.Nx), nch)

#     if abs(float(rotation_deg)) > 1e-12:
#         detector = torch.movedim(detector, -1, 0)
#         detector = tv_rotate(detector, float(rotation_deg))
#         detector = torch.movedim(detector, 0, -1)

#     detector = detector.clone()
#     detector[detector < 1e-2] = 0
#     return detector



# -----------------------------------------------------------------------------
# Saving helpers
# -----------------------------------------------------------------------------
# def save_image(x, path):
#     arr = x.detach().cpu().squeeze().float().numpy()
#     arr = np.clip(arr, 0, None)
#     arr = arr / (arr.max() + 1e-12)
#     plt.figure(figsize=(6, 6))
#     plt.imshow(arr, cmap="gray")
#     plt.axis("off")
#     plt.tight_layout(pad=0)
#     plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0)
#     plt.close()


# def save_stack(stack, path, title):
#     arr = stack.detach().cpu().squeeze(1).numpy()
#     fig, axes = plt.subplots(5, 5, figsize=(10, 10))
#     vmax = float(arr.max())
#     for i, ax in enumerate(axes.flat):
#         ax.imshow(arr[i], cmap="hot", vmin=0, vmax=vmax)
#         ax.set_title(str(i), fontsize=8)
#         ax.axis("off")
#     if title:
#         fig.suptitle(title)
#     plt.tight_layout()
#     plt.savefig(path, dpi=150, bbox_inches="tight")
#     plt.close(fig)


# def save_two_psfs(h_exc, h_em, path):
#     fig, axes = plt.subplots(1, 2, figsize=(8, 4))
#     axes[0].imshow(h_exc.detach().cpu().squeeze(), cmap="hot")
#     axes[0].set_title("Excitation PSF")
#     axes[1].imshow(h_em.detach().cpu().squeeze(), cmap="hot")
#     axes[1].set_title("Emission PSF")
#     for ax in axes:
#         ax.axis("off")
#     plt.tight_layout()
#     plt.savefig(path, dpi=150, bbox_inches="tight")
#     plt.close(fig)


def gaussian_kernel(kh, kw, sigma, device):
    yy = torch.arange(kh, device=device) - (kh - 1) / 2
    xx = torch.arange(kw, device=device) - (kw - 1) / 2
    Y, X = torch.meshgrid(yy, xx, indexing="ij")
    g = torch.exp(-(X * X + Y * Y) / (2 * sigma * sigma))
    return g / g.sum()