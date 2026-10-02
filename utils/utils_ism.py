from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

import ISM.simulation.PSF_sim as ism
import ISM.simulation.detector as ism_detector
from torchvision.transforms.functional import rotate as tv_rotate


EPS = 1e-12
FIXED_MIRRORING = -1
FIXED_ROTATION_DEG = -76.30

def center_crop(x, size):
    h, w = x.shape[-2:]
    if size >= h and size >= w:
        return x
    y0 = max((h - size) // 2, 0)
    x0 = max((w - size) // 2, 0)
    return x[..., y0:y0 + min(size, h), x0:x0 + min(size, w)]


def load_real_data(path, z_idx, crop_size, device):

    data = torch.load(path, map_location="cpu", weights_only=False)
    meas = data["measurment"].float()
    if meas.dim() == 3:
        meas = meas.unsqueeze(1)
    original_shape = tuple(meas.shape)
    meas = center_crop(meas, crop_size)

    psf = data["PSF"].float()
    if psf.dim() == 3:
        psf = psf.unsqueeze(1)
    if not (0 <= z_idx < psf.shape[1]):
        raise ValueError(f"psf_z_idx={z_idx}, but PSF has {psf.shape[1]} plane(s).")
    ref_psf = psf[:, z_idx:z_idx + 1]
    fingerprint = ref_psf.flatten(1).sum(1).view(-1, 1, 1, 1)
    ref_psf_norm = ref_psf / (fingerprint + EPS)

    meta = data.get("metadati", None)
    if meta is None or not hasattr(meta, "dx"):
        raise ValueError("metadati.dx is required to build the real detector grid.")
    pxsizex_nm = float(meta.dx) * 1e3

    return {
        "meas": meas.to(device),
        "ref_psf": ref_psf_norm.to(device),
        "metadata": meta,
        "pxsizex_nm": pxsizex_nm,
        "original_shape": original_shape,
        "optics": {k: data.get(k, None) for k in ("exwl", "emwl", "na")},
    }


def barycentres_xy(stack):
    if stack.dim() == 4:
        stack = stack[:, 0]
    stack = stack.detach().float()
    _, h, w = stack.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=stack.device, dtype=stack.dtype),
        torch.arange(w, device=stack.device, dtype=stack.dtype),
        indexing="ij",
    )
    mass = stack.sum(dim=(-2, -1)).clamp_min(EPS)
    bx = (stack * xx).sum(dim=(-2, -1)) / mass - (w - 1) / 2
    by = (stack * yy).sum(dim=(-2, -1)) / mass - (h - 1) / 2
    return torch.stack((bx, by), dim=1).cpu().numpy()


def make_detector_masks(grid, mirroring=1, rotation_deg=0.0, device="cpu"):
    n = int(grid.N)
    coords = ism_detector.det_coords(grid.N, grid.geometry)
    coords = coords * grid.pxpitch
    coords = torch.round(coords / grid.M / grid.pxsizex).to(torch.int)

    detector = ism_detector.pinhole_array(coords, int(grid.Nx), grid.M, grid.pxsizex, grid.pxdim, grid.pinhole_shape, torch.device(device))

    if int(mirroring) == -1:
        if isinstance(n, tuple):
            nx, ny = n
        else:
            nx = ny = n
        nch = detector.shape[-1]
        detector = detector.reshape(int(grid.Nx), int(grid.Nx), nx, ny)
        detector = torch.flip(detector, dims=(-1,))
        detector = detector.reshape(int(grid.Nx), int(grid.Nx), nch)

    if abs(float(rotation_deg)) > 1e-12:
        detector = torch.movedim(detector, -1, 0)
        detector = tv_rotate(detector, float(rotation_deg))
        detector = torch.movedim(detector, 0, -1)

    detector = detector.clone()
    detector[detector < 1e-2] = 0
    return detector


def similarity_score(p, q):
    """Fit q ~= scale*p + translation; return RMSE, scale, R^2."""
    p0 = p - p.mean(0, keepdims=True)
    q0 = q - q.mean(0, keepdims=True)
    denom = float(np.sum(p0 * p0))
    if denom <= 0:
        return float("inf"), 0.0, -float("inf")
    scale = float(np.sum(p0 * q0) / denom)
    if scale <= 0:
        return float("inf"), scale, -float("inf")
    pred = scale * p0 + q.mean(0, keepdims=True)
    resid = q - pred
    rmse = float(np.sqrt(np.mean(np.sum(resid * resid, axis=1))))
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum(q0 * q0))
    r2 = 1.0 - ss_res / max(ss_tot, EPS)
    return rmse, scale, r2



def pinhole_centres(grid, mirroring, rotation_deg):
    pinholes = make_detector_masks(grid, mirroring, rotation_deg, device="cpu")
    return barycentres_xy(pinholes.permute(2, 0, 1))


def evaluate_fixed_orientation(grid, ref_psf):
    target = barycentres_xy(ref_psf.cpu())
    centres = pinhole_centres(grid, FIXED_MIRRORING, FIXED_ROTATION_DEG)
    rmse, scale, r2 = similarity_score(centres, target)
    return {
        "rotation_deg": float(FIXED_ROTATION_DEG),
        "rotation_rad": float(math.radians(FIXED_ROTATION_DEG)),
        "mirroring": int(FIXED_MIRRORING),
        "scale": float(scale),
        "rmse_px": float(rmse),
        "r2": float(r2),
    }


def build_real_grid(kh, pxsizex_nm, ref_psf):
    grid = ism.GridParameters()
    grid.N = 5
    grid.Nx = kh
    grid.Nz = 1
    grid.pxsizex = float(pxsizex_nm)
    grid.mirroring = int(FIXED_MIRRORING)
    grid.rotation = float(math.radians(FIXED_ROTATION_DEG))
    calibration = evaluate_fixed_orientation(grid, ref_psf)
    return grid, calibration


def scalar(value):
    if torch.is_tensor(value):
        return value.detach().cpu().item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def print_inspection(bundle, grid, calibration):
    meas, ref = bundle["meas"], bundle["ref_psf"]
    print("\n" + "=" * 72)
    print("REAL ISM INSPECTION")
    print("=" * 72)
    print(f"torch version        : {torch.__version__}")
    print(f"torch path           : {torch.__file__}")
    print("detector transform   : local fixed geometry (ISM transform_detector not used)")
    print(f"original measurement : {bundle['original_shape']}")
    print(f"cropped measurement  : {tuple(meas.shape)}")
    print(f"measurement range    : [{meas.min().item():.4g}, {meas.max().item():.4g}]")
    print(f"reference PSF        : {tuple(ref.shape)}")
    print(f"optical metadata     : {bundle['optics']}")
    print(f"pxsizex              : {bundle['pxsizex_nm']:.6f} nm (metadati.dx x 1000)")
    print("\nDetector grid:")
    for key in ("N", "Nx", "Nz", "pxpitch", "pxdim", "M", "geometry", "pinhole_shape"):
        print(f"  {key:14s}: {scalar(getattr(grid, key))}")
    pitch_px = float(scalar(grid.pxpitch)) / float(scalar(grid.M)) / float(scalar(grid.pxsizex))
    print(f"  pitch on PSF : {pitch_px:.6f} px")
    print("\nFixed detector orientation (sanity-checked against stored PSFs):")
    print(f"  mirroring    : {calibration['mirroring']}")
    print(f"  rotation     : {calibration['rotation_deg']:+.2f} deg ({calibration['rotation_rad']:+.4f} rad)")
    print(f"  scale        : {calibration['scale']:.4f}")
    print(f"  RMSE         : {calibration['rmse_px']:.4f} px")
    print(f"  R^2          : {calibration['r2']:.4f}")
    print("=" * 72 + "\n")


# -----------------------------------------------------------------------------
# Saving helpers
# -----------------------------------------------------------------------------
def save_image(x, path):
    arr = x.detach().cpu().squeeze().float().numpy()
    arr = np.clip(arr, 0, None)
    arr = arr / (arr.max() + 1e-12)
    plt.figure(figsize=(6, 6))
    plt.imshow(arr, cmap="gray")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_stack(stack, path, title):
    arr = stack.detach().cpu().squeeze(1).numpy()
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    vmax = float(arr.max())
    for i, ax in enumerate(axes.flat):
        ax.imshow(arr[i], cmap="hot", vmin=0, vmax=vmax)
        ax.set_title(str(i), fontsize=8)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_two_psfs(h_exc, h_em, path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(h_exc.detach().cpu().squeeze(), cmap="hot")
    axes[0].set_title("Excitation PSF")
    axes[1].imshow(h_em.detach().cpu().squeeze(), cmap="hot")
    axes[1].set_title("Emission PSF")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def gaussian_kernel(kh, kw, sigma, device):
    yy = torch.arange(kh, device=device) - (kh - 1) / 2
    xx = torch.arange(kw, device=device) - (kw - 1) / 2
    Y, X = torch.meshgrid(yy, xx, indexing="ij")
    g = torch.exp(-(X * X + Y * Y) / (2 * sigma * sigma))
    return g / g.sum()