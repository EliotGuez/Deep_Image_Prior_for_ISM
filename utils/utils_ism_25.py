from __future__ import annotations
from pathlib import Path
import torch
import deepinv as dinv
from deepinv.physics import Denoising, GaussianNoise, PoissonNoise
from deepinv.optim.data_fidelity import PoissonLikelihood
from deepinv.utils.demo import load_url_image, get_image_url
from deepinv.utils.plotting import plot
from deepinv.loss.metric import SSIM, MSE, PSNR, LPIPS
from deepinv.optim.prior import Prior, PnP
from deepinv.optim import Distance, DataFidelity, Potential
from typing import Callable, TYPE_CHECKING
import torch
import torch.nn.functional as F
from torch import nn

from deepinv.optim.distance import (
    Distance,
    L2Distance,
    L1Distance,
    IndicatorL2Distance,
    AmplitudeLossDistance,
    PoissonLikelihoodDistance,
    LogPoissonLikelihoodDistance,
    ZeroDistance,
)

from deepinv.optim.potential import Potential
from deepinv.physics.functional import dct_2d, idct_2d
from deepinv.optim.optimizers import optim_builder
import ISM.simulation.PSF_sim as ism
import ISM.analysis.Graph_lib as gr
import ISM.simulation.generate_ism_phantom as gen

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave

def generate_25_psf(plot=False):
    grid = ism.GridParameters()
    grid.N = 5
    grid.Nx = 99
    pxsize =  30
    grid.pxsizex =   pxsize
    grid.pxdim = 50e3  
    # grid.pxpitch = 75e3 
    grid.pxpitch = 75e3 

    grid.M = 500
    grid.Nz = 1
    grid.pxsizez = 700

    exPar = ism.simSettings()
    exPar.n = 1.5
    exPar.na = 1.2 

    exPar.wl = 450

    exPar.mask_sampl = 31
    emPar = exPar.copy()
    emPar.wl = 660
    z_shift = 0

    PSF, detPSF, exPSF = ism.SPAD_PSF_2D(grid, exPar, emPar)
    if plot:
        fig = gr.ShowDataset(PSF.cpu(), normalize= False)
    return PSF 

def generate_phantom_data(phantom_type, Nx,Ny, Nz, device, pxsize=40,  flux=40, plot=False):
    image = gen.generate_phantom(phantom_type, Nx, Ny, Nz, pxsizex = pxsize)
    ground_truth = flux * image.to(device)
    if plot:    
        gr.ShowImg(ground_truth.to("cpu"), pxsize*1e-3)
    return ground_truth


def simulate_measurement(PSF, Nx, ground_truth, device, pxsize=40, plot=False, gain=1.0):
    physics_blurr = dinv.physics.BlurFFT(img_size = (1, 1, Nx, Nx), filter = PSF, device=device)
    back = torch.tensor(0.01, device = device)
    blurred_image = physics_blurr(ground_truth.repeat(25,1,1,1)) + back.repeat(25,1,1,1)
    physics_noise = Denoising()
    physics_noise.noise_model = PoissonNoise(gain = gain)
    noise_image = physics_noise(blurred_image)
    if plot:
        gr.ShowDataset(noise_image.cpu(), normalize= False)
        gr.ShowImg(noise_image.sum(0).to("cpu"), pxsize*1e-3)
    return noise_image, blurred_image, physics_blurr

def data_normalization(noise_image, PSF, device):
    '''PSF-Weighted Normalization: 
        1: Compute the Fingerprint: calculate total integrated energy of the theoretical psf for each detector j. 
           This serves as a proxy for the relative quantum efficiency and geometric collection probability of that element
        2: Re-weighting: We normalize the measurement y_j by its local maximum but immediately rescale it by the calculated PSF energy. 
    '''
    noise_image_norm = noise_image.clone()
    finger_print = torch.zeros(25, device = device)
    for j in range(25): 
        finger_print[j] = PSF[j:j+1].sum()
        max_y_i = torch.max(noise_image[j])
        noise_image_norm[j] = (noise_image_norm[j] / max_y_i) * finger_print[j]
    return noise_image_norm

class PoissonNLL(nn.Module):
    def __init__(self, eps: float = 1e-8, norm=False, fingerprint=None, gain=1.):
        super().__init__()
        self.eps = eps
        self.norm = norm
        self.fingerprint = fingerprint
        self.gain = gain
    def forward(self, x, y, physics, *args, **kwargs):
        lam = physics(x).clamp_min(self.eps)
        loss_map = lam - y * torch.log(lam)
        loss_map = lam / self.gain - (y / self.gain) * torch.log(lam) 
        # if self.norm:
        #     w = 1/ self.fingerprint
        #     w = w/ w.mean()
        #     loss_map = w * loss_map
        return loss_map.sum()
    
def save_tensor_image(x, path):
    """
    Save tensor image of shape (1,1,H,W) or (H,W).
    """
    x = x.detach().cpu().squeeze().numpy()
    x = np.clip(x, 0, None)
    x = x / (x.max() + 1e-8)
    imsave(str(path), (255 * x).astype(np.uint8))

def psnr_torch(x, y, data_range=None, eps=1e-12):
    mse = torch.mean((x - y) ** 2)
    if data_range is None:
        data_range = torch.max(y) - torch.min(y)
    data_range = torch.clamp(data_range, min=eps)
    return 10 * torch.log10((data_range ** 2) / torch.clamp(mse, min=eps))


def mse_torch(x, y):
    return torch.mean((x - y) ** 2).item()

def psnr_from_mse(mse, data_range, eps=1e-12):
    mse = max(mse, eps)
    data_range = max(float(data_range), eps)
    return 10.0 * np.log10((data_range ** 2) / mse)

def compute_psf_metrics(pred_k, gt_k):
    mse_list, psnr_list, rel_list = [], [], []
    for i in range(pred_k.shape[0]):
        mse_i = torch.mean((pred_k[i:i+1] - gt_k[i:i+1]) ** 2).item()
        rng_i = (gt_k[i:i+1].max() - gt_k[i:i+1].min()).item()
        psnr_i = psnr_from_mse(mse_i, rng_i if rng_i > 0 else 1.0)
        rel_i = (
            torch.sum((pred_k[i:i+1] - gt_k[i:i+1]) ** 2) /
            (torch.sum(gt_k[i:i+1] ** 2) + 1e-12)
        ).item()
        mse_list.append(mse_i)
        psnr_list.append(psnr_i)
        rel_list.append(rel_i)

    return {
        "psf_mse_mean": float(np.mean(mse_list)),
        "psf_psnr_mean": float(np.mean(psnr_list)),
        "psf_rel_mean": float(np.mean(rel_list)),
        "psf_mse_each": mse_list,
        "psf_psnr_each": psnr_list,
        "psf_rel_each": rel_list,
    }

def save_checkpoint(step, out_x, out_k, pred_y, save_root):
    ckpt_dir = save_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(out_x.detach().cpu(), ckpt_dir / f"x_{step:05d}.pt")
    torch.save(out_k.detach().cpu(), ckpt_dir / f"k_{step:05d}.pt")
    torch.save(pred_y.detach().cpu(), ckpt_dir / f"yhat_{step:05d}.pt")



def save_kernels_grid(kernels, path, nrow=5, figsize=(10, 10)):
    """
    Save kernels of shape (D,1,Kh,Kw) as a grid.
    """
    kernels = kernels.detach().cpu().squeeze(1).numpy()  # (D,Kh,Kw)
    D = kernels.shape[0]
    ncol = nrow
    nrow_plot = int(np.ceil(D / ncol))

    fig, axes = plt.subplots(nrow_plot, ncol, figsize=figsize)
    axes = np.array(axes).reshape(nrow_plot, ncol)

    vmin = kernels.min()
    vmax = kernels.max()

    for i in range(nrow_plot * ncol):
        ax = axes.flat[i]
        ax.axis("off")
        if i < D:
            ax.imshow(kernels[i], cmap="hot", vmin=vmin, vmax=vmax)
            ax.set_title(f"k{i}", fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_metrics(metric_dict, save_path):
    np.save(save_path, metric_dict)


def save_curves(metric_dict, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, val in metric_dict.items():
        if len(val) > 0:
            ax.plot(val, label=key)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Metric")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def save_stack_grid(x, path, nrow=5, figsize=(10,10), cmap="hot"):
    x = x.detach().cpu().squeeze(1).numpy()   # (25,H,W)
    D = x.shape[0]
    ncol = nrow
    nrow_plot = int(np.ceil(D / ncol))

    fig, axes = plt.subplots(nrow_plot, ncol, figsize=figsize)
    axes = np.array(axes).reshape(nrow_plot, ncol)

    vmin = x.min()
    vmax = x.max()

    for i in range(nrow_plot * ncol):
        ax = axes.flat[i]
        ax.axis("off")
        if i < D:
            ax.imshow(x[i], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{i}", fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)



def gaussian_kernel2d(size, sigma):
    """Create a 2D Gaussian kernel."""
    assert size[0] % 2 == 1 and size[1] % 2 == 1, "Kernel size must be odd."
    H,W = size
    ax = torch.arange(-(W//2), W//2+1).view(1, W)
    ay = torch.arange(-(H//2), H//2+1).view(H, 1)
    xx = ax.repeat(H, 1)
    yy = ay.repeat(1, W)
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel



def detector_shifts_5x5(pitch_px=2.0, device="cpu"):
    shifts = []
    for iy in range(5):
        for ix in range(5):
            dx = (ix - 2) * pitch_px
            dy = (iy - 2) * pitch_px
            shifts.append([dx, dy])

    return torch.tensor(shifts, device=device, dtype=torch.float32)


def save_two_psfs(h_exc, h_em, path):
    kernels = torch.cat([h_exc, h_em], dim=0)
    save_kernels_grid(kernels, path, nrow=2, figsize=(6, 3))



def save_best_psnr_summary(best_psnr_summary, save_path, filename="best_psnr_iteration_vs_gain.png"):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    gains = [d["gain"] for d in best_psnr_summary]
    best_iters = [d["best_iter"] for d in best_psnr_summary]
    best_psnrs = [d["best_psnr"] for d in best_psnr_summary]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(gains, best_iters, marker="o")
    ax.set_xlabel("Poisson gain")
    ax.set_ylabel("Iteration of best image PSNR")
    ax.set_title("Best PSNR iteration vs Poisson gain")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path / filename, dpi=150)
    plt.close(fig)

def inspect_real_data(path):

    data = torch.load(path, map_location="cpu", weights_only=False)
    print(f"\n{'='*50}")
    print(f"File: {path}")
    print(f"Keys: {list(data.keys())}")
    
    for k, v in data.items():
        print(f"\n  [{k}]")
        if hasattr(v, "shape"):
            print(f"    shape : {v.shape}")
            print(f"    dtype : {v.dtype}")
            if v.numel() > 0:
                arr = v.float()
                print(f"    min   : {arr.min().item():.4g}")
                print(f"    max   : {arr.max().item():.4g}")
                print(f"    mean  : {arr.mean().item():.4g}")
        elif isinstance(v, dict):
            print(f"    sub-keys: {list(v.keys())}")
        else:
            # Try to print all attributes of metadata objects
            print(f"    type  : {type(v)}")
            attrs = {a: getattr(v, a) for a in dir(v)
                     if not a.startswith("_") and not callable(getattr(v, a))}
            for attr, val in attrs.items():
                print(f"    .{attr} = {val}")
    print(f"{'='*50}\n")


def load_real_data(path, psf_z_idx=0, crop_size=256, device="cpu", dtype=torch.float32, norm_physics=False):
    data = torch.load(path, map_location="cpu", weights_only=False)

    meas = data["measurment"].to(dtype=dtype)           # (25, 1, H, W)  [note: typo in key]
    if meas.dim() == 3:
        meas = meas.unsqueeze(1)

    _, _, H, W = meas.shape
    if crop_size < H or crop_size < W:
        ch, cw = H // 2, W // 2
        h0, w0 = ch - crop_size // 2, cw - crop_size // 2
        meas = meas[:, :, h0:h0+crop_size, w0:w0+crop_size]
    Nx = meas.shape[-1]

    psf_raw = data["PSF"].to(dtype=dtype)               # (25, Nz, Kh, Kw)
    if psf_raw.dim() == 3:
        psf_raw = psf_raw.unsqueeze(1)
    psf_raw = psf_raw[:, psf_z_idx:psf_z_idx+1, :, :]  # (25, 1, Kh, Kw)

    fingerprint = psf_raw.view(25, -1).sum(dim=1).view(25, 1, 1, 1)   # (25,1,1,1)
    PSF = (psf_raw / (fingerprint + 1e-12)).to(device=device)
    fingerprint = fingerprint.to(device=device)
    meas        = meas.to(device=device)

    print(f"[load_real_data] measurement : {tuple(meas.shape)}, "f"range [{meas.min().item():.2g}, {meas.max().item():.2g}]")
    print(f"[load_real_data] PSF         : {tuple(PSF.shape)}, "f"z-plane {psf_z_idx}")
    print(f"[load_real_data] fingerprint : min={fingerprint.min().item():.4g}, "f"max={fingerprint.max().item():.4g}")

    return PSF, fingerprint, meas, Nx