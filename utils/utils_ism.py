from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
from skimage.io import imsave

import deepinv as dinv
from deepinv.physics import Denoising, PoissonNoise

import ISM.simulation.PSF_sim as ism
import ISM.simulation.generate_ism_phantom as gen


def _make_grid(Kh: int = 99, pxpitch=75e3, pxsizex=30):
    grid          = ism.GridParameters()
    grid.N        = 5
    grid.Nx       = Kh
    grid.pxsizex  = 30       # nm/px  taille d'un pixel dans l'espace objet plus c'est petit  plus la psf est résolue dans l'image 
    grid.pxdim    = 50e3     # nm
    grid.pxpitch  = pxpitch    # nm  controlent la taille effective  du pinhole pour chaque détecteur donc pinhole grand fait plus de lumière, élargit la psf de détection
    grid.M        = 500    # Change la taille apparente du pinhole dans l'espace objet 
    grid.Nz       = 1
    grid.pxsizez  = 700      # nm
    return grid


def _make_optical_params(wl, zer_idx=None, zer_ampli=None):
    par            = ism.simSettings()
    par.n          = 1.5   # NA = n · sin(α), NA max = n
    par.na         = 1.2   # plus na est grand plus la psf est petite et piqué : r ~\lambda / NA
    par.wl         = wl    # \lambda grand donne psf plus large  donc psf d'émission est plus large que psf d'excitation
    par.mask_sampl = 31
    if zer_idx is not None:
        par.abe_index = zer_idx
    if zer_ampli is not None:
        par.abe_ampli = zer_ampli
    return par

def generate_25_psf_zernike(Kh=99, zer_coeff_ex=None, zer_coeff_em=None, coeff_scale=0.3, seed=None, ZERNIKE_MAX=15,ZERNIKE_FIXED=1):

    ZERNIKE_FREE = ZERNIKE_MAX - ZERNIKE_FIXED
    rng = np.random.default_rng(seed)

    if zer_coeff_ex is None:
        zer_coeff_ex = torch.tensor(rng.uniform(-coeff_scale, coeff_scale, size=ZERNIKE_FREE), dtype=torch.float32)
    if zer_coeff_em is None:
        zer_coeff_em = torch.tensor(rng.uniform(-coeff_scale, coeff_scale, size=ZERNIKE_FREE), dtype=torch.float32)

    gt_zer_ex = torch.cat([torch.zeros(ZERNIKE_FIXED), zer_coeff_ex])
    gt_zer_em = torch.cat([torch.zeros(ZERNIKE_FIXED), zer_coeff_em])
    zernike_idx = torch.arange(ZERNIKE_MAX)

    grid  = _make_grid(Kh)
    exPar = _make_optical_params(450, zernike_idx, gt_zer_ex)
    emPar = _make_optical_params(660, zernike_idx, gt_zer_em)
    emPar.n  = exPar.n
    emPar.na = exPar.na
    emPar.mask_sampl = exPar.mask_sampl

    PSF, detPSF, exPSF = ism.SPAD_PSF_2D(grid, exPar, emPar)
    return PSF, gt_zer_ex, gt_zer_em, exPSF, detPSF, grid


def reconstruct_single_psf(zer_coeff, wl, Kh=99, device="cpu", dtype=torch.float32):

    grid = _make_grid(Kh)
    zernike_idx = torch.arange(len(zer_coeff))
    par = _make_optical_params(wl, zernike_idx, zer_coeff)

    psf, _ = ism.singlePSF(par, grid.pxsizex, grid.Nx, [0, 0], grid.Nz, device)
    psf = psf.squeeze()
    psf = psf / (psf.sum() + 1e-12)
    return psf.to(device=device, dtype=dtype)


def reconstruct_psf_from_zernike(zer_ex,zer_em, Kh=99, device="cpu", dtype=torch.float32):

    grid = _make_grid(Kh)
    zernike_idx = torch.arange(len(zer_ex))
    exP = _make_optical_params(450, zernike_idx, zer_ex)
    emP = _make_optical_params(660, zernike_idx, zer_em)
    emP.n  = exP.n
    emP.na  = exP.na
    emP.mask_sampl = exP.mask_sampl

    PSF_rec, _, _ = ism.SPAD_PSF_2D(grid, exP, emP)
    return PSF_rec.to(device=device, dtype=dtype)


def generate_phantom_data(phantom_type, Nx, Ny, Nz, device, pxsize=40, flux=40):
    image = gen.generate_phantom(phantom_type, Nx, Ny, Nz, pxsizex=pxsize)
    ground_truth = flux * image.to(device)
    return ground_truth


def simulate_measurement(PSF, Nx, ground_truth, device, gain=1.0, back_lvl=0.01):
    physics_blur  = dinv.physics.BlurFFT(img_size=(1, 1, Nx, Nx), filter=PSF, device=device)
    back = torch.tensor(back_lvl, device=device)
    blurred_image = physics_blur(ground_truth.repeat(25, 1, 1, 1)) + back.repeat(25, 1, 1, 1)

    denoiser = Denoising()
    denoiser.noise_model  = PoissonNoise(gain=gain)
    noise_image = denoiser(blurred_image)

    return noise_image, blurred_image, physics_blur



class PoissonNLL(nn.Module):
    """
    Poisson negative log-likelihood:  sum( lam/gain - (y/gain)*log(lam) )
    """
    def __init__(self, eps=1e-8, gain=1.0):
        super().__init__()
        self.eps         = eps
        self.gain        = gain

    def forward(self, x, y, physics):
        lam = physics(x).clamp_min(self.eps)
        loss_map = lam / self.gain - (y / self.gain) * torch.log(lam)
        return loss_map.sum()



def psnr_torch(x: torch.Tensor, y: torch.Tensor,
               data_range=None, eps: float = 1e-12) -> torch.Tensor:
    mse        = torch.mean((x - y) ** 2)
    if data_range is None:
        data_range = torch.max(y) - torch.min(y)
    data_range = torch.clamp(torch.as_tensor(data_range, dtype=torch.float32), min=eps)
    return 10 * torch.log10((data_range ** 2) / torch.clamp(mse, min=eps))


def compute_grouped_kernel_mse(pred_k: torch.Tensor, gt_k: torch.Tensor) -> dict:
    groups = {
        "outer":  [0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24],
        "inner":  [6, 7, 8, 11, 13, 16, 17, 18],
        "center": [12],
    }
    return {f"psf_mse_{name}": torch.mean((pred_k[idx] - gt_k[idx]) ** 2).item() for name, idx in groups.items()}


def save_tensor_image(x: torch.Tensor, path) -> None:
    """Save any tensor that squeezes to (H, W) as a uint8 PNG."""
    x = x.detach().cpu().squeeze().numpy()
    x = np.clip(x, 0, None)
    x = x / (x.max() + 1e-8)
    imsave(str(path), (255 * x).astype(np.uint8))


def save_kernels_grid(kernels: torch.Tensor, path, nrow: int = 5) -> None:
    """Save (25, 1, Kh, Kw) kernel stack as a 5×5 grid image."""
    k    = kernels.detach().cpu().squeeze(1).numpy()   # (25, Kh, Kw)
    D    = k.shape[0]
    ncol = nrow
    nrow_plot = int(np.ceil(D / ncol))

    fig, axes = plt.subplots(nrow_plot, ncol, figsize=(10, 10))
    axes      = np.array(axes).reshape(nrow_plot, ncol)
    vmin, vmax = k.min(), k.max()

    for i in range(nrow_plot * ncol):
        ax = axes.flat[i]
        ax.axis("off")
        if i < D:
            ax.imshow(k[i], cmap="hot", vmin=vmin, vmax=vmax)
            ax.set_title(f"k{i}", fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_stack_grid(x: torch.Tensor, path, nrow: int = 5, cmap: str = "hot") -> None:
    """Save a (25, 1, H, W) observation stack as a 5×5 grid image."""
    arr  = x.detach().cpu().squeeze(1).numpy()
    D    = arr.shape[0]
    ncol = nrow
    nrow_plot = int(np.ceil(D / ncol))

    fig, axes = plt.subplots(nrow_plot, ncol, figsize=(10, 10))
    axes      = np.array(axes).reshape(nrow_plot, ncol)
    vmin, vmax = arr.min(), arr.max()

    for i in range(nrow_plot * ncol):
        ax = axes.flat[i]
        ax.axis("off")
        if i < D:
            ax.imshow(arr[i], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{i}", fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_two_psfs(h_exc: torch.Tensor, h_em: torch.Tensor, path) -> None:
    """Save excitation and emission PSFs side by side."""
    def _prep(h):
        h = h.detach().cpu()
        while h.dim() > 2:
            h = h.squeeze(0)
        return h.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(_prep(h_exc), cmap="hot")
    axes[0].set_title("Excitation PSF")
    axes[0].axis("off")
    axes[1].imshow(_prep(h_em), cmap="hot")
    axes[1].set_title("Emission PSF")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Saving — curves & Zernike plots
# ---------------------------------------------------------------------------

def save_curves(metric_dict: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, val in metric_dict.items():
        if len(val) > 0:
            ax.plot(val, label=key)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Metric")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def save_zernike_comparison(
    gt_zer_ex, gt_zer_em,
    fit_zer_ex, fit_zer_em,
    path,
) -> None:
    """Side-by-side bar chart: GT vs fitted Zernike coefficients."""
    gt_zer_ex  = torch.as_tensor(gt_zer_ex).detach().cpu().numpy()
    gt_zer_em  = torch.as_tensor(gt_zer_em).detach().cpu().numpy()
    fit_zer_ex = torch.as_tensor(fit_zer_ex).detach().cpu().numpy()
    fit_zer_em = torch.as_tensor(fit_zer_em).detach().cpu().numpy()

    x     = np.arange(len(gt_zer_ex))
    width = 0.4
    labels = [f"Z{i}" for i in x]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, gt, fit, title in zip(
        axes,
        [gt_zer_ex, gt_zer_em],
        [fit_zer_ex, fit_zer_em],
        ["Excitation", "Emission"],
    ):
        ax.bar(x - width / 2, gt,  width, label="Ground Truth", color="steelblue",  alpha=0.8)
        ax.bar(x + width / 2, fit, width, label="Fitted",        color="darkorange", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(f"Zernike Coefficients — {title}")
        ax.set_ylabel("Amplitude (rad)")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.legend()
        ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def save_zernike_gt(
    gt_zer_ex,
    gt_zer_em,
    path,
):
    gt_zer_ex = torch.as_tensor(gt_zer_ex).detach().cpu().numpy()
    gt_zer_em = torch.as_tensor(gt_zer_em).detach().cpu().numpy()

    x = np.arange(len(gt_zer_ex))
    labels = [f"Z{i}" for i in x]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, gt, title in zip(
        axes,
        [gt_zer_ex, gt_zer_em],
        ["Excitation GT", "Emission GT"],
    ):
        ax.bar(x, gt, color="steelblue", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(f"Ground Truth Zernike Coefficients — {title}")
        ax.set_ylabel("Amplitude (rad)")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def save_zernike_history(history: torch.Tensor, path) -> None:
    """Plot multi-start Zernike optimisation loss curves."""
    history = torch.as_tensor(history).detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(history.shape[0]):
        ax.semilogy(history[i], label=f"start {i}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitting loss")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def gaussian_kernel2d(size: tuple, sigma: float) -> torch.Tensor:
    """Return a normalised 2-D Gaussian kernel of shape (H, W)."""
    H, W = size
    assert H % 2 == 1 and W % 2 == 1, "Kernel size must be odd."
    ax = torch.arange(-(W // 2), W // 2 + 1).view(1, W).float()
    ay = torch.arange(-(H // 2), H // 2 + 1).view(H, 1).float()
    kernel = torch.exp(-(ax ** 2 + ay ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()

def gaussian_beads(grid_size, num_beads, sigma, flux=1.0, device="cpu", dtype=torch.float32, seed=None):

    rng = np.random.default_rng(seed)
    coords = rng.integers(0, grid_size, size=(num_beads, 2))  # (N, 2) in (row, col)

    ax = torch.arange(grid_size, dtype=torch.float32)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")  # (H, W) each

    image = torch.zeros(grid_size, grid_size, dtype=torch.float32)
    for row, col in coords:
        image += torch.exp(-((xx - col) ** 2 + (yy - row) ** 2) / (2.0 * sigma ** 2))

    image = image / (image.max() + 1e-8)          # normalise to [0, 1]
    image = (image * flux).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    return image.to(device=device, dtype=dtype)

