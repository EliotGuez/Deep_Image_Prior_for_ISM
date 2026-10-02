from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import deepinv as dinv

from networks.skip import skip
from networks.fcn import SingleKernelFCN
from utils.utils_ism import build_real_grid, make_detector_masks, gaussian_kernel
from utils.utils_train import PhysicsTwoPSF, PoissonNLL

EPS = 1e-12


@dataclass
class DIPConfig:
    num_iter: int = 1000
    lr_x: float = 1e-2
    lr_k: float = 1e-4
    smooth_k: float = 0.0
    reg_noise_std: float = 0.0
    seed: int = 0
    pretraining: bool = False
    pretraining_iter: int = 100
    pretraining_lr: float = 1e-4
    pret_psf: float = 4.0
    print_every: int = 200
    input_depth: int = 8
    latent_k: int = 200
    hidden_k: int = 400
    save_every: int = 100


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def hwc_to_stack(a, device): # (H, W, 25) numpy to (25, 1, H, W) torch.
    t = torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)
    return t.permute(2, 0, 1).unsqueeze(1).contiguous()

# def normalize_stack(k: torch.Tensor) -> torch.Tensor:
#     return k / (k.sum(dim=(-2, -1), keepdim=True) + EPS)


def normalize_psf_stack(k): #     Preserve relative detector sensitivity.    Average PSF integral across detectors becomes 1.
    k = k.clamp_min(0)
    mass = k.sum(dim=(-2, -1), keepdim=True)   # (25,1,1,1)
    return k / (mass.mean() + EPS)
    
def build_image_net(device, input_depth):
    return skip(
        input_depth, 1,
        num_channels_down=[128] * 5, num_channels_up=[128] * 5, num_channels_skip=[16] * 5,
        upsample_mode="bilinear", need_sigmoid=False, need_bias=True,
        pad="reflection", act_fun="LeakyReLU",
    ).to(device)

def gradient_smoothness(k):
    dx, dy = k[:, :, :, 1:] - k[:, :, :, :-1], k[:, :, 1:, :] - k[:, :, :-1, :]
    return (dx ** 2).mean() + (dy ** 2).mean()

# ---------------------------------------------------------------------------
# kernels functions
# ---------------------------------------------------------------------------
class PhysicsFixedKernels(nn.Module):
    def __init__(self, img_size, device, kernels):
        super().__init__()
        self.kernels = kernels
        self.blur = dinv.physics.BlurFFT(img_size=img_size, filter=kernels, device=device)

    def forward(self, x):
        return self.blur(x.repeat(self.kernels.shape[0], 1, 1, 1))


def _pretrain_two_psf(net_exc, net_em, z_exc, z_em, kh, kw, cfg, device):
    target = gaussian_kernel(kh, kw, cfg.pret_psf, device).view(1, 1, kh, kw)
    opt = torch.optim.Adam(list(net_exc.parameters()) + list(net_em.parameters()), lr=cfg.pretraining_lr)
    for _ in range(cfg.pretraining_iter):
        opt.zero_grad()
        e = net_exc(z_exc).view(1, 1, kh, kw)
        m = net_em(z_em).view(1, 1, kh, kw)
        loss = ((e - target) ** 2).sum() + ((m - target) ** 2).sum()
        loss.backward()
        opt.step()


def _np(t): # torch to numpy
    return t.detach().cpu().float().numpy()


# ---------------------------------------------------------------------------
# blind: image + 2 latent PSFs
# ---------------------------------------------------------------------------
def run_dip_blind(y, ref_psf, pxsizex_nm, cfg, device): # the ref_psf is used for orientation check only, not for training.
    set_seed(cfg.seed)
    _, _, h, w = y.shape
    kh, kw = ref_psf.shape[-2:]

    grid, calibration = build_real_grid(kh, pxsizex_nm, ref_psf)
    print(f"[blind] orientation check: {calibration}")
    pinholes = make_detector_masks(grid, calibration["mirroring"], calibration["rotation_deg"], device="cpu").to(device=device, dtype=torch.float32)

    physics = PhysicsTwoPSF((1, 1, h, w), device, pinholes)
    net_x = build_image_net(device, cfg.input_depth)
    z_x = torch.rand(1, cfg.input_depth, h, w, device=device)
    net_exc = SingleKernelFCN(cfg.latent_k, kh * kw, hidden=cfg.hidden_k).to(device)
    net_em = SingleKernelFCN(cfg.latent_k, kh * kw, hidden=cfg.hidden_k).to(device)
    z_exc = torch.rand(1, cfg.latent_k, device=device)
    z_em = torch.rand(1, cfg.latent_k, device=device)

    if cfg.pretraining:
        _pretrain_two_psf(net_exc, net_em, z_exc, z_em, kh, kw, cfg, device)

    optimizer = torch.optim.Adam([
        {"params": net_x.parameters(), "lr": cfg.lr_x},
        {"params": net_exc.parameters(), "lr": cfg.lr_k},
        {"params": net_em.parameters(), "lr": cfg.lr_k},
    ])
    poisson = PoissonNLL()
    
    losses, data_losses, smooth_losses, smooth_raw_losses = [], [], [], []
    snapshot_iters, x_history, h_exc_history, h_em_history, kernels_history = [], [], [], [], []
    gaussian_mse_history, theory_kernel_mse_history = [], []
    gaussian_target = gaussian_kernel(kh, kw, cfg.pret_psf, device).view(1, 1, kh, kw)

    def save_snapshot(iteration):
        with torch.no_grad():
            x_s = F.softplus(net_x(z_x))

            h_exc_s = net_exc(z_exc).view(1, 1, kh, kw)
            h_em_s = net_em(z_em).view(1, 1, kh, kw)

            physics.set_psfs(h_exc_s, h_em_s)
            kernels_s = physics.kernels

            gaussian_mse = (((h_exc_s - gaussian_target) ** 2).mean() + ((h_em_s - gaussian_target) ** 2).mean()).item()

            theory_mse = torch.mean((kernels_s - ref_psf) ** 2).item()

            snapshot_iters.append(iteration)
            x_history.append(_np(x_s)[0, 0])
            h_exc_history.append(_np(h_exc_s)[0, 0])
            h_em_history.append(_np(h_em_s)[0, 0])
            kernels_history.append(_np(kernels_s)[:, 0])

            gaussian_mse_history.append(gaussian_mse)
            theory_kernel_mse_history.append(theory_mse)
    
    save_snapshot(0)  # initial snapshot

    for step in range(cfg.num_iter):
        optimizer.zero_grad()
        out_x = F.softplus(net_x(z_x + cfg.reg_noise_std * torch.randn_like(z_x)))
        h_exc = net_exc(z_exc + cfg.reg_noise_std * torch.randn_like(z_exc)).view(1, 1, kh, kw)
        h_em = net_em(z_em + cfg.reg_noise_std * torch.randn_like(z_em)).view(1, 1, kh, kw)
        physics.set_psfs(h_exc, h_em)
        pred = physics(out_x)

        loss_data = poisson(pred, y)
        smooth_raw = (gradient_smoothness(h_exc) + gradient_smoothness(h_em))

        loss_smooth_k = cfg.smooth_k * smooth_raw
        loss = loss_data + loss_smooth_k

        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        data_losses.append(float(loss_data.item()))
        smooth_losses.append(float(loss_smooth_k.item()))
        smooth_raw_losses.append(float(smooth_raw.item()))

        iteration = step + 1
        if ((cfg.save_every > 0 and iteration % cfg.save_every == 0) or iteration == cfg.num_iter):
            save_snapshot(iteration)
            
        if step % cfg.print_every == 0 or step == cfg.num_iter - 1:
            print(
                f"[blind] step={step:05d} | "
                f"data={loss_data.item():.6g} | "
                f"smooth_raw={smooth_raw.item():.6g} | "
                f"smooth={loss_smooth_k.item():.6g} | "
                f"total={loss.item():.6g}"
            )

    with torch.no_grad():
        out_x = F.softplus(net_x(z_x))
        h_exc = net_exc(z_exc).view(1, 1, kh, kw)
        h_em = net_em(z_em).view(1, 1, kh, kw)
        physics.set_psfs(h_exc, h_em)
        kernels = physics.kernels                      # (25, 1, Kh, Kw)

    return {
        "x": _np(out_x)[0, 0],
        "kernels": _np(kernels)[:, 0],
        "h_exc": _np(h_exc)[0, 0],
        "h_em": _np(h_em)[0, 0],
        "losses": np.asarray(losses, dtype=np.float32),
        "calibration": calibration,
        "data_losses": np.asarray(data_losses, dtype=np.float32),
        "smooth_losses": np.asarray(smooth_losses, dtype=np.float32),
        "smooth_raw_losses": np.asarray(smooth_raw_losses, dtype=np.float32),

        "snapshot_iters": np.asarray(snapshot_iters, dtype=np.int32),
        "x_history": np.stack(x_history),
        "h_exc_history": np.stack(h_exc_history),
        "h_em_history": np.stack(h_em_history),
        "kernels_history": np.stack(kernels_history),

        "gaussian_mse_history": np.asarray( gaussian_mse_history, dtype=np.float32),
        "theory_kernel_mse_history": np.asarray(theory_kernel_mse_history, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# fixed theoretical PSFs: image only
# ---------------------------------------------------------------------------
def run_dip_fixed(y, kernels, cfg, device):
    set_seed(cfg.seed)
    _, _, h, w = y.shape
    physics = PhysicsFixedKernels((1, 1, h, w), device, kernels, )
    net_x = build_image_net(device, cfg.input_depth)
    z_x = torch.rand(1, cfg.input_depth, h, w, device=device)

    optimizer = torch.optim.Adam(net_x.parameters(), lr=cfg.lr_x)
    poisson = PoissonNLL()
    losses = []

    for step in range(cfg.num_iter):
        optimizer.zero_grad()
        out_x = F.softplus(net_x(z_x + cfg.reg_noise_std * torch.randn_like(z_x)))
        loss = poisson(physics(out_x), y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if step % cfg.print_every == 0 or step == cfg.num_iter - 1:
            print(f"[fixed] step={step:05d} | loss={loss.item():.6g}")

    with torch.no_grad():
        out_x = F.softplus(net_x(z_x))

    return {
        "x": _np(out_x)[0, 0],
        "kernels": _np(kernels)[:, 0],
        "losses": np.asarray(losses, dtype=np.float32),
    }
