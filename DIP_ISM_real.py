import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from skimage.io import imsave
from networks.skip import skip
from networks.fcn_25 import fcn_ism
from utils.utils_ism_train_25 import PhysicsISM
import utils.utils_ism_train_25 as uismtrain
from utils.utils_ism import psnr_torch
import warnings
warnings.filterwarnings("ignore")
import utils.utils_ism as uism
from pathlib import Path
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_iter",        type=int,   default=1000)
    p.add_argument("--lr_x",            type=float, default=1e-2)
    p.add_argument("--lr_k",            type=float, default=1e-4)
    p.add_argument("--reg_noise_std",   type=float, default=0.)
    p.add_argument("--Nx",              type=int,   default=256)
    p.add_argument("--seed",            type=int,   default=0)
    p.add_argument("--save_path",       type=str,   default="test_ism/")
    p.add_argument("--loss",            default='poisson_loss',
                   choices=['poisson_loss', 'mse_loss'])
    p.add_argument("--pretraining",     action='store_true')
    p.add_argument("--pretraining_iter",type=int,   default=100)
    p.add_argument("--pret_psf",        type=float, default=4.)
    # saving / printing
    p.add_argument("--save_every",      type=int,   default=200,help="Save images/kernels/curves every N iterations (0 = only at end).")
    # real data
    p.add_argument("--real_data",       type=str,   required=True,help="Path to a .pth file with keys 'measurment' and 'PSF'.")
    p.add_argument("--psf_z_idx",       type=int,   default=None,help="Which z-plane of the PSF to use.")
    p.add_argument("--crop_size",       type=int,   default=None)
    p.add_argument("--inspect",         action='store_true')
    return p.parse_args()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _save_img(tensor, path):
    x = tensor.detach().cpu().squeeze().float().numpy()
    x = np.clip(x, 0, None)
    if x.max() > 0:
        x = x / x.max()
    imsave(str(path), (255 * x).astype(np.uint8))

def _save_observation(save_root, noise_image):
    """Called once at the start to sanity-check the loaded data."""
    _save_img(noise_image[12:13], save_root / "obs_center.png")
    uism.save_stack_grid(noise_image, save_root / "obs_stack.png", nrow=5)

def _periodic_save(step, save_root, out_x, out_k_m):
    """Save reconstruction and kernels at regular intervals."""
    _save_img(out_x, save_root / f"recon_{step:05d}.png")
    uism.save_kernels_grid(out_k_m, save_root / f"kernels_{step:05d}.png", nrow=5)


# ---------------------------------------------------------------------------
# one run for a single z-plane
# ---------------------------------------------------------------------------

def run_one_zplane(hparams, psf_z_idx, save_root, device, dtype):
    np.random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)

    save_root.mkdir(parents=True, exist_ok=True)
    crop_size = hparams.crop_size or hparams.Nx

    PSF, fingerprint, noise_image, Nx = uism.load_real_data(
        path=hparams.real_data,
        psf_z_idx=psf_z_idx,
        crop_size=crop_size,
        device=device,
        dtype=dtype,
        norm_physics=False,
    )

    print(f"\n[z={psf_z_idx}] spatial size: {Nx}×{Nx} | " f"PSF: {tuple(PSF.shape)} | " f"meas range: [{noise_image.min().item():.2g}, {noise_image.max().item():.2g}]")
    print(f"  centre detector  max={noise_image[12].max().item():.1f}  " f"sum={noise_image[12].sum().item():.1f}")
    print(f"  summed (all det) max={noise_image.sum(0).max().item():.1f}")

    num_detectors = 25
    input_depth   = 8
    n_k           = 200
    H = W         = Nx
    Kh, Kw        = PSF.shape[-2], PSF.shape[-1]

    net_k = fcn_ism(
        num_input_channels=n_k,
        num_output_channels=Kh * Kw,
        num_hidden=1000,
        num_detectors=num_detectors,
        norm=False,
        fingerprint=fingerprint,
    ).to(device=device, dtype=dtype)

    z_k_saved = torch.rand(num_detectors, n_k, device=device, dtype=dtype)

    net_x = skip(
        input_depth, 1,
        num_channels_down=[128, 128, 128, 128, 128],
        num_channels_up  =[128, 128, 128, 128, 128],
        num_channels_skip=[16,  16,  16,  16,  16],
        upsample_mode="bilinear",
        need_sigmoid=False,
        need_bias=True,
        pad="reflection",
        act_fun="LeakyReLU",
    ).to(device=device, dtype=dtype)

    optimizer = torch.optim.Adam([
        {"params": net_x.parameters(), "lr": hparams.lr_x},
        {"params": net_k.parameters(), "lr": hparams.lr_k},
    ])
    z_x_saved = torch.rand(1, input_depth, H, W, device=device, dtype=dtype)

    y        = noise_image.to(device=device, dtype=dtype)

    physics = PhysicsISM(img_size=(1, 1, Nx, Nx), device=device, bkg=0.01)

    if hparams.loss == 'mse_loss':
        loss_func = torch.nn.MSELoss(reduction="mean")
    else:
        loss_func = uism.PoissonNLL(norm=False, fingerprint=fingerprint, gain=1.0).to(device)

    if hparams.pretraining:
        kernel_optimizer = torch.optim.Adam(net_k.parameters(), lr=hparams.lr_k)
        gauss_ker   = uism.gaussian_kernel2d(size=(Kh, Kw), sigma=hparams.pret_psf).to(device=device, dtype=dtype)
        pre_kernel  = gauss_ker.unsqueeze(0).unsqueeze(0).repeat(num_detectors, 1, 1, 1)
        pre_kernel  = pre_kernel / (pre_kernel.view(num_detectors, -1).sum(dim=1).view(num_detectors, 1, 1, 1) + 1e-12)
        for _ in range(hparams.pretraining_iter):
            kernel_optimizer.zero_grad()
            pred_k = net_k(z_k_saved).view(num_detectors, 1, Kh, Kw)
            loss_pre = ((pred_k - pre_kernel) ** 2).sum()
            loss_pre.backward()
            kernel_optimizer.step()

        with torch.no_grad():
            pred_k = net_k(z_k_saved).view(num_detectors, 1, Kh, Kw)
            mse = torch.mean((pred_k - pre_kernel) ** 2).item()
        print(f"  [pretraining done] kernel MSE vs Gaussian: {mse:.6e}")


    step_times = []

    for step in range(hparams.num_iter):
        t0 = time.perf_counter()
        optimizer.zero_grad()

        out_k = net_k(z_k_saved)
        out_k_m = out_k.view(num_detectors, 1, Kh, Kw)

        physics.set_kernels(out_k_m)

        z_x = z_x_saved + hparams.reg_noise_std * torch.randn_like(z_x_saved)
        out_x = torch.nn.functional.softplus(net_x(z_x))

        if hparams.loss == 'mse_loss':
            pred_y = physics(out_x)
            loss = loss_func(pred_y, y)
        else:
            loss = loss_func(out_x, y, physics)

        loss.backward()
        optimizer.step()

        step_times.append(time.perf_counter() - t0)

        if step % 200 == 0 or step == hparams.num_iter - 1:
            avg_t = np.mean(step_times) * 1e3
            print(
                f"step={step:04d} | "
                f"loss={loss.item():.6f} | "
                f"avg_step={avg_t:.1f}ms"
            )

        # ---- periodic print & save ----------------------------------------
        is_save_step = ((hparams.save_every > 0 and step % hparams.save_every == 0) or step == hparams.num_iter - 1 )

        if is_save_step:
            _periodic_save(step, save_root, out_x, out_k_m)


    total_s = np.sum(step_times)
    print(
        f"\n[z={psf_z_idx}] done – avg step: {np.mean(step_times)*1e3:.1f} ms | "
        f"total: {total_s:.1f} s ({total_s/60:.1f} min) "
    )

    # ---- final saves -------------------------------------------------------
    _periodic_save(hparams.num_iter - 1, save_root, out_x, out_k_m)

    # ISM sum-shift reconstruction (simple baseline, no PSF knowledge needed)
    with torch.no_grad():
        ism_sum = noise_image.sum(0, keepdim=True)          # (1,1,H,W)
        _save_img(ism_sum, save_root / "ism_open_pinhole.png")

    print(f"  Saved outputs to: {save_root}")
    return None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    hparams = parse_args()
    device, dtype = torch.device("cuda" if torch.cuda.is_available() else "cpu"), torch.float32

    if hparams.inspect:
        uism.inspect_real_data(hparams.real_data)
        # Also tell us how many z-planes the PSF has
        data = torch.load(hparams.real_data, map_location="cpu", weights_only=False)
        psf = data["PSF"]
        if psf.dim() == 3:
            psf = psf.unsqueeze(1)
        nz = psf.shape[1]
        print(f"PSF has {nz} z-plane(s): valid --psf_z_idx values are 0 … {nz-1}")
        return

    _data = torch.load(hparams.real_data, map_location="cpu", weights_only=False)
    _psf  = _data["PSF"]
    if _psf.dim() == 3:
        _psf = _psf.unsqueeze(1)
    nz = _psf.shape[1]
    del _data, _psf

    if hparams.psf_z_idx is not None:
        z_planes = [hparams.psf_z_idx]
    else:
        z_planes = list(range(nz))
        print(f"No --psf_z_idx given → will try all {nz} z-plane(s): {z_planes}")

    save_root_base = Path(hparams.save_path)

    results = {}
    for z_idx in z_planes:
        print(f"\n{'='*60}")
        print(f"  Real data: {hparams.real_data}   |   z-plane {z_idx}/{nz-1}")
        print(f"{'='*60}")

        if len(z_planes) == 1:
            save_root = save_root_base
        else:
            save_root = save_root_base / f"z{z_idx:02d}"

        run_one_zplane(hparams, z_idx, save_root, device, dtype)

if __name__ == "__main__":
    main()