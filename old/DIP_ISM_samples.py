import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from skimage.io import imsave
import cv2
from networks.skip import skip
from networks.fcn import fcn_ism
from utils.utils_ism_train import PhysicsISM
import utils.utils_ism_train as uismtrain
from utils.utils_ism import psnr_torch
import warnings
warnings.filterwarnings("ignore")
import utils.utils_ism as uism
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_iter", type=int, default=1000)
    p.add_argument("--lr_x", type=float, default=1e-2)
    p.add_argument("--lr_k", type=float, default=1e-4)
    p.add_argument("--reg_noise_std", type=float, default=0.)
    p.add_argument("--phantom_types", type=str, default="tubulin", help= "calibration, tubulin, nucleus, membrane, balls, sparse, mitochondria, for everything type mixture")
    p.add_argument("--Nx", type=int, default=256)
    p.add_argument("--flux", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_path", type=str, default="test_ism/")
    p.add_argument("--plot", action='store_true')
    p.add_argument("--calibration", action='store_true')
    p.add_argument("--loss", default='poisson_loss', choices=['poisson_loss', 'mse_loss'])
    p.add_argument("--lambda_tv", type=float, default=0.)
    p.add_argument("--lambda_center", type=float, default=0.)
    p.add_argument("--lambda_radial", type=float, default=0.)
    p.add_argument("--lambda_smooth", type=float, default=0.)
    p.add_argument("--warmup_iters", type=int, default=200)
    p.add_argument("--enforce_symmetry", action='store_true')
    p.add_argument("--pretraining", action='store_true')
    p.add_argument("--pretraining_iter", type=int, default=100)
    p.add_argument("--pret_psf", type=float, default=4.)
    p.add_argument("--norm_physics", action='store_true')
    p.add_argument("--poisson_gain", type=float, nargs="+", default=[1.0])
    return p.parse_args()


def main():
    hparams = parse_args()
    np.random.seed(seed=hparams.seed)
    torch.manual_seed(hparams.seed)
    device, dtype = torch.device("cuda" if torch.cuda.is_available() else "cpu"), torch.float32

    best_psnr_summary = []

    for gain in hparams.poisson_gain:
        best_psnr = -float("inf")
        best_psnr_iter = -1
        print(f"Running with Poisson gain: {gain}")
            
        save_root = Path(hparams.save_path)
        save_root = save_root / f"gain_{gain}"

        img_dir = save_root / "images"
        kernel_dir = save_root / "kernels"
        # pred_dir = save_root / "predictions"
        curve_dir = save_root / "curves"

        img_dir.mkdir(parents=True, exist_ok=True)
        kernel_dir.mkdir(parents=True, exist_ok=True)
        # pred_dir.mkdir(parents=True, exist_ok=True)
        curve_dir.mkdir(parents=True, exist_ok=True)

        Nx = Ny = hparams.Nx
        PSF = uism.generate_25_psf(plot=hparams.plot).to(device=device, dtype=dtype)
        fingerprint = PSF.view(25, -1).sum(dim=1).view(25, 1, 1, 1)
        if hparams.norm_physics:
            PSF = PSF / (PSF.sum() + 1e-12)
        else:
            PSF = PSF / (fingerprint + 1e-12)


        if hparams.phantom_types == 'calibration':
            kernel_size, sigma = 21, 2.0
            bead = uism.gaussian_kernel2d((kernel_size, kernel_size), sigma).to(device=device, dtype=dtype)
            ground_truth = torch.zeros((1, 1, Nx, Ny), device=device, dtype=dtype)
            cx, cy, k = Nx//2, Ny//2, kernel_size//2
            ground_truth[0, 0, cx-k:cx+k+1, cy-k:cy+k+1] = bead
            ground_truth *= hparams.flux
        else: 
            ground_truth = uism.generate_phantom_data(phantom_type=hparams.phantom_types, Nx=Nx, Ny=Ny, Nz=2, flux=10, device=device, plot=hparams.plot)

        noise_image, _, _ = uism.simulate_measurement(PSF, Nx, ground_truth, device, plot=hparams.plot, gain=gain)
        with torch.no_grad():
            center_obs = noise_image[12:13]  # central detector
            obs_psnr = psnr_torch(
                center_obs,
                ground_truth,
                data_range=ground_truth.max() - ground_truth.min()
            ).item()

        print(f"Central detector PSNR vs ground truth: {obs_psnr:.2f} dB")
        uism.save_kernels_grid(PSF, save_root / "ground_truth_psf.png")
        uism.save_tensor_image(ground_truth, save_root / "ground_truth.png")
        # uism.save_tensor_image(noise_image, save_root / "noise_image.png")
        uism.save_tensor_image(noise_image[12:13], save_root / "noise_image_center.png")
        # noise_image_norm = uism.data_normalization(noise_image, PSF, device)
        # uism.save_tensor_image(noise_image_norm.sum(0, keepdim=True), save_root / "measurement_sum.png")

        metric = {
            "loss": [],
            "img_mse": [],
            "img_psnr": [],
            "meas_mse": [],
            "psf_mse_mean": [],
            "psf_psnr_mean": [],
            "psf_rel_mean": [],
            "psf_mse_each": [],
            "psf_psnr_each": [],
            "psf_rel_each": [],
        }


            
        num_detectors = 25
        input_depth = 8
        n_k = 200

        _, _, H, W = ground_truth.shape
        Kh, Kw = PSF.shape[-2], PSF.shape[-1]

        net_k = fcn_ism(
            num_input_channels=n_k,
            num_output_channels=Kh * Kw,
            num_hidden=1000,
            num_detectors=num_detectors,
            norm= hparams.norm_physics,
            fingerprint=fingerprint
        ).to(device=device, dtype=dtype)

        z_k = torch.rand(num_detectors, n_k, device=device, dtype=dtype)
        z_k_saved = z_k.detach().clone()

        if not hparams.calibration:
            net_x = skip(
                input_depth, 1,
                num_channels_down=[128, 128, 128, 128, 128],
                num_channels_up=[128, 128, 128, 128, 128],
                num_channels_skip=[16, 16, 16, 16, 16],
                upsample_mode="bilinear",
                need_sigmoid=False,
                need_bias=True,
                pad="reflection",
                act_fun="LeakyReLU"
            ).to(device=device, dtype=dtype)

            optimizer = torch.optim.Adam([
                {"params": net_x.parameters(), "lr": hparams.lr_x},
                {"params": net_k.parameters(), "lr": hparams.lr_k}
            ])    
            z_x = torch.rand(1, input_depth, H, W, device=device, dtype=dtype)
            z_x_saved = z_x.detach().clone()
        else: 
            optimizer = torch.optim.Adam(params= net_k.parameters(), lr= hparams.lr_k)    

        # y = noise_image_norm.to(device=device, dtype=dtype)
        y = noise_image.to(device=device, dtype=dtype)
        IMG_MSE = lambda out_x: torch.mean((out_x - ground_truth) ** 2).item()
        IMG_PSNR = lambda out_x: psnr_torch(out_x, ground_truth,data_range=ground_truth.max() - ground_truth.min()).item()
        MEAS_MSE = lambda pred_y: torch.mean((pred_y - y) ** 2).item()    

        physics = PhysicsISM(
            img_size=(1, 1, Nx, Nx),
            device=device,
            bkg=0.01,
        )    
        
        if hparams.loss == 'mse_loss':
            loss_func = torch.nn.MSELoss(reduction="mean")
        else: 
            loss_func = uism.PoissonNLL(norm=hparams.norm_physics, fingerprint=fingerprint, gain=gain).to(device)

        losses = []
        if hparams.pretraining:
            kernel_optimizer = torch.optim.Adam(net_k.parameters(), lr=hparams.lr_k)

            gauss_ker = uism.gaussian_kernel2d(size=(Kh, Kw), sigma=hparams.pret_psf).to(device=device, dtype=dtype)
            pre_kernel = gauss_ker.unsqueeze(0).unsqueeze(0).repeat(num_detectors, 1, 1, 1)
            pre_kernel = pre_kernel / (
                pre_kernel.view(num_detectors, -1).sum(dim=1).view(num_detectors, 1, 1, 1) + 1e-12
            )

            for _ in range(hparams.pretraining_iter):
                kernel_optimizer.zero_grad()
                pred_k = net_k(z_k_saved).view(num_detectors, 1, Kh, Kw)
                loss_pre = ((pred_k - pre_kernel) ** 2).sum()
                loss_pre.backward()
                kernel_optimizer.step()

            with torch.no_grad():
                pred_k = net_k(z_k_saved).view(num_detectors, 1, Kh, Kw)
                mse = torch.mean((pred_k - pre_kernel) ** 2).item()

            print(f"[Pretraining done] mean kernel MSE vs Gaussian: {mse:.6e}")
            uism.save_kernels_grid(pred_k, kernel_dir / "pretrained_kernels.png")



        for step in range(hparams.num_iter):
            optimizer.zero_grad()

            out_k = net_k(z_k_saved)                  # (25, Kh*Kw)
            out_k_m = out_k.view(num_detectors, 1, Kh, Kw)
            if hparams.enforce_symmetry:
                if step > hparams.warmup_iters:
                    alpha = uismtrain.schedule_alpha(step, warmup_iters=hparams.warmup_iters, ramp_iters=400)
                    out_k_m_for_physics = uismtrain.recenter_kernels(out_k_m, alpha=alpha)
                    # out_k_m_for_physics = uismtrain.recenter_kernels(out_k_m, alpha=1.0)

                else:
                    out_k_m_for_physics = out_k_m
            else:
                out_k_m_for_physics = out_k_m

            physics.set_kernels(out_k_m_for_physics) 
    

            if hparams.calibration:
                out_x = ground_truth
            else: 
                z_x = z_x_saved + hparams.reg_noise_std * torch.randn_like(z_x_saved)
                out_x = torch.nn.functional.softplus(net_x(z_x))


            if hparams.loss == 'mse_loss':
                pred_y = physics(out_x)
                std_loss = loss_func(pred_y, y)
            else: 
                std_loss = loss_func(out_x, y, physics)

            loss_barycenter = hparams.lambda_center * uismtrain.barycenter_loss(out_k_m)
            loss_k_radial = hparams.lambda_radial * uismtrain.symmetry_loss(out_k_m)
            loss_k_smooth = hparams.lambda_smooth * uismtrain.smoothness_loss(out_k_m)
            loss_x_tv = hparams.lambda_tv * uismtrain.tv_loss(out_x)
            loss = std_loss + loss_barycenter + loss_k_radial + loss_k_smooth + loss_x_tv

            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            metric["loss"].append(loss.item())

            with torch.no_grad():
                pred_y = physics(out_x)
                img_mse = IMG_MSE(out_x)
                img_psnr = IMG_PSNR(out_x)
                meas_mse = MEAS_MSE(pred_y)

                psf_stats = uism.compute_psf_metrics(out_k_m, PSF)

                metric["img_mse"].append(img_mse)
                metric["img_psnr"].append(img_psnr)
                metric["meas_mse"].append(meas_mse)

                metric["psf_mse_mean"].append(psf_stats["psf_mse_mean"])
                metric["psf_psnr_mean"].append(psf_stats["psf_psnr_mean"])
                metric["psf_rel_mean"].append(psf_stats["psf_rel_mean"])

                metric["psf_mse_each"].append(psf_stats["psf_mse_each"])
                metric["psf_psnr_each"].append(psf_stats["psf_psnr_each"])
                metric["psf_rel_each"].append(psf_stats["psf_rel_each"])       

            if img_psnr > best_psnr:
                best_psnr = img_psnr
                best_psnr_iter = step

            if step % 200 == 0 or step == hparams.num_iter - 1 or step == 0:
                if hparams.calibration:
                    print(
                        f"step={step:05d} | "
                        f"loss={metric['loss'][-1]:.6f} | "
                        f"psf_mse={metric['psf_mse_mean'][-1]:.6e} | "
                        f"psf_psnr={metric['psf_psnr_mean'][-1]:.2f} | "
                        f"meas_mse={metric['meas_mse'][-1]:.6e}"
                    )
                else:
                    print(
                        f"step={step:04d} | "
                        f"loss={metric['loss'][-1]:.6f} | "
                        f"img_psnr={metric['img_psnr'][-1]:.2f} | "
                        f"psf_psnr={metric['psf_psnr_mean'][-1]:.2f} | "
                        # f"std_loss={std_loss.item():.3f} | "
                        # f"barycenter_loss={loss_barycenter.item():.3f} | "
                        # f"symmetry_loss={loss_k_radial.item():.3f} | "
                        # f"smoothness_loss={loss_k_smooth.item():.3f}"
                        # f"tv_loss={loss_x_tv.item():.3f}"
                    )

                with torch.no_grad():
                    uism.save_tensor_image(out_x, img_dir / f"recon_{step:05d}.png")
                    uism.save_kernels_grid(out_k_m_for_physics, kernel_dir / f"kernels_{step:05d}.png")
                    # uismtrain.save_kernel_with_markers(out_k_m[12, 0], kernel_dir / f"k12_marked_{step:05d}.png")

        plot_metric = {
            # "loss": metric["loss"],
            # "img_mse": metric["img_mse"],
            "img_psnr": metric["img_psnr"],
            # "meas_mse": metric["meas_mse"],
            # "psf_mse_mean": metric["psf_mse_mean"],
            "psf_psnr_mean": metric["psf_psnr_mean"],
            # "psf_rel_mean": metric["psf_rel_mean"],
        }
        uism.save_curves(plot_metric, curve_dir / "metrics.png")
        best_psnr_summary.append({
        "gain": gain,
        "best_psnr": best_psnr,
        "best_iter": best_psnr_iter,
        })

    uism.save_best_psnr_summary(best_psnr_summary, hparams.save_path)

if __name__ == "__main__":
    main()