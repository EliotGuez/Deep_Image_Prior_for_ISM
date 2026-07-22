import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

from networks.skip import skip
from networks.fcn_2psf import fcn_single_kernel
from utils.utils_ism_train import PhysicsISMFactorized, build_pinholes_tensor, tv_loss_kernel
from utils.utils_ism import (
    generate_25_psf_zernike,
    generate_phantom_data,
    simulate_measurement,
    reconstruct_single_psf,
    reconstruct_psf_from_zernike,
    gaussian_kernel2d,
    gaussian_beads,
    PoissonNLL,
    psnr_torch,
    compute_grouped_kernel_mse,
    save_tensor_image,
    save_kernels_grid,
    save_stack_grid,
    save_two_psfs,
    save_curves,
    save_zernike_comparison,
    save_zernike_history,
    save_zernike_gt,
)
import utils.zernike_fit as zfit
from zernike_net.predict import load_model, predict_zernike
from zernike_net.constants import FIXED_ZERNIKE, ZERNIKE_MAX, ZERNIKE_FREE

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    p = argparse.ArgumentParser(description="ISM blind deconvolution — 2-PSF factorised DIP + Zernike fitting")
    p.add_argument("--num_iter", type=int, default=1000)
    p.add_argument("--lr_x", type=float, default=1e-2)
    p.add_argument("--lr_k", type=float, default=1e-4)
    p.add_argument("--reg_noise_std", type=float, default=0.)
    p.add_argument("--loss_type", type=str, default="poisson", choices=["poisson", "mse"])
    p.add_argument("--lambda_tv_k", type=float, default=0.0)
    p.add_argument("--poisson_gain", type=float, default=.3)
    p.add_argument("--phantom_types",    type=str,   default="tubulin",help="calibration | tubulin | nucleus | membrane | balls | sparse | mitochondria | mixture")
    p.add_argument("--Nx", type=int, default=256)
    p.add_argument("--flux",type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--coeff_scale", type=float, default=0.2)
    p.add_argument("--beads_sigma", type=float, default=0.3)
    p.add_argument("--pretraining", action="store_true")
    p.add_argument("--pretraining_iter", type=int, default=100)
    p.add_argument("--pret_psf", type=float, default=4., help="Sigma of Gaussian used as pretraining target")
    p.add_argument("--zernike_num_starts", type=int, default=10)
    p.add_argument("--zernike_max_iter", type=int, default=200)
    p.add_argument("--zernike_loss", type=str, default="mae",choices=["mae", "mse", "kl"])
    p.add_argument("--zernike_bound", type=float, default=2.0)
    p.add_argument("--zernike_init", type=str, default="sobol", choices=["sobol", "lhs"])
    p.add_argument("--save_path", type=str,  default="test_ism/")
    p.add_argument("--calibration", action="store_true")
    p.add_argument("--do_zernike", action="store_true")
    p.add_argument("--zk_path", type=str, default="zk_net/center/last.pth")
    p.add_argument("--zernike_method", type=str, default="neural", choices=["iit", "neural"])
    return p.parse_args()

def main():
    hparams = parse_args()
    set_seed(hparams.seed)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float32

    save_root  = Path(hparams.save_path)
    img_dir    = save_root / "images"
    kernel_dir = save_root / "kernels"
    curve_dir  = save_root / "curves"
    psf2_dir   = save_root / "two_psfs"
    zk_dir     = save_root / "zernike"
    for d in [img_dir, kernel_dir, curve_dir, psf2_dir, zk_dir]:
        d.mkdir(parents=True, exist_ok=True)


    PSF, gt_zer_ex, gt_zer_em, gt_exPSF, gt_detPSF, grid = generate_25_psf_zernike(seed=hparams.seed, coeff_scale=hparams.coeff_scale)


    # print("GT Zernike excitation:", gt_zer_ex.numpy())
    # print("GT Zernike emission  :", gt_zer_em.numpy())

    gt_detPSF_norm  = gt_detPSF / (gt_detPSF.sum(dim=(0, 1), keepdim=True) + 1e-12)
    gt_det_center   = gt_detPSF_norm[..., 12]   # (Kh, Kw)

    PSF = PSF.to(device=device, dtype=dtype)
    gt_exPSF  = gt_exPSF.to(device=device, dtype=dtype)
    gt_detPSF = gt_detPSF.to(device=device, dtype=dtype)

    fingerprint = PSF.view(25, -1).sum(dim=1).view(25, 1, 1, 1)
    PSF = PSF / (fingerprint + 1e-12)
    Kh, Kw = PSF.shape[-2], PSF.shape[-1]

    #clean for save 
    PSF_clean, _, _, gt_exPSF_clean, gt_detPSF_clean, grid_clean = generate_25_psf_zernike(seed=hparams.seed, coeff_scale=0.)
    gt_detPSF_clean_norm = gt_detPSF_clean / (gt_detPSF_clean.sum(dim=(0, 1), keepdim=True) + 1e-12)
    gt_det_center_clean = gt_detPSF_clean_norm[..., 12]
    fingerprint_clean = PSF_clean.view(25, -1).sum(dim=1).view(25, 1, 1, 1)
    PSF_clean = PSF_clean / (fingerprint_clean + 1e-12)

    save_kernels_grid(PSF_clean, save_root / "gt_psf_25_clean.png")
    save_two_psfs(gt_exPSF_clean, gt_det_center_clean, save_root / "gt_two_psfs_clean.png")
    save_zernike_gt(gt_zer_ex, gt_zer_em, zk_dir / "zernike_gt.png")
    Nx = hparams.Nx

    if hparams.calibration:
        ground_truth = gaussian_beads(grid_size=Kh, num_beads=15, sigma=hparams.beads_sigma, flux=hparams.flux, device=device, dtype=dtype, seed=hparams.seed)
        noise_image, blurred_image, _ = simulate_measurement(PSF, Kh, ground_truth, device, gain=hparams.poisson_gain, back_lvl=1e-5)
    else:
        ground_truth = generate_phantom_data(hparams.phantom_types, Nx, Nx, 2, device, flux=hparams.flux)
        noise_image, blurred_image, _ = simulate_measurement(PSF, Nx, ground_truth, device, gain=hparams.poisson_gain, back_lvl=0.01)

    save_kernels_grid(PSF, save_root / "gt_psf_25.png")
    save_two_psfs(gt_exPSF, gt_det_center, save_root / "gt_two_psfs.png")
    save_tensor_image(ground_truth, save_root / "gt_image.png")

    y = noise_image.to(device=device, dtype=dtype)
    save_tensor_image(y[12], save_root / "central_detector_noise.png")
    save_stack_grid(y, save_root / "observation_grid.png")


    n_k         = 200
    input_depth = 8
    _, _, H, W  = ground_truth.shape

    net_exc = fcn_single_kernel(num_input_channels=n_k, num_output_channels=Kh * Kw, num_hidden=400).to(device=device, dtype=dtype)

    net_em = fcn_single_kernel(num_input_channels=n_k, num_output_channels=Kh * Kw, num_hidden=400).to(device=device, dtype=dtype)

    z_exc = torch.rand(1, n_k, device=device, dtype=dtype)
    z_em  = torch.rand(1, n_k, device=device, dtype=dtype)

    if hparams.calibration:
        optimizer = torch.optim.Adam([
            {"params": net_exc.parameters(), "lr": hparams.lr_k},
            {"params": net_em.parameters(),  "lr": hparams.lr_k},
        ])
    else:
        net_x = skip(
            input_depth, 1,
            num_channels_down=[128, 128, 128, 128, 128],
            num_channels_up=[128, 128, 128, 128, 128],
            num_channels_skip=[16, 16, 16, 16, 16],
            upsample_mode="bilinear",
            need_sigmoid=False, need_bias=True,
            pad="reflection", act_fun="LeakyReLU",
        ).to(device=device, dtype=dtype)
        z_x = torch.rand(1, input_depth, H, W, device=device, dtype=dtype)

        optimizer = torch.optim.Adam([
            {"params": net_x.parameters(),   "lr": hparams.lr_x},
            {"params": net_exc.parameters(), "lr": hparams.lr_k},
            {"params": net_em.parameters(),  "lr": hparams.lr_k},
        ])

    pinholes = build_pinholes_tensor(grid, device=device, dtype=dtype)
    bkg  = 1e-5 if hparams.calibration else 0.01

    physics = PhysicsISMFactorized(img_size=(1, 1, H, W), device=device, bkg=bkg, pinholes=pinholes)

    if hparams.loss_type == "poisson":
        loss_func = PoissonNLL(gain=hparams.poisson_gain).to(device)
    else:
        _mse = nn.MSELoss()
        def loss_func(x, y, physics):
            pred = physics(x)
            return _mse(pred / hparams.poisson_gain, y / hparams.poisson_gain)


    if hparams.pretraining:
        pre_opt = torch.optim.Adam(list(net_exc.parameters()) + list(net_em.parameters()), lr=hparams.lr_k)
        gauss = gaussian_kernel2d((Kh, Kw), hparams.pret_psf).to(device=device, dtype=dtype)
        target_pre = gauss.unsqueeze(0).unsqueeze(0)

        for _ in range(hparams.pretraining_iter):
            pre_opt.zero_grad()
            h_exc_pre = net_exc(z_exc).view(1, 1, Kh, Kw)
            h_em_pre  = net_em(z_em).view(1, 1, Kh, Kw)
            loss_pre  = ((h_exc_pre - target_pre) ** 2).sum()
            loss_pre += ((h_em_pre  - target_pre) ** 2).sum()
            loss_pre.backward()
            pre_opt.step()
        print("[Pretraining done] PSF networks initialised as Gaussian")


    IMG_PSNR = lambda x: psnr_torch(x, ground_truth, data_range=ground_truth.max() - ground_truth.min()).item()

    metric = {
        "loss":           [],
        "img_psnr":       [],
        "psf_mse_outer":  [],
        "psf_mse_inner":  [],
        "psf_mse_center": [],
        "exc_mse":        [],
        "em_mse":         [],
    }
    eps = 1e-12
    if hparams.zernike_method == "neural":
        zk_model  = load_model(hparams.zk_path, device=device)
    clean_exc = reconstruct_single_psf(torch.zeros(ZERNIKE_MAX), wl=450, Kh=Kh, device=device)
    clean_em  = reconstruct_single_psf(torch.zeros(ZERNIKE_MAX), wl=660, Kh=Kh, device=device)
    for step in range(hparams.num_iter):
        optimizer.zero_grad()

        z_exc_noisy = z_exc + hparams.reg_noise_std * torch.randn_like(z_exc)
        z_em_noisy  = z_em  + hparams.reg_noise_std * torch.randn_like(z_em)

        h_exc = net_exc(z_exc_noisy).view(1, 1, Kh, Kw)
        h_em  = net_em(z_em_noisy).view(1, 1, Kh, Kw)

        physics.set_psfs(h_exc, h_em)
        out_k = physics.kernels   # (25, 1, Kh, Kw)

        if hparams.calibration:
            out_x = ground_truth
        else:
            z_x_noisy = z_x + hparams.reg_noise_std * torch.randn_like(z_x)
            out_x     = torch.nn.functional.softplus(net_x(z_x_noisy))

        loss = loss_func(out_x, y, physics) + hparams.lambda_tv_k * tv_loss_kernel(out_k)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            group_mse = compute_grouped_kernel_mse(out_k, PSF)
            exc_mse   = torch.mean((h_exc - gt_exPSF.view(1, 1, Kh, Kw)) ** 2).item()
            em_mse    = torch.mean((h_em  - gt_det_center.view(1, 1, Kh, Kw)) ** 2).item()

            metric["loss"].append(loss.item())
            metric["img_psnr"].append(IMG_PSNR(out_x))
            metric["psf_mse_outer"].append(np.log10(group_mse["psf_mse_outer"]  + eps))
            metric["psf_mse_inner"].append(np.log10(group_mse["psf_mse_inner"]  + eps))
            metric["psf_mse_center"].append(np.log10(group_mse["psf_mse_center"] + eps))
            metric["exc_mse"].append(np.log10(exc_mse + eps))
            metric["em_mse"].append(np.log10(em_mse  + eps))

        if step % 200 == 0 or step == hparams.num_iter - 1:
            psnr_str = "" if hparams.calibration else f"img_psnr={metric['img_psnr'][-1]:.2f} | "
            print(
                f"step={step:05d} | loss={metric['loss'][-1]:.4f} | "
                f"log10(c/i/o mse)=({metric['psf_mse_center'][-1]:.2f}/"
                f"{metric['psf_mse_inner'][-1]:.2f}/"
                f"{metric['psf_mse_outer'][-1]:.2f}) | "
                f"log10(exc/em mse)=({metric['exc_mse'][-1]:.2f}/"
                f"{metric['em_mse'][-1]:.2f}) | "
                + psnr_str
            )
            with torch.no_grad():
                save_tensor_image(out_x,  img_dir    / f"recon_{step:05d}.png")
                save_kernels_grid(out_k,  kernel_dir / f"kernels_{step:05d}.png")
                save_two_psfs(h_exc, h_em, psf2_dir  / f"two_psfs_{step:05d}.png")

    save_curves({k: metric[k] for k in ["psf_mse_center", "psf_mse_inner", "psf_mse_outer", "exc_mse", "em_mse"]},curve_dir / "psf_metrics.png")
    if not hparams.calibration:
        save_curves({"img_psnr": metric["img_psnr"]}, curve_dir / "psnr.png")



    if hparams.do_zernike:

        print("\n" + "=" * 60)
        print("Fitting Zernike coefficients to recovered h_exc and h_em")
        print("=" * 60)

        with torch.no_grad():
            h_exc_final = net_exc(z_exc).view(1, 1, Kh, Kw)
            h_em_final  = net_em(z_em).view(1, 1, Kh, Kw)

        if hparams.zernike_method == "neural":
            zer_ex_fit, zer_em_fit = predict_zernike(zk_model, h_exc_final, h_em_final,clean_exc, clean_em, device=device)
        else:
            zer_ex_fit, zer_em_fit, _, _ = zfit.fit_zernike_from_two_psfs(
                h_exc_final, h_em_final,
                zer_max        = 15,
                num_starts     = hparams.zernike_num_starts,
                max_iter       = hparams.zernike_max_iter,
                loss           = hparams.zernike_loss,
                bound          = hparams.zernike_bound,
                initialization = hparams.zernike_init,
                verbose        = True,
                Kh             = Kh,
            )

        print("\n\n" + "=" * 60)
        print("Zernike evaluation")
        print("=" * 60)
        zer_mse_ex = torch.mean((zer_ex_fit - gt_zer_ex) ** 2).item()
        zer_mse_em = torch.mean((zer_em_fit - gt_zer_em) ** 2).item()
        zer_mae_ex = torch.mean(torch.abs(zer_ex_fit - gt_zer_ex)).item()
        zer_mae_em = torch.mean(torch.abs(zer_em_fit - gt_zer_em)).item()

        print(f"\nZernike MSE — exc: {zer_mse_ex:.6f}  |  em: {zer_mse_em:.6f}")
        print(f"Zernike MAE — exc: {zer_mae_ex:.6f}  |  em: {zer_mae_em:.6f}")

        # Pixel-space comparison: PSF rebuilt from fitted Zernike vs GT PSF
        psf_ex_fit = reconstruct_single_psf(zer_ex_fit, wl=450, Kh=Kh, device=device, dtype=dtype)
        psf_em_fit = reconstruct_single_psf(zer_em_fit, wl=660, Kh=Kh, device=device, dtype=dtype)

        gt_ex_norm = gt_exPSF.squeeze() / (gt_exPSF.squeeze().sum() + eps)
        gt_em_norm = gt_det_center      / (gt_det_center.sum()       + eps)

        psf_ex_mse = torch.mean((psf_ex_fit - gt_ex_norm) ** 2).item()
        psf_em_mse = torch.mean((psf_em_fit - gt_em_norm) ** 2).item()

        h_exc_norm = h_exc_final.squeeze() / (h_exc_final.squeeze().sum() + eps)
        h_em_norm  = h_em_final.squeeze()  / (h_em_final.squeeze().sum()  + eps)
        dip_ex_mse = torch.mean((h_exc_norm.cpu() - gt_ex_norm.cpu()) ** 2).item()
        dip_em_mse = torch.mean((h_em_norm.cpu()  - gt_em_norm.cpu()) ** 2).item()

        # -----------------------------------------------------------------------
        #  Save Zernike outputs
        # -----------------------------------------------------------------------
        save_zernike_comparison(gt_zer_ex, gt_zer_em, zer_ex_fit, zer_em_fit, zk_dir / "zernike_gt_vs_fit.png")
        psf25_from_zernike = reconstruct_psf_from_zernike(zer_ex_fit, zer_em_fit, Kh=Kh, device=device, dtype=dtype)
        save_kernels_grid(psf25_from_zernike, zk_dir / "psf25_from_zernike.png")

if __name__ == "__main__":
    main()
