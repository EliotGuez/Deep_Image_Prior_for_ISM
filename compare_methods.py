from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from theory_mid import run_theory_and_mid, extract_mid_image
from dip_runners import DIPConfig, hwc_to_stack, normalize_psf_stack, run_dip_blind, run_dip_fixed
from figure_compare import make_figure


def parse_args():
    p = argparse.ArgumentParser(description="ISM blind DIP vs MID vs DIP-with-theoretical-PSFs")
    p.add_argument("--stage", choices=["theory", "dip", "figure", "all"], default="all")
    p.add_argument("--work_dir", default="runs/compare")
    p.add_argument("--seed", type=int, default=0)

    # data / theory
    p.add_argument("--path_data", default="data/02_TUB_data.pth")
    p.add_argument("--crop_size", type=int, default=512)
    p.add_argument("--na", type=float, default=1.49,)
    p.add_argument("--exwl", type=float, default=639.0)
    p.add_argument("--emwl", type=float, default=665.0)
    p.add_argument("--mask_sampl", type=int, default=101)
    p.add_argument("--z_in_idx", type=int, default=1)

    # MID
    p.add_argument("--mid_num_iter", type=int, default=15)
    p.add_argument("--mid_iter", type=int, default=None) # None means use the last iteration for the figure

    # DIP (shared by blind and fixed)
    p.add_argument("--num_iter", type=int, default=1000)
    p.add_argument("--lr_x", type=float, default=1e-2)
    p.add_argument("--reg_noise_std", type=float, default=0.0)

    p.add_argument("--print_every", type=int, default=200)
    p.add_argument("--skip_blind", action="store_true")
    p.add_argument("--skip_fixed", action="store_true")
    p.add_argument("--save_every", type=int, default=100)
    
    # DIP only blind
    p.add_argument("--pretraining", action="store_true")
    p.add_argument("--pretraining_iter", type=int, default=100)
    p.add_argument("--pretraining_lr", type=float, default=1e-4)
    p.add_argument("--pret_psf", type=float, default=4.0)
    p.add_argument("--lr_k", type=float, default=1e-4)
    p.add_argument("--smooth_k", type=float, default=0.0)


    # figure
    p.add_argument("--clip_percentile", type=float, default=100.0)
    p.add_argument("--psf_global_norm", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
def stage_theory(args, work):
    if args.path_data is None:
        raise ValueError("--path_data is required for the theory stage")
    run_theory_and_mid(
        path_data=args.path_data, out_npz=str(work / "theory_mid.npz"),
        crop_size=args.crop_size, na=args.na, exwl=args.exwl, emwl=args.emwl,
        mask_sampl=args.mask_sampl, num_iter=args.mid_num_iter, z_in_idx=args.z_in_idx,
    )


def stage_dip(args, work):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    th = np.load(work / "theory_mid.npz")
    dset = th["dset"]                                   # (H, W, 25): same crop as MID
    psf_ism = th["psf_ism"]                             # (Nz, Mx, My, 25)
    z_in = int(th["z_in_idx"])
    pxsizex_nm = float(th["pxsizex_nm"])

    y = hwc_to_stack(dset, device)                                   # (25, 1, H, W)
    ref_raw = hwc_to_stack(psf_ism[z_in], device)
    ref = normalize_psf_stack(ref_raw)       # (25, 1, Kh, Kw)

    # quick sanity check: PSF mass vs measurement mass
    psf_mass, meas_mass = ref_raw.sum(dim=(-2, -1)).flatten(), y.sum(dim=(-2, -1)).flatten()
    psf_rel, meas_rel = psf_mass / psf_mass.mean(), meas_mass / meas_mass.mean()
    print(f"Relative PSF masses: {psf_rel}, Relative measurement masses: {meas_rel}")
    corr = torch.corrcoef(torch.stack([psf_rel, meas_rel]))[0, 1]
    print(f"PSF / measurement mass correlation = {corr.item():.4f}")



    print(f"[dip] y {tuple(y.shape)} | theoretical PSFs {tuple(ref.shape)} (plane {z_in})")

    cfg = DIPConfig(
        num_iter=args.num_iter, lr_x=args.lr_x, lr_k=args.lr_k, smooth_k=args.smooth_k,
        reg_noise_std=args.reg_noise_std, seed=args.seed, pretraining=args.pretraining, 
        pretraining_iter=args.pretraining_iter, pretraining_lr=args.pretraining_lr, pret_psf=args.pret_psf, 
        print_every=args.print_every, save_every=args.save_every,
    )

    if not args.skip_blind:
        out = run_dip_blind(y, ref, pxsizex_nm, cfg, device)
        cal = out.pop("calibration")
        np.savez_compressed(work / "dip_blind.npz", calibration=json.dumps(cal), **out)
        if cal.get("r2", 1.0) < 0.9:
            print(f"[WARNING] orientation R^2 = {cal['r2']:.3f} < 0.9: check the detector "
                  "geometry / channel order for this dataset before trusting the blind result.")

    if not args.skip_fixed:
        out = run_dip_fixed(y, ref, cfg, device)
        np.savez_compressed(work / "dip_fixed.npz", **out)

    with open(work / "run_config.json", "w") as f:
        json.dump({k: v for k, v in vars(args).items()}, f, indent=2, default=str)


def stage_figure(args, work):

    th = np.load(work / "theory_mid.npz")
    dset = th["dset"]
    psf_ism = th["psf_ism"]
    z_in = int(th["z_in_idx"])
    pxsizex_nm = float(th["pxsizex_nm"])

    psf_theory = np.moveaxis(psf_ism[z_in], -1, 0)                   # (25, Kh, Kw)
    psf_theory = np.clip(psf_theory, 0, None)
    psf_mass = psf_theory.sum(axis=(-2, -1), keepdims=True)
    psf_theory = psf_theory / (psf_mass.mean() + 1e-12)
    
    blind = np.load(work / "dip_blind.npz")
    fixed = np.load(work / "dip_fixed.npz")

    x_mid = extract_mid_image(th["mid_all"], args.mid_iter, z_in,
                              n_planes=psf_ism.shape[0], spatial_shape=dset.shape[:2])
    mid_label = f"MID (iter {args.mid_iter})" if args.mid_iter is not None else "MID"

    for ext in ("pdf", "png"):
        make_figure(
            y_stack=dset, psf_theory=psf_theory, psf_blind=blind["kernels"],
            x_blind=blind["x"], x_mid=x_mid, x_fixed=fixed["x"],
            pxsizex_nm=pxsizex_nm, out_path=str(work / f"comparison.{ext}"),
            mid_label=mid_label, clip_percentile=args.clip_percentile,
            psf_per_tile_norm=not args.psf_global_norm,
        )


def main():
    args = parse_args()
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    if args.stage in ("theory", "all"):
        stage_theory(args, work)
    if args.stage in ("dip", "all"):
        stage_dip(args, work)
    if args.stage in ("figure", "all"):
        stage_figure(args, work)


if __name__ == "__main__":
    main()
