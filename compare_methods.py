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
    p = argparse.ArgumentParser(description="ISM blind DIP vs MID vs fixed-PSF DIP")
    p.add_argument("--stage", choices=["theory", "dip", "figure", "all"], default="all")
    p.add_argument("--work_dir", default="runs/compare")
    p.add_argument("--seed", type=int, default=0)

    # Data / MID
    p.add_argument("--path_data", default="data/02_TUB_data.pth")
    p.add_argument("--crop_size", type=int, default=512)
    p.add_argument("--z_in_idx", type=int, default=1)
    p.add_argument("--mid_num_iter", type=int, default=15)
    p.add_argument("--mid_planes", choices=["all", "single"], default="all")
    p.add_argument("--mid_iter", type=int, default=None)

    # DIP
    p.add_argument("--num_iter", type=int, default=1000)
    p.add_argument("--lr_x", type=float, default=1e-2)
    p.add_argument("--lr_k", type=float, default=1e-4)
    p.add_argument("--smooth_k", type=float, default=0.0)
    p.add_argument("--reg_noise_std", type=float, default=0.0)
    p.add_argument("--print_every", type=int, default=200)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--skip_blind", action="store_true")
    p.add_argument("--skip_fixed", action="store_true")

    # Blind DIP
    p.add_argument("--pretraining", action="store_true")
    p.add_argument("--pretraining_iter", type=int, default=100)
    p.add_argument("--pretraining_lr", type=float, default=1e-4)
    p.add_argument("--pret_psf", type=float, default=4.0)

    # Detector geometry
    p.add_argument("--estimate_grid", action="store_true")
    p.add_argument("--magnification", type=float, default=None)
    p.add_argument("--rotation_deg", type=float, default=None)
    p.add_argument("--mirroring", type=int, choices=[-1, 1], default=None)
    p.add_argument("--exwl", type=float, default=639.0)
    p.add_argument("--emwl", type=float, default=665.0)
    p.add_argument("--na", type=float, default=1.49)

    # Figure
    p.add_argument("--clip_percentile", type=float, default=100.0)
    p.add_argument("--psf_per_tile_norm", action="store_true")

    args = p.parse_args()
    if args.skip_blind and args.skip_fixed:
        p.error("Cannot skip both DIP methods")
    return args



def _theory_path(work):
    path = work / "theory_mid.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run --stage theory first.")
    return path


# ---------------------------------------------------------------------------
def stage_theory(args, work):
    run_theory_and_mid(
        args.path_data, str(work / "theory_mid.npz"), crop_size=args.crop_size, 
        # na=args.na, exwl=args.exwl, emwl=args.emwl,  mask_sampl=args.mask_sampl, 
        num_iter=args.mid_num_iter,  z_in_idx=args.z_in_idx, mid_planes= args.mid_planes,
    )


def stage_dip(args, work):
    with np.load(_theory_path(work), allow_pickle=False) as th:
        dset = th["dset"]
        psf = th["psf_ism"][int(th["z_in_idx"])]
        pxsizex_nm = float(th["pxsizex_nm"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y, ref = hwc_to_stack(dset, device)   , hwc_to_stack(psf, device)
    print(f"[dip] device={device}, y={tuple(y.shape)}, PSFs={tuple(ref.shape)}, total PSF mass={ref.sum().item():.6g}")

    cfg = DIPConfig(
        num_iter=args.num_iter, lr_x=args.lr_x, lr_k=args.lr_k, smooth_k=args.smooth_k,
        reg_noise_std=args.reg_noise_std, seed=args.seed, pretraining=args.pretraining, 
        pretraining_iter=args.pretraining_iter, pretraining_lr=args.pretraining_lr, pret_psf=args.pret_psf, 
        print_every=args.print_every, save_every=args.save_every, 
        magnification=args.magnification, rotation_deg=args.rotation_deg, mirroring=args.mirroring, 
        estimate_grid=args.estimate_grid, exwl=args.exwl, emwl=args.emwl, na=args.na,
    )
    if not args.skip_blind:
        result = run_dip_blind(y, ref, pxsizex_nm, cfg, device)
        result["calibration"] = json.dumps(result["calibration"])
        np.savez_compressed(work / "dip_blind.npz", **result)

    if not args.skip_fixed:
        result = run_dip_fixed(y, ref, cfg, device)
        np.savez_compressed(work / "dip_fixed.npz", **result)


def stage_figure(args, work):
    with np.load(_theory_path(work), allow_pickle=False) as th:
        dset = th["dset"]
        z_in_idx = int(th["z_in_idx"])
        psf_theory = np.moveaxis(th["psf_ism"][z_in_idx], -1, 0)
        psf_theory = np.clip(psf_theory, 0, None)
        psf_theory /= psf_theory.sum() + 1e-12

        x_mid = extract_mid_image(th["mid_all"], args.mid_iter, 
            int(th["mid_z_idx"]), int(th["mid_n_planes"]), dset.shape[:2])

        mid_planes = str(th["mid_planes"])
        pxsizex_nm = float(th["pxsizex_nm"])

    with np.load(work / "dip_blind.npz", allow_pickle=False) as blind:
        x_blind = blind["x"]
        psf_blind = blind["kernels"]

    with np.load(work / "dip_fixed.npz", allow_pickle=False) as fixed:
        x_fixed = fixed["x"]

    mid_label = "MID (full PSF stack)" if mid_planes == "all" else "MID (single PSF plane)"

    for ext in ("pdf", "png"):
        make_figure(y_stack=dset, psf_theory=psf_theory, psf_blind=psf_blind,
            x_blind=x_blind, x_mid=x_mid, x_fixed=x_fixed, pxsizex_nm=pxsizex_nm,
            out_path=str(work / f"comparison.{ext}"), mid_label=mid_label,
            clip_percentile=args.clip_percentile, psf_per_tile_norm=args.psf_per_tile_norm,
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
