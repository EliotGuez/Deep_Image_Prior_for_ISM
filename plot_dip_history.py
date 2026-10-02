from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Plot blind-DIP evolution saved in dip_blind.npz")
    p.add_argument("npz")
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.npz)
    out = Path(args.out_dir) if args.out_dir else src.parent
    out.mkdir(parents=True, exist_ok=True)

    d = np.load(src)
    required = ["snapshot_iters", "x_history", "h_exc_history", "h_em_history"]
    missing = [k for k in required if k not in d]
    if missing:
        raise KeyError(
            f"{src} does not contain snapshot history: missing {missing}. "
            "Run the modified blind DIP with --save_every first."
        )

    iters = d["snapshot_iters"]
    xs = d["x_history"]
    h_exc = d["h_exc_history"]
    h_em = d["h_em_history"]

    # Reconstruction evolution. Independent display scaling is intentional:
    # this plot is for seeing structure/noise evolution, not absolute intensity.
    n = len(iters)
    cols = min(6, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.0 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for j, (it, x) in enumerate(zip(iters, xs)):
        ax = axes.ravel()[j]
        vmax = np.percentile(x, 99.5)
        ax.imshow(x, cmap="hot", vmin=0, vmax=vmax)
        ax.set_title(f"iter {int(it)}")
        ax.axis("off")
    fig.suptitle("Blind DIP reconstruction through training")
    fig.tight_layout()
    fig.savefig(out / "history_reconstruction.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Latent PSFs. Use one common scale across all snapshots so broadening /
    # peaking through time is visible, not hidden by per-image normalization.
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.6), squeeze=False)

    for j, it in enumerate(iters):
        axes[0, j].imshow(h_exc[j], cmap="hot", vmin=0, vmax=h_exc[j].max(),)
        axes[1, j].imshow(h_em[j], cmap="hot", vmin=0, vmax=h_em[j].max(),)
        axes[0, j].set_title(f"iter {int(it)}")
        axes[0, j].axis("off")
        axes[1, j].axis("off")
        axes[0, j].set_xlabel("x")
    axes[0, 0].set_ylabel("excitation")
    axes[1, 0].set_ylabel("emission")
    fig.suptitle("Two latent PSFs through training")
    fig.tight_layout()
    fig.savefig(out / "history_latent_psfs.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Scalar diagnostics when available.
    if "data_losses" in d:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(np.arange(1, len(d["data_losses"]) + 1), d["data_losses"], label="data")
        if "smooth_losses" in d and np.any(d["smooth_losses"] != 0):
            ax.plot(np.arange(1, len(d["smooth_losses"]) + 1), d["smooth_losses"], label="weighted smoothness")
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss term")
        ax.set_title("Training loss terms")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "history_losses.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    if "gaussian_mse_history" in d or "theory_kernel_mse_history" in d:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        if "gaussian_mse_history" in d:
            ax.plot(iters, d["gaussian_mse_history"], marker="o", label="latent PSFs vs Gaussian")
        if "theory_kernel_mse_history" in d:
            ax.plot(iters, d["theory_kernel_mse_history"], marker="o", label="25 kernels vs theoretical")
        ax.set_xlabel("iteration")
        ax.set_ylabel("MSE")
        ax.set_title("PSF drift diagnostics")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "history_psf_drift.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved history plots to {out}")


if __name__ == "__main__":
    main()
