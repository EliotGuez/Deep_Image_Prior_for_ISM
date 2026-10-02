from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-12


def mosaic(stack, gap=1, per_tile_norm=True):
    """(D, Kh, Kw) -> one (n*Kh + gaps, n*Kw + gaps) image; NaN in the gaps."""
    d, kh, kw = stack.shape
    n = int(round(np.sqrt(d)))
    canvas = np.full((n * kh + (n - 1) * gap, n * kw + (n - 1) * gap), np.nan, dtype=np.float32)
    for i in range(d):
        r, c = divmod(i, n)
        tile = stack[i].astype(np.float32)
        if per_tile_norm:
            tile = tile / (tile.max() + EPS)
        canvas[r * (kh + gap): r * (kh + gap) + kh, c * (kw + gap): c * (kw + gap) + kw] = tile
    return canvas


def _scale_bar(ax, shape, pxsizex_nm, length_um=10.0):
    h, w = shape
    length_px = length_um * 1e3 / pxsizex_nm
    x1, y = 0.96 * w, 0.06 * h
    ax.plot([x1 - length_px, x1], [y, y], color="white", lw=3, solid_capstyle="butt")
    ax.text(x1 - length_px / 2, y + 0.035 * h, f"{length_um:g} µm", color="white", ha="center", va="top", fontsize=9)


def _show_image(ax, fig, img, title, pxsizex_nm, clip_percentile=100.0, cbar_label="a.u."):
    vmax = np.percentile(img, clip_percentile) if clip_percentile < 100 else img.max()
    im = ax.imshow(img, cmap="hot", vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    _scale_bar(ax, img.shape, pxsizex_nm)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(cbar_label, fontsize=9)


def _show_psfs(ax, stack, title, per_tile_norm):
    cmap = plt.get_cmap("hot").copy()
    cmap.set_bad("white")
    ax.imshow(mosaic(stack, per_tile_norm=per_tile_norm), cmap=cmap, vmin=0,
              vmax=None if per_tile_norm else np.nanmax(stack))
    ax.set_title(title, fontsize=12)
    ax.axis("off")


def make_figure(y_stack, psf_theory, psf_blind, x_blind, x_mid, x_fixed,
                pxsizex_nm, out_path, mid_label="MID", clip_percentile=100.0,
                psf_per_tile_norm=True):
    """
    y_stack     : (H, W, 25) observation (confocal-like image = sum over detectors)
    psf_theory  : (25, Kh, Kw) theoretical PSFs
    psf_blind   : (25, Kh, Kw) PSFs recovered by the blind method
    x_blind, x_mid, x_fixed : (H, W) reconstructions
    """
    y_conf = y_stack.sum(axis=-1)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5))
    _show_image(axes[0, 0], fig, y_conf, "Observation y (sum of 25 detectors)",
                pxsizex_nm, clip_percentile, "Counts")
    _show_psfs(axes[0, 1], psf_theory, "Theoretical PSFs", psf_per_tile_norm)
    _show_psfs(axes[0, 2], psf_blind, "Recovered PSFs (blind DIP)", psf_per_tile_norm)
    _show_image(axes[1, 0], fig, x_blind, "Blind DIP (ours)", pxsizex_nm, clip_percentile)
    _show_image(axes[1, 1], fig, x_mid, mid_label, pxsizex_nm, clip_percentile, "Counts")
    _show_image(axes[1, 2], fig, x_fixed, "DIP, fixed theoretical PSFs", pxsizex_nm, clip_percentile)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] saved {out_path}")
