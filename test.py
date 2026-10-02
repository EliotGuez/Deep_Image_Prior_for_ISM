from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from figure_compare import mosaic

root = Path("runs/kernel_experiments")

for d in sorted(root.glob("lrk_*")):
    f = d / "dip_blind.npz"

    if not f.exists():
        continue

    data = np.load(f)
    kernels = data["kernels"]       # (25, 39, 39)

    fig, ax = plt.subplots(figsize=(7, 7))

    cmap = plt.get_cmap("hot").copy()
    cmap.set_bad("white")

    ax.imshow(
        mosaic(kernels, per_tile_norm=True),
        cmap=cmap,
        vmin=0,
    )

    ax.set_title("Recovered 25 PSFs - blind DIP")
    ax.axis("off")

    fig.tight_layout()
    out = d / "recovered_25_psf.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("saved:", out)
