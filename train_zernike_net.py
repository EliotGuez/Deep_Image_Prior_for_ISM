import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from zernike_net.dataset import ZernikeDataset
from zernike_net.model   import ZernikeNet

def train(hparams):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float32

    save_path = Path(hparams.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    dataset = ZernikeDataset(Kh=hparams.Kh, seed=hparams.seed)
    loader = DataLoader(dataset, batch_size=hparams.batch_size, num_workers=hparams.num_workers, pin_memory=True)

    model = ZernikeNet(base_ch=hparams.base_ch).to(device=device, dtype=dtype)
    print(f"Device     : {device}")
    print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    running   = []

    model.train()
    for step, (inp_ex, inp_em, tgt_ex, tgt_em) in enumerate(loader):
        if step >= hparams.num_steps:
            break

        inp_ex = inp_ex.to(device=device, dtype=dtype)
        inp_em = inp_em.to(device=device, dtype=dtype)
        tgt_ex = tgt_ex.to(device=device, dtype=dtype)
        tgt_em = tgt_em.to(device=device, dtype=dtype)

        pred_ex, pred_em = model(inp_ex, inp_em)
        loss = loss_fn(pred_ex, tgt_ex) + loss_fn(pred_em, tgt_em)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running.append(loss.item())

        if step % hparams.log_every == 0:
            avg = float(np.mean(running[-hparams.log_every:]))
            print(f"step={step:06d} | loss={avg:.6f}")

            if avg < best_loss:
                best_loss = avg

    torch.save({"step": hparams.num_steps, "model": model.state_dict()},save_path / "last.pth")
    print(f"\nTraining done. Best loss : {best_loss:.6f}")
    print(f"Checkpoints saved to     : {save_path}")

def save_eval_examples(inp_ex, inp_em, tgt_ex, tgt_em, pred_ex, pred_em, save_dir: Path, n_examples=4):
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)
    n = min(n_examples, inp_ex.shape[0])

    def save_one_psf(inp, path, title_suffix):
        clean = inp[0].cpu().numpy()
        noisy = inp[1].cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
        vmax = max(clean.max(), noisy.max())

        axes[0].imshow(clean, cmap="hot", vmin=0, vmax=vmax)
        axes[0].set_title(f"Clean PSF {title_suffix}")
        axes[0].axis("off")

        axes[1].imshow(noisy, cmap="hot", vmin=0, vmax=vmax)
        axes[1].set_title(f"Noisy aberrated PSF {title_suffix}")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def save_one_zernike(gt, pred, path, title_suffix):
        gt_np = gt.cpu().numpy()
        pr_np = pred.cpu().numpy()

        x = np.arange(len(gt_np))
        width = 0.4

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(x - width / 2, gt_np, width, label="Ground truth", color="steelblue", alpha=0.8)
        ax.bar(x + width / 2, pr_np, width, label="Predicted", color="darkorange", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([f"Z{j+1}" for j in x], rotation=45, ha="right")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Amplitude (rad)")
        ax.set_title(f"Zernike coefficients — GT vs predicted {title_suffix}")
        ax.legend()
        ax.grid(axis="y", alpha=0.4)

        mae = np.mean(np.abs(gt_np - pr_np))
        ax.set_xlabel(f"MAE = {mae:.4f} rad")

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    for i in range(n):
        save_one_psf(inp_ex[i], save_dir / f"excitation_psf_{i:02d}.png", "(excitation)")
        save_one_zernike(tgt_ex[i], pred_ex[i], save_dir / f"zernike_exc_{i:02d}.png", "(excitation)")
        save_one_psf(inp_em[i], save_dir / f"emission_psf_{i:02d}.png", "(emission)")
        save_one_zernike(tgt_em[i], pred_em[i], save_dir / f"zernike_em_{i:02d}.png", "(emission)")


def evaluate(hparams):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float32
 
    assert hparams.checkpoint is not None, "Provide --checkpoint for eval mode."
    ckpt  = torch.load(hparams.checkpoint, map_location=device)
    model = ZernikeNet(base_ch=hparams.base_ch).to(device=device, dtype=dtype)
    model.load_state_dict(ckpt["model"])
    model.eval()
 
    save_dir = Path(hparams.checkpoint).parent / "eval"
    save_dir.mkdir(parents=True, exist_ok=True)
 
    dataset = ZernikeDataset(Kh=hparams.Kh, seed=hparams.seed + 9999)
    
    loader = DataLoader(dataset, batch_size=hparams.batch_size, num_workers=2, pin_memory=True)
 
    mae_ex_list, mae_em_list = [], []
    examples_saved = False
 
    with torch.no_grad():
        for step, (inp_ex, inp_em, tgt_ex, tgt_em) in enumerate(loader):
            if step >= hparams.eval_steps:
                break
            inp_ex = inp_ex.to(device=device, dtype=dtype)
            inp_em = inp_em.to(device=device, dtype=dtype)
            tgt_ex = tgt_ex.to(device=device, dtype=dtype)
            tgt_em = tgt_em.to(device=device, dtype=dtype)
 
            pred_ex, pred_em = model(inp_ex, inp_em)
            mae_ex_list.append(torch.mean(torch.abs(pred_ex - tgt_ex)).item())
            mae_em_list.append(torch.mean(torch.abs(pred_em - tgt_em)).item())
 
            # Save visual examples from the first batch only
            if not examples_saved:
                save_eval_examples(
                    inp_ex, inp_em, tgt_ex, tgt_em, pred_ex, pred_em,
                    save_dir=save_dir, n_examples=hparams.n_examples,
                )
                examples_saved = True
                print(f"Examples saved to {save_dir}/example_XX/")
 
    print(f"MAE excitation : {np.mean(mae_ex_list):.6f}")
    print(f"MAE emission   : {np.mean(mae_em_list):.6f}")
 

def parse_args():
    p = argparse.ArgumentParser(description="Train/eval ZernikeNet regressor")

    p.add_argument("--mode",        type=str,   default="train", choices=["train", "eval"])
    p.add_argument("--checkpoint",  type=str,   default=None)
    p.add_argument("--save_path",   type=str,   default="zk_net/")
    # Dataset
    p.add_argument("--Kh",          type=int,   default=99)
    p.add_argument("--coeff_scale_min", type=float, default=0.1)
    p.add_argument("--coeff_scale_max", type=float, default=0.9)
    p.add_argument("--na_min",      type=float, default=0.9)
    p.add_argument("--na_max",      type=float, default=1.4)
    p.add_argument("--gain_min",    type=float, default=0.05)
    p.add_argument("--gain_max",    type=float, default=0.5)
    # Training
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--num_steps",   type=int,   default=50_000)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--base_ch",     type=int,   default=32)
    p.add_argument("--log_every",   type=int,   default=500)
    p.add_argument("--seed",        type=int,   default=0)
    # Eval
    p.add_argument("--eval_steps",  type=int,   default=100)
    p.add_argument("--n_examples",  type=int,   default=4, help="Number of visual examples to save during eval")
    return p.parse_args()


if __name__ == "__main__":
    hparams = parse_args()
    if hparams.mode == "train":
        train(hparams)
    else:
        evaluate(hparams)