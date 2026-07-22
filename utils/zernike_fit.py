from __future__ import annotations

import sys
import warnings

import numpy as np
import torch
from scipy.stats import qmc

import ISM.simulation.PSF_sim as ism
from utils.utils_ism import _make_grid, _make_optical_params


FIXED_ZERNIKE = 1    # piston — always forced to zero
ZERNIKE_MAX   = 15   # total number of Zernike orders

def _simulate_single_psf(free_coeff,par,grid):

    full_coeff = torch.cat([torch.zeros(FIXED_ZERNIKE), torch.as_tensor(free_coeff, dtype=torch.float32),])
    par.abe_index = torch.arange(len(full_coeff))
    par.abe_ampli = full_coeff

    psf, _ = ism.singlePSF(par, grid.pxsizex, grid.Nx, [0, 0], grid.Nz, "cpu")
    psf = psf.squeeze()
    psf = psf / (psf.sum() + 1e-12)
    return psf


def _psf_loss(free_coeff, target, par, grid, loss,):
    estimate = _simulate_single_psf(free_coeff, par, grid)
    if loss == "mse":
        return torch.sum((target - estimate) ** 2)
    elif loss == "mae":
        return torch.sum(torch.abs(target - estimate))
    elif loss =="kl":
        eps = 1e-12
        p = target   / (target.sum()   + eps)
        q = estimate / (estimate.sum() + eps)
        return torch.sum(p * torch.log((p + eps) / (q + eps)))
    else:
        raise ValueError(f"Unknown loss '{loss}'. Choose 'kl', 'mse' or 'mae'.")


def _generate_starting_points(ndim, num_starts, bound, initialization):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        if initialization == "lhs":
            samples = qmc.LatinHypercube(d=ndim).random(n=num_starts - 1)
        elif initialization == "sobol":
            samples = qmc.Sobol(d=ndim).random(n=num_starts - 1)
        else:
            raise ValueError(f"Unknown initialization '{initialization}'. "
                             "Choose 'lhs' or 'sobol'.")

    lo = np.full(ndim, -bound)
    hi = np.full(ndim,  bound)
    scaled = qmc.scale(samples, lo, hi)
    scaled = np.concatenate([np.zeros((1, ndim)), scaled], axis=0)
    return torch.tensor(scaled, dtype=torch.float32)


def _minimize_single_psf(start, target, par, grid, max_iter, loss, verbose):
    history   = torch.zeros(max_iter)
    x0        = start.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x0], lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20)

    for step in range(max_iter):
        optimizer.zero_grad()
        lv = _psf_loss(x0, target, par, grid, loss)
        history[step] = lv.item()
        lv.backward()
        optimizer.step()
        scheduler.step(lv)

        if verbose:
            sys.stdout.write(f"\r  iter {step + 1:4d}/{max_iter}  loss={lv.item():.3E}")
            sys.stdout.flush()

    return x0.detach(), history

def _fit_single_psf(target, par, grid, zer_max, num_starts, max_iter, loss, bound, initialization, verbose):
    
    n_free       = zer_max - FIXED_ZERNIKE
    start_points = _generate_starting_points(n_free, num_starts, bound, initialization)
    all_results  = torch.zeros(num_starts, n_free)
    all_history  = torch.zeros(num_starts, max_iter)

    for i, start in enumerate(start_points):
        if verbose:
            print(f"\n  start {i + 1}/{num_starts}")
        x0, h0 = _minimize_single_psf(start, target, par, grid, max_iter, loss, verbose)
        all_results[i] = x0
        all_history[i] = h0

    best_idx   = torch.argmin(all_history[:, -1])
    best_free  = all_results[best_idx]
    full_coeff = torch.cat([torch.zeros(FIXED_ZERNIKE), best_free.cpu()])
    return full_coeff, all_history

def fit_zernike_from_two_psfs(h_exc, h_em, zer_max = ZERNIKE_MAX, num_starts = 5, max_iter = 200, loss = "mae", bound = 2.0, initialization= "sobol", verbose= True, Kh= 99):

    def _prep(h: torch.Tensor) -> torch.Tensor:
        h = h.detach().cpu()
        while h.dim() > 2:
            h = h.squeeze(0)
        return h / (h.sum() + 1e-12)

    target_exc = _prep(h_exc)
    target_em  = _prep(h_em)

    grid   = _make_grid(Kh)
    ex_par = _make_optical_params(wl=450)
    em_par = _make_optical_params(wl=660)

    if verbose:
        print("\n=== Fitting excitation PSF (λ=450 nm) ===")
    zer_ex, hist_ex = _fit_single_psf(target_exc, ex_par, grid, zer_max, num_starts, max_iter, loss, bound, initialization, verbose)

    if verbose:
        print("\n\n=== Fitting emission PSF (λ=660 nm) ===")
    zer_em, hist_em = _fit_single_psf(target_em, em_par, grid, zer_max, num_starts, max_iter, loss, bound, initialization, verbose)

    return zer_ex, zer_em, hist_ex, hist_em
