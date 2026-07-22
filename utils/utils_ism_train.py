import torch
import torch.nn as nn
import deepinv as dinv

import ISM.simulation.PSF_sim as ism
from ISM.simulation.utils import partial_convolution


# ---------------------------------------------------------------------------
# Factorised forward physics model
# ---------------------------------------------------------------------------

class PhysicsISMFactorized(nn.Module):
    def __init__(self, img_size, device, bkg=0.0001, pinholes=None, eps=1e-12):
        super().__init__()
        self.img_size = img_size
        self.device = device
        self.bkg = bkg
        self.eps = eps
        self.pinholes = pinholes
        self.kernels = None
        self._blur = None

    def set_psfs(self, h_exc: torch.Tensor, h_em: torch.Tensor) -> None:
        h_exc = h_exc / (h_exc.sum() + self.eps)
        h_em  = h_em  / (h_em.sum()  + self.eps)

        em_3d  = h_em.squeeze(1)          # (1, Kh, Kw)
        h_det_all = partial_convolution(em_3d, self.pinholes, 'zxy', 'xyc', 'xy')
        h_det_all = h_det_all.squeeze(0)     # (Kh, Kw, 25)

        exc_2d = h_exc.squeeze()       # (Kh, Kw)
        kernels_hwc  = torch.einsum('xyc,xy->xyc', h_det_all, exc_2d)  # (Kh, Kw, 25)
        kernels = kernels_hwc.permute(2, 0, 1).unsqueeze(1)        # (25, 1, Kh, Kw)
        kernels = kernels.clamp(min=0)
        kernels = kernels / (kernels.sum(dim=(-2, -1), keepdim=True) + self.eps)

        self.kernels = kernels
        self._blur   = dinv.physics.BlurFFT(img_size=self.img_size, filter=self.kernels, device=self.device)

    def forward(self, x):
        D = self.kernels.shape[0]
        return self._blur(x.repeat(D, 1, 1, 1)) + self.bkg


def build_pinholes_tensor(grid, device, dtype=torch.float32):
    pinholes = ism.custom_detector(grid, device)
    return pinholes.to(device=device, dtype=dtype)


def tv_loss_kernel(k: torch.Tensor):
    tv_h = torch.mean(torch.abs(k[:, :, 1:, :] - k[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(k[:, :, :, 1:] - k[:, :, :, :-1]))
    return tv_h + tv_w

def entropy_loss(k, eps=1e-12):
    return -torch.sum(k * torch.log(k + eps), dim=-1).mean()