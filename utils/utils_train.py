from typing import Optional, Tuple

import torch
import torch.nn as nn
import deepinv as dinv

from ISM.simulation.utils import partial_convolution
EPS = 1e-12

class Physics25(nn.Module):
    def __init__(self, img_size, device):
        super().__init__()
        self.img_size = img_size
        self.device = device
        self.kernels= None

    def set_kernels(self, kernels):
        self.kernels = kernels

    def forward(self, x):
        if self.kernels is None:
            raise RuntimeError("Kernels have not been set.")
        d = self.kernels.shape[0]
        blur = dinv.physics.BlurFFT(
            img_size=self.img_size,
            filter=self.kernels,
            device=self.device,
        )
        return blur(x.repeat(d, 1, 1, 1))


class PhysicsTwoPSF(nn.Module):
    def __init__(self, img_size, device, pinholes):
        super().__init__()
        self.img_size = img_size
        self.device = device
        self.pinholes = pinholes
        self.kernels = None
        self.blur = None

    def set_psfs(self, h_exc, h_em):
        h_exc = h_exc / (h_exc.sum() + EPS)
        h_em = h_em / (h_em.sum() + EPS)

        # h_em: (1,1,Kh,Kw) -> (1,Kh,Kw), interpreted as zxy by ISM.
        em_3d = h_em.squeeze(1)
        h_det = partial_convolution(em_3d, self.pinholes, "zxy", "xyc", "xy")
        h_det = h_det.squeeze(0)  # (Kh,Kw,25)

        exc_2d = h_exc.squeeze()
        kernels = torch.einsum("xyc,xy->cxy", h_det, exc_2d).unsqueeze(1)
        kernels = kernels.clamp_min(0)
        # kernels = kernels / (kernels.sum(dim=(-2, -1), keepdim=True) + EPS)
        # kernel_mass = kernels.sum(dim=(-2, -1), keepdim=True)
        # kernels = kernels / (kernel_mass.mean() + EPS)
        self.kernels = kernels
        mass = kernels.sum(dim=(-2, -1))
        print("kernel masses =", mass.flatten())
        print("mean kernel mass =", mass.mean())
        
        self.blur = dinv.physics.BlurFFT(
            img_size=self.img_size,
            filter=kernels,
            device=self.device,
        )

    def forward(self, x):
        if self.kernels is None or self.blur is None:
            raise RuntimeError("Latent PSFs have not been set.")
        d = self.kernels.shape[0]
        return self.blur(x.repeat(d, 1, 1, 1))



class PoissonNLL(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        lam = pred.clamp_min(self.eps)
        return (lam - target * torch.log(lam)).mean()