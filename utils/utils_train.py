import torch
import torch.nn as nn
import deepinv as dinv

try:
    from ISM.simulation.utils import partial_convolution
except ModuleNotFoundError:
    from brighteyes_ism.simulation.utils import partial_convolution

EPS = 1e-12

class PhysicsFixedKernels(nn.Module):
    def __init__(self, img_size, kernels):
        super().__init__()
        self.kernels = kernels
        self.img_size = img_size
        h, w = img_size
        kernels_for_blur = torch.flip(kernels, dims=(-2, -1))

        self.blur = dinv.physics.BlurFFT(
            img_size=(1, h, w),
            filter=kernels_for_blur,
            device=kernels.device,
        )

    def forward(self, x):
        return self.blur(x.repeat(self.kernels.shape[0], 1, 1, 1))


class PhysicsTwoPSF(nn.Module):
    def __init__(self, img_size, pinholes):
        super().__init__()
        h,w = img_size
        self.img_size = (1, h, w)
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
        kernels = kernels / (kernels.sum() + EPS)
        self.kernels = kernels

        kernels_for_blur = torch.flip(kernels, dims=(-2, -1))

        self.blur = dinv.physics.BlurFFT(
            img_size= self.img_size,
            filter=kernels_for_blur,
            device=kernels.device,
        )

    def forward(self, x):
        if self.kernels is None or self.blur is None:
            raise RuntimeError("Latent PSFs have not been set.")
        return self.blur(x.repeat(self.kernels.shape[0], 1, 1, 1))



class PoissonNLL(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        lam = pred.clamp_min(self.eps)
        return (lam - target * torch.log(lam)).mean()