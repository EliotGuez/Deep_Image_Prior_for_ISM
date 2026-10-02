import torch
import torch.nn as nn
import torch.nn.functional as F
EPS = 1e-12

class SingleKernelFCN(nn.Module):
    def __init__(self, latent_dim: int, kernel_pixels: int, hidden: int = 400):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU6(),
            nn.Linear(hidden, kernel_pixels),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        k = F.softplus(self.net(z))
        return k / (k.sum(dim=1, keepdim=True) + EPS)


class Independent25FCN(nn.Module):
    def __init__(self, latent_dim: int, kernel_pixels: int, hidden: int = 1000, n_det: int = 25):
        super().__init__()
        self.n_det = n_det
        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, hidden),
                nn.ReLU6(),
                nn.Linear(hidden, kernel_pixels),
            )
            for _ in range(n_det)
        ])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = torch.cat([net(z[d:d + 1]) for d, net in enumerate(self.nets)], dim=0)
        out = F.softplus(out)
        return out / (out.sum(dim=1, keepdim=True) + EPS)