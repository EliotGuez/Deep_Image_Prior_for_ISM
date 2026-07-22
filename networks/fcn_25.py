import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import *

class FCNKernelISM(nn.Module):
    def __init__(self,num_detectors=25, num_input_channels=200, num_output_channels=1, num_hidden=1000, eps=1e-12, norm = False, fingerprint=None):
        super().__init__()
        self.num_detectors = num_detectors
        self.eps = eps

        self.small_network = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_input_channels, num_hidden, bias=True),
                nn.ReLU6(),
                nn.Linear(num_hidden, num_output_channels),
            )
            for _ in range(num_detectors)
        ])

        self.softplus = nn.Softplus()
        self.norm = norm
        self.fingerprint= fingerprint

    def forward(self, z):
        outs = []
        for d in range(self.num_detectors):
            logits = self.small_network[d](z[d:d+1])
            outs.append(logits)

        kernels = torch.cat(outs, dim=0)   # (25, K)
        kernels = self.softplus(kernels)
        kernels = kernels / (kernels.sum(dim=1, keepdim=True) + self.eps)

        if self.norm:
            # kernels = kernels / (kernels.sum() + self.eps)
            kernels = kernels * (self.fingerprint.view(self.num_detectors, 1) + self.eps)
            kernels = kernels / (kernels.sum() + self.eps)
        return kernels

    
def fcn_ism(num_input_channels=200, num_output_channels=1, num_hidden=512, num_detectors=25, norm=False, fingerprint=None):
    return FCNKernelISM(
        num_detectors=num_detectors,
        num_input_channels=num_input_channels,
        num_output_channels=num_output_channels,
        num_hidden=num_hidden,
        norm = norm, 
        fingerprint=fingerprint
    )