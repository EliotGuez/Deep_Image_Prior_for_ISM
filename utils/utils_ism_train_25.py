import torch
import torch.nn as nn
import deepinv as dinv
import torch.nn.functional as F

class PhysicsISM(nn.Module):
    def __init__(self, img_size, device, bkg=0.0, kernels=None):
        super().__init__()
        self.img_size = img_size
        self.device = device
        self.bkg = bkg
        self.kernels = None
        if kernels is not None:
            self.set_kernels(kernels)

    def set_kernels(self, kernels):
        """
        kernels: tensor of shape (D,1,Kh,Kw)
        Do not detach: we want gradients to flow in the blind case.
        """
        self.kernels = kernels

    def forward(self, x):
        if self.kernels is None:
            raise ValueError("PhysicsISM: kernels are not set.")

        D = self.kernels.shape[0]
        x_rep = x.repeat(D, 1, 1, 1)

        blur = dinv.physics.BlurFFT(
            img_size=self.img_size,
            filter=self.kernels,
            device=self.device,
        )

        y = blur(x_rep)
        return y + self.bkg
        
# def _get_center_kernel(kernels, center_idx=12):
#     return kernels[center_idx, 0]   # (H,W)


# def _centered_coords(H, W, device, dtype):
#     x_coords = torch.arange(W, device=device, dtype=dtype) - (W - 1) / 2
#     y_coords = torch.arange(H, device=device, dtype=dtype) - (H - 1) / 2
#     Y, X = torch.meshgrid(y_coords, x_coords, indexing="ij")
#     return X, Y


# def compute_barycenter(kernel):
#     H, W = kernel.shape
#     X, Y = _centered_coords(H, W, kernel.device, kernel.dtype)

#     mass = kernel.sum() + 1e-12
#     mu_x = (kernel * X).sum() / mass
#     mu_y = (kernel * Y).sum() / mass

#     return mu_x, mu_y

# def _shift_kernel_grid_sample(k, dx_norm, dy_norm):
#     k4 = k.unsqueeze(0)
#     H, W = k.shape[-2], k.shape[-1]
#     base_grid = F.affine_grid(torch.eye(2, 3, device=k.device, dtype=k.dtype).unsqueeze(0),k4.shape,align_corners=True)                                             # (1, H, W, 2)
#     shift = torch.tensor([dx_norm, dy_norm], device=k.device, dtype=k.dtype).view(1, 1, 1, 2)
#     shifted_grid = base_grid - shift
#     k_shifted = F.grid_sample(k4, shifted_grid,mode="bilinear",padding_mode="zeros",align_corners=True,).squeeze(0)
#     return k_shifted / (k_shifted.sum() + 1e-12)


# def recenter_kernels(kernels, center_idx=12, alpha=1.0):
#     D, _, Kh, Kw = kernels.shape
#     with torch.no_grad():
#         k_ref = kernels[center_idx, 0].detach()          # (Kh, Kw)
#         mu_x, mu_y = compute_barycenter(k_ref) # We want to shift content by (-mu_x * alpha, -mu_y * alpha)        
#         shift_x = -alpha * mu_x   # pixels
#         shift_y = -alpha * mu_y

#     if abs(shift_x) < 0.05 and abs(shift_y) < 0.05: return kernels
        

#     # Convert pixel shift to grid_sample normalised coords            #
#     dx_norm = shift_x * 2.0 / max(Kw - 1, 1)
#     dy_norm = shift_y * 2.0 / max(Kh - 1, 1)
    
#     # apply same shift to each kernel
#     shifted = [_shift_kernel_grid_sample(kernels[d], dx_norm, dy_norm) for d in range(D)]
#     return torch.stack(shifted, dim=0)

# def schedule_alpha(step, warmup_iters=200, ramp_iters=300, max_alpha=1.0):
#     if step <= warmup_iters:    
#         return 0.0
#     t = min((step - warmup_iters) / max(ramp_iters, 1), 1.0)
#     return max_alpha * t

def save_kernel_with_markers(kernel, save_path):
    """
    Save one kernel with:
    - argmax in cyan
    - barycenter in lime
    """
    import matplotlib.pyplot as plt

    if kernel.dim() == 3:
        kernel = kernel[0]

    k = kernel.detach().cpu()
    H, W = k.shape

    # argmax
    idx = torch.argmax(k)
    iy = (idx // W).item()
    ix = (idx % W).item()

    # barycenter
    mass = k.sum() + 1e-12
    x_coords = torch.arange(W, dtype=k.dtype) - (W - 1) / 2
    y_coords = torch.arange(H, dtype=k.dtype) - (H - 1) / 2
    Y, X = torch.meshgrid(y_coords, x_coords, indexing="ij")
    mu_x = ((k * X).sum() / mass).item()
    mu_y = ((k * Y).sum() / mass).item()
    mu_x_img = mu_x + (W - 1) / 2
    mu_y_img = mu_y + (H - 1) / 2

    plt.figure(figsize=(4, 4))
    plt.imshow(k.numpy(), cmap="hot")
    plt.scatter(ix, iy, marker="x", s=80, label="argmax")
    plt.scatter(mu_x_img, mu_y_img, marker="+", s=120, label="barycenter")
    plt.title("Center kernel k12")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()




# def barycenter_loss(kernels, center_idx=12):
#     mu_x, mu_y = compute_barycenter(_get_center_kernel(kernels, center_idx))
#     return mu_x.pow(2) + mu_y.pow(2)


# def symmetry_loss(kernels):
#     kernel = _get_center_kernel(kernels)

#     loss_90 = F.mse_loss(kernel, torch.rot90(kernel, k=1, dims=[0, 1]))
#     loss_180 = F.mse_loss(kernel, torch.rot90(kernel, k=2, dims=[0, 1]))
#     loss_270 = F.mse_loss(kernel, torch.rot90(kernel, k=3, dims=[0, 1]))

#     return loss_90 + loss_180 + loss_270


# def smoothness_loss(kernels):
#     kernel = _get_center_kernel(kernels)

#     dx = kernel[:, 1:] - kernel[:, :-1]
#     dy = kernel[1:, :] - kernel[:-1, :]

#     return dx.pow(2).mean() + dy.pow(2).mean()


# def tv_loss(x, eps=1e-8):
#     """
#     Isotropic total variation for x of shape (B,C,H,W).
#     """
#     dx = x[:, :, :, 1:] - x[:, :, :, :-1]
#     dy = x[:, :, 1:, :] - x[:, :, :-1, :]

#     dx = dx[:, :, :-1, :]
#     dy = dy[:, :, :, :-1]

#     return torch.sqrt(dx.pow(2) + dy.pow(2) + eps).mean()

# def shift_kernel_pixels(k, dx, dy):
#     """
#     k: (1,1,H,W)
#     dx, dy: shift in pixels
#     """
#     _, _, H, W = k.shape

#     theta = torch.eye(2, 3, device=k.device, dtype=k.dtype).unsqueeze(0)
#     grid = F.affine_grid(theta, k.shape, align_corners=True)

#     dx_norm = 2.0 * dx / max(W - 1, 1)
#     dy_norm = 2.0 * dy / max(H - 1, 1)

#     shift = torch.stack([dx_norm, dy_norm]).view(1, 1, 1, 2)
#     shifted_grid = grid - shift

#     out = F.grid_sample(
#         k,
#         shifted_grid,
#         mode="bilinear",
#         padding_mode="zeros",
#         align_corners=True,
#     )

#     return out / (out.sum() + 1e-12)
    