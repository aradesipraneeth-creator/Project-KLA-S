import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, Any, Tuple

from utils.edge_utils import compute_sobel_edges

try:
    from pytorch_msssim import ssim as ssim_fn
    HAS_PYTORCH_MSSSIM = True
except ImportError:
    HAS_PYTORCH_MSSSIM = False

class HighFrequencyLoss(nn.Module):
    def __init__(self, kernel_size: int = 5, sigma: float = 1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.register_buffer("kernel", self._create_gaussian_kernel(kernel_size, sigma))

    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        g1d = torch.exp(-coords**2 / (2 * sigma**2))
        g1d = g1d / g1d.sum()
        g2d = g1d.unsqueeze(1) @ g1d.unsqueeze(0)
        return g2d.view(1, 1, kernel_size, kernel_size)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p_f = pred.float()
        t_f = target.float()
        kernel = self.kernel.to(device=p_f.device, dtype=torch.float32)
        blur_p = F.conv2d(p_f, kernel, padding=self.kernel_size // 2)
        blur_t = F.conv2d(t_f, kernel, padding=self.kernel_size // 2)
        hf_p = p_f - blur_p
        hf_t = t_f - blur_t
        return F.l1_loss(hf_p, hf_t)

class AIRNetV3AdaptiveLoss(nn.Module):
    """
    Adaptive Multi-Objective Loss for AIR-Net v3 (Section 27-33):
    Dynamically scales loss weights based on router category probabilities [r_edge, r_texture, r_noise, r_smooth, r_sparse]:
      - EDGE: Increases edge & gradient loss weights.
      - TEXTURE: Increases high-frequency & Laplacian loss weights.
      - NOISE: Decreases high-frequency weighting to prevent sharpening noise.
      - SMOOTH: Increases L1 pixel fidelity & SSIM weights.
      - SPARSE: Increases L1 & edge loss weights for small defect preservation.
    """
    def __init__(self, data_range: float = 1.0):
        super().__init__()
        self.data_range = data_range
        self.l1_loss = nn.L1Loss()
        self.hf_loss_fn = HighFrequencyLoss(kernel_size=5, sigma=1.0)

    def forward(
        self,
        pred_dict: Dict[str, Any],
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        restored_pred = pred_dict["restored"]
        edge_pred = pred_dict.get("edge", None)
        routing_probs = pred_dict["routing_probs"]  # (B, 5)

        p_f = restored_pred.float()
        t_f = target.float()

        # Mean routing probabilities across batch
        mean_r = torch.mean(routing_probs, dim=0)  # [edge, texture, noise, smooth, sparse]
        r_edge = float(mean_r[0].item())
        r_texture = float(mean_r[1].item())
        r_noise = float(mean_r[2].item())
        r_smooth = float(mean_r[3].item())
        r_sparse = float(mean_r[4].item())

        # Dynamic Loss Weight Profiles (Bounded & Normalized)
        w_l1 = 0.65 + 0.15 * r_smooth + 0.10 * r_sparse
        w_ssim = 0.20 + 0.05 * r_edge + 0.05 * r_smooth
        w_edge = 0.05 + 0.10 * r_edge + 0.05 * r_sparse
        w_hf = max(0.01, 0.05 + 0.10 * r_texture - 0.04 * r_noise)

        # Normalize weights so sum = 1.0
        total_w = w_l1 + w_ssim + w_edge + w_hf
        w_l1 /= total_w
        w_ssim /= total_w
        w_edge /= total_w
        w_hf /= total_w

        # Compute Losses in Float32
        loss_l1 = self.l1_loss(p_f, t_f)

        if HAS_PYTORCH_MSSSIM:
            ssim_val = ssim_fn(p_f, t_f, data_range=self.data_range, size_average=True)
        else:
            from losses.hybrid_loss import FallbackSSIM
            calc = FallbackSSIM(window_size=11, channel=1, data_range=self.data_range)
            ssim_val = calc(p_f, t_f)
        loss_ssim = 1.0 - ssim_val

        if edge_pred is not None:
            gt_edges = compute_sobel_edges(t_f)
            loss_edge = self.l1_loss(edge_pred.float(), gt_edges.float())
        else:
            loss_edge = torch.tensor(0.0, device=target.device)

        loss_hf = self.hf_loss_fn(p_f, t_f)

        total_loss = (
            w_l1 * loss_l1 +
            w_ssim * loss_ssim +
            w_edge * loss_edge +
            w_hf * loss_hf
        )

        return total_loss, {
            "l1": loss_l1.item(),
            "ssim_loss": loss_ssim.item(),
            "edge": loss_edge.item(),
            "hf": loss_hf.item(),
            "w_l1": w_l1,
            "w_ssim": w_ssim,
            "w_edge": w_edge,
            "w_hf": w_hf
        }
