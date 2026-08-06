import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, Any

from utils.edge_utils import compute_sobel_edges

try:
    from pytorch_msssim import ssim as ssim_fn
    HAS_PYTORCH_MSSSIM = True
except ImportError:
    HAS_PYTORCH_MSSSIM = False


class AIRNetHybridLoss(nn.Module):
    """
    Hybrid Loss function for AIR-Net v1:
        Loss = 0.60 * L1 + 0.25 * (1.0 - SSIM) + 0.15 * EdgeLoss
    
    Optional LPIPS flag available (use_lpips=False by default).
    """
    def __init__(
        self,
        l1_weight: float = 0.60,
        ssim_weight: float = 0.25,
        edge_weight: float = 0.15,
        lpips_weight: float = 0.0,
        use_lpips: bool = False,
        data_range: float = 1.0
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.lpips_weight = lpips_weight
        self.use_lpips = use_lpips
        self.data_range = data_range

        self.l1_loss = nn.L1Loss()
        
        if use_lpips:
            try:
                import lpips
                self.lpips_fn = lpips.LPIPS(net='alex')
            except Exception:
                self.use_lpips = False
                self.lpips_fn = None
        else:
            self.lpips_fn = None

    def forward(
        self,
        pred: Union[torch.Tensor, Dict[str, Any]],
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: Either a tensor (B, 1, 256, 256) or dict from AIRNet containing:
                  "restored": (B, 1, 256, 256)
                  "edge": (B, 1, 256, 256)
            target: GT image tensor of shape (B, 1, 256, 256)
        Returns:
            torch.Tensor: Scalar weighted total loss
        """
        if isinstance(pred, dict):
            restored_pred = pred["restored"]
            edge_pred = pred.get("edge", None)
        else:
            restored_pred = pred
            edge_pred = None

        # 1. L1 Pixel Fidelity Loss
        loss_l1 = self.l1_loss(restored_pred, target)

        # 2. SSIM Structural Loss
        if HAS_PYTORCH_MSSSIM:
            ssim_val = ssim_fn(restored_pred, target, data_range=self.data_range, size_average=True)
        else:
            # Basic Fallback SSIM calculation
            from losses.hybrid_loss import FallbackSSIM
            calc = FallbackSSIM(window_size=11, channel=1, data_range=self.data_range).to(target.device)
            ssim_val = calc(restored_pred, target)
            
        loss_ssim = 1.0 - ssim_val

        # 3. Edge Reconstruction Loss
        if edge_pred is not None:
            gt_edges = compute_sobel_edges(target)
            loss_edge = self.l1_loss(edge_pred, gt_edges)
        else:
            loss_edge = torch.tensor(0.0, device=target.device)

        # 4. Optional LPIPS Perceptual Loss
        loss_lpips = torch.tensor(0.0, device=target.device)
        if self.use_lpips and self.lpips_fn is not None:
            # Map 1-channel grayscale to 3-channel for LPIPS
            restored_3ch = restored_pred.repeat(1, 3, 1, 1) * 2.0 - 1.0
            target_3ch = target.repeat(1, 3, 1, 1) * 2.0 - 1.0
            loss_lpips = self.lpips_fn(restored_3ch, target_3ch).mean()

        total_loss = (
            self.l1_weight * loss_l1 +
            self.ssim_weight * loss_ssim +
            self.edge_weight * loss_edge +
            self.lpips_weight * loss_lpips
        )
        return total_loss

# Alias for backward compatibility
HybridLoss = AIRNetHybridLoss


class FallbackSSIM(nn.Module):
    """Fallback SSIM module when pytorch-msssim is unavailable."""
    def __init__(self, window_size: int = 11, channel: int = 1, data_range: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.data_range = data_range
        self.register_buffer("window", self.create_window(window_size, channel))

    def gaussian(self, window_size: int, sigma: float):
        gauss = torch.exp(torch.tensor([-(x - window_size // 2) ** 2 / float(2 * sigma ** 2) for x in range(window_size)]))
        return gauss / gauss.sum()

    def create_window(self, window_size: int, channel: int):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D_window.expand(channel, 1, window_size, window_size).contiguous()

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2

        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return ssim_map.mean()
