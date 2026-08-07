import torch
import torch.nn.functional as F
import numpy as np

try:
    from pytorch_msssim import ssim as ssim_fn
    HAS_PYTORCH_MSSSIM = True
except ImportError:
    HAS_PYTORCH_MSSSIM = False

def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR) between prediction and ground truth.
    Evaluated in FP32 precision.
    Args:
        pred: Tensor of shape (B, C, H, W)
        target: Tensor of shape (B, C, H, W)
        data_range: Dynamic range of pixel values (default 1.0 for GT)
    Returns:
        float: Average PSNR across batch in dB.
    """
    pred_f = pred.float()
    target_f = target.float()

    mse = F.mse_loss(pred_f, target_f, reduction='none').mean(dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-10)
    psnr = 10.0 * torch.log10((data_range ** 2) / mse)
    return psnr.mean().item()


def calculate_ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """
    Calculates Structural Similarity Index (SSIM) between prediction and ground truth.
    Evaluated in FP32 precision.
    Args:
        pred: Tensor of shape (B, C, H, W)
        target: Tensor of shape (B, C, H, W)
        data_range: Dynamic range (1.0 for GT)
    Returns:
        float: Average SSIM value across batch [0, 1].
    """
    # Ensure float32 casting for evaluation calculation
    pred_f = pred.float()
    target_f = target.float()

    if HAS_PYTORCH_MSSSIM:
        ssim_val = ssim_fn(pred_f, target_f, data_range=data_range, size_average=True)
        return ssim_val.item()
    else:
        # Fallback SSIM calculation
        from losses.hybrid_loss import FallbackSSIM
        calc = FallbackSSIM(window_size=11, channel=1, data_range=data_range)
        return calc(pred_f, target_f).item()
