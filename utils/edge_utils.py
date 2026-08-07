import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelEdgeDetector(nn.Module):
    """
    PyTorch Sobel Edge Detector for GT Edge Target Generation.
    Computes normalized Sobel gradient magnitude maps in [0, 1].
    """
    def __init__(self):
        super().__init__()
        # Sobel Horizontal Kernel (Gx)
        sobel_x = torch.tensor([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        # Sobel Vertical Kernel (Gy)
        sobel_y = torch.tensor([
            [-1.0, -2.0, -1.0],
            [ 0.0,  0.0,  0.0],
            [ 1.0,  2.0,  1.0]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Tensor of shape (B, 1, H, W) or (1, H, W) in range [0, 1]
        Returns:
            torch.Tensor: Normalized Sobel edge map of shape (B, 1, H, W) in range [0, 1]
        """
        if img.ndim == 3:
            img = img.unsqueeze(0)

        sobel_x = self.sobel_x.to(device=img.device, dtype=img.dtype)
        sobel_y = self.sobel_y.to(device=img.device, dtype=img.dtype)

        gx = F.conv2d(img, sobel_x, padding=1)
        gy = F.conv2d(img, sobel_y, padding=1)

        # Compute gradient magnitude
        magnitude = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8)

        # Normalize to [0, 1]
        max_val = magnitude.view(magnitude.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        max_val = torch.clamp(max_val, min=1e-6)
        normalized_magnitude = magnitude / max_val

        return torch.clamp(normalized_magnitude, 0.0, 1.0)

def compute_sobel_edges(gt_img: torch.Tensor) -> torch.Tensor:
    """
    Utility function to compute normalized Sobel edge target map from Ground Truth tensor.
    """
    detector = SobelEdgeDetector().to(gt_img.device)
    return detector(gt_img)
