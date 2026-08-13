import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeExpert(nn.Module):
    """
    Edge Expert (Section 19):
    Specialized for recovering sharp structural boundaries and line geometries.
    Uses depthwise-separable convolutions and high-pass edge refinement.
    """
    def __init__(self, num_channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, groups=num_channels, bias=False)
        self.pointwise = nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(4, num_channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.act(self.norm(self.pointwise(self.conv1(x))))
        out = self.conv2(out)
        return res + out


class TextureExpert(nn.Module):
    """
    Texture Expert (Section 20):
    Specialized for fine microscopic texture and multi-scale detail reconstruction.
    Uses dilated convolutions for multi-receptive field context gathering.
    """
    def __init__(self, num_channels: int = 32):
        super().__init__()
        self.branch1 = nn.Conv2d(num_channels, num_channels // 2, kernel_size=3, padding=1, dilation=1, bias=False)
        self.branch2 = nn.Conv2d(num_channels, num_channels // 2, kernel_size=3, padding=2, dilation=2, bias=False)
        self.fuse = nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(4, num_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        concat = torch.cat([b1, b2], dim=1)
        out = self.act(self.norm(self.fuse(concat)))
        return x + out


class NoiseExpert(nn.Module):
    """
    Noise Expert (Section 21):
    Specialized for noise suppression while preserving underlying structural boundaries.
    Uses gated spatial filtering.
    """
    def __init__(self, num_channels: int = 32):
        super().__init__()
        self.filter_conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.gate_conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(4, num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        filt = self.filter_conv(feat)
        gate = torch.sigmoid(self.gate_conv(x))
        out = self.norm(filt * gate)
        return x + out


class SmoothExpert(nn.Module):
    """
    Smooth / Low-Contrast Expert (Section 22):
    Specialized for subtle low-contrast intensity regions without artificial sharpening.
    Uses soft residual filtering.
    """
    def __init__(self, num_channels: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(4, num_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.norm(self.conv(x)))
        return 0.9 * x + 0.1 * out


class SparseExpert(nn.Module):
    """
    Sparse Feature Expert (Section 23):
    Specialized for preserving isolated tiny semiconductor structures and defects.
    Uses max-pooling spatial attention.
    """
    def __init__(self, num_channels: int = 32):
        super().__init__()
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(4, num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        att = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        out = self.norm(self.conv(x * att))
        return x + out
