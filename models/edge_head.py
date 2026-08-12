import torch
import torch.nn as nn


class EdgeHead(nn.Module):
    """
    Edge Reconstruction Head for AIR-Net v1.
    Reconstructs 256x256 high-frequency semiconductor edge map from decoder feature maps.

    Architecture:
      Conv3x3(C -> C) -> ReLU
      Conv3x3(C -> C * 4) -> PixelShuffle(2)  [Upsamples 128x128 -> 256x256]
      Conv3x3(C -> 1) -> Sigmoid
    """

    def __init__(self, in_channels: int = 32, out_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                in_channels * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.PixelShuffle(2),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Decoder feature map of shape (B, C, 128, 128)
        Returns:
            torch.Tensor: Edge map of shape (B, 1, 256, 256) in [0, 1]
        """
        return self.net(x)
