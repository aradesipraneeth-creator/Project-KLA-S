import torch
import torch.nn as nn
from typing import Dict, Tuple


class DegradationAnalyzer(nn.Module):
    """
    Degradation Analyzer for AIR-Net v1.
    Extracts continuous scalar degradation scores (noise, blur, texture) in [0, 1]
    from noisy LR input images (128x128).

    Architecture:
      Conv(1 -> 16, 3x3) -> ReLU
      Conv(16 -> 32, 3x3) -> ReLU
      AdaptiveAvgPool2D(1)
      Flatten
      Linear(32 -> 16) -> ReLU
      Linear(16 -> 3)
      Sigmoid
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(inplace=True), nn.Linear(16, 3), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (B, 1, 128, 128)
        Returns:
            Tuple:
              - condition_vector: Tensor of shape (B, 3)
              - score_dict: Dict with keys 'noise', 'blur', 'texture', each (B,)
        """
        feat = self.feature_extractor(x)  # (B, 32, 1, 1)
        feat = torch.flatten(feat, 1)  # (B, 32)
        condition_vector = self.fc(feat)  # (B, 3) in [0, 1]

        score_dict = {
            "noise": condition_vector[:, 0],
            "blur": condition_vector[:, 1],
            "texture": condition_vector[:, 2],
        }
        return condition_vector, score_dict
