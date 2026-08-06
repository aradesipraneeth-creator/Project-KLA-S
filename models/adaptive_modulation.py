import torch
import torch.nn as nn

class AdaptiveFeatureModulation(nn.Module):
    """
    Adaptive Feature Modulation (FiLM) for AIR-Net v1.
    Modulates intermediate feature maps using condition vector (B, 3) from Degradation Analyzer.
    
    Formula:
        output = feature * (1 + gamma) + beta
    
    Placement in AIR-Net v1:
        Encoder Output (B, C, H, W) -> Adaptive Feature Modulation -> Latent Restormer Blocks
    """
    def __init__(self, num_channels: int, cond_dim: int = 3):
        super().__init__()
        self.num_channels = num_channels
        self.fc = nn.Linear(cond_dim, num_channels * 2)
        
        # Initialize gamma to 0 and beta to 0 for identity modulation at initialization
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, feature: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature: Tensor of shape (B, C, H, W)
            condition: Tensor of shape (B, 3)
        Returns:
            torch.Tensor: Modulated feature map of shape (B, C, H, W)
        """
        b, c, h, w = feature.shape
        params = self.fc(condition)  # (B, 2*C)
        params = params.view(b, 2, c, 1, 1)

        gamma = params[:, 0, :, :, :]  # (B, C, 1, 1)
        beta = params[:, 1, :, :, :]   # (B, C, 1, 1)

        modulated_feature = feature * (1.0 + gamma) + beta
        return modulated_feature
