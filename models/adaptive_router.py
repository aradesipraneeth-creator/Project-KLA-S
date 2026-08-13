import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveRouter(nn.Module):
    """
    Soft Adaptive Router for AIR-Net v3:
    Input: 10-D normalized characteristic vector computed strictly from 128x128 INPUT.
    Output: 5 routing probabilities [r_edge, r_texture, r_noise, r_smooth, r_sparse]
            enforcing Softmax sum = 1.0.
    """
    def __init__(self, input_dim: int = 10, hidden_dim1: int = 32, hidden_dim2: int = 16, num_experts: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.GELU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LayerNorm(hidden_dim2),
            nn.GELU(),
            nn.Linear(hidden_dim2, num_experts)
        )
        self.categories = [
            "EDGE_DOMINANT",
            "TEXTURE_DOMINANT",
            "NOISE_DOMINANT",
            "SMOOTH_LOW_CONTRAST",
            "SPARSE_FEATURE"
        ]

    def forward(self, index_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            index_vector: Tensor of shape (B, 10) or (10,)
        Returns:
            torch.Tensor: Shape (B, 5) or (5,) softmax routing probabilities
        """
        is_unbatched = index_vector.ndim == 1
        if is_unbatched:
            index_vector = index_vector.unsqueeze(0)
        
        logits = self.net(index_vector.float())
        probs = F.softmax(logits, dim=-1)

        if is_unbatched:
            probs = probs.squeeze(0)
        return probs

    def get_dominant_category(self, probs: torch.Tensor) -> str:
        if probs.ndim == 2:
            probs = probs.squeeze(0)
        idx = torch.argmax(probs).item()
        return self.categories[idx]
