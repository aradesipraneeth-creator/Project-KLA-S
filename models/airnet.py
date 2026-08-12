import torch
import torch.nn as nn
from typing import Dict, Any

from models.degradation_analyzer import DegradationAnalyzer
from models.adaptive_modulation import AdaptiveFeatureModulation
from models.edge_head import EdgeHead
from models.restormer_baseline import (
    LayerNorm2d,
    TransformerBlock,
    Downsample,
    Upsample,
)


class AIRNet(nn.Module):
    """
    AIR-Net v1: Adaptive Image Restoration Network for Semiconductor Inspection.

    Pipeline:
      Input (128x128)
           ↓
      Degradation Analyzer (Predicts noise, blur, texture scores in [0, 1])
           ↓
      Restormer Encoder (32 -> 64 -> 128 -> 192 channels, downsampled to 16x16)
           ↓
      Adaptive Feature Modulation (Applied once at 192-channel bottleneck)
           ↓
      Restormer Latent Blocks (8 Blocks @ 192 channels, 16x16)
           ↓
      Restormer Decoder (192 -> 128 -> 64 -> 32 channels with Skip Connections)
           ↓
      Dual Reconstruction Heads:
        ├── Restored Image Head (PixelShuffle 2x -> 256x256 Image)
        └── Edge Reconstruction Head (PixelShuffle 2x -> 256x256 Edge Map)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        dim: int = 32,
        channels: list = [32, 64, 128, 192],
        heads: list = [1, 2, 4, 6],
        enc_blocks: list = [2, 2, 4],
        latent_blocks: int = 8,
        dec_blocks: list = [4, 2, 2],
        ffn_expansion_factor: float = 2.66,
    ):
        super().__init__()

        # 1. Degradation Analyzer
        self.degradation_analyzer = DegradationAnalyzer(in_channels=in_channels)

        # 2. Restormer Encoder
        self.input_conv = nn.Conv2d(
            in_channels, channels[0], kernel_size=3, stride=1, padding=1, bias=False
        )

        self.encoder_level1 = nn.Sequential(
            *[
                TransformerBlock(channels[0], heads[0], ffn_expansion_factor)
                for _ in range(enc_blocks[0])
            ]
        )
        self.down1_2 = Downsample(channels[0], channels[1])  # 32 -> 64 (64x64)

        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(channels[1], heads[1], ffn_expansion_factor)
                for _ in range(enc_blocks[1])
            ]
        )
        self.down2_3 = Downsample(channels[1], channels[2])  # 64 -> 128 (32x32)

        self.encoder_level3 = nn.Sequential(
            *[
                TransformerBlock(channels[2], heads[2], ffn_expansion_factor)
                for _ in range(enc_blocks[2])
            ]
        )
        self.down3_latent = Downsample(channels[2], channels[3])  # 128 -> 192 (16x16)

        # 3. Adaptive Feature Modulation (Placed ONCE between Encoder & Latent)
        self.adaptive_modulation = AdaptiveFeatureModulation(
            num_channels=channels[3], cond_dim=3
        )

        # 4. Restormer Latent Blocks (192 channels @ 16x16)
        self.latent = nn.Sequential(
            *[
                TransformerBlock(channels[3], heads[3], ffn_expansion_factor)
                for _ in range(latent_blocks)
            ]
        )

        # 5. Restormer Decoder
        self.up_latent_dec1 = Upsample(channels[3], channels[2])
        self.reduce_dec1 = nn.Conv2d(
            channels[2] * 2, channels[2], kernel_size=1, bias=False
        )
        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock(channels[2], heads[2], ffn_expansion_factor)
                for _ in range(dec_blocks[0])
            ]
        )

        self.up_dec1_dec2 = Upsample(channels[2], channels[1])
        self.reduce_dec2 = nn.Conv2d(
            channels[1] * 2, channels[1], kernel_size=1, bias=False
        )
        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock(channels[1], heads[1], ffn_expansion_factor)
                for _ in range(dec_blocks[1])
            ]
        )

        self.up_dec2_dec3 = Upsample(channels[1], channels[0])
        self.reduce_dec3 = nn.Conv2d(
            channels[0] * 2, channels[0], kernel_size=1, bias=False
        )
        self.decoder_level3 = nn.Sequential(
            *[
                TransformerBlock(channels[0], heads[0], ffn_expansion_factor)
                for _ in range(dec_blocks[2])
            ]
        )

        # 6. Dual Reconstruction Heads
        # Head A: Restored Image Head (B, 1, 256, 256)
        self.restored_pre_conv = nn.Conv2d(
            channels[0], channels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.restored_pixel_shuffle = nn.Sequential(
            nn.Conv2d(
                channels[0],
                channels[0] * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.PixelShuffle(2),  # (B, 32, 256, 256)
        )
        self.restored_post_conv = nn.Conv2d(
            channels[0], out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )

        # Head B: Edge Reconstruction Head (B, 1, 256, 256)
        self.edge_head = EdgeHead(in_channels=channels[0], out_channels=1)

    def forward(self, inp: torch.Tensor) -> Dict[str, Any]:
        """
        Args:
            inp: Input tensor of shape (B, 1, 128, 128)
        Returns:
            Dict containing:
              - "restored": (B, 1, 256, 256) Restored Image
              - "edge": (B, 1, 256, 256) Edge Map
              - "noise": (B,) Noise degradation score in [0, 1]
              - "blur": (B,) Blur degradation score in [0, 1]
              - "texture": (B,) Texture degradation score in [0, 1]
        """
        # Step 1: Degradation Analyzer
        cond_vector, score_dict = self.degradation_analyzer(inp)

        # Step 2: Encoder
        x = self.input_conv(inp)  # (B, 32, 128, 128)

        enc1 = self.encoder_level1(x)  # (B, 32, 128, 128)
        x_down1 = self.down1_2(enc1)  # (B, 64, 64, 64)

        enc2 = self.encoder_level2(x_down1)  # (B, 64, 64, 64)
        x_down2 = self.down2_3(enc2)  # (B, 128, 32, 32)

        enc3 = self.encoder_level3(x_down2)  # (B, 128, 32, 32)
        encoder_out = self.down3_latent(enc3)  # (B, 192, 16, 16)

        # Step 3: Adaptive Feature Modulation (Applied once at 192 ch bottleneck)
        modulated_feat = self.adaptive_modulation(
            encoder_out, cond_vector
        )  # (B, 192, 16, 16)

        # Step 4: Latent Restormer Blocks
        latent_feat = self.latent(modulated_feat)  # (B, 192, 16, 16)

        # Step 5: Decoder
        # (B, 128, 32, 32)
        dec1 = self.up_latent_dec1(latent_feat)
        # (B, 256, 32, 32)
        dec1 = torch.cat([dec1, enc3], dim=1)
        # (B, 128, 32, 32)
        dec1 = self.reduce_dec1(dec1)
        # (B, 128, 32, 32)
        dec1 = self.decoder_level1(dec1)

        # (B, 64, 64, 64)
        dec2 = self.up_dec1_dec2(dec1)
        # (B, 128, 64, 64)
        dec2 = torch.cat([dec2, enc2], dim=1)
        # (B, 64, 64, 64)
        dec2 = self.reduce_dec2(dec2)
        # (B, 64, 64, 64)
        dec2 = self.decoder_level2(dec2)

        # (B, 32, 128, 128)
        dec3 = self.up_dec2_dec3(dec2)
        # (B, 64, 128, 128)
        dec3 = torch.cat([dec3, enc1], dim=1)
        # (B, 32, 128, 128)
        dec3 = self.reduce_dec3(dec3)
        dec3_feat = self.decoder_level3(dec3)  # (B, 32, 128, 128)

        # Step 6: Dual Reconstruction Heads
        # Head A: Restored Image
        res_feat = self.restored_pre_conv(dec3_feat)
        res_upsampled = self.restored_pixel_shuffle(res_feat)  # (B, 32, 256, 256)
        restored_img = self.restored_post_conv(res_upsampled)  # (B, 1, 256, 256)

        # Head B: Edge Map
        # (B, 1, 256, 256)
        edge_map = self.edge_head(dec3_feat)

        return {
            "restored": restored_img,
            "edge": edge_map,
            "noise": score_dict["noise"],
            "blur": score_dict["blur"],
            "texture": score_dict["texture"],
        }
