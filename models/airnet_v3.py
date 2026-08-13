import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from models.image_indexer import ImageIndexer
from models.adaptive_router import AdaptiveRouter
from models.experts import (
    EdgeExpert,
    TextureExpert,
    NoiseExpert,
    SmoothExpert,
    SparseExpert
)
from models.degradation_analyzer import DegradationAnalyzer
from models.adaptive_modulation import AdaptiveFeatureModulation
from models.edge_head import EdgeHead
from models.restormer_baseline import (
    LayerNorm2d,
    TransformerBlock,
    Downsample,
    Upsample
)

class AIRNetV3(nn.Module):
    """
    AIR-Net v3: Content-Adaptive Multi-Expert Semiconductor Image Restoration Network.
    
    Pipeline:
      1. Image Characteristic Indexer: Computes 10 normalized metrics from INPUT (128x128).
      2. Soft Adaptive Router: Computes 5 routing probabilities [r_edge, r_texture, r_noise, r_smooth, r_sparse].
      3. Shared Restormer Backbone: Feature extraction across encoder, latent, and decoder levels.
      4. Lightweight Specialized Experts: 5 parallel expert branches operating on decoder features.
      5. Soft MoE Feature Fusion: Weighted sum of expert outputs: F = sum(r_i * E_i).
      6. Reconstruction Head: Output = Bicubic(Input) + Residual(Input).
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
        norm_params: dict = None,
        use_residual_learning: bool = True
    ):
        super().__init__()
        self.use_residual_learning = use_residual_learning

        # 1. Content Indexer & Soft Router
        self.indexer = ImageIndexer(norm_params=norm_params)
        self.router = AdaptiveRouter(input_dim=10, num_experts=5)

        # 2. Degradation Analyzer
        self.degradation_analyzer = DegradationAnalyzer(in_channels=in_channels)

        # 3. Restormer Shared Encoder
        self.input_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(channels[0], heads[0], ffn_expansion_factor) for _ in range(enc_blocks[0])]
        )
        self.down1_2 = Downsample(channels[0], channels[1])

        self.encoder_level2 = nn.Sequential(
            *[TransformerBlock(channels[1], heads[1], ffn_expansion_factor) for _ in range(enc_blocks[1])]
        )
        self.down2_3 = Downsample(channels[1], channels[2])

        self.encoder_level3 = nn.Sequential(
            *[TransformerBlock(channels[2], heads[2], ffn_expansion_factor) for _ in range(enc_blocks[2])]
        )
        self.down3_latent = Downsample(channels[2], channels[3])

        # 4. Adaptive Feature Modulation
        self.adaptive_modulation = AdaptiveFeatureModulation(num_channels=channels[3], cond_dim=3)

        # 5. Restormer Latent Blocks
        self.latent = nn.Sequential(
            *[TransformerBlock(channels[3], heads[3], ffn_expansion_factor) for _ in range(latent_blocks)]
        )

        # 6. Restormer Shared Decoder
        self.up_latent_dec1 = Upsample(channels[3], channels[2])
        self.reduce_dec1 = nn.Conv2d(channels[2] * 2, channels[2], kernel_size=1, bias=False)
        self.decoder_level1 = nn.Sequential(
            *[TransformerBlock(channels[2], heads[2], ffn_expansion_factor) for _ in range(dec_blocks[0])]
        )

        self.up_dec1_dec2 = Upsample(channels[2], channels[1])
        self.reduce_dec2 = nn.Conv2d(channels[1] * 2, channels[1], kernel_size=1, bias=False)
        self.decoder_level2 = nn.Sequential(
            *[TransformerBlock(channels[1], heads[1], ffn_expansion_factor) for _ in range(dec_blocks[1])]
        )

        self.up_dec2_dec3 = Upsample(channels[1], channels[0])
        self.reduce_dec3 = nn.Conv2d(channels[0] * 2, channels[0], kernel_size=1, bias=False)
        self.decoder_level3 = nn.Sequential(
            *[TransformerBlock(channels[0], heads[0], ffn_expansion_factor) for _ in range(dec_blocks[2])]
        )

        # 7. Specialized Restoration Experts (5 Lightweight Branches)
        self.expert_edge = EdgeExpert(num_channels=channels[0])
        self.expert_texture = TextureExpert(num_channels=channels[0])
        self.expert_noise = NoiseExpert(num_channels=channels[0])
        self.expert_smooth = SmoothExpert(num_channels=channels[0])
        self.expert_sparse = SparseExpert(num_channels=channels[0])

        # 8. Dual Reconstruction Heads
        self.restored_pre_conv = nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.restored_pixel_shuffle = nn.Sequential(
            nn.Conv2d(channels[0], channels[0] * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )
        self.restored_post_conv = nn.Conv2d(channels[0], out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.edge_head = EdgeHead(in_channels=channels[0], out_channels=1)

    def set_norm_params(self, norm_params: dict):
        self.indexer.norm_params = norm_params

    def forward(self, inp: torch.Tensor, precomputed_index_vec: torch.Tensor = None) -> Dict[str, Any]:
        b, c, h, w = inp.shape

        # Step 1: Input Characteristic Indexing & Routing
        if precomputed_index_vec is None:
            # Batch-level characteristic calculation
            index_vecs = []
            for i in range(b):
                raw = self.indexer.compute_indices(inp[i:i+1])
                norm_vec = self.indexer.normalize_indices(raw).to(inp.device)
                index_vecs.append(norm_vec)
            index_vec_batch = torch.stack(index_vecs, dim=0)
        else:
            index_vec_batch = precomputed_index_vec.to(inp.device)

        # Step 2: Soft Adaptive Router
        routing_probs = self.router(index_vec_batch)  # (B, 5)

        # Step 3: Shared Backbone Encoder
        cond_vector, score_dict = self.degradation_analyzer(inp)

        x = self.input_conv(inp)
        enc1 = self.encoder_level1(x)
        x_down1 = self.down1_2(enc1)

        enc2 = self.encoder_level2(x_down1)
        x_down2 = self.down2_3(enc2)

        enc3 = self.encoder_level3(x_down2)
        encoder_out = self.down3_latent(enc3)

        modulated_feat = self.adaptive_modulation(encoder_out, cond_vector)
        latent_feat = self.latent(modulated_feat)

        # Step 4: Shared Backbone Decoder
        dec1 = self.up_latent_dec1(latent_feat)
        dec1 = torch.cat([dec1, enc3], dim=1)
        dec1 = self.reduce_dec1(dec1)
        dec1 = self.decoder_level1(dec1)

        dec2 = self.up_dec1_dec2(dec1)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.reduce_dec2(dec2)
        dec2 = self.decoder_level2(dec2)

        dec3 = self.up_dec2_dec3(dec2)
        dec3 = torch.cat([dec3, enc1], dim=1)
        dec3 = self.reduce_dec3(dec3)
        dec3_feat = self.decoder_level3(dec3)  # (B, 32, 128, 128)

        # Step 5: Specialized Expert Processing & Soft MoE Fusion
        feat_edge = self.expert_edge(dec3_feat)
        feat_texture = self.expert_texture(dec3_feat)
        feat_noise = self.expert_noise(dec3_feat)
        feat_smooth = self.expert_smooth(dec3_feat)
        feat_sparse = self.expert_sparse(dec3_feat)

        r = routing_probs.view(b, 5, 1, 1, 1)
        fused_feat = (
            r[:, 0] * feat_edge +
            r[:, 1] * feat_texture +
            r[:, 2] * feat_noise +
            r[:, 3] * feat_smooth +
            r[:, 4] * feat_sparse
        )

        # Step 6: Reconstruction
        res_feat = self.restored_pre_conv(fused_feat)
        res_upsampled = self.restored_pixel_shuffle(res_feat)
        residual_img = self.restored_post_conv(res_upsampled)

        if self.use_residual_learning:
            bicubic_base = F.interpolate(inp.float(), size=(256, 256), mode='bicubic', align_corners=False)
            restored_img = bicubic_base + residual_img
        else:
            restored_img = residual_img

        edge_map = self.edge_head(fused_feat)

        return {
            "restored": restored_img,
            "residual": residual_img,
            "edge": edge_map,
            "routing_probs": routing_probs,
            "index_vector": index_vec_batch,
            "noise": score_dict["noise"],
            "blur": score_dict["blur"],
            "texture": score_dict["texture"]
        }
