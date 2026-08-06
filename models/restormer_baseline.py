import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    """
    Channel-first LayerNorm for (B, C, H, W) tensors.
    """
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias


class MDTA(nn.Module):
    """
    Multi-Dhead Transposed Attention (MDTA).
    Calculates attention maps across channel dimensions rather than spatial dimensions.
    Spatial complexity is linear O(H x W).
    """
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 1x1 conv followed by 3x3 depthwise conv for Q, K, V
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, C, H, W)
        b, c, h, w = x.shape

        qkv = self.qkv_dw(self.qkv(x))  # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)    # Each: (B, C, H, W)

        # Reshape to (B, heads, C_per_head, H*W)
        c_per_head = c // self.num_heads
        q = q.view(b, self.num_heads, c_per_head, h * w)
        k = k.view(b, self.num_heads, c_per_head, h * w)
        v = v.view(b, self.num_heads, c_per_head, h * w)

        # Normalize along spatial dimension
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Transposed attention across channels: (B, heads, C_per_head, C_per_head)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # Context aggregation: (B, heads, C_per_head, H*W)
        out = attn @ v
        out = out.view(b, c, h, w)       # (B, C, H, W)

        out = self.project_out(out)       # (B, C, H, W)
        return out


class GDFN(nn.Module):
    """
    Gated-Dconv Feed-Forward Network (GDFN).
    Controls information flow with depthwise convolutions and GELU gating.
    """
    def __init__(self, dim: int, ffn_expansion_factor: float = 2.66):
        super().__init__()
        hidden_dim = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(
            hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=1, padding=1, groups=hidden_dim * 2, bias=False
        )
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, C, H, W)
        x_proj = self.project_in(x)                   # (B, 2*hidden_dim, H, W)
        x_dw = self.dwconv(x_proj)                     # (B, 2*hidden_dim, H, W)
        x1, x2 = x_dw.chunk(2, dim=1)                  # Each: (B, hidden_dim, H, W)
        out = F.gelu(x1) * x2                          # Gating: (B, hidden_dim, H, W)
        out = self.project_out(out)                    # (B, C, H, W)
        return out


class TransformerBlock(nn.Module):
    """
    Restormer Transformer Block.
    Contains LayerNorm -> MDTA -> LayerNorm -> GDFN with residual connections.
    """
    def __init__(self, dim: int, num_heads: int, ffn_expansion_factor: float = 2.66):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim, num_heads)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim, ffn_expansion_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual MDTA
        x = x + self.attn(self.norm1(x))
        # Residual GDFN
        x = x + self.ffn(self.norm2(x))
        return x


class Downsample(nn.Module):
    """
    Strided 3x3 Convolution for Downsampling.
    Maps (B, C_in, H, W) -> (B, C_out, H/2, W/2).
    Robust feature extractor for noisy semiconductor images.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.body = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    """
    PixelShuffle 2x Upsampling Module.
    Maps (B, C_in, H, W) -> (B, C_out, 2H, 2W).
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class RestormerBaseline(nn.Module):
    """
    Stage-1 Restormer Baseline Network for Semiconductor Image Restoration.
    
    Architecture Summary:
    - Input: (B, 1, 128, 128)
    - Input Conv: 1 -> 32 channels
    - Encoder: [2, 2, 4] Restormer blocks across 3 stages
      - Stage 1: 32 channels, 1 head, (128x128) -> Downsample 3x3 Conv (stride=2) -> 64 ch (64x64)
      - Stage 2: 64 channels, 2 heads, (64x64) -> Downsample 3x3 Conv (stride=2) -> 128 ch (32x32)
      - Stage 3: 128 channels, 4 heads, (32x32) -> Downsample 3x3 Conv (stride=2) -> 192 ch (16x16)
    - Latent Stage: 8 Restormer blocks at 192 channels, 6 heads, (16x16)
    - Decoder: [4, 2, 2] Restormer blocks across 3 stages with PixelShuffle 2x upsampling & skip connections
      - Stage 1: Upsample to 128 ch (32x32) + Skip Concatenation (256 ch) -> 1x1 Conv -> 128 ch -> 4 blocks (4 heads)
      - Stage 2: Upsample to 64 ch (64x64) + Skip Concatenation (128 ch) -> 1x1 Conv -> 64 ch -> 2 blocks (2 heads)
      - Stage 3: Upsample to 32 ch (128x128) + Skip Concatenation (64 ch) -> 1x1 Conv -> 32 ch -> 2 blocks (1 head)
    - Final Output Head: Conv 3x3 -> PixelShuffle x2 (256x256) -> Conv 3x3 -> Output (B, 1, 256, 256)
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
        ffn_expansion_factor: float = 2.66
    ):
        super().__init__()

        # Input stem: (B, 1, 128, 128) -> (B, 32, 128, 128)
        self.input_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1, padding=1, bias=False)

        # Encoder Stage 1: (B, 32, 128, 128)
        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(channels[0], heads[0], ffn_expansion_factor) for _ in range(enc_blocks[0])]
        )
        self.down1_2 = Downsample(channels[0], channels[1])  # (B, 64, 64, 64)

        # Encoder Stage 2: (B, 64, 64, 64)
        self.encoder_level2 = nn.Sequential(
            *[TransformerBlock(channels[1], heads[1], ffn_expansion_factor) for _ in range(enc_blocks[1])]
        )
        self.down2_3 = Downsample(channels[1], channels[2])  # (B, 128, 32, 32)

        # Encoder Stage 3: (B, 128, 32, 32)
        self.encoder_level3 = nn.Sequential(
            *[TransformerBlock(channels[2], heads[2], ffn_expansion_factor) for _ in range(enc_blocks[2])]
        )
        self.down3_latent = Downsample(channels[2], channels[3])  # (B, 192, 16, 16)

        # Bottleneck / Latent Stage: (B, 192, 16, 16)
        self.latent = nn.Sequential(
            *[TransformerBlock(channels[3], heads[3], ffn_expansion_factor) for _ in range(latent_blocks)]
        )

        # Decoder Stage 1: 192 (16x16) -> Upsample 2x -> 128 (32x32)
        self.up_latent_dec1 = Upsample(channels[3], channels[2])
        self.reduce_dec1 = nn.Conv2d(channels[2] * 2, channels[2], kernel_size=1, bias=False)
        self.decoder_level1 = nn.Sequential(
            *[TransformerBlock(channels[2], heads[2], ffn_expansion_factor) for _ in range(dec_blocks[0])]
        )

        # Decoder Stage 2: 128 (32x32) -> Upsample 2x -> 64 (64x64)
        self.up_dec1_dec2 = Upsample(channels[2], channels[1])
        self.reduce_dec2 = nn.Conv2d(channels[1] * 2, channels[1], kernel_size=1, bias=False)
        self.decoder_level2 = nn.Sequential(
            *[TransformerBlock(channels[1], heads[1], ffn_expansion_factor) for _ in range(dec_blocks[1])]
        )

        # Decoder Stage 3: 64 (64x64) -> Upsample 2x -> 32 (128x128)
        self.up_dec2_dec3 = Upsample(channels[1], channels[0])
        self.reduce_dec3 = nn.Conv2d(channels[0] * 2, channels[0], kernel_size=1, bias=False)
        self.decoder_level3 = nn.Sequential(
            *[TransformerBlock(channels[0], heads[0], ffn_expansion_factor) for _ in range(dec_blocks[2])]
        )

        # Final 2x PixelShuffle Upsampling Head: (B, 32, 128, 128) -> (B, 1, 256, 256)
        self.output_pre_conv = nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.output_pixel_shuffle = nn.Sequential(
            nn.Conv2d(channels[0], channels[0] * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)  # (B, 32, 256, 256)
        )
        self.output_post_conv = nn.Conv2d(channels[0], out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # inp shape: (B, 1, 128, 128)
        
        # Stem
        x = self.input_conv(inp)             # (B, 32, 128, 128)

        # Encoder
        enc1 = self.encoder_level1(x)         # (B, 32, 128, 128)
        x_down1 = self.down1_2(enc1)          # (B, 64, 64, 64)

        enc2 = self.encoder_level2(x_down1)   # (B, 64, 64, 64)
        x_down2 = self.down2_3(enc2)          # (B, 128, 32, 32)

        enc3 = self.encoder_level3(x_down2)   # (B, 128, 32, 32)
        x_down3 = self.down3_latent(enc3)     # (B, 192, 16, 16)

        # Latent
        latent_feat = self.latent(x_down3)    # (B, 192, 16, 16)

        # Decoder Stage 1
        dec1 = self.up_latent_dec1(latent_feat)                 # (B, 128, 32, 32)
        dec1 = torch.cat([dec1, enc3], dim=1)                   # (B, 256, 32, 32)
        dec1 = self.reduce_dec1(dec1)                           # (B, 128, 32, 32)
        dec1 = self.decoder_level1(dec1)                        # (B, 128, 32, 32)

        # Decoder Stage 2
        dec2 = self.up_dec1_dec2(dec1)                          # (B, 64, 64, 64)
        dec2 = torch.cat([dec2, enc2], dim=1)                   # (B, 128, 64, 64)
        dec2 = self.reduce_dec2(dec2)                           # (B, 64, 64, 64)
        dec2 = self.decoder_level2(dec2)                        # (B, 64, 64, 64)

        # Decoder Stage 3
        dec3 = self.up_dec2_dec3(dec2)                          # (B, 32, 128, 128)
        dec3 = torch.cat([dec3, enc1], dim=1)                   # (B, 64, 128, 128)
        dec3 = self.reduce_dec3(dec3)                           # (B, 32, 128, 128)
        dec3 = self.decoder_level3(dec3)                        # (B, 32, 128, 128)

        # Output Upsampling Head
        out_feat = self.output_pre_conv(dec3)                   # (B, 32, 128, 128)
        out_upsampled = self.output_pixel_shuffle(out_feat)     # (B, 32, 256, 256)
        out = self.output_post_conv(out_upsampled)              # (B, 1, 256, 256)

        return out
