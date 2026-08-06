import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from configs.config import Config
from models.airnet import AIRNet

def generate_model_summary(config: Config = None, model: torch.nn.Module = None) -> str:
    """
    Computes module names, parameter counts, trainable parameters,
    and dictionary output shapes for AIR-Net v1.
    Saves report to outputs/model_summary.txt.
    """
    if config is None:
        config = Config()
    config.create_dirs()

    if model is None:
        model = AIRNet(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            dim=config.dim,
            channels=config.channels,
            heads=config.heads,
            enc_blocks=config.enc_blocks,
            latent_blocks=config.latent_blocks,
            dec_blocks=config.dec_blocks,
            ffn_expansion_factor=config.ffn_expansion_factor
        )

    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    input_shape = (1, config.in_channels, 128, 128)
    dummy_input = torch.randn(*input_shape)

    with torch.no_grad():
        out_dict = model(dummy_input)

    output_shapes_str = "\n".join([
        f"   - {k:<10}: {tuple(v.shape)}" for k, v in out_dict.items()
    ])

    module_breakdown_lines = []
    for name, module in model.named_children():
        mod_params = sum(p.numel() for p in module.parameters())
        module_breakdown_lines.append(f"   - {name:<25}: {mod_params:,} params ({mod_params / 1e6:.3f} M)")

    module_breakdown_str = "\n".join(module_breakdown_lines)

    report = (
        "====================================================\n"
        "KLA SEMICONDUCTOR RESTORATION - AIR-NET V1 SUMMARY REPORT\n"
        "====================================================\n"
        f"Architecture:            AIR-Net v1 (Adaptive Image Restoration Network)\n"
        f"Input Shape:             {input_shape}\n"
        "Output Shapes:\n"
        f"{output_shapes_str}\n"
        "----------------------------------------------------\n"
        "MODULE PARAMETER BREAKDOWN:\n"
        f"{module_breakdown_str}\n"
        "----------------------------------------------------\n"
        f"Total Parameters:        {total_params:,} ({total_params / 1e6:.2f} M)\n"
        f"Trainable Parameters:    {trainable_params:,}\n"
        "====================================================\n"
    )

    with open(config.model_summary_file, "w") as f:
        f.write(report)

    print(f"AIR-Net v1 model summary generated and saved to {config.model_summary_file}")
    return report

if __name__ == "__main__":
    print(generate_model_summary())
