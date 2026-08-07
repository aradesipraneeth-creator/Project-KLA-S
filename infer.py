import os
import argparse
import numpy as np
import torch

from configs.config import Config
from models.restormer_baseline import RestormerBaseline

from utils.device import get_device

def run_test_inference(checkpoint_path: str, config: Config = None):
    if config is None:
        config = Config()

    config.create_dirs()
    device = get_device()

    test_lr_dir = config.test_lr_dir
    output_dir = config.test_results_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Running inference on test dataset: {test_lr_dir}")
    print(f"Loading checkpoint: {checkpoint_path}")

    # Load Model
    model = RestormerBaseline(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        dim=config.dim,
        channels=config.channels,
        heads=config.heads,
        enc_blocks=config.enc_blocks,
        latent_blocks=config.latent_blocks,
        dec_blocks=config.dec_blocks,
        ffn_expansion_factor=config.ffn_expansion_factor
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    if 'ema_state_dict' in state_dict:
        state_dict = state_dict['ema_state_dict']
    elif 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    model.load_state_dict(state_dict)
    model.eval()

    test_files = sorted([f for f in os.listdir(test_lr_dir) if f.endswith(".npy")])
    print(f"Found {len(test_files)} test samples.")

    with torch.no_grad():
        for count, fname in enumerate(test_files, start=1):
            lr_path = os.path.join(test_lr_dir, fname)
            lr_arr = np.load(lr_path).astype(np.float32)

            if lr_arr.ndim == 2:
                lr_arr = np.expand_dims(lr_arr, axis=0)

            lr_tensor = torch.from_numpy(lr_arr).unsqueeze(0).to(device)  # (1, 1, 128, 128)

            pred_tensor = model(lr_tensor)  # (1, 1, 256, 256)
            pred_clamped = torch.clamp(pred_tensor, 0.0, 1.0)

            pred_np = pred_clamped.squeeze().cpu().numpy().astype(np.float32)

            save_path = os.path.join(output_dir, fname)
            np.save(save_path, pred_np)

            if count % 50 == 0 or count == len(test_files):
                print(f"  Processed [{count:03d}/{len(test_files):03d}] test predictions -> {save_path}")

    print(f"\nInference complete! Saved {len(test_files)} predictions to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for Test_NoisyLR dataset")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join("outputs", "checkpoints", "ema_best_model.pth"),
        help="Path to trained model checkpoint"
    )
    args = parser.parse_args()
    cfg = Config()
    run_test_inference(args.checkpoint, cfg)
