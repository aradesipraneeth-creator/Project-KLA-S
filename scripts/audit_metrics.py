import os
import sys
import csv
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config import Config
from datasets.kla_dataset import get_train_val_datasets
from models.airnet import AIRNet
from utils.metrics import calculate_psnr, calculate_ssim
from utils.device import get_device

def main():
    print("====================================================")
    print("KLA AIR-NET V1 - EVALUATION AUDIT & DIAGNOSTIC SCRIPT")
    print("====================================================")

    config = Config()
    device = get_device()
    print(f"Audit Execution Device: {device}")

    # Directories
    audit_vis_dir = os.path.join(config.output_dir, "evaluation_audit")
    os.makedirs(audit_vis_dir, exist_ok=True)

    # 1. Load Validation Dataset
    _, val_dataset = get_train_val_datasets(
        train_lr_dir=config.train_lr_dir,
        train_gt_dir=config.train_gt_dir,
        seed=config.seed,
        train_split=config.train_split,
        val_split=config.val_split
    )

    # 2. Checkpoint Verification & Model Loading
    ckpt_candidates = [
        os.path.join(config.checkpoint_dir, "airnet_ema_best_model.pth"),
        os.path.join(config.checkpoint_dir, "ema_best_model.pth"),
        os.path.join(config.checkpoint_dir, "best_model.pth"),
        os.path.join(config.checkpoint_dir, "last_model.pth"),
    ]

    ckpt_path = None
    for cand in ckpt_candidates:
        if os.path.exists(cand):
            ckpt_path = cand
            break

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
    ).to(device)

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading AIR-Net Checkpoint for Audit: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        if isinstance(state_dict, dict) and "ema_state_dict" in state_dict:
            state_dict = state_dict["ema_state_dict"]
        elif isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)
    else:
        print("Notice: No trained checkpoint file found in outputs/checkpoints/. Auditing architecture & evaluation pipeline with active model weights.")

    model.eval()

    # 3. Select Audit Samples (5 fixed tracking indices + 5 random samples)
    random.seed(config.seed)
    num_val = len(val_dataset)
    fixed_indices = [idx for idx in config.fixed_val_indices if idx < num_val]
    random_indices = random.sample([i for i in range(num_val) if i not in fixed_indices], 5)
    audit_indices = fixed_indices + random_indices

    csv_path = os.path.join(config.output_dir, "metric_sample_audit.csv")
    csv_fieldnames = [
        "sample_index", "filename",
        "gt_min", "gt_max", "gt_mean", "gt_std",
        "lr_min", "lr_max", "lr_mean", "lr_std",
        "bicubic_min", "bicubic_max", "bicubic_mean", "bicubic_std",
        "airnet_min", "airnet_max", "airnet_mean", "airnet_std",
        "bicubic_psnr", "bicubic_ssim",
        "airnet_psnr", "airnet_ssim",
        "psnr_diff", "ssim_diff"
    ]

    sample_rows = []

    print("\n--- SAMPLE-BY-SAMPLE EVALUATION CROSS-CHECK ---")
    with torch.no_grad():
        for count, idx in enumerate(audit_indices, start=1):
            lr_tensor, gt_tensor, fname = val_dataset[idx]
            lr_batch = lr_tensor.unsqueeze(0).to(device)  # (1, 1, 128, 128)
            gt_batch = gt_tensor.unsqueeze(0).to(device)  # (1, 1, 256, 256)

            # Bicubic 2x
            bicubic_batch = F.interpolate(lr_batch, size=(256, 256), mode='bicubic', align_corners=False)
            bicubic_clamped = torch.clamp(bicubic_batch, 0.0, 1.0)

            # AIR-Net prediction
            out = model(lr_batch)
            airnet_batch = out["restored"] if isinstance(out, dict) else out
            airnet_clamped = torch.clamp(airnet_batch, 0.0, 1.0)

            # Metrics
            bicubic_psnr = calculate_psnr(bicubic_clamped, gt_batch, data_range=1.0)
            bicubic_ssim = calculate_ssim(bicubic_clamped, gt_batch, data_range=1.0)

            airnet_psnr = calculate_psnr(airnet_clamped, gt_batch, data_range=1.0)
            airnet_ssim = calculate_ssim(airnet_clamped, gt_batch, data_range=1.0)

            psnr_diff = airnet_psnr - bicubic_psnr
            ssim_diff = airnet_ssim - bicubic_ssim

            # Array statistics
            gt_np = gt_batch.squeeze().cpu().numpy()
            lr_np = lr_batch.squeeze().cpu().numpy()
            bicubic_np = bicubic_clamped.squeeze().cpu().numpy()
            airnet_np = airnet_clamped.squeeze().cpu().numpy()

            row = {
                "sample_index": idx,
                "filename": fname,
                "gt_min": round(float(gt_np.min()), 6),
                "gt_max": round(float(gt_np.max()), 6),
                "gt_mean": round(float(gt_np.mean()), 6),
                "gt_std": round(float(gt_np.std()), 6),
                "lr_min": round(float(lr_np.min()), 6),
                "lr_max": round(float(lr_np.max()), 6),
                "lr_mean": round(float(lr_np.mean()), 6),
                "lr_std": round(float(lr_np.std()), 6),
                "bicubic_min": round(float(bicubic_np.min()), 6),
                "bicubic_max": round(float(bicubic_np.max()), 6),
                "bicubic_mean": round(float(bicubic_np.mean()), 6),
                "bicubic_std": round(float(bicubic_np.std()), 6),
                "airnet_min": round(float(airnet_np.min()), 6),
                "airnet_max": round(float(airnet_np.max()), 6),
                "airnet_mean": round(float(airnet_np.mean()), 6),
                "airnet_std": round(float(airnet_np.std()), 6),
                "bicubic_psnr": round(bicubic_psnr, 4),
                "bicubic_ssim": round(bicubic_ssim, 4),
                "airnet_psnr": round(airnet_psnr, 4),
                "airnet_ssim": round(airnet_ssim, 4),
                "psnr_diff": round(psnr_diff, 4),
                "ssim_diff": round(ssim_diff, 4)
            }
            sample_rows.append(row)

            print(f"Sample [{idx:03d}] {fname}:")
            print(f"  Bicubic: PSNR={bicubic_psnr:.4f} dB, SSIM={bicubic_ssim:.4f}")
            print(f"  AIR-Net: PSNR={airnet_psnr:.4f} dB, SSIM={airnet_ssim:.4f} (Diff: PSNR {psnr_diff:+.4f} dB, SSIM {ssim_diff:+.4f})")
            print(f"  GT Stats: min={gt_np.min():.4f}, max={gt_np.max():.4f}, mean={gt_np.mean():.4f}, std={gt_np.std():.4f}")
            print(f"  AIR-Net Stats: min={airnet_np.min():.4f}, max={airnet_np.max():.4f}, mean={airnet_np.mean():.4f}, std={airnet_np.std():.4f}")

            # Generate 6-panel Audit Image (Input LR, Bicubic, AIR-Net, GT, Bicubic Error, AIR-Net Error)
            if count <= 5:
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))

                # Top Row: Images
                axes[0, 0].imshow(lr_np, cmap='gray')
                axes[0, 0].set_title(f"Input LR (128x128)")
                axes[0, 0].axis('off')

                axes[0, 1].imshow(bicubic_np, cmap='gray')
                axes[0, 1].set_title(f"Bicubic (PSNR: {bicubic_psnr:.2f}dB, SSIM: {bicubic_ssim:.4f})")
                axes[0, 1].axis('off')

                axes[0, 2].imshow(gt_np, cmap='gray')
                axes[0, 2].set_title("Ground Truth (256x256)")
                axes[0, 2].axis('off')

                # Bottom Row: AIR-Net Restored + Absolute Error Maps
                axes[1, 0].imshow(airnet_np, cmap='gray')
                axes[1, 0].set_title(f"AIR-Net Restored (PSNR: {airnet_psnr:.2f}dB, SSIM: {airnet_ssim:.4f})")
                axes[1, 0].axis('off')

                bicubic_err = np.abs(bicubic_np - gt_np)
                im1 = axes[1, 1].imshow(bicubic_err, cmap='magma', vmin=0, vmax=0.3)
                axes[1, 1].set_title(f"Bicubic Abs Error (Mean: {bicubic_err.mean():.4f})")
                axes[1, 1].axis('off')
                plt.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)

                airnet_err = np.abs(airnet_np - gt_np)
                im2 = axes[1, 2].imshow(airnet_err, cmap='magma', vmin=0, vmax=0.3)
                axes[1, 2].set_title(f"AIR-Net Abs Error (Mean: {airnet_err.mean():.4f})")
                axes[1, 2].axis('off')
                plt.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.04)

                plt.tight_layout()
                vis_save_path = os.path.join(audit_vis_dir, f"audit_sample_{idx:03d}_{fname.replace('.npy', '.png')}")
                plt.savefig(vis_save_path, dpi=150)
                plt.close(fig)
                print(f"  Saved audit visualization to: {vis_save_path}")

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows(sample_rows)
    print(f"\nSaved sample audit CSV to: {csv_path}")

if __name__ == "__main__":
    main()
