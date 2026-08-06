import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from typing import Tuple
from configs.config import Config
from datasets.kla_dataset import get_train_val_datasets
from utils.metrics import calculate_psnr, calculate_ssim

def compute_bicubic_baseline(config: Config) -> Tuple[float, float]:
    """
    Evaluates Bicubic 2x Upsampling PSNR and SSIM exclusively on the 320 validation samples.
    Writes summary to bicubic_baseline.txt.
    """
    config.create_dirs()
    print("Computing Bicubic baseline metrics on validation split (320 samples)...")

    _, val_ds = get_train_val_datasets(
        train_lr_dir=config.train_lr_dir,
        train_gt_dir=config.train_gt_dir,
        seed=config.seed,
        train_split=config.train_split,
        val_split=config.val_split
    )

    psnr_list = []
    ssim_list = []

    for i in range(len(val_ds)):
        lr_tensor, gt_tensor, _ = val_ds[i]
        lr_batch = lr_tensor.unsqueeze(0)  # (1, 1, 128, 128)
        gt_batch = gt_tensor.unsqueeze(0)  # (1, 1, 256, 256)

        # Bicubic 2x upsampling from 128x128 to 256x256
        bicubic_pred = F.interpolate(lr_batch, size=(256, 256), mode='bicubic', align_corners=False)

        # Clip bicubic output to [0, 1] range for metric calculation against GT
        bicubic_pred_clamped = torch.clamp(bicubic_pred, 0.0, 1.0)

        psnr_val = calculate_psnr(bicubic_pred_clamped, gt_batch, data_range=1.0)
        ssim_val = calculate_ssim(bicubic_pred_clamped, gt_batch, data_range=1.0)

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

    avg_psnr = float(sum(psnr_list) / len(psnr_list))
    avg_ssim = float(sum(ssim_list) / len(ssim_list))

    report = (
        "====================================================\n"
        "KLA SEMICONDUCTOR RESTORATION - BICUBIC VALIDATION BASELINE\n"
        "====================================================\n"
        f"Validation Set Size:  {len(val_ds)} samples\n"
        f"Bicubic Average PSNR: {avg_psnr:.4f} dB\n"
        f"Bicubic Average SSIM: {avg_ssim:.4f}\n"
        "====================================================\n"
    )

    with open(config.bicubic_baseline_file, "w") as f:
        f.write(report)

    print(f"Bicubic baseline computed: PSNR={avg_psnr:.4f} dB, SSIM={avg_ssim:.4f}")
    print(f"Saved report to {config.bicubic_baseline_file}")
    return avg_psnr, avg_ssim

if __name__ == "__main__":
    cfg = Config()
    compute_bicubic_baseline(cfg)
