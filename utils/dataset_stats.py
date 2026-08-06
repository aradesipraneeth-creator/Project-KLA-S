import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from configs.config import Config
from datasets.kla_dataset import get_train_val_datasets

def generate_dataset_stats(config: Config) -> str:
    """
    Computes global statistics for LR and GT on the training split (2880 samples)
    and verifies file integrity (no NaNs or Infs).
    Writes output to train_stats.txt.
    """
    config.create_dirs()
    print("Generating dataset statistics (train_stats.txt)...")

    train_ds, _ = get_train_val_datasets(
        train_lr_dir=config.train_lr_dir,
        train_gt_dir=config.train_gt_dir,
        seed=config.seed,
        train_split=config.train_split,
        val_split=config.val_split
    )

    lr_mins, lr_maxs, lr_means, lr_stds = [], [], [], []
    gt_mins, gt_maxs, gt_means, gt_stds = [], [], [], []

    nan_inf_count = 0

    for i in range(len(train_ds)):
        lr_tensor, gt_tensor, fname = train_ds[i]
        lr_np = lr_tensor.numpy()
        gt_np = gt_tensor.numpy()

        if np.isnan(lr_np).any() or np.isinf(lr_np).any() or np.isnan(gt_np).any() or np.isinf(gt_np).any():
            nan_inf_count += 1

        lr_mins.append(float(lr_np.min()))
        lr_maxs.append(float(lr_np.max()))
        lr_means.append(float(lr_np.mean()))
        lr_stds.append(float(lr_np.std()))

        gt_mins.append(float(gt_np.min()))
        gt_maxs.append(float(gt_np.max()))
        gt_means.append(float(gt_np.mean()))
        gt_stds.append(float(gt_np.std()))

    report = (
        "====================================================\n"
        "KLA SEMICONDUCTOR RESTORATION - DATASET SANITY REPORT\n"
        "====================================================\n"
        f"Total Training Samples Analyzed: {len(train_ds)}\n"
        f"Corrupted/NaN/Inf Files Found: {nan_inf_count}\n\n"
        "--- NoisyLR Statistics (128x128 float32) ---\n"
        f"Min Value:   {np.min(lr_mins):.6f} (Avg sample min: {np.mean(lr_mins):.6f})\n"
        f"Max Value:   {np.max(lr_maxs):.6f} (Avg sample max: {np.mean(lr_maxs):.6f})\n"
        f"Mean Value:  {np.mean(lr_means):.6f}\n"
        f"Std Value:   {np.mean(lr_stds):.6f}\n\n"
        "--- Ground Truth Statistics (256x256 float32) ---\n"
        f"Min Value:   {np.min(gt_mins):.6f}\n"
        f"Max Value:   {np.max(gt_maxs):.6f}\n"
        f"Mean Value:  {np.mean(gt_means):.6f}\n"
        f"Std Value:   {np.mean(gt_stds):.6f}\n"
        "====================================================\n"
    )

    with open(config.train_stats_file, "w") as f:
        f.write(report)

    print(f"Dataset statistics report saved to {config.train_stats_file}")
    return report

if __name__ == "__main__":
    cfg = Config()
    print(generate_dataset_stats(cfg))
