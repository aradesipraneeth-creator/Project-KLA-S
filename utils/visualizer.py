import os
import numpy as np
import torch
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def save_visualizations_and_predictions(
    model: torch.nn.Module,
    val_dataset,
    fixed_indices: list,
    epoch: int,
    vis_dir: str,
    val_preds_dir: str,
    device: torch.device
):
    """
    Evaluates model on fixed validation samples and saves:
      1. 4-panel PNG image grids (LR, Bicubic, Prediction, Ground Truth).
      2. Raw float32 numpy prediction arrays (.npy).
    """
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(val_preds_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for sample_count, val_idx in enumerate(fixed_indices, start=1):
            if val_idx >= len(val_dataset):
                continue

            lr_tensor, gt_tensor, fname = val_dataset[val_idx]
            lr_batch = lr_tensor.unsqueeze(0).to(device)  # (1, 1, 128, 128)
            gt_batch = gt_tensor.unsqueeze(0).to(device)  # (1, 1, 256, 256)

            # Model prediction
            pred_batch = model(lr_batch)  # (1, 1, 256, 256)
            pred_batch_clamped = torch.clamp(pred_batch, 0.0, 1.0)

            # Bicubic reference
            bicubic_batch = F.interpolate(lr_batch, size=(256, 256), mode='bicubic', align_corners=False)
            bicubic_clamped = torch.clamp(bicubic_batch, 0.0, 1.0)

            # Save raw prediction array (.npy)
            pred_np = pred_batch_clamped.squeeze().cpu().numpy().astype(np.float32)
            raw_npy_path = os.path.join(val_preds_dir, f"sample_{sample_count:03d}_epoch_{epoch:02d}.npy")
            np.save(raw_npy_path, pred_np)

            # Save 4-panel visual comparison PNG
            if HAS_MATPLOTLIB:
                lr_display = bicubic_clamped.squeeze().cpu().numpy()
                bicubic_display = bicubic_clamped.squeeze().cpu().numpy()
                pred_display = pred_np
                gt_display = gt_batch.squeeze().cpu().numpy()

                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                
                axes[0].imshow(lr_display, cmap='gray')
                axes[0].set_title("Input LR (Upsampled)")
                axes[0].axis('off')

                axes[1].imshow(bicubic_display, cmap='gray')
                axes[1].set_title("Bicubic Baseline")
                axes[1].axis('off')

                axes[2].imshow(pred_display, cmap='gray')
                axes[2].set_title(f"Prediction (Epoch {epoch:02d})")
                axes[2].axis('off')

                axes[3].imshow(gt_display, cmap='gray')
                axes[3].set_title("Ground Truth")
                axes[3].axis('off')

                plt.tight_layout()
                png_path = os.path.join(vis_dir, f"epoch_{epoch:02d}_sample_{sample_count:03d}.png")
                plt.savefig(png_path, dpi=150)
                plt.close(fig)
