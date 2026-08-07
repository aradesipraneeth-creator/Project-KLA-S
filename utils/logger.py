import os
import csv
import json
from typing import Optional, List, Dict, Any

class CSVLogger:
    """
    Automated logging to results.csv with synchronized columns for AIR-Net v1:
    epoch, train_loss, val_loss, psnr, ssim, learning_rate,
    gpu_allocated_mb, gpu_reserved_mb, gpu_peak_mb,
    images_per_second, batches_per_second
    """
    DEFAULT_FIELDNAMES = [
        "epoch",
        "train_loss",
        "val_loss",
        "psnr",
        "ssim",
        "learning_rate",
        "gpu_allocated_mb",
        "gpu_reserved_mb",
        "gpu_peak_mb",
        "images_per_second",
        "batches_per_second"
    ]

    def __init__(self, csv_filepath: str, extra_fieldnames: Optional[List[str]] = None):
        self.csv_filepath = csv_filepath
        self.fieldnames = list(self.DEFAULT_FIELDNAMES)
        if extra_fieldnames:
            for fn in extra_fieldnames:
                if fn not in self.fieldnames:
                    self.fieldnames.append(fn)
        
        # Initialize CSV file with headers if it doesn't exist
        os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)
        if not os.path.exists(csv_filepath):
            with open(csv_filepath, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        psnr: float,
        ssim: float,
        lr: float,
        gpu_allocated_mb: float = 0.0,
        gpu_reserved_mb: float = 0.0,
        gpu_peak_mb: Optional[float] = None,
        gpu_max_allocated_mb: Optional[float] = None,
        images_per_second: Optional[float] = None,
        batches_per_second: Optional[float] = None,
        **kwargs: Any
    ):
        # Backward compatibility for legacy gpu_max_allocated_mb parameter
        if gpu_peak_mb is None and gpu_max_allocated_mb is not None:
            gpu_peak_mb = gpu_max_allocated_mb
        elif gpu_peak_mb is None:
            gpu_peak_mb = 0.0

        row: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "psnr": round(psnr, 4),
            "ssim": round(ssim, 4),
            "learning_rate": f"{lr:.6e}",
            "gpu_allocated_mb": round(gpu_allocated_mb or 0.0, 2),
            "gpu_reserved_mb": round(gpu_reserved_mb or 0.0, 2),
            "gpu_peak_mb": round(gpu_peak_mb or 0.0, 2),
            "images_per_second": round(images_per_second or 0.0, 2) if images_per_second is not None else 0.0,
            "batches_per_second": round(batches_per_second or 0.0, 2) if batches_per_second is not None else 0.0,
        }

        # Dynamically map any additional registered fieldnames from kwargs
        for key in self.fieldnames:
            if key not in row and key in kwargs:
                val = kwargs[key]
                if isinstance(val, float):
                    row[key] = round(val, 4)
                else:
                    row[key] = val

        with open(self.csv_filepath, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction='ignore')
            writer.writerow(row)

def save_json(data: dict, filepath: str):
    """Saves dictionary data to a formatted JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def print_epoch_summary(epoch: int, total_epochs: int, train_loss: float, val_loss: float, psnr: float, ssim: float, lr: float):
    """Prints a clean epoch summary table line to console."""
    print(f"[Epoch {epoch:02d}/{total_epochs:02d}] "
          f"Train Loss: {train_loss:.6f} | "
          f"Val Loss: {val_loss:.6f} | "
          f"PSNR: {psnr:.4f} dB | "
          f"SSIM: {ssim:.4f} | "
          f"LR: {lr:.2e}")
