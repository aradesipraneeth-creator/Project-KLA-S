import os
import csv
import json

class CSVLogger:
    """
    Automated logging to results.csv with columns:
    epoch, train_loss, val_loss, psnr, ssim, learning_rate
    """
    def __init__(self, csv_filepath: str):
        self.csv_filepath = csv_filepath
        self.fieldnames = ["epoch", "train_loss", "val_loss", "psnr", "ssim", "learning_rate"]
        
        # Initialize CSV file with headers if it doesn't exist
        os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)
        if not os.path.exists(csv_filepath):
            with open(csv_filepath, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, psnr: float, ssim: float, lr: float):
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "psnr": round(psnr, 4),
            "ssim": round(ssim, 4),
            "learning_rate": f"{lr:.6e}"
        }
        with open(self.csv_filepath, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
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
