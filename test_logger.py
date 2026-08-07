import os
import csv
import sys

# Ensure UTF-8 output formatting for Windows console
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from utils.logger import CSVLogger

def main():
    print("====================================================")
    print("AIR-NET V1 - CSVLOGGER UPGRADE VERIFICATION")
    print("====================================================")

    test_csv = "outputs/test_results.csv"
    if os.path.exists(test_csv):
        os.remove(test_csv)

    # 1. Initialize Logger
    logger = CSVLogger(test_csv)
    print("[OK] CSVLogger initialized successfully.")

    # 2. Test standard logging with new columns
    logger.log_epoch(
        epoch=1,
        train_loss=0.54321,
        val_loss=0.43210,
        psnr=28.50,
        ssim=0.8500,
        lr=0.0002,
        gpu_allocated_mb=1200.5,
        gpu_reserved_mb=2048.0,
        gpu_peak_mb=1500.0,
        images_per_second=142.5,
        batches_per_second=35.6
    )
    print("[OK] Log epoch with new columns (gpu_peak_mb, throughput) passed.")

    # 3. Test backward compatibility with legacy parameter gpu_max_allocated_mb
    logger.log_epoch(
        epoch=2,
        train_loss=0.44321,
        val_loss=0.33210,
        psnr=29.10,
        ssim=0.8700,
        lr=0.00015,
        gpu_allocated_mb=1210.0,
        gpu_reserved_mb=2048.0,
        gpu_max_allocated_mb=1600.0  # Legacy parameter
    )
    print("[OK] Backward compatibility test with gpu_max_allocated_mb passed.")

    # 4. Test resilience against unknown **kwargs
    logger.log_epoch(
        epoch=3,
        train_loss=0.34321,
        val_loss=0.23210,
        psnr=30.20,
        ssim=0.8900,
        lr=0.0001,
        gpu_allocated_mb=1220.0,
        gpu_reserved_mb=2048.0,
        gpu_peak_mb=1650.0,
        unknown_metric_1=999.9,
        unknown_metric_2="dummy_val"
    )
    print("[OK] Resilience test with unknown **kwargs passed without exceptions.")

    # 5. Verify CSV Header and Row Contents
    with open(test_csv, "r") as f:
        reader = list(csv.reader(f))
        header = reader[0]
        row1 = reader[1]
        row2 = reader[2]

    expected_header = [
        "epoch", "train_loss", "val_loss", "psnr", "ssim", "learning_rate",
        "gpu_allocated_mb", "gpu_reserved_mb", "gpu_peak_mb",
        "images_per_second", "batches_per_second"
    ]
    assert header == expected_header, f"Header mismatch!\nExpected: {expected_header}\nGot: {header}"

    # Row 2 check (legacy gpu_max_allocated_mb mapped to gpu_peak_mb)
    peak_idx = header.index("gpu_peak_mb")
    assert float(row2[peak_idx]) == 1600.0, f"Expected 1600.0 for mapped peak memory, got {row2[peak_idx]}"

    print("----------------------------------------------------")
    print("✓ CSVLogger updated successfully.")
    print("✓ New columns (gpu_peak_mb, images_per_second, batches_per_second) verified.")
    print("✓ Legacy gpu_max_allocated_mb mapped correctly.")
    print("✓ Unknown **kwargs handled gracefully.")
    print("✓ Complete backward compatibility maintained.")
    print("====================================================")

if __name__ == "__main__":
    main()
