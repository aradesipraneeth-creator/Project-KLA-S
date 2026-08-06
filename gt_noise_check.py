import os
import cv2
import numpy as np
import random

GT_FOLDER = r"Train\train\GT"
LR_FOLDER = r"Train\train\NoisyLR"

files = sorted([
    f for f in os.listdir(GT_FOLDER)
    if f.endswith(".npy")
])

# Randomly choose 100 samples
samples = random.sample(files, 100)

gt_lap = []
lr_lap = []

for f in samples:

    gt = np.load(os.path.join(GT_FOLDER, f))
    lr = np.load(os.path.join(LR_FOLDER, f))

    lr_up = cv2.resize(
        lr,
        (256,256),
        interpolation=cv2.INTER_CUBIC
    )

    gt_var = cv2.Laplacian(
        gt.astype(np.float64),
        cv2.CV_64F
    ).var()

    lr_var = cv2.Laplacian(
        lr_up.astype(np.float64),
        cv2.CV_64F
    ).var()

    gt_lap.append(gt_var)
    lr_lap.append(lr_var)

print("="*50)
print("GT NOISE CHECK")
print("="*50)

print(f"GT Average Laplacian Variance : {np.mean(gt_lap):.6f}")
print(f"LR Average Laplacian Variance : {np.mean(lr_lap):.6f}")

print(f"\nGT Min : {np.min(gt_lap):.6f}")
print(f"GT Max : {np.max(gt_lap):.6f}")

print(f"\nLR Min : {np.min(lr_lap):.6f}")
print(f"LR Max : {np.max(lr_lap):.6f}")

ratio = np.mean(gt_lap) / np.mean(lr_lap)

print(f"\nGT/LR Ratio : {ratio:.4f}")

if ratio > 1:
    print("\nGT contains more high-frequency content than LR")
else:
    print("\nGT is smoother than LR")