import os
import numpy as np

gt_folder = r"Train\train\GT"
lr_folder = r"Train\train\NoisyLR"

gt_min = float("inf")
gt_max = float("-inf")

lr_min = float("inf")
lr_max = float("-inf")

for f in os.listdir(gt_folder):
    if f.endswith(".npy"):
        x = np.load(os.path.join(gt_folder, f))
        gt_min = min(gt_min, x.min())
        gt_max = max(gt_max, x.max())

for f in os.listdir(lr_folder):
    if f.endswith(".npy"):
        x = np.load(os.path.join(lr_folder, f))
        lr_min = min(lr_min, x.min())
        lr_max = max(lr_max, x.max())

print("GT Min:", gt_min)
print("GT Max:", gt_max)

print("LR Min:", lr_min)
print("LR Max:", lr_max)
