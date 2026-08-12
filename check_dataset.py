import os
import numpy as np

gt_folder = r"Train\train\GT"
lr_folder = r"Train\train\NoisyLR"

gt_files = [f for f in os.listdir(gt_folder) if f.endswith(".npy")]
lr_files = [f for f in os.listdir(lr_folder) if f.endswith(".npy")]

print(f"GT Files: {len(gt_files)}")
print(f"NoisyLR Files: {len(lr_files)}")

gt_path = os.path.join(gt_folder, gt_files[0])
lr_path = os.path.join(lr_folder, lr_files[0])

gt = np.load(gt_path)
lr = np.load(lr_path)

print("\n===== GT =====")
print("File:", gt_files[0])
print("Shape:", gt.shape)
print("Dtype:", gt.dtype)
print("Min:", gt.min())
print("Max:", gt.max())
print("Mean:", gt.mean())

print("\n===== NoisyLR =====")
print("File:", lr_files[0])
print("Shape:", lr.shape)
print("Dtype:", lr.dtype)
print("Min:", lr.min())
print("Max:", lr.max())
print("Mean:", lr.mean())
