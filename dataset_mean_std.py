import os
import numpy as np

lr_folder = r"Train\train\NoisyLR"

total_sum = 0
total_sq_sum = 0
total_pixels = 0

files = [f for f in os.listdir(lr_folder) if f.endswith(".npy")]

for f in files:
    img = np.load(os.path.join(lr_folder, f))

    total_sum += img.sum()
    total_sq_sum += (img ** 2).sum()
    total_pixels += img.size

mean = total_sum / total_pixels
std = ((total_sq_sum / total_pixels) - mean**2) ** 0.5

print("Dataset Mean:", mean)
print("Dataset Std:", std)