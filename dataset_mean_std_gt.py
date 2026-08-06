import os
import numpy as np

folder = r"Train\train\GT"

total_sum = 0
total_sq = 0
pixels = 0

for f in os.listdir(folder):
    if f.endswith(".npy"):
        img = np.load(os.path.join(folder,f))
        total_sum += img.sum()
        total_sq += (img**2).sum()
        pixels += img.size

mean = total_sum / pixels
std = ((total_sq/pixels)-mean**2)**0.5

print("GT Mean:", mean)
print("GT Std:", std)