import os
import numpy as np

lr_folder = r"Train\train\NoisyLR"

mins = []
maxs = []

for f in os.listdir(lr_folder):
    if f.endswith(".npy"):
        img = np.load(os.path.join(lr_folder, f))
        mins.append(img.min())
        maxs.append(img.max())

print("Average Min:", np.mean(mins))
print("Average Max:", np.mean(maxs))
print("Min of Mins:", np.min(mins))
print("Max of Maxs:", np.max(maxs))
