import numpy as np
import os

gt_folder = r"Train\train\GT"
lr_folder = r"Train\train\NoisyLR"

gt = np.load(os.path.join(gt_folder, "000000.npy"))
lr = np.load(os.path.join(lr_folder, "000000.npy"))

print("GT:", gt.shape)
print("LR:", lr.shape)

print("GT unique dimensions:", len(gt.shape))
print("LR unique dimensions:", len(lr.shape))