import os
import numpy as np

gt_folder = r"Train\train\GT"
lr_folder = r"Train\train\NoisyLR"

gt_shapes = set()
lr_shapes = set()

for f in os.listdir(gt_folder):
    if f.endswith(".npy"):
        gt_shapes.add(np.load(os.path.join(gt_folder, f)).shape)

for f in os.listdir(lr_folder):
    if f.endswith(".npy"):
        lr_shapes.add(np.load(os.path.join(lr_folder, f)).shape)

print("GT Shapes:", gt_shapes)
print("NoisyLR Shapes:", lr_shapes)