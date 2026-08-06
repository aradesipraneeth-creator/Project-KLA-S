import numpy as np
import cv2

gt = np.load(r"Train\train\GT\000000.npy")
lr = np.load(r"Train\train\NoisyLR\000000.npy")

lr_up = cv2.resize(
    lr,
    (256,256),
    interpolation=cv2.INTER_CUBIC
)

diff = np.abs(gt - lr_up)

print("Mean Difference:", diff.mean())
print("Max Difference:", diff.max())