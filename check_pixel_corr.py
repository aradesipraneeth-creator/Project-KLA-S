import numpy as np
import cv2

gt = np.load(r"Train\train\GT\000000.npy")
lr = np.load(r"Train\train\NoisyLR\000000.npy")

lr_up = cv2.resize(lr,(256,256))

corr = np.corrcoef(
    gt.flatten(),
    lr_up.flatten()
)[0,1]

print("Correlation:", corr)