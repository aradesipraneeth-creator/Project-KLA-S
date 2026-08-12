import numpy as np
import cv2

gt = np.load(r"Train\train\GT\000000.npy").astype(np.float64)

lr = np.load(r"Train\train\NoisyLR\000000.npy")
lr_up = cv2.resize(lr, (256, 256), interpolation=cv2.INTER_CUBIC)
lr_up = lr_up.astype(np.float64)

gt_var = cv2.Laplacian(gt, cv2.CV_64F).var()
lr_var = cv2.Laplacian(lr_up, cv2.CV_64F).var()

print("GT Sharpness:", gt_var)
print("LR Sharpness:", lr_var)
