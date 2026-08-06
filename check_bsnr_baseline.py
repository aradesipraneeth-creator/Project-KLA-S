import numpy as np
import cv2

gt = np.load(r"Train\train\GT\000000.npy")

lr = np.load(r"Train\train\NoisyLR\000000.npy")
lr_up = cv2.resize(lr,(256,256), interpolation=cv2.INTER_CUBIC)

mse = np.mean((gt - lr_up)**2)

psnr = 10 * np.log10(1.0 / mse)

print("Baseline PSNR:", psnr)