from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2

gt = np.load(r"Train\train\GT\000000.npy")

lr = np.load(r"Train\train\NoisyLR\000000.npy")
lr_up = cv2.resize(lr, (256, 256))

score = ssim(gt, lr_up, data_range=1.0)

print("Baseline SSIM:", score)
