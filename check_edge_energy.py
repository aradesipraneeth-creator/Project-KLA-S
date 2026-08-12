import numpy as np
import cv2

gt = np.load(r"Train\train\GT\000000.npy")

sobelx = cv2.Sobel(gt, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(gt, cv2.CV_64F, 0, 1)

edge_energy = np.mean(np.sqrt(sobelx**2 + sobely**2))

print("Edge Energy:", edge_energy)
