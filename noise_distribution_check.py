import numpy as np
import cv2
import matplotlib.pyplot as plt

gt = np.load(r"Train\train\GT\000000.npy")
lr = np.load(r"Train\train\NoisyLR\000000.npy")

lr_up = cv2.resize(lr, (256, 256), interpolation=cv2.INTER_CUBIC)

noise = gt - lr_up

print("Noise Mean:", noise.mean())
print("Noise Std:", noise.std())

plt.hist(noise.flatten(), bins=100)
plt.title("Noise Distribution")
plt.show()
