import numpy as np
import matplotlib.pyplot as plt

gt = np.load(r"Train\train\GT\000000.npy")
lr = np.load(r"Train\train\NoisyLR\000000.npy")

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(lr, cmap="gray")
plt.title("NoisyLR")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(gt, cmap="gray")
plt.title("GT")
plt.axis("off")

plt.tight_layout()
plt.show()