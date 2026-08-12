import numpy as np
import matplotlib.pyplot as plt

gt = np.load(r"Train\train\GT\000000.npy")

fft = np.fft.fftshift(np.fft.fft2(gt))
magnitude = np.log(np.abs(fft) + 1)

plt.imshow(magnitude, cmap="gray")
plt.title("FFT Spectrum")
plt.colorbar()
plt.show()
