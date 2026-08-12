import os
import numpy as np

test_folder = r"Test_NoisyLR"

files = [f for f in os.listdir(test_folder) if f.endswith(".npy")]

print("Test Files:", len(files))

sample = np.load(os.path.join(test_folder, files[0]))

print("Shape:", sample.shape)
print("Dtype:", sample.dtype)
print("Min:", sample.min())
print("Max:", sample.max())
