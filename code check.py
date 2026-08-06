import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# Windows dataset path
dataset_path = r"C:\Users\arade\OneDrive\Documents\KLA Project S\Test_NoisyLR\NoisyLR"
# Example:
# dataset_path = r"D:\Datasets\Test_NoisyLR\NoisyLR"

# Check if folder exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset folder not found:\n{dataset_path}")

# Get all .npy files
files = sorted(glob.glob(os.path.join(dataset_path, "*.npy")))

print(f"Dataset Path : {dataset_path}")
print(f"Total Files  : {len(files)}")

if len(files) == 0:
    raise ValueError("No .npy files found!")

# -----------------------------
# Sample File Information
# -----------------------------
sample = np.load(files[0])

print("\nSample File:", os.path.basename(files[0]))
print("Shape :", sample.shape)
print("Dtype :", sample.dtype)
print("Min   :", sample.min())
print("Max   :", sample.max())

plt.figure(figsize=(5, 5))
plt.imshow(sample, cmap="gray")
plt.title("Sample Image")
plt.colorbar()
plt.show()

# -----------------------------
# Check Corrupted Files
# -----------------------------
corrupted = 0

for file in files:
    try:
        np.load(file)
    except Exception:
        corrupted += 1

print("\nCorrupted Files:", corrupted)

# -----------------------------
# Check NaN Values
# -----------------------------
nan_files = 0

for file in files:
    img = np.load(file)
    if np.isnan(img).any():
        nan_files += 1

print("Files with NaN:", nan_files)

# -----------------------------
# Check Infinite Values
# -----------------------------
inf_files = 0

for file in files:
    img = np.load(file)
    if np.isinf(img).any():
        inf_files += 1

print("Files with Inf:", inf_files)

# -----------------------------
# Compute Dataset Mean & Std
# -----------------------------
pixel_sum = 0.0
pixel_sq_sum = 0.0
total_pixels = 0

for file in files:
    img = np.load(file).astype(np.float64)

    pixel_sum += img.sum()
    pixel_sq_sum += np.square(img).sum()
    total_pixels += img.size

mean = pixel_sum / total_pixels
std = np.sqrt((pixel_sq_sum / total_pixels) - mean ** 2)

print("\nDataset Statistics")
print("------------------")
print("Mean :", mean)
print("Std  :", std)

# -----------------------------
# Check Image Shapes
# -----------------------------
shapes = set()

for file in files:
    img = np.load(file)
    shapes.add(img.shape)

print("\nUnique Image Shapes:")
print(shapes)

# -----------------------------
# Pixel Distribution
# -----------------------------
pixels = []

for file in files:
    img = np.load(file)
    pixels.append(img.ravel()[::10])  # Sample every 10th pixel

pixels = np.concatenate(pixels)

plt.figure(figsize=(8, 5))
plt.hist(pixels, bins=100, color="steelblue", edgecolor="black")
plt.title("Pixel Distribution")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()