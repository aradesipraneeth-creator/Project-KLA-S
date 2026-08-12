import os
import cv2
import numpy as np
from scipy.stats import entropy
from skimage.metrics import structural_similarity as ssim

# =====================================================
# PATHS
# =====================================================

GT_FOLDER = r"Train\train\GT"
LR_FOLDER = r"Train\train\NoisyLR"
TEST_FOLDER = r"Test_NoisyLR\NoisyLR"

# =====================================================
# DATASET STATS
# =====================================================

gt_sum = 0
gt_sq_sum = 0

lr_sum = 0
lr_sq_sum = 0

gt_pixels = 0
lr_pixels = 0

psnr_scores = []
ssim_scores = []
correlations = []

noise_stds = []

residual_values = []

entropies = []

edge_densities = []

worst_psnr = []
worst_ssim = []

all_files = sorted([f for f in os.listdir(GT_FOLDER) if f.endswith(".npy")])

print("Processing", len(all_files), "training samples...")

for idx, f in enumerate(all_files):

    gt = np.load(os.path.join(GT_FOLDER, f))
    lr = np.load(os.path.join(LR_FOLDER, f))

    # ==========================================
    # Basic Statistics
    # ==========================================

    gt_sum += gt.sum()
    gt_sq_sum += (gt**2).sum()
    gt_pixels += gt.size

    lr_sum += lr.sum()
    lr_sq_sum += (lr**2).sum()
    lr_pixels += lr.size

    # ==========================================
    # Bicubic Upscale
    # ==========================================

    lr_up = cv2.resize(lr, (256, 256), interpolation=cv2.INTER_CUBIC)

    # ==========================================
    # PSNR
    # ==========================================

    mse = np.mean((gt - lr_up) ** 2)

    if mse > 0:
        psnr = 10 * np.log10(1.0 / mse)
    else:
        psnr = 100

    psnr_scores.append(psnr)

    # ==========================================
    # SSIM
    # ==========================================

    ssim_score = ssim(gt, lr_up, data_range=1.0)

    ssim_scores.append(ssim_score)

    # ==========================================
    # Correlation
    # ==========================================

    corr = np.corrcoef(gt.flatten(), lr_up.flatten())[0, 1]

    correlations.append(corr)

    # ==========================================
    # Residual Analysis
    # ==========================================

    residual = gt - lr_up

    residual_values.append(residual.flatten())

    noise_stds.append(residual.std())

    # ==========================================
    # Entropy
    # ==========================================

    hist, _ = np.histogram(gt, bins=256, range=(0, 1), density=True)

    entropies.append(entropy(hist + 1e-10))

    # ==========================================
    # Edge Density
    # ==========================================

    gt_uint8 = np.clip(gt * 255, 0, 255).astype(np.uint8)

    edges = cv2.Canny(gt_uint8, 50, 150)

    edge_density = np.mean(edges > 0)

    edge_densities.append(edge_density)

    # ==========================================
    # Worst Samples Tracking
    # ==========================================

    worst_psnr.append((f, psnr))

    worst_ssim.append((f, ssim_score))

    if (idx + 1) % 200 == 0:
        print(f"Processed {idx+1}/{len(all_files)}")

# =====================================================
# FINAL CALCULATIONS
# =====================================================

gt_mean = gt_sum / gt_pixels
gt_std = np.sqrt((gt_sq_sum / gt_pixels) - gt_mean**2)

lr_mean = lr_sum / lr_pixels
lr_std = np.sqrt((lr_sq_sum / lr_pixels) - lr_mean**2)

residual_values = np.concatenate(residual_values)

worst_psnr = sorted(worst_psnr, key=lambda x: x[1])[:20]

worst_ssim = sorted(worst_ssim, key=lambda x: x[1])[:20]

# =====================================================
# TEST SET COUNT
# =====================================================

if os.path.exists(TEST_FOLDER):
    test_count = len([f for f in os.listdir(TEST_FOLDER) if f.endswith(".npy")])
else:
    test_count = "Unknown"

# =====================================================
# REPORT
# =====================================================

print("\n")
print("=" * 50)
print("KLA DATASET REPORT")
print("=" * 50)

print(f"\nTrain Samples: {len(all_files)}")
print(f"Test Samples: {test_count}")

print("\n----- DATASET STATS -----")

print(f"GT Mean: {gt_mean:.6f}")
print(f"GT Std : {gt_std:.6f}")

print(f"LR Mean: {lr_mean:.6f}")
print(f"LR Std : {lr_std:.6f}")

print("\n----- BASELINE METRICS -----")

print(f"Average PSNR: {np.mean(psnr_scores):.6f}")
print(f"Average SSIM: {np.mean(ssim_scores):.6f}")

print(f"Min PSNR: {np.min(psnr_scores):.6f}")
print(f"Max PSNR: {np.max(psnr_scores):.6f}")

print(f"Min SSIM: {np.min(ssim_scores):.6f}")
print(f"Max SSIM: {np.max(ssim_scores):.6f}")

print("\n----- CORRELATION -----")

print(f"Average Correlation: {np.mean(correlations):.6f}")
print(f"Min Correlation: {np.min(correlations):.6f}")
print(f"Max Correlation: {np.max(correlations):.6f}")

print("\n----- RESIDUAL ANALYSIS -----")

print(f"Residual Mean: {residual_values.mean():.6f}")
print(f"Residual Std : {residual_values.std():.6f}")

print(f"Residual Min : {residual_values.min():.6f}")
print(f"Residual Max : {residual_values.max():.6f}")

print("\n----- NOISE ANALYSIS -----")

print(f"Average Noise Std: {np.mean(noise_stds):.6f}")
print(f"Min Noise Std: {np.min(noise_stds):.6f}")
print(f"Max Noise Std: {np.max(noise_stds):.6f}")

print("\n----- IMAGE COMPLEXITY -----")

print(f"Average Entropy: {np.mean(entropies):.6f}")
print(f"Min Entropy: {np.min(entropies):.6f}")
print(f"Max Entropy: {np.max(entropies):.6f}")

print("\n----- EDGE ANALYSIS -----")

print(f"Average Edge Density: {np.mean(edge_densities):.6f}")
print(f"Min Edge Density: {np.min(edge_densities):.6f}")
print(f"Max Edge Density: {np.max(edge_densities):.6f}")

print("\n----- WORST 20 PSNR SAMPLES -----")

for item in worst_psnr:
    print(item)

print("\n----- WORST 20 SSIM SAMPLES -----")

for item in worst_ssim:
    print(item)

print("\n" + "=" * 50)
print("REPORT COMPLETE")
print("=" * 50)
