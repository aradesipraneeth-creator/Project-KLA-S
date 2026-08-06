import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# ==========================================
# PATHS
# ==========================================

GT_FOLDER = r"Train\train\GT"
LR_FOLDER = r"Train\train\NoisyLR"

OUTPUT_DIR = "analysis_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# DATA COLLECTION
# ==========================================

psnr_scores = []
ssim_scores = []
corr_scores = []

sample_data = []

files = sorted([
    f for f in os.listdir(GT_FOLDER)
    if f.endswith(".npy")
])

print("Processing dataset...")

for idx, f in enumerate(files):

    gt = np.load(
        os.path.join(GT_FOLDER, f)
    )

    lr = np.load(
        os.path.join(LR_FOLDER, f)
    )

    lr_up = cv2.resize(
        lr,
        (256,256),
        interpolation=cv2.INTER_CUBIC
    )

    # ======================
    # PSNR
    # ======================

    mse = np.mean(
        (gt - lr_up)**2
    )

    psnr = (
        10*np.log10(1.0/mse)
        if mse > 0
        else 100
    )

    # ======================
    # SSIM
    # ======================

    ssim_score = ssim(
        gt,
        lr_up,
        data_range=1.0
    )

    # ======================
    # Correlation
    # ======================

    corr = np.corrcoef(
        gt.flatten(),
        lr_up.flatten()
    )[0,1]

    psnr_scores.append(psnr)
    ssim_scores.append(ssim_score)
    corr_scores.append(corr)

    sample_data.append(
        {
            "file": f,
            "psnr": psnr,
            "ssim": ssim_score,
            "corr": corr
        }
    )

    if (idx+1) % 200 == 0:
        print(
            f"Processed {idx+1}/{len(files)}"
        )

# ==========================================
# FIND HARDEST SAMPLES
# ==========================================

worst_psnr = sorted(
    sample_data,
    key=lambda x: x["psnr"]
)[:10]

worst_ssim = sorted(
    sample_data,
    key=lambda x: x["ssim"]
)[:10]

# ==========================================
# SAVE WORST PSNR SAMPLES
# ==========================================

print("\nSaving Worst PSNR Samples...")

for item in worst_psnr:

    file = item["file"]

    gt = np.load(
        os.path.join(GT_FOLDER,file)
    )

    lr = np.load(
        os.path.join(LR_FOLDER,file)
    )

    lr_up = cv2.resize(
        lr,
        (256,256),
        interpolation=cv2.INTER_CUBIC
    )

    fig, ax = plt.subplots(
        1,2,
        figsize=(8,4)
    )

    ax[0].imshow(
        lr_up,
        cmap="gray"
    )

    ax[0].set_title(
        f"LR\nPSNR={item['psnr']:.2f}"
    )

    ax[1].imshow(
        gt,
        cmap="gray"
    )

    ax[1].set_title(
        "GT"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"WORST_PSNR_{file}.png"
        )
    )

    plt.close()

# ==========================================
# SAVE WORST SSIM SAMPLES
# ==========================================

print("Saving Worst SSIM Samples...")

for item in worst_ssim:

    file = item["file"]

    gt = np.load(
        os.path.join(GT_FOLDER,file)
    )

    lr = np.load(
        os.path.join(LR_FOLDER,file)
    )

    lr_up = cv2.resize(
        lr,
        (256,256),
        interpolation=cv2.INTER_CUBIC
    )

    fig, ax = plt.subplots(
        1,2,
        figsize=(8,4)
    )

    ax[0].imshow(
        lr_up,
        cmap="gray"
    )

    ax[0].set_title(
        f"LR\nSSIM={item['ssim']:.3f}"
    )

    ax[1].imshow(
        gt,
        cmap="gray"
    )

    ax[1].set_title(
        "GT"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"WORST_SSIM_{file}.png"
        )
    )

    plt.close()

# ==========================================
# CHECK 27
# DISTRIBUTIONS
# ==========================================

print("Creating Histograms...")

plt.figure(figsize=(8,5))
plt.hist(psnr_scores,bins=50)
plt.title("PSNR Distribution")
plt.xlabel("PSNR")
plt.ylabel("Count")
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "PSNR_Distribution.png"
    )
)
plt.close()

plt.figure(figsize=(8,5))
plt.hist(ssim_scores,bins=50)
plt.title("SSIM Distribution")
plt.xlabel("SSIM")
plt.ylabel("Count")
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "SSIM_Distribution.png"
    )
)
plt.close()

plt.figure(figsize=(8,5))
plt.hist(corr_scores,bins=50)
plt.title("Correlation Distribution")
plt.xlabel("Correlation")
plt.ylabel("Count")
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "Correlation_Distribution.png"
    )
)
plt.close()

# ==========================================
# SUMMARY
# ==========================================

print("\n================================")
print("HARDEST PSNR SAMPLES")
print("================================")

for item in worst_psnr:
    print(
        item["file"],
        "PSNR=",
        round(item["psnr"],3)
    )

print("\n================================")
print("HARDEST SSIM SAMPLES")
print("================================")

for item in worst_ssim:
    print(
        item["file"],
        "SSIM=",
        round(item["ssim"],3)
    )

print("\n================================")
print("OUTPUT SAVED TO")
print("================================")
print(OUTPUT_DIR)