# KLA Project S — Semiconductor Image Restoration (AIR-Net v3)

Adaptive Image Restoration Network (AIR-Net v3) — Content-Adaptive Multi-Expert Semiconductor Image Restoration System for high-fidelity $128 \times 128 \to 256 \times 256$ super-resolution.

## 🚀 Key Features (AIR-Net v3)

- **Image Characteristic Indexer** ([`models/image_indexer.py`](models/image_indexer.py)): Computes 10 normalized metrics strictly from the $128 \times 128$ INPUT image (Sobel edge index, gradient energy, Laplacian energy, high-frequency energy, texture index, noise index, contrast index, entropy, edge density, sparse feature index).
- **Soft Adaptive Router** ([`models/adaptive_router.py`](models/adaptive_router.py)): Maps normalized input characteristics to 5 soft restoration category probabilities:
  - `EDGE_DOMINANT`
  - `TEXTURE_DOMINANT`
  - `NOISE_DOMINANT`
  - `SMOOTH_LOW_CONTRAST`
  - `SPARSE_FEATURE`
- **Lightweight Specialized Experts & Soft MoE Fusion** ([`models/experts/`](models/experts)): Shared Restormer backbone + 5 parallel expert branches fused via $F = \sum r_i \cdot \text{Expert}_i$.
- **Sample-Adaptive Dynamic Loss** ([`losses/adaptive_loss.py`](losses/adaptive_loss.py)): Loss weights dynamically scale based on category routing.
- **Float32 Precision & Locked Validation**: Modern PyTorch AMP (`torch.amp.autocast("cuda")` & `torch.amp.GradScaler("cuda")`) with canonical 320-sample validation split locked via SHA-256 (`d3c8c112fb...`).

---

## 🛠️ Installation

```bash
git clone https://github.com/aradesipraneeth-creator/Project-KLA-S.git
cd Project-KLA-S
pip install -r requirements.txt
```

---

## 🏋️ Training

To execute AIR-Net v3 content-adaptive training locally or on remote GPU hardware (NVIDIA T4 / DGX B200):

```bash
python scripts/execute_v3_pipeline.py
```

### Google Colab Remote GPU Execution:
Open [`AIRNet_v3_Colab_T4_Training.ipynb`](AIRNet_v3_Colab_T4_Training.ipynb) directly in Google Colab with T4 GPU runtime enabled.

---

## 🔬 Standalone Inference (No GT Required!)

Run content-adaptive restoration on any $128 \times 128$ grayscale semiconductor image:

```bash
python inference/restore_v3.py path/to/128x128_image.npy output_v3_256.png
```

---

## 🖥️ Streamlit Web Application

Launch the interactive multi-model and content-adaptive dashboard:

```bash
streamlit run app.py
```

Features:
- Single $128 \times 128$ image upload (.npy, .png, .jpg).
- Displays identified Category, Soft Routing Probabilities bar chart, and 10 Characteristic Features.
- Multi-model comparative grid view (NoisyLR, Bicubic, AIR-Net v1, v1.2, v2, v3, Ground Truth).
- Live metric evaluation (PSNR, SSIM, LPIPS).
- 1-click download of $256 \times 256$ restored image.
