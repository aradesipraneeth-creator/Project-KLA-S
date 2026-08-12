# KLA Project S — Semiconductor Image Restoration (AIR-Net v2)

Adaptive Image Restoration Network (AIR-Net v2) for high-fidelity $128 \times 128 \to 256 \times 256$ semiconductor inspection image super-resolution.

## 🚀 Key Features

- **Residual Reconstruction Branch**: Learns $\text{Restored} = \text{Bicubic}(\text{Input}) + \text{Residual}(\text{Input})$, ensuring an immediate baseline of $\text{PSNR} \ge 22.98\text{ dB}$ on Epoch 1 while targeting $\text{PSNR} \ge 25\text{ dB}$.
- **Multi-Objective Loss Formulation**:
  $$\text{Total Loss} = 0.70 \cdot L_1 + 0.20 \cdot \text{SSIM} + 0.05 \cdot \text{Edge} + 0.05 \cdot \text{HighFrequency}$$
- **Float32 Analytical Precision**: Safe mixed precision training with `torch.amp.autocast("cuda")` and `torch.amp.GradScaler("cuda")`.
- **Authoritative Validation Basis**: Evaluated on 320 canonical validation samples locked via SHA-256 (`d3c8c112fb...`).

---

## 🛠️ Installation

```bash
git clone https://github.com/aradesipraneeth-creator/Project-KLA-S.git
cd Project-KLA-S
pip install -r requirements.txt
```

---

## 🏋️ Training

To train AIR-Net v2 locally or on remote GPU hardware (NVIDIA T4 / DGX B200):

```bash
python scripts/execute_v2_pipeline.py
```

### Google Colab Remote GPU Execution:
Open [`AIRNet_v2_Colab_T4_Training.ipynb`](AIRNet_v2_Colab_T4_Training.ipynb) directly in Google Colab with T4 GPU runtime enabled.

---

## 🔬 Standalone Inference

Run restoration on any $128 \times 128$ grayscale semiconductor image:

```bash
python inference/restore_v2.py path/to/128x128_image.npy output_256.png
```

---

## 🖥️ Streamlit Web Application

Launch the interactive multi-model comparison dashboard:

```bash
streamlit run app.py
```

Features:
- Single $128 \times 128$ image upload (.npy, .png, .jpg).
- Live 6-panel grid comparing NoisyLR, Bicubic, AIR-Net v1, AIR-Net v1.2, AIR-Net v2, and Ground Truth.
- Live quantitative metric evaluation (PSNR, SSIM, LPIPS).
- 1-click download of $256 \times 256$ restored image.
