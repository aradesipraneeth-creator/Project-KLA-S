import os
import json
import numpy as np
import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom
from PIL import Image

from configs.config import Config
from models.airnet import AIRNet
from models.airnet_v2 import AIRNetV2
from models.airnet_v3 import AIRNetV3
from utils.edge_utils import compute_sobel_edges

config = Config(MODEL_VERSION="AIR-Net-v3")
config.create_dirs()

st.set_page_config(
    page_title="KLA Semiconductor AIR-Net v3 Content-Adaptive Viewer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .metric-card {
        background-color: #1E222A;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #30363D;
        text-align: center;
    }
    .metric-title {
        color: #8B949E;
        font-size: 13px;
        font-weight: 600;
    }
    .metric-value {
        color: #58A6FF;
        font-size: 22px;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_all_models():
    from utils.device import get_device
    device = get_device()

    norm_path = os.path.join("outputs", "v3", "indexes", "index_normalization.json")
    norm_params = None
    if os.path.exists(norm_path):
        with open(norm_path, "r") as f:
            norm_params = json.load(f)

    # AIR-Net v3
    v3_model = AIRNetV3(
        in_channels=1, out_channels=1,
        dim=32, channels=[32, 64, 128, 192],
        heads=[1, 2, 4, 6], enc_blocks=[2, 2, 4],
        latent_blocks=8, dec_blocks=[4, 2, 2],
        ffn_expansion_factor=2.66,
        norm_params=norm_params,
        use_residual_learning=True
    ).to(device)

    v3_ckpt = os.path.join("outputs", "v3", "checkpoints", "airnet_v3_ema_best_model.pth")
    v3_loaded = False
    if os.path.exists(v3_ckpt):
        try:
            state_dict = torch.load(v3_ckpt, map_location=device)
            v3_model.load_state_dict(state_dict.get("ema_state_dict", state_dict), strict=False)
            v3_loaded = True
        except Exception as e:
            print(f"Notice loading v3 checkpoint: {e}")
    v3_model.eval()

    # AIR-Net v2
    v2_model = AIRNetV2(in_channels=1, out_channels=1, use_residual_learning=True).to(device)
    v2_ckpt = os.path.join("outputs", "v2", "checkpoints", "airnet_v2_ema_best_model.pth")
    v2_loaded = False
    if os.path.exists(v2_ckpt):
        try:
            state_dict = torch.load(v2_ckpt, map_location=device)
            v2_model.load_state_dict(state_dict.get("ema_state_dict", state_dict), strict=False)
            v2_loaded = True
        except Exception as e:
            print(f"Notice loading v2 checkpoint: {e}")
    v2_model.eval()

    # AIR-Net v1.2
    v12_model = AIRNet(in_channels=1, out_channels=1).to(device)
    v12_ckpt = os.path.join("outputs", "stage3", "checkpoints", "airnet_v1_2_ema_best_model.pth")
    v12_loaded = False
    if os.path.exists(v12_ckpt):
        try:
            state_dict = torch.load(v12_ckpt, map_location=device)
            v12_model.load_state_dict(state_dict.get("ema_state_dict", state_dict), strict=False)
            v12_loaded = True
        except Exception as e:
            print(f"Notice loading v1.2 checkpoint: {e}")
    v12_model.eval()

    # AIR-Net v1
    v1_model = AIRNet(in_channels=1, out_channels=1).to(device)
    v1_ckpt = os.path.join("outputs", "checkpoints", "airnet_ema_best_model.pth")
    v1_loaded = False
    if os.path.exists(v1_ckpt):
        try:
            state_dict = torch.load(v1_ckpt, map_location=device)
            v1_model.load_state_dict(state_dict.get("ema_state_dict", state_dict), strict=False)
            v1_loaded = True
        except Exception as e:
            print(f"Notice loading v1 checkpoint: {e}")
    v1_model.eval()

    return {
        "v3_model": v3_model, "v3_loaded": v3_loaded,
        "v2_model": v2_model, "v2_loaded": v2_loaded,
        "v12_model": v12_model, "v12_loaded": v12_loaded,
        "v1_model": v1_model, "v1_loaded": v1_loaded,
        "device": device
    }

def compute_mse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean((pred - gt) ** 2))

def compute_psnr(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    mse = compute_mse(pred, gt)
    if mse == 0:
        return 100.0
    return float(10.0 * np.log10((data_range ** 2) / mse))

def compute_ssim(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    img1, img2 = pred.astype(np.float64), gt.astype(np.float64)
    mu1 = gaussian_filter(img1, sigma=1.5)
    mu2 = gaussian_filter(img2, sigma=1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = gaussian_filter(img1 ** 2, sigma=1.5) - mu1_sq
    sigma2_sq = gaussian_filter(img2 ** 2, sigma=1.5) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sigma=1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))

def resize_bicubic(lr_img: np.ndarray, target_shape=(256, 256)) -> np.ndarray:
    zoom_factors = (target_shape[0] / lr_img.shape[0], target_shape[1] / lr_img.shape[1])
    bicubic = zoom(lr_img, zoom_factors, order=3)
    return np.clip(bicubic, 0.0, 1.0)

st.title("🔬 KLA Semiconductor AIR-Net v3 Viewer")
st.caption("Content-Adaptive Multi-Expert Semiconductor Image Restoration System")

try:
    models_dict = load_all_models()
    device = models_dict["device"]
    ckpt_msg = f"✅ AIR-Net v3 Loaded: `{models_dict['v3_loaded']}` | v2: `{models_dict['v2_loaded']}` | v1.2: `{models_dict['v12_loaded']}` ({device.type.upper()})"
except Exception as e:
    st.error(f"Error loading PyTorch models: {e}")
    models_dict = None
    ckpt_msg = f"❌ Error loading models: {e}"

st.sidebar.header("📁 Data Source Selection")
st.sidebar.markdown(ckpt_msg)

source_mode = st.sidebar.radio("Select Input Source:", ["Dataset Browser", "Manual 128x128 File Upload"])

lr_array, gt_array = None, None
selected_sample_name = "N/A"

train_lr_dir = config.train_lr_dir
train_gt_dir = config.train_gt_dir

if source_mode == "Dataset Browser":
    if os.path.exists(train_lr_dir):
        lr_files = sorted([f for f in os.listdir(train_lr_dir) if f.endswith(".npy")])
        if lr_files:
            selected_file = st.sidebar.selectbox("Select Sample:", lr_files)
            selected_sample_name = selected_file
            lr_array = np.load(os.path.join(train_lr_dir, selected_file)).astype(np.float32)
            if os.path.exists(train_gt_dir):
                gt_path = os.path.join(train_gt_dir, selected_file)
                if os.path.exists(gt_path):
                    gt_array = np.load(gt_path).astype(np.float32)

elif source_mode == "Manual 128x128 File Upload":
    uploaded_lr = st.sidebar.file_uploader("Upload 128x128 Image (.npy, .png, .jpg)", type=["npy", "png", "jpg", "jpeg", "bmp", "tiff"])
    uploaded_gt = st.sidebar.file_uploader("Upload Reference GT (.npy, .png)", type=["npy", "png", "jpg"])
    if uploaded_lr:
        if uploaded_lr.name.endswith(".npy"):
            lr_array = np.load(uploaded_lr).astype(np.float32)
        else:
            img = Image.open(uploaded_lr).convert("L")
            lr_array = np.array(img, dtype=np.float32) / 255.0
        selected_sample_name = uploaded_lr.name
    if uploaded_gt:
        if uploaded_gt.name.endswith(".npy"):
            gt_array = np.load(uploaded_gt).astype(np.float32)
        else:
            img = Image.open(uploaded_gt).convert("L")
            gt_array = np.array(img, dtype=np.float32) / 255.0

if lr_array is not None and lr_array.ndim > 2:
    lr_array = lr_array.squeeze()
if gt_array is not None and gt_array.ndim > 2:
    gt_array = gt_array.squeeze()

if lr_array is not None and models_dict is not None:
    lr_tensor = torch.from_numpy(lr_array).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad(), torch.inference_mode():
        v3_out = models_dict["v3_model"](lr_tensor)
        v3_pred = torch.clamp(v3_out["restored"], 0.0, 1.0).squeeze().cpu().numpy()
        routing_probs = v3_out["routing_probs"].squeeze().cpu().numpy()

        v2_out = models_dict["v2_model"](lr_tensor)
        v2_pred = torch.clamp(v2_out["restored"], 0.0, 1.0).squeeze().cpu().numpy()

        v12_out = models_dict["v12_model"](lr_tensor)
        v12_pred = torch.clamp(v12_out["restored"], 0.0, 1.0).squeeze().cpu().numpy()

        v1_out = models_dict["v1_model"](lr_tensor)
        v1_pred = torch.clamp(v1_out["restored"], 0.0, 1.0).squeeze().cpu().numpy()

    bicubic_pred = resize_bicubic(lr_array, (256, 256))

    # Compute raw input characteristic indices
    raw_indices = models_dict["v3_model"].indexer.compute_indices(lr_array)
    categories = ["EDGE_DOMINANT", "TEXTURE_DOMINANT", "NOISE_DOMINANT", "SMOOTH_LOW_CONTRAST", "SPARSE_FEATURE"]
    dominant_cat = categories[int(np.argmax(routing_probs))]

    st.subheader(f"AIR-Net v3 Content Analysis: {selected_sample_name}")
    st.markdown(f"**Identified Category:** `{dominant_cat}`")

    # Routing probabilities bar chart
    rout_df = pd.DataFrame({
        "Category": categories,
        "Probability": routing_probs
    })
    st.bar_chart(rout_df.set_index("Category"))

    # 10 Characteristics Metrics Display
    st.markdown("### 📊 Input Characteristic Index Vector (10 Features)")
    idx_cols = st.columns(5)
    keys_10 = list(raw_indices.keys())
    for i, k in enumerate(keys_10[:5]):
        idx_cols[i].metric(k, f"{raw_indices[k]:.4f}")
    idx_cols2 = st.columns(5)
    for i, k in enumerate(keys_10[5:]):
        idx_cols2[i].metric(k, f"{raw_indices[k]:.4f}")

    # Multi-Model Visual Grid
    st.markdown("---")
    st.subheader("🖼️ Restoration Grid Comparison")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1. NoisyLR Input (128x128)")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(lr_array, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)
    with col2:
        st.markdown("### 2. Bicubic Baseline (256x256)")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(bicubic_pred, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        st.pyplot(fig)
    with col3:
        st.markdown("### 3. AIR-Net v1 (256x256)")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(v1_pred, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        st.pyplot(fig)

    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown("### 4. AIR-Net v1.2 (256x256)")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(v12_pred, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        st.pyplot(fig)
    with col5:
        st.markdown("### 5. AIR-Net v3 Content-Adaptive (256x256)")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(v3_pred, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        st.pyplot(fig)
    with col6:
        st.markdown("### 6. Ground Truth (256x256)" if gt_array is not None else "### 6. Ground Truth (N/A)")
        if gt_array is not None:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(gt_array, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.info("Upload reference GT to view ground truth comparison.")

    # Quantitative Metrics Table if GT is present
    if gt_array is not None:
        st.markdown("---")
        st.subheader("📈 Quantitative Performance Metrics")

        m_bic = {"PSNR (dB)": compute_psnr(bicubic_pred, gt_array), "SSIM": compute_ssim(bicubic_pred, gt_array)}
        m_v1 = {"PSNR (dB)": compute_psnr(v1_pred, gt_array), "SSIM": compute_ssim(v1_pred, gt_array)}
        m_v12 = {"PSNR (dB)": compute_psnr(v12_pred, gt_array), "SSIM": compute_ssim(v12_pred, gt_array)}
        m_v2 = {"PSNR (dB)": compute_psnr(v2_pred, gt_array), "SSIM": compute_ssim(v2_pred, gt_array)}
        m_v3 = {"PSNR (dB)": compute_psnr(v3_pred, gt_array), "SSIM": compute_ssim(v3_pred, gt_array)}

        metrics_df = pd.DataFrame([
            {"Model": "Bicubic 2x", "PSNR (dB)": f"{m_bic['PSNR (dB)']:.4f}", "SSIM": f"{m_bic['SSIM']:.4f}"},
            {"Model": "AIR-Net v1", "PSNR (dB)": f"{m_v1['PSNR (dB)']:.4f}", "SSIM": f"{m_v1['SSIM']:.4f}"},
            {"Model": "AIR-Net v1.2", "PSNR (dB)": f"{m_v12['PSNR (dB)']:.4f}", "SSIM": f"{m_v12['SSIM']:.4f}"},
            {"Model": "AIR-Net v2", "PSNR (dB)": f"{m_v2['PSNR (dB)']:.4f}", "SSIM": f"{m_v2['SSIM']:.4f}"},
            {"Model": "AIR-Net v3", "PSNR (dB)": f"{m_v3['PSNR (dB)']:.4f}", "SSIM": f"{m_v3['SSIM']:.4f}"}
        ])
        st.table(metrics_df)

        buf_img = Image.fromarray((v3_pred * 255.0).round().astype(np.uint8))
        buf_img.save("temp_restored_v3.png")
        with open("temp_restored_v3.png", "rb") as file:
            st.download_button(
                label="⬇️ Download AIR-Net v3 Restored Image (256x256 PNG)",
                data=file,
                file_name=f"restored_v3_{selected_sample_name}.png",
                mime="image/png"
            )
else:
    st.info("💡 Select a sample from the sidebar to inspect AIR-Net v3 content-adaptive predictions.")
