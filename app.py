import os
import json
import numpy as np
import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, laplace, gaussian_filter, zoom

from configs.config import Config
from models.airnet import AIRNet
from utils.edge_utils import compute_sobel_edges

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="KLA Semiconductor AIR-Net v1 Viewer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
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


# --- Cached AIR-Net PyTorch Model Loader ---
@st.cache_resource
def load_airnet_model():
    """
    Instantiates AIRNet v1 model and loads checkpoint if available.
    """
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AIRNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        dim=config.dim,
        channels=config.channels,
        heads=config.heads,
        enc_blocks=config.enc_blocks,
        latent_blocks=config.latent_blocks,
        dec_blocks=config.dec_blocks,
        ffn_expansion_factor=config.ffn_expansion_factor
    ).to(device)

    checkpoint_dir = config.checkpoint_dir
    checkpoints_to_check = [
        ("airnet_ema_best_model.pth", os.path.join(checkpoint_dir, "airnet_ema_best_model.pth")),
        ("ema_best_model.pth", os.path.join(checkpoint_dir, "ema_best_model.pth")),
        ("best_model.pth", os.path.join(checkpoint_dir, "best_model.pth")),
        ("last_model.pth", os.path.join(checkpoint_dir, "last_model.pth"))
    ]

    loaded_checkpoint_name = None
    for name, path in checkpoints_to_check:
        if os.path.exists(path):
            try:
                state_dict = torch.load(path, map_location=device)
                if 'ema_state_dict' in state_dict:
                    state_dict = state_dict['ema_state_dict']
                elif 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                
                model.load_state_dict(state_dict, strict=False)
                loaded_checkpoint_name = name
                break
            except Exception as e:
                print(f"Notice loading checkpoint {path}: {e}")

    model.eval()
    return model, device, loaded_checkpoint_name


def run_airnet_inference(model, device, lr_arr: np.ndarray):
    """
    Runs live PyTorch inference using AIR-Net v1 on a 2D LR array (128x128).
    Returns dict containing 2D restored array, 2D edge map, and float degradation scores.
    """
    if lr_arr.ndim == 2:
        lr_tensor = torch.from_numpy(lr_arr).unsqueeze(0).unsqueeze(0).to(device)
    elif lr_arr.ndim == 3:
        lr_tensor = torch.from_numpy(lr_arr).unsqueeze(0).to(device)
    else:
        lr_tensor = torch.from_numpy(lr_arr).to(device)

    with torch.no_grad():
        out_dict = model(lr_tensor)

    restored_arr = torch.clamp(out_dict["restored"], 0.0, 1.0).squeeze().cpu().numpy().astype(np.float32)
    edge_arr = torch.clamp(out_dict["edge"], 0.0, 1.0).squeeze().cpu().numpy().astype(np.float32)
    noise_score = float(out_dict["noise"].item())
    blur_score = float(out_dict["blur"].item())
    texture_score = float(out_dict["texture"].item())

    return {
        "restored": restored_arr,
        "edge": edge_arr,
        "noise": noise_score,
        "blur": blur_score,
        "texture": texture_score
    }


# --- Helper Metrics Functions ---
def compute_mse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean((pred - gt) ** 2))

def compute_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - gt)))

def compute_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(compute_mse(pred, gt)))

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


# --- Application Header ---
st.title("🔬 KLA Semiconductor AIR-Net v1 Viewer")
st.caption("Adaptive Image Restoration Network (AIR-Net v1) Evaluation & Degradation Diagnostics")

# --- Load AIR-Net Model ---
try:
    airnet_model, exec_device, loaded_ckpt_name = load_airnet_model()
    if loaded_ckpt_name:
        ckpt_status_msg = f"✅ Loaded checkpoint: `{loaded_ckpt_name}` ({exec_device.type.upper()})"
    else:
        ckpt_status_msg = f"⚠️ No trained checkpoint found. Using initialized AIR-Net v1 weights."
except Exception as e:
    st.error("Error initializing AIR-Net v1 PyTorch model:")
    st.exception(e)
    airnet_model, exec_device, loaded_ckpt_name = None, torch.device("cpu"), None
    ckpt_status_msg = f"❌ Model initialization error: {e}"

# --- Sidebar Controls ---
st.sidebar.header("📁 Data Source Selection")
st.sidebar.markdown(ckpt_status_msg)

source_mode = st.sidebar.radio("Select Input Source:", ["Training/Test Dataset Browser", "Validation Predictions Browser", "Manual File Upload"])

lr_array, gt_array, pred_array, edge_array = None, None, None, None
noise_val, blur_val, texture_val = 0.0, 0.0, 0.0
selected_sample_name = "N/A"

cfg = Config()
outputs_dir = cfg.output_dir
train_lr_dir = cfg.train_lr_dir
train_gt_dir = cfg.train_gt_dir
test_lr_dir = cfg.test_lr_dir

if source_mode == "Training/Test Dataset Browser":
    browser_folder = st.sidebar.selectbox("Select Dataset Folder:", ["Train/train (NoisyLR + GT)", "Test_NoisyLR"])
    if browser_folder == "Train/train (NoisyLR + GT)" and os.path.exists(train_lr_dir):
        lr_files = sorted([f for f in os.listdir(train_lr_dir) if f.endswith(".npy")])
        selected_file = st.sidebar.selectbox("Select Sample File:", lr_files)
        selected_sample_name = selected_file
        lr_array = np.load(os.path.join(train_lr_dir, selected_file)).astype(np.float32)
        if os.path.exists(train_gt_dir):
            gt_path = os.path.join(train_gt_dir, selected_file)
            if os.path.exists(gt_path):
                gt_array = np.load(gt_path).astype(np.float32)
    elif browser_folder == "Test_NoisyLR" and os.path.exists(test_lr_dir):
        test_files = sorted([f for f in os.listdir(test_lr_dir) if f.endswith(".npy")])
        selected_file = st.sidebar.selectbox("Select Test Sample:", test_files)
        selected_sample_name = selected_file
        lr_array = np.load(os.path.join(test_lr_dir, selected_file)).astype(np.float32)

elif source_mode == "Validation Predictions Browser":
    val_preds_dir = os.path.join(outputs_dir, "validation_predictions")
    if os.path.exists(val_preds_dir):
        npy_files = sorted([f for f in os.listdir(val_preds_dir) if f.endswith(".npy")])
        if npy_files:
            selected_pred_file = st.sidebar.selectbox("Select Validation Prediction:", npy_files)
            selected_sample_name = selected_pred_file
            try:
                cfg = Config()
                from datasets.kla_dataset import get_train_val_datasets
                _, val_ds = get_train_val_datasets(train_lr_dir=cfg.train_lr_dir, train_gt_dir=cfg.train_gt_dir, seed=cfg.seed)
                parts = selected_pred_file.split("_")
                sample_idx_num = int(parts[1]) - 1
                if 0 <= sample_idx_num < len(val_ds):
                    lr_t, gt_t, fname = val_ds[sample_idx_num]
                    lr_array = lr_t.squeeze().numpy()
                    gt_array = gt_t.squeeze().numpy()
            except Exception as e:
                st.sidebar.warning(f"Could not auto-match GT/LR: {e}")

elif source_mode == "Manual File Upload":
    uploaded_lr = st.sidebar.file_uploader("Upload LR (.npy)", type=["npy"])
    uploaded_gt = st.sidebar.file_uploader("Upload Ground Truth (.npy)", type=["npy"])
    if uploaded_lr:
        lr_array = np.load(uploaded_lr).astype(np.float32)
        selected_sample_name = uploaded_lr.name
    if uploaded_gt:
        gt_array = np.load(uploaded_gt).astype(np.float32)


# --- Squeeze 2D Shapes ---
if lr_array is not None and lr_array.ndim > 2:
    lr_array = lr_array.squeeze()
if gt_array is not None and gt_array.ndim > 2:
    gt_array = gt_array.squeeze()


# --- AIR-NET V1 LIVE INFERENCE TRIGGER ---
if lr_array is not None and airnet_model is not None:
    try:
        airnet_out = run_airnet_inference(airnet_model, exec_device, lr_array)
        pred_array = airnet_out["restored"]
        edge_array = airnet_out["edge"]
        noise_val = airnet_out["noise"]
        blur_val = airnet_out["blur"]
        texture_val = airnet_out["texture"]
    except Exception as e:
        st.sidebar.error("AIR-Net v1 Inference Error:")
        st.sidebar.exception(e)

# Bicubic baseline reference
bicubic_array = resize_bicubic(lr_array, (256, 256)) if lr_array is not None else None

# Fallback Preview if no data selected
if lr_array is None:
    st.info("💡 Select a sample from the sidebar to inspect AIR-Net v1 predictions.")
    np.random.seed(42)
    lr_array = np.random.rand(128, 128).astype(np.float32)
    gt_array = np.random.rand(256, 256).astype(np.float32)
    if airnet_model is not None:
        airnet_out = run_airnet_inference(airnet_model, exec_device, lr_array)
        pred_array = airnet_out["restored"]
        edge_array = airnet_out["edge"]
        noise_val, blur_val, texture_val = airnet_out["noise"], airnet_out["blur"], airnet_out["texture"]
    bicubic_array = resize_bicubic(lr_array, (256, 256))
    selected_sample_name = "Demo Preview Sample"


# --- SIDEBAR DEGRADATION SCORES PROGRESS BARS ---
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Degradation Analyzer Scores")

st.sidebar.markdown(f"**Noise Score:** `{noise_val * 100:.1f}%`")
st.sidebar.progress(min(max(noise_val, 0.0), 1.0))

st.sidebar.markdown(f"**Blur Score:** `{blur_val * 100:.1f}%`")
st.sidebar.progress(min(max(blur_val, 0.0), 1.0))

st.sidebar.markdown(f"**Texture Complexity:** `{texture_val * 100:.1f}%`")
st.sidebar.progress(min(max(texture_val, 0.0), 1.0))


# --- MAIN LAYOUT 3 COLUMNS ---
st.subheader(f"AIR-Net v1 Live Output: {selected_sample_name}")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 1. Input Image (128x128)")
    if lr_array is not None:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(lr_array, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)

with col2:
    st.markdown("### 2. Restored Image (256x256)")
    if pred_array is not None:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(pred_array, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        st.pyplot(fig)

with col3:
    st.markdown("### 3. Predicted Edge Map (256x256)")
    if edge_array is not None:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(edge_array, cmap="magma", vmin=0, vmax=1)
        ax.axis("off")
        st.pyplot(fig)


# --- TABS FOR EXPANDED ANALYSIS ---
tab1, tab2, tab3 = st.tabs([
    "🔍 Detailed Comparison & Error Heatmaps",
    "📊 Quantitative Metrics & Statistics",
    "📄 AIR-Net v1 Reports & Metadata"
])

with tab1:
    if pred_array is not None and gt_array is not None:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Ground Truth (256x256)**")
            fig, ax = plt.subplots()
            ax.imshow(gt_array, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            st.pyplot(fig)
        with c2:
            st.markdown("**GT Sobel Edge Map**")
            gt_edge_sobel = compute_sobel_edges(torch.from_numpy(gt_array).unsqueeze(0).unsqueeze(0)).squeeze().numpy()
            fig, ax = plt.subplots()
            ax.imshow(gt_edge_sobel, cmap="magma", vmin=0, vmax=1)
            ax.axis("off")
            st.pyplot(fig)
        with c3:
            st.markdown("**Absolute Error Map |Pred - GT|**")
            abs_err = np.abs(pred_array - gt_array)
            fig, ax = plt.subplots()
            im = ax.imshow(abs_err, cmap="hot", vmin=0.0, vmax=0.25)
            plt.colorbar(im, ax=ax)
            ax.axis("off")
            st.pyplot(fig)

with tab2:
    if pred_array is not None and gt_array is not None:
        psnr_v = compute_psnr(pred_array, gt_array)
        ssim_v = compute_ssim(pred_array, gt_array)
        mse_v = compute_mse(pred_array, gt_array)
        mae_v = compute_mae(pred_array, gt_array)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PSNR (dB)", f"{psnr_v:.4f} dB")
        m2.metric("SSIM", f"{ssim_v:.4f}")
        m3.metric("MSE", f"{mse_v:.6f}")
        m4.metric("MAE", f"{mae_v:.6f}")

with tab3:
    st.markdown("### 📋 Model Architecture & Summary")
    summary_path = os.path.join(outputs_dir, "model_summary.txt")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            st.code(f.read(), language="text")
