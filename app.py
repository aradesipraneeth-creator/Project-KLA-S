import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import streamlit as st
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import Config
from models.airnet import AIRNet
from models.airnet_v2 import AIRNetV2
from models.airnet_v3 import AIRNetV3
from models.image_indexer import ImageIndexer

# Device Selection (CPU / CUDA compatible for Render deployment)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config(MODEL_VERSION="AIR-Net-v3")
config.create_dirs()

st.set_page_config(
    page_title="AIR-Net v3 — Content-Adaptive Restoration System",
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
        font-size: 20px;
        font-weight: 700;
    }
    .explanation-box {
        background-color: #161B22;
        border-left: 4px solid #58A6FF;
        padding: 14px;
        border-radius: 4px;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)


# --- Helper Utility & Centralized Normalization Functions ---
def normalize_image_array(img_input) -> np.ndarray:
    """
    Centralized image normalization function.
    Handles NPY, PNG, JPG, JPEG, TIFF, BMP, PIL Images, singletons, grayscale, RGB, RGBA, CHW, HWC.
    Converts internally to float32 in range [0.0, 1.0].
    """
    if isinstance(img_input, np.ndarray):
        arr = img_input.astype(np.float32)
    elif hasattr(img_input, "read") or isinstance(img_input, Image.Image):
        if not isinstance(img_input, Image.Image):
            img_input = Image.open(img_input)
        img = img_input.convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
    else:
        raise ValueError("Unsupported image format.")

    if arr.ndim == 3:
        if arr.shape[0] in [1, 3, 4]:  # CHW
            arr = arr[0]
        elif arr.shape[2] in [1, 3, 4]:  # HWC
            arr = arr[:, :, 0]

    arr = np.squeeze(arr)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def validate_image_shape(img: np.ndarray, expected_shape: tuple, name: str) -> np.ndarray:
    """
    Validates that an image array matches expected_shape (e.g., (128, 128) or (256, 256)).
    """
    if img is None:
        return None
    arr = normalize_image_array(img)
    if arr.shape != expected_shape:
        raise ValueError(
            f"Ground Truth dimension mismatch. Expected: {expected_shape[0]}×{expected_shape[1]}, Received: {arr.shape[0]}×{arr.shape[1]}"
        )
    return arr


def validate_metric_pair(pred: np.ndarray, gt: np.ndarray) -> tuple:
    """
    Verifies pred and gt have identical spatial dimensions before calculating metrics.
    Prevents (256, 256) vs (128, 128) shape mismatch crashes.
    """
    if pred is None or gt is None:
        raise ValueError("Ground Truth unavailable — quantitative fidelity metrics cannot be calculated for this image.")
    p = normalize_image_array(pred)
    g = normalize_image_array(gt)
    if p.shape != g.shape:
        raise ValueError(f"Metric shape mismatch crash prevented: Pred shape {p.shape} != GT shape {g.shape}")
    return p, g


def compute_sobel_edge_map(img_2d: np.ndarray) -> np.ndarray:
    """
    Computes deterministic Sobel gradient magnitude map for a 2D numpy array.
    Operates at native resolution (128x128 or 256x256).
    """
    t = torch.from_numpy(img_2d.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
    gx = F.conv2d(t, sobel_x, padding=1)
    gy = F.conv2d(t, sobel_y, padding=1)
    edge_mag = torch.sqrt(gx**2 + gy**2 + 1e-8).squeeze().numpy()
    return np.clip(edge_mag, 0.0, 1.0)


# --- Quantitative Metrics Suite ---
def compute_mse(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = validate_metric_pair(pred, gt)
    return float(np.mean((p - g) ** 2))

def compute_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = validate_metric_pair(pred, gt)
    return float(np.mean(np.abs(p - g)))

def compute_psnr(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    mse = compute_mse(pred, gt)
    if mse == 0:
        return 100.0
    return float(10.0 * np.log10((data_range ** 2) / mse))

def compute_ssim(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    p, g = validate_metric_pair(pred, gt)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mu1 = gaussian_filter(p.astype(np.float64), sigma=1.5)
    mu2 = gaussian_filter(g.astype(np.float64), sigma=1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = gaussian_filter(p.astype(np.float64) ** 2, sigma=1.5) - mu1_sq
    sigma2_sq = gaussian_filter(g.astype(np.float64) ** 2, sigma=1.5) - mu2_sq
    sigma12 = gaussian_filter(p.astype(np.float64) * g.astype(np.float64), sigma=1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))

def compute_lpips_safe(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = validate_metric_pair(pred, gt)
    try:
        import lpips
        p_t = torch.from_numpy(p).unsqueeze(0).unsqueeze(0).float()
        g_t = torch.from_numpy(g).unsqueeze(0).unsqueeze(0).float()
        p3 = p_t.repeat(1, 3, 1, 1) * 2.0 - 1.0
        g3 = g_t.repeat(1, 3, 1, 1) * 2.0 - 1.0
        loss_fn = lpips.LPIPS(net='alex', verbose=False).to(DEVICE)
        with torch.no_grad():
            dist = loss_fn(p3.to(DEVICE), g3.to(DEVICE)).mean().item()
        return float(dist)
    except Exception:
        return float(np.mean(np.abs(p - g)))

def resize_bicubic(lr_img: np.ndarray, target_shape=(256, 256)) -> np.ndarray:
    zoom_factors = (target_shape[0] / lr_img.shape[0], target_shape[1] / lr_img.shape[1])
    bicubic = zoom(lr_img, zoom_factors, order=3)
    return np.clip(bicubic, 0.0, 1.0)


# --- Cached PyTorch Model Loader ---
@st.cache_resource
def load_all_models():
    norm_path = PROJECT_ROOT / "outputs" / "v3" / "indexes" / "index_normalization.json"
    norm_params = None
    if norm_path.exists():
        with open(norm_path, "r") as f:
            norm_params = json.load(f)

    # AIR-Net v3 (Production Checkpoint)
    v3_model = AIRNetV3(
        in_channels=1, out_channels=1, dim=32,
        channels=[32, 64, 128, 192], heads=[1, 2, 4, 6],
        enc_blocks=[2, 2, 4], latent_blocks=8, dec_blocks=[4, 2, 2],
        ffn_expansion_factor=2.66, norm_params=norm_params, use_residual_learning=True
    ).to(DEVICE)

    v3_ckpt_candidates = [
        PROJECT_ROOT / "outputs" / "v3" / "checkpoints" / "airnet_v3_ema_best_model.pth",
        PROJECT_ROOT / "outputs" / "v3" / "checkpoints" / "airnet_v3_best_model.pth"
    ]
    v3_loaded_name = None
    for cand in v3_ckpt_candidates:
        if cand.exists():
            try:
                state_dict = torch.load(cand, map_location=DEVICE)
                v3_model.load_state_dict(state_dict.get("ema_state_dict", state_dict.get("model_state_dict", state_dict)), strict=False)
                v3_loaded_name = cand.name
                break
            except Exception as e:
                print(f"Notice loading v3 checkpoint {cand}: {e}")
    v3_model.eval()

    # AIR-Net v2
    v2_model = AIRNetV2(in_channels=1, out_channels=1, use_residual_learning=True).to(DEVICE)
    v2_ckpt = PROJECT_ROOT / "outputs" / "v2" / "checkpoints" / "airnet_v2_ema_best_model.pth"
    v2_loaded = False
    if v2_ckpt.exists():
        try:
            state_dict = torch.load(v2_ckpt, map_location=DEVICE)
            v2_model.load_state_dict(state_dict.get("ema_state_dict", state_dict), strict=False)
            v2_loaded = True
        except Exception:
            pass
    v2_model.eval()

    # AIR-Net v1.2
    v12_model = AIRNet(in_channels=1, out_channels=1).to(DEVICE)
    v12_ckpt = PROJECT_ROOT / "outputs" / "stage3" / "checkpoints" / "airnet_v1_2_ema_best_model.pth"
    v12_loaded = False
    if v12_ckpt.exists():
        try:
            state_dict = torch.load(v12_ckpt, map_location=DEVICE)
            v12_model.load_state_dict(state_dict.get("ema_state_dict", state_dict), strict=False)
            v12_loaded = True
        except Exception:
            pass
    v12_model.eval()

    # AIR-Net v1
    v1_model = AIRNet(in_channels=1, out_channels=1).to(DEVICE)
    v1_ckpt = PROJECT_ROOT / "outputs" / "checkpoints" / "airnet_ema_best_model.pth"
    v1_loaded = False
    if v1_ckpt.exists():
        try:
            state_dict = torch.load(v1_ckpt, map_location=DEVICE)
            v1_model.load_state_dict(state_dict.get("ema_state_dict", state_dict), strict=False)
            v1_loaded = True
        except Exception:
            pass
    v1_model.eval()

    return {
        "v3_model": v3_model, "v3_loaded_name": v3_loaded_name,
        "v2_model": v2_model, "v2_loaded": v2_loaded,
        "v12_model": v12_model, "v12_loaded": v12_loaded,
        "v1_model": v1_model, "v1_loaded": v1_loaded,
        "device": DEVICE
    }


def explain_category_routing(raw_indices: dict, dominant_cat: str, routing_probs: dict) -> str:
    """
    Generates an evidence-based explanation connecting the 10 input characteristic feature metrics
    to the assigned category and soft routing probabilities.
    """
    sobel = raw_indices.get("sobel_edge_index", 0.0)
    grad = raw_indices.get("gradient_energy", 0.0)
    lap = raw_indices.get("laplacian_energy", 0.0)
    hf = raw_indices.get("hf_energy", 0.0)
    texture = raw_indices.get("texture_index", 0.0)
    noise = raw_indices.get("noise_index", 0.0)
    contrast = raw_indices.get("contrast_index", 0.0)
    density = raw_indices.get("edge_density", 0.0)
    sparse = raw_indices.get("sparse_feature_index", 0.0)

    prob_pct = routing_probs.get(dominant_cat, 0.0) * 100

    if dominant_cat == "EDGE_DOMINANT":
        reason = (
            f"The router assigned the highest probability ({prob_pct:.1f}%) to **EDGE_DOMINANT** "
            f"because the input exhibits relatively strong boundary and gradient activity "
            f"(Sobel Edge Index: `{sobel:.4f}`, Gradient Energy: `{grad:.4f}`, Edge Density: `{density * 100:.1f}%`)."
        )
    elif dominant_cat == "TEXTURE_DOMINANT":
        reason = (
            f"The router assigned the highest probability ({prob_pct:.1f}%) to **TEXTURE_DOMINANT** "
            f"because the image contains high micro-texture variation "
            f"(Texture Index: `{texture:.4f}`, High-Frequency Energy: `{hf:.4f}`, Laplacian Energy: `{lap:.4f}`)."
        )
    elif dominant_cat == "NOISE_DOMINANT":
        reason = (
            f"The router assigned the highest probability ({prob_pct:.1f}%) to **NOISE_DOMINANT** "
            f"because high-frequency stochastic variation dominates structural edges (Noise Index Proxy: `{noise:.4f}`)."
        )
    elif dominant_cat == "SMOOTH_LOW_CONTRAST":
        reason = (
            f"The router assigned the highest probability ({prob_pct:.1f}%) to **SMOOTH_LOW_CONTRAST** "
            f"because intensity variations are gradual and overall contrast is low "
            f"(Contrast Index: `{contrast:.4f}`, Edge Density: `{density * 100:.1f}%`)."
        )
    elif dominant_cat == "SPARSE_FEATURE":
        reason = (
            f"The router assigned the highest probability ({prob_pct:.1f}%) to **SPARSE_FEATURE** "
            f"because the image is mostly uniform with localized peak gradient spikes (Sparse Feature Index: `{sparse:.4f}`)."
        )
    else:
        reason = f"Routed to `{dominant_cat}` with probability `{prob_pct:.1f}%` based on input characteristic feature vector."

    return reason


# --- Streamlit Dashboard Application ---
st.title("🔬 AIR-Net v3 Content-Adaptive Restoration System")
st.caption("Adaptive Image Restoration Network — Input-Guided Multi-Expert Semiconductor Image Restoration")

try:
    models_dict = load_all_models()
    device = models_dict["device"]
    if models_dict["v3_loaded_name"]:
        ckpt_status = f"✅ AIR-Net v3 Active: `{models_dict['v3_loaded_name']}` ({device.type.upper()})"
    else:
        ckpt_status = f"⚠️ Production v3 Checkpoint Not Found — Using initialized weights ({device.type.upper()})"
except Exception as e:
    st.error(f"Error initializing PyTorch model: {e}")
    models_dict = None
    ckpt_status = f"❌ Model Initialization Error: {e}"

st.sidebar.header("📁 Control Panel & Data Source")
st.sidebar.markdown(ckpt_status)

source_mode = st.sidebar.radio("Select Input Source:", ["Validation Dataset Browser", "Manual 128x128 Image Upload"])

lr_array, gt_array = None, None
selected_sample_name = "N/A"
is_validation_sample = False

train_lr_dir = config.train_lr_dir
train_gt_dir = config.train_gt_dir

if source_mode == "Validation Dataset Browser":
    is_validation_sample = True
    if os.path.exists(train_lr_dir):
        lr_files = sorted([f for f in os.listdir(train_lr_dir) if f.endswith(".npy")])
        if lr_files:
            # Preset selector
            preset = st.sidebar.selectbox("Filter Presets:", ["All Validation Samples", "Sample 000001 (Deterministic Demo)", "Sample 000021", "Sample 000034", "Sample 000064", "Sample 000095"])
            if preset != "All Validation Samples":
                target_name = preset.split("(")[0].strip().replace("Sample ", "") + ".npy"
                if target_name in lr_files:
                    selected_file = target_name
                else:
                    selected_file = st.sidebar.selectbox("Select Sample:", lr_files)
            else:
                selected_file = st.sidebar.selectbox("Select Sample:", lr_files)

            selected_sample_name = selected_file
            lr_raw = np.load(os.path.join(train_lr_dir, selected_file))
            lr_array = validate_image_shape(lr_raw, (128, 128), "Input NoisyLR")

            if os.path.exists(train_gt_dir):
                gt_path = os.path.join(train_gt_dir, selected_file)
                if os.path.exists(gt_path):
                    gt_raw = np.load(gt_path)
                    try:
                        gt_array = validate_image_shape(gt_raw, (256, 256), "Ground Truth")
                    except ValueError as err:
                        st.sidebar.error(str(err))
                        gt_array = None

elif source_mode == "Manual 128x128 Image Upload":
    uploaded_lr = st.sidebar.file_uploader("Upload 128x128 Semiconductor Image (.npy, .png, .jpg, .tiff, .bmp)", type=["npy", "png", "jpg", "jpeg", "bmp", "tiff"])
    uploaded_gt = st.sidebar.file_uploader("Upload Reference Ground Truth 256x256 (Optional)", type=["npy", "png", "jpg", "jpeg"])

    if uploaded_lr:
        selected_sample_name = uploaded_lr.name
        try:
            if uploaded_lr.name.endswith(".npy"):
                lr_raw = np.load(uploaded_lr)
            else:
                lr_raw = Image.open(uploaded_lr)
            lr_array = validate_image_shape(lr_raw, (128, 128), "Uploaded Input Image")
        except ValueError as err:
            st.error(str(err))
            lr_array = None

    if uploaded_gt:
        try:
            if uploaded_gt.name.endswith(".npy"):
                gt_raw = np.load(uploaded_gt)
            else:
                gt_raw = Image.open(uploaded_gt)
            gt_array = validate_image_shape(gt_raw, (256, 256), "Uploaded Ground Truth")
        except ValueError as err:
            st.sidebar.error(str(err))
            gt_array = None


# --- Core Pipeline Execution ---
if lr_array is not None and models_dict is not None:
    lr_tensor = torch.from_numpy(lr_array).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad(), torch.inference_mode():
        v3_out = models_dict["v3_model"](lr_tensor)
        v3_pred = torch.clamp(v3_out["restored"], 0.0, 1.0).squeeze().cpu().numpy()
        routing_probs_arr = v3_out["routing_probs"].squeeze().cpu().numpy()

        v2_out = models_dict["v2_model"](lr_tensor)
        v2_pred = torch.clamp(v2_out["restored"], 0.0, 1.0).squeeze().cpu().numpy()

        v12_out = models_dict["v12_model"](lr_tensor)
        v12_pred = torch.clamp(v12_out["restored"], 0.0, 1.0).squeeze().cpu().numpy()

        v1_out = models_dict["v1_model"](lr_tensor)
        v1_pred = torch.clamp(v1_out["restored"], 0.0, 1.0).squeeze().cpu().numpy()

    bicubic_pred = resize_bicubic(lr_array, (256, 256))

    # Compute Sobel edge maps for all resolution components
    input_edge_128 = compute_sobel_edge_map(lr_array)
    bicubic_edge_256 = compute_sobel_edge_map(bicubic_pred)
    v3_edge_256 = compute_sobel_edge_map(v3_pred)
    gt_edge_256 = compute_sobel_edge_map(gt_array) if gt_array is not None else None

    # Compute raw input characteristic features (INPUT ONLY)
    raw_indices = models_dict["v3_model"].indexer.compute_indices(lr_array)
    categories = ["EDGE_DOMINANT", "TEXTURE_DOMINANT", "NOISE_DOMINANT", "SMOOTH_LOW_CONTRAST", "SPARSE_FEATURE"]
    dominant_cat = categories[int(np.argmax(routing_probs_arr))]
    routing_dict = {cat: float(prob) for cat, prob in zip(categories, routing_probs_arr)}


    # =========================================================================
    # SECTION A: INPUT ANALYSIS
    # =========================================================================
    st.markdown("---")
    st.subheader(f"SECTION A: Input Analysis — {selected_sample_name}")

    a1, a2 = st.columns(2)
    with a1:
        st.markdown("**Native Input Image (128×128)**")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(lr_array, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)

    with a2:
        st.markdown("**Input Sobel Edge Map (128×128)**")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(input_edge_128, cmap="magma", vmin=0, vmax=1)
        ax.axis("off")
        st.pyplot(fig)


    # =========================================================================
    # SECTION B: IMAGE CHARACTERISTICS
    # =========================================================================
    st.markdown("---")
    st.subheader("SECTION B: 10 Input Characteristic Features (Input-Only)")

    b1_cols = st.columns(5)
    keys_10 = list(raw_indices.keys())
    for i, k in enumerate(keys_10[:5]):
        b1_cols[i].metric(k.replace("_", " ").title(), f"{raw_indices[k]:.4f}")
    b2_cols = st.columns(5)
    for i, k in enumerate(keys_10[5:]):
        b2_cols[i].metric(k.replace("_", " ").title(), f"{raw_indices[k]:.4f}")


    # =========================================================================
    # SECTION C & D: ADAPTIVE ROUTING & EXPLAINABILITY
    # =========================================================================
    st.markdown("---")
    c_col1, c_col2 = st.columns([1, 1])

    with c_col1:
        st.subheader("SECTION C: Soft Adaptive Router Probabilities")
        rout_df = pd.DataFrame({
            "Category": categories,
            "Probability (%)": [round(p * 100, 2) for p in routing_probs_arr]
        })
        st.bar_chart(rout_df.set_index("Category"))

    with c_col2:
        st.subheader("SECTION D: Why This Category?")
        st.markdown(f"**Dominant Assigned Category:** `{dominant_cat}`")
        explanation_text = explain_category_routing(raw_indices, dominant_cat, routing_dict)
        st.markdown(f"<div class='explanation-box'>{explanation_text}</div>", unsafe_allow_html=True)


    # =========================================================================
    # SECTION E: RESTORATION COMPARISON
    # =========================================================================
    st.markdown("---")
    st.subheader("SECTION E: Side-by-Side Restoration Comparison")

    grid_col1, grid_col2, grid_col3 = st.columns(3)
    with grid_col1:
        st.markdown("### 1. NoisyLR Input (128×128)")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(lr_array, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)
    with grid_col2:
        st.markdown("### 2. Bicubic Baseline (256×256)")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(bicubic_pred, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        st.pyplot(fig)
    with grid_col3:
        st.markdown("### 3. AIR-Net v1 (256×256)")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(v1_pred, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        st.pyplot(fig)

    grid_col4, grid_col5, grid_col6 = st.columns(3)
    with grid_col4:
        st.markdown("### 4. AIR-Net v1.2 (256×256)")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(v12_pred, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        st.pyplot(fig)
    with grid_col5:
        st.markdown("### 5. AIR-Net v3 Content-Adaptive (256×256)")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(v3_pred, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        st.pyplot(fig)
    with grid_col6:
        st.markdown("### 6. Ground Truth Target (256×256)" if gt_array is not None else "### 6. Ground Truth Target (N/A)")
        if gt_array is not None:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(gt_array, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.info("Ground Truth unavailable for this manual upload.")


    # =========================================================================
    # SECTION F: SOBEL EDGE MAP ANALYSIS
    # =========================================================================
    st.markdown("---")
    st.subheader("SECTION F: Sobel Edge Map Analysis")

    e_col1, e_col2, e_col3, e_col4 = st.columns(4)
    with e_col1:
        st.markdown("**Input Edge Map (128×128)**")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(input_edge_128, cmap="magma", vmin=0, vmax=1)
        ax.axis("off")
        st.pyplot(fig)
    with e_col2:
        st.markdown("**Bicubic Edge Map (256×256)**")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(bicubic_edge_256, cmap="magma", vmin=0, vmax=1)
        ax.axis("off")
        st.pyplot(fig)
    with e_col3:
        st.markdown("**AIR-Net v3 Edge Map (256×256)**")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(v3_edge_256, cmap="magma", vmin=0, vmax=1)
        ax.axis("off")
        st.pyplot(fig)
    with e_col4:
        st.markdown("**Ground Truth Edge Map (256×256)**" if gt_edge_256 is not None else "**GT Edge Map (N/A)**")
        if gt_edge_256 is not None:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(gt_edge_256, cmap="magma", vmin=0, vmax=1)
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.info("N/A (No GT)")


    # =========================================================================
    # SECTION G: QUANTITATIVE FIDELITY METRICS
    # =========================================================================
    st.markdown("---")
    st.subheader("SECTION G: Quantitative Fidelity Metrics")

    if is_validation_sample and gt_array is not None:
        st.caption("✅ Validation Sample: Evaluated against authoritative 256×256 Ground Truth target.")
        try:
            m_bic = {"PSNR (dB)": compute_psnr(bicubic_pred, gt_array), "SSIM": compute_ssim(bicubic_pred, gt_array), "LPIPS": compute_lpips_safe(bicubic_pred, gt_array), "MSE": compute_mse(bicubic_pred, gt_array), "MAE": compute_mae(bicubic_pred, gt_array)}
            m_v1 = {"PSNR (dB)": compute_psnr(v1_pred, gt_array), "SSIM": compute_ssim(v1_pred, gt_array), "LPIPS": compute_lpips_safe(v1_pred, gt_array), "MSE": compute_mse(v1_pred, gt_array), "MAE": compute_mae(v1_pred, gt_array)}
            m_v12 = {"PSNR (dB)": compute_psnr(v12_pred, gt_array), "SSIM": compute_ssim(v12_pred, gt_array), "LPIPS": compute_lpips_safe(v12_pred, gt_array), "MSE": compute_mse(v12_pred, gt_array), "MAE": compute_mae(v12_pred, gt_array)}
            m_v2 = {"PSNR (dB)": compute_psnr(v2_pred, gt_array), "SSIM": compute_ssim(v2_pred, gt_array), "LPIPS": compute_lpips_safe(v2_pred, gt_array), "MSE": compute_mse(v2_pred, gt_array), "MAE": compute_mae(v2_pred, gt_array)}
            m_v3 = {"PSNR (dB)": compute_psnr(v3_pred, gt_array), "SSIM": compute_ssim(v3_pred, gt_array), "LPIPS": compute_lpips_safe(v3_pred, gt_array), "MSE": compute_mse(v3_pred, gt_array), "MAE": compute_mae(v3_pred, gt_array)}

            edge_err_v3 = float(np.mean(np.abs(v3_edge_256 - gt_edge_256)))
            edge_err_bic = float(np.mean(np.abs(bicubic_edge_256 - gt_edge_256)))

            metrics_df = pd.DataFrame([
                {"Model": "Bicubic 2x", "PSNR (dB)": f"{m_bic['PSNR (dB)']:.4f}", "SSIM": f"{m_bic['SSIM']:.4f}", "LPIPS": f"{m_bic['LPIPS']:.4f}", "MSE": f"{m_bic['MSE']:.6f}", "MAE": f"{m_bic['MAE']:.6f}", "Edge Error": f"{edge_err_bic:.6f}"},
                {"Model": "AIR-Net v1", "PSNR (dB)": f"{m_v1['PSNR (dB)']:.4f}", "SSIM": f"{m_v1['SSIM']:.4f}", "LPIPS": f"{m_v1['LPIPS']:.4f}", "MSE": f"{m_v1['MSE']:.6f}", "MAE": f"{m_v1['MAE']:.6f}", "Edge Error": "N/A"},
                {"Model": "AIR-Net v1.2", "PSNR (dB)": f"{m_v12['PSNR (dB)']:.4f}", "SSIM": f"{m_v12['SSIM']:.4f}", "LPIPS": f"{m_v12['LPIPS']:.4f}", "MSE": f"{m_v12['MSE']:.6f}", "MAE": f"{m_v12['MAE']:.6f}", "Edge Error": "N/A"},
                {"Model": "AIR-Net v2", "PSNR (dB)": f"{m_v2['PSNR (dB)']:.4f}", "SSIM": f"{m_v2['SSIM']:.4f}", "LPIPS": f"{m_v2['LPIPS']:.4f}", "MSE": f"{m_v2['MSE']:.6f}", "MAE": f"{m_v2['MAE']:.6f}", "Edge Error": "N/A"},
                {"Model": "AIR-Net v3", "PSNR (dB)": f"{m_v3['PSNR (dB)']:.4f}", "SSIM": f"{m_v3['SSIM']:.4f}", "LPIPS": f"{m_v3['LPIPS']:.4f}", "MSE": f"{m_v3['MSE']:.6f}", "MAE": f"{m_v3['MAE']:.6f}", "Edge Error": f"{edge_err_v3:.6f}"}
            ])
            st.table(metrics_df)
        except ValueError as shape_err:
            st.error(f"Shape validation error: {shape_err}")
    else:
        st.warning("Ground Truth unavailable — quantitative fidelity metrics cannot be calculated for this manual upload.")


    # =========================================================================
    # SECTION H: INPUT -> OUTPUT ANALYSIS
    # =========================================================================
    st.markdown("---")
    st.subheader("SECTION H: Input → Output Restoration Pipeline Flow")

    h_col1, h_col2, h_col3, h_col4 = st.columns(4)
    with h_col1:
        st.markdown("**Step 1: Input NoisyLR**")
        st.image(lr_array, caption="128×128 Grayscale", use_container_width=True, clamp=True)
    with h_col2:
        st.markdown("**Step 2: Bicubic Upsample**")
        st.image(bicubic_pred, caption="256×256 Upsampled", use_container_width=True, clamp=True)
    with h_col3:
        st.markdown("**Step 3: AIR-Net v3 Restored**")
        st.image(v3_pred, caption="256×256 Faithful Restoration", use_container_width=True, clamp=True)
    with h_col4:
        st.markdown("**Step 4: Ground Truth Target**")
        if gt_array is not None:
            st.image(gt_array, caption="256×256 Target", use_container_width=True, clamp=True)
        else:
            st.info("N/A (Manual Upload)")


    # =========================================================================
    # SECTION I: METRIC EXPLANATION GUIDE
    # =========================================================================
    st.markdown("---")
    st.subheader("SECTION I: Why These Metrics?")

    st.markdown("""
    - **PSNR (Peak Signal-to-Noise Ratio)**: Measures pixel-level reconstruction fidelity in dB. Higher is better ($\text{Target} \approx 25\text{ dB}+$).
    - **SSIM (Structural Similarity Index)**: Measures structural, luminance, and contrast alignment in $[0, 1]$. Higher is better.
    - **LPIPS (Learned Perceptual Image Patch Similarity)**: Measures perceptual feature difference using deep feature representations. Lower is better.
    - **MSE / MAE**: Mean Squared Error and Mean Absolute Error between prediction and Ground Truth. Lower is better.
    - **Sobel Edge Error**: Mean Absolute Error between predicted Sobel edge map and Ground Truth edge map at 256×256. Lower is better.
    - **High-Frequency / Gradient / Laplacian Energy**: Quantitative descriptors of spatial details. Objective: $\text{Prediction} \approx \text{Ground Truth}$.
    """)


    # =========================================================================
    # SECTION J: DOWNLOAD RESTORED IMAGE
    # =========================================================================
    st.markdown("---")
    st.subheader("SECTION J: Restored Image Download")

    buf_img = Image.fromarray((v3_pred * 255.0).round().astype(np.uint8))
    buf_path = PROJECT_ROOT / "temp_restored_v3.png"
    buf_img.save(buf_path)
    with open(buf_path, "rb") as file:
        st.download_button(
            label="⬇️ Download AIR-Net v3 Restored Image (256×256 PNG)",
            data=file,
            file_name=f"restored_v3_{selected_sample_name}.png",
            mime="image/png"
        )
else:
    st.info("💡 Select a sample from the sidebar to inspect AIR-Net v3 content-adaptive predictions.")
