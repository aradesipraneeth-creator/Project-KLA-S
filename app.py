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
    page_title="AIR-Net v3 — Semiconductor Restoration Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- High-Contrast Accessible Styling (NO Black-on-Black / Low Contrast Text) ---
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stApp {
        color: #E6EDF3;
    }
    .high-contrast-card {
        background-color: #161B22;
        color: #F0F6FC;
        border-left: 5px solid #58A6FF;
        padding: 18px;
        border-radius: 6px;
        margin-top: 10px;
        margin-bottom: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }
    .card-heading {
        color: #58A6FF;
        font-weight: 700;
        font-size: 17px;
        margin-bottom: 8px;
    }
    .card-text {
        color: #E6EDF3;
        font-size: 14px;
        line-height: 1.5;
    }
    .metric-badge {
        background-color: #21262D;
        border: 1px solid #30363D;
        border-radius: 6px;
        padding: 8px;
        text-align: center;
    }
    .metric-title {
        color: #8B949E;
        font-size: 12px;
        font-weight: 600;
    }
    .metric-num {
        color: #58A6FF;
        font-size: 18px;
        font-weight: 700;
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
    Validates that an image array matches expected_shape (e.g. (128, 128) or (256, 256)).
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
    Prevents shape mismatch crashes.
    """
    if pred is None or gt is None:
        raise ValueError("Ground Truth unavailable — quantitative fidelity metrics cannot be calculated for this image.")
    p = normalize_image_array(pred)
    g = normalize_image_array(gt)
    if p.shape != g.shape:
        raise ValueError(f"Metric shape mismatch crash prevented: Pred shape {p.shape} != GT shape {g.shape}")
    return p, g


def compute_sobel_edge_map(img_2d: np.ndarray) -> tuple:
    """
    Computes deterministic Sobel gradient magnitude map for a 2D numpy array.
    Operates at native resolution (128x128 or 256x256).
    Returns (raw_magnitude, normalized_visual_magnitude, max_val).
    """
    t = torch.from_numpy(img_2d.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
    gx = F.conv2d(t, sobel_x, padding=1)
    gy = F.conv2d(t, sobel_y, padding=1)
    edge_mag = torch.sqrt(gx**2 + gy**2 + 1e-8).squeeze().numpy()

    max_val = float(np.max(edge_mag))
    if max_val == 0:
        edge_vis = np.zeros_like(edge_mag)
    else:
        edge_vis = np.clip(edge_mag / max_val, 0.0, 1.0)

    return edge_mag, edge_vis, max_val


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
def load_airnet_v3_model():
    norm_path = PROJECT_ROOT / "outputs" / "v3" / "indexes" / "index_normalization.json"
    norm_params = None
    if norm_path.exists():
        with open(norm_path, "r") as f:
            norm_params = json.load(f)

    model = AIRNetV3(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        dim=config.dim,
        channels=config.channels,
        heads=config.heads,
        enc_blocks=config.enc_blocks,
        latent_blocks=config.latent_blocks,
        dec_blocks=config.dec_blocks,
        ffn_expansion_factor=config.ffn_expansion_factor,
        norm_params=norm_params,
        use_residual_learning=True
    ).to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters())

    ckpt_candidates = [
        PROJECT_ROOT / "outputs" / "v3" / "checkpoints" / "airnet_v3_ema_best_model.pth",
        PROJECT_ROOT / "outputs" / "v3" / "checkpoints" / "airnet_v3_best_model.pth",
        PROJECT_ROOT / "outputs" / "v3" / "checkpoints" / "airnet_v3_last_model.pth",
        PROJECT_ROOT / "outputs" / "v3" / "checkpoints" / "training_state_latest.pth"
    ]

    loaded_path = None
    loaded_size_mb = 0.0
    for cand in ckpt_candidates:
        if cand.exists():
            try:
                state_dict = torch.load(cand, map_location=DEVICE)
                weight_dict = state_dict.get("ema_state_dict", state_dict.get("model_state_dict", state_dict))
                model.load_state_dict(weight_dict, strict=True)
                loaded_path = str(cand)
                loaded_size_mb = os.path.getsize(cand) / (1024 * 1024)
                break
            except Exception as e:
                print(f"Notice loading checkpoint {cand}: {e}")

    model.eval()
    return {
        "model": model,
        "num_params": num_params,
        "loaded_path": loaded_path,
        "loaded_size_mb": loaded_size_mb,
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
            f"The adaptive router assigned the highest probability (**{prob_pct:.1f}%**) to **EDGE_DOMINANT** "
            f"because the input exhibits strong boundary and structural gradient activity "
            f"(Sobel Edge Index: `{sobel:.4f}`, Gradient Energy: `{grad:.4f}`, Edge Density: `{density * 100:.1f}%`)."
        )
    elif dominant_cat == "TEXTURE_DOMINANT":
        reason = (
            f"The adaptive router assigned the highest probability (**{prob_pct:.1f}%**) to **TEXTURE_DOMINANT** "
            f"because the input contains high micro-texture variation "
            f"(Texture Index: `{texture:.4f}`, High-Frequency Energy: `{hf:.4f}`, Laplacian Energy: `{lap:.4f}`)."
        )
    elif dominant_cat == "NOISE_DOMINANT":
        reason = (
            f"The adaptive router assigned the highest probability (**{prob_pct:.1f}%**) to **NOISE_DOMINANT** "
            f"because high-frequency stochastic noise dominates structured boundaries (Noise Index Proxy: `{noise:.4f}`)."
        )
    elif dominant_cat == "SMOOTH_LOW_CONTRAST":
        reason = (
            f"The adaptive router assigned the highest probability (**{prob_pct:.1f}%**) to **SMOOTH_LOW_CONTRAST** "
            f"because intensity transitions are gradual and overall contrast is low "
            f"(Contrast Index: `{contrast:.4f}`, Edge Density: `{density * 100:.1f}%`)."
        )
    elif dominant_cat == "SPARSE_FEATURE":
        reason = (
            f"The adaptive router assigned the highest probability (**{prob_pct:.1f}%**) to **SPARSE_FEATURE** "
            f"because the image is predominantly uniform with localized peak gradient spikes (Sparse Feature Index: `{sparse:.4f}`)."
        )
    else:
        reason = f"Routed to **{dominant_cat}** with probability **{prob_pct:.1f}%** based on the 10-D characteristic vector."

    return reason


# --- Dashboard App Core ---
st.title("🔬 AIR-Net v3 Content-Adaptive Restoration Viewer")
st.caption("Adaptive Multi-Expert Semiconductor Image Restoration Network (128×128 → 256×256)")

try:
    info_dict = load_airnet_v3_model()
    model = info_dict["model"]
    num_params = info_dict["num_params"]
    loaded_path = info_dict["loaded_path"]
    device_obj = info_dict["device"]

    if loaded_path:
        ckpt_status = f"✅ AIR-Net v3 Checkpoint Loaded: `{os.path.basename(loaded_path)}` ({info_dict['loaded_size_mb']:.1f} MB | {num_params:,} Params | {device_obj.type.upper()})"
    else:
        ckpt_status = f"⚠️ Checkpoint File Not Found — Operating in Architecture Demonstration Mode ({num_params:,} Params | {device_obj.type.upper()})"
except Exception as e:
    st.error(f"ERROR: AIR-Net v3 initialization failed: {e}")
    st.stop()

st.sidebar.header("📁 Control Panel & Data Source")
st.sidebar.markdown(ckpt_status)

source_mode = st.sidebar.radio("Select Input Source:", ["Validation Dataset Browser", "Manual 128×128 File Upload"])

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
            preset = st.sidebar.selectbox("Filter Presets:", ["All Validation Samples", "Sample 000001 (Demo)", "Sample 000021", "Sample 000034", "Sample 000064", "Sample 000095"])
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

elif source_mode == "Manual 128×128 File Upload":
    uploaded_lr = st.sidebar.file_uploader("Upload 128×128 Semiconductor Image (.npy, .png, .jpg, .tiff, .bmp)", type=["npy", "png", "jpg", "jpeg", "bmp", "tiff"])
    uploaded_gt = st.sidebar.file_uploader("Upload Reference Ground Truth 256×256 (Optional)", type=["npy", "png", "jpg", "jpeg"])

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


# --- Core Inference & Display Execution ---
if lr_array is not None:
    lr_tensor = torch.from_numpy(lr_array).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # AIR-Net v3 Programmatic Inference Verification
    try:
        with torch.no_grad(), torch.inference_mode():
            v3_out = model(lr_tensor)
            restored_raw = v3_out["restored"].squeeze().cpu().numpy()  # Raw Float32 for metrics
            routing_probs_arr = v3_out["routing_probs"].squeeze().cpu().numpy()

        r_min, r_max, r_mean = float(restored_raw.min()), float(restored_raw.max()), float(restored_raw.mean())
        print(f"[AIR-NET V3 INFERENCE] Input: (128,128) | Output: (256,256) | Min: {r_min:.6f} | Max: {r_max:.6f} | Mean: {r_mean:.6f}")

        if np.isnan(r_min) or np.isnan(r_max) or restored_raw.shape != (256, 256):
            st.error("AIR-Net v3 produced an invalid output (NaN or shape mismatch).")
            st.stop()
    except Exception as inf_err:
        st.error(f"ERROR: AIR-Net v3 inference failed: {inf_err}")
        st.stop()

    restored_vis = np.clip(restored_raw, 0.0, 1.0)
    bicubic_pred = resize_bicubic(lr_array, (256, 256))

    # Sobel Edge Calculations
    input_edge_raw, input_edge_vis, _ = compute_sobel_edge_map(lr_array)
    bicubic_edge_raw, bicubic_edge_vis, _ = compute_sobel_edge_map(bicubic_pred)
    v3_edge_raw, v3_edge_vis, _ = compute_sobel_edge_map(restored_raw)
    gt_edge_raw, gt_edge_vis, _ = compute_sobel_edge_map(gt_array) if gt_array is not None else (None, None, 0.0)

    # Compute raw 10 input characteristic features (INPUT ONLY)
    raw_indices = model.indexer.compute_indices(lr_array)
    categories = ["EDGE_DOMINANT", "TEXTURE_DOMINANT", "NOISE_DOMINANT", "SMOOTH_LOW_CONTRAST", "SPARSE_FEATURE"]
    dominant_cat = categories[int(np.argmax(routing_probs_arr))]
    routing_dict = {cat: float(prob) for cat, prob in zip(categories, routing_probs_arr)}


    # =========================================================================
    # SECTION 1: DEDICATED AIR-NET V3 RESTORATION
    # =========================================================================
    st.markdown("---")
    st.subheader("AIR-Net v3 Restoration")

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.markdown("**Input Image (128×128)**")
        st.image(lr_array, caption=f"NoisyLR Input: {selected_sample_name} (128×128)", use_container_width=True, clamp=True)
    with res_col2:
        st.markdown("**AIR-Net v3 Restored (256×256)**")
        st.image(restored_vis, caption=f"AIR-Net v3 Restored Output (256×256) | Range: [{r_min:.3f}, {r_max:.3f}]", use_container_width=True, clamp=True)


    # =========================================================================
    # SECTION 2: MAIN COMPARISON GRID
    # =========================================================================
    st.markdown("---")
    st.subheader("Restoration Comparison Grid")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("### NoisyLR")
        st.caption("128×128")
        st.image(lr_array, use_container_width=True, clamp=True)
    with c2:
        st.markdown("### Bicubic")
        st.caption("256×256")
        st.image(bicubic_pred, use_container_width=True, clamp=True)
    with c3:
        st.markdown("### AIR-Net v3")
        st.caption("256×256")
        st.image(restored_vis, use_container_width=True, clamp=True)
    with c4:
        st.markdown("### Ground Truth")
        st.caption("256×256")
        if gt_array is not None:
            st.image(gt_array, use_container_width=True, clamp=True)
        else:
            st.info("Ground Truth N/A")


    # =========================================================================
    # SECTION 3: SOBEL EDGE MAP ANALYSIS
    # =========================================================================
    st.markdown("---")
    st.subheader("Edge Map Analysis")

    e1, e2, e3, e4 = st.columns(4)
    with e1:
        st.markdown("**Input Edge Map**")
        st.caption("128×128 Native Resolution")
        st.image(input_edge_vis, use_container_width=True, clamp=True)
    with e2:
        st.markdown("**Bicubic Edge Map**")
        st.caption("256×256 Native Resolution")
        st.image(bicubic_edge_vis, use_container_width=True, clamp=True)
    with e3:
        st.markdown("**AIR-Net v3 Edge Map**")
        st.caption("256×256 Native Resolution")
        st.image(v3_edge_vis, use_container_width=True, clamp=True)
    with e4:
        st.markdown("**Ground Truth Edge Map**")
        st.caption("256×256 Native Resolution")
        if gt_edge_vis is not None:
            st.image(gt_edge_vis, use_container_width=True, clamp=True)
        else:
            st.info("No GT Edge Map")


    # =========================================================================
    # SECTION 4: INPUT -> OUTPUT TRANSFORMATION
    # =========================================================================
    st.markdown("---")
    st.subheader("Input → Output Transformation")

    t1, t2, t3 = st.columns([1, 1, 1])
    with t1:
        st.markdown("**Input Semiconductor Image**")
        st.caption("128×128 Resolution")
        st.image(lr_array, use_container_width=True, clamp=True)
    with t2:
        st.markdown("### ➔ AIR-Net v3 ➔")
        st.caption("Shared Restormer Backbone + 5 Experts")
        st.info(f"Category: **{dominant_cat}**\n\nParameters: **{num_params:,}**")
    with t3:
        st.markdown("**Restored Output Image**")
        st.caption("256×256 Super-Resolution Target")
        st.image(restored_vis, use_container_width=True, clamp=True)


    # =========================================================================
    # SECTION 5: WHY THIS CATEGORY? (HIGH-CONTRAST VISIBLE CARD)
    # =========================================================================
    st.markdown("---")
    st.subheader("Why This Category?")

    explanation_md = explain_category_routing(raw_indices, dominant_cat, routing_dict)
    st.markdown(f"""
        <div class="high-contrast-card">
            <div class="card-heading">CLASSIFICATION ANALYSIS: {dominant_cat}</div>
            <div class="card-text">
                {explanation_md}
            </div>
        </div>
    """, unsafe_allow_html=True)

    why_c1, why_c2 = st.columns([1, 1])
    with why_c1:
        st.markdown("**Soft Adaptive Routing Distribution**")
        rout_df = pd.DataFrame({
            "Category": categories,
            "Probability (%)": [round(p * 100, 2) for p in routing_probs_arr]
        })
        st.bar_chart(rout_df.set_index("Category"))

    with why_c2:
        st.markdown("**10 Characteristic Input Features (Input-Only)**")
        idx_cols1 = st.columns(5)
        keys_10 = list(raw_indices.keys())
        for i, k in enumerate(keys_10[:5]):
            idx_cols1[i].metric(k.replace("_", " ").title(), f"{raw_indices[k]:.4f}")
        idx_cols2 = st.columns(5)
        for i, k in enumerate(keys_10[5:]):
            idx_cols2[i].metric(k.replace("_", " ").title(), f"{raw_indices[k]:.4f}")


    # =========================================================================
    # SECTION 6: QUANTITATIVE FIDELITY METRICS
    # =========================================================================
    st.markdown("---")
    st.subheader("Quantitative Fidelity Metrics")

    if is_validation_sample and gt_array is not None:
        st.caption("Evaluated against authoritative 256×256 Ground Truth target.")
        try:
            m_bic = {"PSNR (dB)": compute_psnr(bicubic_pred, gt_array), "SSIM": compute_ssim(bicubic_pred, gt_array), "LPIPS": compute_lpips_safe(bicubic_pred, gt_array), "MSE": compute_mse(bicubic_pred, gt_array), "MAE": compute_mae(bicubic_pred, gt_array)}
            m_v3 = {"PSNR (dB)": compute_psnr(restored_raw, gt_array), "SSIM": compute_ssim(restored_raw, gt_array), "LPIPS": compute_lpips_safe(restored_raw, gt_array), "MSE": compute_mse(restored_raw, gt_array), "MAE": compute_mae(restored_raw, gt_array)}

            edge_err_v3 = float(np.mean(np.abs(v3_edge_raw - gt_edge_raw)))
            edge_err_bic = float(np.mean(np.abs(bicubic_edge_raw - gt_edge_raw)))

            metrics_df = pd.DataFrame([
                {"Model": "Bicubic 2x", "PSNR (dB)": f"{m_bic['PSNR (dB)']:.4f}", "SSIM": f"{m_bic['SSIM']:.4f}", "LPIPS": f"{m_bic['LPIPS']:.4f}", "MSE": f"{m_bic['MSE']:.6f}", "MAE": f"{m_bic['MAE']:.6f}", "Edge Error": f"{edge_err_bic:.6f}"},
                {"Model": "AIR-Net v3", "PSNR (dB)": f"{m_v3['PSNR (dB)']:.4f}", "SSIM": f"{m_v3['SSIM']:.4f}", "LPIPS": f"{m_v3['LPIPS']:.4f}", "MSE": f"{m_v3['MSE']:.6f}", "MAE": f"{m_v3['MAE']:.6f}", "Edge Error": f"{edge_err_v3:.6f}"}
            ])
            st.table(metrics_df)
        except ValueError as shape_err:
            st.error(f"Shape validation error: {shape_err}")
    else:
        st.warning("Ground Truth unavailable — quantitative fidelity metrics cannot be calculated for this manual upload.")


    # =========================================================================
    # SECTION 7: DEBUG & RUNTIME INFORMATION EXPANDER
    # =========================================================================
    st.markdown("---")
    with st.expander("🛠️ AIR-Net v3 Debug & Runtime Information"):
        st.json({
            "checkpoint_path": loaded_path if loaded_path else "INITIALIZED_WEIGHTS",
            "checkpoint_loaded": loaded_path is not None,
            "parameter_count": f"{num_params:,}",
            "device": str(DEVICE),
            "input_shape": list(lr_tensor.shape),
            "output_shape": list(v3_out["restored"].shape),
            "output_dtype": str(restored_raw.dtype),
            "output_min": f"{r_min:.6f}",
            "output_max": f"{r_max:.6f}",
            "output_mean": f"{r_mean:.6f}"
        })

    # Download Button
    buf_img = Image.fromarray((restored_vis * 255.0).round().astype(np.uint8))
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
