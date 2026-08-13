import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = os.environ.get("KLA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import Config
from models.airnet_v3 import AIRNetV3
from utils.device import get_device

def load_airnet_v3_model(device=None):
    if device is None:
        device = get_device()
    config = Config(MODEL_VERSION="AIR-Net-v3")

    norm_path = os.path.join(PROJECT_ROOT, "outputs", "v3", "indexes", "index_normalization.json")
    norm_params = None
    if os.path.exists(norm_path):
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
    ).to(device)

    ckpt_candidates = [
        os.path.join(PROJECT_ROOT, "outputs", "v3", "checkpoints", "airnet_v3_ema_best_model.pth"),
        os.path.join(PROJECT_ROOT, "outputs", "v3", "checkpoints", "airnet_v3_best_model.pth"),
        os.path.join(PROJECT_ROOT, "outputs", "v2", "checkpoints", "airnet_v2_ema_best_model.pth"),
    ]

    loaded_path = None
    for cand in ckpt_candidates:
        if os.path.exists(cand):
            ckpt_data = torch.load(cand, map_location=device)
            state_dict = ckpt_data.get("ema_state_dict", ckpt_data.get("model_state_dict", ckpt_data))
            try:
                model.load_state_dict(state_dict, strict=False)
                loaded_path = cand
                break
            except Exception as e:
                print(f"Notice loading checkpoint {cand}: {e}")

    model.eval()
    return model, loaded_path, device

def restore_v3_image(input_image_path_or_array, save_path=None):
    """
    Content-Adaptive AIR-Net v3 Restoration Pipeline:
      - Input: 128x128 image (path or numpy array)
      - Computes 10 characteristic indices (INPUT ONLY, no GT required!)
      - Computes soft category routing probabilities
      - Outputs 256x256 restored uint8 image array
    """
    model, loaded_ckpt, device = load_airnet_v3_model()

    if isinstance(input_image_path_or_array, str):
        if input_image_path_or_array.endswith(".npy"):
            arr = np.load(input_image_path_or_array).astype(np.float32)
        else:
            img = Image.open(input_image_path_or_array).convert("L")
            arr = np.array(img, dtype=np.float32) / 255.0
    elif isinstance(input_image_path_or_array, np.ndarray):
        arr = input_image_path_or_array.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
    else:
        raise ValueError("Unsupported input format.")

    if arr.ndim == 3:
        arr = arr.squeeze()

    h, w = arr.shape
    if h != 128 or w != 128:
        raise ValueError(f"AIR-Net v3 strictly requires 128x128 input images, got {w}x{h}.")

    img_tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad(), torch.inference_mode():
        out_dict = model(img_tensor)
        pred_tensor = torch.clamp(out_dict["restored"], 0.0, 1.0)
        routing_probs = out_dict["routing_probs"].squeeze().cpu().numpy()

    categories = ["EDGE", "TEXTURE", "NOISE", "SMOOTH", "SPARSE"]
    routing_info = {cat: round(float(prob), 4) for cat, prob in zip(categories, routing_probs)}
    dominant_cat = categories[int(np.argmax(routing_probs))]

    restored_np = pred_tensor.squeeze().cpu().numpy()
    assert restored_np.shape == (256, 256), f"Expected 256x256 output, got {restored_np.shape}"
    restored_uint8 = (restored_np * 255.0).round().astype(np.uint8)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        Image.fromarray(restored_uint8).save(save_path)
        print(f"[OK] AIR-Net v3 Restored 256x256 saved to: {save_path}")
        print(f"     Dominant Category: {dominant_cat} | Routing: {routing_info}")

    return restored_uint8, restored_np, routing_info

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inp_path = sys.argv[1]
        out_path = sys.argv[2] if len(sys.argv) > 2 else "restored_v3_256.png"
        restore_v3_image(inp_path, out_path)
        print("Status: AIR-NET V3 CONTENT-ADAPTIVE RESTORATION SUCCESSFUL")
    else:
        print("Usage: python inference/restore_v3.py <path_to_128x128_image> [output_path]")
