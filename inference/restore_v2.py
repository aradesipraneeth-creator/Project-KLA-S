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
from models.airnet_v2 import AIRNetV2
from utils.device import get_device

def load_airnet_v2_model(device=None):
    if device is None:
        device = get_device()
    config = Config(MODEL_VERSION="AIR-Net-v2")
    model = AIRNetV2(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        dim=config.dim,
        channels=config.channels,
        heads=config.heads,
        enc_blocks=config.enc_blocks,
        latent_blocks=config.latent_blocks,
        dec_blocks=config.dec_blocks,
        ffn_expansion_factor=config.ffn_expansion_factor,
        use_residual_learning=True
    ).to(device)

    # Checkpoint resolution priority for AIR-Net v2
    ckpt_candidates = [
        os.path.join(PROJECT_ROOT, "outputs", "v2", "checkpoints", "airnet_v2_ema_best_model.pth"),
        os.path.join(PROJECT_ROOT, "outputs", "v2", "checkpoints", "airnet_v2_best_model.pth"),
        os.path.join(PROJECT_ROOT, "outputs", "stage3", "checkpoints", "airnet_v1_2_ema_best_model.pth"),
    ]

    loaded_path = None
    for cand in ckpt_candidates:
        if os.path.exists(cand) and "quarantine" not in cand:
            ckpt_data = torch.load(cand, map_location=device)
            state_dict = ckpt_data.get("ema_state_dict", ckpt_data.get("model_state_dict", ckpt_data))
            try:
                model.load_state_dict(state_dict, strict=True)
                loaded_path = cand
                break
            except Exception as e:
                print(f"Notice loading checkpoint {cand}: {e}")

    model.eval()
    return model, loaded_path, device

def restore_v2_image(input_image_path_or_array, save_path=None):
    """
    AIR-Net v2 Multi-Objective High-Fidelity Restoration Pipeline:
      Input: 128x128 grayscale image (path or numpy array)
      Output: 256x256 restored uint8 image array
    """
    model, loaded_ckpt, device = load_airnet_v2_model()

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
        raise ValueError("Unsupported input format. Expected file path or numpy array.")

    if arr.ndim == 3:
        arr = arr.squeeze()

    h, w = arr.shape
    if h != 128 or w != 128:
        raise ValueError(f"AIR-Net v2 strictly requires 128x128 input images, got {w}x{h}.")

    img_tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad(), torch.inference_mode():
        out_dict = model(img_tensor)
        pred_tensor = out_dict["restored"] if isinstance(out_dict, dict) else out_dict
        pred_tensor = torch.clamp(pred_tensor, 0.0, 1.0)

    restored_np = pred_tensor.squeeze().cpu().numpy()
    assert restored_np.shape == (256, 256), f"Expected 256x256 output, got {restored_np.shape}"
    restored_uint8 = (restored_np * 255.0).round().astype(np.uint8)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        Image.fromarray(restored_uint8).save(save_path)
        print(f"[OK] AIR-Net v2 Restored 256x256 image saved to: {save_path}")

    return restored_uint8, restored_np

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inp_path = sys.argv[1]
        out_path = sys.argv[2] if len(sys.argv) > 2 else "restored_v2_256.png"
        restore_v2_image(inp_path, out_path)
        print("Status: AIR-NET V2 RESTORATION SUCCESSFUL")
    else:
        print("Usage: python inference/restore_v2.py <path_to_128x128_image> [output_path]")
