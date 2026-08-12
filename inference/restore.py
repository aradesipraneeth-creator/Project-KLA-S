from utils.device import get_device
from models.airnet import AIRNet
from configs.config import Config
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = os.environ.get(
    "KLA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_airnet_v1_2_model(device=None):
    if device is None:
        device = get_device()
    config = Config(MODEL_VERSION="AIR-Net-v1.2")
    model = AIRNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        dim=config.dim,
        channels=config.channels,
        heads=config.heads,
        enc_blocks=config.enc_blocks,
        latent_blocks=config.latent_blocks,
        dec_blocks=config.dec_blocks,
        ffn_expansion_factor=config.ffn_expansion_factor,
    ).to(device)

    # Checkpoint resolution priority
    ckpt_candidates = [
        os.path.join(
            PROJECT_ROOT,
            "outputs",
            "stage3",
            "checkpoints",
            "airnet_v1_2_ema_best_model.pth",
        ),
        os.path.join(
            PROJECT_ROOT,
            "outputs",
            "stage3",
            "checkpoints",
            "airnet_v1_2_best_model.pth",
        ),
        os.path.join(
            PROJECT_ROOT,
            "outputs",
            "v1_2",
            "checkpoints",
            "airnet_v1_2_ema_best_model.pth",
        ),
    ]

    loaded_path = None
    for cand in ckpt_candidates:
        if os.path.exists(cand) and "quarantine" not in cand:
            ckpt_data = torch.load(cand, map_location=device)
            state_dict = ckpt_data.get(
                "ema_state_dict", ckpt_data.get("model_state_dict", ckpt_data)
            )
            model.load_state_dict(state_dict, strict=True)
            loaded_path = cand
            break

    model.eval()
    return model, loaded_path, device


def restore_image(input_image_path_or_array, save_path=None):
    """
    Standalone AIR-Net v1.2 restoration pipeline:
      Input: 128x128 grayscale image (path to file or numpy array)
      Output: 256x256 restored uint8 image array (and saved PNG if save_path provided)
    """
    model, loaded_ckpt, device = load_airnet_v1_2_model()

    # 1. Load and Preprocess Input
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
        raise ValueError(
            "Unsupported input format. Expected file path (str) or numpy array."
        )

    # Squeeze to 2D
    if arr.ndim == 3:
        arr = arr.squeeze()

    h, w = arr.shape
    if h != 128 or w != 128:
        raise ValueError(
            f"AIR-Net v1.2 strictly requires 128x128 input images, got {w}x{h}."
        )

    # 2. Tensor Prep
    img_tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

    # 3. Inference Mode
    with torch.no_grad(), torch.inference_mode():
        out_dict = model(img_tensor)
        pred_tensor = out_dict["restored"] if isinstance(out_dict, dict) else out_dict
        pred_tensor = torch.clamp(pred_tensor, 0.0, 1.0)

    restored_np = pred_tensor.squeeze().cpu().numpy()
    assert restored_np.shape == (
        256,
        256,
    ), f"Expected 256x256 output, got {restored_np.shape}"

    restored_uint8 = (restored_np * 255.0).round().astype(np.uint8)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        Image.fromarray(restored_uint8).save(save_path)
        print(f"[OK] Restored 256x256 image saved to: {save_path}")

    return restored_uint8, restored_np


if __name__ == "__main__":
    if len(sys.argv) > 1:
        inp_path = sys.argv[1]
        out_path = sys.argv[2] if len(sys.argv) > 2 else "restored_256.png"
        restore_image(inp_path, out_path)
        print("Status: RESTORATION SUCCESSFUL")
    else:
        print(
            "Usage: python inference/restore.py <path_to_128x128_image> [output_path]"
        )
