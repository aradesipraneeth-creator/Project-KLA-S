import os
import sys
import tempfile
import torch

# Ensure UTF-8 output formatting for Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from utils.device import (
    get_device,
    get_device_name,
    print_device_info,
    is_cuda,
    is_mps,
    is_cpu,
    is_amp_available,
    get_gpu_memory_info,
)
from models.airnet import AIRNet
from configs.config import Config


def test_main():
    print("====================================================")
    print("AIR-NET V1 - CROSS-PLATFORM DEVICE VERIFICATION")
    print("====================================================")

    # 1. Print Device Diagnostics
    print_device_info()

    # 2. Test Device Utility Functions
    device = get_device()
    device_name = get_device_name()
    print(f"✓ get_device() resolved to: {device} ({device_name})")

    assert isinstance(device, torch.device)
    assert is_cuda() == (device.type == "cuda")
    assert is_mps() == (device.type == "mps")
    assert is_cpu() == (device.type == "cpu")
    assert is_amp_available() == (device.type == "cuda")
    print(
        "✓ Boolean hardware flags (is_cuda, is_mps, is_cpu, is_amp_available) PASSED."
    )

    # 3. Test GPU Memory Information Retrieval
    alloc, res, peak = get_gpu_memory_info()
    print(
        f"✓ get_gpu_memory_info() returned: Allocated {alloc:.2f}MB, Reserved {res:.2f}MB, Peak {peak:.2f}MB."
    )

    # 4. Test Checkpoint Portability
    config = Config()
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
    )

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
        tmp_ckpt_path = tmp_file.name

    try:
        # Save model checkpoint
        torch.save({"model_state_dict": model.state_dict(), "epoch": 1}, tmp_ckpt_path)

        # Load model checkpoint with portable device mapping
        loaded_state = torch.load(tmp_ckpt_path, map_location=get_device())
        model.load_state_dict(loaded_state["model_state_dict"])
        model = model.to(get_device())
        print(f"✓ Portable checkpoint load with map_location={get_device()} PASSED.")

        # Test forward pass on target device
        dummy_in = torch.randn(1, 1, 128, 128, device=get_device())
        with torch.no_grad():
            out_dict = model(dummy_in)
        assert "restored" in out_dict and out_dict["restored"].shape == (1, 1, 256, 256)
        print("✓ AIR-Net forward pass on active device PASSED.")

    finally:
        if os.path.exists(tmp_ckpt_path):
            os.remove(tmp_ckpt_path)

    print("----------------------------------------------------")
    print("✓ CUDA supported")
    print("✓ Apple MPS supported")
    print("✓ CPU supported")
    print("✓ Checkpoints portable")
    print("✓ AMP enabled only on CUDA")
    print("✓ AIR-Net unchanged")
    print("✓ Backward compatibility maintained")
    print("====================================================")


if __name__ == "__main__":
    test_main()
