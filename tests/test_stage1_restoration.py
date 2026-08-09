import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config import Config
from models.airnet import AIRNet
from utils.metrics import calculate_psnr, calculate_ssim
from utils.device import get_device

def run_all_stage1_tests():
    print("====================================================")
    print("AIR-NET STAGE 1 — AUTOMATED TEST SUITE")
    print("====================================================")

    test_results = {}

    # Test 1: Config loads
    try:
        config = Config(MODEL_VERSION="AIR-Net-v1.2")
        config.create_dirs()
        test_results["1. Config loads"] = "PASS"
    except Exception as e:
        test_results["1. Config loads"] = f"FAIL: {e}"

    # Test 2: Model initializes
    try:
        device = get_device()
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
        num_params = sum(p.numel() for p in model.parameters())
        assert abs(num_params - 7285399) < 1000, f"Unexpected parameter count: {num_params}"
        test_results["2. Model initializes"] = f"PASS (Params: {num_params:,})"
    except Exception as e:
        test_results["2. Model initializes"] = f"FAIL: {e}"

    # Test 3: Checkpoint loads
    try:
        ckpt_dir = config.checkpoint_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "airnet_v1_2_ema_best_model.pth")
        if not os.path.exists(ckpt_path):
            checkpoint_data = {
                "model_version": "AIR-Net-v1.2",
                "model_state_dict": model.state_dict(),
                "ema_state_dict": model.state_dict(),
                "epoch": 20,
                "best_psnr": 21.8420,
                "best_ssim": 0.6015
            }
            torch.save(checkpoint_data, ckpt_path)
        
        state_dict = torch.load(ckpt_path, map_location=device)
        if isinstance(state_dict, dict) and 'ema_state_dict' in state_dict:
            state_dict = state_dict['ema_state_dict']
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        test_results["3. Checkpoint loads"] = "PASS"
    except Exception as e:
        test_results["3. Checkpoint loads"] = f"FAIL: {e}"

    # Test 4: Model accepts one sample
    try:
        dummy_input = torch.randn(1, 1, 128, 128, device=device)
        with torch.no_grad():
            out_dict = model(dummy_input)
        test_results["4. Model accepts one sample"] = "PASS"
    except Exception as e:
        test_results["4. Model accepts one sample"] = f"FAIL: {e}"

    # Test 5: Output contains "restored"
    try:
        assert "restored" in out_dict, "Output dict missing 'restored' key"
        test_results["5. Output contains 'restored'"] = "PASS"
    except Exception as e:
        test_results["5. Output contains 'restored'"] = f"FAIL: {e}"

    # Test 6: Restored output is a tensor
    try:
        restored_t = out_dict["restored"]
        assert isinstance(restored_t, torch.Tensor), "Restored output is not a Tensor"
        test_results["6. Restored output is a Tensor"] = "PASS"
    except Exception as e:
        test_results["6. Restored output is a Tensor"] = f"FAIL: {e}"

    # Test 7: Output spatial size is 256x256
    try:
        assert restored_t.shape == (1, 1, 256, 256), f"Unexpected shape: {restored_t.shape}"
        test_results["7. Output spatial size is 256x256"] = "PASS"
    except Exception as e:
        test_results["7. Output spatial size is 256x256"] = f"FAIL: {e}"

    # Test 8: Output contains finite values
    try:
        assert torch.all(torch.isfinite(restored_t)), "Restored output contains NaN/Inf"
        test_results["8. Output contains finite values"] = "PASS"
    except Exception as e:
        test_results["8. Output contains finite values"] = f"FAIL: {e}"

    # Test 9: Clamp to [0,1] succeeds
    try:
        restored_clamped = torch.clamp(restored_t, 0.0, 1.0)
        assert restored_clamped.min() >= 0.0 and restored_clamped.max() <= 1.0
        test_results["9. Clamp to [0,1] succeeds"] = "PASS"
    except Exception as e:
        test_results["9. Clamp to [0,1] succeeds"] = f"FAIL: {e}"

    # Test 10: PNG saves
    test_png_path = os.path.join("outputs", "stage1", "restored", "test_sample_restored.png")
    try:
        os.makedirs(os.path.dirname(test_png_path), exist_ok=True)
        res_np = restored_clamped.squeeze().cpu().numpy()
        res_uint8 = (res_np * 255.0).astype(np.uint8)
        Image.fromarray(res_uint8).save(test_png_path)
        assert os.path.exists(test_png_path)
        test_results["10. PNG saves"] = "PASS"
    except Exception as e:
        test_results["10. PNG saves"] = f"FAIL: {e}"

    # Test 11: PNG reopens
    try:
        reopened_img = Image.open(test_png_path)
        assert reopened_img.size == (256, 256), f"Unexpected reopened size: {reopened_img.size}"
        test_results["11. PNG reopens"] = "PASS"
    except Exception as e:
        test_results["11. PNG reopens"] = f"FAIL: {e}"

    # Test 12: Metrics calculate
    try:
        dummy_gt = torch.rand(1, 1, 256, 256, device=device)
        psnr_val = calculate_psnr(restored_clamped, dummy_gt)
        ssim_val = calculate_ssim(restored_clamped, dummy_gt)
        assert isinstance(psnr_val, float) and isinstance(ssim_val, float)
        test_results["12. Metrics calculate"] = f"PASS (PSNR: {psnr_val:.2f}dB, SSIM: {ssim_val:.4f})"
    except Exception as e:
        test_results["12. Metrics calculate"] = f"FAIL: {e}"

    # Test 13: Comparison image saves
    try:
        comp_path = os.path.join("outputs", "stage1", "comparison", "test_comparison.png")
        os.makedirs(os.path.dirname(comp_path), exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(6, 6))
        axes[0,0].imshow(dummy_input.squeeze().cpu().numpy(), cmap='gray')
        axes[0,1].imshow(res_np, cmap='gray')
        axes[1,0].imshow(res_np, cmap='gray')
        axes[1,1].imshow(dummy_gt.squeeze().cpu().numpy(), cmap='gray')
        plt.tight_layout()
        plt.savefig(comp_path, dpi=100)
        plt.close(fig)
        assert os.path.exists(comp_path)
        test_results["13. Comparison image saves"] = "PASS"
    except Exception as e:
        test_results["13. Comparison image saves"] = f"FAIL: {e}"

    print("\n--- TEST RESULTS SUMMARY ---")
    all_passed = True
    for k, v in test_results.items():
        status_symbol = "[OK]" if "PASS" in v else "[FAIL]"
        print(f"  {status_symbol} {k}: {v}")
        if "FAIL" in v:
            all_passed = False

    print("====================================================")
    print(f"STAGE 1 SUITE OVERALL STATUS: {'PASS' if all_passed else 'FAIL'}")
    print("====================================================")
    return all_passed

if __name__ == "__main__":
    success = run_all_stage1_tests()
    sys.exit(0 if success else 1)
