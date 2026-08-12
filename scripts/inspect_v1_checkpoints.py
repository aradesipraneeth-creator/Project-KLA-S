from utils.device import get_device
from models.airnet import AIRNet
from configs.config import Config
import os
import sys
import glob
import time
import torch

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    print("====================================================")
    print("AIR-NET V1 — CHECKPOINT INVENTORY & RESUME AUDIT")
    print("====================================================")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    outputs_dir = os.path.join(project_root, "outputs")

    pth_files = []
    if os.path.exists(outputs_dir):
        for root, dirs, files in os.walk(outputs_dir):
            for f in files:
                if f.endswith(".pth") or f.endswith(".pt"):
                    pth_files.append(os.path.join(root, f))

    print(f"Found {len(pth_files)} checkpoint file(s) in outputs/ directory:\n")

    config = Config(MODEL_VERSION="AIR-Net-v1.0")

    inventory = []
    for idx, path in enumerate(pth_files, start=1):
        rel_path = os.path.relpath(path, project_root)
        size_bytes = os.path.getsize(path)
        mtime = time.ctime(os.path.getmtime(path))

        epoch_val = "NOT AVAILABLE"
        best_psnr_val = "NOT AVAILABLE"
        best_ssim_val = "NOT AVAILABLE"
        model_version_val = "NOT AVAILABLE"
        ema_status = "NOT AVAILABLE"
        opt_status = "NOT AVAILABLE"
        sched_status = "NOT AVAILABLE"
        scaler_status = "NOT AVAILABLE"
        param_count = None
        arch_compat = "NO"

        try:
            ckpt_data = torch.load(path, map_location="cpu")
            if isinstance(ckpt_data, dict):
                epoch_val = ckpt_data.get("epoch", "NOT AVAILABLE")
                best_psnr_val = ckpt_data.get("best_psnr", "NOT AVAILABLE")
                best_ssim_val = ckpt_data.get("best_ssim", "NOT AVAILABLE")
                model_version_val = ckpt_data.get("model_version", "NOT AVAILABLE")

                ema_status = "YES" if "ema_state_dict" in ckpt_data else "NO"
                opt_status = "YES" if "optimizer_state_dict" in ckpt_data else "NO"
                sched_status = "YES" if "scheduler_state_dict" in ckpt_data else "NO"
                scaler_status = "YES" if "scaler_state_dict" in ckpt_data else "NO"

                state_dict = ckpt_data.get(
                    "ema_state_dict", ckpt_data.get("model_state_dict", ckpt_data)
                )
                if isinstance(state_dict, dict):
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
                    load_res = model.load_state_dict(state_dict, strict=False)
                    if (
                        len(load_res.missing_keys) == 0
                        and len(load_res.unexpected_keys) == 0
                    ):
                        arch_compat = "YES"
                        param_count = sum(p.numel() for p in model.parameters())

        except Exception as e:
            arch_compat = f"ERROR: {e}"

        item = {
            "index": idx,
            "path": rel_path,
            "full_path": path,
            "size": size_bytes,
            "modified": mtime,
            "epoch": epoch_val,
            "best_psnr": best_psnr_val,
            "best_ssim": best_ssim_val,
            "model_version": model_version_val,
            "ema": ema_status,
            "optimizer": opt_status,
            "scheduler": sched_status,
            "scaler": scaler_status,
            "param_count": param_count,
            "arch_compat": arch_compat,
        }
        inventory.append(item)

        print(f"[{idx:02d}] Path:     {rel_path}")
        print(f"     Size:     {size_bytes:,} bytes ({size_bytes/(1024*1024):.2f} MB)")
        print(f"     Modified: {mtime}")
        print(
            f"     Epoch:    {epoch_val} | Best PSNR: {best_psnr_val} | Best SSIM: {best_ssim_val}"
        )
        print(
            f"     EMA:      {ema_status} | Opt: {opt_status} | Sched: {sched_status} | Scaler: {scaler_status}"
        )
        params_str = f"{param_count:,}" if param_count is not None else "None"
        print(f"     Params:   {params_str} | Arch Compat: {arch_compat}\n")

    # Select best v1 resume candidate
    valid_candidates = [
        item
        for item in inventory
        if item["arch_compat"] == "YES" and "quarantine" not in item["path"]
    ]

    print("====================================================")
    if valid_candidates:
        # Sort by epoch descending
        valid_candidates.sort(
            key=lambda x: x["epoch"] if isinstance(x["epoch"], int) else -1,
            reverse=True,
        )
        best_cand = valid_candidates[0]
        print(f"CHOSEN V1 RESUME CHECKPOINT:")
        print(f"  Path:       {best_cand['path']}")
        print(f"  Epoch:      {best_cand['epoch']}")
        print(f"  Best PSNR:  {best_cand['best_psnr']}")
        print(f"  Best SSIM:  {best_cand['best_ssim']}")
        print(f"  EMA:        {best_cand['ema']}")
        print(f"  Optimizer:  {best_cand['optimizer']}")
        print(f"  Params:     {best_cand['param_count']:,}")
    else:
        print("NO PRE-EXISTING VALID V1 CHECKPOINT FOUND IN OUTPUTS/")
    print("====================================================")


if __name__ == "__main__":
    main()
