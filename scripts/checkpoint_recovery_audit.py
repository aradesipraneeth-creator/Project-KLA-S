from utils.device import get_device
from models.airnet import AIRNet
from configs.config import Config
import os
import sys
import glob
import hashlib
import time
import json
import torch

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def get_sha256(filepath: str) -> str:
    """Computes SHA256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def scan_for_checkpoints():
    print("====================================================")
    print("AIR-NET V1.2 — FOCUSED CHECKPOINT RECOVERY SEARCH")
    print("====================================================")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    search_dirs = [
        project_root,
        os.path.join(project_root, "outputs"),
        os.path.join(project_root, "outputs", "checkpoints"),
        os.path.join(project_root, "outputs", "v1_1"),
        os.path.join(project_root, "outputs", "v1_1", "checkpoints"),
        os.path.join(project_root, "outputs", "v1_2"),
        os.path.join(project_root, "outputs", "v1_2", "checkpoints"),
        "C:\\Users\\arade\\Downloads",
        "C:\\Users\\arade\\Desktop",
    ]

    found_files = set()
    for d in search_dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith(".pth") or f.endswith(".pt"):
                    found_files.add(os.path.abspath(os.path.join(d, f)))

    candidates = sorted(list(found_files))
    print(
        f"Found {len(candidates)} candidate checkpoint file(s) across target locations.\n"
    )

    recovery_dir = os.path.join("outputs", "stage1", "checkpoint_recovery")
    os.makedirs(recovery_dir, exist_ok=True)

    # 1. Quarantine Suspected Random Checkpoint FIRST if it exists
    quarantine_dir = os.path.join("outputs", "v1_2", "checkpoints", "quarantine")
    os.makedirs(quarantine_dir, exist_ok=True)
    quarantine_txt_path = os.path.join(recovery_dir, "quarantine_report.txt")

    suspect_path = os.path.join(
        "outputs", "v1_2", "checkpoints", "airnet_v1_2_ema_best_model.pth"
    )
    quarantined_path = os.path.join(
        quarantine_dir, "airnet_v1_2_ema_best_model_RANDOM_UNTRAINED.pth"
    )

    if os.path.exists(suspect_path):
        os.replace(suspect_path, quarantined_path)
        print(f"Quarantined suspect checkpoint to: {quarantined_path}")

        q_report_text = (
            "====================================================\n"
            "AIR-NET V1.2 CHECKPOINT QUARANTINE REPORT\n"
            "====================================================\n\n"
            f"Original Path: {suspect_path}\n"
            f"New Path:      {quarantined_path}\n"
            f"Size:          {os.path.getsize(quarantined_path):,} bytes\n"
            f"SHA256:        {get_sha256(quarantined_path)}\n"
            "Reason:        Checkpoint was created by fallback logic from randomly initialized\n"
            "               model.state_dict() and is not a genuine trained v1.2 checkpoint.\n"
            "====================================================\n"
        )
        with open(quarantine_txt_path, "w") as f:
            f.write(q_report_text)
        print(f"Saved quarantine report to: {quarantine_txt_path}")

    # Re-scan candidate list after quarantine move
    candidates_post = []
    for d in search_dirs + [quarantine_dir]:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith(".pth") or f.endswith(".pt"):
                    candidates_post.append(os.path.abspath(os.path.join(d, f)))
    candidates_post = sorted(list(set(candidates_post)))

    # 2. Write candidate_checkpoints.txt
    cand_txt_path = os.path.join(recovery_dir, "candidate_checkpoints.txt")
    cand_lines = [
        "====================================================\n",
        "AIR-NET V1.2 CANDIDATE CHECKPOINTS SEARCH REPORT\n",
        "====================================================\n\n",
    ]

    detailed_info = []
    config = Config(MODEL_VERSION="AIR-Net-v1.2")
    device = get_device()

    for idx, path in enumerate(candidates_post, start=1):
        size_bytes = os.path.getsize(path)
        mtime = time.ctime(os.path.getmtime(path))
        sha256 = get_sha256(path)

        cand_lines.append(f"Candidate {idx}:\n")
        cand_lines.append(f"  Path:     {path}\n")
        cand_lines.append(
            f"  Size:     {size_bytes:,} bytes ({size_bytes / (1024*1024):.2f} MB)\n"
        )
        cand_lines.append(f"  Modified: {mtime}\n")
        cand_lines.append(f"  SHA256:   {sha256}\n\n")

        # Inspect internal structure
        param_count = None
        arch_compat = "NO"
        keys_list = []
        epoch_val = "NOT AVAILABLE"
        best_psnr_val = "NOT AVAILABLE"
        best_ssim_val = "NOT AVAILABLE"
        model_version_val = "NOT AVAILABLE"
        ema_status = "NOT AVAILABLE"
        status_label = "UNKNOWN"

        try:
            ckpt_data = torch.load(path, map_location="cpu")
            if isinstance(ckpt_data, dict):
                keys_list = list(ckpt_data.keys())
                epoch_val = str(ckpt_data.get("epoch", "NOT AVAILABLE"))
                best_psnr_val = str(ckpt_data.get("best_psnr", "NOT AVAILABLE"))
                best_ssim_val = str(ckpt_data.get("best_ssim", "NOT AVAILABLE"))
                model_version_val = str(ckpt_data.get("model_version", "NOT AVAILABLE"))

                ema_status = (
                    "AVAILABLE" if "ema_state_dict" in ckpt_data else "NOT AVAILABLE"
                )

                # Check weight parameters
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

            if "quarantine" in path or "RANDOM" in path:
                status_label = "RANDOM_UNTRAINED (QUARANTINED)"
            elif arch_compat == "YES":
                dummy_in = torch.ones(1, 1, 128, 128)
                model.eval()
                with torch.no_grad():
                    out = model(dummy_in)["restored"]
                if out.max().item() < 0.15 and out.mean().item() < 0.05:
                    status_label = "RANDOM_UNTRAINED"
                else:
                    status_label = "TRAINED"

        except Exception as e:
            status_label = f"NON_MODEL_TENSOR: {e}"

        info_dict = {
            "index": idx,
            "path": path,
            "size": size_bytes,
            "modified": mtime,
            "sha256": sha256,
            "arch_compat": arch_compat,
            "param_count": param_count,
            "model_version": model_version_val,
            "epoch": epoch_val,
            "ema_status": ema_status,
            "best_psnr": best_psnr_val,
            "best_ssim": best_ssim_val,
            "status": status_label,
            "keys": keys_list,
        }
        detailed_info.append(info_dict)

    with open(cand_txt_path, "w") as f:
        f.writelines(cand_lines)
    print(f"Saved candidate list report to: {cand_txt_path}")

    # 3. Write checkpoint_identity_report.txt
    ident_txt_path = os.path.join(recovery_dir, "checkpoint_identity_report.txt")
    ident_lines = [
        "====================================================\n",
        "AIR-NET V1.2 CHECKPOINT IDENTITY & DISCOVERY REPORT\n",
        "====================================================\n\n",
    ]

    for item in detailed_info:
        ident_lines.append("----------------------------------------------------\n")
        ident_lines.append(f"CHECKPOINT CANDIDATE #{item['index']}\n")
        ident_lines.append("----------------------------------------------------\n")
        ident_lines.append(f"Path:                    {item['path']}\n")
        ident_lines.append(f"Size:                    {item['size']:,} bytes\n")
        ident_lines.append(f"SHA256:                  {item['sha256']}\n")
        ident_lines.append(f"Architecture compatible: {item['arch_compat']}\n")
        ident_lines.append(f"Parameter count:         {item['param_count']}\n")
        ident_lines.append(f"Model version:           {item['model_version']}\n")
        ident_lines.append(f"Epoch:                   {item['epoch']}\n")
        ident_lines.append(f"EMA available:           {item['ema_status']}\n")
        ident_lines.append(f"Best PSNR:               {item['best_psnr']}\n")
        ident_lines.append(f"Best SSIM:               {item['best_ssim']}\n")
        ident_lines.append(f"Status:                  {item['status']}\n\n")

    with open(ident_txt_path, "w") as f:
        f.writelines(ident_lines)
    print(f"Saved identity report to: {ident_txt_path}")

    # 4. Final Recovery Summary
    trained_found = [item for item in detailed_info if item["status"] == "TRAINED"]
    print("\n---------------- RECOVERY SUMMARY ----------------")
    print(f"Total Candidates Analyzed:          {len(detailed_info)}")
    print(f"Genuine Trained Checkpoints Found:  {len(trained_found)}")
    print("--------------------------------------------------")


if __name__ == "__main__":
    scan_for_checkpoints()
