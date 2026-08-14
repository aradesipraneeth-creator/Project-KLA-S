import os
import sys
import json
import csv
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = os.environ.get("KLA_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import Config
from models.image_indexer import ImageIndexer
from models.adaptive_router import AdaptiveRouter
from utils.device import get_device

def main():
    start_time = time.time()
    device = get_device()
    print("==============================================================================")
    print("AIR-Net v3 — ROUTER DISTRIBUTION DIAGNOSTIC")
    print("==============================================================================")

    config = Config(MODEL_VERSION="AIR-Net-v3")
    indexes_dir = os.path.join(PROJECT_ROOT, "outputs", "v3", "indexes")
    os.makedirs(indexes_dir, exist_ok=True)

    norm_path = os.path.join(indexes_dir, "index_normalization.json")
    norm_params = None
    if os.path.exists(norm_path):
        with open(norm_path, "r") as f:
            norm_params = json.load(f)

    indexer = ImageIndexer(norm_params=norm_params)
    router = AdaptiveRouter(input_dim=10, num_experts=5).to(device)

    v3_ckpt_path = os.path.join(PROJECT_ROOT, "outputs", "v3", "checkpoints", "airnet_v3_ema_best_model.pth")
    if not os.path.exists(v3_ckpt_path):
        v3_ckpt_path = os.path.join(PROJECT_ROOT, "outputs", "v3", "checkpoints", "airnet_v3_best_model.pth")

    if os.path.exists(v3_ckpt_path):
        ckpt_data = torch.load(v3_ckpt_path, map_location=device)
        state_dict = ckpt_data.get("ema_state_dict", ckpt_data.get("model_state_dict", ckpt_data))
        router_state = {k.replace("router.", ""): v for k, v in state_dict.items() if k.startswith("router.")}
        if router_state:
            router.load_state_dict(router_state, strict=False)
            print(f"[OK] Loaded router weights from '{v3_ckpt_path}'")

    router.eval()

    lr_files = sorted([f for f in os.listdir(config.train_lr_dir) if f.endswith(".npy")])
    mapping_csv = os.path.join(PROJECT_ROOT, "outputs", "stage1", "stage1_reconstruction", "authoritative_validation_mapping.csv")
    val_files_set = set()
    if os.path.exists(mapping_csv):
        with open(mapping_csv, "r") as f:
            val_files_set = set(row["filename"] for row in csv.DictReader(f))

    train_files = [f for f in lr_files if f not in val_files_set]
    val_files = [f for f in lr_files if f in val_files_set]

    categories = ["EDGE_DOMINANT", "TEXTURE_DOMINANT", "NOISE_DOMINANT", "SMOOTH_LOW_CONTRAST", "SPARSE_FEATURE"]

    def evaluate_file_list(split_name, file_list):
        print(f"Processing {split_name} ({len(file_list)} files)...")
        all_probs = []
        winner_counts = {c: 0 for c in categories}

        for fname in file_list:
            arr = np.load(os.path.join(config.train_lr_dir, fname)).astype(np.float32)
            raw_idx = indexer.compute_indices(arr)
            norm_vec = indexer.normalize_indices(raw_idx).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = router(norm_vec).squeeze().cpu().numpy()
            all_probs.append(probs)
            win_cat = categories[int(np.argmax(probs))]
            winner_counts[win_cat] += 1

        probs_arr = np.array(all_probs)
        means = np.mean(probs_arr, axis=0)
        medians = np.median(probs_arr, axis=0)
        mins = np.min(probs_arr, axis=0)
        maxs = np.max(probs_arr, axis=0)

        return {
            "split": split_name,
            "count": len(file_list),
            "means": means,
            "medians": medians,
            "mins": mins,
            "maxs": maxs,
            "winner_counts": winner_counts
        }

    train_res = evaluate_file_list("Training Set", train_files)
    val_res = evaluate_file_list("Validation Set", val_files)

    stats_csv_path = os.path.join(indexes_dir, "router_statistics.csv")
    with open(stats_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "category", "winner_count", "winner_pct", "mean_prob", "median_prob", "min_prob", "max_prob"])
        for res in [train_res, val_res]:
            n = res["count"]
            for i, cat in enumerate(categories):
                cnt = res["winner_counts"][cat]
                pct = round(cnt / n * 100, 2)
                writer.writerow([
                    res["split"], cat, cnt, f"{pct}%",
                    round(float(res["means"][i]), 6),
                    round(float(res["medians"][i]), 6),
                    round(float(res["mins"][i]), 6),
                    round(float(res["maxs"][i]), 6)
                ])

    print(f"[OK] Router statistics saved to: '{stats_csv_path}'")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(categories))
    width = 0.35
    train_pcts = [train_res["winner_counts"][c] / train_res["count"] * 100 for c in categories]
    val_pcts = [val_res["winner_counts"][c] / val_res["count"] * 100 for c in categories]

    axes[0].bar(x - width/2, train_pcts, width, label='Training Set (2880)', color='#1f77b4')
    axes[0].bar(x + width/2, val_pcts, width, label='Validation Set (320)', color='#ff7f0e')
    axes[0].set_ylabel('Winner Frequency (%)', fontsize=11, fontweight='bold')
    axes[0].set_title('Router Category Winner Frequency', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=8, fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    axes[1].plot(categories, train_res["means"], marker='o', label='Training Mean Prob', color='#1f77b4', linewidth=2)
    axes[1].plot(categories, val_res["means"], marker='s', label='Validation Mean Prob', color='#ff7f0e', linewidth=2)
    axes[1].set_ylabel('Mean Softmax Probability', fontsize=11, fontweight='bold')
    axes[1].set_title('Mean Softmax Probabilities Across Dataset', fontsize=12, fontweight='bold')
    axes[1].set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=8, fontweight='bold')
    axes[1].legend()
    axes[1].grid(linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_path = os.path.join(indexes_dir, "router_distribution.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Router distribution chart saved to: '{plot_path}' (Elapsed: {time.time()-start_time:.1f}s)")

if __name__ == "__main__":
    main()
