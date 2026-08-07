import os
from dataclasses import dataclass, field
from typing import List

def resolve_existing_dir(candidates: List[str], default_fallback: str) -> str:
    """Helper function to find the first existing directory from a list of candidate paths."""
    for path in candidates:
        if path and os.path.exists(path) and os.path.isdir(path):
            return os.path.abspath(path)
    return os.path.abspath(default_fallback)

@dataclass
class Config:
    # --- Dataset & Paths ---
    project_root: str = field(default_factory=lambda: os.getcwd())
    train_lr_dir: str = ""
    train_gt_dir: str = ""
    test_lr_dir: str = ""

    # Outputs & Artifacts
    output_dir: str = "outputs"
    checkpoint_dir: str = os.path.join("outputs", "checkpoints")
    vis_dir: str = os.path.join("outputs", "visualizations")
    val_preds_dir: str = os.path.join("outputs", "validation_predictions")
    results_csv: str = os.path.join("outputs", "results.csv")
    train_stats_file: str = os.path.join("outputs", "train_stats.txt")
    bicubic_baseline_file: str = os.path.join("outputs", "bicubic_baseline.txt")
    model_summary_file: str = os.path.join("outputs", "model_summary.txt")
    experiment_info_file: str = os.path.join("outputs", "experiment_info.json")
    benchmark_report_file: str = os.path.join("outputs", "benchmark_report.txt")
    best_metrics_file: str = os.path.join("outputs", "best_metrics.json")
    final_report_file: str = os.path.join("outputs", "final_report.txt")
    test_results_dir: str = os.path.join("outputs", "results", "pred_baseline")

    # --- Dataset Split ---
    seed: int = 42
    total_samples: int = 3200
    train_split: int = 2880
    val_split: int = 320

    # --- Architecture Specifications ---
    in_channels: int = 1
    out_channels: int = 1
    dim: int = 32
    channels: List[int] = field(default_factory=lambda: [32, 64, 128, 192])
    heads: List[int] = field(default_factory=lambda: [1, 2, 4, 6])
    enc_blocks: List[int] = field(default_factory=lambda: [2, 2, 4])
    latent_blocks: int = 8
    dec_blocks: List[int] = field(default_factory=lambda: [4, 2, 2])
    ffn_expansion_factor: float = 2.66

    # --- Training Hyperparameters ---
    batch_size: int = 4
    grad_accum_steps: int = 8  # Effective batch size = 32
    epochs: int = 20
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    min_lr: float = 1e-6
    max_grad_norm: float = 1.0
    ema_decay: float = 0.999
    loss_l1_weight: float = 0.8
    loss_ssim_weight: float = 0.2

    # Fixed 5 validation sample indices for tracking
    fixed_val_indices: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])

    # --- Optimization & Pipeline Control Flags ---
    SKIP_PRECOMPUTATION: bool = True
    AUTO_RESUME: bool = True
    RUN_BENCHMARK_AFTER_TRAINING: bool = True
    VISUALIZATION_INTERVAL: int = 5

    def __post_init__(self):
        root = os.path.abspath(self.project_root)
        script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # 1. Resolve train_lr_dir
        if not self.train_lr_dir:
            lr_candidates = [
                os.path.join(root, "train", "train", "NoisyLR"),
                os.path.join(root, "Train", "train", "NoisyLR"),
                os.path.join(root, "train", "Train", "NoisyLR"),
                os.path.join(root, "Train", "Train", "NoisyLR"),
                os.path.join(root, "train", "NoisyLR"),
                os.path.join(root, "Train", "NoisyLR"),
                os.path.join(script_dir, "train", "train", "NoisyLR"),
                os.path.join(script_dir, "Train", "train", "NoisyLR"),
                os.path.join(script_dir, "train", "NoisyLR"),
                os.path.join(script_dir, "Train", "NoisyLR"),
                os.path.join("train", "train", "NoisyLR"),
            ]
            self.train_lr_dir = resolve_existing_dir(lr_candidates, os.path.join("train", "train", "NoisyLR"))

        # 2. Resolve train_gt_dir
        if not self.train_gt_dir:
            gt_candidates = [
                os.path.join(root, "train", "train", "GT"),
                os.path.join(root, "Train", "train", "GT"),
                os.path.join(root, "train", "Train", "GT"),
                os.path.join(root, "Train", "Train", "GT"),
                os.path.join(root, "train", "GT"),
                os.path.join(root, "Train", "GT"),
                os.path.join(script_dir, "train", "train", "GT"),
                os.path.join(script_dir, "Train", "train", "GT"),
                os.path.join(script_dir, "train", "GT"),
                os.path.join(script_dir, "Train", "GT"),
                os.path.join("train", "train", "GT"),
            ]
            self.train_gt_dir = resolve_existing_dir(gt_candidates, os.path.join("train", "train", "GT"))

        # 3. Resolve test_lr_dir
        if not self.test_lr_dir:
            test_candidates = [
                os.path.join(root, "Test_NoisyLR", "NoisyLR"),
                os.path.join(root, "test_NoisyLR", "NoisyLR"),
                os.path.join(root, "Test_NoisyLR"),
                os.path.join(root, "test_NoisyLR"),
                os.path.join(script_dir, "Test_NoisyLR", "NoisyLR"),
                os.path.join(script_dir, "Test_NoisyLR"),
                os.path.join("Test_NoisyLR", "NoisyLR"),
            ]
            self.test_lr_dir = resolve_existing_dir(test_candidates, os.path.join("Test_NoisyLR", "NoisyLR"))

    def create_dirs(self):
        """Creates all necessary output directories."""
        for d in [
            self.output_dir,
            self.checkpoint_dir,
            self.vis_dir,
            self.val_preds_dir,
            self.test_results_dir,
        ]:
            os.makedirs(d, exist_ok=True)
