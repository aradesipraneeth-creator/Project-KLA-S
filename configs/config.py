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
    # --- Experiment & Model Version ---
    MODEL_VERSION: str = "AIR-Net-v1"

    # Loss weights for AIR-Net v1 (Sum = 1.00)
    L1_WEIGHT: float = 0.60
    SSIM_WEIGHT: float = 0.25
    EDGE_WEIGHT: float = 0.15

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
    loss_l1_weight: float = 0.80
    loss_ssim_weight: float = 0.15
    loss_edge_weight: float = 0.05

    # Fixed 5 validation sample indices for tracking
    fixed_val_indices: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])

    # --- Optimization & Pipeline Control Flags ---
    SKIP_PRECOMPUTATION: bool = True
    AUTO_RESUME: bool = True
    RUN_BENCHMARK_AFTER_TRAINING: bool = True
    VISUALIZATION_INTERVAL: int = 5
    USE_GRADIENT_CHECKPOINTING: bool = False
    VALIDATE_EVERY: int = 1
    EARLY_STOPPING: bool = False
    EARLY_STOP_PATIENCE: int = 20
    EXPORT_ONNX: bool = False
    USE_TORCH_COMPILE: bool = False
    COPY_DATASET_TO_LOCAL: bool = True
    USE_TTA: bool = False

    def __post_init__(self):
        root = os.path.abspath(self.project_root)
        script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # Dynamic output routing for AIR-Net versions
        if self.MODEL_VERSION == "AIR-Net-v1.1":
            self.output_dir = os.path.join("outputs", "v1_1")
        elif self.MODEL_VERSION == "AIR-Net-v1.2":
            self.output_dir = os.path.join("outputs", "v1_2")

        if self.MODEL_VERSION in ["AIR-Net-v1.1", "AIR-Net-v1.2"]:
            self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
            self.vis_dir = os.path.join(self.output_dir, "visualizations")
            self.val_preds_dir = os.path.join(self.output_dir, "validation_predictions")
            self.results_csv = os.path.join(self.output_dir, "results", "results.csv")
            self.train_stats_file = os.path.join(self.output_dir, "reports", "train_stats.txt")
            self.bicubic_baseline_file = os.path.join(self.output_dir, "reports", "bicubic_baseline.txt")
            self.model_summary_file = os.path.join(self.output_dir, "reports", "model_summary.txt")
            self.experiment_info_file = os.path.join(self.output_dir, "reports", "experiment_info.json")
            self.benchmark_report_file = os.path.join(self.output_dir, "reports", "benchmark_report.txt")
            self.best_metrics_file = os.path.join(self.output_dir, "reports", "best_metrics.json")
            self.final_report_file = os.path.join(self.output_dir, "reports", "final_report.txt")
            self.test_results_dir = os.path.join(self.output_dir, "results", "pred_baseline")

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
        reports_dir = os.path.join(self.output_dir, "reports")
        results_dir = os.path.join(self.output_dir, "results")
        for d in [
            self.output_dir,
            self.checkpoint_dir,
            self.vis_dir,
            self.val_preds_dir,
            self.test_results_dir,
            reports_dir,
            results_dir
        ]:
            os.makedirs(d, exist_ok=True)
