from utils.logger import save_json
from configs.config import Config
import time
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def generate_experiment_info(config: Config) -> dict:
    """
    Saves complete reproducibility metadata to experiment_info.json.
    """
    config.create_dirs()
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_stage": "Stage-1 Baseline (Restormer + 2x PixelShuffle)",
        "seed": config.seed,
        "dataset_split": {
            "total_samples": config.total_samples,
            "train_samples": config.train_split,
            "val_samples": config.val_split,
        },
        "model_config": {
            "in_channels": config.in_channels,
            "out_channels": config.out_channels,
            "dim": config.dim,
            "channels": config.channels,
            "heads": config.heads,
            "enc_blocks": config.enc_blocks,
            "latent_blocks": config.latent_blocks,
            "dec_blocks": config.dec_blocks,
            "ffn_expansion_factor": config.ffn_expansion_factor,
        },
        "training_config": {
            "batch_size": config.batch_size,
            "grad_accum_steps": config.grad_accum_steps,
            "effective_batch_size": config.batch_size * config.grad_accum_steps,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "min_lr": config.min_lr,
            "max_grad_norm": config.max_grad_norm,
            "ema_decay": config.ema_decay,
            "loss_weights": {
                "L1": config.loss_l1_weight,
                "SSIM": config.loss_ssim_weight,
            },
        },
    }

    save_json(metadata, config.experiment_info_file)
    print(f"Reproducibility package saved to {config.experiment_info_file}")
    return metadata


if __name__ == "__main__":
    cfg = Config()
    generate_experiment_info(cfg)
