import os
import argparse
import torch
from torch.utils.data import DataLoader

from configs.config import Config
from datasets.kla_dataset import get_train_val_datasets
from models.restormer_baseline import RestormerBaseline
from losses.hybrid_loss import HybridLoss
from utils import calculate_psnr, calculate_ssim

from utils.device import get_device

def validate_checkpoint(checkpoint_path: str, config: Config = None):
    if config is None:
        config = Config()

    device = get_device()
    print(f"Validating checkpoint: {checkpoint_path} on device: {device}")

    # Load Validation Dataset
    _, val_dataset = get_train_val_datasets(
        train_lr_dir=config.train_lr_dir,
        train_gt_dir=config.train_gt_dir,
        seed=config.seed,
        train_split=config.train_split,
        val_split=config.val_split
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Load Model
    model = RestormerBaseline(
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

    state_dict = torch.load(checkpoint_path, map_location=device)
    if 'ema_state_dict' in state_dict:
        state_dict = state_dict['ema_state_dict']
    elif 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    model.load_state_dict(state_dict)
    model.eval()

    criterion = HybridLoss().to(device)

    running_loss = 0.0
    psnr_list = []
    ssim_list = []

    with torch.no_grad():
        for lr_batch, gt_batch, _ in val_loader:
            lr_batch = lr_batch.to(device)
            gt_batch = gt_batch.to(device)

            pred_batch = model(lr_batch)
            loss = criterion(pred_batch, gt_batch)
            running_loss += loss.item()

            pred_clamped = torch.clamp(pred_batch, 0.0, 1.0)
            psnr_list.append(calculate_psnr(pred_clamped, gt_batch, data_range=1.0))
            ssim_list.append(calculate_ssim(pred_clamped, gt_batch, data_range=1.0))

    avg_loss = running_loss / len(val_loader)
    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)

    print("====================================================")
    print("CHECKPOINT VALIDATION RESULTS")
    print("====================================================")
    print(f"Checkpoint File: {checkpoint_path}")
    print(f"Validation Loss: {avg_loss:.6f}")
    print(f"Average PSNR:    {avg_psnr:.4f} dB")
    print(f"Average SSIM:    {avg_ssim:.4f}")
    print("====================================================")

    return avg_loss, avg_psnr, avg_ssim

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Restormer Baseline Checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join("outputs", "checkpoints", "ema_best_model.pth"),
        help="Path to .pth checkpoint file"
    )
    args = parser.parse_args()
    cfg = Config()
    validate_checkpoint(args.checkpoint, cfg)
