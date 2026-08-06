import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.config import Config
from datasets.kla_dataset import get_train_val_datasets
from models.airnet import AIRNet
from losses.hybrid_loss import AIRNetHybridLoss
from utils import (
    ModelEMA,
    calculate_psnr,
    calculate_ssim,
    CSVLogger,
    print_epoch_summary,
    save_json,
    generate_dataset_stats,
    compute_bicubic_baseline,
    save_visualizations_and_predictions,
    generate_model_summary,
    generate_experiment_info,
    run_inference_benchmark
)

def set_seed(seed: int = 42):
    """Sets fixed random seed across python, numpy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(
    model: nn.Module,
    ema: ModelEMA,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    grad_accum_steps: int = 8,
    max_grad_norm: float = 1.0
) -> float:
    """Runs one training epoch with Mixed Precision and Gradient Accumulation."""
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    for step, (lr_batch, gt_batch, _) in enumerate(dataloader, start=1):
        lr_batch = lr_batch.to(device)
        gt_batch = gt_batch.to(device)

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            out_dict = model(lr_batch)
            loss = criterion(out_dict, gt_batch)
            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()
        running_loss += loss.item() * grad_accum_steps

        if step % grad_accum_steps == 0 or step == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            ema.update(model)

    return running_loss / len(dataloader)

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
):
    """Evaluates model on validation set and computes Loss, PSNR, and SSIM."""
    model.eval()
    running_loss = 0.0
    psnr_list = []
    ssim_list = []

    with torch.no_grad():
        for lr_batch, gt_batch, _ in dataloader:
            lr_batch = lr_batch.to(device)
            gt_batch = gt_batch.to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                out_dict = model(lr_batch)
                loss = criterion(out_dict, gt_batch)

            running_loss += loss.item()
            pred_img = out_dict["restored"] if isinstance(out_dict, dict) else out_dict
            pred_clamped = torch.clamp(pred_img, 0.0, 1.0)

            psnr_val = calculate_psnr(pred_clamped, gt_batch, data_range=1.0)
            ssim_val = calculate_ssim(pred_clamped, gt_batch, data_range=1.0)

            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)

    avg_loss = running_loss / len(dataloader)
    avg_psnr = float(sum(psnr_list) / len(psnr_list))
    avg_ssim = float(sum(ssim_list) / len(ssim_list))

    return avg_loss, avg_psnr, avg_ssim

def main():
    config = Config()
    config.create_dirs()
    set_seed(config.seed)

    print("====================================================")
    print("KLA AIR-NET V1 SEMICONDUCTOR RESTORATION TRAINING PIPELINE")
    print("====================================================")

    # 1. Generate Dataset Sanity Report
    generate_dataset_stats(config)

    # 2. Compute Bicubic Validation Baseline
    bicubic_psnr, bicubic_ssim = compute_bicubic_baseline(config)

    # 3. Load Datasets and DataLoaders
    train_dataset, val_dataset = get_train_val_datasets(
        train_lr_dir=config.train_lr_dir,
        train_gt_dir=config.train_gt_dir,
        seed=config.seed,
        train_split=config.train_split,
        val_split=config.val_split
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # 4. Initialize AIR-Net v1 Model, EMA, Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")

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

    ema = ModelEMA(model, decay=config.ema_decay)

    # 5. Generate Model Summary and Reproducibility Metadata
    generate_model_summary(config, model)
    generate_experiment_info(config)

    # 6. Loss, Optimizer, Scheduler, AMP Scaler, Logger
    criterion = AIRNetHybridLoss(
        l1_weight=0.60,
        ssim_weight=0.25,
        edge_weight=0.15,
        use_lpips=False,
        data_range=1.0
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.min_lr
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    csv_logger = CSVLogger(config.results_csv)

    # Tracking Variables
    best_psnr = -1.0
    best_ssim = -1.0
    best_epoch = -1

    epoch_times = []
    start_total_time = time.time()

    print(f"\nStarting {config.epochs} Epoch Training Loop...")
    for epoch in range(1, config.epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(
            model=model,
            ema=ema,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_accum_steps=config.grad_accum_steps,
            max_grad_norm=config.max_grad_norm
        )

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # Evaluate EMA model on validation set
        val_loss, ema_psnr, ema_ssim = evaluate(
            model=ema.ema_model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )

        t1 = time.time()
        epoch_dur = t1 - t0
        epoch_times.append(epoch_dur)

        # Log to CSV
        csv_logger.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            psnr=ema_psnr,
            ssim=ema_ssim,
            lr=current_lr
        )

        # Console Output
        print_epoch_summary(
            epoch=epoch,
            total_epochs=config.epochs,
            train_loss=train_loss,
            val_loss=val_loss,
            psnr=ema_psnr,
            ssim=ema_ssim,
            lr=current_lr
        )

        # Visualizations & Prediction Dumps (using EMA model)
        save_visualizations_and_predictions(
            model=ema.ema_model,
            val_dataset=val_dataset,
            fixed_indices=config.fixed_val_indices,
            epoch=epoch,
            vis_dir=config.vis_dir,
            val_preds_dir=config.val_preds_dir,
            device=device
        )

        # Save Last Model Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'ema_state_dict': ema.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, os.path.join(config.checkpoint_dir, "last_model.pth"))

        # Best Model Tracking based on EMA PSNR
        if ema_psnr > best_psnr:
            best_psnr = ema_psnr
            best_ssim = ema_ssim
            best_epoch = epoch

            # Save AIR-Net EMA Best Model and standard checkpoints
            torch.save(ema.state_dict(), os.path.join(config.checkpoint_dir, "airnet_ema_best_model.pth"))
            torch.save(ema.state_dict(), os.path.join(config.checkpoint_dir, "ema_best_model.pth"))
            torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, "best_model.pth"))

            # Save Best Metrics JSON
            save_json({
                "best_epoch": best_epoch,
                "best_psnr": round(best_psnr, 4),
                "best_ssim": round(best_ssim, 4),
                "best_model": "airnet_ema_best_model.pth"
            }, config.best_metrics_file)

            print(f"  --> [NEW BEST] Saved airnet_ema_best_model.pth (PSNR: {best_psnr:.4f} dB, SSIM: {best_ssim:.4f})")

    total_training_time = time.time() - start_total_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    # 7. Run Inference Benchmarking
    run_inference_benchmark(config)

    # 8. Generate Final Post-Training Report
    final_report = (
        "====================================================\n"
        "KLA SEMICONDUCTOR AIR-NET V1 - FINAL REPORT\n"
        "====================================================\n"
        f"Total Training Epochs:    {config.epochs}\n"
        f"Total Training Time:      {total_training_time / 60.0:.2f} minutes ({total_training_time:.1f} s)\n"
        f"Average Epoch Time:       {avg_epoch_time:.2f} s\n"
        "----------------------------------------------------\n"
        "METRIC BREAKDOWN & COMPARISON\n"
        "----------------------------------------------------\n"
        f"Bicubic Baseline PSNR:    {bicubic_psnr:.4f} dB\n"
        f"Bicubic Baseline SSIM:    {bicubic_ssim:.4f}\n\n"
        f"Best Epoch:               Epoch {best_epoch}\n"
        f"Best Model PSNR (EMA):    {best_psnr:.4f} dB (Gain over Bicubic: +{best_psnr - bicubic_psnr:.4f} dB)\n"
        f"Best Model SSIM (EMA):    {best_ssim:.4f} (Gain over Bicubic: +{best_ssim - bicubic_ssim:.4f})\n\n"
        f"Final Epoch PSNR (EMA):   {ema_psnr:.4f} dB\n"
        f"Final Epoch SSIM (EMA):   {ema_ssim:.4f}\n"
        "====================================================\n"
    )

    with open(config.final_report_file, "w") as f:
        f.write(final_report)

    print("\nAIR-Net v1 training completed successfully!")
    print(final_report)

if __name__ == "__main__":
    main()
