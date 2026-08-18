"""
train.py -- Production training script for DegradationAwareNAFNet.

KLA / Semicon India 2026 Hackathon
Task: 2x super-resolution + blind denoising (speckle + Gaussian + downsampling)
Model: DegradationAwareNAFNet (32.4M params)
Hardware: NVIDIA RTX 4060 Laptop, 8GB VRAM, CUDA 12.1

Usage:
    python train.py                                     # full training
    python train.py --debug                             # sanity check + 2 epochs
    python train.py --no_wandb                          # disable wandb
    python train.py --resume weights/best_model.pth     # resume from checkpoint
    python train.py --epochs 50 --lr 1e-4 --no_wandb    # quick test
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import sys
import time
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Optional wandb -- training works without it
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

# Project imports
from src.dataset import SemiconDataset, make_dataloaders
from src.losses import CompositeLoss
from src.metrics import RestorationMetrics, MetricTracker
from src.model import DegradationAwareNAFNet, build_model
from src.utils import (
    set_seed, get_device, get_grad_scaler,
    enable_benchmark_mode, CheckpointManager,
    save_visual_comparison, log_wandb_images,
    load_yaml_config, print_model_summary
)


# ==============================================================================
# SYNTHETIC DEGRADATION
# ==============================================================================

def apply_synthetic_degradation(
    gt_batch: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Applies three degradations in random order to a GT batch to produce
    synthetic NoisyLR. Used to augment training with on-the-fly generated pairs.

    The three degradations:
        A: Downsample 2x (bicubic antialias) -- 256x256 -> 128x128
        B: Speckle noise (multiplicative) -- alpha ~ U(0.05, 0.25)
        C: Gaussian noise (additive) -- sigma ~ U(0.01, 0.15)

    Degradation order is randomised per call to simulate unknown real-world order.
    Output values may exceed [0,1] -- this is intentional (matches real NoisyLR).

    Args:
        gt_batch: (B, 1, 256, 256) tensor on GPU, values in [0,1]
        device:   torch.device

    Returns:
        (B, 1, 128, 128) tensor, float32, values may exceed [0,1]
    """
    with torch.no_grad():
        # Sample noise parameters (one per batch, not per image)
        alpha = random.uniform(0.05, 0.25)   # speckle strength
        sigma = random.uniform(0.01, 0.15)   # gaussian strength

        # Track current tensor and spatial resolution
        x = gt_batch.clone()   # (B, 1, 256, 256), float32
        is_256 = True

        # Helper closures for each degradation operation
        def do_downsample(t, currently_256):
            if currently_256:
                t = F.interpolate(t, size=(128, 128),
                                  mode='bicubic', antialias=True)
            return t, False   # now at 128x128

        def do_speckle(t):
            noise = torch.randn_like(t)
            return t * (1.0 + alpha * noise)  # values may exceed [0,1]

        def do_gaussian(t):
            noise = torch.randn_like(t)
            return t + sigma * noise           # values may exceed [0,1]

        # Generate random order: 0=downsample, 1=speckle, 2=gaussian
        order = list(range(3))
        random.shuffle(order)

        # Execute operations in shuffled order
        for op_id in order:
            if op_id == 0:
                x, is_256 = do_downsample(x, is_256)
            elif op_id == 1:
                x = do_speckle(x)
            else:
                x = do_gaussian(x)

        # Safety: if downsample never fired (should not happen, but guard)
        if is_256:
            x, _ = do_downsample(x, True)

        assert x.shape[-2:] == (128, 128), \
            f"Synthetic aug output wrong shape: {x.shape}"

        return x   # (B, 1, 128, 128), float32, values may exceed [0,1]


# ==============================================================================
# VALIDATION
# ==============================================================================

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    metrics_calc: RestorationMetrics,
    device: torch.device,
    epoch: int,
    wandb_run,
    save_dir: str,
    log_images: bool = False
) -> dict:
    """
    Evaluates model on val_loader. Returns dict with psnr, ssim, lpips, loss.
    Logs visual comparisons to disk and optionally to wandb.

    Args:
        model:        The model in training mode (will be set to eval internally).
        val_loader:   Validation DataLoader.
        criterion:    CompositeLoss instance.
        metrics_calc: RestorationMetrics instance.
        device:       torch.device.
        epoch:        Current epoch number (1-indexed for display).
        wandb_run:    wandb run object or None.
        save_dir:     Directory to save visual comparisons.
        log_images:   Whether to save/log visual comparisons this epoch.

    Returns:
        dict with keys: psnr, ssim, lpips, loss (all float).
    """
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    model.eval()

    loss_tracker = MetricTracker(["loss"])
    metric_tracker = MetricTracker(["psnr", "ssim", "lpips"])

    first_noisylr = None
    first_pred = None
    first_gt = None
    first_fnames = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            noisylr = batch["noisylr"].to(device, non_blocking=True)  # (B,1,128,128)
            gt = batch["gt"].to(device, non_blocking=True)             # (B,1,256,256)
            fnames = batch["filename"]

            with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
                pred = model(noisylr)   # (B,1,256,256), may be FP16

            # Convert BOTH to float32 before loss and metrics
            pred_f = pred.float()
            gt_f = gt.float()

            # Compute loss (float32, outside autocast)
            loss, _ = criterion(pred_f, gt_f)

            # Compute metrics
            metrics = metrics_calc.compute_all(pred_f, gt_f)
            metric_tracker.update(metrics, batch_size=noisylr.shape[0])
            loss_tracker.update({"loss": loss.item()}, batch_size=noisylr.shape[0])

            # Store first batch for visualization
            if batch_idx == 0:
                first_noisylr = noisylr.cpu()
                first_pred = pred_f.clamp(0.0, 1.0).cpu()
                first_gt = gt_f.cpu()
                first_fnames = list(fnames)

    avg_metrics = metric_tracker.compute()
    avg_metrics["loss"] = loss_tracker.compute()["loss"]

    # Visual logging
    if log_images and first_pred is not None:
        n_vis = min(4, first_pred.shape[0])
        for i in range(n_vis):
            save_visual_comparison(
                noisylr=first_noisylr[i],
                pred=first_pred[i],
                gt=first_gt[i],
                filename=first_fnames[i],
                save_dir=save_dir,
                epoch=epoch
            )
        log_wandb_images(
            wandb_run=wandb_run,
            noisylr_batch=first_noisylr[:n_vis],
            pred_batch=first_pred[:n_vis],
            gt_batch=first_gt[:n_vis],
            filenames=first_fnames[:n_vis],
            epoch=epoch,
            max_images=4
        )

    model.train()
    return avg_metrics


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def train(args: argparse.Namespace) -> None:
    """Full training loop with FP16, gradient accumulation, wandb, checkpointing."""

    # -------------------------------------------------------------------------
    # 4.1 CONFIG
    # -------------------------------------------------------------------------
    config = load_yaml_config(args.config)
    train_cfg = config.get('train', {})
    data_cfg = config.get('data', {})
    loss_cfg = config.get('loss', {})
    wandb_cfg = config.get('wandb', {})

    # CLI overrides -- only when explicitly passed (not None)
    if args.epochs is not None:
        train_cfg['num_epochs'] = args.epochs
    if args.batch_size is not None:
        train_cfg['batch_size'] = args.batch_size
    if args.lr is not None:
        train_cfg['lr'] = args.lr
    if args.debug:
        train_cfg['num_epochs'] = 2
        train_cfg['val_every_n_epochs'] = 1
        train_cfg['save_every_n_epochs'] = 999   # don't save in debug
        train_cfg['log_images_every'] = 1

    # -------------------------------------------------------------------------
    # 4.2 REPRODUCIBILITY & DEVICE
    # -------------------------------------------------------------------------
    set_seed(train_cfg.get('seed', 42))
    device = get_device(cuda_device_id=0)
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    scaler = get_grad_scaler()
    enable_benchmark_mode()

    # -------------------------------------------------------------------------
    # 4.3 DATA PATHS
    # -------------------------------------------------------------------------
    script_dir = Path(__file__).parent.resolve()
    gt_folder = str((script_dir / data_cfg['gt_folder']).resolve())
    noisylr_folder = str((script_dir / data_cfg['noisylr_folder']).resolve())

    data_config = {
        "gt_folder": gt_folder,
        "noisylr_folder": noisylr_folder,
        "batch_size": train_cfg.get('batch_size', 16),
        "patch_size": train_cfg.get('patch_size', 128),
        "val_split": data_cfg.get('val_split', 0.1),
        "num_workers": 0,
        "seed": train_cfg.get('seed', 42)
    }
    train_loader, val_loader = make_dataloaders(data_config)

    # -------------------------------------------------------------------------
    # 4.4 MODEL
    # -------------------------------------------------------------------------
    model = build_model(config, device)
    print_model_summary(model, device, input_size=(1, 1, 128, 128))

    # -------------------------------------------------------------------------
    # 4.5 LOSS, METRICS, OPTIMIZER, SCHEDULER
    # -------------------------------------------------------------------------
    criterion = CompositeLoss(
        charbonnier_weight=loss_cfg.get('charbonnier_weight', 0.6),
        ssim_weight=loss_cfg.get('ssim_weight', 0.2),
        fft_weight=loss_cfg.get('fft_weight', 0.2),
        eps=loss_cfg.get('charbonnier_eps', 0.001)
    ).to(device)

    metrics_calc = RestorationMetrics(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.get('lr', 3e-4),
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=train_cfg.get('T_0', 50),
        T_mult=train_cfg.get('T_mult', 1),
        eta_min=train_cfg.get('eta_min', 1e-6)
    )

    # -------------------------------------------------------------------------
    # 4.6 CHECKPOINT MANAGER & OPTIONAL RESUME
    # -------------------------------------------------------------------------
    ckpt_manager = CheckpointManager(
        save_dir=str(script_dir / train_cfg.get('save_dir', 'weights')),
        max_keep=3
    )
    start_epoch = 0
    best_psnr = 0.0

    if args.resume:
        # Load ONCE with all states
        ckpt = ckpt_manager.load(
            args.resume, model, optimizer, scheduler, scaler
        )
        start_epoch = ckpt.get('epoch', 0) + 1
        best_psnr = ckpt.get('metrics', {}).get('psnr', 0.0)
        print(f"Resuming from epoch {start_epoch} | "
              f"best PSNR: {best_psnr:.4f} dB")

    # -------------------------------------------------------------------------
    # 4.7 TRAINING CONSTANTS
    # -------------------------------------------------------------------------
    num_epochs = train_cfg.get('num_epochs', 200)
    accumulation_steps = train_cfg.get('accumulation_steps', 2)
    grad_clip_norm = train_cfg.get('grad_clip_norm', 1.0)
    early_stop_patience = train_cfg.get('early_stop_patience', 30)
    val_every = train_cfg.get('val_every_n_epochs', 1)
    save_every = train_cfg.get('save_every_n_epochs', 10)
    log_images_every = train_cfg.get('log_images_every', 10)

    # -------------------------------------------------------------------------
    # 4.8 WANDB
    # -------------------------------------------------------------------------
    wandb_run = None
    if not args.no_wandb and WANDB_AVAILABLE:
        try:
            wandb_run = wandb.init(
                project=wandb_cfg.get('project', 'kla-image-restoration'),
                entity=wandb_cfg.get('entity', None) or None,
                name=args.run_name or f"nafnet_w64_ep{num_epochs}",
                config={
                    "model": config.get('model', {}),
                    "train": train_cfg,
                    "loss": loss_cfg,
                    "dataset": {"train_pairs": 2880, "val_pairs": 320},
                    "hardware": {"gpu": "RTX4060Laptop", "batch_eff": 32}
                },
                resume="allow" if args.resume else None
            )
            print(f"wandb run: {wandb_run.name}")
        except Exception as e:
            print(f"wandb init failed (continuing without): {e}")
            wandb_run = None
    elif not WANDB_AVAILABLE:
        print("wandb not installed -- logging to console only")

    # -------------------------------------------------------------------------
    # 4.9 TRAINING LOOP
    # -------------------------------------------------------------------------
    model.train()
    no_improve_count = 0

    print("")
    print("=" * 60)
    print(f"Starting training: {num_epochs} epochs | "
          f"batch={train_cfg.get('batch_size', 16)} | "
          f"effective_batch={train_cfg.get('batch_size', 16) * accumulation_steps}")
    print("=" * 60)

    for epoch in range(start_epoch, num_epochs):

        epoch_loss = 0.0
        epoch_charb = 0.0
        epoch_ssim_l = 0.0
        epoch_fft_l = 0.0
        n_batches = 0

        optimizer.zero_grad()

        # tqdm progress bar -- ascii=True + dynamic_ncols=False for Windows
        pbar = tqdm(
            train_loader,
            desc=f"Ep {epoch + 1:03d}/{num_epochs}",
            leave=False,
            ncols=100,
            ascii=True,
            dynamic_ncols=False
        )

        last_step_accumulated = False   # track whether final step fired

        for step, batch in enumerate(pbar):
            noisylr = batch["noisylr"].to(device, non_blocking=True)  # (B,1,128,128)
            gt = batch["gt"].to(device, non_blocking=True)             # (B,1,256,256)

            # --- SYNTHETIC AUGMENTATION: 50/50 mixing ---
            # On even steps: generate synthetic NoisyLR from GT on-the-fly
            # On odd steps:  use the real loaded NoisyLR
            # Both cases use the same GT as target
            if step % 2 == 0:
                noisylr = apply_synthetic_degradation(gt, device)

            # --- FORWARD (FP16) ---
            with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
                pred = model(noisylr)   # (B,1,256,256)

            # Loss in float32 (outside autocast for precision)
            loss, loss_dict = criterion(pred.float(), gt.float())
            loss_scaled = loss / accumulation_steps

            # --- BACKWARD ---
            scaler.scale(loss_scaled).backward()

            # --- ACCUMULATION STEP ---
            last_step_accumulated = False
            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                last_step_accumulated = True

            # --- STATS ---
            epoch_loss += loss_dict["total"]
            epoch_charb += loss_dict["charbonnier"]
            epoch_ssim_l += loss_dict["ssim"]
            epoch_fft_l += loss_dict["fft"]
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{loss_dict['total']:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.1e}"
            })

        pbar.close()

        # Fire any leftover gradients at epoch end
        # ONLY if the last step did NOT already fire an accumulation update
        if not last_step_accumulated:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Step scheduler AFTER optimizer step, ONCE per epoch
        # Pass (epoch + 1) so the scheduler correctly tracks completed epochs
        scheduler.step(epoch + 1)

        # --- EPOCH AVERAGES ---
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_charb = epoch_charb / max(n_batches, 1)
        avg_ssim = epoch_ssim_l / max(n_batches, 1)
        avg_fft = epoch_fft_l / max(n_batches, 1)
        current_lr = optimizer.param_groups[0]['lr']

        # --- VALIDATION ---
        val_metrics = None
        is_best = False

        if (epoch + 1) % val_every == 0:
            do_log_images = ((epoch + 1) % log_images_every == 0)
            val_metrics = validate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                metrics_calc=metrics_calc,
                device=device,
                epoch=epoch + 1,
                wandb_run=wandb_run,
                save_dir=str(script_dir / "results" / "sample_outputs"),
                log_images=do_log_images
            )

            val_psnr = val_metrics["psnr"]
            val_ssim = val_metrics["ssim"]
            val_lpips = val_metrics["lpips"]

            is_best = val_psnr > best_psnr
            if is_best:
                best_psnr = val_psnr
                no_improve_count = 0
            else:
                no_improve_count += 1

            print(
                f"Epoch {epoch + 1:03d}/{num_epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"PSNR: {val_psnr:.4f} dB | "
                f"SSIM: {val_ssim:.4f} | "
                f"LPIPS: {val_lpips:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Best: {best_psnr:.4f} dB"
                + (" [BEST]" if is_best else "")
            )

        else:
            print(
                f"Epoch {epoch + 1:03d}/{num_epochs} | "
                f"Loss: {avg_loss:.4f} "
                f"(c={avg_charb:.4f} s={avg_ssim:.4f} f={avg_fft:.4f}) | "
                f"LR: {current_lr:.2e}"
            )

        # --- CHECKPOINT ---
        if val_metrics is not None:
            should_save = ((epoch + 1) % save_every == 0) or is_best
            if should_save:
                ckpt_manager.save(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch + 1,
                    metrics=val_metrics,
                    config=config,
                    is_best=is_best
                )

        # --- WANDB LOGGING ---
        if wandb_run is not None:
            log_dict = {
                "train/loss": avg_loss,
                "train/loss_charbonnier": avg_charb,
                "train/loss_ssim": avg_ssim,
                "train/loss_fft": avg_fft,
                "train/lr": current_lr,
                "epoch": epoch + 1,
            }
            if val_metrics is not None:
                log_dict.update({
                    "val/psnr": val_metrics["psnr"],
                    "val/ssim": val_metrics["ssim"],
                    "val/lpips": val_metrics["lpips"],
                    "val/loss": val_metrics["loss"],
                    "val/best_psnr": best_psnr,
                })
            # Degradation estimate tracking (from last training batch)
            deg_est = model.last_degradation_estimate
            if deg_est is not None:
                log_dict["train/deg_speckle"] = deg_est[:, 0].mean().item()
                log_dict["train/deg_gaussian"] = deg_est[:, 1].mean().item()
                log_dict["train/deg_blur"] = deg_est[:, 2].mean().item()
            wandb_run.log(log_dict, step=epoch + 1)

        # --- EARLY STOPPING ---
        if val_metrics is not None and no_improve_count >= early_stop_patience:
            print(f"Early stopping: no PSNR improvement for "
                  f"{early_stop_patience} epochs")
            print(f"Best val PSNR: {best_psnr:.4f} dB")
            break

    # --- DONE ---
    print("")
    print("=" * 60)
    print("Training complete.")
    print(f"  Best val PSNR: {best_psnr:.4f} dB")
    print(f"  Weights:       {ckpt_manager.save_dir}/best_model.pth")
    print("=" * 60)

    if wandb_run is not None:
        wandb_run.finish()   # use run object, not module-level wandb.finish()


# ==============================================================================
# ARGUMENT PARSER
# ==============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for train.py."""
    parser = argparse.ArgumentParser(
        description="Train DegradationAwareNAFNet for semiconductor image restoration"
    )
    parser.add_argument(
        "--config", type=str, default="configs/nafnet_base.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint .pth to resume training from"
    )
    parser.add_argument(
        "--run_name", type=str, default=None,
        help="wandb run name (auto-generated if not provided)"
    )
    parser.add_argument(
        "--no_wandb", action="store_true", default=False,
        help="Disable wandb logging"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override num_epochs from config"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override batch_size from config"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate from config"
    )
    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="Debug mode: 2 epochs, no wandb, runs sanity check first"
    )
    return parser


# ==============================================================================
# SANITY CHECK
# ==============================================================================

def run_sanity_check(config_path: str = "configs/nafnet_base.yaml") -> None:
    """
    Runs 2 training steps + 1 full validation pass to confirm the entire
    pipeline is correctly wired before launching a long training run.
    Does NOT save checkpoints or log to wandb.
    Exits with sys.exit(1) if any step fails.
    """
    print("=" * 55)
    print("SANITY CHECK: 2 train steps + 1 val pass")
    print("=" * 55)

    try:
        device = get_device(0)
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        set_seed(42)
        print("[OK] Device and seed initialized")

        config = load_yaml_config(config_path)
        script_dir = Path(__file__).parent.resolve()
        gt_folder = str((script_dir / config['data']['gt_folder']).resolve())
        nlr_folder = str((script_dir / config['data']['noisylr_folder']).resolve())
        print("[OK] Config loaded")

        data_cfg = {
            "gt_folder": gt_folder,
            "noisylr_folder": nlr_folder,
            "batch_size": 4,
            "patch_size": 128,
            "val_split": 0.1,
            "num_workers": 0,
            "seed": 42
        }
        train_loader, val_loader = make_dataloaders(data_cfg)
        print("[OK] DataLoaders created")

        model = build_model(config, device)
        criterion = CompositeLoss().to(device)
        metrics_c = RestorationMetrics(device)
        scaler = get_grad_scaler()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        print("[OK] Model, loss, metrics, optimizer, scaler ready")

        # 2 training steps with gradient accumulation
        model.train()
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            if step >= 2:
                break
            noisylr = batch["noisylr"].to(device)
            gt = batch["gt"].to(device)

            # Test synthetic augmentation on step 0
            if step == 0:
                synth = apply_synthetic_degradation(gt, device)
                assert synth.shape == (noisylr.shape[0], 1, 128, 128), \
                    f"Synthetic aug shape wrong: {synth.shape}"
                noisylr = synth
                print(f"  [OK] Synthetic augmentation: shape={tuple(synth.shape)} "
                      f"range=[{synth.min():.3f}, {synth.max():.3f}]")

            with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
                pred = model(noisylr)
                loss, ldict = criterion(pred.float(), gt.float())
            scaler.scale(loss / 2).backward()

            if (step + 1) % 2 == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            print(f"  [OK] Train step {step + 1}/2: "
                  f"loss={ldict['total']:.4f} "
                  f"(c={ldict['charbonnier']:.4f} "
                  f"s={ldict['ssim']:.4f} "
                  f"f={ldict['fft']:.4f})")

        print("[OK] 2 training steps completed with gradient accumulation")

        # 1 validation pass
        save_dir_sanity = str(script_dir / "results" / "sanity_check")
        val_m = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            metrics_calc=metrics_c,
            device=device,
            epoch=0,
            wandb_run=None,
            save_dir=save_dir_sanity,
            log_images=True
        )
        print(f"[OK] Validation: "
              f"PSNR={val_m['psnr']:.4f} dB | "
              f"SSIM={val_m['ssim']:.4f} | "
              f"LPIPS={val_m['lpips']:.4f} | "
              f"Loss={val_m['loss']:.4f}")

        # Verify visual output was saved
        sanity_dir = Path(save_dir_sanity)
        png_files = list(sanity_dir.glob("*.png")) if sanity_dir.exists() else []
        if png_files:
            print(f"[OK] Visual comparisons saved: {len(png_files)} PNG(s) "
                  f"in results/sanity_check/")
        else:
            print("[WARN] No PNG files found in results/sanity_check/ "
                  "-- check save_visual_comparison")

        print("")
        print("=" * 55)
        print("SANITY CHECK PASSED [OK]")
        print("=" * 55)
        print("")
        print("Full training commands:")
        print("  python train.py                              # standard run")
        print("  python train.py --no_wandb                  # disable wandb")
        print("  python train.py --resume weights/best_model.pth  # resume")
        print("  python train.py --epochs 50 --no_wandb      # quick test")

    except Exception as e:
        import traceback
        print(f"\nSANITY CHECK FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.debug:
        args.no_wandb = True
        print("DEBUG MODE: running sanity check then 2 training epochs")
        print("")
        run_sanity_check(args.config)

    train(args)
