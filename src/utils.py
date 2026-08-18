"""
src/utils.py — Training Utilities for Semiconductor Image Restoration

Provides essential training infrastructure:
  - set_seed():             Full reproducibility (random, numpy, torch, CUDA)
  - get_device():           GPU initialization with detailed logging
  - enable_benchmark_mode(): Switch from deterministic to inference mode
  - get_grad_scaler():      FP16 mixed precision GradScaler
  - CheckpointManager:      Save/load checkpoints with best-model tracking
  - save_visual_comparison(): 3-panel comparison figures (LR | Restored | GT)
  - log_wandb_images():     W&B visual logging with graceful fallback
  - load_yaml_config():     YAML config file loader
  - print_model_summary():  Parameter count and shape verification

Hardware: RTX 4060 Laptop (8GB VRAM), CUDA 12.1, Windows OS
"""

import os
import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — prevents display issues on headless/Windows
import matplotlib.pyplot as plt
import yaml


def set_seed(seed: int = 42) -> None:
    """Sets all random seeds for full reproducibility.

    Called once at the start of train.py before any tensor operations.
    NOTE: sets cudnn.deterministic=True which slightly reduces throughput.
    Switch to enable_benchmark_mode() for inference.

    Args:
        seed: Random seed value (default 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed} -- deterministic training mode enabled")


def get_device(cuda_device_id: int = 0) -> torch.device:
    """Initializes and returns the best available device.

    Prints full GPU specs for logging in training runs.
    RTX 4060 Laptop confirmed: 8GB VRAM, CUDA 12.1, cuDNN 90100.

    Args:
        cuda_device_id: CUDA device index (default 0).

    Returns:
        torch.device for computation.
    """
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available -- falling back to CPU")
        print("Training will be ~17x slower than RTX 4060 GPU")
        return torch.device('cpu')

    torch.cuda.set_device(cuda_device_id)
    device = torch.device(f'cuda:{cuda_device_id}')
    props = torch.cuda.get_device_properties(cuda_device_id)

    total_mb = props.total_memory / 1024**2
    free_mb = (props.total_memory - torch.cuda.memory_allocated(cuda_device_id)) / 1024**2

    print(f"=== DEVICE INITIALIZED ===")
    print(f"  Name:             {props.name}")
    print(f"  VRAM total:       {total_mb:.0f} MB ({total_mb / 1024:.1f} GB)")
    print(f"  VRAM free:        {free_mb:.0f} MB")
    print(f"  CUDA version:     {torch.version.cuda}")
    print(f"  cuDNN version:    {torch.backends.cudnn.version()}")
    print(f"  Compute cap:      {props.major}.{props.minor}")
    print(f"  Multiprocessors:  {props.multi_processor_count}")
    print(f"==========================")
    return device


def enable_benchmark_mode() -> None:
    """Switches from deterministic (training) to benchmark (inference) mode.

    cuDNN benchmark finds the fastest convolution algorithm for your input sizes.
    Safe to call after training is complete, before inference.
    Do NOT call during training — will break reproducibility.
    """
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    print("cuDNN benchmark mode enabled -- optimized for inference speed")


def get_grad_scaler() -> torch.cuda.amp.GradScaler:
    """Returns a GradScaler for mixed precision training.

    RTX 4060 Laptop supports FP16 natively — this gives ~30% training speedup.

    Usage in train.py::

        scaler = get_grad_scaler()
        with torch.cuda.amp.autocast():
            pred = model(noisylr)
            loss, loss_dict = criterion(pred, gt)
        scaler.scale(loss / accumulation_steps).backward()
        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    Returns:
        Configured GradScaler instance.
    """
    if not torch.cuda.is_available():
        print("WARNING: GradScaler created but CUDA unavailable -- will be a no-op")
    scaler = torch.amp.GradScaler(
        device='cuda',
        init_scale=2**16,          # start high, reduce on overflow
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
        enabled=torch.cuda.is_available()
    )
    print(f"GradScaler initialized | FP16 mixed precision: {torch.cuda.is_available()}")
    return scaler


class CheckpointManager:
    """Saves and loads model checkpoints.

    Keeps only max_keep most recent regular checkpoints, but always preserves
    best_model.pth separately.

    Args:
        save_dir: Directory path to save checkpoints.
        max_keep: Maximum number of regular checkpoints to retain (default 3).
    """

    def __init__(self, save_dir: str, max_keep: int = 3) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        self.checkpoint_files: List[Path] = []
        print(f"CheckpointManager: saving to {self.save_dir} | max_keep={max_keep}")

    def save(self,
             model: nn.Module,
             optimizer: torch.optim.Optimizer,
             scheduler,                         # any lr scheduler or None
             scaler: torch.cuda.amp.GradScaler,
             epoch: int,
             metrics: Dict[str, float],
             config: Dict = None,
             is_best: bool = False) -> None:
        """Save a training checkpoint.

        Args:
            model:     The model to checkpoint.
            optimizer: Optimizer state.
            scheduler: LR scheduler (or None).
            scaler:    GradScaler for mixed precision state.
            epoch:     Current epoch number.
            metrics:   Dict of metric values (e.g. psnr, ssim).
            config:    Training config dict (optional).
            is_best:   If True, also saves as best_model.pth.
        """
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict(),   # preserve FP16 scale factor
            "metrics": metrics,
            "config": config or {}
        }

        fname = self.save_dir / f"checkpoint_epoch_{epoch:04d}.pth"
        torch.save(ckpt, fname)
        self.checkpoint_files.append(fname)

        # Prune old checkpoints
        while len(self.checkpoint_files) > self.max_keep:
            oldest = self.checkpoint_files.pop(0)
            if oldest.exists():
                oldest.unlink()
                print(f"  Pruned old checkpoint: {oldest.name}")

        psnr = metrics.get('psnr', 0.0)
        ssim = metrics.get('ssim', 0.0)
        print(f"Checkpoint saved: epoch={epoch:04d} | PSNR={psnr:.4f} dB | SSIM={ssim:.4f}")

        if is_best:
            best_path = self.save_dir / "best_model.pth"
            torch.save(ckpt, best_path)
            print(f"*  New best saved -> best_model.pth | PSNR={psnr:.4f} dB")

    def load(self,
             path: str,
             model: nn.Module,
             optimizer: torch.optim.Optimizer = None,
             scheduler=None,
             scaler: torch.cuda.amp.GradScaler = None) -> Dict:
        """Load a checkpoint from disk.

        Args:
            path:      Path to checkpoint file.
            model:     Model to load state into.
            optimizer: Optimizer to restore state (optional).
            scheduler: LR scheduler to restore state (optional).
            scaler:    GradScaler to restore state (optional).

        Returns:
            Full checkpoint dict with epoch, metrics, config.
        """
        torch.serialization.add_safe_globals([
            dict, list, tuple, int, float, str
        ])
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        if optimizer and ckpt.get('optimizer_state_dict'):
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler and ckpt.get('scheduler_state_dict'):
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if scaler and ckpt.get('scaler_state_dict'):
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        epoch = ckpt.get('epoch', 0)
        metrics = ckpt.get('metrics', {})
        print(f"Loaded: {Path(path).name} | epoch={epoch} | "
              f"PSNR={metrics.get('psnr', 0):.4f} dB")
        return ckpt

    def load_best(self, model: nn.Module, **kwargs) -> Dict:
        """Load the best model checkpoint.

        Args:
            model: Model to load state into.
            **kwargs: Additional args passed to load().

        Returns:
            Full checkpoint dict.

        Raises:
            FileNotFoundError: If best_model.pth doesn't exist.
        """
        best_path = self.save_dir / "best_model.pth"
        if not best_path.exists():
            raise FileNotFoundError(f"best_model.pth not found in {self.save_dir}")
        return self.load(str(best_path), model, **kwargs)

    def __repr__(self) -> str:
        return f"CheckpointManager(save_dir={self.save_dir}, max_keep={self.max_keep})"


def save_visual_comparison(noisylr: torch.Tensor,
                           pred: torch.Tensor,
                           gt: torch.Tensor,
                           filename: str,
                           save_dir: str,
                           epoch: int) -> None:
    """Saves a 3-panel comparison figure: NoisyLR (upscaled) | Restored | GT.

    NoisyLR (128×128) is upscaled to 256×256 with nearest-neighbor for display only.
    Computes and displays PSNR between pred and GT in the panel title.
    All inputs: (1, H, W) single-sample tensors on any device.

    Args:
        noisylr:  (1, 128, 128) NoisyLR tensor.
        pred:     (1, 256, 256) Restored prediction tensor.
        gt:       (1, 256, 256) Ground truth tensor.
        filename: Sample filename stem for the figure title.
        save_dir: Directory to save the figure.
        epoch:    Current epoch for labeling.
    """
    import torch.nn.functional as F_nn

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Convert to numpy — detach from graph, move to CPU
    lr_np = noisylr.detach().cpu().float().numpy().squeeze()          # (128, 128)
    pred_np = pred.detach().cpu().float().clamp(0, 1).numpy().squeeze()  # (256, 256)
    gt_np = gt.detach().cpu().float().clamp(0, 1).numpy().squeeze()      # (256, 256)

    # Upsample NoisyLR 128→256 for display only (nearest neighbor preserves pixel structure)
    lr_t = noisylr.detach().cpu().float().unsqueeze(0)               # (1, 1, 128, 128)
    lr_display = F_nn.interpolate(lr_t, size=(256, 256), mode='nearest').squeeze().numpy()

    # Compute PSNR for subtitle
    mse = np.mean((pred_np - gt_np) ** 2)
    psnr_val = 100.0 if mse < 1e-10 else float(10.0 * np.log10(1.0 / mse))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Epoch {epoch:04d}  |  {filename}", fontsize=13, fontweight='bold')

    axes[0].imshow(lr_display, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("NoisyLR Input\n(2x upscaled for display, nearest-neighbor)", fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(pred_np, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"Model Output (Restored)\nPSNR vs GT: {psnr_val:.2f} dB", fontsize=10)
    axes[1].axis('off')

    axes[2].imshow(gt_np, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title("Ground Truth (Clean)", fontsize=10)
    axes[2].axis('off')

    plt.tight_layout()
    out_path = Path(save_dir) / f"epoch_{epoch:04d}_{filename}.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)   # critical — prevents matplotlib memory leak in long training runs


def log_wandb_images(wandb_run,
                     noisylr_batch: torch.Tensor,
                     pred_batch: torch.Tensor,
                     gt_batch: torch.Tensor,
                     filenames: List[str],
                     epoch: int,
                     max_images: int = 4) -> None:
    """Logs side-by-side comparison images to wandb.

    Creates a horizontal strip: [NoisyLR_upscaled | Restored | GT] per sample.
    Fails gracefully if wandb is not initialized — never crashes training.

    Args:
        wandb_run:     Active wandb run object (or None to skip).
        noisylr_batch: (B, 1, 128, 128) NoisyLR batch.
        pred_batch:    (B, 1, 256, 256) Restored prediction batch.
        gt_batch:      (B, 1, 256, 256) Ground truth batch.
        filenames:     List of filename stems.
        epoch:         Current epoch for labeling.
        max_images:    Maximum number of images to log (default 4).
    """
    if wandb_run is None:
        return
    try:
        import wandb
        import torch.nn.functional as F_nn

        images = []
        n = min(max_images, noisylr_batch.shape[0])

        for i in range(n):
            lr = noisylr_batch[i].detach().cpu().float()          # (1, 128, 128)
            pred = pred_batch[i].detach().cpu().float().clamp(0, 1)  # (1, 256, 256)
            gt_i = gt_batch[i].detach().cpu().float()             # (1, 256, 256)

            # Upsample LR for display
            lr_up = F_nn.interpolate(lr.unsqueeze(0), size=(256, 256),
                                     mode='nearest').squeeze(0)   # (1, 256, 256)

            # Stack horizontally: (1, 256, 768)
            grid = torch.cat([lr_up, pred, gt_i], dim=2)
            grid_np = grid.squeeze().numpy()  # (256, 768)

            # Compute PSNR for caption
            mse = np.mean((pred.squeeze().numpy() - gt_i.squeeze().numpy())**2)
            psnr_cap = 100.0 if mse < 1e-10 else float(10 * np.log10(1.0 / mse))

            images.append(wandb.Image(
                grid_np,
                caption=f"{filenames[i]} | Epoch {epoch} | "
                        f"PSNR: {psnr_cap:.2f} dB | [NoisyLR_up | Restored | GT]"
            ))

        wandb_run.log({"val/visual_comparison": images}, step=epoch)

    except Exception as e:
        print(f"wandb image logging skipped (non-fatal): {type(e).__name__}: {e}")


def load_yaml_config(config_path: str) -> Dict:
    """Load a YAML configuration file.

    Args:
        config_path: Path to .yaml config file.

    Returns:
        Parsed config dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}\n"
            f"Expected location: {path.resolve()}"
        )
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Config loaded: {config_path}")
    return config


def print_model_summary(model: nn.Module,
                        device: torch.device,
                        input_size: Tuple = (1, 1, 128, 128)) -> None:
    """Prints parameter count and verifies input→output shape contract.

    Args:
        model:      The model to summarize.
        device:     Device the model is on.
        input_size: Input tensor shape (B, C, H, W). Use B=1 for summary.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    print(f"=== MODEL SUMMARY: {model.__class__.__name__} ===")
    print(f"  Total params:     {total / 1e6:.3f}M")
    print(f"  Trainable:        {trainable / 1e6:.3f}M")
    print(f"  Frozen:           {frozen / 1e6:.3f}M")

    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(*input_size).to(device)
        output = model(dummy)
        print(f"  Input  shape:     {tuple(dummy.shape)}")
        print(f"  Output shape:     {tuple(output.shape)}")
        assert output.shape[-2:] == (input_size[-2] * 2, input_size[-1] * 2), \
            f"Output spatial dims should be 2x input: got {output.shape}"
        print(f"  2x SR contract:   [OK] ({input_size[-2]}->{output.shape[-2]}, "
              f"{input_size[-1]}->{output.shape[-1]})")
    print(f"===========================================")


if __name__ == "__main__":
    import tempfile

    device = get_device(0)
    set_seed(42)

    # Test GradScaler
    scaler = get_grad_scaler()
    print(f"GradScaler enabled: {scaler.is_enabled()} [OK]\n")

    # Test CheckpointManager with GradScaler state
    dummy_model = nn.Sequential(nn.Conv2d(1, 1, 3, padding=1)).to(device)
    dummy_opt = torch.optim.Adam(dummy_model.parameters(), lr=3e-4)
    tmpdir = tempfile.mkdtemp()
    cm = CheckpointManager(save_dir=tmpdir, max_keep=2)

    for ep in [1, 2, 3]:
        cm.save(dummy_model, dummy_opt, None, scaler,
                epoch=ep, metrics={"psnr": 20.0 + ep, "ssim": 0.7 + ep * 0.01},
                is_best=(ep == 3))

    remaining = list(Path(tmpdir).glob("checkpoint_epoch_*.pth"))
    assert len(remaining) <= 2, f"max_keep=2 violated: {len(remaining)} files"
    assert (Path(tmpdir) / "best_model.pth").exists(), "best_model.pth not saved"
    print(f"\nCheckpointManager: {len(remaining)} regular + 1 best kept [OK]")

    # Load best and verify
    loaded = cm.load_best(dummy_model)
    assert loaded['epoch'] == 3, "Wrong epoch loaded from best"
    print(f"load_best: epoch={loaded['epoch']}, PSNR={loaded['metrics']['psnr']:.1f} [OK]")

    # Test visual comparison
    tmpvis = tempfile.mkdtemp()
    fake_lr = torch.rand(1, 128, 128)
    fake_pred = torch.rand(1, 256, 256)
    fake_gt = torch.rand(1, 256, 256)
    save_visual_comparison(fake_lr, fake_pred, fake_gt, "test_000001", tmpvis, epoch=5)
    saved = [f for f in os.listdir(tmpvis) if f.endswith('.png')]
    assert len(saved) == 1, f"Expected 1 PNG, got: {saved}"
    print(f"Visual comparison saved: {saved[0]} [OK]")

    # Test enable_benchmark_mode
    enable_benchmark_mode()
    assert torch.backends.cudnn.benchmark == True
    print("Benchmark mode: [OK]")

    print("\nUtils verification PASSED [OK]")
