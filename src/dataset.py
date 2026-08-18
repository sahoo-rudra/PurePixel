"""
src/dataset.py — Semiconductor Inspection Dataset & DataLoader Factory

Production-ready PyTorch Dataset for the KLA / Semicon India 2026 hackathon.
Loads paired NoisyLR (128×128) and GT (256×256) .npy images with deterministic
train/val splitting and geometry-aware paired augmentations (flip, rotate, crop).

Architecture contract:
  - Dataset returns NoisyLR at (1, 128, 128) and GT at (1, 256, 256)
  - No resizing — the model handles 2× upsampling internally via PixelShuffle
  - NoisyLR values may exceed [0,1] — preserved as real signal from noise
  - GT values clamped to [0,1] for loss computation

Hardware: RTX 4060 Laptop (8GB VRAM), Windows OS
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SemiconDataset(Dataset):
    """PyTorch Dataset for paired NoisyLR / GT semiconductor inspection images.

    Loads .npy arrays, applies deterministic train/val split, and performs
    paired geometric augmentations (horizontal flip, vertical flip, 90° rotation,
    random crop) during training.

    Args:
        gt_dir:      Path to directory containing GT .npy files (256×256).
        noisylr_dir: Path to directory containing NoisyLR .npy files (128×128).
        split:       'train' or 'val'. Controls which subset of stems to use.
        val_split:   Fraction of data reserved for validation (default 0.1).
        seed:        Random seed for deterministic split (default 42).
        augment:     Whether to apply paired augmentations (default True).
        patch_size:  Crop size for NoisyLR patches (default 128 = full image).
    """

    def __init__(self,
                 gt_dir: str,
                 noisylr_dir: str,
                 split: str = "train",
                 val_split: float = 0.1,
                 seed: int = 42,
                 augment: bool = True,
                 patch_size: int = 128) -> None:
        super().__init__()

        # 1. Resolve and verify directories
        self.gt_dir = Path(gt_dir).resolve()
        self.noisylr_dir = Path(noisylr_dir).resolve()

        if not self.gt_dir.exists():
            raise FileNotFoundError(
                f"GT directory not found: {self.gt_dir}\n"
                f"Expected .npy files at: {self.gt_dir}/*.npy"
            )
        if not self.noisylr_dir.exists():
            raise FileNotFoundError(
                f"NoisyLR directory not found: {self.noisylr_dir}\n"
                f"Expected .npy files at: {self.noisylr_dir}/*.npy"
            )

        # 2. Collect file stems
        gt_stems = sorted({p.stem for p in self.gt_dir.glob("*.npy")})
        noisylr_stems = sorted({p.stem for p in self.noisylr_dir.glob("*.npy")})

        # 3. Intersection — only use paired images
        gt_set = set(gt_stems)
        noisylr_set = set(noisylr_stems)
        valid_stems = sorted(gt_set & noisylr_set)

        # 4. Warn on mismatches
        if len(gt_set) != len(noisylr_set):
            unmatched_gt = gt_set - noisylr_set
            unmatched_lr = noisylr_set - gt_set
            total_unmatched = len(unmatched_gt) + len(unmatched_lr)
            print(f"WARNING: {total_unmatched} unmatched files "
                  f"(GT-only: {len(unmatched_gt)}, NoisyLR-only: {len(unmatched_lr)})")

        if len(valid_stems) == 0:
            raise ValueError(
                f"No matching pairs found between:\n"
                f"  GT:      {self.gt_dir} ({len(gt_stems)} files)\n"
                f"  NoisyLR: {self.noisylr_dir} ({len(noisylr_stems)} files)"
            )

        # 5. Deterministic split
        random.seed(seed)

        # 6. Split — last val_split fraction alphabetically for validation
        n_val = int(len(valid_stems) * val_split)
        val_stems = valid_stems[-n_val:]        # last 10% alphabetically
        train_stems = valid_stems[:-n_val]      # first 90%

        # 7. Select split
        self.stems = train_stems if split == "train" else val_stems

        # 8-9. Store instance variables
        self.split = split
        self.val_split = val_split
        self.seed = seed
        self.augment = augment
        self.patch_size = patch_size

        # 10. Summary
        print(f"SemiconDataset | split={split} | pairs={len(self.stems)} | augment={augment}")

    def __getitem__(self, idx: int) -> Dict:
        stem = self.stems[idx]

        # Load raw arrays — preserve float32 precision
        noisylr = np.load(self.noisylr_dir / f"{stem}.npy").astype(np.float32)
        gt = np.load(self.gt_dir / f"{stem}.npy").astype(np.float32)

        # PAIRED AUGMENTATION — only for training
        if self.augment and self.split == "train":

            # 1. Random horizontal flip
            if random.random() < 0.5:
                noisylr = np.fliplr(noisylr).copy()
                gt = np.fliplr(gt).copy()

            # 2. Random vertical flip
            if random.random() < 0.5:
                noisylr = np.flipud(noisylr).copy()
                gt = np.flipud(gt).copy()

            # 3. Random 90° rotation (k=0,1,2,3) — applied identically to both
            k = random.randint(0, 3)
            noisylr = np.rot90(noisylr, k).copy()
            gt = np.rot90(gt, k).copy()

            # 4. Random paired crop — ONLY when patch_size < 128
            #    patch_size == 128 means full image — skip crop
            if self.patch_size < 128:
                max_offset = 128 - self.patch_size
                x = random.randint(0, max_offset)
                y = random.randint(0, max_offset)
                noisylr = noisylr[y: y + self.patch_size,
                                  x: x + self.patch_size]
                # GT crop is exactly 2x the NoisyLR crop in pixel space
                gt = gt[y * 2: (y + self.patch_size) * 2,
                        x * 2: (x + self.patch_size) * 2]

        # Convert to tensors — add channel dim
        noisylr_t = torch.from_numpy(noisylr).unsqueeze(0)    # (1, 128, 128)
        gt_t = torch.from_numpy(gt).unsqueeze(0)              # (1, 256, 256)

        # Clamp GT only — it must be [0,1] for loss computation
        # Do NOT clamp NoisyLR — out-of-range values are real signal from noise
        gt_t = gt_t.clamp(0.0, 1.0)

        return {"noisylr": noisylr_t, "gt": gt_t, "filename": stem}

    def __len__(self) -> int:
        return len(self.stems)

    def __repr__(self) -> str:
        return (f"SemiconDataset(split={self.split}, pairs={len(self.stems)}, "
                f"augment={self.augment}, patch_size={self.patch_size})")


def make_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader]:
    """Creates optimized train and val DataLoaders for RTX 4060 Laptop.

    Handles Windows multiprocessing constraints automatically.
    On Windows, num_workers is forced to 0 to avoid BrokenPipeError.

    Args:
        config: Dictionary with keys:
            gt_folder (str):      Path to GT directory.
            noisylr_folder (str): Path to NoisyLR directory.
            batch_size (int):     Batch size (default 16).
            patch_size (int):     Crop size for training (default 128).
            val_split (float):    Validation fraction (default 0.1).
            num_workers (int):    DataLoader workers (default 2, forced 0 on Windows).
            seed (int):           Random seed (default 42).

    Returns:
        Tuple of (train_loader, val_loader).
    """
    # WINDOWS SAFETY — critical for Antigravity on Windows
    # Using num_workers > 0 without if __name__ == '__main__' guard causes
    # BrokenPipeError on Windows. Force num_workers=0 on Windows.
    is_windows = os.name == 'nt'
    if is_windows:
        actual_workers = 0
        persistent = False
        print("Windows detected -- num_workers=0 (avoids BrokenPipeError)")
    else:
        actual_workers = config.get('num_workers', 2)
        persistent = actual_workers > 0

    train_ds = SemiconDataset(
        gt_dir=config['gt_folder'],
        noisylr_dir=config['noisylr_folder'],
        split="train",
        val_split=config.get('val_split', 0.1),
        seed=config.get('seed', 42),
        augment=True,
        patch_size=config.get('patch_size', 128)
    )
    val_ds = SemiconDataset(
        gt_dir=config['gt_folder'],
        noisylr_dir=config['noisylr_folder'],
        split="val",
        val_split=config.get('val_split', 0.1),
        seed=config.get('seed', 42),
        augment=False,
        patch_size=128   # always use full image for validation
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.get('batch_size', 16),
        shuffle=True,
        num_workers=actual_workers,
        pin_memory=True,          # faster CPU->GPU transfer
        drop_last=True,           # keeps batch sizes consistent for accumulation
        persistent_workers=persistent
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.get('batch_size', 16),
        shuffle=False,
        num_workers=actual_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=persistent
    )

    print(f"Train loader: {len(train_loader)} batches x batch_size={config.get('batch_size', 16)}")
    print(f"Val loader:   {len(val_loader)} batches x batch_size={config.get('batch_size', 16)}")
    print(f"Effective batch size (with grad accumulation x2): {config.get('batch_size', 16) * 2}")
    return train_loader, val_loader


if __name__ == "__main__":
    # Must be inside __main__ guard for Windows DataLoader safety
    config = {
        "gt_folder":      "../../GT",
        "noisylr_folder": "../../NoisyLR",
        "batch_size":     4,
        "patch_size":     128,
        "val_split":      0.1,
        "num_workers":    0,
        "seed":           42
    }
    train_loader, val_loader = make_dataloaders(config)

    # Verify train batch
    train_batch = next(iter(train_loader))
    assert train_batch["noisylr"].shape == (4, 1, 128, 128), \
        f"NoisyLR shape wrong: {train_batch['noisylr'].shape}"
    assert train_batch["gt"].shape == (4, 1, 256, 256), \
        f"GT shape wrong: {train_batch['gt'].shape}"

    lr_min = train_batch["noisylr"].min().item()
    lr_max = train_batch["noisylr"].max().item()
    gt_min = train_batch["gt"].min().item()
    gt_max = train_batch["gt"].max().item()

    print(f"\nTrain batch shapes OK:")
    print(f"  NoisyLR: {train_batch['noisylr'].shape} | range [{lr_min:.4f}, {lr_max:.4f}]")
    print(f"  GT:      {train_batch['gt'].shape}      | range [{gt_min:.4f}, {gt_max:.4f}]")
    print(f"  Filenames: {train_batch['filename'][:2]}")
    if lr_max > 1.0 or lr_min < 0.0:
        print("  [OK] NoisyLR correctly preserves out-of-range noise values")
    assert gt_max <= 1.0 and gt_min >= 0.0, "GT values outside [0,1] -- clamping failed"

    # Verify val batch
    val_batch = next(iter(val_loader))
    assert val_batch["noisylr"].shape[1:] == (1, 128, 128)
    assert val_batch["gt"].shape[1:] == (1, 256, 256)
    print(f"\nVal batch shapes OK: {val_batch['noisylr'].shape}, {val_batch['gt'].shape}")

    print("\nDataset verification PASSED [OK]")
