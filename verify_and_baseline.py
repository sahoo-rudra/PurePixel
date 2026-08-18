"""
verify_and_baseline.py -- Model size verification + bicubic baseline metrics
KLA / Semicon India 2026 Image Restoration

Two sections:
  1. Verifies model handles 128x128, 256x256, and 64x64 inputs correctly
  2. Computes bicubic upsampling baseline metrics on the 320-image val set

Run from kla-image-restoration/ directory:
  C:\\Users\\SUNANDAN\\miniconda3\\envs\\semicon\\python.exe verify_and_baseline.py
"""

# =======================================================
# SECTION 1: IMPORTS
# =======================================================

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.model import build_model
from src.metrics import RestorationMetrics, MetricTracker
from src.utils import get_device, load_yaml_config


# =======================================================
# SECTION 2: MODEL SIZE VERIFICATION
# =======================================================

def verify_model_sizes(model, device):
    """
    Tests the model with three different input sizes and reports output shapes.
    Critical to confirm behaviour before KLA evaluates on potentially 512x512 GT.

    Your model: input (B,1,H,W) -> output (B,1,H*2,W*2) via PixelShuffle 2x
    Expected:
      128x128 input  -> 256x256 output  (standard training size)
      256x256 input  -> 512x512 output  (possible 512x512 GT scenario)
      64x64 input    -> 128x128 output  (sanity check for smaller inputs)
    """
    print("=" * 60)
    print("SECTION 1: MODEL INPUT SIZE VERIFICATION")
    print("=" * 60)

    test_sizes = [
        (128, 128, "Standard training size -- NoisyLR -> 256x256 GT"),
        (256, 256, "Large input -- would produce 512x512 output"),
        (64, 64, "Small input sanity check"),
    ]

    model.eval()
    all_passed = True

    for H, W, description in test_sizes:
        try:
            dummy = torch.zeros(1, 1, H, W, device=device)
            with torch.no_grad():
                out = model(dummy)
            expected_H = H * 2
            expected_W = W * 2
            shape_ok = (out.shape == (1, 1, expected_H, expected_W))
            status = "[OK]" if shape_ok else "[FAIL]"
            print(f"  {status} Input ({H}x{W}) -> Output {tuple(out.shape)}")
            print(f"       Expected: (1,1,{expected_H},{expected_W})")
            print(f"       Note: {description}")
            if not shape_ok:
                all_passed = False
                print(f"       WARNING: Shape mismatch -- check PixelShuffle head")
        except Exception as e:
            print(f"  [FAIL] Input ({H}x{W}) -> ERROR: {type(e).__name__}: {e}")
            all_passed = False
        print()

    if all_passed:
        print("[OK] All size checks passed")
        print("     Model correctly handles variable input sizes via PixelShuffle 2x")
        print("     If KLA evaluates on 512x512 GT (256x256 NoisyLR input),")
        print("     your inference.py will produce correct 512x512 output automatically.")
    else:
        print("[WARN] Some size checks failed -- review model architecture")

    print()
    return all_passed


# =======================================================
# SECTION 3: BICUBIC BASELINE COMPUTATION
# =======================================================

def compute_bicubic_baseline(device):
    """
    Computes bicubic upsampling PSNR/SSIM/LPIPS on the 320-image val set.

    Bicubic baseline: take NoisyLR (128x128), upsample to 256x256 via bicubic,
    compare against GT (256x256). No denoising -- pure upsampling baseline.

    This is a MANDATORY requirement per KLA evaluation criteria:
    "Compare at least one baseline with the final method."

    Val split: last 320 stems from sorted list of 3200 pairs (last 10%)
    Same deterministic split used in dataset.py (seed=42, sorted alphabetically)
    Uses intersection of GT and NoisyLR stems -- matches dataset.py exactly.
    """
    print("=" * 60)
    print("SECTION 2: BICUBIC BASELINE METRICS (320 val images)")
    print("=" * 60)

    script_dir = Path(__file__).parent.resolve()
    gt_dir = (script_dir / "../GT").resolve()
    lr_dir = (script_dir / "../NoisyLR").resolve()

    if not gt_dir.exists():
        print(f"[FAIL] GT directory not found: {gt_dir}")
        return None
    if not lr_dir.exists():
        print(f"[FAIL] NoisyLR directory not found: {lr_dir}")
        return None

    # Deterministic val split -- must match dataset.py exactly
    # dataset.py uses intersection of GT and NoisyLR stems, sorted
    gt_stems_set = {p.stem for p in gt_dir.glob("*.npy")}
    lr_stems_set = {p.stem for p in lr_dir.glob("*.npy")}
    all_stems = sorted(gt_stems_set & lr_stems_set)
    n_val = int(len(all_stems) * 0.1)
    val_stems = all_stems[-n_val:]

    print(f"  Total pairs:  {len(all_stems)}")
    print(f"  Val stems:    {len(val_stems)} (last {n_val} alphabetically)")
    print(f"  First val:    {val_stems[0]}")
    print(f"  Last val:     {val_stems[-1]}")
    print()

    metrics_calc = RestorationMetrics(device)
    tracker = MetricTracker(["psnr", "ssim", "lpips"])

    errors = 0
    t_start = time.perf_counter()

    print(f"  Computing bicubic baseline... (this takes 2-4 minutes)")

    for i, stem in enumerate(val_stems):
        try:
            # Load arrays
            lr_arr = np.load(lr_dir / f"{stem}.npy").astype(np.float32)
            gt_arr = np.load(gt_dir / f"{stem}.npy").astype(np.float32)

            # Bicubic upsampling: 128x128 -> 256x256
            lr_t = torch.from_numpy(lr_arr).unsqueeze(0).unsqueeze(0)
            bicubic = F.interpolate(lr_t, size=(256, 256),
                                    mode='bicubic', antialias=True)
            bicubic = bicubic.clamp(0.0, 1.0).to(device)

            gt_t = torch.from_numpy(gt_arr).unsqueeze(0).unsqueeze(0).to(device)
            gt_t = gt_t.clamp(0.0, 1.0)

            # Compute all three metrics
            m = metrics_calc.compute_all(bicubic, gt_t)
            tracker.update(m, batch_size=1)

            # Progress every 50 images
            if (i + 1) % 50 == 0:
                current = tracker.compute()
                print(f"  [{i + 1:3d}/320] Running avg -- "
                      f"PSNR: {current['psnr']:.4f} dB | "
                      f"SSIM: {current['ssim']:.4f} | "
                      f"LPIPS: {current['lpips']:.4f}")

        except Exception as e:
            print(f"  [WARN] Error on {stem}: {e}")
            errors += 1

    elapsed = time.perf_counter() - t_start
    baseline = tracker.compute()

    print()
    print("=" * 60)
    print("BICUBIC BASELINE RESULTS")
    print("=" * 60)
    print(f"  Val images:  {len(val_stems) - errors} / {len(val_stems)}")
    print(f"  Errors:      {errors}")
    print(f"  Time:        {elapsed:.1f}s")
    print()
    print(f"  PSNR:        {baseline['psnr']:.4f} dB")
    print(f"  SSIM:        {baseline['ssim']:.4f}")
    print(f"  LPIPS:       {baseline['lpips']:.4f}")
    print()
    print("=" * 60)
    print("IMPROVEMENT OVER BICUBIC (your DegradationAwareNAFNet)")
    print("=" * 60)
    model_psnr = 24.7450
    model_ssim = 0.6971
    model_lpips = 0.3760
    print(f"  PSNR:  {baseline['psnr']:.4f} -> {model_psnr:.4f} dB "
          f"(+{model_psnr - baseline['psnr']:.4f} dB)")
    print(f"  SSIM:  {baseline['ssim']:.4f} -> {model_ssim:.4f} "
          f"(+{model_ssim - baseline['ssim']:.4f})")
    print(f"  LPIPS: {baseline['lpips']:.4f} -> {model_lpips:.4f} "
          f"(-{baseline['lpips'] - model_lpips:.4f}, lower=better)")
    print()
    print("  Copy these numbers into your README baseline comparison table")
    print("  and PPT Slide 8 (Experiment Tracking and Baseline Comparison)")
    print()

    return baseline


# =======================================================
# SECTION 4: MAIN
# =======================================================

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("KLA Image Restoration -- Verification and Baseline Script")
    print("=" * 60)
    print()

    # Setup
    device = get_device(cuda_device_id=0)
    script_dir = Path(__file__).parent.resolve()

    # Load model
    print("Loading model...")
    config = load_yaml_config(str(script_dir / "configs" / "nafnet_base.yaml"))
    model = build_model(config, device)
    ckpt = torch.load(
        str(script_dir / "weights" / "best_model.pth"),
        map_location=device, weights_only=False
    )
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        epoch = ckpt.get("epoch", "unknown")
        metrics = ckpt.get("metrics", {})
        psnr_val = metrics.get("psnr", None)
        if psnr_val is not None:
            print(f"  Checkpoint: epoch={epoch} | "
                  f"val_PSNR={psnr_val:.4f} dB")
        else:
            print(f"  Checkpoint: epoch={epoch} | val_PSNR=N/A")
    else:
        model.load_state_dict(ckpt, strict=True)
        print("  Checkpoint: raw state dict")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print()

    # Run Section 1: size verification
    size_ok = verify_model_sizes(model, device)

    # Run Section 2: bicubic baseline
    baseline = compute_bicubic_baseline(device)

    # Final summary
    print("=" * 60)
    print("SCRIPT COMPLETE")
    print("=" * 60)
    if size_ok:
        print("[OK] Model handles variable input sizes correctly")
    else:
        print("[WARN] Model has input size issues -- review before submission")
    if baseline:
        print(f"[OK] Bicubic baseline computed: "
              f"PSNR={baseline['psnr']:.4f} dB | "
              f"SSIM={baseline['ssim']:.4f} | "
              f"LPIPS={baseline['lpips']:.4f}")
    print()
    print("Next steps:")
    print("  1. Copy baseline numbers into README.md and PPT Slide 8")
    print("  2. Start Fix 1 training: python train.py --no_wandb --epochs 80")
    print("  3. Generate frontend: backend.py + frontend/index.html")
