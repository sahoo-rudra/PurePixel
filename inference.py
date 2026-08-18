"""
Inference script for a production-grade AI image restoration pipeline (KLA / Semicon India 2026).
"""

import os, sys, time, argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.model import build_model
from src.utils import get_device, enable_benchmark_mode, load_yaml_config


def load_model(
    weights_path: str,
    config_path:  str,
    device:       torch.device
) -> torch.nn.Module:
  """
  Loads DegradationAwareNAFNet from checkpoint.
  Verifies input/output shape contract with a dummy forward pass.
  """
  config = load_yaml_config(config_path)
  model  = build_model(config, device)

  ckpt = torch.load(weights_path, map_location=device, weights_only=False)

  # Handle full checkpoint dict OR raw state dict
  if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
      state_dict = ckpt["model_state_dict"]
      epoch      = ckpt.get("epoch", "unknown")
      metrics    = ckpt.get("metrics", {})
      psnr_str   = f"{metrics.get('psnr', 'N/A')}"
      print(f"  Checkpoint: epoch={epoch} | val_PSNR={psnr_str} dB")
  else:
      state_dict = ckpt
      print("  Checkpoint: raw state dict")

  model.load_state_dict(state_dict, strict=True)
  model.eval()

  # Freeze all parameters — no grad computation needed at inference
  for p in model.parameters():
      p.requires_grad_(False)

  # Verify shape contract
  with torch.no_grad():
      dummy = torch.zeros(1, 1, 128, 128, device=device)
      out   = model(dummy)
  assert out.shape == (1, 1, 256, 256), \
      f"Shape contract failed: expected (1,1,256,256), got {out.shape}"

  total = sum(p.numel() for p in model.parameters())
  print(f"  Params: {total/1e6:.3f}M | device={device}")
  print(f"  Contract: (B,1,128,128) -> (B,1,256,256) [OK]")
  return model


def discover_input_files(input_dir: str) -> List[Path]:
  """
  Returns sorted list of .npy Paths in input_dir.
  Raises clearly if directory missing or empty.
  """
  p = Path(input_dir)
  if not p.exists():
      raise FileNotFoundError(
          f"Input directory not found: {p.resolve()}"
      )
  files = sorted(p.glob("*.npy"))
  if not files:
      raise ValueError(f"No .npy files found in: {p.resolve()}")
  print(f"  Input files: {len(files)} .npy files in {p.resolve()}")
  return files


def load_npy_as_tensor(path: Path) -> torch.Tensor:
  """
  Loads .npy and returns (1, 1, 128, 128) float32 CPU tensor.
  Handles (128,128), (1,128,128), (1,1,128,128) input shapes.
  Does NOT clip values -- preserves out-of-range noise signal.
  """
  arr = np.load(str(path)).astype(np.float32)

  if arr.ndim == 2:
      pass                                     # (H, W) -- expected
  elif arr.ndim == 3 and arr.shape[0] == 1:
      arr = arr.squeeze(0)                     # (1,H,W) -> (H,W)
  elif arr.ndim == 4 and arr.shape[:2] == (1, 1):
      arr = arr[0, 0]                          # (1,1,H,W) -> (H,W)
  else:
      raise ValueError(
          f"{path.name}: unexpected shape {arr.shape}. "
          f"Expected (128,128) or variant."
      )

  # (H,W) -> (1,1,H,W): add batch + channel dims
  return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def save_tensor_as_npy(tensor: torch.Tensor, output_path: Path) -> None:
  """
  Saves tensor as (256, 256) float32 .npy file.
  Clamps to [0,1] here -- this is the ONLY place clamping occurs.
  KLA scores exactly what this function saves.
  """
  # Detach, move to CPU, cast to float32
  arr = tensor.detach().cpu().float()

  # Clamp INSIDE this function -- mandatory per KLA scoring rules
  arr = arr.clamp(0.0, 1.0)

  # Squeeze all dims down to (256, 256)
  arr = arr.squeeze()   # removes all size-1 dims
  assert arr.ndim == 2, f"Expected 2D after squeeze, got shape {arr.shape}"

  # Save
  np.save(str(output_path), arr.numpy())


def apply_tta(
    model:       torch.nn.Module,
    x:           torch.Tensor,       # (1, 1, 128, 128) on device, NOT clamped
    device_type: str                 # 'cuda' or 'cpu'
) -> torch.Tensor:                   # returns (1, 1, 256, 256) float32
  """
  Test-Time Augmentation: 8 augmented forward passes averaged.
  Augmentations: 4 rotations (0,90,180,270) x 2 flips (none, horizontal).
  Forward: flip -> rotate. Inverse: rotate-inverse -> flip.
  Averaging in float32 before any clamping.
  Free PSNR boost ~0.3-0.5 dB at zero training cost.
  """
  outputs = []

  for flip in [False, True]:
      for k in range(4):    # k=0,1,2,3 -> rotate by 0,90,180,270 degrees

          # --- FORWARD AUGMENTATION ---
          aug = x.clone()   # (1, 1, 128, 128)

          # Step 1: horizontal flip
          if flip:
              aug = torch.flip(aug, dims=[3])

          # Step 2: rotate k*90 degrees (dims=[2,3] = H,W axes)
          if k > 0:
              aug = torch.rot90(aug, k=k, dims=[2, 3])

          # --- MODEL FORWARD ---
          with torch.no_grad():
              with torch.amp.autocast(
                  device_type=device_type, dtype=torch.float16
              ):
                  pred = model(aug)    # (1, 1, 256, 256), may be FP16
          pred = pred.float()          # convert to FP32 before inverse + averaging

          # --- INVERSE AUGMENTATION (reverse order of forward) ---
          # pred is (1, 1, 256, 256) -- inverse ops on H,W dims=[2,3]

          # Undo rotation first (inverse of k is 4-k, mod 4)
          if k > 0:
              pred = torch.rot90(pred, k=(4 - k), dims=[2, 3])

          # Undo flip (flip is self-inverse)
          if flip:
              pred = torch.flip(pred, dims=[3])

          outputs.append(pred)   # (1, 1, 256, 256)

  # Stack and average: 8 x (1,1,256,256) -> average -> (1,1,256,256)
  # torch.stack on dim=0: (8, 1, 1, 256, 256) -> mean(dim=0): (1, 1, 256, 256)
  avg = torch.stack(outputs, dim=0).mean(dim=0)
  return avg   # (1, 1, 256, 256), float32, NOT yet clamped


def run_inference(
    input_dir:    str,
    output_dir:   str,
    weights_path: str,
    config_path:  str,
    batch_size:   int  = 32,
    use_tta:      bool = True,
    device_id:    int  = 0
) -> dict:
  """
  Full inference pipeline. Returns timing and file count summary.
  """

  # --- SETUP ---
  device      = get_device(cuda_device_id=device_id)
  device_type = 'cuda' if device.type == 'cuda' else 'cpu'
  enable_benchmark_mode()   # cuDNN auto-selects fastest kernels for inference

  # Create output directory
  out_path = Path(output_dir)
  out_path.mkdir(parents=True, exist_ok=True)

  # --- MODEL ---
  print("Loading model...")
  model = load_model(weights_path, config_path, device)

  # --- INPUT FILES ---
  print("Scanning input directory...")
  input_files = discover_input_files(input_dir)
  n_files     = len(input_files)

  print("")
  print(f"Output dir:  {out_path.resolve()}")
  print(f"Mode:        {'TTA (8x augmentations)' if use_tta else 'batch (no TTA)'}")
  print(f"Batch size:  {batch_size} (used in batch mode only)")
  print(f"Processing {n_files} files...")
  print("")

  # --- INFERENCE ---
  t_start         = time.perf_counter()
  files_processed = 0
  files_skipped   = 0

  if use_tta:
    # TTA: one image at a time (safe on 8GB VRAM for any model size)
    pbar = tqdm(
        input_files,
        desc          = "Restoring (TTA x8)",
        ascii         = True,
        ncols         = 100,
        dynamic_ncols = False
    )
    for fpath in pbar:
        try:
            x    = load_npy_as_tensor(fpath).to(device)  # (1,1,128,128)
            pred = apply_tta(model, x, device_type)       # (1,1,256,256)
            save_tensor_as_npy(pred, out_path / fpath.name)
            files_processed += 1
        except Exception as e:
            print(f"\n  ERROR {fpath.name}: {e} -- skipping")
            files_skipped += 1
        pbar.set_postfix({"ok": files_processed, "skip": files_skipped})

  else:
    # BATCH MODE: faster, processes batch_size images at once
    pbar = tqdm(
        range(0, n_files, batch_size),
        desc          = "Restoring (batch)",
        ascii         = True,
        ncols         = 100,
        dynamic_ncols = False
    )
    for batch_start in pbar:
        batch_paths = input_files[batch_start : batch_start + batch_size]
        tensors     = []
        valid_paths = []

        for fpath in batch_paths:
            try:
                tensors.append(load_npy_as_tensor(fpath))
                valid_paths.append(fpath)
            except Exception as e:
                print(f"\n  ERROR loading {fpath.name}: {e} -- skipping")
                files_skipped += 1

        if not tensors:
            continue

        batch_in = torch.cat(tensors, dim=0).to(device)  # (B,1,128,128)

        with torch.no_grad():
            with torch.amp.autocast(
                device_type=device_type, dtype=torch.float16
            ):
                pred_batch = model(batch_in)   # (B,1,256,256), may be FP16
        pred_batch = pred_batch.float()

        for i, fpath in enumerate(valid_paths):
            try:
                save_tensor_as_npy(
                    pred_batch[i : i+1],       # (1,1,256,256)
                    out_path / fpath.name
                )
                files_processed += 1
            except Exception as e:
                print(f"\n  ERROR saving {fpath.name}: {e}")
                files_skipped += 1

        pbar.set_postfix({"ok": files_processed, "skip": files_skipped})

  # Synchronize GPU before stopping timer
  # (GPU ops are async -- without this, timing is too optimistic)
  if device.type == 'cuda':
      torch.cuda.synchronize()

  t_total     = time.perf_counter() - t_start
  ms_per_img  = (t_total / max(files_processed, 1)) * 1000
  throughput  = files_processed / max(t_total, 1e-9)

  print("")
  print("="*60)
  print("Inference complete.")
  print(f"  Files processed: {files_processed} / {n_files}")
  print(f"  Files skipped:   {files_skipped}")
  print(f"  Total time:      {t_total:.2f} s")
  print(f"  Per image:       {ms_per_img:.1f} ms")
  print(f"  Throughput:      {throughput:.1f} images/sec")
  print(f"  Mode:            {'TTA x8' if use_tta else 'batch'}")
  print("="*60)

  # --- VERIFY OUTPUT ---
  saved = sorted(out_path.glob("*.npy"))
  if len(saved) == files_processed:
      print(f"[OK] Output: {len(saved)} .npy files in {out_path.name}/")
  else:
      print(f"[WARN] Count mismatch: processed={files_processed}, "
            f"saved={len(saved)}")

  # --- VERIFY FILENAME PRESERVATION ---
  # Check that first output filename matches first input filename
  if saved and input_files:
      first_in  = input_files[0].name
      first_out = saved[0].name
      if first_in == first_out:
          print(f"[OK] Filename preserved: {first_in}")
      else:
          print(f"[WARN] Filename mismatch: in={first_in}, out={first_out}")

  # --- VERIFY OUTPUT SHAPE & RANGE ---
  # Sample-check the first saved file
  if saved:
      sample     = np.load(str(saved[0]))
      shape_ok   = (sample.shape == (256, 256))
      range_ok   = (sample.min() >= 0.0 and sample.max() <= 1.0)
      dtype_ok   = (sample.dtype == np.float32)
      print(f"[OK] Sample check: shape={sample.shape} "
            f"range=[{sample.min():.4f},{sample.max():.4f}] "
            f"dtype={sample.dtype}"
            + (" [OK]" if (shape_ok and range_ok and dtype_ok) else " [WARN]"))

  return {
      "files_processed": files_processed,
      "files_skipped":   files_skipped,
      "total_time_s":    t_total,
      "ms_per_image":    ms_per_img,
      "throughput":      throughput,
      "use_tta":         use_tta,
      "output_dir":      str(out_path.resolve()),
      "model":           model    # returned so __main__ can reuse for benchmark
  }


def build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description="KLA Semiconductor Image Restoration -- Inference"
  )

  # REQUIRED (judges use only these two)
  parser.add_argument(
      "--input_dir", type=str, required=True,
      help="Directory containing input .npy files"
  )
  parser.add_argument(
      "--output_dir", type=str, required=True,
      help="Directory to save restored .npy files"
  )

  # OPTIONAL
  parser.add_argument(
      "--weights", type=str, default="weights/best_model.pth",
      help="Path to model checkpoint (default: weights/best_model.pth)"
  )
  parser.add_argument(
      "--config", type=str, default="configs/nafnet_base.yaml",
      help="Path to YAML config (default: configs/nafnet_base.yaml)"
  )
  parser.add_argument(
      "--batch_size", type=int, default=32,
      help="Batch size for non-TTA inference (default: 32)"
  )
  parser.add_argument(
      "--tta", action="store_true", default=False,
      help="Enable TTA (8x augmentations, +0.3-0.5 dB, 8x slower)"
  )
  parser.add_argument(
      "--device_id", type=int, default=0,
      help="CUDA device ID (default: 0)"
  )
  parser.add_argument(
      "--benchmark", action="store_true", default=False,
      help="Run speed benchmark after inference"
  )
  return parser


if __name__ == "__main__":
  parser = build_arg_parser()
  args   = parser.parse_args()

  # Resolve all paths relative to script location
  script_dir   = Path(__file__).parent.resolve()
  weights_path = str((script_dir / args.weights).resolve())
  config_path  = str((script_dir / args.config).resolve())

  # Validate critical files before starting
  if not Path(weights_path).exists():
      print(f"ERROR: weights not found: {weights_path}")
      print(f"  Train first:   python train.py")
      print(f"  Or specify:    --weights path/to/checkpoint.pth")
      sys.exit(1)

  if not Path(config_path).exists():
      print(f"ERROR: config not found: {config_path}")
      sys.exit(1)

  print("="*60)
  print("KLA Image Restoration -- Inference")
  print(f"  Weights:  {weights_path}")
  print(f"  Input:    {args.input_dir}")
  print(f"  Output:   {args.output_dir}")
  print(f"  TTA:      {'enabled (8x)' if args.tta else 'disabled (batch mode)'}")
  print("="*60)
  print("")

  # Run inference -- returns summary dict including the model object
  result = run_inference(
      input_dir    = args.input_dir,
      output_dir   = args.output_dir,
      weights_path = weights_path,
      config_path  = config_path,
      batch_size   = args.batch_size,
      use_tta      = args.tta,
      device_id    = args.device_id
  )

  # Optional speed benchmark -- reuses model from result (no double-load)
  if args.benchmark and result["files_processed"] > 0:
      print("")
      print("-"*50)
      print("SPEED BENCHMARK (single image, no TTA, 20 runs)")
      print("-"*50)

      model       = result["model"]
      device      = next(model.parameters()).device
      device_type = 'cuda' if device.type == 'cuda' else 'cpu'
      dummy       = torch.rand(1, 1, 128, 128, device=device)

      # Warmup passes
      with torch.no_grad():
          with torch.amp.autocast(
              device_type=device_type, dtype=torch.float16
          ):
              for _ in range(5):
                  _ = model(dummy)
      if device.type == 'cuda':
          torch.cuda.synchronize()

      # Timed passes
      n_bench = 20
      t0 = time.perf_counter()
      with torch.no_grad():
          with torch.amp.autocast(
              device_type=device_type, dtype=torch.float16
          ):
              for _ in range(n_bench):
                  _ = model(dummy)
      if device.type == 'cuda':
          torch.cuda.synchronize()
      t_bench = time.perf_counter() - t0

      print(f"  Runs:          {n_bench}")
      print(f"  Total time:    {t_bench*1000:.1f} ms")
      print(f"  Per image:     {t_bench/n_bench*1000:.2f} ms")
      print(f"  Throughput:    {n_bench/t_bench:.1f} img/s (single image)")
      print("-"*50)

  # Natural exit (no sys.exit(0) -- avoids Windows CUDA cleanup issues)
