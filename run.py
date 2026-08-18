import sys
import time
import numpy as np
import torch
from pathlib import Path

from inference import load_model, discover_input_files, load_npy_as_tensor, save_tensor_as_npy
from src.utils import get_device

def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Hardcoded paths relative to script location
    base_dir = Path(__file__).parent.resolve()
    weights_path = str((base_dir / "weights/best_model.pth").resolve())
    config_path = str((base_dir / "configs/nafnet_base.yaml").resolve())

    # Create output directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = get_device(cuda_device_id=0)
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'

    # Load model
    print("Loading model...")
    model = load_model(weights_path, config_path, device)

    # Discover files
    print("Scanning input directory...")
    try:
        input_files = discover_input_files(input_dir)
    except Exception as e:
        print(f"Error finding files: {e}")
        sys.exit(1)
        
    n_files = len(input_files)
    print(f"\nProcessing {n_files} files...")

    # Batch mode default parameters
    batch_size = 32
    
    t_start = time.perf_counter()
    files_processed = 0
    files_skipped = 0
    
    for batch_start in range(0, n_files, batch_size):
        batch_paths = input_files[batch_start : batch_start + batch_size]
        tensors = []
        valid_paths = []

        for fpath in batch_paths:
            try:
                tensors.append(load_npy_as_tensor(fpath))
                valid_paths.append(fpath)
            except Exception as e:
                print(f"  ERROR loading {fpath.name}: {e} -- skipping")
                files_skipped += 1

        if not tensors:
            continue

        batch_in = torch.cat(tensors, dim=0).to(device)  # (B,1,128,128)

        with torch.no_grad():
            with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
                pred_batch = model(batch_in)
        pred_batch = pred_batch.float()

        for i, fpath in enumerate(valid_paths):
            filename = fpath.name
            try:
                t_out = pred_batch[i : i+1]
                # Pre-calculate what save_tensor_as_npy will do to validate
                t_out_cpu = t_out.detach().cpu().float().clamp(0.0, 1.0).squeeze()
                restored = t_out_cpu.numpy()
                
                # 6. CRITICAL -- explicit NaN/Inf validation
                assert not np.isnan(restored).any(), f"NaN detected in {filename}"
                assert not np.isinf(restored).any(), f"Inf detected in {filename}"
                
                # 7. Ensure output arrays are exactly (H, W) 2D shape
                assert restored.shape == (256, 256) or restored.shape == (512, 512), \
                    f"Unexpected output shape {restored.shape} for {filename}"
                
                save_tensor_as_npy(t_out, out_path / filename)
                files_processed += 1
            except Exception as e:
                print(f"  ERROR processing {filename}: {e} -- skipping")
                files_skipped += 1

    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    t_total = time.perf_counter() - t_start
    print("\n" + "="*60)
    print("Inference complete.")
    print(f"  Files processed: {files_processed} / {n_files}")
    print(f"  Files skipped:   {files_skipped}")
    print(f"  Total time:      {t_total:.2f} s")
    print("="*60)
    
    sys.exit(0)

if __name__ == "__main__":
    main()
