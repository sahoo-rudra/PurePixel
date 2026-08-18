import os
import sys
import io
import time
import base64
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

from src.model import build_model
from src.utils import get_device, enable_benchmark_mode, load_yaml_config

# ---------------------------------------------------------------------------
# Model Loading (module level, runs once at startup)
# ---------------------------------------------------------------------------
script_dir   = Path(__file__).parent.resolve()
config_path  = script_dir / "configs" / "nafnet_base.yaml"
weights_path = script_dir / "weights" / "best_model.pth"

device      = get_device(cuda_device_id=0)
device_type = 'cuda' if device.type == 'cuda' else 'cpu'
config      = load_yaml_config(str(config_path))
model       = build_model(config, device)
ckpt        = torch.load(str(weights_path), map_location=device,
                          weights_only=False)
model.load_state_dict(ckpt["model_state_dict"], strict=True)
loaded_epoch   = ckpt.get("epoch", "unknown")
loaded_metrics = ckpt.get("metrics", {})
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
import torch._dynamo
torch._dynamo.config.suppress_errors = True
if hasattr(torch, 'compile') and sys.platform != 'win32':
    try:
        model = torch.compile(model, mode='reduce-overhead')
        print("torch.compile: enabled")
    except Exception as e:
        print(f"torch.compile skipped: {e}")
else:
    print("torch.compile: skipped (Windows -- Triton not available)")
enable_benchmark_mode()
print(f"Model ready | epoch={loaded_epoch} | "
      f"PSNR={loaded_metrics.get('psnr', 24.7450):.4f} dB | device={device}")

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(title="KLA Image Restoration API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

frontend_dir = script_dir / "frontend"
frontend_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def apply_tta(x: torch.Tensor) -> torch.Tensor:
    """8-aug TTA. x: (1,1,128,128) on device. Returns (1,1,256,256) float32."""
    outputs = []
    for flip in [False, True]:
        for k in range(4):
            aug = x.clone()
            if flip:
                aug = torch.flip(aug, dims=[3])
            if k > 0:
                aug = torch.rot90(aug, k=k, dims=[2, 3])
            with torch.no_grad():
                with torch.amp.autocast(device_type=device_type,
                                        dtype=torch.float16):
                    pred = model(aug)
            pred = pred.float()
            if k > 0:
                pred = torch.rot90(pred, k=(4 - k), dims=[2, 3])
            if flip:
                pred = torch.flip(pred, dims=[3])
            outputs.append(pred)
    return torch.stack(outputs, dim=0).mean(dim=0)


def array_to_base64_png(arr: np.ndarray) -> str:
    """(H,W) float32 [0,1] -> base64 PNG data URL string."""
    arr_u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    img    = Image.fromarray(arr_u8, mode='L')
    buf    = io.BytesIO()
    img.save(buf, format='PNG')
    b64    = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{b64}"


def bicubic_upsample(arr: np.ndarray) -> np.ndarray:
    """(128,128) float32 -> (256,256) float32 via bicubic."""
    t  = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    up = F.interpolate(t, size=(256, 256), mode='bicubic', antialias=True)
    return up.squeeze().clamp(0, 1).numpy()


def normalise_input_shape(arr: np.ndarray) -> np.ndarray:
    """Handles (H,W), (1,H,W), (1,1,H,W) -> (H,W)."""
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr.squeeze(0)
    if arr.ndim == 4:
        arr = arr[0, 0]
    return arr

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({"error": "frontend/index.html not found"})


@app.get("/api/health")
async def health():
    """Frontend polls this every 10s to show connection status."""
    vram_used = vram_total = 0.0
    if torch.cuda.is_available():
        vram_used  = torch.cuda.memory_allocated() / 1024**2
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    return {
        "status":          "ready",
        "device":          str(device),
        "device_name":     torch.cuda.get_device_name(0) if torch.cuda.is_available()
                           else "CPU",
        "vram_used_mb":    round(vram_used,  1),
        "vram_total_mb":   round(vram_total, 1),
        "model_params":    "32.439M",
        "model_epoch":     loaded_epoch,
        "val_psnr":        24.7450,
        "val_ssim":        0.6971,
        "val_lpips":       0.3760,
        "bicubic_psnr":    23.1535,
        "psnr_gain":       1.5915,
        "batch_throughput": "44.1 img/sec (RTX 4060)",
        "h100_estimate":   "180-265 img/sec (batch)"
    }


@app.post("/api/restore")
async def restore(
    file:    UploadFile = File(...),
    use_tta: str        = Form(default="false")
):
    """
    Main restoration endpoint.
    Input:  .npy file upload + use_tta ("true"/"false")
    Output: JSON with base64 PNG images, metrics, degradation estimate
    """
    if not file.filename.endswith('.npy'):
        raise HTTPException(400, "Only .npy files are accepted")

    use_tta_bool = use_tta.lower() == "true"

    try:
        contents = await file.read()
        arr      = np.load(io.BytesIO(contents)).astype(np.float32)
        arr      = normalise_input_shape(arr)

        if arr.shape != (128, 128):
            raise HTTPException(400,
                f"Expected (128,128), got {arr.shape}. "
                f"Upload a valid NoisyLR .npy file.")

        t_start = time.perf_counter()

        # Build tensor -- do NOT clip NoisyLR
        inp = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

        if use_tta_bool:
            pred = apply_tta(inp)           # (1,1,256,256) float32
        else:
            with torch.no_grad():
                with torch.amp.autocast(device_type=device_type,
                                        dtype=torch.float16):
                    pred = model(inp)
            pred = pred.float()

        # Degradation estimate from DegradationHead
        deg_est = model.last_degradation_estimate
        deg     = [0.0, 0.0, 0.0]
        if deg_est is not None:
            d   = deg_est.cpu().float().numpy()[0]
            deg = [round(float(d[0]), 4),
                   round(float(d[1]), 4),
                   round(float(d[2]), 4)]

        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        restored   = pred.squeeze().cpu().clamp(0, 1).numpy()  # (256,256)
        lr_display = bicubic_upsample(arr)                      # (256,256)

        # Difference stats
        diff_mean = float(np.abs(restored - lr_display).mean()) * 100

        # Save .npy for download
        tmp_dir   = tempfile.mkdtemp()
        stem      = Path(file.filename).stem
        npy_path  = os.path.join(tmp_dir, f"{stem}_restored.npy")
        np.save(npy_path, restored.astype(np.float32))
        app.state.last_npy_path = npy_path
        app.state.last_npy_name = f"{stem}_restored.npy"

        return JSONResponse({
            "success":              True,
            "lr_image":             array_to_base64_png(lr_display),
            "restored_image":       array_to_base64_png(restored),
            "elapsed_ms":           round(elapsed_ms, 1),
            "mode":                 "TTA x8" if use_tta_bool else "Batch (single pass)",
            "input_shape":          [128, 128],
            "output_shape":         [256, 256],
            "diff_mean_pct":        round(diff_mean, 2),
            "degradation_speckle":  deg[0],
            "degradation_gaussian": deg[1],
            "degradation_blur":     deg[2],
            "val_psnr":             24.7450,
            "val_ssim":             0.6971,
            "val_lpips":            0.3760,
            "bicubic_psnr":         23.1535,
            "filename":             file.filename
        })

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Inference error: {str(e)}")


@app.get("/api/download")
async def download_restored():
    path = getattr(app.state, 'last_npy_path', None)
    name = getattr(app.state, 'last_npy_name', 'restored.npy')
    if not path or not os.path.exists(path):
        raise HTTPException(404, "No restored file available. Run inference first.")
    return FileResponse(path, filename=name,
                        media_type='application/octet-stream')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("KLA Image Restoration -- Web Demo")
    print("=" * 55)
    print(f"  GPU:    {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  Model:  DegradationAwareNAFNet (32.439M params)")
    print(f"  PSNR:   24.7450 dB (val, epoch 246)")
    print(f"  Gain:   +1.5915 dB over bicubic baseline")
    print("")
    print("  Open: http://127.0.0.1:8000")
    print("  Docs: http://127.0.0.1:8000/docs")
    print("")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
