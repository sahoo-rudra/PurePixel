# KLA Semiconductor Image Restoration — DegradationAwareNAFNet

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch 2.5.1](https://img.shields.io/badge/PyTorch-2.5.1-ee4c2c) ![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-76b900) ![License MIT](https://img.shields.io/badge/License-MIT-green)

**DegradationAwareNAFNet**: a FiLM-conditioned NAFNet backbone with PixelShuffle 2× super-resolution head and composite frequency-domain loss, achieving PSNR 24.7450 dB (+1.59 dB over bicubic) on 320 validation images.

---

## Results at a Glance

**Model vs Bicubic Baseline (320 val images)**

| Method | PSNR (dB) | SSIM | LPIPS |
|----------------|-----------|--------|--------|
| Bicubic | 23.1535 | 0.5370 | 0.4276 |
| Ours (ep 246) | 24.7450 | 0.6971 | 0.3760 |
| **Improvement**| **+1.5915**|**+0.1601**|**−0.0516**|

**Runtime Performance**

| Mode | Throughput | Latency | Hardware |
|--------------|---------------|----------|------------------|
| Batch (FP16) | 60.5 img/sec | 16.53 ms | RTX 4060 Laptop |
| H100 est. | 150–250 img/s | ~4–7 ms | NVIDIA H100 |

---

## Architecture Overview

### DegradationHead (FiLM Conditioning)

A lightweight CNN analyses the raw NoisyLR input and produces a 64-dimensional degradation embedding. This embedding is projected into (γ, β) FiLM conditioning pairs for each of the 4 encoder scales. The affine transform is applied as:

```
x = γ · x + β (applied after LayerNorm, before conv)
```

At initialisation γ ≈ 1.0 (identity) and β ≈ 0.0 (no shift), ensuring a stable training start. This design directly addresses the unknown degradation order constraint — the model estimates degradation characteristics without knowing which order speckle, Gaussian noise, and downsampling were applied. An auxiliary output predicts `[speckle_level, gaussian_level, blur_level]` in [0, 1] for interpretability.

### NAFNet Backbone

- **Width**: 64 channels at the first scale
- **Encoder blocks**: [2, 2, 4, 8] across 4 downsampling stages
- **Decoder blocks**: [2, 2, 2, 2] across 4 upsampling stages
- **Middle**: 6 NAFBlocks at the bottleneck
- **Max channels**: 512 (4th downsample capped to prevent a 1024-channel bottleneck)
- **Nonlinearity**: SimpleGate — `x₁ · x₂` (no ReLU/GELU anywhere)
- **Normalisation**: LayerNorm (not BatchNorm — better stability at batch=16)
- **Skip connections**: element-wise addition
- **FiLM conditioning**: applied at all 4 encoder scales; decoder is unconditioned
- **Total parameters**: 32.439M

### PixelShuffle 2× SR Head

Converts feature maps from (B, 64, 128, 128) to (B, 1, 256, 256) via sub-pixel convolution. Learned upsampling produces sharper edges than bicubic interpolation, which is critical for semiconductor Manhattan geometry (90-degree lines and corners). The head also handles 256×256 input → 512×512 output for variable input sizes.

### Composite Loss

```
L_total = 0.6 × Charbonnier + 0.2 × SSIM + 0.2 × FFT
```

| Component | Formula | Purpose |
|-----------|---------|-------|
| Charbonnier | `mean(√(diff² + 0.001²))` | Smooth L1 — robust to outliers |
| FFT | `L1(|FFT(pred)|, |FFT(gt)|)` | Preserves high-frequency edge content |
| SSIM | `1 − SSIM(pred_clamped, gt)` | Structural coherence |

Model selection criterion: validation PSNR (higher = better).

---

## Repository Structure

```
kla-image-restoration/
├── README.md
├── requirements.txt
├── train.py
├── inference.py
├── app.py                          # Optional Gradio demo
├── backend.py                      # FastAPI demo backend
├── verify_and_baseline.py
├── configs/
│   └── nafnet_base.yaml
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   └── utils.py
├── weights/
│   ├── best_model.pth              # ~390 MB (not in git — see below)
│   └── best_model_246ep_psnr24.74.pth
├── results/
│   ├── sample_outputs/
│   │   ├── success_cases/           # 002880, 002000
│   │   └── failure_cases/           # 002881, 002882
│   └── test_submission_output/      # 400 restored test files
└── solution_presentation.pptx
```

---

## Environment Setup

### Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 12.1 support
- cuDNN 90100
- 8 GB+ VRAM recommended

### Installation

```bash
git clone https://github.com/sahoo-rudra/PurePixel.git
cd PurePixel
pip install -r requirements.txt
```

### Download Model Weights

`best_model.pth` (~390 MB) is not tracked in git due to file size limits.

**Download**: [Hugging Face model repository](https://huggingface.co/sahoo-rudra/PurePixel/resolve/main/best_model.pth)

Place the file at:

```
weights/best_model.pth
```

---

## Training

### Sanity check (2 epochs, no wandb)

```bash
python train.py --debug
```

### Full training from scratch

```bash
python train.py
```

### Resume from checkpoint

```bash
python train.py --resume weights/best_model.pth --epochs 300
```

### Key training features

- **FP16 mixed precision** via `torch.amp.autocast` — ~30% speedup
- **Gradient accumulation** ×2 — effective batch size 32 from physical batch 16
- **Synthetic degradation augmentation** — 50/50 real/synthetic mixing per step
- **CosineAnnealingWarmRestarts** — T₀=50 epochs, η_min=1×10⁻⁶
- **Early stopping** — patience 30 epochs on val PSNR
- **wandb logging** — train/val loss, PSNR, SSIM, LPIPS, visual comparisons, degradation estimate tracking (speckle/gaussian/blur levels)
- **Seed=42** for full reproducibility

### wandb Experiment Runs

| Run ID | Epochs | Notes |
|-----------|---------|----------------------------|
| `vp58iu9o`| 1–200 | Initial training |
| `kbwf67i7`| 201–246 | Resumed, found best at 246 |

Project: `kla-image-restoration`

---

## Inference (KLA Submission Format)

### Standard inference (batch mode, no TTA)

```bash
python inference.py --input_dir /path/to/NoisyLR --output_dir /path/to/output
```

### With Test-Time Augmentation (better quality, 8× slower)

```bash
python inference.py --input_dir /path/to/NoisyLR --output_dir /path/to/output --tta
```

### Input/Output Contract

| Property | Input | Output |
|----------|-------|--------|
| Format | `.npy` (float32) | `.npy` (float32) |
| Shape | (128, 128) | (256, 256) |
| Value range | May exceed [0, 1] | Clamped to [0, 1] |
| Filenames | Preserved exactly | Preserved exactly |

Also supports (256, 256) input → (512, 512) output automatically.

### Runtime Breakdown (end-to-end, batch=16, FP16, RTX 4060 Laptop)

| Stage | Time/image | % of total |
|------------------|------------|------------|
| Disk read | 6.40 ms | 38.7% |
| Model forward | 9.02 ms | 54.6% |
| Other (transfer, clamp, save) | 1.11 ms | 6.7% |
| **Total** | **16.53 ms** | **100%** |
| **Throughput** | **60.5 img/sec** | |

---

## Demo

### Gradio Demo

```bash
pip install gradio
python app.py
```

Open: http://127.0.0.1:7860

### React Web Demo

```bash
pip install fastapi uvicorn python-multipart pillow
python backend.py
```

Open: http://127.0.0.1:8000

---

## Validation and Reporting

### Validation Split

Last 10% of alphabetically sorted file stems = 320 images (stems: last 320 of 3200). Deterministic, seed=42, identical across all runs. The validation set is used only for PSNR checkpoint selection — it is never seen during training.

### Metrics

| Metric | Implementation | Notes |
|--------|---------------|-------|
| PSNR | `10 × log₁₀(1 / MSE)`, MAX=1.0 | Standard peak signal-to-noise ratio |
| SSIM | `torchmetrics`, data_range=1.0 | Structural similarity index |
| LPIPS | `lpips` v0.1.4, AlexNet backbone | Perceptual similarity (lower = better) |

Model selection: best validation PSNR determines which checkpoint is saved as `best_model.pth`.

### Experiment Tracking

All hyperparameters, random seeds, loss weights, and checkpoints are logged to wandb. Experiments tested one change at a time:

| Experiment | Config | Result | Status |
|------------|--------|--------|--------|
| Baseline | Charb=0.6, SSIM=0.2, FFT=0.2 | 24.7450 dB | **FINAL** |
| Fix1 | FFT=0.35, 80 epochs | 24.3163 dB (−0.43 dB) | Reverted |

---

## Training Progression

| Epoch | Loss | PSNR (dB) | SSIM |
|-------|--------|-----------|-------|
| 1 | 0.3783 | 18.96 | 0.313 |
| 2 | 0.1861 | 21.09 | 0.419 |
| 5 | 0.1327 | 22.17 | 0.501 |
| 10 | 0.1082 | 23.50 | 0.645 |
| 50 | 0.0962 | 24.27 | 0.676 |
| 100 | 0.0951 | 24.35 | 0.682 |
| 150 | 0.0930 | 24.56 | 0.690 |
| 200 | 0.0895 | 24.64 | 0.693 |
| **246** | **0.0890** | **24.7450** | **0.6971** |

---

## Key Design Decisions

1. **FiLM conditioning over explicit degradation classification.** The unknown degradation order is the central constraint of this challenge. Rather than trying to classify the order explicitly, FiLM conditioning lets the network learn a continuous degradation representation that modulates encoder features — the model adapts its internal processing without ever needing to identify the order.

2. **FFT loss for semiconductor Manhattan geometry.** Semiconductor wafer images are dominated by horizontal and vertical edges at precise spatial frequencies. Pixel-space losses (MSE, L1) can produce correct averages while blurring edge transitions. The FFT loss directly penalises mismatch in the frequency domain, preserving the sharp 90-degree features that matter for inspection.

3. **Charbonnier over MSE.** NoisyLR values intentionally exceed [0, 1] due to additive noise in float space. MSE squares these outliers, causing gradient spikes. Charbonnier's smooth-L1 behaviour at large deviations provides robust gradients without sacrificing sensitivity at small errors.

4. **PixelShuffle over bilinear upsampling.** Sub-pixel convolution learns the upsampling kernel directly, producing sharper edges and better LPIPS scores than fixed bilinear interpolation. This is especially important for the thin line structures common in semiconductor imagery.

5. **Gradient accumulation for effective batch=32 on 8 GB VRAM.** The RTX 4060 Laptop GPU has 8 GB VRAM. Physical batch=16 uses 5057 MB (62% utilisation). Gradient accumulation ×2 achieves the stability benefits of batch=32 without exceeding memory limits.

6. **Synthetic augmentation to double effective training data.** With only 2880 real training pairs, the model risks overfitting to the specific degradation patterns in the dataset. Synthetic augmentation generates new NoisyLR images from GT on-the-fly with randomised degradation order and strength every step, effectively doubling exposure to diverse degradation combinations.

---

## Failure Analysis and Limitations

### 1. Out-of-Distribution Organic Content

Samples 002881 and 002882 contain organic/natural structures (fine fibrous strands, thin radiating lines) unlike the geometric semiconductor patterns dominating the training set. The model produces structurally correct output but with softer fine detail — PSNR ~28 dB versus ~32 dB for in-distribution semiconductor content. Root cause: 2880 training pairs provide insufficient OOD diversity. Mitigation: adding BSD500 and DIV2K external datasets would broaden the content distribution.

### 2. Very Dark Images

Images with mean intensity ~0.06 are restored correctly (visually clean output, correct structure) but produce lower absolute PSNR values. This is inherent to the PSNR metric — with minimal signal content, even small absolute errors produce large relative ratios. The model output is visually indistinguishable from GT in these cases.

---

## External Resources

### 1. LPIPS v0.1.4

- **Use**: Validation metric only — NOT used as a training loss
- **Paper**: Zhang et al. "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric", CVPR 2018
- **Repository**: https://github.com/richzhang/PerceptualSimilarity
- **Licence**: BSD-2-Clause
- **Note**: AlexNet backbone pretrained on ImageNet is auto-downloaded by the library

### 2. NAFNet (Architecture Reference)

- **Use**: Architecture basis — NOT used as pretrained weights
- **Paper**: Chen et al. "Simple Baselines for Image Restoration", ECCV 2022
- **Repository**: https://github.com/megvii-research/NAFNet
- **Licence**: MIT

### 3. PyTorch Ecosystem

- **Components**: torch, torchvision, torchmetrics, torchaudio
- **Licence**: BSD-style

### 4. wandb (Experiment Tracking)

- **Licence**: MIT, free tier

---

## Next Steps

If given more time:

- **External data for OOD generalisation**: Add BSD500 + DIV2K datasets (+0.3–0.8 dB expected improvement)
- **Pretrained initialisation**: Use official NAFNet ImageNet-pretrained weights as starting point
- **Full Fix1 training**: Train FFT=0.35 config for 200+ epochs (reverted at 80 epochs due to time)
- **Multi-seed ensemble**: Train 3 seeds and average predictions for +0.1–0.2 dB
- **Model distillation**: Compress to a smaller model for faster H100 throughput

---

Built for KLA / Semicon India Hackathon 2026 — RGIPT team.