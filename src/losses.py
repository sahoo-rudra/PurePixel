"""
src/losses.py — Composite Loss Functions for Semiconductor Image Restoration

Designed specifically for semiconductor Manhattan geometry images (90° lines,
rectangular contacts, sharp edges). Three complementary loss components:

  1. Charbonnier: robust pixel fidelity (smooth L1, handles noise outliers)
  2. SSIM:       structural coherence and contrast preservation
  3. FFT:        high-frequency edge sharpness in frequency domain

Default weights: Charbonnier=0.6 / SSIM=0.2 / FFT=0.2
  - Charbonnier dominates for PSNR-friendly pixel accuracy
  - FFT preserves semiconductor edge sharpness (Manhattan geometry)
  - SSIM ensures structural coherence across the upscaled image

Hardware: RTX 4060 Laptop (8GB VRAM), FP16 mixed precision compatible
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Smooth approximation of L1 loss.

    More robust than MSE to outlier pixels (e.g. extreme noise spikes in
    NoisyLR that bleed into predictions).

    Formula: mean(sqrt(diff² + eps²))
    At diff=0: gradient = 0 (unlike L1 which is undefined)
    At large diff: behaves like L1 (unlike L2 which over-penalizes)

    Args:
        eps: Small constant for numerical stability (default 0.001).
    """

    def __init__(self, eps: float = 0.001) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))

    def __repr__(self) -> str:
        return f"CharbonnierLoss(eps={self.eps})"


class FFTLoss(nn.Module):
    """L1 loss on FFT magnitude spectra.

    Critical for semiconductor images because:
    - Manhattan geometry (90° lines) creates strong horizontal/vertical
      frequency peaks
    - Pixel losses (L1/L2) are blind to phase/frequency — they allow blurry
      predictions that have correct pixel averages but missing edge sharpness
    - FFT loss directly penalizes missing high-frequency edge content

    Uses ortho normalization so loss scale is independent of image size.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Both inputs: (B, 1, H, W)
        pred_fft = torch.fft.fft2(pred, norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')
        # Magnitude spectrum — torch.abs handles complex tensors correctly
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        return F.l1_loss(pred_mag, target_mag)

    def __repr__(self) -> str:
        return "FFTLoss(norm='ortho')"


class SSIMLoss(nn.Module):
    """Returns (1 - SSIM) so it's minimizable.

    SSIM captures luminance, contrast, and structure simultaneously —
    complementary to pixel-wise Charbonnier.

    Clamps pred to [0,1] internally ONLY for SSIM computation (SSIM requires
    bounded input). The clamp is NOT applied globally — other losses receive
    raw predictions.

    Args:
        data_range: The value range of the input images (default 1.0).
    """

    def __init__(self, data_range: float = 1.0) -> None:
        super().__init__()
        self.data_range = data_range

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
        pred_clamped = pred.clamp(0.0, 1.0)   # local clamp for SSIM only
        ssim_val = ssim_fn(pred_clamped, target, data_range=self.data_range)
        return 1.0 - ssim_val

    def __repr__(self) -> str:
        return f"SSIMLoss(data_range={self.data_range})"


class CompositeLoss(nn.Module):
    """Weighted sum of Charbonnier + SSIM + FFT losses.

    Default weights (0.6 / 0.2 / 0.2) are tuned for the SR+denoising task:
    - Charbonnier dominates to ensure pixel accuracy (PSNR-friendly)
    - FFT at 0.2 is strong enough to preserve semiconductor edge sharpness
    - SSIM at 0.2 ensures structural coherence across the upscaled image

    Weights should sum to 1.0 for interpretable loss scale.

    Args:
        charbonnier_weight: Weight for Charbonnier loss (default 0.6).
        ssim_weight:        Weight for SSIM loss (default 0.2).
        fft_weight:         Weight for FFT loss (default 0.2).
        eps:                Charbonnier epsilon (default 0.001).
    """

    def __init__(self,
                 charbonnier_weight: float = 0.6,
                 ssim_weight: float = 0.2,
                 fft_weight: float = 0.2,
                 eps: float = 0.001) -> None:
        super().__init__()
        self.w_c = charbonnier_weight
        self.w_s = ssim_weight
        self.w_f = fft_weight

        # Validate weights
        total = charbonnier_weight + ssim_weight + fft_weight
        if abs(total - 1.0) > 1e-3:
            print(f"WARNING: Loss weights sum to {total:.3f}, not 1.0. "
                  f"Loss scale will be {total:.3f}x expected.")

        self.charbonnier = CharbonnierLoss(eps=eps)
        self.ssim_loss = SSIMLoss(data_range=1.0)
        self.fft_loss = FFTLoss()

        print(f"CompositeLoss | charbonnier={charbonnier_weight} | "
              f"ssim={ssim_weight} | fft={fft_weight} | eps={eps}")

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute composite loss.

        Args:
            pred:   (B, 1, 256, 256) — model output, may have values outside [0,1]
            target: (B, 1, 256, 256) — ground truth, values in [0,1]

        Returns:
            total_loss: scalar tensor with gradient graph attached
            loss_dict:  detached floats for logging — does NOT affect backprop
        """
        l_charb = self.charbonnier(pred, target)
        l_ssim = self.ssim_loss(pred, target)     # internal clamping for SSIM
        l_fft = self.fft_loss(pred, target)

        total = self.w_c * l_charb + self.w_s * l_ssim + self.w_f * l_fft

        loss_dict = {
            "total": total.item(),
            "charbonnier": l_charb.item(),
            "ssim": l_ssim.item(),
            "fft": l_fft.item()
        }
        return total, loss_dict

    def __repr__(self) -> str:
        return f"CompositeLoss(w_c={self.w_c}, w_s={self.w_s}, w_f={self.w_f})"


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing losses on {device}")

    criterion = CompositeLoss(
        charbonnier_weight=0.6, ssim_weight=0.2, fft_weight=0.2
    ).to(device)

    # Simulate a realistic training batch
    # pred: model output — may exceed [0,1] before clamping
    pred = torch.rand(4, 1, 256, 256, device=device, requires_grad=True)
    pred_noisy = pred + 0.1 * torch.randn_like(pred)   # add some out-of-range vals
    target = torch.rand(4, 1, 256, 256, device=device)

    total, loss_dict = criterion(pred_noisy, target)
    print(f"\nLoss values:")
    for k, v in loss_dict.items():
        print(f"  {k:12s}: {v:.6f}")

    # Verify gradient flows back through all components
    total.backward()
    # pred is a leaf with requires_grad=True, and pred_noisy = pred + noise
    # creates a computation graph, so gradients flow back to pred
    assert pred.grad is not None, "Gradient did not flow through CompositeLoss!"
    print(f"\nGradient flow: OK (grad norm = {pred.grad.norm().item():.6f})")

    # Double-check: direct pred with requires_grad also works
    pred2 = torch.rand(4, 1, 256, 256, device=device, requires_grad=True)
    total2, _ = criterion(pred2, target)
    total2.backward()
    assert pred2.grad is not None, "Gradient did not flow through CompositeLoss!"
    print(f"Gradient flow (direct): OK (grad norm = {pred2.grad.norm().item():.6f})")

    # Verify loss decreases when pred → target (sanity check)
    perfect_pred = target.clone().requires_grad_(True)
    total_perfect, dict_perfect = criterion(perfect_pred, target)
    assert dict_perfect["charbonnier"] < 1.5e-3, \
        f"Charbonnier not near-zero at perfect pred: {dict_perfect['charbonnier']}"
    assert dict_perfect["ssim"] < 1e-3, "SSIM loss not near-zero at perfect pred"
    print(f"  Perfect prediction loss: {dict_perfect['total']:.8f} (near-zero [OK])")

    print("\nLoss verification PASSED [OK]")
