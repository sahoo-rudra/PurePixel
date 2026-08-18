"""
src/metrics.py — Restoration Quality Metrics for Semiconductor Inspection

Computes PSNR, SSIM, and LPIPS for single-channel float32 restoration outputs.
All methods accept raw model output (may be slightly outside [0,1]) and handle
clamping internally per metric requirements.

LPIPS requires (B, 3, H, W) input in [-1, 1]. Our data: (B, 1, H, W) in [0, 1].
Every metric method handles this conversion internally — the caller never needs
to think about channel conversion.

Includes MetricTracker for accumulating weighted running averages across batches.

Hardware: RTX 4060 Laptop (8GB VRAM), CUDA 12.1
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import lpips


class RestorationMetrics:
    """Computes PSNR, SSIM, and LPIPS for single-channel float32 restoration outputs.

    All methods accept raw model output (may be slightly outside [0,1]) and
    handle clamping internally per metric requirements.
    Thread-safe: no mutable state, all computations are stateless.

    Args:
        device: torch.device to run computations on.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        # LPIPS with AlexNet backbone — fastest and well-calibrated for perceptual quality
        self.lpips_fn = lpips.LPIPS(net='alex', pretrained=True).to(device)
        self.lpips_fn.eval()
        # Freeze LPIPS weights permanently — these are fixed perceptual features
        for param in self.lpips_fn.parameters():
            param.requires_grad_(False)
        print(f"RestorationMetrics initialized on {device}")
        print(f"  LPIPS: AlexNet backbone, pretrained, frozen")

    def _to_lpips_input(self, tensor: torch.Tensor) -> torch.Tensor:
        """Converts (B, 1, H, W) in [0,1] → (B, 3, H, W) in [-1,1] for LPIPS.

        Step 1: clamp to [0,1] (LPIPS expects bounded input)
        Step 2: expand channel dim 1→3 (AlexNet expects RGB)
        Step 3: rescale [0,1] → [-1,1]
        """
        t = tensor.clamp(0.0, 1.0)
        t = t.repeat(1, 3, 1, 1)       # (B, 1, H, W) → (B, 3, H, W)
        t = t * 2.0 - 1.0              # [0,1] → [-1,1]
        return t

    def compute_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Peak Signal-to-Noise Ratio in dB. Higher is better.

        Formula: 10 * log10(MAX² / MSE) where MAX=1.0 for normalized images.
        Returns 100.0 for perfect prediction (MSE=0) to avoid log(0).
        """
        with torch.no_grad():
            pred_c = pred.clamp(0.0, 1.0)
            mse = F.mse_loss(pred_c, target.clamp(0.0, 1.0))
            if mse.item() < 1e-10:
                return 100.0
            psnr = 10.0 * torch.log10(torch.tensor(1.0, device=self.device) / mse)
            return psnr.item()

    def compute_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Structural Similarity Index. Range [0,1], higher is better.

        data_range=1.0 because GT is normalized to [0,1].
        """
        from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
        with torch.no_grad():
            pred_c = pred.clamp(0.0, 1.0)
            ssim_val = ssim_fn(pred_c, target.clamp(0.0, 1.0), data_range=1.0)
            return ssim_val.item()

    def compute_lpips(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Learned Perceptual Image Patch Similarity. Range [0,1], lower is better.

        Uses AlexNet features. Automatically handles grayscale→3ch conversion.
        """
        with torch.no_grad():
            pred_lp = self._to_lpips_input(pred)
            target_lp = self._to_lpips_input(target)
            score = self.lpips_fn(pred_lp, target_lp)
            return score.mean().item()

    def compute_all(self,
                    pred: torch.Tensor,
                    target: torch.Tensor) -> Dict[str, float]:
        """Compute all three metrics in one call.

        Both tensors must be on self.device.
        Returns dict with keys: psnr, ssim, lpips
        """
        return {
            "psnr": self.compute_psnr(pred, target),
            "ssim": self.compute_ssim(pred, target),
            "lpips": self.compute_lpips(pred, target)
        }

    def __repr__(self) -> str:
        return f"RestorationMetrics(device={self.device}, lpips=alex)"


class MetricTracker:
    """Accumulates weighted running averages of metrics across batches.

    Reset between epochs. Thread-safe for single-threaded training loops.

    Args:
        metric_names: List of metric names to track (default: psnr, ssim, lpips).
    """

    def __init__(self, metric_names: List[str] = None) -> None:
        self.metric_names = metric_names or ["psnr", "ssim", "lpips"]
        self.reset()

    def reset(self) -> None:
        """Reset all accumulators to zero."""
        self.sums = {k: 0.0 for k in self.metric_names}
        self.counts = {k: 0 for k in self.metric_names}

    def update(self, metrics_dict: Dict[str, float], batch_size: int) -> None:
        """Update running averages with a new batch of metrics."""
        for k, v in metrics_dict.items():
            if k in self.sums:
                self.sums[k] += v * batch_size
                self.counts[k] += batch_size

    def compute(self) -> Dict[str, float]:
        """Compute current running averages."""
        return {
            k: (self.sums[k] / self.counts[k] if self.counts[k] > 0 else 0.0)
            for k in self.metric_names
        }

    def pretty_print(self, prefix: str = "", epoch: Optional[int] = None) -> None:
        """Print formatted metric summary."""
        avgs = self.compute()
        ep_str = f"Epoch {epoch} | " if epoch is not None else ""
        print(f"{prefix}{ep_str}"
              f"PSNR: {avgs.get('psnr', 0):.4f} dB | "
              f"SSIM: {avgs.get('ssim', 0):.4f} | "
              f"LPIPS: {avgs.get('lpips', 0):.4f}")

    def __repr__(self) -> str:
        return f"MetricTracker(metrics={self.metric_names}, samples={list(self.counts.values())})"


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing metrics on {device}\n")

    rm = RestorationMetrics(device)

    # Test 1: random pred vs target (should give moderate scores)
    pred = torch.rand(4, 1, 256, 256, device=device)
    target = torch.rand(4, 1, 256, 256, device=device)
    metrics = rm.compute_all(pred, target)
    print(f"Random pred vs random target:")
    print(f"  PSNR:  {metrics['psnr']:.4f} dB  (expect ~7-9 dB for random)")
    print(f"  SSIM:  {metrics['ssim']:.4f}     (expect ~0.0 for random)")
    print(f"  LPIPS: {metrics['lpips']:.4f}    (expect ~0.7+ for random)")

    # Test 2: perfect prediction
    perfect_metrics = rm.compute_all(target.clone(), target)
    print(f"\nPerfect prediction (pred == target):")
    print(f"  PSNR:  {perfect_metrics['psnr']:.2f} dB  (expect 100.0)")
    print(f"  SSIM:  {perfect_metrics['ssim']:.4f}     (expect 1.0)")
    print(f"  LPIPS: {perfect_metrics['lpips']:.6f}    (expect ~0.0)")
    assert perfect_metrics['psnr'] >= 99.0, "PSNR wrong at perfect prediction"
    assert perfect_metrics['ssim'] >= 0.999, "SSIM wrong at perfect prediction"

    # Test 3: MetricTracker accumulation
    tracker = MetricTracker()
    for i in range(5):
        noisy_pred = target + 0.05 * torch.randn_like(target)
        tracker.update(rm.compute_all(noisy_pred.clamp(0, 1), target), batch_size=4)
    print(f"\nMetricTracker after 5 batches (slightly noisy pred):")
    tracker.pretty_print(prefix="  ", epoch=10)
    avg = tracker.compute()
    assert avg['psnr'] > 20.0, f"PSNR too low for slightly noisy pred: {avg['psnr']}"

    # Test 4: Out-of-range NoisyLR handling
    oor_pred = target + 0.3 * torch.randn_like(target)   # values outside [0,1]
    oor_metrics = rm.compute_all(oor_pred, target)
    print(f"\nOut-of-range pred (clamped internally):")
    print(f"  PSNR: {oor_metrics['psnr']:.4f} dB")
    print(f"  (No crash = internal clamping works [OK])")

    print("\nMetrics verification PASSED [OK]")
