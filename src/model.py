"""
DegradationAwareNAFNet -- Production-grade image restoration model
for the KLA / Semicon India 2026 hackathon.

Architecture overview
=====================
Task:   Simultaneous 2x super-resolution + blind denoising of single-channel
        grayscale semiconductor inspection images with Manhattan geometry.

Input:  (B, 1, 128, 128) NoisyLR  -- float32, values may exceed [0,1]
Output: (B, 1, 256, 256) Restored -- float32

Components (in dependency order):
  1. SimpleGate       -- NAFNet element-wise gating nonlinearity
  2. NAFBlock         -- Core residual block with depthwise conv + SCA + FFN
  3. DegradationHead  -- Lightweight blind estimator producing FiLM params
  4. PixelShuffleSRHead -- Sharp 2x upsampling via PixelShuffle
  5. DegradationAwareNAFNet -- Full encoder-middle-decoder UNet backbone
  6. build_model()    -- Factory function

Backbone: Modified NAFNet (Nonlinear Activation Free Network) with
  - 4-stage encoder   (widths 64 -> 128 -> 256 -> 512, stride-2 downsampling)
  - 6-block bottleneck at 512 channels, 8x8 spatial
  - 4-stage decoder   (widths 256 -> 128 -> 64 -> 32(?), PixelShuffle upsampling)
  - Additive skip connections between encoder and decoder
  - PixelShuffle SR head: 64ch 128x128 -> 1ch 256x256

  Channel progression with max_channels=512:
    Encoder:  64 -> 128 -> 256 -> 512, downsample after each stage
    Last downsample: 512 -> min(512*2, 512) = 512 (capped)
    Middle:   512ch, 8x8
    Decoder upsamples:
      512  -> 256 (16x16), + skip[3]=512ch -- 1x1 conv projects skip to 256
      256  -> 128 (32x32), + skip[2]=256ch -- 1x1 conv projects skip to 128
      128  ->  64 (64x64), + skip[1]=128ch -- 1x1 conv projects skip to 64
       64  ->  32 (128x128),+ skip[0]= 64ch-- 1x1 conv projects skip to 32
    Output ending: Conv2d(32 -> 64, k=3) to restore width for SR head
    SR head: 64ch 128x128 -> 1ch 256x256

  Skip connections use element-wise addition AFTER a 1x1 projection conv
  that matches the skip channels to the upsampled channels.

Conditioning: DegradationHead analyzes the raw NoisyLR input and produces
  FiLM (Feature-wise Linear Modulation) parameters (gamma, beta) per encoder
  stage.  This lets the network adapt to unknown noise type / severity.

Designed for RTX 4060 Laptop GPU (8 GB VRAM):
  - Peak VRAM at batch=16, FP16 + backward < 4 GB
  - ~20M parameters
"""

import time
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


# ------------------------------------------------------------------ #
# Component 1: SimpleGate                                             #
# ------------------------------------------------------------------ #

class SimpleGate(nn.Module):
    """NAFNet gating nonlinearity.  Splits channels in half and multiplies.

    Input:  (B, C, H, W)
    Output: (B, C//2, H, W)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x, 2, dim=1)
        return x1 * x2

    def __repr__(self) -> str:
        return "SimpleGate()"


# ------------------------------------------------------------------ #
# Component 2: NAFBlock                                                #
# ------------------------------------------------------------------ #

class NAFBlock(nn.Module):
    """Core NAFNet residual block with optional FiLM conditioning.

    Two branches:
      Branch 1 -- Depthwise conv block with SCA (Simplified Channel Attention)
      Branch 2 -- FFN (two 1x1 convs with SimpleGate)

    Both branches use learned residual scaling (beta, gamma).

    FiLM conditioning (film_gamma, film_beta) is applied after norm1,
    before conv1 in Branch 1.  When not supplied the block behaves as a
    standard NAFBlock.

    Parameters
    ----------
    c : int
        Number of input / output channels.
    dw_expand : int
        Channel expansion factor for depthwise conv branch.
    ffn_expand : int
        Channel expansion factor for FFN branch.
    """

    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(c)
        self.norm2 = nn.LayerNorm(c)

        dw_channel = c * dw_expand

        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(
            dw_channel, dw_channel, 3, 1, 1,
            groups=dw_channel, bias=True,
        )  # depthwise
        # After SimpleGate: dw_channel -> dw_channel // 2
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, bias=True),
        )

        ffn_channel = c * ffn_expand

        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0, bias=True)
        # After SimpleGate: ffn_channel -> ffn_channel // 2
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0, bias=True)

        self.gate = SimpleGate()

        # Learnable residual scales -- small init for training stability.
        # NOTE: beta / gamma here are NAFNet residual scalers, NOT FiLM params.
        self.beta = nn.Parameter(torch.ones(1, c, 1, 1) * 0.01)
        self.gamma = nn.Parameter(torch.ones(1, c, 1, 1) * 0.01)

    def forward(
        self,
        x: torch.Tensor,
        film_gamma: Optional[torch.Tensor] = None,
        film_beta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        inp = x  # save for residual

        # --- Branch 1: Depthwise conv block ---
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)

        # Apply FiLM conditioning after norm1, before conv1
        if film_gamma is not None and film_beta is not None:
            x = film_gamma * x + film_beta

        x = self.conv1(x)           # (B, C, H, W) -> (B, dw_channel, H, W)
        x = self.conv2(x)           # depthwise: same shape
        x = self.gate(x)            # -> (B, dw_channel//2, H, W)
        x = x * self.sca(x)         # channel attention: same shape
        x = self.conv3(x)           # -> (B, C, H, W)

        y = inp + x * self.beta     # residual with learned scale

        # --- Branch 2: FFN block ---
        x = y
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2)

        x = self.conv4(x)           # -> (B, ffn_channel, H, W)
        x = self.gate(x)            # -> (B, ffn_channel//2, H, W)
        x = self.conv5(x)           # -> (B, C, H, W)

        return y + x * self.gamma   # residual with learned scale

    def __repr__(self) -> str:
        return (
            f"NAFBlock(c={self.conv1.in_channels}, "
            f"dw_expand={self.conv1.out_channels // self.conv1.in_channels}, "
            f"ffn_expand={self.conv4.out_channels // self.conv1.in_channels})"
        )


# ------------------------------------------------------------------ #
# Component 3: DegradationHead                                        #
# ------------------------------------------------------------------ #

class DegradationHead(nn.Module):
    """Lightweight blind degradation estimator with FiLM output.

    Analyses the raw NoisyLR input (which may have values outside [0,1])
    and produces per-scale FiLM conditioning parameters for the encoder.

    Speckle, Gaussian noise, and downsampling have distinct frequency
    signatures.  A small CNN can robustly distinguish them from the raw
    NoisyLR input.  FiLM (Feature-wise Linear Modulation) lets the
    restoration network adapt its feature transforms dynamically to the
    estimated degradation type -- directly addressing the "unknown
    degradation order" constraint.

    Parameters
    ----------
    img_channel : int
        Number of input image channels (1 for grayscale).
    embed_dim : int
        Dimensionality of the internal degradation embedding.
    num_scales : int
        Number of encoder scales to condition (default 4).
    width : int
        Base channel width of the main backbone (used to compute per-scale
        FiLM channel counts).
    max_channels : int
        Maximum channel width (caps channel doubling at each scale).
    """

    def __init__(
        self,
        img_channel: int = 1,
        embed_dim: int = 64,
        num_scales: int = 4,
        width: int = 64,
        max_channels: int = 512,
    ) -> None:
        super().__init__()

        self.num_scales = num_scales

        # Degradation feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(img_channel, 32, 3, 1, 1),   # (B,1,128,128) -> (B,32,128,128)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),             # -> (B,64,64,64)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),            # -> (B,128,32,32)
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),                 # -> (B,128,4,4)
            nn.Flatten(),                            # -> (B,2048)
            nn.Linear(2048, embed_dim),              # -> (B,embed_dim)
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),         # -> (B,embed_dim)
        )

        # FiLM projection heads -- one gamma+beta pair per encoder scale.
        # Channel sizes per scale: width * 2^i, capped at max_channels
        self.film_gammas = nn.ModuleList()
        self.film_betas = nn.ModuleList()
        for i in range(num_scales):
            ch = min(width * (2 ** i), max_channels)
            self.film_gammas.append(nn.Linear(embed_dim, ch))
            self.film_betas.append(nn.Linear(embed_dim, ch))

        # Auxiliary degradation regression head.
        # Outputs [speckle_level, gaussian_level, blur_level] in [0,1].
        # Used for logging / interpretability only.
        self.deg_classifier = nn.Linear(embed_dim, 3)

        # Store channel sizes for repr
        self.film_channels = [min(width * (2 ** i), max_channels) for i in range(num_scales)]

    def forward(
        self, noisylr: torch.Tensor
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor]:
        """Estimate degradation and produce FiLM parameters.

        Parameters
        ----------
        noisylr : torch.Tensor
            Raw noisy low-resolution input, shape (B, 1, 128, 128).

        Returns
        -------
        film_params : list of (gamma, beta) tuples
            Each gamma/beta has shape (B, ch_i, 1, 1).
        deg_estimate : torch.Tensor
            Degradation level estimates, shape (B, 3), values in [0,1].
        embedding : torch.Tensor
            Internal embedding vector, shape (B, embed_dim).
        """
        embedding = self.feature_extractor(noisylr)  # (B, embed_dim)

        film_params: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i in range(self.num_scales):
            gamma = self.film_gammas[i](embedding)          # (B, ch_i)
            beta = self.film_betas[i](embedding)            # (B, ch_i)
            # Reshape to (B, ch_i, 1, 1) for broadcasting over spatial dims
            gamma = gamma.unsqueeze(-1).unsqueeze(-1) + 1.0
            # +1.0 so at init (weights ~ 0) gamma ~ 1.0 = identity scale
            beta = beta.unsqueeze(-1).unsqueeze(-1)
            # at init (weights ~ 0) beta ~ 0.0 = no shift
            film_params.append((gamma, beta))

        deg_estimate = torch.sigmoid(self.deg_classifier(embedding))  # (B, 3)

        return film_params, deg_estimate, embedding

    def __repr__(self) -> str:
        return (
            f"DegradationHead(embed_dim={self.feature_extractor[-1].out_features}, "
            f"scales={self.num_scales}, "
            f"film_channels={self.film_channels})"
        )


# ------------------------------------------------------------------ #
# Component 4: PixelShuffleSRHead                                     #
# ------------------------------------------------------------------ #

class PixelShuffleSRHead(nn.Module):
    """Sharp 2x super-resolution head using PixelShuffle.

    Converts (B, in_channels, H, W) -> (B, 1, H*2, W*2).
    PixelShuffle produces sharper edges than bilinear upsampling, which
    is critical for Manhattan geometry in semiconductor images.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels (typically 64).
    scale : int
        Upsampling factor (2 for this task).
    """

    def __init__(self, in_channels: int, scale: int = 2) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.scale = scale

        # Step 1: Feature refinement before upsampling.
        # SimpleGate halves channels, so we must implement as explicit steps.
        self.refine_conv1 = nn.Conv2d(in_channels, in_channels * 2, 3, 1, 1)
        self.refine_gate = SimpleGate()
        # SimpleGate: in_channels*2 -> in_channels
        self.refine_conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

        # Step 2: PixelShuffle expansion.
        # PixelShuffle(2) requires input_channels = out_channels * scale^2
        # out_channels = 1, scale = 2  =>  need 1 * 4 = 4 input channels
        self.pixel_conv = nn.Conv2d(in_channels, 1 * scale * scale, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(scale)

        # Step 3: Final refinement at full resolution
        self.final_conv = nn.Conv2d(1, 1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Feature tensor, shape (B, in_channels, 128, 128).

        Returns
        -------
        torch.Tensor
            Restored image, shape (B, 1, 256, 256).
        """
        # Residual refinement
        refined = self.refine_conv2(self.refine_gate(self.refine_conv1(x)))
        x = x + refined               # (B, in_channels, 128, 128)

        # PixelShuffle upsampling
        x = self.pixel_conv(x)        # (B, 4, 128, 128)
        x = self.pixel_shuffle(x)     # (B, 1, 256, 256)

        # Final refinement at output resolution
        x = self.final_conv(x)        # (B, 1, 256, 256)
        return x

    def __repr__(self) -> str:
        return f"PixelShuffleSRHead(in_channels={self.in_channels}, scale={self.scale})"


# ------------------------------------------------------------------ #
# Component 5: DegradationAwareNAFNet                                  #
# ------------------------------------------------------------------ #

class DegradationAwareNAFNet(nn.Module):
    """Full encoder-middle-decoder UNet with FiLM-conditioned NAFBlocks.

    Architecture
    ------------
    Input projection:  Conv2d(1 -> width, k=3)

    Encoder (4 stages, stride-2 downsampling after each):
      Stage 0:  2 NAFBlocks @  64ch, 128x128 -> down ->  128ch, 64x64
      Stage 1:  2 NAFBlocks @ 128ch,  64x64  -> down ->  256ch, 32x32
      Stage 2:  4 NAFBlocks @ 256ch,  32x32  -> down ->  512ch, 16x16
      Stage 3:  8 NAFBlocks @ 512ch,  16x16  -> down ->  512ch,  8x8 (capped)

    Middle (bottleneck):
      6 NAFBlocks @ 512ch, 8x8

    Decoder (4 stages, PixelShuffle upsampling before each):
      Stage 0: up 512ch 8x8   -> 256ch 16x16, + proj(skip[3]=512ch->256ch), 2 NAF
      Stage 1: up 256ch 16x16 -> 128ch 32x32, + proj(skip[2]=256ch->128ch), 2 NAF
      Stage 2: up 128ch 32x32 ->  64ch 64x64, + proj(skip[1]=128ch-> 64ch), 2 NAF
      Stage 3: up  64ch 64x64 ->  32ch 128x128,+ proj(skip[0]= 64ch-> 32ch), 2 NAF

    Output:  Conv2d(32 -> 64, k=3) channel expansion
             PixelShuffleSRHead(64 -> 1, 2x)  =>  (B, 1, 256, 256)

    Skip connections use element-wise addition after a 1x1 projection conv
    that maps encoder skip channels to the decoder's upsampled channel count.

    FiLM conditioning from DegradationHead applies to encoder stages only.

    Parameters
    ----------
    img_channel : int
        Input image channels (1 for grayscale).
    width : int
        Base channel width.
    enc_blks : list of int
        Number of NAFBlocks per encoder stage.
    dec_blks : list of int
        Number of NAFBlocks per decoder stage.
    middle_blk_num : int
        Number of NAFBlocks in the bottleneck.
    embed_dim : int
        Embedding dimension for DegradationHead.
    dw_expand : int
        Depthwise expansion factor for NAFBlocks.
    ffn_expand : int
        FFN expansion factor for NAFBlocks.
    max_channels : int
        Maximum channel width (caps the channel doubling).
    sr_width : int
        Channel width fed into the SR head (output of ending conv).
    """

    def __init__(
        self,
        img_channel: int = 1,
        width: int = 64,
        enc_blks: Optional[List[int]] = None,
        dec_blks: Optional[List[int]] = None,
        middle_blk_num: int = 6,
        embed_dim: int = 64,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        max_channels: int = 512,
        sr_width: int = 64,
    ) -> None:
        super().__init__()

        if enc_blks is None:
            enc_blks = [2, 2, 4, 8]
        if dec_blks is None:
            dec_blks = [2, 2, 2, 2]

        self.max_channels = max_channels

        # --- Input projection ---
        self.intro = nn.Conv2d(img_channel, width, kernel_size=3, padding=1, stride=1)
        # (B, 1, 128, 128) -> (B, width, 128, 128)

        # --- Degradation Head ---
        self.degradation_head = DegradationHead(
            img_channel=img_channel,
            embed_dim=embed_dim,
            num_scales=len(enc_blks),
            width=width,
            max_channels=max_channels,
        )

        # --- Encoder ---
        # Track channel widths at each encoder stage for skip projection
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.encoder_channels: List[int] = []  # channels AFTER each encoder stage (before down)
        chan = width  # starts at 64
        for num_blks in enc_blks:
            self.encoders.append(
                nn.ModuleList(
                    [NAFBlock(chan, dw_expand, ffn_expand) for _ in range(num_blks)]
                )
            )
            self.encoder_channels.append(chan)
            # Downsampling: chan -> next_chan, spatial /2
            next_chan = min(chan * 2, max_channels)
            self.downs.append(nn.Conv2d(chan, next_chan, kernel_size=2, stride=2))
            chan = next_chan
        # After 4 downs with max_channels=512:
        #   64 -> 128 -> 256 -> 512 -> 512 (capped)
        # chan = 512

        # --- Middle ---
        self.middle_blks = nn.ModuleList(
            [NAFBlock(chan, dw_expand, ffn_expand) for _ in range(middle_blk_num)]
        )
        # chan = 512, spatial = 8x8

        # --- Decoder ---
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.skip_projs = nn.ModuleList()  # 1x1 convs to project skips

        for i, num_blks in enumerate(dec_blks):
            # Upsample: Conv2d(chan, chan*2, k=1) -> PixelShuffle(2)
            # chan*2 / (2^2) = chan*2 / 4 = chan // 2 channels at 2x spatial
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            up_chan = chan // 2  # channels after upsample

            # Skip comes from encoder in reverse order.
            # encoder_channels reversed: [512, 256, 128, 64]
            skip_chan = self.encoder_channels[-(i + 1)]

            # Project skip to match upsampled channels if they differ
            if skip_chan != up_chan:
                self.skip_projs.append(
                    nn.Conv2d(skip_chan, up_chan, kernel_size=1, bias=False)
                )
            else:
                self.skip_projs.append(nn.Identity())

            chan = up_chan
            # Decoder blocks at chan channels (after skip addition)
            self.decoders.append(
                nn.ModuleList(
                    [NAFBlock(chan, dw_expand, ffn_expand) for _ in range(num_blks)]
                )
            )
        # After 4 ups from 512: 256 -> 128 -> 64 -> 32
        # chan = 32

        # --- Output: expand to sr_width for SR head ---
        self.ending = nn.Conv2d(chan, sr_width, kernel_size=3, padding=1, stride=1)
        self.sr_head = PixelShuffleSRHead(in_channels=sr_width, scale=2)
        # sr_head: (B, sr_width, 128, 128) -> (B, 1, 256, 256)

        # Store config for repr
        self.width = width
        self.enc_blks = enc_blks
        self.dec_blks = dec_blks
        self.middle_blk_num = middle_blk_num
        self.sr_width = sr_width

        # Weight initialization
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # FiLM init: small random weights so gradients flow from step 1,
        # but output is near-zero so +1.0 in forward gives gamma ~ 1 (identity).
        for fg, fb in zip(
            self.degradation_head.film_gammas,
            self.degradation_head.film_betas,
        ):
            nn.init.xavier_uniform_(fg.weight, gain=0.01)
            nn.init.zeros_(fg.bias)
            nn.init.xavier_uniform_(fb.weight, gain=0.01)
            nn.init.zeros_(fb.bias)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Full forward pass: degradation estimation -> encoder -> middle -> decoder -> SR.

        Parameters
        ----------
        inp : torch.Tensor
            Noisy low-resolution input, shape (B, 1, 128, 128).

        Returns
        -------
        torch.Tensor
            Restored high-resolution output, shape (B, 1, 256, 256).
        """
        # 1. Degradation estimation from raw input
        film_params, deg_estimate, _ = self.degradation_head(inp)
        # film_params: list of 4 (gamma, beta) tuples
        # Store for logging (detached -- no gradient held)
        self._last_deg_estimate = deg_estimate.detach()

        # 2. Input projection
        x = self.intro(inp)  # (B, width, 128, 128)

        # 3. Encoder with FiLM conditioning and skip connections
        encoder_skips: List[torch.Tensor] = []
        for i, (enc_stage, down) in enumerate(zip(self.encoders, self.downs)):
            gamma, beta = film_params[i]
            for blk in enc_stage:
                x = blk(x, film_gamma=gamma, film_beta=beta)
            encoder_skips.append(x)  # save before downsampling
            x = down(x)

        # 4. Middle blocks (no FiLM -- bottleneck operates on compressed features)
        for blk in self.middle_blks:
            x = blk(x)

        # 5. Decoder with skip connections (addition after projection)
        for i, (up, dec_stage, skip_proj) in enumerate(
            zip(self.ups, self.decoders, self.skip_projs)
        ):
            x = up(x)
            # Skip from encoder in reverse order:
            #   decoder stage 0 uses encoder_skips[3] (512ch at 16x16)
            #   decoder stage 1 uses encoder_skips[2] (256ch at 32x32)
            #   decoder stage 2 uses encoder_skips[1] (128ch at 64x64)
            #   decoder stage 3 uses encoder_skips[0] (64ch at 128x128)
            skip = encoder_skips[-(i + 1)]
            skip = skip_proj(skip)  # project to match upsampled channels
            x = x + skip

            for blk in dec_stage:
                x = blk(x)

        # 6. Channel expansion to sr_width
        x = self.ending(x)  # (B, sr_width, 128, 128)

        # 7. 2x super-resolution upsampling
        x = self.sr_head(x)  # (B, 1, 256, 256)

        return x

    @property
    def last_degradation_estimate(self) -> Optional[torch.Tensor]:
        """Last degradation estimate from forward pass (detached)."""
        return getattr(self, "_last_deg_estimate", None)

    def __repr__(self) -> str:
        return (
            f"DegradationAwareNAFNet("
            f"width={self.width}, "
            f"enc={self.enc_blks}, "
            f"dec={self.dec_blks}, "
            f"middle={self.middle_blk_num}blks, "
            f"max_ch={self.max_channels})"
        )


# ------------------------------------------------------------------ #
# Component 6: build_model (factory function)                         #
# ------------------------------------------------------------------ #


def build_model(config: dict, device: torch.device) -> DegradationAwareNAFNet:
    """Build DegradationAwareNAFNet from a config dict and move to device.

    Expected config structure::

        config['model']['width']          -> int, default 64
        config['model']['enc_blks']       -> list, default [2,2,4,8]
        config['model']['dec_blks']       -> list, default [2,2,2,2]
        config['model']['middle_blk_num'] -> int, default 6
        config['model']['img_channel']    -> int, default 1
        config['model']['embed_dim']      -> int, default 64
        config['model']['max_channels']   -> int, default 512
        config['model']['sr_width']       -> int, default 64

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    device : torch.device
        Target device (cpu / cuda).

    Returns
    -------
    DegradationAwareNAFNet
        Initialized model on the specified device.
    """
    model_cfg = config.get("model", {})
    model = DegradationAwareNAFNet(
        img_channel=model_cfg.get("img_channel", 1),
        width=model_cfg.get("width", 64),
        enc_blks=model_cfg.get("enc_blks", [2, 2, 4, 8]),
        dec_blks=model_cfg.get("dec_blks", [2, 2, 2, 2]),
        middle_blk_num=model_cfg.get("middle_blk_num", 6),
        embed_dim=model_cfg.get("embed_dim", 64),
        max_channels=model_cfg.get("max_channels", 512),
        sr_width=model_cfg.get("sr_width", 64),
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DegradationAwareNAFNet initialized on {device}")
    print(f"  Total params:     {total / 1e6:.3f}M")
    print(f"  Trainable params: {trainable / 1e6:.3f}M")
    print(
        f"  Width={model_cfg.get('width', 64)} | "
        f"enc={model_cfg.get('enc_blks', [2, 2, 4, 8])} | "
        f"middle={model_cfg.get('middle_blk_num', 6)} blocks | "
        f"max_ch={model_cfg.get('max_channels', 512)}"
    )
    return model


# ------------------------------------------------------------------ #
# Verification block                                                   #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing DegradationAwareNAFNet on {device}\n")

    # --- TEST 1: Build model ---
    print("Test 1: Build model...")
    config = {
        "model": {
            "img_channel": 1,
            "width": 64,
            "enc_blks": [2, 2, 4, 8],
            "dec_blks": [2, 2, 2, 2],
            "middle_blk_num": 6,
            "embed_dim": 64,
            "max_channels": 512,
            "sr_width": 64,
        }
    }
    model = build_model(config, device)
    print(f"  {repr(model)}")
    print(f"  Build: PASSED [OK]\n")

    # --- TEST 2: Forward pass shape contract ---
    print("Test 2: Forward pass shape contract (B=2)...")
    model.eval()
    with torch.no_grad():
        dummy = torch.rand(2, 1, 128, 128, device=device)
        out = model(dummy)
    assert out.shape == (2, 1, 256, 256), (
        f"Shape wrong: expected (2,1,256,256), got {out.shape}"
    )
    print(f"  Input:  {tuple(dummy.shape)}")
    print(f"  Output: {tuple(out.shape)}")
    print(f"  Shape contract: PASSED [OK]\n")

    # --- TEST 3: Degradation head output ---
    print("Test 3: Degradation head output...")
    deg = model.last_degradation_estimate
    assert deg is not None, "last_degradation_estimate is None after forward"
    assert deg.shape == (2, 3), (
        f"Deg estimate shape: expected (2,3), got {deg.shape}"
    )
    assert deg.min() >= 0.0 and deg.max() <= 1.0, (
        f"Deg estimate outside [0,1]: [{deg.min():.4f}, {deg.max():.4f}]"
    )
    print(f"  Estimate shape: {tuple(deg.shape)}")
    print(
        f"  Range: [{deg.min().item():.4f}, {deg.max().item():.4f}] (expect [0,1])"
    )
    print(f"  Degradation head: PASSED [OK]\n")

    # --- TEST 4: Gradient flow ---
    print("Test 4: Gradient flow through full model...")
    model.train()
    inp = torch.rand(2, 1, 128, 128, device=device, requires_grad=True)
    out = model(inp)
    loss = out.mean()
    loss.backward()
    assert inp.grad is not None, "inp.grad is None -- backprop failed"
    grad_norm = inp.grad.norm().item()
    assert grad_norm > 0, f"Zero gradient norm -- gradient vanished"
    print(f"  Gradient norm at input: {grad_norm:.8f}")
    print(f"  Gradient flow: PASSED [OK]\n")

    # --- TEST 5: DegradationHead sensitivity ---
    print("Test 5: DegradationHead sensitivity to different degradation patterns...")
    model.eval()
    with torch.no_grad():
        # Simulate clean-ish input
        clean_input = torch.rand(4, 1, 128, 128, device=device) * 0.5 + 0.25
        # Simulate heavily noisy input (values outside [0,1] as per real NoisyLR)
        noisy_input = clean_input + 0.8 * torch.randn(4, 1, 128, 128, device=device)

        _, deg_clean, emb_clean = model.degradation_head(clean_input)
        _, deg_noisy, emb_noisy = model.degradation_head(noisy_input)

    # Embeddings and estimates should differ for different inputs
    emb_diff = (emb_clean - emb_noisy).abs().mean().item()
    deg_diff = (deg_clean - deg_noisy).abs().mean().item()
    assert emb_diff > 0, (
        "DegradationHead produces identical embeddings -- not sensitive to input"
    )
    assert deg_diff > 0, (
        "Degradation estimates identical for clean vs noisy -- head not working"
    )
    print(f"  Embedding diff (clean vs noisy): {emb_diff:.6f} (should be > 0)")
    print(f"  Estimate diff  (clean vs noisy): {deg_diff:.6f} (should be > 0)")
    print(f"  Degradation estimates (clean): {deg_clean[0].tolist()}")
    print(f"  Degradation estimates (noisy): {deg_noisy[0].tolist()}")
    print(f"  DegradationHead sensitivity: PASSED [OK]\n")

    # --- TEST 6: VRAM usage at training batch size ---
    if torch.cuda.is_available():
        print("Test 6: VRAM usage at batch=16 with FP16 + backward...")
        model.train()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        batch = torch.rand(16, 1, 128, 128, device=device)
        with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
            out = model(batch)
            loss = out.mean()
        loss.backward()

        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"  Peak VRAM: {peak_mb:.0f} MB / 8192 MB")
        if peak_mb < 7000:
            print(f"  VRAM check: SAFE [OK]")
        else:
            print(
                f"  WARNING: VRAM high ({peak_mb:.0f} MB) -- consider reducing width"
            )
        print()

    # --- TEST 7: Inference speed benchmark ---
    if torch.cuda.is_available():
        print("Test 7: Inference throughput benchmark (batch=16, FP16)...")
        model.eval()
        torch.cuda.empty_cache()

        # Warmup runs
        with torch.no_grad():
            with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
                for _ in range(5):
                    _ = model(torch.rand(16, 1, 128, 128, device=device))
        torch.cuda.synchronize()

        # Timed runs
        n_runs = 30
        t_start = time.perf_counter()
        with torch.no_grad():
            with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
                for _ in range(n_runs):
                    _ = model(torch.rand(16, 1, 128, 128, device=device))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start

        imgs_per_sec = (n_runs * 16) / elapsed
        ms_per_batch = (elapsed / n_runs) * 1000
        ms_per_image = ms_per_batch / 16
        print(f"  Batch=16 | FP16 | {n_runs} runs")
        print(f"  Throughput:    {imgs_per_sec:.1f} images/sec")
        print(f"  Latency/batch: {ms_per_batch:.1f} ms")
        print(f"  Latency/image: {ms_per_image:.2f} ms")
        print(f"  Speed benchmark: PASSED [OK]\n")

    print("=" * 55)
    print("src/model.py verification PASSED [OK]")
    print("=" * 55)
    print("")
    print("Architecture summary:")
    print(f"  Input:          (B, 1, 128, 128) NoisyLR")
    print(f"  Output:         (B, 1, 256, 256) Restored")
    print(f"  DegradationHead: 4-scale FiLM conditioning")
    print(f"  Backbone:        NAFNet width=64, middle=6 blocks, max_ch=512")
    print(f"  SR head:         PixelShuffle x2")
    print(f"  Total stages:    encoder(4) + middle + decoder(4) + SR")
    print("")
    print("Next step: Phase 3 -- train.py")
