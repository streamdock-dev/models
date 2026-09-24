"""
export_basicvsrpp_onnx.py -- Export BasicVSR++ to ONNX for offline video upscaling
in StreamDock's batch processing queue.

Usage:
    py export_basicvsrpp_onnx.py [--out DIR] [--seq-lens 5 10] [--tile-size 256] [--no-simplify]

Requirements:
    pip install torch torchvision onnx onnxsim onnxruntime

basicsr is NOT required. The architecture is implemented fully inline using
torchvision.ops.deform_conv2d for the second-order deformable alignment modules.
No custom CUDA compilation is needed; export runs on CPU.

The script:
  1. Downloads BasicVSR++ REDS4 4x weights from the OpenMMLab model zoo (~230 MB, Apache-2.0).
  2. Implements the architecture standalone (no basicsr or mmcv required).
  3. Exports one ONNX per sequence length T with fixed tile input (default 256x256).
  4. Optionally simplifies with onnxsim.

Output files (default: ThirdParty/models/basicvsrpp/):
    basicvsrpp_x4_t5.onnx    5-frame input sequence, 256x256 tiles, 4x upscale
    basicvsrpp_x4_t10.onnx  10-frame input sequence, 256x256 tiles, 4x upscale

Input tensor shape:  (1, T, 3, tile_h, tile_w)  -- T RGB frames, normalised [0, 1], float32
Output tensor shape: (1, T, 3, tile_h*4, tile_w*4) -- T upscaled frames, float32

Offline inference in C# (BasicVsrPpUpscaler, StreamDock.Core):
    - Split each frame into 256x256 spatial tiles with overlap.
    - Batch T tiles across time (same tile position across T consecutive frames).
    - Run the model; blend and stitch output tiles.
    - Advance by T/2 frames (50% temporal overlap) to reduce clip-boundary artefacts.
    - Requires opset 18 and DirectML 1.8+ (Windows 11 22H2+) for GPU acceleration.

Architecture notes:
    BasicVSR++ uses SPyNet optical flow + second-order grid propagation with
    torchvision.ops.deform_conv2d (DCNv2) alignment. All ops are standard
    PyTorch/torchvision with ONNX symbolic support in opset 18.
    Python for-loops over T unroll to a fully static graph at trace time.
"""

from __future__ import annotations

import argparse
import math
import sys
import urllib.request
from pathlib import Path

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx

try:
    from onnxsim import simplify as onnxsim_simplify
except ImportError:
    onnxsim_simplify = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore[assignment]

# BasicVSR++ 4x REDS4 weights from OpenMMLab model zoo (Apache-2.0).
_WEIGHTS_URL = (
    "https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/"
    "basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth"
)
_WEIGHTS_NAME = "basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth"

# Architecture hyper-parameters matching the pre-trained checkpoint.
_MID_CH = 64
_NUM_BLOCKS = 7        # residual blocks per backbone branch
_DEFORM_GROUPS = 16    # deformable_groups in SecondOrderDeformableAlignment
_MAX_RESIDUE = 10      # max_residue_magnitude in SecondOrderDeformableAlignment

# Export defaults.
_TILE_SIZE = 256
_DEFAULT_SEQ_LENS = [5, 10]
_OPSET = 18

_DEFAULT_OUT = str(Path(__file__).resolve().parent / "basicvsrpp")


# ---------------------------------------------------------------------------
# Shared building blocks (same key layout as BasicSR / MMEditing v1)
# ---------------------------------------------------------------------------

class _ConvAct(nn.Module):
    """Conv2d wrapper storing conv under .conv (matches SPyNet checkpoint keys)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int = 1,
        padding: int = 0,
        act: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=True)
        self._act = nn.ReLU(inplace=True) if act else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self._act is not None:
            x = self._act(x)
        return x


class _ResBlockNoBN(nn.Module):
    """Residual block without BN (maps to ResidualBlockNoBN in basicsr)."""

    def __init__(self, mid_channels: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.relu(self.conv1(x)))


class _ConvResidualBlocks(nn.Module):
    """Maps to ConvResidualBlocks in basicsr.

    Key layout:
        main.0.weight / main.0.bias   -- projection Conv2d
        main.2.{i}.conv1.weight       -- residual block i, first conv
        main.2.{i}.conv2.weight       -- residual block i, second conv
    """

    def __init__(
        self, in_channels: int, out_channels: int = 64, num_blocks: int = 15
    ) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),  # index 0
            nn.LeakyReLU(negative_slope=0.1, inplace=True),             # index 1
            nn.Sequential(                                               # index 2
                *[_ResBlockNoBN(mid_channels=out_channels) for _ in range(num_blocks)]
            ),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.main(feat)


def _flow_warp(
    x: torch.Tensor,
    flow: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    """Warp feature map x with optical flow (pixel units, (n, h, w, 2) layout)."""
    _, _, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, device=flow.device, dtype=x.dtype),
        torch.arange(0, w, device=flow.device, dtype=x.dtype),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), dim=2)
    grid_flow = grid + flow
    grid_flow_x = 2.0 * grid_flow[..., 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[..., 1] / max(h - 1, 1) - 1.0
    grid_norm = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    return F.grid_sample(
        x, grid_norm, mode=mode, padding_mode=padding_mode, align_corners=align_corners
    )


class _SPyNetBasicModule(nn.Module):
    """One level of the SPyNet pyramid."""

    def __init__(self) -> None:
        super().__init__()
        self.basic_module = nn.Sequential(
            _ConvAct(8, 32, 7, 1, 3, act=True),
            _ConvAct(32, 64, 7, 1, 3, act=True),
            _ConvAct(64, 32, 7, 1, 3, act=True),
            _ConvAct(32, 16, 7, 1, 3, act=True),
            _ConvAct(16, 2, 7, 1, 3, act=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.basic_module(x)


class _SPyNet(nn.Module):
    """SPyNet optical flow estimator (6-level spatial pyramid)."""

    def __init__(self) -> None:
        super().__init__()
        self.basic_module = nn.ModuleList(
            [_SPyNetBasicModule() for _ in range(6)]
        )
        self.register_buffer(
            "mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _compute_flow(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        n, _, h, w = ref.size()
        ref_norm = [(ref - self.mean) / self.std]
        supp_norm = [(supp - self.mean) / self.std]
        for _ in range(5):
            ref_norm.append(
                F.avg_pool2d(ref_norm[-1], kernel_size=2, stride=2, count_include_pad=False)
            )
            supp_norm.append(
                F.avg_pool2d(supp_norm[-1], kernel_size=2, stride=2, count_include_pad=False)
            )
        ref_norm = ref_norm[::-1]
        supp_norm = supp_norm[::-1]
        flow = ref_norm[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref_norm)):
            flow_up = flow if level == 0 else (
                F.interpolate(flow, scale_factor=2, mode="bilinear", align_corners=True) * 2.0
            )
            flow = flow_up + self.basic_module[level](
                torch.cat(
                    [
                        ref_norm[level],
                        _flow_warp(
                            supp_norm[level],
                            flow_up.permute(0, 2, 3, 1),
                            padding_mode="border",
                        ),
                        flow_up,
                    ],
                    dim=1,
                )
            )
        return flow

    def forward(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        h, w = ref.shape[2], ref.shape[3]
        w_up = w if w % 32 == 0 else 32 * (w // 32 + 1)
        h_up = h if h % 32 == 0 else 32 * (h // 32 + 1)
        ref_padded = F.interpolate(ref, size=(h_up, w_up), mode="bilinear", align_corners=False)
        supp_padded = F.interpolate(supp, size=(h_up, w_up), mode="bilinear", align_corners=False)
        flow = F.interpolate(
            self._compute_flow(ref_padded, supp_padded),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        flow[:, 0] *= float(w) / float(w_up)
        flow[:, 1] *= float(h) / float(h_up)
        return flow


# ---------------------------------------------------------------------------
# Second-order deformable alignment (DCNv2 -- pure PyTorch, ONNX-compatible)
# ---------------------------------------------------------------------------

def _deform_conv2d(
    x: torch.Tensor,
    offset: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    mask: torch.Tensor,
    offset_groups: int,
) -> torch.Tensor:
    """DCNv2 (deformable conv with modulation mask) implemented via F.grid_sample.

    Drop-in replacement for torchvision.ops.deform_conv2d that exports cleanly
    to ONNX using only standard ops (Pad, Range, GridSample, Conv, etc.).
    The bilinear interpolation matches the behaviour of the C++ DCNv2 kernel:
    pixel-coordinate indexing with align_corners=True and zeros padding.

    Args:
        x:             (B, in_C, H, W) input feature map.
        offset:        (B, 2*offset_groups*kH*kW, H_out, W_out).
                       Channel layout: (y0, x0, y1, x1, ...) per offset group,
                       groups ordered from 0 to offset_groups-1.
        weight:        (out_C, in_C, kH, kW) -- conv_groups=1.
        bias:          (out_C,) bias.
        stride:        (sH, sW).
        padding:       (pH, pW).
        dilation:      (dH, dW).
        mask:          (B, offset_groups*kH*kW, H_out, W_out) modulation masks.
        offset_groups: Number of deformable groups (splits input channels equally).

    Returns:
        (B, out_C, H_out, W_out).
    """
    B, in_C, H, W = x.shape
    out_C, _, kH, kW = weight.shape
    sH, sW = stride
    pH, pW = padding
    dH, dW = dilation
    in_C_per_og = in_C // offset_groups
    H_pad = H + 2 * pH
    W_pad = W + 2 * pW
    H_out = offset.shape[2]
    W_out = offset.shape[3]

    x_pad = F.pad(x, (pW, pW, pH, pH))
    # Treat each offset group's channels as an independent batch element.
    x_grp = x_pad.view(B * offset_groups, in_C_per_og, H_pad, W_pad)

    # Separate y and x offsets and masks, shaped (B, og, kH*kW, H_out, W_out).
    # The alternating (y, x) layout means even channels are y-offsets.
    off_y = offset[:, 0::2].view(B, offset_groups, kH * kW, H_out, W_out)
    off_x = offset[:, 1::2].view(B, offset_groups, kH * kW, H_out, W_out)
    msk   = mask.view(B, offset_groups, kH * kW, H_out, W_out)

    out = torch.zeros(B, out_C, H_out, W_out, dtype=x.dtype, device=x.device)

    for kh in range(kH):
        for kw in range(kW):
            k = kh * kW + kw

            # Base sampling location in padded-input coordinates for each output position.
            base_y = torch.arange(H_out, dtype=x.dtype, device=x.device) * sH + kh * dH
            base_x = torch.arange(W_out, dtype=x.dtype, device=x.device) * sW + kw * dW

            # (B, og, H_out, W_out) -- sampling coordinates with offsets applied.
            sy = base_y.view(1, 1, H_out, 1) + off_y[:, :, k]  # (B, og, H_out, W_out)
            sx = base_x.view(1, 1, 1, W_out) + off_x[:, :, k]  # (B, og, H_out, W_out)

            # Normalise to [-1, 1] using align_corners=True convention.
            sy_n = sy / (H_pad - 1) * 2.0 - 1.0
            sx_n = sx / (W_pad - 1) * 2.0 - 1.0

            # (B * og, H_out, W_out, 2) -- grid_sample expects (x, y) order.
            grid = torch.stack([sx_n, sy_n], dim=-1).view(B * offset_groups, H_out, W_out, 2)

            # Sample input channels for all offset groups simultaneously.
            samp = F.grid_sample(x_grp, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            # (B * og, in_C_per_og, H_out, W_out) -> (B, og, in_C_per_og, H_out, W_out)
            samp = samp.view(B, offset_groups, in_C_per_og, H_out, W_out)

            # Modulation mask: (B, og, H_out, W_out) -> (B, og, 1, H_out, W_out).
            samp = samp * msk[:, :, k].unsqueeze(2)

            # Flatten back to (B, in_C, H_out, W_out) for the 1x1 conv.
            samp = samp.view(B, in_C, H_out, W_out)

            # Accumulate contribution of kernel position (kh, kw).
            out = out + F.conv2d(samp, weight[:, :, kh:kh+1, kw:kw+1])

    if bias is not None:
        out = out + bias.view(1, out_C, 1, 1)

    return out

class _SecondOrderDeformableAlignment(nn.Module):
    """Second-order deformable alignment module.

    Implements SecondOrderDeformableAlignment from basicsr without any mmcv
    dependency. Uses torchvision.ops.deform_conv2d (DCNv2) which exports to
    ONNX opset-18 DeformConv with mask support.

    Checkpoint key layout (under deform_align.{module_name}.):
        weight                  -- deformable conv kernel  [out_ch, in_ch, 3, 3]
        bias                    -- [out_ch]
        conv_offset.0.weight    -- [out_ch, 3*out_ch+4, 3, 3]
        conv_offset.0.bias
        conv_offset.2.weight    -- [out_ch, out_ch, 3, 3]
        conv_offset.2.bias
        conv_offset.4.weight    -- [out_ch, out_ch, 3, 3]
        conv_offset.4.bias
        conv_offset.6.weight    -- [27*dg, out_ch, 3, 3]
        conv_offset.6.bias
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        deformable_groups: int = 16,
        max_residue_magnitude: int = 10,
    ) -> None:
        super().__init__()
        self.max_residue_magnitude = max_residue_magnitude
        self.deformable_groups = deformable_groups
        self.stride = (1, 1)
        self.padding = (padding, padding)
        self.dilation = (1, 1)

        # Deformable conv weight and bias (correspond to ModulatedDeformConvPack
        # parent class parameters in the basicsr checkpoint).
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_channels * kernel_size * kernel_size
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

        # Offset prediction network.
        # Input: cat([cond_n1(C), feat_current(C), cond_n2(C), flow_1(2), flow_2(2)])
        #      = 3*out_channels + 4 channels.
        # Output: 27*dg channels split 9:9:9 into o1, o2, mask.
        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * out_channels + 4, out_channels, 3, 1, 1),  # index 0
            nn.LeakyReLU(negative_slope=0.1, inplace=True),           # index 1
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),           # index 2
            nn.LeakyReLU(negative_slope=0.1, inplace=True),           # index 3
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),           # index 4
            nn.LeakyReLU(negative_slope=0.1, inplace=True),           # index 5
            nn.Conv2d(out_channels, 27 * deformable_groups, 3, 1, 1), # index 6
        )
        # Zero-init last conv so offsets start at zero (as in basicsr).
        nn.init.constant_(self.conv_offset[-1].weight, 0)
        nn.init.constant_(self.conv_offset[-1].bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        extra_feat: torch.Tensor,
        flow_1: torch.Tensor,
        flow_2: torch.Tensor,
    ) -> torch.Tensor:
        """Apply second-order deformable alignment.

        Args:
            x:          (n, 2*C, H, W) -- cat([feat_prop, feat_n2]) to transform.
            extra_feat: (n, 3*C, H, W) -- cat([cond_n1, feat_current, cond_n2]).
            flow_1:     (n, 2, H, W)   -- t-1 to t flow (x, y channel order).
            flow_2:     (n, 2, H, W)   -- composed second-order flow.

        Returns:
            (n, C, H, W) aligned feature map.
        """
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)  # (n, 3C+4, H, W)
        out = self.conv_offset(extra_feat)                            # (n, 27*dg, H, W)
        o1, o2, mask = torch.chunk(out, 3, dim=1)                    # each (n, 9*dg, H, W)

        # Tanh-clamped offset with flow residue (Eq. 6 in the paper).
        offset = self.max_residue_magnitude * torch.tanh(torch.cat([o1, o2], dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        # flow_1 is (x, y); flip(1) gives (y, x) to match (row, col) offset convention.
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)               # (n, 18*dg, H, W)
        mask = torch.sigmoid(mask)                                     # (n, 9*dg, H, W)

        return _deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
            offset_groups=self.deformable_groups,
        )


# ---------------------------------------------------------------------------
# BasicVSR++ main model
# ---------------------------------------------------------------------------

class _BasicVSRPlusPlus(nn.Module):
    """BasicVSR++ (CVPR 2022): second-order grid propagation with deformable alignment.

    Four propagation passes (backward_1, forward_1, backward_2, forward_2) each with
    a SecondOrderDeformableAlignment and a ResidualBlocks backbone.
    """

    def __init__(
        self,
        mid_channels: int = 64,
        num_blocks: int = 7,
        max_residue_magnitude: int = 10,
        deformable_groups: int = 16,
    ) -> None:
        super().__init__()
        C = mid_channels
        self.mid_channels = C

        # Optical flow network.
        self.spynet = _SPyNet()

        # Initial spatial feature extraction (3 -> C, 5 residual blocks).
        self.feat_extract = _ConvResidualBlocks(3, C, 5)

        # Four propagation branches: deformable alignment + residual backbone.
        # backbone input channels: (2+i)*C for branch index i = 0,1,2,3.
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        _branch_names = ["backward_1", "forward_1", "backward_2", "forward_2"]
        for i, name in enumerate(_branch_names):
            self.deform_align[name] = _SecondOrderDeformableAlignment(
                2 * C, C, 3,
                padding=1,
                deformable_groups=deformable_groups,
                max_residue_magnitude=max_residue_magnitude,
            )
            self.backbone[name] = _ConvResidualBlocks((2 + i) * C, C, num_blocks)

        # Reconstruction (5*C -> C, 5 residual blocks).
        self.reconstruction = _ConvResidualBlocks(5 * C, C, 5)

        # Pixel-shuffle upsampling x2 x2 = x4 total.
        self.upconv1 = nn.Conv2d(C, C * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(C, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def _compute_flow(
        self, lqs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute forward and backward optical flows for all adjacent frame pairs."""
        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:].reshape(-1, c, h, w)
        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)
        flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)
        return flows_forward, flows_backward

    def _propagate(
        self,
        feats: dict[str, list[torch.Tensor]],
        flows: torch.Tensor,
        module_name: str,
    ) -> dict[str, list[torch.Tensor]]:
        """Run one directional propagation pass.

        Args:
            feats:       Feature lists keyed by branch name plus 'spatial'.
            flows:       (n, T-1, 2, H, W) optical flows for this direction.
            module_name: 'backward_1', 'forward_1', 'backward_2', or 'forward_2'.

        Returns:
            Updated feats dict with feats[module_name] filled in chronological order.
        """
        n, num_flows, _, h, w = flows.size()
        T = num_flows + 1  # number of frames

        # frame_idx: which frame to process at each step.
        # flow_idx:  which flow tensor to use at each step (index into flows dim-1).
        frame_idx = list(range(T))         # [0, 1, ..., T-1]
        flow_idx = list(range(-1, T - 1))  # [-1, 0, ..., T-2]
        if "backward" in module_name:
            frame_idx = frame_idx[::-1]    # [T-1, T-2, ..., 0]
            flow_idx = frame_idx[:]        # same reversed order

        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        feats[module_name] = []

        for i, idx in enumerate(frame_idx):
            feat_current = feats["spatial"][idx]

            if i > 0:
                # First-order: warp feat_prop by flow from the previous step.
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                cond_n1 = _flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                # Second-order features (zero for i == 1, real for i >= 2).
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:
                    feat_n2 = feats[module_name][-2]
                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    flow_n2 = flow_n1 + _flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = _flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)  # (n, 3C, H, W)
                x = torch.cat([feat_prop, feat_n2], dim=1)                  # (n, 2C, H, W)
                feat_prop = self.deform_align[module_name](x, cond, flow_n1, flow_n2)

            # Gather features from all previously completed branches + current spatial.
            feat_parts: list[torch.Tensor] = [feat_current]
            for k in feats:
                if k not in ("spatial", module_name):
                    feat_parts.append(feats[k][idx])
            feat_parts.append(feat_prop)
            feat = torch.cat(feat_parts, dim=1)  # (n, (2+branch_i)*C, H, W)
            feat_prop = feat_prop + self.backbone[module_name](feat)
            feats[module_name].append(feat_prop)

        # Backward passes are built in reverse order; restore chronological order.
        if "backward" in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def forward(self, lqs: torch.Tensor) -> torch.Tensor:
        """Process T low-resolution frames and return T 4x super-resolved frames.

        Args:
            lqs: (1, T, 3, H, W) float32 tensor normalised to [0, 1].

        Returns:
            (1, T, 3, 4H, 4W) float32 tensor.
        """
        n, t, c, h, w = lqs.size()

        feats_flat = self.feat_extract(lqs.view(-1, c, h, w))
        feats: dict[str, list[torch.Tensor]] = {
            "spatial": [feats_flat.view(n, t, -1, h, w)[:, i] for i in range(t)]
        }

        flows_forward, flows_backward = self._compute_flow(lqs)

        for iteration in [1, 2]:
            for direction in ["backward", "forward"]:
                module = f"{direction}_{iteration}"
                flows = flows_backward if direction == "backward" else flows_forward
                feats = self._propagate(feats, flows, module)

        outputs: list[torch.Tensor] = []
        for i in range(t):
            hr = torch.cat(
                [
                    feats["spatial"][i],
                    feats["backward_1"][i],
                    feats["forward_1"][i],
                    feats["backward_2"][i],
                    feats["forward_2"][i],
                ],
                dim=1,
            )
            hr = self.reconstruction(hr)
            hr = self.lrelu(self.pixel_shuffle(self.upconv1(hr)))
            hr = self.lrelu(self.pixel_shuffle(self.upconv2(hr)))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr) + self.img_upsample(lqs[:, i])
            outputs.append(hr)

        return torch.stack(outputs, dim=1)


# ---------------------------------------------------------------------------
# Weights loading
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path, label: str) -> None:
    """Download url to dest with progress display, skipping if already present."""
    if dest.exists():
        print(f"  [skip] {label} already downloaded.")
        return
    print(f"  Downloading {label} ...", end=" ", flush=True)

    def _hook(count: int, block: int, total: int) -> None:
        if total > 0:
            pct = min(100, count * block * 100 // total)
            print(f"\r  Downloading {label} ... {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, _hook)
    print(f"\r  Downloaded  {label} ({dest.stat().st_size // (1024 * 1024)} MB)")


def _build_and_load(weights_path: Path) -> _BasicVSRPlusPlus:
    """Instantiate the architecture, load pre-trained weights, return the model."""
    model = _BasicVSRPlusPlus(
        mid_channels=_MID_CH,
        num_blocks=_NUM_BLOCKS,
        max_residue_magnitude=_MAX_RESIDUE,
        deformable_groups=_DEFORM_GROUPS,
    )

    raw: dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    if "state_dict" in raw:
        raw = raw["state_dict"]
    state = raw.get("params_ema", raw.get("params", raw))

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  WARNING: missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"  WARNING: unexpected keys (first 5): {unexpected[:5]}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _export(
    model: _BasicVSRPlusPlus,
    seq_len: int,
    tile_size: int,
    out_path: Path,
    simplify: bool,
) -> None:
    """Trace and export one (seq_len, tile_size) ONNX model."""
    dummy = torch.rand(1, seq_len, 3, tile_size, tile_size, dtype=torch.float32)

    model.eval()
    print(f"  Exporting {out_path.name} ...", end=" ", flush=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),
            str(out_path),
            input_names=["lqs"],
            output_names=["output"],
            opset_version=_OPSET,
            do_constant_folding=True,
            dynamo=False,
        )
    print("done.")

    if simplify:
        if onnxsim_simplify is None:
            print("  [skip] onnxsim not installed -- skipping simplification.")
        else:
            print(f"  Simplifying {out_path.name} ...", end=" ", flush=True)
            proto = onnx.load(str(out_path))
            simplified, ok = onnxsim_simplify(proto)
            if ok:
                onnx.save(simplified, str(out_path))
                print("done.")
            else:
                print("check failed, keeping original.")

    # Quick CPU inference sanity-check.
    if ort is not None:
        try:
            sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
            inp = torch.rand(1, seq_len, 3, tile_size, tile_size).numpy()
            result = sess.run(None, {"lqs": inp})
            expected = (1, seq_len, 3, tile_size * 4, tile_size * 4)
            got = result[0].shape
            status = "PASS" if got == expected else f"SHAPE MISMATCH {got}"
            print(f"  CPU inference check (T={seq_len}, {tile_size}x{tile_size}): {status}")
        except Exception as exc:
            print(f"  CPU inference check FAILED: {exc}", file=sys.stderr)
    else:
        print("  [skip] onnxruntime not installed -- skipping inference check.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments, download weights, and export all requested ONNX models."""
    parser = argparse.ArgumentParser(
        description="Export BasicVSR++ to ONNX for offline video upscaling."
    )
    parser.add_argument(
        "--out",
        default=_DEFAULT_OUT,
        help="Output directory (default: basicvsrpp/ next to this script).",
    )
    parser.add_argument(
        "--seq-lens",
        nargs="+",
        type=int,
        default=_DEFAULT_SEQ_LENS,
        help="Sequence lengths T to export. Default: 5 10.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=_TILE_SIZE,
        help=f"Spatial tile size in pixels (must be a multiple of 32). Default: {_TILE_SIZE}.",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip onnxsim simplification pass.",
    )
    parser.add_argument(
        "--weights-dir",
        default=str(Path(__file__).resolve().parent / "_weights_cache"),
        help="Directory to cache downloaded .pth weights.",
    )
    args = parser.parse_args()

    tile_size: int = args.tile_size
    if tile_size % 32 != 0:
        print(
            f"ERROR: --tile-size {tile_size} is not a multiple of 32. "
            "SPyNet requires 32-aligned inputs.",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path(args.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / _WEIGHTS_NAME

    print(f"\n=== BasicVSR++ ONNX export ===")
    print(f"Tile size : {tile_size}x{tile_size}")
    print(f"Seq lens  : {args.seq_lens}")
    print(f"Output dir: {out_dir}")
    print()

    _download(_WEIGHTS_URL, weights_path, _WEIGHTS_NAME)
    print()

    print("Loading weights ...", end=" ", flush=True)
    model = _build_and_load(weights_path)
    print("done.")
    print()

    simplify = not args.no_simplify
    for t in args.seq_lens:
        name = f"basicvsrpp_x4_t{t}.onnx"
        out_path = out_dir / name
        print(f"[T={t}]")
        _export(model, t, tile_size, out_path, simplify)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"  Saved -> {out_path}  ({size_mb:.1f} MB)")
        print()

    print("All exports complete.")


if __name__ == "__main__":
    main()
