"""
export_edvr_onnx.py -- Export EDVR to ONNX for offline video upscaling
in StreamDock's batch processing queue.

Usage:
    py export_edvr_onnx.py [--out DIR] [--variant M] [--tile-size 256] [--no-simplify]
    py export_edvr_onnx.py --variant L   # export EDVR-L (larger, slower, higher quality)

Requirements:
    pip install torch onnx onnxsim onnxruntime

The script:
  1. Downloads EDVR-M or EDVR-L weights from the BasicSR model zoo (~12-60 MB, Apache-2.0).
  2. Builds the EDVR architecture inline (no basicsr / mmcv dependency).
  3. Exports an ONNX with fixed tile input (default 256x256) and T=5 frame window.
  4. Optionally simplifies with onnxsim and verifies with onnxruntime.

Output files (default: ThirdParty/models/edvr/):
    edvr_m_x4.onnx   EDVR-M, 4x upscale, T=5 window, 256x256 tiles
    edvr_l_x4.onnx   EDVR-L, 4x upscale, T=5 window, 256x256 tiles

Input tensor name/shape:  "lqs"    (1, 5, 3, tile_h, tile_w)   -- 5 RGB frames, [0,1], float32
Output tensor name/shape: "output" (1, 3, tile_h*4, tile_w*4)  -- 4x center frame, float32

Architecture note:
    EDVR uses Pyramid Cascading Deformable (PCD) alignment and Temporal Spatial
    Attention (TSA) fusion. Deformable convolutions (DCNv2 / modulated deform conv)
    are implemented via F.grid_sample to stay within ONNX opset 18, avoiding the
    opset-19-only DeformConv operator. This matches the approach used in export_basicvsrpp_onnx.py.

Offline inference in C# (EdvrUpscaler, StreamDock.Core):
    EdvrUpscaler uses a stride-1 sliding window: for each input frame, a T=5 window
    centered on that frame is assembled (with edge-replication padding at clip boundaries)
    and fed to this ONNX model, which returns the single upscaled center frame.
    N inference runs produce N output frames for N input frames.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import gdown
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
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

# EDVR official weights from BasicSR Google Drive model zoo (Apache-2.0).
# Source: https://drive.google.com/drive/folders/11WCwsdkY-CKoIQDG-V49XaADaQckagZF
_VARIANTS: dict[str, dict[str, object]] = {
    "M": {
        "gdrive_id": "1dd6aFj-5w2v08VJTq5mS9OFsD-wALYD6",
        "weights_name": "EDVR_M_x4_SR_REDS_official-32075921.pth",
        "out_name": "edvr_m_x4.onnx",
        "num_feat": 64,
        "num_frame": 5,
        "deformable_groups": 8,
        "num_extract_block": 5,
        "num_reconstruct_block": 10,
    },
    "L": {
        "gdrive_id": "127KXEjlCwfoPC1aXyDkluNwr9elwyHNb",
        "weights_name": "EDVR_L_x4_SR_REDS_official-9f5f5039.pth",
        "out_name": "edvr_l_x4.onnx",
        "num_feat": 128,
        "num_frame": 5,
        "deformable_groups": 8,
        "num_extract_block": 5,
        "num_reconstruct_block": 40,
    },
}

_TILE_SIZE = 256
_OPSET = 18
_DEFAULT_OUT = str(Path(__file__).resolve().parent / "edvr")


# ---------------------------------------------------------------------------
# Architecture -- standalone, no basicsr / mmcv dependency
# ---------------------------------------------------------------------------

class _ResBlockNoBN(nn.Module):
    """Residual block without batch normalisation.

    Matches basicsr.archs.arch_util.ResidualBlockNoBN.
    Checkpoint keys under the parent path:
        conv1.weight / conv1.bias
        conv2.weight / conv2.bias
    """

    def __init__(self, num_feat: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.relu(self.conv1(x)))


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

    Drop-in replacement for mmcv modulated_deform_conv2d that exports cleanly to
    ONNX opset 18 using only standard ops (Pad, Range, GridSample, Conv, etc.).
    Bilinear interpolation matches the DCNv2 C++ kernel with align_corners=True
    and zeros padding.

    Args:
        x:             (B, in_C, H, W) input feature map.
        offset:        (B, 2*offset_groups*kH*kW, H_out, W_out).
                       Channel layout: (y0, x0, y1, x1, ...) per offset group.
        weight:        (out_C, in_C, kH, kW).
        bias:          (out_C,) bias.
        stride:        (sH, sW).
        padding:       (pH, pW).
        dilation:      (dH, dW).
        mask:          (B, offset_groups*kH*kW, H_out, W_out) modulation masks.
        offset_groups: Number of deformable groups.

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
    x_grp = x_pad.view(B * offset_groups, in_C_per_og, H_pad, W_pad)

    off_y = offset[:, 0::2].view(B, offset_groups, kH * kW, H_out, W_out)
    off_x = offset[:, 1::2].view(B, offset_groups, kH * kW, H_out, W_out)
    msk   = mask.view(B, offset_groups, kH * kW, H_out, W_out)

    out = torch.zeros(B, out_C, H_out, W_out, dtype=x.dtype, device=x.device)

    for kh in range(kH):
        for kw in range(kW):
            k = kh * kW + kw

            base_y = torch.arange(H_out, dtype=x.dtype, device=x.device) * sH + kh * dH
            base_x = torch.arange(W_out, dtype=x.dtype, device=x.device) * sW + kw * dW

            sy = base_y.view(1, 1, H_out, 1) + off_y[:, :, k]
            sx = base_x.view(1, 1, 1, W_out) + off_x[:, :, k]

            sy_n = sy / (H_pad - 1) * 2.0 - 1.0
            sx_n = sx / (W_pad - 1) * 2.0 - 1.0

            grid = torch.stack([sx_n, sy_n], dim=-1).view(B * offset_groups, H_out, W_out, 2)

            samp = F.grid_sample(x_grp, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            samp = samp.view(B, offset_groups, in_C_per_og, H_out, W_out)
            samp = samp * msk[:, :, k].unsqueeze(2)
            samp = samp.view(B, in_C, H_out, W_out)

            out = out + F.conv2d(samp, weight[:, :, kh:kh+1, kw:kw+1])

    if bias is not None:
        out = out + bias.view(1, out_C, 1, 1)

    return out


class _DCNv2Pack(nn.Module):
    """Modulated deformable convolution pack used in EDVR's PCD alignment.

    Matches basicsr.archs.edvr_arch.DCNv2Pack (which inherits from
    mmcv ModulatedDeformConvPack). Takes feature x to be warped and a
    separate feature feat from which offsets and mask are predicted.

    Checkpoint keys under the parent path:
        weight                 [num_feat, num_feat, 3, 3]
        bias                   [num_feat]
        conv_offset.weight     [3*deformable_groups*9, num_feat, 3, 3]
        conv_offset.bias       [3*deformable_groups*9]
    """

    def __init__(self, num_feat: int = 64, deformable_groups: int = 8) -> None:
        super().__init__()
        kH, kW = 3, 3
        self.deformable_groups = deformable_groups
        self.stride = (1, 1)
        self.padding = (1, 1)
        self.dilation = (1, 1)

        # Deformable conv weight and bias stored directly (not in a sub-module),
        # matching mmcv ModulatedDeformConv2d's parameter layout.
        self.weight = nn.Parameter(torch.empty(num_feat, num_feat, kH, kW))
        self.bias = nn.Parameter(torch.zeros(num_feat))
        fan_in = num_feat * kH * kW
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

        # Predicts (offset_y, offset_x, mask) per kernel position per group.
        self.conv_offset = nn.Conv2d(
            num_feat,
            3 * deformable_groups * kH * kW,
            kH,
            stride=1,
            padding=1,
            bias=True,
        )
        nn.init.constant_(self.conv_offset.weight, 0)
        nn.init.constant_(self.conv_offset.bias, 0)

    def forward(self, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """Apply modulated deformable convolution.

        Args:
            x:    (B, C, H, W) feature to be warped / aligned.
            feat: (B, C, H, W) feature used to predict offsets and mask.

        Returns:
            (B, C, H, W) aligned feature.
        """
        out = self.conv_offset(feat)              # (B, 3*dg*9, H, W)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat([o1, o2], dim=1)       # (B, 2*dg*9, H, W)
        mask = torch.sigmoid(mask)                # (B, dg*9, H, W)

        return _deform_conv2d(
            x, offset, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
            offset_groups=self.deformable_groups,
        )


class _PCDAlignment(nn.Module):
    """Pyramid, Cascading, Deformable (PCD) alignment for EDVR.

    Aligns neighboring frame features to reference features using a 3-level
    pyramid (L1 full-res, L2 1/2, L3 1/4) followed by a cascading refinement.

    Matches basicsr.archs.edvr_arch.PCDAlignment.
    """

    def __init__(self, num_feat: int = 64, deformable_groups: int = 8) -> None:
        super().__init__()

        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()

        for i in range(3, 0, -1):
            level = f"l{i}"
            self.offset_conv1[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
            if i == 3:
                self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            else:
                self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
            self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.dcn_pack[level] = _DCNv2Pack(num_feat, deformable_groups)
            if i < 3:
                self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)

        # Cascading refinement at full resolution.
        self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.cas_offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.cas_dcnpack = _DCNv2Pack(num_feat, deformable_groups)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(
        self,
        nbr_feat_l: list[torch.Tensor],
        ref_feat_l: list[torch.Tensor],
    ) -> torch.Tensor:
        """Align neighboring frame features to the reference frame.

        Args:
            nbr_feat_l: [L1, L2, L3] features for the neighboring frame.
            ref_feat_l: [L1, L2, L3] features for the reference frame.

        Returns:
            (B, C, H, W) aligned L1-resolution feature.
        """
        upsampled_offset: torch.Tensor | None = None
        upsampled_feat: torch.Tensor | None = None

        for i in range(3, 0, -1):
            level = f"l{i}"
            offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
            offset = self.lrelu(self.offset_conv1[level](offset))
            if i == 3:
                offset = self.lrelu(self.offset_conv2[level](offset))
            else:
                offset = self.lrelu(
                    self.offset_conv2[level](torch.cat([offset, upsampled_offset], dim=1))
                )
                offset = self.lrelu(self.offset_conv3[level](offset))

            feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset)
            if i < 3:
                feat = self.feat_conv[level](torch.cat([feat, upsampled_feat], dim=1))
            if i > 1:
                feat = self.lrelu(feat)
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)

        # Cascading refinement.
        offset = torch.cat([feat, ref_feat_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset))))
        feat = self.lrelu(self.cas_dcnpack(feat, offset))
        return feat


class _TSAFusion(nn.Module):
    """Temporal Spatial Attention (TSA) fusion for EDVR.

    Computes per-frame temporal attention via feature correlation, then applies
    a spatial attention pyramid before fusing all frames.

    Matches basicsr.archs.edvr_arch.TSAFusion.
    """

    def __init__(
        self,
        num_feat: int = 64,
        num_frame: int = 5,
        center_frame_idx: int = 2,
    ) -> None:
        super().__init__()
        self.center_frame_idx = center_frame_idx

        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = nn.Conv2d(num_frame * num_feat, num_feat, 1)
        self.spatial_attn2 = nn.Conv2d(num_feat * 2, num_feat, 1)
        self.spatial_attn3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn4 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_l1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.spatial_attn_l3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_add1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_add2 = nn.Conv2d(num_feat, num_feat, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, aligned_feat: torch.Tensor) -> torch.Tensor:
        """Fuse aligned features with temporal and spatial attention.

        Args:
            aligned_feat: (B, T, C, H, W) aligned features from PCD alignment.

        Returns:
            (B, C, H, W) fused feature.
        """
        b, t, c, h, w = aligned_feat.size()

        embedding_ref = self.temporal_attn1(
            aligned_feat[:, self.center_frame_idx, :, :, :].clone()
        )
        embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        embedding = embedding.view(b, t, -1, h, w)

        corr_l = []
        for i in range(t):
            corr = torch.sum(embedding[:, i] * embedding_ref, dim=1)  # (B, H, W)
            corr_l.append(corr.unsqueeze(1))
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))            # (B, T, H, W)
        corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w).contiguous()
        corr_prob = corr_prob.view(b, -1, h, w)                        # (B, T*C, H, W)

        aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob      # (B, T*C, H, W)
        feat = self.lrelu(self.feat_fusion(aligned_feat))               # (B, C, H, W)

        attn = self.lrelu(self.spatial_attn1(aligned_feat))             # (B, C, H, W)
        attn_max = self.max_pool(attn)                                  # (B, C, H/2, W/2)
        attn_avg = self.avg_pool(attn)
        attn = self.lrelu(
            self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1))
        )                                                               # (B, C, H/2, W/2)

        attn_level = self.lrelu(self.spatial_attn_l1(attn))
        attn_max = self.max_pool(attn_level)                            # (B, C, H/4, W/4)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.lrelu(
            self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1))
        )
        attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
        attn_level = self.upsample(attn_level)                         # (B, C, H/2, W/2)

        attn = self.lrelu(self.spatial_attn3(attn)) + attn_level       # (B, C, H/2, W/2)
        attn = self.lrelu(self.spatial_attn4(attn))
        attn = self.upsample(attn)                                      # (B, C, H, W)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(
            self.lrelu(self.spatial_attn_add1(attn))
        )
        attn = torch.sigmoid(attn)

        # * 2 keeps (attn * 2) close to 1.0 after init (scale factor from paper).
        feat = feat * attn * 2 + attn_add
        return feat


class _EDVR(nn.Module):
    """EDVR: Enhanced Deformable Video Restoration (CVPR 2019 Workshop).

    Standalone implementation matching basicsr.archs.edvr_arch.EDVR
    with with_predeblur=False and with_tsa=True (used by the REDS weights).

    Input:  (B, T, 3, H, W) -- T LR RGB frames, float32 in [0, 1].
    Output: (B, 3, H*4, W*4) -- 4x upscaled center frame, float32.

    Checkpoint key layout (after stripping params_ema / params):
        conv_first.*
        feature_extraction.{0..n}.conv1/conv2.*
        conv_l2_1.* / conv_l2_2.*
        conv_l3_1.* / conv_l3_2.*
        pcd_align.offset_conv1/2/3.l1/l2/l3.*
        pcd_align.dcn_pack.l1/l2/l3.{weight,bias,conv_offset.*}
        pcd_align.feat_conv.l1/l2.*
        pcd_align.cas_offset_conv1/2.*
        pcd_align.cas_dcnpack.{weight,bias,conv_offset.*}
        fusion.{temporal_attn1/2,feat_fusion,spatial_attn*}.*
        reconstruction.{0..n}.conv1/conv2.*
        upconv1.* / upconv2.*
        conv_hr.* / conv_last.*
    """

    def __init__(
        self,
        num_feat: int = 64,
        num_frame: int = 5,
        deformable_groups: int = 8,
        num_extract_block: int = 5,
        num_reconstruct_block: int = 10,
        center_frame_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.center_frame_idx = num_frame // 2 if center_frame_idx is None else center_frame_idx

        self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
        self.feature_extraction = nn.Sequential(
            *[_ResBlockNoBN(num_feat) for _ in range(num_extract_block)]
        )

        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.pcd_align = _PCDAlignment(num_feat, deformable_groups)
        self.fusion = _TSAFusion(num_feat, num_frame, self.center_frame_idx)

        self.reconstruction = nn.Sequential(
            *[_ResBlockNoBN(num_feat) for _ in range(num_reconstruct_block)]
        )

        # 4x upsampling: two pixel-shuffle x2 stages.
        # upconv2 always outputs 64*4 regardless of num_feat (matches basicsr).
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """4x super-resolve the center frame of a T-frame window.

        Args:
            x: (B, T, 3, H, W) LR input frames, float32 in [0, 1].

        Returns:
            (B, 3, H*4, W*4) super-resolved center frame.
        """
        b, t, c, h, w = x.size()
        x_center = x[:, self.center_frame_idx].contiguous()

        # Extract per-frame features (all T frames as one batch).
        feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        feat_l1 = self.feature_extraction(feat_l1)

        # L2 (1/2) and L3 (1/4) pyramid levels.
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        feat_l1 = feat_l1.view(b, t, -1, h, w)
        feat_l2 = feat_l2.view(b, t, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, t, -1, h // 4, w // 4)

        ref_feat_l = [
            feat_l1[:, self.center_frame_idx].clone(),
            feat_l2[:, self.center_frame_idx].clone(),
            feat_l3[:, self.center_frame_idx].clone(),
        ]

        aligned_feat = []
        for i in range(t):
            nbr_feat_l = [
                feat_l1[:, i].clone(),
                feat_l2[:, i].clone(),
                feat_l3[:, i].clone(),
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat_t = torch.stack(aligned_feat, dim=1)  # (B, T, C, H, W)

        feat = self.fusion(aligned_feat_t)
        out = self.reconstruction(feat)

        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))   # x2
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))   # x2 -> x4
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)

        base = F.interpolate(x_center, scale_factor=4, mode="bilinear", align_corners=False)
        return out + base



def _download(gdrive_id: str, dest: Path, label: str) -> None:
    """Download a Google Drive file to dest, skipping if already present."""
    if dest.exists():
        print(f"  {label}: already cached at {dest}")
        return
    print(f"  Downloading {label} from Google Drive ...", flush=True)
    gdown.download(id=gdrive_id, output=str(dest), quiet=False)
    print(f"  done ({dest.stat().st_size / 1024 / 1024:.1f} MB).")


def _build_and_load(cfg: dict[str, object], weights_path: Path) -> _EDVR:
    """Instantiate _EDVR from cfg and load pre-trained weights."""
    model = _EDVR(
        num_feat=int(cfg["num_feat"]),
        num_frame=int(cfg["num_frame"]),
        deformable_groups=int(cfg["deformable_groups"]),
        num_extract_block=int(cfg["num_extract_block"]),
        num_reconstruct_block=int(cfg["num_reconstruct_block"]),
    )
    state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    params = state.get("params_ema", state.get("params", state))
    missing, unexpected = model.load_state_dict(params, strict=False)
    if missing:
        print(f"  WARNING: missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"  INFO: unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    model.eval()
    return model


def _export(
    model: _EDVR,
    tile_size: int,
    out_path: Path,
    num_frame: int,
    simplify: bool,
) -> None:
    """Trace and export the EDVR model to ONNX, then optionally verify."""
    dummy = torch.rand(1, num_frame, 3, tile_size, tile_size, dtype=torch.float32)

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
            print("  [skip] onnxsim not installed.")
        else:
            print(f"  Simplifying {out_path.name} ...", end=" ", flush=True)
            model_proto = onnx.load(str(out_path))
            simplified, ok = onnxsim_simplify(model_proto)
            if ok:
                onnx.save(simplified, str(out_path))
                print("done.")
            else:
                print("WARNING: onnxsim simplification failed; keeping unsimplified model.")

    if _ORT_AVAILABLE:
        print("  Running ONNX Runtime inference check ...", end=" ", flush=True)
        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        out_np = sess.run(None, {"lqs": dummy.numpy()})[0]
        exp_h = tile_size * 4
        exp_w = tile_size * 4
        assert out_np.shape == (1, 3, exp_h, exp_w), (
            f"Output shape mismatch: expected (1, 3, {exp_h}, {exp_w}), got {out_np.shape}"
        )
        print(f"PASS  output shape {out_np.shape}")
    else:
        print("  [skip] onnxruntime not installed -- skipping inference check.")


def main() -> None:
    """Entry point: parse args, download, build, export."""
    parser = argparse.ArgumentParser(description="Export EDVR to ONNX.")
    parser.add_argument("--out", default=_DEFAULT_OUT, help="Output directory.")
    parser.add_argument(
        "--variant",
        choices=["M", "L"],
        default="M",
        help="EDVR variant to export: M (medium, default) or L (large).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=_TILE_SIZE,
        help="Tile spatial size in pixels (default: 256).",
    )
    parser.add_argument("--no-simplify", action="store_true", help="Skip onnxsim.")
    parser.add_argument(
        "--weights-dir",
        default=str(Path(__file__).resolve().parent / "_weights_cache"),
        help="Directory to cache downloaded .pth weights.",
    )
    args = parser.parse_args()

    cfg = _VARIANTS[args.variant]

    if args.tile_size % 4 != 0:
        print(
            f"ERROR: --tile-size {args.tile_size} is not a multiple of 4.",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path(args.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / str(cfg["weights_name"])

    print(f"\n=== EDVR-{args.variant} ONNX export ===")
    print(f"Tile size : {args.tile_size}x{args.tile_size}")
    print(f"T (frames): {cfg['num_frame']}")
    print(f"Output dir: {out_dir}")
    print()

    _download(str(cfg["gdrive_id"]), weights_path, str(cfg["weights_name"]))
    print()

    print("Loading weights ...", end=" ", flush=True)
    model = _build_and_load(cfg, weights_path)
    print("done.")
    print()

    out_path = out_dir / str(cfg["out_name"])
    print(f"[Variant={args.variant}]")
    _export(model, args.tile_size, out_path, int(cfg["num_frame"]), not args.no_simplify)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  Saved -> {out_path}  ({size_mb:.1f} MB)")
    print()

    print("Export complete.")


if __name__ == "__main__":
    main()
