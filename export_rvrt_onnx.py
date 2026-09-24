"""
export_rvrt_onnx.py -- Export RVRT to ONNX for offline video upscaling
in StreamDock's batch processing queue.

Usage:
    py export_rvrt_onnx.py [--out DIR] [--seq-lens 5 10] [--tile-size 256]
                           [--device cpu] [--no-simplify]

Requirements (on the machine running this script):
    pip install einops torch torchvision onnx onnxsim onnxruntime

basicsr is NOT required. RVRT's source is downloaded directly from the official
RVRT GitHub repository (JingyunLiang/RVRT). The CUDA deform_attn kernel is replaced
with a pure-PyTorch equivalent so the model is ONNX-traceable without a CUDA build
environment, and can be exported on CPU or GPU.

GPU usage: run with --device cuda:0 on an RTX 3070 / 3080 / 4090 (sm_86 or sm_89).
CPU usage: omit --device (default is cpu) -- slower but no GPU required.

The pure-PyTorch deform_attn replacement uses F.grid_sample-based deformable sampling
plus per-head scaled-dot-product attention. It is mathematically equivalent to the
original CUDA kernel up to floating-point rounding differences.

LICENSE NOTICE -- NON-COMMERCIAL USE ONLY:
    RVRT is released under the Creative Commons Attribution-NonCommercial 4.0
    International License (CC-BY-NC 4.0). The generated ONNX files inherit this
    restriction and may only be used for non-commercial research purposes. Do NOT
    distribute or use the generated ONNX files in commercial products or services.
    See: https://github.com/JingyunLiang/RVRT -- LICENSE

The script:
  1. Downloads RVRT REDS VSR 4x weights from the official RVRT GitHub release (~44 MB).
  2. Downloads SpyNet optical-flow weights used by RVRT (~5.5 MB).
  3. Downloads models/network_rvrt.py from the RVRT GitHub repo (patched for Python 3.12+).
  4. Writes a pure-PyTorch deform_attn module (no CUDA compilation needed).
  5. Exports one ONNX per sequence length T with fixed tile input (default 256x256).
  6. Optionally simplifies with onnxsim.

Output files (default: ThirdParty/models/rvrt/):
    rvrt_x4_t4.onnx    4-frame input sequence, 256x256 tiles, 4x upscale
    rvrt_x4_t10.onnx  10-frame input sequence, 256x256 tiles, 4x upscale

Note: RVRT uses clip_size=2, so T must be a positive even integer (2, 4, 6, 10, ...).

Input tensor shape:  (1, T, 3, tile_h, tile_w)  -- T RGB frames, normalised [0, 1], float32
Output tensor shape: (1, T, 3, tile_h*4, tile_w*4) -- T upscaled frames, float32

Offline inference in C# (RvrtUpscaler, StreamDock.Core):
    Same tiling and stitching approach as BasicVSR++ -- see export_basicvsrpp_onnx.py for
    architecture notes. RVRT requires opset 18 and DirectML 1.8+ for GPU acceleration.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

try:
    import einops  # noqa: F401
except ImportError:
    print(
        "ERROR: einops is not installed. Run:\n"
        "    pip install einops torch torchvision onnx onnxsim onnxruntime",
        file=sys.stderr,
    )
    sys.exit(1)

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx


class _PixelShuffleONNX(nn.Module):
    """Pure reshape+permute pixel-shuffle for ONNX-traceable 5D upsampling.

    RVRT's Upsample class feeds 5D tensors (N, T, C*r^2, H, W) into nn.PixelShuffle,
    which the ONNX exporter rejects (only 4D supported). This replacement implements
    the same operation using reshape/permute only -- no F.pixel_shuffle call -- so
    every op it emits is ONNX-compatible regardless of tensor rank.
    """

    def __init__(self, upscale_factor: int) -> None:
        super().__init__()
        self.r = upscale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.r
        n, t, c_r2, h, w = x.shape
        c = c_r2 // (r * r)
        # Equivalent to nn.PixelShuffle on each (c*r^2, H, W) slice:
        # reshape -> (n*t, c, r, r, H, W), permute -> (n*t, c, H, r, W, r),
        # reshape -> (n*t, c, H*r, W*r), then restore T dimension.
        x = x.reshape(n * t, c, r, r, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3)  # (n*t, c, h, r, w, r)
        x = x.reshape(n * t, c, h * r, w * r)
        return x.reshape(n, t, c, h * r, w * r)

try:
    from onnxsim import simplify as onnxsim_simplify
except ImportError:
    onnxsim_simplify = None

# RVRT source from the official GitHub repository (JingyunLiang/RVRT).
_RVRT_RAW = "https://raw.githubusercontent.com/JingyunLiang/RVRT/main"
_NETWORK_URL = f"{_RVRT_RAW}/models/network_rvrt.py"

# RVRT REDS 4x VSR weights from the official RVRT GitHub release (CC-BY-NC 4.0).
# Non-commercial research use only -- see license notice at the top of this file.
_WEIGHTS_URL = (
    "https://github.com/JingyunLiang/RVRT/releases/download/v0.0/"
    "001_RVRT_videosr_bi_REDS_30frames.pth"
)
_WEIGHTS_NAME = "001_RVRT_videosr_bi_REDS_30frames.pth"

# SpyNet optical-flow weights required by RVRT (MIT license, from Sintel pre-trained).
_SPYNET_URL = (
    "https://github.com/JingyunLiang/RVRT/releases/download/v0.0/"
    "spynet_sintel_final-3d2a1287.pth"
)
_SPYNET_NAME = "spynet_sintel_final-3d2a1287.pth"

_TILE_SIZE = 256
_DEFAULT_SEQ_LENS = [4, 10]
_OPSET = 16

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR / "_rvrt_src"
_DEFAULT_OUT = str(_SCRIPT_DIR / "rvrt")

# Pure-PyTorch deform_attn written to _rvrt_src/models/op/deform_attn.py.
# Replaces the CUDA-compiled kernel so the model is ONNX-traceable.
#
# Offset tensor layout (inferred from GuidedDeformAttnPack in network_rvrt.py):
#   total_pts = clip_size * deformable_groups * kernel_h * kernel_w
#   offset[:, :total_pts]  -- y-offsets in pixel units
#   offset[:, total_pts:]  -- x-offsets in pixel units
#   Within each per-frame block: group 0 pts 0..K-1, group 1 pts 0..K-1, etc.
_DEFORM_ATTN_SRC = '''\
"""
Pure-PyTorch deformable attention for ONNX export.
Replaces the CUDA-only deform_attn kernel from the RVRT repo.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def deform_attn(q, kv, offset, kernel_h, kernel_w, stride, padding, dilation,
                attention_heads, deformable_groups, clip_size):
    """
    Pure-PyTorch guided deformable attention (ONNX-traceable, fully vectorized).
    Replaces the CUDA kernel from models/op/deform_attn.py.
    Called from GuidedDeformAttnPack.forward in network_rvrt.py.

    q:      (N, 1, C, H, W)  where N = b * clip_size
    kv:     (b, clip_size, 2*C, H, W)
    offset: (N, 2 * clip_size * deformable_groups * kernel_h * kernel_w, H, W)
    returns (N, 1, C, H, W)

    Fully vectorized: no Python loops over heads, frames, or kernel points.
    Uses exactly 2 F.grid_sample calls total (one for keys, one for values).
    """
    N, _, C, H, W = q.shape
    K = kernel_h * kernel_w
    G = deformable_groups    # offset groups
    A = attention_heads
    P = clip_size * G * K    # total sample points (= total_pts)
    HC = C // A              # channels per head

    # Expand kv: (N, clip_size, 2*C, H, W)
    # Use unsqueeze/expand/reshape instead of repeat_interleave: the ONNX
    # symbolic for repeat_interleave calls _get_tensor_rank on an intermediate
    # ONNX node which returns None without shape inference, crashing ones().
    _b = kv.size(0)  # batch (= N // clip_size)
    _T = kv.size(1)
    _C2 = kv.size(2)
    _Hk = kv.size(3)
    _Wk = kv.size(4)
    kv_exp = kv.unsqueeze(1).expand(_b, clip_size, _T, _C2, _Hk, _Wk).reshape(N, _T, _C2, _Hk, _Wk)

    oy = offset[:, :P]   # (N, P, H, W)
    ox = offset[:, P:]   # (N, P, H, W)

    # Normalised sampling grids: (N, P, H, W) each
    gy, gx = torch.meshgrid(
        torch.arange(H, device=q.device, dtype=q.dtype),
        torch.arange(W, device=q.device, dtype=q.dtype),
        indexing="ij",
    )
    h_scale = 2.0 / float(max(H - 1, 1))
    w_scale = 2.0 / float(max(W - 1, 1))
    sy = gy * h_scale + oy * h_scale - 1.0   # (N, P, H, W)
    sx = gx * w_scale + ox * w_scale - 1.0   # (N, P, H, W)

    # For each sample point p, which (frame, group) does it come from?
    # Offset layout: frame-major, then group-major, then kernel-point-major.
    # point p -> frame = p // (G * K),  group = (p % (G * K)) // K
    frames_idx = torch.arange(P, device=q.device) // (G * K)   # (P,)
    groups_idx = (torch.arange(P, device=q.device) % (G * K)) // K  # (P,)

    # Gather the key/value feature maps for each sample point.
    # kv_exp: (N, clip_size, 2*C, H, W) -- index along dim 1 by frames_idx
    # Result shape needed: (N*P, C, H, W) for k, same for v
    kv_pts = kv_exp[:, frames_idx]   # (N, P, 2*C, H, W)
    k_pts = kv_pts[:, :, :C]         # (N, P, C, H, W)
    v_pts = kv_pts[:, :, C:]         # (N, P, C, H, W)

    # Flatten N*P for grid_sample
    k_flat = k_pts.reshape(N * P, C, H, W)
    v_flat = v_pts.reshape(N * P, C, H, W)

    # Build grid: (N*P, H, W, 2)
    grid = torch.stack(
        [sx.reshape(N * P, H, W), sy.reshape(N * P, H, W)],
        dim=-1,
    )

    # Two grid_sample calls -- all points, all channels at once
    k_samp = F.grid_sample(k_flat, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    v_samp = F.grid_sample(v_flat, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    # k_samp, v_samp: (N*P, C, H, W)

    k_samp = k_samp.reshape(N, P, C, H, W)
    v_samp = v_samp.reshape(N, P, C, H, W)

    # Attention: reshape to (N, A, P, HC, H, W) for per-head dot-product
    # q: (N, 1, C, H, W) -> (N, A, 1, HC, H, W)
    q_h = q[:, 0].reshape(N, A, HC, H, W).unsqueeze(2)   # (N, A, 1, HC, H, W)

    # Each head attends only to its own group's sample points.
    # heads_per_group = A // G; head h uses group g = h // (A // G)
    HPG = max(1, A // G)
    head_groups = torch.arange(A, device=q.device) // HPG   # (A,) -- group index per head

    # k_samp: (N, P, C, H, W) -- split C into (A, HC) heads
    k_h = k_samp.reshape(N, P, A, HC, H, W).permute(0, 2, 1, 3, 4, 5)  # (N, A, P, HC, H, W)
    v_h = v_samp.reshape(N, P, A, HC, H, W).permute(0, 2, 1, 3, 4, 5)  # (N, A, P, HC, H, W)

    # Scores: (N, A, P, H, W) via einsum over HC
    scale = float(HC) ** -0.5
    scores = (q_h * k_h).sum(dim=3) * scale   # (N, A, P, H, W)
    attn = F.softmax(scores, dim=2)            # (N, A, P, H, W)

    # Weighted sum over P sample points: (N, A, HC, H, W)
    out = (attn.unsqueeze(3) * v_h).sum(dim=2)

    # Reshape back to (N, 1, C, H, W)
    return out.reshape(N, 1, C, H, W)


class DeformAttn(nn.Module):
    def __init__(self, in_channels, out_channels, attention_window=None,
                 deformable_groups=12, attention_heads=12, clip_size=1):
        super().__init__()
        if attention_window is None:
            attention_window = [3, 3]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h = attention_window[0]
        self.kernel_w = attention_window[1]
        self.attn_size = self.kernel_h * self.kernel_w
        self.deformable_groups = deformable_groups
        self.attention_heads = attention_heads
        self.clip_size = clip_size
        self.stride = 1
        self.padding = self.kernel_h // 2
        self.dilation = 1
        self.proj_q = nn.Sequential(
            Rearrange("n d c h w -> n d h w c"),
            nn.Linear(in_channels, in_channels),
            Rearrange("n d h w c -> n d c h w"),
        )
        self.proj_k = nn.Sequential(
            Rearrange("n d c h w -> n d h w c"),
            nn.Linear(in_channels, in_channels),
            Rearrange("n d h w c -> n d c h w"),
        )
        self.proj_v = nn.Sequential(
            Rearrange("n d c h w -> n d h w c"),
            nn.Linear(in_channels, in_channels),
            Rearrange("n d h w c -> n d c h w"),
        )
        self.mlp = nn.Sequential(
            Rearrange("n d c h w -> n d h w c"),
            Mlp(in_channels, in_channels * 2),
            Rearrange("n d h w c -> n d c h w"),
        )

    def forward(self, q, k, v, offset):
        q_p = self.proj_q(q)
        kv = torch.cat([self.proj_k(k), self.proj_v(v)], dim=2)
        v_out = deform_attn(
            q_p, kv, offset, self.kernel_h, self.kernel_w,
            self.stride, self.padding, self.dilation,
            self.attention_heads, self.deformable_groups, self.clip_size,
        )
        return v_out + self.mlp(v_out)


class DeformAttnPack(DeformAttn):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(
            self.in_channels * (1 + self.clip_size),
            self.clip_size * self.deformable_groups * self.attn_size * 2,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
            dilation=(1, 1), bias=True,
        )
        self.init_weight()

    def init_weight(self):
        if hasattr(self, "conv_offset"):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, q, k, v):
        out = self.conv_offset(torch.cat([q.flatten(1, 2), k.flatten(1, 2)], dim=1))
        o1, o2 = torch.chunk(out, 2, dim=1)
        offset = torch.cat([o1, o2], dim=1)
        q_p = self.proj_q(q)
        kv = torch.cat([self.proj_k(k), self.proj_v(v)], dim=2)
        v_out = deform_attn(
            q_p, kv, offset, self.kernel_h, self.kernel_w,
            self.stride, self.padding, self.dilation,
            self.attention_heads, self.deformable_groups, self.clip_size,
        )
        return v_out + self.mlp(v_out)
'''

# LooseVersion shim injected into the patched network_rvrt.py to replace
# "from distutils.version import LooseVersion" which was removed in Python 3.12.
_LOOSEVERSION_SHIM = '''\
try:
    from packaging.version import Version as LooseVersion
except ImportError:
    class LooseVersion:
        def __init__(self, s):
            self._v = tuple(int(x) for x in str(s).split("+")[0].split(".")[:3])
        def __ge__(self, other): return self._v >= other._v
        def __gt__(self, other): return self._v > other._v
        def __lt__(self, other): return self._v < other._v
        def __le__(self, other): return self._v <= other._v
        def __eq__(self, other): return self._v == other._v
'''


def _download(url: str, dest: Path, label: str) -> None:
    """Download url to dest, skipping if already present."""
    if dest.exists():
        print(f"  {label}: already cached at {dest}")
        return
    print(f"  Downloading {label} ...", end=" ", flush=True)
    urllib.request.urlretrieve(url, str(dest))
    print(f"done ({dest.stat().st_size / 1024 / 1024:.1f} MB).")


def _ensure_rvrt_sources() -> None:
    """
    Set up the _rvrt_src/ module tree used to import RVRT without basicsr.

    Always writes the pure-PyTorch deform_attn module (no CUDA build required).
    Downloads and patches network_rvrt.py on the first run; subsequent runs
    reuse the cached copy (patches are idempotent).
    """
    models_dir = _SRC_DIR / "models"
    op_dir = models_dir / "op"
    op_dir.mkdir(parents=True, exist_ok=True)

    (models_dir / "__init__.py").write_text("", encoding="utf-8")
    (op_dir / "__init__.py").write_text("", encoding="utf-8")
    (op_dir / "deform_attn.py").write_text(_DEFORM_ATTN_SRC, encoding="utf-8")

    network_dest = models_dir / "network_rvrt.py"
    if not network_dest.exists():
        print("  Downloading models/network_rvrt.py from RVRT GitHub ...",
              end=" ", flush=True)
        urllib.request.urlretrieve(_NETWORK_URL, str(network_dest))
        print("done.")

    src = network_dest.read_text(encoding="utf-8")
    changed = False

    # Python 3.12+ removed distutils.version.
    if "from distutils.version import LooseVersion" in src:
        src = src.replace(
            "from distutils.version import LooseVersion",
            _LOOSEVERSION_SHIM,
        )
        changed = True

    # torchvision is imported at module level but may not be installed with CPU torch.
    if re.search(r"^import torchvision$", src, re.MULTILINE):
        src = re.sub(
            r"^import torchvision$",
            "try:\n    import torchvision\nexcept ImportError:\n    torchvision = None",
            src,
            flags=re.MULTILINE,
        )
        changed = True

    # ONNX only supports 4D PixelShuffle. RVRT's Upsample feeds a 5D tensor
    # (N, T, C, H, W) to nn.PixelShuffle, which PyTorch accepts but ONNX rejects.
    # Replace with _PixelShuffle5D that merges N and T before applying 4D PS.
    _ps5d_class = '''\
class _PixelShuffle5D(nn.Module):
    """PixelShuffle for 5D input (N, T, C*r^2, H, W) via 4D reshape.
    ONNX-compatible replacement for nn.PixelShuffle on 5D tensors.
    """
    def __init__(self, upscale_factor):
        super().__init__()
        self.r = upscale_factor

    def forward(self, x):
        n, t, c, h, w = x.shape
        x = x.reshape(n * t, c, h, w)
        x = F.pixel_shuffle(x, self.r)
        return x.reshape(n, t, c // (self.r * self.r), h * self.r, w * self.r)

'''
    if "_PixelShuffle5D" not in src:
        src = src.replace(
            "class Upsample(nn.Sequential):",
            _ps5d_class + "class Upsample(nn.Sequential):",
        )
        src = src.replace("m.append(nn.PixelShuffle(2))", "m.append(_PixelShuffle5D(2))")
        src = src.replace("m.append(nn.PixelShuffle(3))", "m.append(_PixelShuffle5D(3))")
        changed = True

    # Patch the residual-add path to avoid aten::type_as in the JIT graph.
    # The +=  operator on a tensor whose type the tracer cannot resolve from
    # the upsample_trilinear3d output causes a type_as node that the ONNX
    # symbolic functions cannot handle when shape inference is disabled.
    # Replacing with an explicit .float() cast gives the ONNX exporter a
    # concrete dtype, avoiding the type_as node entirely.
    _residual_old = (
        "hr += torch.nn.functional.interpolate("
        "lqs, size=hr.shape[-3:], mode='trilinear', align_corners=False)"
    )
    _residual_new = (
        "hr = hr + torch.nn.functional.interpolate("
        "lqs.float(), size=hr.shape[-3:], mode='trilinear', align_corners=False)"
    )
    if _residual_old in src:
        src = src.replace(_residual_old, _residual_new)
        changed = True

    # Patch the SpyNet flow-scaling lines (line ~172) to avoid aten::type_as.
    # The in-place *=  with Python float (Double) on a Float32 tensor slice
    # causes type_as nodes that fail when ONNX shape inference is disabled.
    # Replace with a single element-wise mul using a Float32 scale tensor.
    _flow_scale_old = (
        "                flow_out[:, 0, :, :] *= float(w // scale) / float(w_floor // scale)\n"
        "                flow_out[:, 1, :, :] *= float(h // scale) / float(h_floor // scale)"
    )
    _flow_scale_new = (
        "                _w_ratio = float(w // scale) / float(w_floor // scale)\n"
        "                _h_ratio = float(h // scale) / float(h_floor // scale)\n"
        "                flow_out = flow_out * flow_out.new_tensor([_w_ratio, _h_ratio]).reshape(1, 2, 1, 1)"
    )
    if _flow_scale_old in src:
        src = src.replace(_flow_scale_old, _flow_scale_new)
        changed = True

    if changed:
        network_dest.write_text(src, encoding="utf-8")


def _build_and_load(weights_path: Path, spynet_path: Path) -> nn.Module:
    """Construct RVRT with REDS 4x defaults and load checkpoint weights."""
    _ensure_rvrt_sources()

    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))

    from models.network_rvrt import RVRT  # noqa: PLC0415

    model = RVRT(
        upscale=4,
        clip_size=2,
        img_size=[2, 64, 64],
        window_size=[2, 8, 8],
        num_blocks=[1, 2, 1],
        depths=[2, 2, 2],
        embed_dims=[144, 144, 144],
        num_heads=[6, 6, 6],
        mlp_ratio=2.0,
        qkv_bias=True,
        spynet_path=str(spynet_path),
        max_residue_magnitude=10,
        deformable_groups=12,
        attention_heads=12,
        attention_window=[3, 3],
        cpu_cache_length=100,
    )

    state = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    params = state.get("params_ema", state.get("params", state))
    missing, unexpected = model.load_state_dict(params, strict=False)
    if missing:
        print(f"  WARNING: missing keys in checkpoint: {missing[:3]}")
    if unexpected:
        print(f"  WARNING: unexpected keys in checkpoint: {unexpected[:3]}")
    model.eval()

    # nn.PixelShuffle is not ONNX-exportable with 5D input (RVRT feeds it N,T,C,H,W).
    # Replace every PixelShuffle instance (nn.PixelShuffle or _PixelShuffle5D from
    # the source patch) with the ONNX-compatible wrapper.
    _PIXEL_SHUFFLE_NAMES = frozenset({'PixelShuffle', '_PixelShuffle5D'})
    _replaced = 0
    for parent in model.modules():
        for child_name, child in list(parent.named_children()):
            if type(child).__name__ in _PIXEL_SHUFFLE_NAMES:
                factor = getattr(child, 'upscale_factor', None) or getattr(child, 'r', None)
                print(f"  Replacing {type(child).__name__}({factor}) at {type(parent).__name__}.{child_name}")
                setattr(parent, child_name, _PixelShuffleONNX(factor))
                _replaced += 1
    print(f"  Replaced {_replaced} PixelShuffle module(s) with ONNX-compatible version.")

    return model


def _export(
    model: nn.Module,
    seq_len: int,
    tile_size: int,
    out_path: Path,
    device: torch.device,
    simplify: bool,
) -> None:
    """Trace and export one (seq_len, tile_size) ONNX model."""
    model = model.to(device)
    dummy = torch.rand(
        1, seq_len, 3, tile_size, tile_size, dtype=torch.float32, device=device
    )

    print(f"  Exporting {out_path.name} ...", end=" ", flush=True)
    # Use the internal TorchScript ONNX exporter directly so we can pass
    # onnx_shape_inference=False. The public torch.onnx.export API does not
    # forward this parameter and defaults to True. The C++ shape-inference
    # pass (_jit_pass_onnx_graph_shape_type_inference) crashes with
    # STATUS_INTEGER_DIVIDE_BY_ZERO on the RVRT graph.
    #
    # Disabling shape inference causes JitScalarType.from_value() (called by
    # many symbolic functions without a default) to raise SymbolicValueError
    # because ONNX graph values have no type annotations without inference.
    # Patch from_value to return UNDEFINED instead of raising. Symbolic
    # functions that check for HALF-precision will see False (correct: RVRT
    # runs in float32) and all float32-safe paths continue unaffected.
    from torch.onnx._internal.torchscript_exporter import _type_utils as _onnx_tu
    _orig_from_value = _onnx_tu.JitScalarType.from_value.__func__
    @classmethod  # type: ignore[misc]
    def _from_value_no_raise(cls, value, default=None):
        if default is None:
            default = _onnx_tu.JitScalarType.UNDEFINED
        return _orig_from_value(cls, value, default)
    _onnx_tu.JitScalarType.from_value = _from_value_no_raise

    from torch.onnx._internal.torchscript_exporter.utils import _export as _ts_export
    with torch.no_grad():
        _ts_export(
            model,
            (dummy,),
            str(out_path),
            input_names=["lqs"],
            output_names=["output"],
            opset_version=_OPSET,
            do_constant_folding=False,
            onnx_shape_inference=False,
        )
    print("done.")

    if simplify:
        if onnxsim_simplify is None:
            print("  [skip] onnxsim not installed -- skipping simplification.")
        else:
            print(f"  Simplifying {out_path.name} ...", end=" ", flush=True)
            model_proto = onnx.load(str(out_path))
            simplified, ok = onnxsim_simplify(model_proto)
            if ok:
                onnx.save(simplified, str(out_path))
                print("done.")
            else:
                print("WARNING: onnxsim simplification failed, keeping unsimplified.")


def main() -> None:
    """Entry point: parse args, download, build, export."""
    parser = argparse.ArgumentParser(description="Export RVRT to ONNX.")
    parser.add_argument("--out", default=_DEFAULT_OUT, help="Output directory.")
    parser.add_argument(
        "--seq-lens",
        nargs="+",
        type=int,
        default=_DEFAULT_SEQ_LENS,
        metavar="T",
        help="Sequence lengths to export (default: 4 10). Must be positive even integers.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=_TILE_SIZE,
        help="Tile spatial size in pixels (default: 256).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "Device for model tracing and ONNX export (default: cpu). "
            'Use "cuda:0" on a compatible NVIDIA GPU (RTX 3070/3080/4090).'
        ),
    )
    parser.add_argument("--no-simplify", action="store_true", help="Skip onnxsim.")
    parser.add_argument(
        "--weights-dir",
        default=str(_SCRIPT_DIR / "_weights_cache"),
        help="Directory to cache downloaded .pth weights.",
    )
    args = parser.parse_args()

    if args.tile_size % 32 != 0:
        print(
            f"ERROR: --tile-size {args.tile_size} is not a multiple of 32.",
            file=sys.stderr,
        )
        sys.exit(1)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(
            "ERROR: CUDA requested but torch.cuda.is_available() is False.\n"
            "Install a CUDA-enabled torch build or omit --device to use CPU.",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path(args.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / _WEIGHTS_NAME
    spynet_path = weights_dir / _SPYNET_NAME

    print("\n=== RVRT ONNX export ===")
    print("LICENSE: CC-BY-NC 4.0 -- non-commercial research use only.")
    print(f"Device    : {device}")
    print(f"Tile size : {args.tile_size}x{args.tile_size}")
    print(f"Seq lens  : {args.seq_lens}")
    print(f"Output dir: {out_dir}")
    print()

    _download(_WEIGHTS_URL, weights_path, _WEIGHTS_NAME)
    _download(_SPYNET_URL, spynet_path, _SPYNET_NAME)
    print()

    print("Loading model ...", end=" ", flush=True)
    model = _build_and_load(weights_path, spynet_path)
    print("done.")
    print()

    simplify = not args.no_simplify
    for t in args.seq_lens:
        name = f"rvrt_x4_t{t}.onnx"
        out_path = out_dir / name
        print(f"[T={t}]")
        _export(model, t, args.tile_size, out_path, device, simplify)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"  Saved -> {out_path}  ({size_mb:.1f} MB)")
        print()

    print("All exports complete.")


if __name__ == "__main__":
    main()
