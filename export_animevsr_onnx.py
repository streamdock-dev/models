"""
export_animevsr_onnx.py -- Export AnimeSR to ONNX for offline video upscaling
in StreamDock's batch processing queue.

Usage:
    py export_animevsr_onnx.py [--out DIR] [--seq-lens 5 10] [--tile-size 256] [--no-simplify]

Requirements:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install onnx onnxsim onnxruntime gdown

The script:
  1. Downloads AnimeSR v1 weights from Google Drive (~6 MB, Apache-2.0).
  2. Implements the MSRSWVSR architecture standalone (no basicsr required).
  3. Exports one ONNX per sequence length T with fixed tile input (default 256x256).
  4. Optionally simplifies with onnxsim.
  5. Runs a CPU inference sanity-check via onnxruntime.

Output files (default: ThirdParty/models/realbasicvsr/):
    animevsr_x4_t5.onnx    5-frame input sequence, 256x256 tiles, 4x upscale
    animevsr_x4_t10.onnx  10-frame input sequence, 256x256 tiles, 4x upscale

NOTE: AnimeSR output files are placed in the realbasicvsr/ subfolder because they
use the "RealBasicVsr" pipeline in model-bundle.json -- the C# RealBasicVsrUpscaler
handles both RealBasicVSR and AnimeSR models identically (same I/O shape).

AnimeSR (TencentARC, NeurIPS 2022) trains on high-quality anime video and uses the
MSRSWVSR (Multi-Scale, unidirectional Recurrent, Sliding Window VSR) architecture.
Unlike RealBasicVSR, it has no optical flow, no deformable convolutions, and no
image-cleaning pre-pass. Propagation is unidirectional (forward only), with a
sliding window that sees the previous, current, and next frame at each step.

Input tensor shape:  (1, T, 3, tile_h, tile_w)  -- T RGB frames, normalised [0, 1], float32
Output tensor shape: (1, T, 3, tile_h*4, tile_w*4) -- T upscaled frames, float32

Architecture notes:
    MSRSWVSR uses a multi-scale recurrent cell (RightAlignMSConvResidualBlocks) that
    operates on pixel_unshuffle-downsampled SR feedback + hidden state + 3-frame input.
    No alignment/optical-flow modules. Python for-loops over T unroll at trace time to
    produce a fully static ONNX graph for a fixed T and tile resolution.
"""

from __future__ import annotations

import argparse
import sys
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

try:
    import gdown
except ImportError:
    gdown = None  # type: ignore[assignment]

# AnimeSR weights on Google Drive (TencentARC, Apache-2.0 licence).
# Google Drive folder: https://drive.google.com/drive/folders/1gwNTbKLUjt5FlgT6PQQnBz5wFzmNUX8g
# AnimeSR v1 (paper model) uses a BasicVSR backbone with 30 propagation blocks,
# which matches the architecture implemented here. v2 uses a different recurrent
# architecture (MSRSWVSR) and is NOT compatible with this script.
# If the file ID below stops working, download AnimeSR_v1-PaperModel.pth manually
# from the link above and place it in _weights_cache/.
_WEIGHTS_GDRIVE_ID = "1wpBXS5PIKDAC8IvMuCGNnrrA_fMTzF9P"
_WEIGHTS_GDRIVE_FOLDER = "https://drive.google.com/drive/folders/1gwNTbKLUjt5FlgT6PQQnBz5wFzmNUX8g"
_WEIGHTS_NAME = "AnimeSR_v1-PaperModel.pth"

# Architecture hyper-parameters matching the AnimeSR v1 pre-trained weights.
# num_feat=64, num_block=(5, 3, 2), netscale=4 -- matches checkpoint tensor shapes.
_NUM_FEAT = 64
_NUM_BLOCK = (5, 3, 2)
_NETSCALE = 4

# Export defaults.
_TILE_SIZE = 256
_DEFAULT_SEQ_LENS = [5, 10]
_OPSET = 18


# ---------------------------------------------------------------------------
# Standalone architecture (MSRSWVSR -- matches checkpoint key structure)
# ---------------------------------------------------------------------------

class _ResBlockNoBN(nn.Module):
    """Residual block without batch normalisation."""

    def __init__(self, num_feat: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.relu(self.conv1(x)))


class _RightAlignMSConvResidualBlocks(nn.Module):
    """Multi-scale residual cell used by MSRSWVSR (maps to RightAlignMSConvResidualBlocks).

    Processes the input at three spatial scales (s1=full, s2=1/2, s4=1/4) through
    residual blocks that are progressively merged right-to-left (the coarser scales
    start processing later in the sequence, hence "right-align").
    """

    def __init__(
        self,
        num_in_ch: int = 121,
        num_state_ch: int = 64,
        num_out_ch: int = 112,
        num_block: tuple = (5, 3, 2),
    ) -> None:
        super().__init__()
        assert len(num_block) == 3
        assert num_block[0] >= num_block[1] >= num_block[2]
        self.num_block = num_block

        self.conv_s1_first = nn.Sequential(
            nn.Conv2d(num_in_ch, num_state_ch, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.conv_s2_first = nn.Sequential(
            nn.Conv2d(num_state_ch, num_state_ch, 3, 2, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.conv_s4_first = nn.Sequential(
            nn.Conv2d(num_state_ch, num_state_ch, 3, 2, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.body_s1_first = nn.ModuleList(
            [_ResBlockNoBN(num_state_ch) for _ in range(num_block[0])]
        )
        self.body_s2_first = nn.ModuleList(
            [_ResBlockNoBN(num_state_ch) for _ in range(num_block[1])]
        )
        self.body_s4_first = nn.ModuleList(
            [_ResBlockNoBN(num_state_ch) for _ in range(num_block[2])]
        )

        self.upsample_x2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.upsample_x4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)

        self.fusion = nn.Sequential(
            nn.Conv2d(3 * num_state_ch, 2 * num_out_ch, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(2 * num_out_ch, num_out_ch, 3, 1, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_s1 = self.conv_s1_first(x)
        x_s2 = self.conv_s2_first(x_s1)
        x_s4 = self.conv_s4_first(x_s2)

        flag_s2 = False
        flag_s4 = False
        for i in range(self.num_block[0]):
            x_s1 = self.body_s1_first[i](
                x_s1
                + (self.upsample_x2(x_s2) if flag_s2 else 0)
                + (self.upsample_x4(x_s4) if flag_s4 else 0)
            )
            if i >= self.num_block[0] - self.num_block[1]:
                j2 = i - (self.num_block[0] - self.num_block[1])
                x_s2 = self.body_s2_first[j2](
                    x_s2 + (self.upsample_x2(x_s4) if flag_s4 else 0)
                )
                flag_s2 = True
            if i >= self.num_block[0] - self.num_block[2]:
                j4 = i - (self.num_block[0] - self.num_block[2])
                x_s4 = self.body_s4_first[j4](x_s4)
                flag_s4 = True

        x_fusion = self.fusion(
            torch.cat(
                (x_s1, self.upsample_x2(x_s2), self.upsample_x4(x_s4)),
                dim=1,
            )
        )
        return x_fusion


class _MSRSWVSR(nn.Module):
    """MSRSWVSR: Multi-Scale, unidirectional Recurrent, Sliding Window VSR (4x upscale).

    Each frame is processed using the previous SR output and hidden state as feedback,
    with a 3-frame sliding window (previous, current, next) as input. No optical flow.
    """

    def __init__(
        self,
        num_feat: int = 64,
        num_block: tuple = (5, 3, 2),
        netscale: int = 4,
    ) -> None:
        super().__init__()
        self.num_feat = num_feat
        self.netscale = netscale

        # Input channels: 3*3 frames + 3*netscale^2 SR feedback (pixel_unshuffle) + num_feat state
        num_in_ch = 3 * 3 + 3 * netscale * netscale + num_feat
        num_out_ch = num_feat + 3 * netscale * netscale

        self.recurrent_cell = _RightAlignMSConvResidualBlocks(
            num_in_ch=num_in_ch,
            num_state_ch=num_feat,
            num_out_ch=num_out_ch,
            num_block=num_block,
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.pixel_shuffle = nn.PixelShuffle(netscale)

    def _cell(
        self,
        x: torch.Tensor,
        fb: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple:
        """Process one step: x=(prev,cur,next) frames, fb=previous SR frame, state=hidden."""
        res = x[:, 3:6]  # current LR frame (RGB)
        inp = torch.cat(
            (x, F.pixel_unshuffle(fb, self.netscale), state),
            dim=1,
        )
        out = self.recurrent_cell(inp)
        sr_ch = 3 * self.netscale * self.netscale
        out_img = self.pixel_shuffle(out[:, :sr_ch]) + F.interpolate(
            res, scale_factor=float(self.netscale), mode="bilinear", align_corners=False
        )
        out_state = self.lrelu(out[:, sr_ch:])
        return out_img, out_state

    def forward(self, lqs: torch.Tensor) -> torch.Tensor:
        """Process T low-resolution frames and return T 4x super-resolved frames.

        Args:
            lqs: (n, t, 3, h, w) float32 tensor normalised to [0, 1].

        Returns:
            (n, t, 3, h*4, w*4) float32 tensor.
        """
        n, t, c, h, w = lqs.size()
        out = lqs.new_zeros(n, c, h * self.netscale, w * self.netscale)
        state = lqs.new_zeros(n, self.num_feat, h, w)
        outputs: list[torch.Tensor] = []

        for i in range(t):
            if i == 0:
                x_in = torch.cat((lqs[:, i], lqs[:, i], lqs[:, i + 1]), dim=1)
            elif i == t - 1:
                x_in = torch.cat((lqs[:, i - 1], lqs[:, i], lqs[:, i]), dim=1)
            else:
                x_in = torch.cat((lqs[:, i - 1], lqs[:, i], lqs[:, i + 1]), dim=1)
            out, state = self._cell(x_in, out, state)
            outputs.append(out)

        return torch.stack(outputs, dim=1)


# ---------------------------------------------------------------------------
# Weights loading
# ---------------------------------------------------------------------------

def _download(dest: Path, label: str) -> None:
    """Download AnimeSR weights via gdown (Google Drive), skipping if already present."""
    if dest.exists():
        print(f"  [skip] {label} already downloaded.")
        return

    if gdown is None:
        print(
            f"ERROR: {label} not found and gdown is not installed.\n"
            f"  Install gdown and retry:  pip install gdown\n"
            f"  Or download {label} manually from:\n"
            f"    {_WEIGHTS_GDRIVE_FOLDER}\n"
            f"  and place it in {dest.parent}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  Downloading {label} from Google Drive via gdown ...")
    try:
        gdown.download(id=_WEIGHTS_GDRIVE_ID, output=str(dest), quiet=False, use_cookies=False)
    except Exception as exc:
        print(f"  gdown download failed: {exc}", file=sys.stderr)
        print(
            f"  Please download {label} manually from:\n"
            f"    {_WEIGHTS_GDRIVE_FOLDER}\n"
            f"  and place it in {dest.parent}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not dest.exists():
        print(
            f"  ERROR: {label} not found after gdown download.\n"
            f"  Expected: {dest}\n"
            f"  Please download manually from {_WEIGHTS_GDRIVE_FOLDER}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"  Downloaded {label} ({dest.stat().st_size // 1024} KB)")


def _build_and_load(weights_path: Path) -> _MSRSWVSR:
    """Instantiate _MSRSWVSR and load AnimeSR weights."""
    model = _MSRSWVSR(num_feat=_NUM_FEAT, num_block=_NUM_BLOCK, netscale=_NETSCALE)

    raw: dict = torch.load(weights_path, map_location="cpu", weights_only=False)

    # AnimeSR checkpoint is a flat state dict (no params_ema / params wrapper).
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    state = raw.get("params_ema", raw.get("params", raw))

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  WARNING: missing keys: {missing[:3]}")
    if unexpected:
        print(f"  WARNING: unexpected keys: {unexpected[:3]}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _export(
    model: _MSRSWVSR,
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
            model_proto = onnx.load(str(out_path))
            simplified, ok = onnxsim_simplify(model_proto)
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
            expected_h = tile_size * 4
            expected_w = tile_size * 4
            got = result[0].shape
            status = "PASS" if got == (1, seq_len, 3, expected_h, expected_w) else f"SHAPE MISMATCH {got}"
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
        description="Export AnimeSR to ONNX for offline video upscaling."
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "realbasicvsr"),
        help="Output directory (default: realbasicvsr/ next to this script).",
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
        help=f"Spatial tile size (must be a multiple of 32). Default: {_TILE_SIZE}.",
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

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = Path(args.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / _WEIGHTS_NAME

    print(f"\n=== AnimeSR ONNX export ===")
    print(f"Tile size : {tile_size}x{tile_size}")
    print(f"Seq lens  : {args.seq_lens}")
    print(f"Output dir: {out_dir}")
    print(f"(Files saved to realbasicvsr/ -- uses 'RealBasicVsr' pipeline in StreamDock)")
    print()

    _download(weights_path, _WEIGHTS_NAME)
    print()

    print("Loading weights ...", end=" ", flush=True)
    model = _build_and_load(weights_path)
    print("done.")
    print()

    simplify = not args.no_simplify
    for t in args.seq_lens:
        name = f"animevsr_x4_t{t}.onnx"
        out_path = out_dir / name
        print(f"[T={t}]")
        _export(model, t, tile_size, out_path, simplify)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"  Saved -> {out_path}  ({size_mb:.1f} MB)")
        print()

    print("All exports complete.")


if __name__ == "__main__":
    main()
