"""
export_realbasicvsr_onnx.py -- Export RealBasicVSR to ONNX for offline video upscaling
in StreamDock's batch processing queue.

Usage:
    py export_realbasicvsr_onnx.py [--out DIR] [--seq-lens 5 10] [--tile-size 256] [--no-simplify]

Requirements:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install onnx onnxsim onnxruntime

The script:
  1. Downloads RealBasicVSR.pth from the official Dropbox release (~56 MB, Apache-2.0).
  2. Implements the RealBasicVSR + BasicVSR + SPyNet architecture standalone
     (no mmcv or mmagic required).
  3. Exports one ONNX per sequence length T with fixed tile input (default 256x256).
  4. Optionally simplifies with onnxsim.

Output files (default: ThirdParty/models/realbasicvsr/):
    realbasicvsr_x4_t5.onnx    5-frame input sequence, 256x256 tiles, 4x upscale
    realbasicvsr_x4_t10.onnx  10-frame input sequence, 256x256 tiles, 4x upscale

Input tensor shape:  (1, T, 3, tile_h, tile_w)  -- T RGB frames, normalised [0, 1], float32
Output tensor shape: (1, T, 3, tile_h*4, tile_w*4) -- T upscaled frames, float32

Offline inference in C# (RealBasicVsrUpscaler, StreamDock.Core):
    - Split each video frame into 256x256 spatial tiles with 50% overlap.
    - Batch T tiles across time (same tile position across T consecutive frames).
    - Run the model, producing 1024x1024 tiles.
    - Blend adjacent tiles with a Hann window and stitch into the full upscaled frame.
    - Advance by T/2 frames (50% temporal overlap) to reduce clip-boundary artefacts.

Architecture notes:
    BasicVSR uses SPyNet optical flow + bilinear warp (grid_sample, ONNX opset 16+) +
    residual blocks. No deformable convolutions. Python for-loops unroll to a static
    graph at trace time, producing a large but fully deterministic ONNX graph.
    Tile size must be a multiple of 32 so SPyNet's 32-pad step is a no-op and gets
    eliminated by constant-folding.
"""

from __future__ import annotations

import argparse
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

# Official weights -- Apache-2.0 licence, Kelvin C.K. Chan et al., CVPR 2022.
# Dropbox direct-download URL (?dl=1 serves the raw file without the preview page).
_WEIGHTS_URL = "https://www.dropbox.com/s/eufigxmmkv5woop/RealBasicVSR.pth?dl=1"
_WEIGHTS_NAME = "RealBasicVSR.pth"

# Architecture hyper-parameters matching the pre-trained weights.
_MID_CH = 64
_N_PROP = 20   # num_propagation_blocks per BasicVSR direction
_N_CLEAN = 20  # num_cleaning_blocks in the image-cleaning head

# Export defaults.
_TILE_SIZE = 256          # pixels -- must be a multiple of 32 for SPyNet padding no-op
_DEFAULT_SEQ_LENS = [5, 10]
_OPSET = 18


# ---------------------------------------------------------------------------
# Standalone architecture (matches the mmagic / MMEditing v1 key structure)
# ---------------------------------------------------------------------------

class _ConvAct(nn.Module):
    """Conv2d wrapper that stores the conv under the attribute name 'conv'.

    This mirrors the ConvModule layout used in mmcv, which is what the
    SPyNet checkpoint keys expect (e.g. basic_module.0.conv.weight).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int = 1,
        padding: int = 0,
        act: bool = True,
    ) -> None:
        """Build a Conv2d with an optional in-place ReLU activation."""
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=True)
        self._act = nn.ReLU(inplace=True) if act else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply conv then optional ReLU."""
        x = self.conv(x)
        if self._act is not None:
            x = self._act(x)
        return x


class _ResBlockNoBN(nn.Module):
    """Residual block without batch normalisation (BasicVSR building block)."""

    def __init__(self, mid_channels: int = 64) -> None:
        """Build two 3x3 convolutions with identity shortcut."""
        super().__init__()
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual forward: x + conv2(relu(conv1(x)))."""
        return x + self.conv2(self.relu(self.conv1(x)))


class _ResBlocksWithInputConv(nn.Module):
    """Input-projection conv + N residual blocks (maps to ResidualBlocksWithInputConv)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 64,
        num_blocks: int = 20,
    ) -> None:
        """Build projection conv, LeakyReLU, then a Sequential of N residual blocks."""
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),  # index 0
            nn.LeakyReLU(negative_slope=0.1, inplace=True),             # index 1 (no params)
            nn.Sequential(                                               # index 2
                *[_ResBlockNoBN(mid_channels=out_channels) for _ in range(num_blocks)]
            ),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Apply the projection + residual blocks."""
        return self.main(feat)


class _PixelShufflePack(nn.Module):
    """Sub-pixel upscaling: conv -> pixel_shuffle (maps to PixelShufflePack)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int,
        upsample_kernel: int,
    ) -> None:
        """Build the upscaling conv, padding to keep spatial size before shuffle."""
        super().__init__()
        self.upsample_conv = nn.Conv2d(
            in_channels,
            out_channels * scale_factor * scale_factor,
            upsample_kernel,
            padding=(upsample_kernel - 1) // 2,
        )
        self._scale = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Conv then pixel-shuffle to increase spatial resolution."""
        return F.pixel_shuffle(self.upsample_conv(x), self._scale)


def _flow_warp(
    x: torch.Tensor,
    flow: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    """Warp feature map x with optical flow.

    Args:
        x:    (n, c, h, w) feature map.
        flow: (n, h, w, 2) optical flow in pixel units (not normalised).
        mode: interpolation mode for F.grid_sample.
        padding_mode: padding mode for F.grid_sample.
        align_corners: passed to F.grid_sample.

    Returns:
        Warped tensor with the same shape as x.
    """
    _, _, h, w = x.size()
    # Create absolute coordinate grid.
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, device=flow.device, dtype=x.dtype),
        torch.arange(0, w, device=flow.device, dtype=x.dtype),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), dim=2)  # (h, w, 2)
    grid_flow = grid + flow                        # (n, h, w, 2) via broadcasting

    # Normalise to [-1, 1] for F.grid_sample.
    grid_flow_x = 2.0 * grid_flow[..., 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[..., 1] / max(h - 1, 1) - 1.0
    grid_norm = torch.stack((grid_flow_x, grid_flow_y), dim=3)

    return F.grid_sample(
        x,
        grid_norm,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


class _SPyNetBasicModule(nn.Module):
    """One level of the SPyNet pyramid (5 ConvModules, channels 8->32->64->32->16->2)."""

    def __init__(self) -> None:
        """Build the 5-layer conv stack using _ConvAct wrappers for key compatibility."""
        super().__init__()
        self.basic_module = nn.Sequential(
            _ConvAct(8, 32, 7, 1, 3, act=True),   # 0 -> keys: basic_module.0.conv.*
            _ConvAct(32, 64, 7, 1, 3, act=True),  # 1
            _ConvAct(64, 32, 7, 1, 3, act=True),  # 2
            _ConvAct(32, 16, 7, 1, 3, act=True),  # 3
            _ConvAct(16, 2, 7, 1, 3, act=False),  # 4 (no activation on output)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the 5-conv stack and return the 2-channel flow delta."""
        return self.basic_module(x)


class _SPyNet(nn.Module):
    """SPyNet optical flow estimator (6-level spatial pyramid)."""

    def __init__(self) -> None:
        """Build the 6-level pyramid and register normalisation buffers."""
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
        """Compute coarse-to-fine flow from ref to supp (both already 32-padded)."""
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
        """Estimate optical flow from ref to supp.

        Pads input to a multiple of 32 (a no-op for tile sizes already 32-aligned),
        computes the pyramid flow, then rescales back to the original resolution.
        """
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


class _BasicVSRNet(nn.Module):
    """BasicVSR network (bidirectional recurrent super-resolution, 4x scale).

    Processes an entire sequence of T frames in one forward call.
    The Python for-loops over T unroll into the ONNX graph at trace time,
    producing a fully static graph for a fixed T and tile resolution.
    """

    def __init__(self, mid_channels: int = 64, num_blocks: int = 20) -> None:
        """Build SPyNet, backward/forward propagation branches, and the upsampling head."""
        super().__init__()
        self.mid_channels = mid_channels

        self.spynet = _SPyNet()

        self.backward_resblocks = _ResBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)
        self.forward_resblocks = _ResBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)

        self.fusion = nn.Conv2d(mid_channels * 2, mid_channels, 1, 1, 0, bias=True)
        self.upsample1 = _PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = _PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def _compute_flow(
        self, lrs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute forward and backward optical flows for all adjacent frame pairs."""
        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1].reshape(-1, c, h, w)  # frames 0 .. t-2
        lrs_2 = lrs[:, 1:].reshape(-1, c, h, w)   # frames 1 .. t-1

        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)
        flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)
        return flows_forward, flows_backward

    def forward(self, lrs: torch.Tensor) -> torch.Tensor:
        """Process T low-resolution frames and return T 4x super-resolved frames.

        Args:
            lrs: (n, t, 3, h, w) float32 tensor normalised to [0, 1].

        Returns:
            (n, t, 3, 4h, 4w) float32 tensor.
        """
        n, t, c, h, w = lrs.size()
        flows_forward, flows_backward = self._compute_flow(lrs)

        # Backward-time propagation.
        back_feats: list[torch.Tensor] = []
        feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:
                feat_prop = _flow_warp(feat_prop, flows_backward[:, i].permute(0, 2, 3, 1))
            feat_prop = self.backward_resblocks(torch.cat([lrs[:, i], feat_prop], dim=1))
            back_feats.append(feat_prop)
        back_feats = back_feats[::-1]  # reverse so back_feats[i] corresponds to frame i

        # Forward-time propagation and upsampling.
        outputs: list[torch.Tensor] = []
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(t):
            if i > 0:
                feat_prop = _flow_warp(feat_prop, flows_forward[:, i - 1].permute(0, 2, 3, 1))
            feat_prop = self.forward_resblocks(torch.cat([lrs[:, i], feat_prop], dim=1))

            out = self.lrelu(self.fusion(torch.cat([back_feats[i], feat_prop], dim=1)))
            out = self.lrelu(self.upsample1(out))
            out = self.lrelu(self.upsample2(out))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out) + self.img_upsample(lrs[:, i])
            outputs.append(out)

        return torch.stack(outputs, dim=1)


class _RealBasicVsrExportWrapper(nn.Module):
    """Export wrapper that applies exactly one cleaning pass then BasicVSR.

    The original RealBasicVSRNet uses a dynamic loop (up to 3 cleaning passes,
    stopping early when residue magnitude drops below a threshold). That conditional
    early exit does not trace to a deterministic ONNX graph. This wrapper hard-codes
    a single cleaning pass, which is the usual outcome for real video inputs.
    """

    def __init__(
        self,
        image_cleaning: nn.Module,
        basicvsr: _BasicVSRNet,
    ) -> None:
        """Store the two sub-networks."""
        super().__init__()
        self.image_cleaning = image_cleaning
        self.basicvsr = basicvsr

    def forward(self, lqs: torch.Tensor) -> torch.Tensor:
        """One cleaning pass + BasicVSR 4x super-resolution.

        Args:
            lqs: (1, T, 3, H, W) float32, values in [0, 1].

        Returns:
            (1, T, 3, H*4, W*4) float32.
        """
        n, t, c, h, w = lqs.size()

        # One cleaning pass -- non-mutating, compatible with ONNX tracing.
        cleaned: list[torch.Tensor] = []
        for i in range(t):
            frame = lqs[:, i]
            residue = self.image_cleaning(frame)
            cleaned.append(frame + residue)
        lqs_cleaned = torch.stack(cleaned, dim=1)

        return self.basicvsr(lqs_cleaned)


# ---------------------------------------------------------------------------
# Weights loading
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path, label: str) -> None:
    """Download url to dest with a progress display, skipping if it already exists."""
    if dest.exists():
        print(f"  [skip] {label} already downloaded.")
        return
    print(f"  Downloading {label} ...", end=" ", flush=True)

    def _hook(count: int, block: int, total: int) -> None:
        if total > 0:
            pct = min(100, count * block * 100 // total)
            print(f"\r  Downloading {label} ... {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, _hook)
    print(f"\r  Downloaded  {label} ({dest.stat().st_size // 1024} KB)")


def _build_and_load(weights_path: Path) -> _RealBasicVsrExportWrapper:
    """Instantiate the architecture, load pre-trained weights, return the export wrapper."""
    image_cleaning = nn.Sequential(
        _ResBlocksWithInputConv(3, _MID_CH, _N_CLEAN),
        nn.Conv2d(_MID_CH, 3, 3, 1, 1, bias=True),
    )
    basicvsr = _BasicVSRNet(mid_channels=_MID_CH, num_blocks=_N_PROP)

    raw: dict = torch.load(weights_path, map_location="cpu", weights_only=False)

    # The checkpoint may be the full GAN state dict (with generator.* keys) or just
    # the generator. Prefer generator_ema (EMA weights give slightly better quality).
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]

    def _strip_prefix(d: dict, prefix: str) -> dict:
        return {k[len(prefix):]: v for k, v in d.items() if k.startswith(prefix)}

    if any(k.startswith("generator_ema.") for k in raw):
        state = _strip_prefix(raw, "generator_ema.")
    elif any(k.startswith("generator.") for k in raw):
        state = _strip_prefix(raw, "generator.")
    else:
        state = raw  # assume keys are already flat (generator-only checkpoint)

    # Split into the two sub-module state dicts.
    cleaning_state = _strip_prefix(state, "image_cleaning.")
    basicvsr_state = _strip_prefix(state, "basicvsr.")

    missing_c, unexpected_c = image_cleaning.load_state_dict(cleaning_state, strict=False)
    missing_b, unexpected_b = basicvsr.load_state_dict(basicvsr_state, strict=False)

    if missing_c or missing_b:
        print(
            f"  WARNING: missing keys -- cleaning: {missing_c[:3]}, basicvsr: {missing_b[:3]}"
        )
    if unexpected_c or unexpected_b:
        print(
            f"  WARNING: unexpected keys -- cleaning: {unexpected_c[:3]}, basicvsr: {unexpected_b[:3]}"
        )

    image_cleaning.eval()
    basicvsr.eval()
    return _RealBasicVsrExportWrapper(image_cleaning, basicvsr)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _export(
    wrapper: _RealBasicVsrExportWrapper,
    seq_len: int,
    tile_size: int,
    out_path: Path,
    simplify: bool,
) -> None:
    """Trace and export one (seq_len, tile_size) ONNX model."""
    # Use random (not zeros) so that the mirror-extended detection inside BasicVSR
    # evaluates as False and traces the standard forward path.
    dummy = torch.rand(1, seq_len, 3, tile_size, tile_size, dtype=torch.float32)

    wrapper.eval()
    print(f"  Exporting {out_path.name} ...", end=" ", flush=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
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
            model = onnx.load(str(out_path))
            simplified, ok = onnxsim_simplify(model)
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
        description="Export RealBasicVSR to ONNX for offline video upscaling."
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

    print(f"\n=== RealBasicVSR ONNX export ===")
    print(f"Tile size : {tile_size}x{tile_size}")
    print(f"Seq lens  : {args.seq_lens}")
    print(f"Output dir: {out_dir}")
    print()

    # Step 1: Download weights.
    _download(_WEIGHTS_URL, weights_path, _WEIGHTS_NAME)
    print()

    # Step 2: Build and load.
    print("Loading weights ...", end=" ", flush=True)
    wrapper = _build_and_load(weights_path)
    print("done.")
    print()

    # Step 3: Export one ONNX per seq-len.
    simplify = not args.no_simplify
    for t in args.seq_lens:
        name = f"realbasicvsr_x4_t{t}.onnx"
        out_path = out_dir / name
        print(f"[T={t}]")
        _export(wrapper, t, tile_size, out_path, simplify)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"  Saved -> {out_path}  ({size_mb:.1f} MB)")
        print()

    print("All exports complete.")


if __name__ == "__main__":
    main()
