"""
export_nafnet_deblur_onnx.py -- Export NAFNet blind-deblur to ONNX.

Usage:
    python export_nafnet_deblur_onnx.py [--out DIR] [--no-simplify] [--keep-tmp]

Requirements:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install onnx onnxsim

    The pre-exported NAFNet ONNX is downloaded directly from HuggingFace
    (opencv/deblurring_nafnet). Spatial axes are made dynamic if they are not already,
    then the model is optionally simplified with onnxsim and saved to the output dir.
    No PyTorch or CUDA is required.

The script will:
  1. Download the NAFNet-GoPro-width32 ONNX from HuggingFace (opencv/deblurring_nafnet)
     to ~/.cache/nafnet/deblurring_nafnet_2025may.onnx, or use the cached copy.
  2. Ensure H and W axes are declared dynamic so OnnxRuntime can run at any resolution.
  3. Run ONNX shape inference.
  4. Optionally simplify with onnxsim.
  5. Save as nafnet_deblur.onnx in the output directory.

ONNX tensor interface:
    Input  "input"   [1, 3, H, W]  float32  blurred frame, normalised [0, 1]
    Output "output"  [1, 3, H, W]  float32  deblurred frame, normalised [0, 1]

Dynamic axes on H and W are declared so the model accepts any frame resolution
without re-export. The C# runner (OnnxRestorationFilterBase) pads the frame to a
multiple of alignStride (32) before inference, which satisfies NAFNet's internal
requirement that spatial dimensions be multiples of 16 (padder_size = 2^4 encoder
levels). The C# runner crops back to the original frame size after inference.

Output file (default):
    ThirdParty/deblur/nafnet_deblur.onnx

License:
    NAFNet is released under the MIT licence.
    See https://github.com/megvii-research/NAFNet
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.onnx
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

try:
    import onnx
    from onnxsim import simplify as onnxsim_simplify
except ImportError:
    onnx = None  # type: ignore[assignment]
    onnxsim_simplify = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "ThirdParty" / "deblur"
# Pre-exported NAFNet ONNX from opencv/deblurring_nafnet on HuggingFace
HUGGINGFACE_ONNX_URL = (
    "https://huggingface.co/opencv/deblurring_nafnet/resolve/main/"
    "deblurring_nafnet_2025may.onnx"
)
ONNX_CACHE = Path.home() / ".cache" / "nafnet" / "deblurring_nafnet_2025may.onnx"

# Export spatial size: a representative 16:9 HD resolution.
# Must be a multiple of padder_size (16) to pass the sanity check.
EXPORT_H = 480
EXPORT_W = 720

# NAFNet-GoPro-width32 architecture configuration
_NAFNET_CONFIG: dict[str, object] = dict(
    img_channel=3,
    width=32,
    middle_blk_num=1,
    enc_blk_nums=[1, 1, 1, 28],
    dec_blk_nums=[1, 1, 1, 1],
)


class LayerNorm2d(nn.Module):
    """Per-channel layer normalisation for 2D feature maps (NCHW layout).

    Equivalent to the NAFNet LayerNorm2d but implemented as a plain nn.Module
    (without a custom autograd.Function) so it traces cleanly through torch.onnx.export.

    Parameters
    ----------
    channels:
        Number of feature channels to normalise over.
    eps:
        Small constant added to the variance for numerical stability.
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        """Initialise learnable scale and bias parameters."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise x per channel across the H and W spatial dimensions.

        Parameters
        ----------
        x:
            Input feature map, shape [N, C, H, W].

        Returns
        -------
        torch.Tensor
            Normalised feature map, same shape as x.
        """
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + self.eps).sqrt()
        return self.weight.view(1, -1, 1, 1) * y + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    """Split the channel dimension in half and return the elementwise product.

    Replaces the standard nonlinear activation in NAFNet. Given x of shape
    [N, 2C, H, W], splits into two halves along the channel axis and returns
    x1 * x2 of shape [N, C, H, W].
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated activation: split channels in half, return x1 * x2.

        Parameters
        ----------
        x:
            Input tensor, shape [N, 2C, H, W].

        Returns
        -------
        torch.Tensor
            Gated output, shape [N, C, H, W].
        """
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """NAFNet basic building block with depthwise convolution and channel attention.

    Implements the core residual block from the NAFNet paper:
    depthwise separable convolution with SimpleGate activation and simplified
    channel attention, followed by a feed-forward branch with another SimpleGate.

    Parameters
    ----------
    c:
        Number of feature channels.
    dw_expand:
        Channel expansion factor for the depthwise separable convolution branch.
    ffn_expand:
        Channel expansion factor for the feed-forward (FFN) branch.
    """

    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2) -> None:
        """Build the NAFBlock layers with the given channel width."""
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(
            dw_channel, dw_channel, 3, padding=1, groups=dw_channel, bias=True
        )
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, padding=0, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        ffn_channel = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, padding=0, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, padding=0, bias=True)

        self.gate = SimpleGate()

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, bias=True),
        )

        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Apply the NAFBlock residual transformation.

        Parameters
        ----------
        inp:
            Input feature map, shape [N, C, H, W].

        Returns
        -------
        torch.Tensor
            Output feature map, same shape as inp.
        """
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gate(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.gate(x)
        x = self.conv5(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    """NAFNet U-Net image restoration model (encoder-decoder with skip connections).

    Implements the architecture from "Simple Baselines for Image Restoration"
    (Chen et al., 2022). The model takes a degraded image and returns a restored
    image via a global residual skip connection: output = encoder_decoder(inp) + inp.

    The forward pass requires that H and W are multiples of padder_size
    (16 for the 4-level encoder configuration). Padding to satisfy this constraint
    is handled externally -- the C# OnnxRestorationFilterBase pads the input frame
    to a multiple of alignStride before calling the ONNX session.

    Parameters
    ----------
    img_channel:
        Number of image channels (3 for RGB).
    width:
        Base feature channel width.
    middle_blk_num:
        Number of NAFBlocks in the bottleneck.
    enc_blk_nums:
        Number of NAFBlocks per encoder stage (length determines the number of scales).
    dec_blk_nums:
        Number of NAFBlocks per decoder stage (must match len(enc_blk_nums)).
    """

    def __init__(
        self,
        img_channel: int = 3,
        width: int = 16,
        middle_blk_num: int = 1,
        enc_blk_nums: list[int] | None = None,
        dec_blk_nums: list[int] | None = None,
    ) -> None:
        """Build the NAFNet encoder-decoder architecture."""
        super().__init__()
        if enc_blk_nums is None:
            enc_blk_nums = []
        if dec_blk_nums is None:
            dec_blk_nums = []

        self.intro = nn.Conv2d(img_channel, width, 3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(enc_blk_nums)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Run the NAFNet forward pass.

        H and W must be exact multiples of padder_size (16) when this is called.
        No internal padding is performed; the caller is responsible for satisfying
        the alignment requirement.

        Parameters
        ----------
        inp:
            Shape [1, 3, H, W], float32, normalised to [0, 1].

        Returns
        -------
        torch.Tensor
            Shape [1, 3, H, W], float32, restored frame.
        """
        x = self.intro(inp)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, reversed(encs)):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        return x + inp


def download_onnx(path: Path) -> None:
    """Download the pre-exported NAFNet ONNX from HuggingFace if not already cached.

    Parameters
    ----------
    path:
        Local file path to save the downloaded ONNX to.
    """
    if path.exists():
        print(f"Using cached ONNX: {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading NAFNet ONNX from HuggingFace...\n  {HUGGINGFACE_ONNX_URL}")
    try:
        req = urllib.request.Request(
            HUGGINGFACE_ONNX_URL,
            headers={"User-Agent": "python-urllib/3"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp, open(path, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"Saved to {path} ({size_mb:.1f} MB)")
    except Exception as exc:
        if path.exists():
            path.unlink()
        print(
            f"ERROR: Failed to download ONNX: {exc}\n"
            "Download manually from:\n"
            f"  {HUGGINGFACE_ONNX_URL}\n"
            f"and place at:\n  {path}",
            file=sys.stderr,
        )
        sys.exit(1)


def ensure_dynamic_axes(model_proto: object) -> object:
    """Make H and W axes dynamic on all graph inputs and outputs if they are static.

    Parameters
    ----------
    model_proto:
        An onnx.ModelProto to modify in-place.

    Returns
    -------
    object
        The same ModelProto with dynamic H/W axes applied.
    """
    for io in list(model_proto.graph.input) + list(model_proto.graph.output):  # type: ignore[attr-defined]
        t = io.type.tensor_type
        if t.HasField("shape") and len(t.shape.dim) == 4:
            for dim_idx, dim_name in [(2, "height"), (3, "width")]:
                dim = t.shape.dim[dim_idx]
                if dim.HasField("dim_value"):
                    dim.ClearField("dim_value")
                    dim.dim_param = dim_name
    return model_proto


def prepare_onnx(out_dir: Path, simplify: bool) -> None:
    """Download, post-process and save the NAFNet ONNX as nafnet_deblur.onnx.

    Downloads the pre-exported ONNX from HuggingFace, ensures spatial axes are
    dynamic, runs shape inference, optionally simplifies, and writes the output.

    Parameters
    ----------
    out_dir:
        Directory to write nafnet_deblur.onnx into.
    simplify:
        Whether to run onnxsim simplification after post-processing.
    """
    if onnx is None:
        print(
            "ERROR: onnx package is not installed. Install it with:\n"
            "  pip install onnx onnxsim",
            file=sys.stderr,
        )
        sys.exit(1)

    download_onnx(ONNX_CACHE)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "nafnet_deblur.onnx"

    print(f"Loading ONNX from cache: {ONNX_CACHE}")
    model_proto = onnx.load(str(ONNX_CACHE))

    print("Ensuring H and W axes are dynamic...")
    model_proto = ensure_dynamic_axes(model_proto)

    print("Running ONNX shape inference...")
    model_proto = onnx.shape_inference.infer_shapes(model_proto)

    if simplify and onnxsim_simplify is not None:
        print("Simplifying with onnxsim...")
        simplified, check_ok = onnxsim_simplify(model_proto)
        if check_ok:
            model_proto = simplified
            print("Simplification succeeded.")
        else:
            print("WARNING: onnxsim check failed -- keeping un-simplified model.")
    elif simplify:
        print("onnxsim not installed -- skipping simplification.")

    onnx.save(model_proto, str(out_path))
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Saved: {out_path}  ({size_mb:.1f} MB)")


def main() -> None:
    """Parse CLI arguments and export NAFNet blind-deblur to ONNX."""
    parser = argparse.ArgumentParser(
        description=(
            "Export NAFNet blind-deblur (GoPro-width32) to ONNX "
            "for use with OnnxDeblurFilter in StreamDock."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip the onnxsim graph simplification pass.",
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Keep the downloaded ONNX cache after export.",
    )
    args = parser.parse_args()

    prepare_onnx(out_dir=args.out, simplify=not args.no_simplify)

    if not args.keep_tmp and ONNX_CACHE.exists():
        print(f"Removing cached ONNX: {ONNX_CACHE}")
        ONNX_CACHE.unlink()

    print("\nAll done.")


if __name__ == "__main__":
    main()
