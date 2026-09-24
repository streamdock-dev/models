"""
export_zerodce_onnx.py -- Export Zero-DCE++ to ONNX.

Usage:
    python export_zerodce_onnx.py [--out DIR] [--no-simplify]

Requirements:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install onnx onnxsim

    Zero-DCE++ (Zero-DCE_extension) must be cloned from GitHub:
        git clone https://github.com/Li-Chongyi/Zero-DCE_extension
    Then place the model definition file on the Python path:
        set PYTHONPATH=Zero-DCE_extension/Zero-DCE++/code   (Windows)
        export PYTHONPATH=Zero-DCE_extension/Zero-DCE++/code  (macOS/Linux)
    Or copy model.py from that directory alongside this script.

    Pretrained weights (Epoch99.pth) can be placed manually at
    ~/.cache/zerodcepp/Epoch99.pth or will be downloaded automatically
    if the --weights-url flag points to a direct download link.

The script will:
  1. Load the Zero-DCE++ enhance_net_nopool model weights.
  2. Wrap the model in a single-input/single-output module that:
       - Accepts one RGB frame, shape [1, 3, H, W] float32, normalised to [0, 1].
       - Applies the learned per-pixel illumination enhancement curves.
       - Returns the enhanced frame, shape [1, 3, H, W] float32, range [0, 1].
  3. Export the wrapper to ONNX at opset 17 with a fixed spatial size of 256x256
     (matching the tile size used at inference). Dynamic spatial axes are also
     enabled so the model can be run full-frame when the GPU has sufficient VRAM.
  4. Optionally simplify with onnxsim.

ONNX tensor interface:
    Input  "input"   [1, 3, H, W]  float32  normalised RGB [0, 1]
    Output "output"  [1, 3, H, W]  float32  enhanced RGB   [0, 1]

Dynamic axes on H and W allow OnnxRuntime to run the model at any spatial
resolution at inference time. The C# ZeroDceEnhancer uses 256x256 tiling with
16 px linear-ramp overlap blending for fixed-axis models, or full-frame inference
when the model metadata indicates dynamic spatial axes and the frame fits in VRAM.

Output file (default):
    ThirdParty/models/zerodcepp/zerodcepp.onnx

License:
    Zero-DCE++ is released under the MIT licence.
    See https://github.com/Li-Chongyi/Zero-DCE_extension
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.onnx

try:
    import onnx
    from onnxsim import simplify as onnxsim_simplify
except ImportError:
    onnx = None  # type: ignore[assignment]
    onnxsim_simplify = None

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "ThirdParty" / "models" / "zerodcepp"
WEIGHTS_CACHE = Path.home() / ".cache" / "zerodcepp" / "Epoch99.pth"

# Export spatial size: matches the tile size used by ZeroDceEnhancer at inference.
# Dynamic axes allow the model to run at any other resolution at runtime.
EXPORT_H = 256
EXPORT_W = 256


class ZeroDcePlusPlusWrapper(nn.Module):
    """Single-input/single-output wrapper around Zero-DCE++ enhance_net_nopool.

    Accepts one RGB frame ([1, 3, H, W]) and returns the enhanced frame
    ([1, 3, H, W]). The underlying model returns a tuple; this wrapper
    unpacks the enhanced image (first element) so that the ONNX graph has
    a single clean output tensor.

    Parameters
    ----------
    model:
        Loaded enhance_net_nopool instance from the Zero-DCE_extension repo.
    """

    def __init__(self, model: Any) -> None:
        """Initialise the wrapper with a loaded Zero-DCE++ model."""
        super().__init__()
        self.model = model

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Enhance a single low-light RGB frame.

        Parameters
        ----------
        input:
            Shape [1, 3, H, W], float32, normalised to [0, 1].

        Returns
        -------
        torch.Tensor
            Shape [1, 3, H, W], float32, normalised to [0, 1].
            Values are clamped to [0, 1] so the output is always valid.
        """
        result = self.model(input)
        enhanced = result[0] if isinstance(result, (tuple, list)) else result
        return torch.clamp(enhanced, 0.0, 1.0)


def load_model(weights_path: Path, device: str = "cpu") -> Any:
    """Load the Zero-DCE++ model from enhance_net_nopool and pretrained weights.

    Parameters
    ----------
    weights_path:
        Absolute path to the Epoch99.pth weights file.
    device:
        PyTorch device string, e.g. ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    Any
        Loaded enhance_net_nopool model in eval mode.
    """
    try:
        from model import enhance_net_nopool  # type: ignore[import]
    except ImportError as exc:
        print(
            "ERROR: Zero-DCE++ model.py is not on the Python path.\n"
            "Clone the repository and add its code directory to PYTHONPATH:\n"
            "  git clone https://github.com/Li-Chongyi/Zero-DCE_extension\n"
            "  set PYTHONPATH=Zero-DCE_extension/Zero-DCE++/code  (Windows)\n"
            "  export PYTHONPATH=Zero-DCE_extension/Zero-DCE++/code  (macOS/Linux)\n"
            "Or copy model.py alongside this script.\n"
            f"Details: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not weights_path.exists():
        print(
            f"ERROR: Weights file not found: {weights_path}\n"
            "Download Epoch99.pth from the Zero-DCE_extension repository releases\n"
            "or Google Drive link provided in the repository README, then place it at:\n"
            f"  {weights_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading Zero-DCE++ weights from {weights_path} ...")
    model = enhance_net_nopool(scale_factor=1)
    state_dict = torch.load(str(weights_path), map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def export_model(model: Any, out_dir: Path, simplify: bool) -> None:
    """Export the Zero-DCE++ model to ONNX.

    Parameters
    ----------
    model:
        Loaded enhance_net_nopool model.
    out_dir:
        Directory to write the ONNX file into.
    simplify:
        Whether to run onnxsim graph simplification after export.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "zerodcepp.onnx"

    wrapper = ZeroDcePlusPlusWrapper(model)
    wrapper.eval()

    dummy_input = torch.zeros(1, 3, EXPORT_H, EXPORT_W, dtype=torch.float32)

    print(
        f"Exporting Zero-DCE++ to ONNX (opset 17)...\n"
        f"  Input  shape: [1, 3, H, W]\n"
        f"  Output shape: [1, 3, H, W]\n"
        f"  Export size:  {EXPORT_W}x{EXPORT_H}"
    )

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input,),
            str(out_path),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {2: "height", 3: "width"},
                "output": {2: "height", 3: "width"},
            },
            do_constant_folding=True,
        )

    print(f"Saved: {out_path}")

    if onnx is None:
        print("onnx not installed; skipping model verification.")
        return

    model_onnx = onnx.load(str(out_path))
    onnx.checker.check_model(model_onnx)
    print("ONNX model check passed.")

    if simplify and onnxsim_simplify is not None:
        print("Running onnxsim simplification...")
        model_simplified, check = onnxsim_simplify(model_onnx)
        if check:
            onnx.save(model_simplified, str(out_path))
            print("Simplification succeeded.")
        else:
            print("WARNING: onnxsim simplification check failed; keeping original graph.")
    elif simplify and onnxsim_simplify is None:
        print("onnxsim not installed; skipping simplification.")


def main() -> None:
    """Parse arguments and run the export."""
    parser = argparse.ArgumentParser(
        description="Export Zero-DCE++ (Low-Light Enhancement) to ONNX."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        metavar="DIR",
        help=f"Output directory. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=WEIGHTS_CACHE,
        metavar="PATH",
        help=f"Path to Epoch99.pth weights file. Default: {WEIGHTS_CACHE}",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip onnxsim graph simplification.",
    )
    args = parser.parse_args()

    model = load_model(weights_path=args.weights, device="cpu")
    export_model(model=model, out_dir=args.out, simplify=not args.no_simplify)

    print("\nDone.")


if __name__ == "__main__":
    main()
