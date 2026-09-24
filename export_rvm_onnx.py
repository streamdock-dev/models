"""
export_rvm_onnx.py -- Export RobustVideoMatting (mobilenetv3) to ONNX with explicit
recurrent state I/O.

Usage:
    python src/Scripts/export_rvm_onnx.py [--out DIR] [--no-simplify]

Requirements:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install onnx onnxsim

    RobustVideoMatting must be cloned from GitHub:
        git clone https://github.com/PeterL1997/RobustVideoMatting
    Then either:
        - Add it to PYTHONPATH: set PYTHONPATH=RobustVideoMatting  (Windows)
        - Or place model/model.py and model/mobilenetv3.py alongside this script.

The script will:
  1. Download pretrained mobilenetv3 weights from the official GitHub release (or
     load from cache at ~/.cache/rvm/rvm_mobilenetv3.pth).
  2. Wrap the model to expose explicit recurrent state inputs/outputs:
         r1i, r2i, r3i, r4i  (initial states, zeros on first frame)
         r1o, r2o, r3o, r4o  (updated states to feed back next frame)
  3. Export to ONNX at opset 17 with dynamic spatial and batch dimensions.
  4. Optionally simplify with onnxsim.

ONNX tensor interface (agreed with RobustVideoMattingFilter.cs):
    Inputs:
        "src"   [1, 3, H, W]       float32   normalised RGB source frame [0, 1]
        "r1i"   [1, 16, H/2, W/2]  float32   recurrent state 1 (zeros on first frame)
        "r2i"   [1, 20, H/4, W/4]  float32   recurrent state 2
        "r3i"   [1, 40, H/8, W/8]  float32   recurrent state 3
        "r4i"   [1, 64, H/16,W/16] float32   recurrent state 4
    Outputs:
        "fgr"   [1, 3, H, W]       float32   foreground RGB [0, 1]
        "pha"   [1, 1, H, W]       float32   alpha mask [0, 1]
        "r1o"   [1, 16, H/2, W/2]  float32   updated recurrent state 1
        "r2o"   [1, 20, H/4, W/4]  float32   updated recurrent state 2
        "r3o"   [1, 40, H/8, W/8]  float32   updated recurrent state 3
        "r4o"   [1, 64, H/16,W/16] float32   updated recurrent state 4

The recurrent states allow the model to track foreground over time. Pass zeros on
the first frame; feed back each r_o as the r_i for the next frame.

Output file (default):
    ThirdParty/models/rvm/rvm-mobilenetv3.onnx

Models are discovered automatically by ModelCatalog.DiscoverForPipeline(Matting).

License:
    RobustVideoMatting is released under the MIT licence.
    See https://github.com/PeterL1997/RobustVideoMatting
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
    onnxsim_simplify = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "ThirdParty" / "models" / "rvm"
WEIGHTS_CACHE = Path.home() / ".cache" / "rvm" / "rvm_mobilenetv3.pth"

WEIGHTS_URL = (
    "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth"
)

EXPORT_H = 480
EXPORT_W = 640


class RvmWrapper(nn.Module):
    """Thin wrapper around RVM mobilenetv3 that routes recurrent states as explicit I/O.

    The original model signature is::

        fgr, pha, r1, r2, r3, r4 = model(src, r1, r2, r3, r4, downsample_ratio=1.0)

    This wrapper presents the states as named ONNX inputs/outputs so OnnxRuntime can
    consume them without requiring a Python runtime.
    """

    def __init__(self, model: nn.Module) -> None:
        """Initialize the wrapper with the base RVM model.

        Args:
            model: The loaded RobustVideoMatting mobilenetv3 model.
        """
        super().__init__()
        self.model = model

    def forward(
        self,
        src: torch.Tensor,
        r1i: torch.Tensor,
        r2i: torch.Tensor,
        r3i: torch.Tensor,
        r4i: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one frame of RVM inference.

        Args:
            src:  Source frame [1, 3, H, W] float32 RGB [0, 1].
            r1i:  Recurrent state 1 from previous frame (or zeros).
            r2i:  Recurrent state 2 from previous frame (or zeros).
            r3i:  Recurrent state 3 from previous frame (or zeros).
            r4i:  Recurrent state 4 from previous frame (or zeros).

        Returns:
            fgr: Foreground RGB [1, 3, H, W] float32 [0, 1].
            pha: Alpha mask [1, 1, H, W] float32 [0, 1].
            r1o: Updated recurrent state 1.
            r2o: Updated recurrent state 2.
            r3o: Updated recurrent state 3.
            r4o: Updated recurrent state 4.
        """
        fgr, pha, r1o, r2o, r3o, r4o = self.model(
            src, r1i, r2i, r3i, r4i, downsample_ratio=torch.tensor(1.0)
        )
        return fgr, pha, r1o, r2o, r3o, r4o


def download_weights(weights_url: str, dest: Path) -> None:
    """Download pretrained weights to dest, creating parent dirs as needed.

    Args:
        weights_url: Direct download URL for the .pth file.
        dest: Local path to save the weights.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading weights from {weights_url} -> {dest}")
    urllib.request.urlretrieve(weights_url, str(dest))
    print("Download complete.")


def load_model(weights_path: Path) -> Any:
    """Load and return the RVM mobilenetv3 model with pretrained weights.

    Attempts to import MattingNetwork from the RobustVideoMatting repository.
    If the package is not on PYTHONPATH, prints an installation hint and exits.

    Args:
        weights_path: Path to the .pth checkpoint file.

    Returns:
        The model in eval mode.
    """
    try:
        from model import MattingNetwork  # type: ignore[import]
    except ImportError:
        print(
            "ERROR: Could not import MattingNetwork. Clone RobustVideoMatting and add it to PYTHONPATH:\n"
            "  git clone https://github.com/PeterL1997/RobustVideoMatting\n"
            "  set PYTHONPATH=RobustVideoMatting  (Windows)\n"
            "  export PYTHONPATH=RobustVideoMatting  (Linux/macOS)",
            file=sys.stderr,
        )
        sys.exit(1)

    model = MattingNetwork("mobilenetv3").eval()
    state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model


def export(out_dir: Path, simplify: bool) -> None:
    """Run the full export pipeline: load weights -> wrap -> trace -> save ONNX.

    Args:
        out_dir: Directory to write the .onnx file.
        simplify: Whether to run onnxsim after export.
    """
    if not WEIGHTS_CACHE.exists():
        download_weights(WEIGHTS_URL, WEIGHTS_CACHE)
    else:
        print(f"Using cached weights at {WEIGHTS_CACHE}")

    model = load_model(WEIGHTS_CACHE)
    wrapper = RvmWrapper(model).eval()

    h, w = EXPORT_H, EXPORT_W
    src = torch.zeros(1, 3, h, w)
    r1i = torch.zeros(1, 16, h // 2, w // 2)
    r2i = torch.zeros(1, 20, h // 4, w // 4)
    r3i = torch.zeros(1, 40, h // 8, w // 8)
    r4i = torch.zeros(1, 64, h // 16, w // 16)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rvm-mobilenetv3.onnx"
    print(f"Exporting to {out_path} ...")

    torch.onnx.export(
        wrapper,
        (src, r1i, r2i, r3i, r4i),
        str(out_path),
        dynamo=False,
        input_names=["src", "r1i", "r2i", "r3i", "r4i"],
        output_names=["fgr", "pha", "r1o", "r2o", "r3o", "r4o"],
        dynamic_axes={
            "src":  {0: "batch", 2: "height", 3: "width"},
            "r1i":  {0: "batch", 2: "h2",  3: "w2"},
            "r2i":  {0: "batch", 2: "h4",  3: "w4"},
            "r3i":  {0: "batch", 2: "h8",  3: "w8"},
            "r4i":  {0: "batch", 2: "h16", 3: "w16"},
            "fgr":  {0: "batch", 2: "height", 3: "width"},
            "pha":  {0: "batch", 2: "height", 3: "width"},
            "r1o":  {0: "batch", 2: "h2",  3: "w2"},
            "r2o":  {0: "batch", 2: "h4",  3: "w4"},
            "r3o":  {0: "batch", 2: "h8",  3: "w8"},
            "r4o":  {0: "batch", 2: "h16", 3: "w16"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Exported: {out_path}")

    if simplify and onnx is not None and onnxsim_simplify is not None:
        print("Simplifying with onnxsim ...")
        model_proto = onnx.load(str(out_path))
        simplified, check_ok = onnxsim_simplify(model_proto)
        if check_ok:
            onnx.save(simplified, str(out_path))
            print("Simplified OK.")
        else:
            print("WARNING: onnxsim check failed; keeping original export.")
    elif simplify:
        print("WARNING: onnx/onnxsim not installed; skipping simplification.")

    print("Done.")


def main() -> None:
    """Parse command-line arguments and run the export."""
    parser = argparse.ArgumentParser(description="Export RobustVideoMatting mobilenetv3 to ONNX.")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip onnxsim simplification.",
    )
    args = parser.parse_args()

    export(args.out, simplify=not args.no_simplify)


if __name__ == "__main__":
    main()
