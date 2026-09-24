"""
export_fastdvdnet_onnx.py -- Export FastDVDnet to ONNX.

Usage:
    python export_fastdvdnet_onnx.py [--out DIR] [--sigma {15,25,50,all}] [--no-simplify]

Requirements:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install onnx onnxsim

    FastDVDnet must be installed from the m-tassano/fastdvdnet repository:
        git clone https://github.com/m-tassano/fastdvdnet
        pip install -e ./fastdvdnet

    Pretrained weights (fastdvdnet_nodvd.pth) are downloaded automatically on first run
    from the GitHub releases of m-tassano/fastdvdnet, or can be placed manually in
    ~/.cache/fastdvdnet/fastdvdnet_nodvd.pth.

The script will:
  1. Load the FastDVDnet color model weights.
  2. Wrap the model in a fixed-sigma inference module that:
       - Accepts 5 consecutive RGB frames concatenated along the channel axis,
         shape [1, 15, H, W] float32, normalised to [0, 1].
       - Has the noise sigma baked in as a constant tensor (no runtime sigma input),
         so one ONNX file corresponds to one noise level.
       - Produces the denoised center frame, shape [1, 3, H, W] float32, range [0, 1].
  3. Export the wrapper to ONNX at opset 17 with a fixed spatial size of 480x272
     (a representative 16:9 HD resolution). Dynamic spatial axes are also enabled so
     the exported model accepts any resolution that is a multiple of 1.
  4. Optionally simplify with onnxsim.

ONNX tensor interface:
    Input  "frames_in"  [1, 15, H, W]  float32  5 frames x 3 RGB channels, [0, 1]
    Output "frame_out"  [1,  3, H, W]  float32  denoised center frame, [0, 1]

Dynamic axes on both H and W are declared so OnnxRuntime can run the model at any
frame size without re-export, as long as the hardware has sufficient VRAM.

Sigma variants:
    sigma 15 -> fastdvdnet-sigma15.onnx  (light noise, faint ISO grain)
    sigma 25 -> fastdvdnet-sigma25.onnx  (medium noise, compression artefacts)
    sigma 50 -> fastdvdnet-sigma50.onnx  (heavy noise, archival / VHS footage)

Output files (default):
    ThirdParty/models/fastdvdnet/fastdvdnet-sigma15.onnx
    ThirdParty/models/fastdvdnet/fastdvdnet-sigma25.onnx
    ThirdParty/models/fastdvdnet/fastdvdnet-sigma50.onnx

License:
    FastDVDnet is released under the BSD 3-Clause licence.
    See https://github.com/m-tassano/fastdvdnet
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
DEFAULT_OUT_DIR = REPO_ROOT / "ThirdParty" / "models" / "fastdvdnet"
WEIGHTS_CACHE = Path.home() / ".cache" / "fastdvdnet" / "fastdvdnet_nodvd.pth"
WEIGHTS_URL = (
    "https://github.com/m-tassano/fastdvdnet/releases/download/v0.1/fastdvdnet_nodvd.pth"
)

SIGMA_VARIANTS = [15, 25, 50]

# Export spatial size: a representative 16:9 HD resolution.
# Dynamic axes allow runtime to accept any other size.
EXPORT_H = 272
EXPORT_W = 480

NUM_FRAMES = 5
CHANNELS_PER_FRAME = 3
INPUT_CHANNELS = NUM_FRAMES * CHANNELS_PER_FRAME  # 15


class FastDvdNetWrapper(nn.Module):
    """Single-sigma FastDVDnet wrapper with baked-in noise level.

    Accepts 5 consecutive RGB frames concatenated along the channel axis
    ([1, 15, H, W]) and returns the denoised center frame ([1, 3, H, W]).
    The noise level (sigma) is baked in as a constant tensor at trace time so
    the exported model does not require a separate sigma input at inference.

    Parameters
    ----------
    model:
        Loaded FastDVDnet model instance (FastDVDnet or FastDVDnetSEQ from the
        m-tassano/fastdvdnet repository).
    sigma:
        Noise standard deviation on the [0, 1] scale. Typical values: 15/255,
        25/255, 50/255 corresponding to sigma=15, 25, 50 on [0, 255].
    """

    def __init__(self, model: Any, sigma: float) -> None:
        """Initialise the wrapper with a loaded model and a fixed sigma value."""
        super().__init__()
        self.model = model
        self.register_buffer("sigma_map", torch.full((1, 1, 1, 1), sigma))

    def forward(self, frames_in: torch.Tensor) -> torch.Tensor:
        """Denoise the center frame of a 5-frame window.

        Parameters
        ----------
        frames_in:
            Shape [1, 15, H, W], float32, normalised to [0, 1].
            Layout: [frame0_R, frame0_G, frame0_B, frame1_R, ..., frame4_B].

        Returns
        -------
        torch.Tensor
            Shape [1, 3, H, W], float32, normalised to [0, 1].
        """
        h = frames_in.shape[2]
        w = frames_in.shape[3]
        noise_map = self.sigma_map.expand(1, 1, h, w)
        return self.model(frames_in, noise_map)


def download_weights(path: Path) -> None:
    """Download FastDVDnet pretrained weights if not already cached.

    Parameters
    ----------
    path:
        Local path to save the weights file.
    """
    if path.exists():
        print(f"Using cached weights: {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading FastDVDnet weights from {WEIGHTS_URL} ...")
    try:
        urllib.request.urlretrieve(WEIGHTS_URL, path)
        print(f"Saved to {path}")
    except Exception as exc:
        print(
            f"ERROR: Failed to download weights: {exc}\n"
            f"Download manually from:\n  {WEIGHTS_URL}\n"
            f"and place at:\n  {path}",
            file=sys.stderr,
        )
        sys.exit(1)


def load_model(device: str = "cpu") -> Any:
    """Load FastDVDnet color model from the m-tassano/fastdvdnet package.

    Parameters
    ----------
    device:
        PyTorch device string, e.g. ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    Any
        Loaded FastDVDnet model in eval mode.
    """
    try:
        from fastdvdnet import FastDVDnet
    except ImportError as exc:
        print(
            "ERROR: fastdvdnet is not installed.\n"
            "Clone the repo and install it with:  pip install -e ./fastdvdnet\n"
            f"Details: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    download_weights(WEIGHTS_CACHE)

    print("Loading FastDVDnet model weights...")
    model = FastDVDnet(num_input_frames=NUM_FRAMES)
    state_dict = torch.load(str(WEIGHTS_CACHE), map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def export_sigma(
    model: Any,
    sigma: int,
    out_dir: Path,
    simplify: bool,
) -> None:
    """Export a single sigma variant to ONNX.

    Parameters
    ----------
    model:
        Loaded FastDVDnet model.
    sigma:
        Integer noise level (15, 25, or 50). Normalised to [0,1] internally.
    out_dir:
        Directory to write the ONNX file into.
    simplify:
        Whether to run onnxsim simplification after export.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fastdvdnet-sigma{sigma}.onnx"

    sigma_norm = sigma / 255.0
    wrapper = FastDvdNetWrapper(model, sigma=sigma_norm)
    wrapper.eval()

    dummy_input = torch.zeros(1, INPUT_CHANNELS, EXPORT_H, EXPORT_W, dtype=torch.float32)

    print(
        f"Exporting sigma={sigma} variant to ONNX (opset 17)...\n"
        f"  Input  shape: [1, {INPUT_CHANNELS}, H, W]\n"
        f"  Output shape: [1, {CHANNELS_PER_FRAME}, H, W]\n"
        f"  Export size:  {EXPORT_W}x{EXPORT_H}"
    )

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input,),
            str(out_path),
            opset_version=17,
            input_names=["frames_in"],
            output_names=["frame_out"],
            dynamic_axes={
                "frames_in": {2: "height", 3: "width"},
                "frame_out": {2: "height", 3: "width"},
            },
            do_constant_folding=True,
        )

    print(f"Saved: {out_path}")

    if onnx is None:
        print("onnx not installed -- skipping shape inference and simplification.")
        return

    print("Running ONNX shape inference...")
    model_proto = onnx.load(str(out_path))
    model_proto = onnx.shape_inference.infer_shapes(model_proto)
    onnx.save(model_proto, str(out_path))

    if simplify and onnxsim_simplify is not None:
        print("Simplifying with onnxsim...")
        simplified, check_ok = onnxsim_simplify(model_proto)
        if check_ok:
            onnx.save(simplified, str(out_path))
            print("Simplification succeeded.")
        else:
            print("WARNING: onnxsim check failed -- keeping un-simplified model.")
    elif simplify:
        print("onnxsim not installed -- skipping simplification.")

    print(f"Done. Output: {out_path}")


def main() -> None:
    """Parse CLI arguments and run the export."""
    parser = argparse.ArgumentParser(
        description="Export FastDVDnet to ONNX for use with StreamDock."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--sigma",
        choices=["15", "25", "50", "all"],
        default="all",
        help="Noise sigma variant to export: 15, 25, 50, or all (default: all)",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip the onnxsim graph simplification pass.",
    )
    args = parser.parse_args()

    sigmas_to_export: list[int]
    if args.sigma == "all":
        sigmas_to_export = SIGMA_VARIANTS
    else:
        sigmas_to_export = [int(args.sigma)]

    model = load_model(device="cpu")

    for sigma in sigmas_to_export:
        print(f"\n--- Exporting sigma={sigma} ---")
        export_sigma(
            model=model,
            sigma=sigma,
            out_dir=args.out,
            simplify=not args.no_simplify,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
