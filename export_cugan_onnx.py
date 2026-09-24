"""
export_cugan_onnx.py -- Export Real-CUGAN weights to ONNX for use with CuganUpscaler.

Usage:
    python export_cugan_onnx.py [--out DIR] [--models all|no-denoise|conservative|denoise2x|denoise3x] [--no-simplify]

Requirements:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install onnx onnxruntime onnxsim

The script will:
  1. Download the Real-CUGAN architecture (upcunet_v3.py) from the bilibili/ailab repo.
  2. Download the updated_weights.zip from the GitHub release (~35 MB) containing all 2x
     denoise variants (no-denoise, conservative, denoise2x, denoise3x).
  3. Wrap each selected model into a single-input [1, 3, H, W] -> [1, 3, H*2, W*2] module.
  4. Export to ONNX at 128x128 (the tile size used by CuganUpscaler), optionally simplified.

Output files (default: ThirdParty/models/cugan/ relative to repo root):
    up2x-latest-no-denoise.onnx       2x scale, no denoising (noise level 0)
    up2x-latest-conservative.onnx     2x scale, conservative denoising (noise level 1)
    up2x-latest-denoise2x.onnx        2x scale, moderate denoising (noise level 2)
    up2x-latest-denoise3x.onnx        2x scale, heavy denoising (noise level 3)

CuganUpscaler usage:
    Models are discovered automatically by ModelCatalog.DiscoverForPipeline(Cugan).
    Place output files in ThirdParty/models/cugan/ and select the desired
    variant in the model picker, or set CUGAN_MODEL_PATH in app.local.cfg to specify a model.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import types
import urllib.request
import zipfile
from pathlib import Path

import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.onnx
from torch.nn.functional import pad

try:
    from onnxsim import simplify as onnxsim_simplify
except ImportError:
    onnxsim_simplify = None

AILAB_RAW = "https://raw.githubusercontent.com/bilibili/ailab/main/Real-CUGAN"
WEIGHTS_ZIP_URL = (
    "https://github.com/bilibili/ailab/releases/download/Real-CUGAN/updated_weights.zip"
)
ARCH_URL = f"{AILAB_RAW}/upcunet_v3.py"

TILE_SIZE = 128  # must match CuganUpscaler.TileSize

# Model registry: key -> (zip entry path, output filename)
_MODELS: dict[str, tuple[str, str]] = {
    "no-denoise":   ("updated_weights/up2x-latest-no-denoise.pth",   "up2x-latest-no-denoise.onnx"),
    "conservative": ("updated_weights/up2x-latest-conservative.pth", "up2x-latest-conservative.onnx"),
    "denoise2x":    ("updated_weights/up2x-latest-denoise2x.pth",    "up2x-latest-denoise2x.onnx"),
    "denoise3x":    ("updated_weights/up2x-latest-denoise3x.pth",    "up2x-latest-denoise3x.onnx"),
}


def download(url: str, dest: Path, label: str) -> None:
    """Download url to `dest`, printing progress. Skips if `dest` already exists."""
    if dest.exists():
        print(f"  [skip] {label} already downloaded.")
        return
    print(f"  Downloading {label} ...", end=" ", flush=True)

    def reporthook(count: int, block: int, total: int) -> None:
        if total > 0:
            pct = min(100, count * block * 100 // total)
            print(f"\r  Downloading {label} ... {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook)
    print(f"\r  Downloaded  {label} ({dest.stat().st_size // 1024} KB)")


def load_model(arch_path: Path, weights_path: Path) -> nn.Module:
    """
    Dynamically import upcunet_v3.py, instantiate RealWaifuUpScaler (2x),
    and return the inner UpCunet2x nn.Module in eval mode.
    """
    spec = importlib.util.spec_from_file_location("upcunet_v3", arch_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)

    # upcunet_v3 imports cv2 at the module level for the CLI helper functions.
    # We only need the model classes, so stub out cv2 if it is not installed.
    if "cv2" not in sys.modules:
        stub = types.ModuleType("cv2")
        sys.modules["cv2"] = stub

    spec.loader.exec_module(mod)

    # RealWaifuUpScaler wraps the inner UpCunet2x nn.Module and calls
    # torch.load internally, so instantiate it to get the loaded model.
    inner = mod.RealWaifuUpScaler(2, str(weights_path), half=False, device="cpu")
    model = inner.model
    model.eval()
    return model


class CuganWrapper(nn.Module):
    """
    Traceable wrapper that accepts a [1, 3, H, W] float32 RGB input (values 0-1)
    and returns [1, 3, H*2, W*2] float32 output in the same range.

    Bypasses UpCunet2x.forward() (which has untraceable control flow) and
    instead directly calls unet1/unet2 using the tile_mode=0 (no-tile) path
    with alpha=1.0 and pro=False. CuganUpscaler feeds normalised float tiles
    directly so no additional normalisation is needed.
    """

    def __init__(self, model: nn.Module) -> None:
        """Initialise wrapper by extracting `unet1` and `unet2` from the loaded model."""
        super().__init__()
        self.unet1: nn.Module = getattr(model, "unet1")
        self.unet2: nn.Module = getattr(model, "unet2")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the no-tile 2x upscale pass and return the clamped output tensor."""
        _n, _c, h0, w0 = x.shape
        # Pad to nearest multiple of 2 and add 18-pixel reflect border.
        ph = ((h0 - 1) // 2 + 1) * 2
        pw = ((w0 - 1) // 2 + 1) * 2
        x = pad(x, (18, 18 + pw - w0, 18, 18 + ph - h0), "reflect")
        feat = self.unet1(x)
        out = self.unet2(feat)  # alpha equal to 1.0 (default)
        feat = pad(feat, (-20, -20, -20, -20))
        out = torch.add(out, feat)
        # Crop to 2x original resolution and clamp to [0, 1].
        return out[:, :, : h0 * 2, : w0 * 2].clamp(0.0, 1.0)


def export(out_dir: Path, simplify: bool, model_keys: list[str], keep_tmp: bool = False) -> None:
    """Download weights, build ONNX models for each selected key, and write them to out_dir."""
    tmp_dir = out_dir / "_cugan_export_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    arch_path = tmp_dir / "upcunet_v3.py"
    weights_zip = tmp_dir / "updated_weights.zip"

    print("Step 1: downloading architecture ...")
    download(ARCH_URL, arch_path, "upcunet_v3.py")

    print("Step 2: downloading weights zip ...")
    download(WEIGHTS_ZIP_URL, weights_zip, "updated_weights.zip")

    for model_key in model_keys:
        zip_entry, out_name = _MODELS[model_key]
        weights_path = tmp_dir / Path(zip_entry).name

        if not weights_path.exists():
            print(f"  Extracting {zip_entry} ...", end=" ", flush=True)
            with zipfile.ZipFile(weights_zip) as zf:
                data = zf.read(zip_entry)
                weights_path.write_bytes(data)
            print(f"done ({weights_path.stat().st_size // 1024} KB)")

        out_path = out_dir / out_name
        _export_single(arch_path, weights_path, out_path, model_key, simplify)

    if not keep_tmp and tmp_dir.exists():
        shutil.rmtree(tmp_dir)
        print(f"Cleaned up temporary directory: {tmp_dir}")


def _export_single(
    arch_path: Path,
    weights_path: Path,
    out_path: Path,
    model_key: str,
    simplify: bool,
) -> None:
    """Load, wrap, export, and sanity-check a single CUGAN variant."""
    print(f"\nStep 3 [{model_key}]: loading model ...")
    model = load_model(arch_path, weights_path)
    wrapper = CuganWrapper(model)
    wrapper.eval()

    if out_path.exists():
        print(f"  [skip] {out_path} already exists.")
        return

    dummy = torch.zeros(1, 3, TILE_SIZE, TILE_SIZE)

    print(f"Step 4 [{model_key}]: exporting ONNX to {out_path} ...")
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out_path),
        dynamo=False,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "N"}, "output": {0: "N"}},
    )
    print(f"  Exported ({out_path.stat().st_size // 1024} KB)")

    if simplify:
        if onnxsim_simplify is None:
            print("  [skip] onnxsim not installed, skipping simplification.")
        else:
            print("  Simplifying with onnxsim ...")
            model_proto = onnx.load(str(out_path))
            if onnxsim_simplify is None:
                print("  onnxsim not available, skipping simplification.")
            else:
                simplified, ok = onnxsim_simplify(model_proto)
                if ok:
                    onnx.save(simplified, str(out_path))
                    print(f"  Simplified ({out_path.stat().st_size // 1024} KB)")
                else:
                    print("  Simplification failed, keeping original.")

    print(f"  Running sanity check [{model_key}] ...")
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": dummy.numpy()})[0]
    expected_size = TILE_SIZE * 2
    assert out.shape == (1, 3, expected_size, expected_size), (
        f"Unexpected output shape {out.shape}, expected (1, 3, {expected_size}, {expected_size})"
    )
    print(f"  Sanity check passed: input={dummy.shape} -> output={out.shape}")
    print(f"  Model written to: {out_path}")


def main() -> None:
    """Export Real-CUGAN variants to ONNX."""
    default_out = Path(__file__).resolve().parent / "cugan"

    parser = argparse.ArgumentParser(description="Export Real-CUGAN variants to ONNX.")
    parser.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help=f"Output directory (default: {default_out})",
    )
    parser.add_argument(
        "--models",
        default="all",
        help=(
            "Comma-separated list of model keys to export, or 'all'. "
            f"Available: {', '.join(_MODELS)}. Default: all."
        ),
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip onnxsim simplification pass.",
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Keep the temporary directory with downloaded weights and arch files after export.",
    )
    args = parser.parse_args()

    if args.models == "all":
        model_keys = list(_MODELS.keys())
    else:
        model_keys = [k.strip() for k in args.models.split(",")]
        unknown = [k for k in model_keys if k not in _MODELS]
        if unknown:
            parser.error(f"Unknown model key(s): {', '.join(unknown)}. Available: {', '.join(_MODELS)}")

    export(args.out, simplify=not args.no_simplify, model_keys=model_keys, keep_tmp=args.keep_tmp)


if __name__ == "__main__":
    main()
