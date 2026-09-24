"""
export_waifu2x_onnx.py -- Export Waifu2x cunet weights to ONNX for use with Waifu2xUpscaler.

Usage:
    python export_waifu2x_onnx.py [--out DIR] [--models all|noise0|noise1|noise2|noise3] [--no-simplify]

Requirements:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install onnx onnxruntime onnxsim

The script will:
  1. Download the cunet architecture from nagadomi/nunif.
  2. Download waifu2x_pretrained_models_20250502.zip from nunif releases (437 MB, one-time).
  3. Extract the cunet/art noise+scale2x .pth weights for each selected noise variant.
  4. Wrap each model into a single-input [1, 3, H, W] -> [1, 3, H*2, W*2] traceable module.
  5. Export to ONNX at 128x128 (the tile size used by Waifu2xUpscaler), optionally simplified.

Output files (default: ThirdParty/models/waifu2x/ relative to repo root):
    waifu2x-cunet-scale2-noise0.onnx    2x scale, no denoising
    waifu2x-cunet-scale2-noise1.onnx    2x scale, light denoising
    waifu2x-cunet-scale2-noise2.onnx    2x scale, moderate denoising
    waifu2x-cunet-scale2-noise3.onnx    2x scale, heavy denoising

Waifu2xUpscaler usage:
    Models are discovered automatically by ModelCatalog.DiscoverForPipeline(Waifu2x).
    Place output files in ThirdParty/models/waifu2x/ and select the desired
    variant in the model picker.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import zipfile
from types import ModuleType
from typing import Any
import urllib.request
from pathlib import Path

import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx

try:
    from onnxsim import simplify as onnxsim_simplify
except ImportError:
    onnxsim_simplify = None

_CUNET_ARCH_URL = (
    "https://raw.githubusercontent.com/nagadomi/nunif/master"
    "/waifu2x/models/cunet.py"
)

# Pretrained models zip from the nunif official releases.
# Contains pretrained_models/cunet/art/noise{N}_scale2x.pth
_PRETRAINED_ZIP_URL = (
    "https://github.com/nagadomi/nunif/releases/download/0.0.0"
    "/waifu2x_pretrained_models_20250502.zip"
)

TILE_SIZE = 128

# Map from model key to (zip internal path, output filename).
_MODELS: dict[str, tuple[str, str]] = {
    "noise0": (
        "pretrained_models/cunet/art/noise0_scale2x.pth",
        "waifu2x-cunet-scale2-noise0.onnx",
    ),
    "noise1": (
        "pretrained_models/cunet/art/noise1_scale2x.pth",
        "waifu2x-cunet-scale2-noise1.onnx",
    ),
    "noise2": (
        "pretrained_models/cunet/art/noise2_scale2x.pth",
        "waifu2x-cunet-scale2-noise2.onnx",
    ),
    "noise3": (
        "pretrained_models/cunet/art/noise3_scale2x.pth",
        "waifu2x-cunet-scale2-noise3.onnx",
    ),
}


def download(url: str, dest: Path, label: str) -> None:
    """Download url to `dest`, printing progress. Skips if `dest` already exists."""
    if dest.exists():
        print(f"  [skip] {label} already downloaded.")
        return
    print(f"  Downloading {label} ...", end=" ", flush=True)
    urllib.request.urlretrieve(url, dest)
    print(f"done ({dest.stat().st_size // 1024} KB)")


def extract_from_zip(zip_path: Path, member: str, dest: Path) -> None:
    """Extract a single member from a zip archive to dest. Skips if dest already exists."""
    if dest.exists():
        print(f"  [skip] {dest.name} already extracted.")
        return
    print(f"  Extracting {member} ...", end=" ", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        data = zf.read(member)
    dest.write_bytes(data)
    print(f"done ({dest.stat().st_size // 1024} KB)")


def load_cunet_arch(arch_path: Path) -> ModuleType:
    """
    Dynamically load the cunet.py architecture module.
    The nunif cunet.py defines UpCUNet (2x scale+denoise) and CUNet (noise-only).
    Stubs out nunif dependencies so no nunif install is required.
    """
    spec = importlib.util.spec_from_file_location("cunet_arch", arch_path)
    if spec is None:
        raise RuntimeError(f"Failed to load cunet architecture from {arch_path}")
    mod = importlib.util.module_from_spec(spec)

    # Minimal SEBlock (Squeeze-and-Excitation) matching nunif's implementation.
    class _SEBlock(nn.Module):
        def __init__(self, in_channels: int, reduction: int = 8, bias: bool = False) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1, 1, 0, bias=bias)
            self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, 1, 1, 0, bias=bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            z = F.adaptive_avg_pool2d(x, 1)
            z = self.conv1(z)
            z = F.relu(z, inplace=True)
            z = self.conv2(z)
            z = torch.sigmoid(z)
            return x * z.expand(x.shape)

    # Minimal I2IBaseModel stub -- provides only what cunet.py's UpCUNet/CUNet needs.
    class _I2IBaseModel(nn.Module):
        def __init__(self, kwargs: dict[str, Any], scale: int, offset: int,
                     in_channels: int = 3, **_kw: object) -> None:
            super().__init__()
            self.i2i_scale = scale
            self.i2i_offset = offset
            self.kwargs = {k: v for k, v in kwargs.items() if k not in {"self", "__class__"}}

        def register_tile_size_validator(self, validator: object) -> None:
            pass

        def get_kwargs(self) -> dict:
            return self.kwargs

    def _register_model(cls: type) -> type:
        return cls

    # Build stub modules for all nunif imports cunet.py needs.
    nunif_models_stub = ModuleType("nunif.models")
    nunif_models_stub.I2IBaseModel = _I2IBaseModel  # type: ignore[attr-defined]
    nunif_models_stub.register_model = _register_model  # type: ignore[attr-defined]

    nunif_modules_stub = ModuleType("nunif.modules")
    nunif_modules_stub.SEBlock = _SEBlock  # type: ignore[attr-defined]

    for mod_name, stub in [
        ("nunif", ModuleType("nunif")),
        ("nunif.models", nunif_models_stub),
        ("nunif.modules", nunif_modules_stub),
        ("nunif.modules.res_block", ModuleType("nunif.modules.res_block")),
        ("nunif.modules.permute", ModuleType("nunif.modules.permute")),
        ("nunif.modules.attention", ModuleType("nunif.modules.attention")),
    ]:
        sys.modules[mod_name] = stub

    if spec.loader is not None:
        spec.loader.exec_module(mod)
    return mod


class Waifu2xCunetWrapper(nn.Module):
    """
    Traceable wrapper for the Waifu2x cunet 2x model.

    Accepts [1, 3, H, W] float32 RGB input (values 0-1) and returns
    [1, 3, H*2, W*2] float32 output in the same range.

    The cunet architecture requires a 128-pixel context window with reflect padding.
    We use 36 pixels of padding on each side (matching the cunet receptive field) so
    that the tiled 128x128 input produces a clean 256x256 output without border
    artefacts when assembled by Waifu2xUpscaler.
    """

    PAD = 36

    def __init__(self, model: nn.Module) -> None:
        """Wrap a loaded cunet model (UNet2 instance)."""
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run 2x upscale with reflect padding and crop to exact 2x output size."""
        _n, _c, h0, w0 = x.shape
        p = self.PAD
        x_padded = F.pad(x, (p, p, p, p), "reflect")
        out = self.model(x_padded)
        # Crop to 2x input size from the padded output (2*p offset on each side).
        return out[:, :, 2 * p : 2 * p + h0 * 2, 2 * p : 2 * p + w0 * 2].clamp(0.0, 1.0)


def load_model(arch_path: Path, weights_path: Path) -> nn.Module:
    """Load UpCUNet architecture and nunif-format weights, return model in eval mode."""
    arch_mod = load_cunet_arch(arch_path)

    # nunif cunet.py defines UpCUNet for 2x scale + denoise.
    model_cls = getattr(arch_mod, "UpCUNet", None)
    if model_cls is None:
        raise RuntimeError(
            f"UpCUNet class not found in {arch_path}. "
            "Check the nunif cunet.py architecture file."
        )

    # no_clip=True: clamp the intermediate z1 output (default in training; always correct).
    model: nn.Module = model_cls(in_channels=3, out_channels=3, no_clip=True)
    data = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    # nunif save_model stores weights under the "state_dict" key.
    if isinstance(data, dict) and "nunif_model" in data:
        state = data["state_dict"]
    elif isinstance(data, dict) and "state_dict" in data:
        state = data["state_dict"]
    else:
        state = data
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def export(out_dir: Path, simplify: bool, model_keys: list[str], keep_tmp: bool = False) -> None:
    """Download weights, build ONNX models for each selected key, and write them to out_dir."""
    tmp_dir = out_dir / "_waifu2x_export_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    arch_path = tmp_dir / "cunet.py"
    print("Step 1: downloading cunet architecture ...")
    download(_CUNET_ARCH_URL, arch_path, "cunet.py")

    # Download the pretrained models zip (contains all noise variants).
    zip_path = tmp_dir / "waifu2x_pretrained_models_20250502.zip"
    print("Step 2: downloading pretrained models zip (437 MB) ...")
    download(_PRETRAINED_ZIP_URL, zip_path, "waifu2x_pretrained_models_20250502.zip")

    for model_key in model_keys:
        zip_member, out_name = _MODELS[model_key]
        weights_path = tmp_dir / f"waifu2x-cunet-{model_key}.pth"
        out_path = out_dir / out_name

        if out_path.exists():
            print(f"  [skip] {out_name} already exists.")
            continue

        print(f"\nStep 3 [{model_key}]: extracting weights from zip ...")
        extract_from_zip(zip_path, zip_member, weights_path)

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
    """Load, wrap, export, and sanity-check a single Waifu2x cunet variant."""
    print(f"\nStep 4 [{model_key}]: loading model ...")
    model = load_model(arch_path, weights_path)
    wrapper = Waifu2xCunetWrapper(model)
    wrapper.eval()

    dummy = torch.zeros(1, 3, TILE_SIZE, TILE_SIZE)

    print(f"Step 5 [{model_key}]: exporting ONNX to {out_path} ...")
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out_path),
        opset_version=17,
        dynamo=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    if simplify and onnxsim_simplify is not None:
        print(f"Step 6 [{model_key}]: simplifying ONNX graph ...")
        model_onnx = onnx.load(str(out_path))
        simplified, check = onnxsim_simplify(model_onnx)
        if check:
            onnx.save(simplified, str(out_path))
            print("  Simplification OK.")
        else:
            print("  Simplification check failed -- keeping original graph.")
    elif simplify:
        print("  [skip] onnxsim not installed; skipping simplification.")

    print(f"Step 7 [{model_key}]: sanity-check via CPUExecutionProvider ...")
    sess_opts = ort.SessionOptions()
    sess = ort.InferenceSession(str(out_path), sess_opts, providers=["CPUExecutionProvider"])
    inp = dummy.numpy()
    out = sess.run(None, {"input": inp})[0]
    assert out.shape == (1, 3, TILE_SIZE * 2, TILE_SIZE * 2), (
        f"Unexpected output shape {out.shape}"
    )
    print(f"  Output shape: {out.shape} -- OK")
    print(f"  Saved: {out_path} ({out_path.stat().st_size // 1024} KB)")


def main() -> None:
    """Parse arguments and run export."""
    here = Path(__file__).resolve().parent
    default_out = here.parent.parent / "ThirdParty" / "models" / "waifu2x"

    parser = argparse.ArgumentParser(
        description="Export Waifu2x cunet weights to ONNX for use with Waifu2xUpscaler."
    )
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
            "Which noise variants to export. "
            "Choices: all | noise0 | noise1 | noise2 | noise3 "
            "(comma-separated for multiple, e.g. noise1,noise3)"
        ),
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip onnxsim simplification step.",
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Keep the temporary directory with downloaded weights and arch files after export.",
    )
    args = parser.parse_args()

    all_keys = list(_MODELS.keys())
    if args.models.strip().lower() == "all":
        keys = all_keys
    else:
        keys = [k.strip() for k in args.models.split(",")]
        unknown = [k for k in keys if k not in _MODELS]
        if unknown:
            parser.error(f"Unknown model keys: {unknown}. Valid keys: {all_keys}")

    args.out.mkdir(parents=True, exist_ok=True)
    export(args.out, simplify=not args.no_simplify, model_keys=keys, keep_tmp=args.keep_tmp)
    print("\nDone.")


if __name__ == "__main__":
    main()
