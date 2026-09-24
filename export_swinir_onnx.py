"""
export_swinir_onnx.py -- Export SwinIR real-world SR weights to ONNX for use with SwinIrUpscaler.

Usage:
    python export_swinir_onnx.py [--out DIR] [--models all|x2|x4] [--no-simplify]

Requirements:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install onnx onnxruntime onnxsim timm

The script will:
  1. Download the SwinIR architecture (models/network_swinir.py) from JingyunLiang/SwinIR.
  2. Download the real-world SR weights for the selected scale variants.
  3. Wrap each model into a fixed-tile [1, 3, TILE, TILE] -> [1, 3, TILE*N, TILE*N] module.
     The forward pass pads to window_size multiples internally; with TILE_SIZE=128 and
     window_size=8 no extra padding is needed (128 / 8 = 16 exactly).
  4. Export to ONNX at the fixed tile size, baking in all attention masks.
  5. Optionally simplify with onnxsim and sanity-check via CPUExecutionProvider.

Output files (default: ThirdParty/models/swinir/ relative to repo root):
    swinir-real-sr-x2.onnx    2x real-world SR (SwinIR-M, BSRGAN degradation)
    swinir-real-sr-x4.onnx    4x real-world SR (SwinIR-L, BSRGAN degradation + more filters)

SwinIrUpscaler usage:
    Models are discovered automatically by ModelCatalog.DiscoverForPipeline(SwinIr).
    Place output files in ThirdParty/models/swinir/ and select the desired
    variant in the model picker.

Notes on ONNX export:
    SwinIR uses a window-based attention mechanism. The relative_position_bias_table is a
    learned parameter embedded in the ONNX graph. The attention_mask for cyclic-shifted
    windows depends on H and W but is also embedded at trace time since we export at a
    fixed tile size. No dynamic control flow remains in the exported graph.

    The SwinIR forward() calls check_image_size() to pad H/W to window_size multiples.
    With TILE_SIZE=128 and window_size=8 this is a no-op. The final crop slice is
    traced as a concrete slice since H and W are constants at trace time.

Source: https://github.com/JingyunLiang/SwinIR (Apache 2.0)
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import types
import urllib.request
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.onnx

try:
    from onnxsim import simplify as onnxsim_simplify
except ImportError:
    onnxsim_simplify = None

_SWINIR_RAW = (
    "https://raw.githubusercontent.com/JingyunLiang/SwinIR/main"
)
_ARCH_URL = f"{_SWINIR_RAW}/models/network_swinir.py"

_RELEASES_BASE = (
    "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0"
)

TILE_SIZE = 128

# Model registry: key -> (weights URL, output filename, model_kwargs)
_MODELS: dict[str, tuple[str, str, dict[str, Any]]] = {
    "x2": (
        f"{_RELEASES_BASE}/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth",
        "swinir-real-sr-x2.onnx",
        {
            "upscale": 2,
            "in_chans": 3,
            "img_size": 64,
            "window_size": 8,
            "img_range": 1.0,
            "depths": [6, 6, 6, 6, 6, 6],
            "embed_dim": 180,
            "num_heads": [6, 6, 6, 6, 6, 6],
            "mlp_ratio": 2,
            "upsampler": "nearest+conv",
            "resi_connection": "1conv",
        },
    ),
    "x4": (
        f"{_RELEASES_BASE}/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
        "swinir-real-sr-x4.onnx",
        {
            "upscale": 4,
            "in_chans": 3,
            "img_size": 64,
            "window_size": 8,
            "img_range": 1.0,
            "depths": [6, 6, 6, 6, 6, 6, 6, 6, 6],
            "embed_dim": 240,
            "num_heads": [8, 8, 8, 8, 8, 8, 8, 8, 8],
            "mlp_ratio": 2,
            "upsampler": "nearest+conv",
            "resi_connection": "3conv",
        },
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


def load_swinir_arch(arch_path: Path) -> types.ModuleType:
    """
    Dynamically load the network_swinir.py architecture module and return the SwinIR class.
    """
    spec = importlib.util.spec_from_file_location("network_swinir", arch_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)

    # network_swinir.py depends on timm for DropPath; ensure it is importable.
    try:
        import timm
    except ImportError:
        print(
            "  WARNING: 'timm' package not found. Install with: pip install timm"
        )

    spec.loader.exec_module(mod)
    return mod


class SwinIrWrapper(nn.Module):
    """
    Traceable wrapper for a SwinIR model at a fixed tile size.

    Accepts [1, 3, TILE, TILE] float32 RGB input (values 0-1) and returns
    [1, 3, TILE*N, TILE*N] float32 output, where N is the model upscale factor.

    SwinIR's forward() handles internal padding to window_size multiples via
    check_image_size(). With TILE_SIZE=128 and window_size=8, 128 % 8 == 0 so
    no padding occurs. The output slice x[:, :, :H*scale, :W*scale] is constant
    at trace time because H and W are known from the fixed-size input.
    """

    def __init__(self, model: nn.Module) -> None:
        """Wrap a loaded SwinIR model instance."""
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run upscale and clamp output to [0, 1]."""
        return self.model(x).clamp(0.0, 1.0)


def load_model(arch_mod: Any, weights_path: Path, model_kwargs: dict[str, Any]) -> nn.Module:
    """Instantiate SwinIR, load weights, return in eval mode."""
    model_cls = getattr(arch_mod, "SwinIR", None)
    if model_cls is None:
        raise RuntimeError(
            "SwinIR class not found in network_swinir.py. "
            "Check the JingyunLiang/SwinIR architecture file."
        )

    model: nn.Module = model_cls(**model_kwargs)
    state = torch.load(str(weights_path), map_location="cpu")

    # Official SwinIR weights store params under 'params' or 'params_ema'.
    if isinstance(state, dict):
        if "params_ema" in state:
            state = state["params_ema"]
        elif "params" in state:
            state = state["params"]

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def export(out_dir: Path, simplify: bool, model_keys: list[str], keep_tmp: bool = False) -> None:
    """Download weights, build ONNX models for each selected key, and write them to out_dir."""
    tmp_dir = out_dir / "_swinir_export_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    arch_path = tmp_dir / "network_swinir.py"
    print("Step 1: downloading SwinIR architecture ...")
    download(_ARCH_URL, arch_path, "network_swinir.py")

    print("Step 2: loading architecture module ...")
    arch_mod = load_swinir_arch(arch_path)

    for model_key in model_keys:
        weights_url, out_name, model_kwargs = _MODELS[model_key]
        weights_path = tmp_dir / f"swinir-{model_key}.pth"
        out_path = out_dir / out_name

        if out_path.exists():
            print(f"  [skip] {out_name} already exists.")
            continue

        print(f"\nStep 3 [{model_key}]: downloading weights ...")
        download(weights_url, weights_path, f"swinir-{model_key}.pth")

        _export_single(arch_mod, weights_path, out_path, model_key, model_kwargs, simplify)

    if not keep_tmp and tmp_dir.exists():
        shutil.rmtree(tmp_dir)
        print(f"Cleaned up temporary directory: {tmp_dir}")


def _export_single(
    arch_mod: Any,
    weights_path: Path,
    out_path: Path,
    model_key: str,
    model_kwargs: dict[str, Any],
    simplify: bool,
) -> None:
    """Load, wrap, export, and sanity-check a single SwinIR variant."""
    scale: int = model_kwargs["upscale"]

    print(f"\nStep 4 [{model_key}]: loading model (scale={scale}x) ...")
    model = load_model(arch_mod, weights_path, model_kwargs)
    wrapper = SwinIrWrapper(model)
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
    expected_h = TILE_SIZE * scale
    expected_w = TILE_SIZE * scale
    assert out.shape == (1, 3, expected_h, expected_w), (
        f"Unexpected output shape {out.shape}, expected (1, 3, {expected_h}, {expected_w})"
    )
    print(f"  Output shape: {out.shape} -- OK")
    print(f"  Saved: {out_path} ({out_path.stat().st_size // 1024} KB)")


def main() -> None:
    """Parse arguments and run export."""
    here = Path(__file__).resolve().parent
    default_out = here.parent.parent / "ThirdParty" / "models" / "swinir"

    parser = argparse.ArgumentParser(
        description="Export SwinIR real-world SR weights to ONNX for use with SwinIrUpscaler."
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
            "Which scale variants to export. "
            "Choices: all | x2 | x4 "
            "(comma-separated for multiple, e.g. x2,x4)"
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
        unknown = [k for k in keys if k not in all_keys]
        if unknown:
            parser.error(f"Unknown model keys: {unknown}. Valid keys: {all_keys}")

    args.out.mkdir(parents=True, exist_ok=True)
    export(args.out, simplify=not args.no_simplify, model_keys=keys, keep_tmp=args.keep_tmp)
    print("\nDone.")


if __name__ == "__main__":
    main()
