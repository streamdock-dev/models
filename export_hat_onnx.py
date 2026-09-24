"""
export_hat_onnx.py -- Export HAT (Hybrid Attention Transformer) real-world SR weights to ONNX
for use with HatUpscaler.

Usage:
    python export_hat_onnx.py [--out DIR] [--models all|x4|x4sharper] [--no-simplify]

Requirements:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install onnx onnxruntime onnxsim einops gdown

Weights:
    HAT real-world SR weights are hosted on Google Drive (Apache 2.0 license):
    https://drive.google.com/drive/folders/1HpmReFfoUqUbnAOQ7rvOeNU3uf_m69w0

    The script will attempt to download the selected .pth files automatically using gdown.
    If the automatic download fails (e.g. due to download limits), manually download the
    file from the Google Drive folder above and place it in:
        <out>/_hat_export_tmp/Real_HAT_GAN_SRx4.pth     (for --models x4)
        <out>/_hat_export_tmp/Real_HAT_GAN_SRx4_sharper.pth  (for --models x4sharper)
    Then re-run the script -- it will skip the download and use the local file.

The script will:
  1. Download hat_arch.py from XPixelGroup/HAT on GitHub.
  2. Patch out the basicsr registry decorator and provide lightweight stubs for
     to_2tuple and trunc_normal_ so the full basicsr package is not required.
  3. Download the selected real-world SR weights (via gdown from Google Drive).
  4. Precompute the attention mask for the fixed tile size and patch forward_features
     to use the baked-in mask, eliminating dynamic x_size-dependent tensor creation.
  5. Export to ONNX at the fixed tile size with a batch dynamic axis.
  6. Optionally simplify with onnxsim and sanity-check via CPUExecutionProvider.

Output files (default: ThirdParty/models/hat/ relative to repo root):
    hat-real-sr-x4.onnx          4x real-world SR (Real_HAT_GAN_SRx4, balanced quality)
    hat-real-sr-x4-sharper.onnx  4x real-world SR (Real_HAT_GAN_SRx4_sharper, higher perceptual quality)

HatUpscaler usage:
    Models are discovered automatically by ModelCatalog.DiscoverForPipeline(Hat).
    Place output files in ThirdParty/models/hat/ and select the desired
    variant in the model picker.

Notes on ONNX export:
    HAT uses a window_size of 16 and an overlapping cross-attention block (OCAB) with
    overlap_ratio=0.5. The attention mask for cyclic-shifted windows and the relative
    position index buffers are all fixed tensors at a given tile size. forward_features
    is patched to use precomputed buffers rather than calling calculate_mask at inference
    time, eliminating dynamic control flow. All operations (including einops rearrange
    and F.unfold in OCAB) reduce to concrete tensor ops at trace time.

    TILE_SIZE=64 is required: window_size=16, so tile must be a multiple of 16.
    64 = 4 * 16 is the safe choice for mid-range VRAM budgets (~4 GB).

Source: https://github.com/XPixelGroup/HAT (Apache 2.0)
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import textwrap
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

try:
    import gdown

    _has_gdown: bool = True
except ImportError:
    _has_gdown = False

_HAT_RAW = (
    "https://raw.githubusercontent.com/XPixelGroup/HAT/main"
)
_ARCH_URL = f"{_HAT_RAW}/hat/archs/hat_arch.py"

# Google Drive folder: https://drive.google.com/drive/folders/1HpmReFfoUqUbnAOQ7rvOeNU3uf_m69w0
_GDRIVE_FOLDER_ID = "1HpmReFfoUqUbnAOQ7rvOeNU3uf_m69w0"

TILE_SIZE = 64

# Model registry: key -> (gdrive_file_id_or_none, fallback_filename, output_filename, model_kwargs)
# Google Drive file IDs are resolved from the shared folder at runtime via gdown folder listing.
# If the IDs below are stale, use manual download as documented in the module docstring.
_MODELS: dict[str, tuple[str, str, dict[str, Any]]] = {
    "x2": (
        "HAT_SRx2.pth",
        "hat-sr-x2.onnx",
        {
            "upscale": 2,
            "in_chans": 3,
            "img_size": 64,
            "window_size": 16,
            "compress_ratio": 3,
            "squeeze_factor": 30,
            "conv_scale": 0.01,
            "overlap_ratio": 0.5,
            "img_range": 1.0,
            "depths": [6, 6, 6, 6, 6, 6],
            "embed_dim": 180,
            "num_heads": [6, 6, 6, 6, 6, 6],
            "mlp_ratio": 2,
            "upsampler": "pixelshuffle",
            "resi_connection": "1conv",
            "num_feat": 64,
        },
    ),
    "x4": (
        "Real_HAT_GAN_SRx4.pth",
        "hat-real-sr-x4.onnx",
        {
            "upscale": 4,
            "in_chans": 3,
            "img_size": 64,
            "window_size": 16,
            "compress_ratio": 3,
            "squeeze_factor": 30,
            "conv_scale": 0.01,
            "overlap_ratio": 0.5,
            "img_range": 1.0,
            "depths": [6, 6, 6, 6, 6, 6],
            "embed_dim": 180,
            "num_heads": [6, 6, 6, 6, 6, 6],
            "mlp_ratio": 2,
            "upsampler": "pixelshuffle",
            "resi_connection": "1conv",
            "num_feat": 64,
        },
    ),
    "x4sharper": (
        "Real_HAT_GAN_sharper.pth",
        "hat-real-sr-x4-sharper.onnx",
        {
            "upscale": 4,
            "in_chans": 3,
            "img_size": 64,
            "window_size": 16,
            "compress_ratio": 3,
            "squeeze_factor": 30,
            "conv_scale": 0.01,
            "overlap_ratio": 0.5,
            "img_range": 1.0,
            "depths": [6, 6, 6, 6, 6, 6],
            "embed_dim": 180,
            "num_heads": [6, 6, 6, 6, 6, 6],
            "mlp_ratio": 2,
            "upsampler": "pixelshuffle",
            "resi_connection": "1conv",
            "num_feat": 64,
        },
    ),
}


def download_arch(arch_path: Path) -> None:
    """Download hat_arch.py from XPixelGroup/HAT. Skips if already present."""
    if arch_path.exists():
        print("  [skip] hat_arch.py already downloaded.")
        return
    print("  Downloading hat_arch.py ...", end=" ", flush=True)
    urllib.request.urlretrieve(_ARCH_URL, arch_path)
    print(f"done ({arch_path.stat().st_size // 1024} KB)")


def download_weights(weights_filename: str, dest: Path) -> None:
    """
    Attempt to download a weights file from the Google Drive model zoo via gdown.
    Falls back to manual-download instructions if gdown is not installed or fails.
    """
    if dest.exists():
        print(f"  [skip] {weights_filename} already downloaded.")
        return

    if not _has_gdown:
        _print_manual_download(weights_filename, dest)
        sys.exit(1)

    print(f"  Downloading {weights_filename} from Google Drive ...")
    print(f"  Folder ID: {_GDRIVE_FOLDER_ID}")
    try:
        # Download the entire folder and look for the target file.
        # gdown will create a subdirectory in dest.parent named after the folder.
        gdown.download_folder(
            id=_GDRIVE_FOLDER_ID,
            output=str(dest.parent),
            quiet=False,
            use_cookies=False,
        )
        # gdown places files in dest.parent/<folder_name>/*.pth
        # Search for the file in any subdirectory created.
        found = list(dest.parent.rglob(weights_filename))
        if not found:
            raise FileNotFoundError(f"{weights_filename} not found after folder download.")
        # Move/rename to expected location.
        found[0].rename(dest)
        print(f"  Placed at {dest}")
    except Exception as exc:
        print(f"  gdown failed: {exc}")
        _print_manual_download(weights_filename, dest)
        sys.exit(1)


def _print_manual_download(weights_filename: str, dest: Path) -> None:
    print(
        textwrap.dedent(
            f"""
    Manual download required:
      1. Open the HAT model zoo on Google Drive:
         https://drive.google.com/drive/folders/{_GDRIVE_FOLDER_ID}
      2. Download '{weights_filename}' from that folder.
      3. Place the file at:
         {dest}
      4. Re-run this script.
    """
        )
    )


def patch_and_load_arch(arch_path: Path) -> types.ModuleType:
    """
    Load hat_arch.py, patching out the basicsr dependency so the full package
    is not required. Provides lightweight stubs for to_2tuple and trunc_normal_.
    """
    source = arch_path.read_text(encoding="utf-8")

    # Remove basicsr imports -- replaced by stubs below.
    source = source.replace(
        "from basicsr.utils.registry import ARCH_REGISTRY\n", ""
    )
    source = source.replace(
        "from basicsr.archs.arch_util import to_2tuple, trunc_normal_\n", ""
    )
    # Remove the registry decorator so HAT can be instantiated as a plain nn.Module.
    source = source.replace("@ARCH_REGISTRY.register()\n", "")

    stubs = textwrap.dedent(
        """
        import math as _math

        def to_2tuple(x):
            if isinstance(x, (list, tuple)):
                return tuple(x)
            return (x, x)

        def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
            import torch.nn.init as _init
            with torch.no_grad():
                _init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)
            return tensor

        """
    )
    source = stubs + source

    tmp_path = arch_path.parent / "_hat_arch_patched.py"
    tmp_path.write_text(source, encoding="utf-8")

    spec = importlib.util.spec_from_file_location("hat_arch", tmp_path)
    if spec is None:
        raise RuntimeError(f"Failed to load hat_arch from {tmp_path}")
    mod = importlib.util.module_from_spec(spec)

    try:
        import einops
    except ImportError:
        print("  WARNING: 'einops' not found. Install with: pip install einops")

    if spec.loader is not None:
        spec.loader.exec_module(mod)
    return mod


def load_model(arch_mod: types.ModuleType, weights_path: Path, model_kwargs: dict[str, Any]) -> nn.Module:
    """Instantiate HAT, load weights, return in eval mode."""
    model_cls = getattr(arch_mod, "HAT", None)
    if model_cls is None:
        raise RuntimeError(
            "HAT class not found in hat_arch.py. "
            "Check the XPixelGroup/HAT architecture file."
        )

    model: nn.Module = model_cls(**model_kwargs)
    state = torch.load(str(weights_path), map_location="cpu")

    # Official HAT weights may store params under 'params', 'params_ema', or be a raw state dict.
    if isinstance(state, dict):
        if "params_ema" in state:
            state = state["params_ema"]
        elif "params" in state:
            state = state["params"]

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


class HatWrapper(nn.Module):
    """
    Traceable wrapper for a HAT model at a fixed tile size.

    Accepts [1, 3, TILE, TILE] float32 RGB input (values 0-1) and returns
    [1, 3, TILE*N, TILE*N] float32 output, where N is the model upscale factor.

    HAT's forward_features calls calculate_mask(x_size) at inference time, which
    creates a zeros tensor of shape (1, h, w, 1) from dynamic dimensions. This wrapper
    precomputes the attention mask for the fixed tile size and patches forward_features
    to use the baked-in constant, eliminating dynamic tensor creation during tracing.

    The relative position index buffers (rpi_sa, rpi_oca) are already registered
    buffers in the model and will be embedded as constants in the ONNX graph.
    """

    def __init__(self, model: nn.Module, tile_size: int) -> None:
        """Wrap a loaded HAT model, precomputing the fixed-size attention mask."""
        super().__init__()
        self.model = model

        # Cast to Any for attribute access on the dynamically-loaded HAT architecture.
        _model_any: Any = model
        _wrapper_any: Any = self

        # Precompute the attention mask for the fixed tile and register as a buffer
        # so it is embedded as a constant in the ONNX graph.
        attn_mask = _model_any.calculate_mask((tile_size, tile_size))
        self.register_buffer("_attn_mask", attn_mask)
        self._tile_size = tile_size

        # Patch forward_features to use the precomputed mask instead of calling
        # calculate_mask dynamically. This is the core export patch for HAT.
        def patched_forward_features(x: torch.Tensor) -> torch.Tensor:
            x_size = (_wrapper_any._tile_size, _wrapper_any._tile_size)
            params = {
                "attn_mask": _wrapper_any._attn_mask,
                "rpi_sa": _model_any.relative_position_index_SA,
                "rpi_oca": _model_any.relative_position_index_OCA,
            }
            x = _model_any.patch_embed(x)
            if _model_any.ape:
                x = x + _model_any.absolute_pos_embed
            x = _model_any.pos_drop(x)
            for layer in _model_any.layers:
                x = layer(x, x_size, params)
            x = _model_any.norm(x)
            x = _model_any.patch_unembed(x, x_size)
            return x

        setattr(model, "forward_features", patched_forward_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run upscale and clamp output to [0, 1]."""
        return self.model(x).clamp(0.0, 1.0)


def export(out_dir: Path, simplify: bool, model_keys: list[str], keep_tmp: bool = False) -> None:
    """Download weights and architecture, build ONNX models for each selected key."""
    tmp_dir = out_dir / "_hat_export_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    arch_path = tmp_dir / "hat_arch.py"
    print("Step 1: downloading HAT architecture ...")
    download_arch(arch_path)

    print("Step 2: loading and patching architecture module ...")
    arch_mod = patch_and_load_arch(arch_path)

    for model_key in model_keys:
        weights_filename, out_name, model_kwargs = _MODELS[model_key]
        weights_path = tmp_dir / weights_filename
        out_path = out_dir / out_name

        if out_path.exists():
            print(f"  [skip] {out_name} already exists.")
            continue

        print(f"\nStep 3 [{model_key}]: downloading weights ...")
        download_weights(weights_filename, weights_path)

        _export_single(arch_mod, weights_path, out_path, model_key, model_kwargs, simplify)

    if not keep_tmp and tmp_dir.exists():
        shutil.rmtree(tmp_dir)
        print(f"Cleaned up temporary directory: {tmp_dir}")


def _export_single(
    arch_mod: types.ModuleType,
    weights_path: Path,
    out_path: Path,
    model_key: str,
    model_kwargs: dict[str, Any],
    simplify: bool,
) -> None:
    """Load, wrap, export, and sanity-check a single HAT variant."""
    scale: int = model_kwargs["upscale"]
    window_size: int = model_kwargs["window_size"]

    if TILE_SIZE % window_size != 0:
        raise ValueError(
            f"TILE_SIZE={TILE_SIZE} must be a multiple of window_size={window_size}."
        )

    print(f"\nStep 4 [{model_key}]: loading model (scale={scale}x, window={window_size}) ...")
    model = load_model(arch_mod, weights_path, model_kwargs)
    wrapper = HatWrapper(model, TILE_SIZE)
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
    result = sess.run(None, {"input": inp})
    out_h, out_w = result[0].shape[2], result[0].shape[3]
    assert out_h == TILE_SIZE * scale, f"Expected {TILE_SIZE * scale}, got {out_h}"
    assert out_w == TILE_SIZE * scale, f"Expected {TILE_SIZE * scale}, got {out_w}"
    sz_mb = out_path.stat().st_size / 1_048_576
    print(
        f"  OK: output shape {result[0].shape}, model size {sz_mb:.1f} MB"
    )


def main() -> None:
    """Parse arguments and run the export pipeline."""
    # Resolve default output directory relative to this script's location.
    script_dir = Path(__file__).parent
    default_out = script_dir / "hat"

    parser = argparse.ArgumentParser(
        description="Export HAT real-world SR weights to ONNX."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help=f"Output directory for .onnx files (default: {default_out})",
    )
    parser.add_argument(
        "--models",
        default="x2,x4",
        help=(
            "Comma-separated model keys to export: x2, x4, x4sharper, all "
            "(default: x2,x4)"
        ),
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip onnxsim graph simplification.",
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
            parser.error(f"Unknown model key(s): {unknown}. Choose from: {list(_MODELS)}")

    args.out.mkdir(parents=True, exist_ok=True)
    export(args.out, simplify=not args.no_simplify, model_keys=model_keys, keep_tmp=args.keep_tmp)
    print("\nDone.")


if __name__ == "__main__":
    main()
