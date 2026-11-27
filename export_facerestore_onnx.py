"""
export_facerestore_onnx.py -- Export GFPGANv1.4 to ONNX for use with
                              `FaceEnhancementFilter`.

Usage:
    python export_facerestore_onnx.py [--out DIR] [--models all|gfpgan] [--no-simplify]

Requirements:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install onnx onnxsim numpy

The script will:
  1. Download the GFPGAN v1.4 architecture from TencentARC/GFPGAN on GitHub.
  2. Download GFPGANv1.4.pth (~350 MB) from the GitHub release.
  3. Wrap the model so that normalisation (BGR pixel values in [0, 255] -> [-1, 1]) is
     included inside the ONNX graph, matching the `FaceEnhancementFilter` contract exactly.
  4. Export a fixed-shape [1, 3, 512, 512] ONNX file, optionally simplified with onnxsim.

`FaceEnhancementFilter` restoration model contract (must match exactly):
  Input:  [1, 3, 512, 512] float32, BGR channels in [-1, 1] range
  Output: same shape, same normalisation

Output files (default: ThirdParty/models/facerestore/ relative to repo root):
    GFPGANv1.4.onnx       GFPGAN v1.4 face restoration (~350 MB)
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
import urllib.request
from pathlib import Path
from typing import Callable

import onnx
import torch
import torch.nn as nn
import torch.onnx

try:
    from onnxsim import simplify as onnxsim_simplify
except ImportError:
    onnxsim_simplify = None


def _inject_basicsr_stubs() -> None:
    """Inject minimal basicsr stubs so the GFPGAN arch files can be
    loaded without the real basicsr package (which is broken on Python 3.13+).
    Only the symbols actually referenced by the arch files are stubbed out."""

    if "basicsr" in sys.modules:
        return

    class _Registry:
        def register(self) -> Callable[[type], type]:
            def decorator(cls: type) -> type:
                return cls

            return decorator

    def _default_init_weights(module_list: object, **kwargs: object) -> None:
        pass

    class _NoOpLogger:
        def info(self, *a: object, **k: object) -> None:
            pass

        def warning(self, *a: object, **k: object) -> None:
            pass

        def debug(self, *a: object, **k: object) -> None:
            pass

        def error(self, *a: object, **k: object) -> None:
            pass

    _noop_logger = _NoOpLogger()

    def _get_root_logger(*args: object, **kwargs: object) -> _NoOpLogger:
        return _noop_logger

    # basicsr.utils.registry
    _utils_registry = types.ModuleType("basicsr.utils.registry")
    _utils_registry.ARCH_REGISTRY = _Registry()

    # basicsr.archs.arch_util
    _archs_arch_util = types.ModuleType("basicsr.archs.arch_util")
    _archs_arch_util.default_init_weights = _default_init_weights

    # basicsr.utils
    _utils = types.ModuleType("basicsr.utils")
    _utils.registry = _utils_registry
    _utils.get_root_logger = _get_root_logger

    # basicsr.archs
    _archs = types.ModuleType("basicsr.archs")
    _archs.arch_util = _archs_arch_util

    # basicsr root
    _basicsr = types.ModuleType("basicsr")
    _basicsr.utils = _utils
    _basicsr.archs = _archs

    sys.modules["basicsr"] = _basicsr
    sys.modules["basicsr.utils"] = _utils
    sys.modules["basicsr.utils.registry"] = _utils_registry
    sys.modules["basicsr.archs"] = _archs
    sys.modules["basicsr.archs.arch_util"] = _archs_arch_util


_inject_basicsr_stubs()

_GFPGAN_ARCH_URL: str = "https://raw.githubusercontent.com/TencentARC/GFPGAN/master/gfpgan/archs/gfpganv1_clean_arch.py"
_GFPGAN_STYLEGAN2_URL: str = "https://raw.githubusercontent.com/TencentARC/GFPGAN/master/gfpgan/archs/stylegan2_clean_arch.py"
_GFPGAN_PTH_URL: str = (
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
)

RESTORE_SIZE: int = 512


def download(url: str, dest: Path, label: str) -> None:
    """Download `url` to `dest`. Prints a done message. Skips silently if `dest` already exists."""
    if dest.exists():
        print(f"  [skip] {label} already downloaded.")
        return
    print(f"  Downloading {label} ...", end=" ", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"done ({dest.stat().st_size // 1_048_576} MB)")


def load_module_from_file(name: str, path: Path) -> types.ModuleType:
    """Dynamically load a Python source file as a module registered under name in `sys.modules`."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class GfpganWrapper(nn.Module):
    """
    Wraps a `GFPGANCleanArch` so the ONNX graph consumes [1, 3, 512, 512] BGR float32
    in the [-1, 1] range and produces output in the same range, matching
    `FaceEnhancementFilter`'s restoration model contract exactly.

    `GFPGANv1.4.generate()` returns `(output, None)`; we take only `output[0]`.
    """

    def __init__(self, gfpgan: nn.Module) -> None:
        """Wrap a `GFPGANv1Clean` instance for single-input ONNX export."""
        super().__init__()
        self.gfpgan = gfpgan

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run GFPGAN and return the restored face tensor, discarding the auxiliary output."""
        output, _ = self.gfpgan(x, return_rgb=False)
        return output


def export_gfpgan(tmp: Path, out_dir: Path, simplify: bool) -> None:
    """Download GFPGANv1.4 weights, build the ONNX wrapper, and write GFPGANv1.4.onnx to `out_dir`."""
    print("\n[GFPGAN v1.4]")
    stylegan2_path = tmp / "stylegan2_clean_arch.py"
    arch_raw_path = tmp / "gfpganv1_clean_arch.py"
    arch_patched_path = tmp / "gfpganv1_clean_arch_patched.py"
    pth_path = tmp / "GFPGANv1.4.pth"
    out_path = out_dir / "GFPGANv1.4.onnx"

    if out_path.exists():
        print("  [skip] GFPGANv1.4.onnx already exists.")
        return

    download(_GFPGAN_STYLEGAN2_URL, stylegan2_path, "stylegan2_clean_arch.py")
    download(_GFPGAN_ARCH_URL, arch_raw_path, "gfpganv1_clean_arch.py")
    download(_GFPGAN_PTH_URL, pth_path, "GFPGANv1.4.pth")

    # Load the stylegan2 dependency first so the patched relative import resolves.
    load_module_from_file("stylegan2_clean_arch", stylegan2_path)

    # gfpganv1_clean_arch.py uses a relative import that fails when loaded
    # outside of its package. Patch it to an absolute import before loading.
    patched_src = arch_raw_path.read_text(encoding="utf-8").replace(
        "from .stylegan2_clean_arch import",
        "from stylegan2_clean_arch import",
    )
    arch_patched_path.write_text(patched_src, encoding="utf-8")

    arch_module = load_module_from_file("gfpganv1_clean_arch", arch_patched_path)
    gfpgan_clean_arch = arch_module.GFPGANv1Clean

    net = gfpgan_clean_arch(
        out_size=RESTORE_SIZE,
        num_style_feat=512,
        channel_multiplier=2,
        decoder_load_path=None,
        fix_decoder=False,
        num_mlp=8,
        input_is_latent=True,
        different_w=True,
        narrow=1,
        sft_half=True,
    )

    state = torch.load(pth_path, map_location="cpu")
    params = state.get("params_ema") or state.get("params") or state
    net.load_state_dict(params, strict=True)
    net.eval()

    model = GfpganWrapper(net)

    dummy = torch.zeros(1, 3, RESTORE_SIZE, RESTORE_SIZE)
    _export(model, dummy, out_path, simplify, input_name="input", output_name="output")


def _export(
    model: nn.Module,
    dummy: torch.Tensor,
    out_path: Path,
    simplify: bool,
    input_name: str,
    output_name: str,
) -> None:
    """Export model to ONNX at out_path, validate it, and optionally simplify with onnxsim."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Exporting ONNX -> {out_path} ...", end=" ", flush=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(out_path),
            opset_version=17,
            dynamo=False,
            input_names=[input_name],
            output_names=[output_name],
            dynamic_axes=None,
        )

    print("done")

    proto = onnx.load(str(out_path))
    onnx.checker.check_model(proto)
    print("  ONNX check passed.")

    if simplify:
        if onnxsim_simplify is not None:
            print("  Simplifying ...", end=" ", flush=True)
            proto_sim, ok = onnxsim_simplify(proto)
            if ok:
                onnx.save(proto_sim, str(out_path))
                print("done")
            else:
                print("skipped (onnxsim returned not-ok)")
        else:
            print("  [skip] onnxsim not installed; skipping simplification.")

    size_mb = out_path.stat().st_size // 1_048_576
    print(f"  Output: {out_path}  ({size_mb} MB)")


def main() -> None:
    """Parse arguments and export the selected face restoration models to ONNX."""
    parser = argparse.ArgumentParser(
        description="Export face restoration models to ONNX"
    )
    parser.add_argument(
        "--models",
        default="all",
        choices=["all", "gfpgan"],
        help="Which models to export (default: all)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output directory (default: ThirdParty/models/facerestore/ relative "
            "to repo root)"
        ),
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip onnxsim simplification pass",
    )
    args = parser.parse_args()

    out_dir = (
        Path(args.out) if args.out else Path(__file__).resolve().parent / "facerestore"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp = Path(__file__).resolve().parent / ".cache" / "facerestore_export"
    tmp.mkdir(parents=True, exist_ok=True)

    simplify = not args.no_simplify

    if args.models in ("all", "gfpgan"):
        export_gfpgan(tmp, out_dir, simplify)

    print(
        "\nDone. Place the output .onnx files in ThirdParty/models/facerestore/ "
        "and select them"
    )
    print("as the Restore Model in the Face Enhancement settings.")


if __name__ == "__main__":
    main()
