"""
export_esrgan_onnx.py -- Export Real-ESRGAN weights to ONNX for use with EsrganUpscaler.

Usage:
    py export_esrgan_onnx.py [--out DIR] [--models all|x2plus|x4plus|animevideov3] [--tile-size N] [--no-simplify]

Requirements:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install onnx onnxsim numpy onnxruntime

The script will:
  1. Download the RRDBNet and SRVGGNetCompact architecture source files from GitHub.
  2. Download the .pth weights for each selected model from Real-ESRGAN GitHub releases.
  3. Load each architecture and export to ONNX at the specified tile size (default: 128x128),
     with a dynamic batch axis, optionally simplified with onnxsim.
  4. Strip stale Resize-node shapes from the ONNX graph so DirectML computes correct
     buffer sizes at runtime (incorporates the logic from patch_esrgan_resize.py).

Output files (default: ThirdParty/models/esrgan/ relative to repo root):
    realesrgan-x2plus.onnx             2x general-purpose upscaler (tile=128, default)
    realesrgan-x4plus.onnx             4x general-purpose upscaler (tile=128, default)
    realesr-animevideov3.onnx          4x anime/animation upscaler (tile=128, default)
    realesrgan-x2plus-tile64.onnx      2x with 64px tiles (--tile-size 64)
    realesrgan-x4plus-tile64.onnx      4x with 64px tiles (--tile-size 64)

Non-128 tile sizes get a -tileN suffix in the output filename. To use a custom-tile model
in EsrganUpscaler, pass TileSize in EsrganUpscalerOptions matching the exported tile size.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import types
import urllib.request
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.onnx
from onnx import numpy_helper, shape_inference

try:
    from onnxsim import simplify as onnxsim_simplify
except ImportError:
    onnxsim_simplify = None

_RRDBNET_ARCH_URL: str = (
    "https://raw.githubusercontent.com/XPixelGroup/"
    "BasicSR/master/basicsr/archs/rrdbnet_arch.py"
)
_SRVGG_ARCH_URL: str = (
    "https://raw.githubusercontent.com/xinntao/"
    "Real-ESRGAN/master/realesrgan/archs/srvgg_arch.py"
)

_MODELS: dict[str, dict[str, object]] = {
    "x2plus": {
        "out_name": "realesrgan-x2plus.onnx",
        "arch": "rrdbnet",
        "scale": 2,
        "num_block": 23,
        "url": (
            "https://github.com/xinntao/Real-ESRGAN/"
            "releases/download/v0.2.1/RealESRGAN_x2plus.pth"
        ),
        "pth_name": "RealESRGAN_x2plus.pth",
    },
    "x4plus": {
        "out_name": "realesrgan-x4plus.onnx",
        "arch": "rrdbnet",
        "scale": 4,
        "num_block": 23,
        "url": (
            "https://github.com/xinntao/Real-ESRGAN/"
            "releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        ),
        "pth_name": "RealESRGAN_x4plus.pth",
    },
    "animevideov3": {
        "out_name": "realesr-animevideov3.onnx",
        "arch": "srvgg",
        "scale": 4,
        "num_conv": 16,
        "url": (
            "https://github.com/xinntao/Real-ESRGAN/"
            "releases/download/v0.2.5.0/realesr-animevideov3.pth"
        ),
        "pth_name": "realesr-animevideov3.pth",
    },
}

_TILE_SIZE: int = 128


def _inject_arch_stubs() -> None:
    """
    Inject minimal stubs for basicsr and realesrgan registry imports.

    The architecture source files (rrdbnet_arch.py, srvgg_arch.py) only need the
    ARCH_REGISTRY decorator (a no-op pass-through) and two arch_util helpers used
    during model construction. Stubbing these avoids pip-installing the full
    basicsr and realesrgan packages, which pull in heavy dependencies.
    """

    def _make_registry_module(name: str) -> types.ModuleType:
        """Return a module containing a no-op ARCH_REGISTRY stub."""
        mod = types.ModuleType(name)

        class _Reg:
            """Stub registry: register() returns the decorated class unchanged."""

            def register(self, *args: object, **kwargs: object) -> object:
                """Act as a pass-through class decorator."""

                def deco(cls: type) -> type:
                    """Return cls unchanged."""
                    return cls

                return args[0] if (len(args) == 1 and callable(args[0])) else deco

        mod.ARCH_REGISTRY = _Reg()
        return mod

    def _make_arch_util_module() -> types.ModuleType:
        """Return a module containing the two basicsr arch_util helpers needed at runtime."""
        mod = types.ModuleType("basicsr.archs.arch_util")

        def make_layer(
            basic_block: type, num_basic_block: int, **kwarg: object
        ) -> nn.Sequential:
            """Build a Sequential of num_basic_block instances of basic_block(**kwarg)."""
            return nn.Sequential(
                *[basic_block(**kwarg) for _ in range(num_basic_block)]
            )

        def default_init_weights(
            module_list: object,
            scale: float = 1.0,
            bias_fill: float = 0.0,
            **kwargs: object,
        ) -> None:
            """No-op stub; weight initialisation is not needed for ONNX inference export."""

        # basicsr calls pixel_unshuffle(x, scale=N) but torch uses downscale_factor=N
        def pixel_unshuffle(input: torch.Tensor, scale: int) -> torch.Tensor:
            """Wrapper that maps basicsr's scale= kwarg to torch's downscale_factor=."""
            return torch.nn.functional.pixel_unshuffle(input, scale)

        mod.make_layer = make_layer
        mod.default_init_weights = default_init_weights
        mod.pixel_unshuffle = pixel_unshuffle
        return mod

    stub_map: list[tuple[str, types.ModuleType]] = [
        ("basicsr", types.ModuleType("basicsr")),
        ("basicsr.utils", types.ModuleType("basicsr.utils")),
        ("basicsr.utils.registry", _make_registry_module("basicsr.utils.registry")),
        ("basicsr.archs", types.ModuleType("basicsr.archs")),
        ("basicsr.archs.arch_util", _make_arch_util_module()),
        ("realesrgan", types.ModuleType("realesrgan")),
        ("realesrgan.utils", types.ModuleType("realesrgan.utils")),
        (
            "realesrgan.utils.registry",
            _make_registry_module("realesrgan.utils.registry"),
        ),
    ]
    for name, mod in stub_map:
        sys.modules.setdefault(name, mod)


def download(url: str, dest: Path, label: str) -> None:
    """Download url to dest, printing progress. Skip silently if dest already exists."""
    if dest.exists():
        print(f"  [skip] {label} already downloaded.")
        return

    print(f"  Downloading {label} ...", end=" ", flush=True)

    def _hook(count: int, block: int, total: int) -> None:
        """Report download progress as a percentage."""
        if total > 0:
            pct = min(100, count * block * 100 // total)
            print(f"\r  Downloading {label} ... {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, _hook)
    print(f"\r  Downloaded  {label} ({dest.stat().st_size // 1024} KB)")


def _load_arch_module(arch_path: Path, module_name: str) -> types.ModuleType:
    """
    Dynamically load an architecture source file as a Python module.

    The loaded module is registered in sys.modules under module_name so that
    subsequent imports within the same process see the loaded version.
    Sets __package__ on the module before execution so that relative imports
    (e.g. ``from .arch_util import ...``) resolve correctly via a sibling stub
    pre-registered under ``{module_name}.arch_util``.
    """
    spec = importlib.util.spec_from_file_location(module_name, arch_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # treat this module as its own package so relative imports work
    mod.__package__ = module_name
    sys.modules[module_name] = mod
    # pre-inject arch_util sibling so "from .arch_util import ..." resolves
    arch_util_stub = sys.modules.get("basicsr.archs.arch_util")
    if arch_util_stub is not None:
        sys.modules.setdefault(f"{module_name}.arch_util", arch_util_stub)
    spec.loader.exec_module(mod)
    return mod


def _load_rrdbnet(
    arch_path: Path,
    weights_path: Path,
    scale: int,
    num_block: int,
) -> nn.Module:
    """
    Instantiate and load a Real-ESRGAN RRDBNet model from a .pth checkpoint.

    Handles the three common state-dict key conventions used across Real-ESRGAN
    releases: params_ema (preferred), params, and a flat top-level state dict.
    """
    arch_mod = _load_arch_module(arch_path, "_rrdbnet_arch")
    model: nn.Module = arch_mod.RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=num_block,
        num_grow_ch=32,
        scale=scale,
    )
    raw = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        if "params_ema" in raw:
            raw = raw["params_ema"]
        elif "params" in raw:
            raw = raw["params"]
    model.load_state_dict(raw, strict=True)
    model.eval()
    return model


def _load_srvgg(
    arch_path: Path,
    weights_path: Path,
    scale: int,
    num_conv: int,
) -> nn.Module:
    """
    Instantiate and load a Real-ESRGAN SRVGGNetCompact model from a .pth checkpoint.

    Handles the same state-dict key conventions as _load_rrdbnet.
    """
    arch_mod = _load_arch_module(arch_path, "_srvgg_arch")
    model: nn.Module = arch_mod.SRVGGNetCompact(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_conv=num_conv,
        upscale=scale,
        act_type="prelu",
    )
    raw = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        if "params_ema" in raw:
            raw = raw["params_ema"]
        elif "params" in raw:
            raw = raw["params"]
    model.load_state_dict(raw, strict=True)
    model.eval()
    return model


def _save_inline(proto: onnx.ModelProto, path: Path) -> None:
    """Save an ONNX proto as a single self-contained file with all weights inline.

    Clears any external-data location markers left over from a previous load so
    that onnx.save does not re-split weights into a separate model.data file.
    """
    for t in proto.graph.initializer:
        t.ClearField("data_location")
        del t.external_data[:]
    onnx.save(proto, str(path), save_as_external_data=False)


def apply_batch_patch(model_proto: onnx.ModelProto) -> onnx.ModelProto:
    """
    Make the batch dimension dynamic and convert size-based Resize nodes to scale-based
    ones so that DirectML computes correct buffer sizes at runtime for any batch size N.

    This incorporates the logic from patch_esrgan_resize.py and must be called after
    the initial ONNX export which uses a fixed batch=1 dummy input.
    """
    # Make batch dimension dynamic on I/O tensors.
    model_proto.graph.input[0].type.tensor_type.shape.dim[0].ClearField("dim_value")
    model_proto.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "N"
    model_proto.graph.output[0].type.tensor_type.shape.dim[0].ClearField("dim_value")
    model_proto.graph.output[0].type.tensor_type.shape.dim[0].dim_param = "N"

    # Strip all inferred intermediate shapes so DirectML recomputes them from the
    # actual runtime input rather than using stale batch=1 buffer sizes.
    del model_proto.graph.value_info[:]

    # Run shape inference on a temporary copy to read Resize initializer shapes.
    inferred: onnx.ModelProto = shape_inference.infer_shapes(model_proto)

    shape_map: dict[str, list[int | None]] = {}
    for vi in (
        list(inferred.graph.value_info)
        + list(inferred.graph.input)
        + list(inferred.graph.output)
    ):
        try:
            dims: list[int | None] = [
                (d.dim_value if (d.HasField("dim_value") and d.dim_value > 0) else None)
                for d in vi.type.tensor_type.shape.dim
            ]
            shape_map[vi.name] = dims
        except Exception:
            pass

    init_arrays: dict[str, np.ndarray] = {
        init.name: numpy_helper.to_array(init) for init in inferred.graph.initializer
    }

    new_inits: list[onnx.TensorProto] = []
    patched: int = 0

    for node in model_proto.graph.node:
        if node.op_type != "Resize":
            continue
        if len(node.input) < 4 or not node.input[3]:
            continue
        sizes_name: str = node.input[3]
        if sizes_name not in init_arrays:
            continue
        input_name: str = node.input[0]
        if input_name not in shape_map:
            continue

        sizes: np.ndarray = init_arrays[sizes_name].flatten().astype(np.float64)
        in_shape: list[int | None] = shape_map[input_name]
        if len(in_shape) != len(sizes):
            continue

        scales: list[float] = [
            1.0 if (in_dim is None or in_dim == 0) else float(out_dim) / float(in_dim)
            for in_dim, out_dim in zip(in_shape, sizes)
        ]

        scales_name: str = (
            f"__batched_scales_{node.name.replace('/', '_').strip('_')}__"
        )
        new_inits.append(
            numpy_helper.from_array(
                np.array(scales, dtype=np.float32), name=scales_name
            )
        )

        while len(node.input) < 4:
            node.input.append("")
        node.input[2] = scales_name
        node.input[3] = ""
        patched += 1

    if patched > 0:
        print(f"  Patched {patched} Resize node(s) for dynamic batch.")
    else:
        print("  No Resize nodes required patching.")

    model_proto.graph.initializer.extend(new_inits)
    return model_proto


def export_model(
    model_key: str,
    out_dir: Path,
    tmp_dir: Path,
    rrdbnet_arch_path: Path,
    srvgg_arch_path: Path,
    simplify: bool,
    tile_size: int = 128,
) -> None:
    """
    Export a single Real-ESRGAN model to ONNX.

    Downloads the .pth weights if not already present, loads the appropriate
    architecture, exports to ONNX, applies the dynamic-batch patch, optionally
    simplifies, and runs a CPU sanity check.
    """
    cfg: dict[str, object] = _MODELS[model_key]
    base_name: str = str(cfg["out_name"])
    if tile_size != 128:
        stem, suffix = base_name.rsplit(".", 1)
        base_name = f"{stem}-tile{tile_size}.{suffix}"
    out_path: Path = out_dir / base_name

    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists.")
        return

    pth_path: Path = tmp_dir / str(cfg["pth_name"])
    download(str(cfg["url"]), pth_path, str(cfg["pth_name"]))

    arch: str = str(cfg["arch"])
    scale: int = int(cfg["scale"])

    print(f"  Loading {arch} ({model_key}) ...")
    if arch == "rrdbnet":
        model: nn.Module = _load_rrdbnet(
            rrdbnet_arch_path,
            pth_path,
            scale,
            int(cfg["num_block"]),
        )
    else:
        model = _load_srvgg(
            srvgg_arch_path,
            pth_path,
            scale,
            int(cfg["num_conv"]),
        )

    dummy: torch.Tensor = torch.zeros(1, 3, tile_size, tile_size)

    print(f"  Exporting ONNX to {out_path} ...")
    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        dynamo=False,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "N"}, "output": {0: "N"}},
    )
    print(f"  Exported ({out_path.stat().st_size // 1024} KB)")

    print("  Applying dynamic-batch patch ...")
    proto: onnx.ModelProto = onnx.load(str(out_path), load_external_data=True)
    proto = apply_batch_patch(proto)
    onnx.checker.check_model(proto)
    _save_inline(proto, out_path)

    if simplify:
        if onnxsim_simplify is None:
            print("  [skip] onnxsim not installed, skipping simplification.")
        else:
            print("  Simplifying with onnxsim ...")
            simplified, ok = onnxsim_simplify(
                onnx.load(str(out_path), load_external_data=True)
            )
            if ok:
                _save_inline(simplified, out_path)
                print(f"  Simplified ({out_path.stat().st_size // 1024} KB)")
            else:
                print("  onnxsim failed, keeping patched model.")

    print("  Running sanity check ...")
    sess: ort.InferenceSession = ort.InferenceSession(
        str(out_path), providers=["CPUExecutionProvider"]
    )
    out_arr: np.ndarray = sess.run(None, {"input": dummy.numpy()})[0]
    expected: int = tile_size * scale
    assert out_arr.shape == (1, 3, expected, expected), (
        f"Unexpected output shape {out_arr.shape}, "
        f"expected (1, 3, {expected}, {expected})"
    )
    print(
        f"  Sanity check passed: input={tuple(dummy.shape)} -> output={out_arr.shape}"
    )
    print(f"\nDone. Model written to: {out_path}")


def main() -> None:
    """Parse arguments and export the selected Real-ESRGAN models to ONNX."""
    default_out: Path = Path(__file__).resolve().parent / "esrgan"

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Export Real-ESRGAN weights to ONNX for EsrganUpscaler."
    )
    parser.add_argument(
        "--out",
        default=str(default_out),
        help=f"Output directory (default: {default_out})",
    )
    parser.add_argument(
        "--models",
        default="all",
        help=(
            "Comma-separated list of model keys to export: "
            "x2plus, x4plus, animevideov3, or 'all' (default: all)."
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
    parser.add_argument(
        "--tile-size",
        type=int,
        default=128,
        help=(
            "Tile size (spatial input dimension) for the exported ONNX model. "
            "Default: 128. Non-128 values produce a filename suffix, e.g. realesrgan-x2plus-tile64.onnx."
        ),
    )
    args: argparse.Namespace = parser.parse_args()

    out_dir: Path = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    keys: list[str] = (
        list(_MODELS.keys())
        if args.models == "all"
        else [k.strip() for k in args.models.split(",")]
    )
    invalid: list[str] = [k for k in keys if k not in _MODELS]
    if invalid:
        print(
            f"Unknown model keys: {invalid}. Valid choices: {list(_MODELS.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    tmp_dir: Path = out_dir / "_esrgan_export_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    _inject_arch_stubs()

    rrdbnet_arch_path: Path = tmp_dir / "rrdbnet_arch.py"
    srvgg_arch_path: Path = tmp_dir / "srvgg_arch.py"

    print("Step 1: downloading architecture source files ...")
    download(_RRDBNET_ARCH_URL, rrdbnet_arch_path, "rrdbnet_arch.py")
    download(_SRVGG_ARCH_URL, srvgg_arch_path, "srvgg_arch.py")

    for step, key in enumerate(keys, start=2):
        print(f"\nStep {step}: exporting {key} ...")
        export_model(
            key,
            out_dir,
            tmp_dir,
            rrdbnet_arch_path,
            srvgg_arch_path,
            not args.no_simplify,
            args.tile_size,
        )

    if not args.keep_tmp and tmp_dir.exists():
        shutil.rmtree(tmp_dir)
        print(f"Cleaned up temporary directory: {tmp_dir}")


if __name__ == "__main__":
    main()
