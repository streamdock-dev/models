"""
export_bsrgan_onnx.py -- Export BSRGAN weights to ONNX for use with EsrganUpscaler.

Usage:
    py export_bsrgan_onnx.py [--out DIR] [--no-simplify]

Requirements:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install onnx onnxsim numpy onnxruntime

The script will:
  1. Download the RRDBNet architecture source file from GitHub (BasicSR).
  2. Download the .pth weights for each selected model from the BSRGAN GitHub releases.
  3. Load the architecture and export to ONNX at 128x128 (the tile size used by
     EsrganUpscaler), with a dynamic batch axis, optionally simplified with onnxsim.
  4. Strip stale Resize-node shapes from the ONNX graph so DirectML computes correct
     buffer sizes at runtime (same patch logic used by export_esrgan_onnx.py).

Output files (default: ThirdParty/models/bsrgan/ relative to repo root):
    bsrgan-x4.onnx        4x blind-degradation upscaler (RRDBNet, 23 blocks)
    bsrgan-x2.onnx        2x blind-degradation upscaler (RRDBNet, 23 blocks)

EsrganUpscaler usage:
    Models are discovered automatically by ModelCatalog.DiscoverForPipeline(Esrgan).
    Place output files in ThirdParty/models/bsrgan/ and select them in the
    model picker, or set ESRGAN_MODEL_PATH in app.local.cfg to use a specific model.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
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

# BSRGAN's own architecture (no pixel-unshuffle for x2, original ESRGAN key names).
_BSRGAN_NATIVE_ARCH_URL: str = (
    "https://raw.githubusercontent.com/cszn/BSRGAN/main/models/network_rrdbnet.py"
)

_MODELS: dict[str, dict[str, object]] = {
    "x4": {
        "out_name": "bsrgan-x4.onnx",
        "scale": 4,
        "num_block": 23,
        "url": (
            "https://github.com/cszn/KAIR/"
            "releases/download/v1.0/BSRGAN.pth"
        ),
        "pth_name": "BSRGAN.pth",
    },
    "x2": {
        "out_name": "bsrgan-x2.onnx",
        "scale": 2,
        "num_block": 23,
        "url": (
            "https://github.com/cszn/KAIR/"
            "releases/download/v1.0/BSRGANx2.pth"
        ),
        "pth_name": "BSRGANx2.pth",
        "use_native_arch": True,
    },
}

_TILE_SIZE: int = 128


def _inject_arch_stubs() -> None:
    """
    Inject minimal stubs for basicsr registry imports.

    The architecture source file (rrdbnet_arch.py) only needs the ARCH_REGISTRY
    decorator (a no-op pass-through) and two arch_util helpers used during model
    construction. Stubbing these avoids pip-installing the full basicsr package.
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
    mod.__package__ = module_name
    sys.modules[module_name] = mod
    arch_util_stub = sys.modules.get("basicsr.archs.arch_util")
    if arch_util_stub is not None:
        sys.modules.setdefault(f"{module_name}.arch_util", arch_util_stub)
    spec.loader.exec_module(mod)
    return mod


def _remap_bsrgan_keys(state_dict: dict) -> dict:
    """
    Remap BSRGAN checkpoint keys from the original ESRGAN naming convention to
    the BasicSR RRDBNet naming convention used by rrdbnet_arch.py.

    Original ESRGAN naming:      BasicSR naming:
      RRDB_trunk.X.RDB1.*          body.X.rdb1.*
      RRDB_trunk.X.RDB2.*          body.X.rdb2.*
      RRDB_trunk.X.RDB3.*          body.X.rdb3.*
      trunk_conv.*                 conv_body.*
      upconv1.*                    conv_up1.*
      upconv2.*                    conv_up2.*
      HRconv.*                     conv_hr.*
    """
    remapped: dict = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("RRDB_trunk."):
            nk = "body." + nk[len("RRDB_trunk."):]
            nk = nk.replace(".RDB1.", ".rdb1.")
            nk = nk.replace(".RDB2.", ".rdb2.")
            nk = nk.replace(".RDB3.", ".rdb3.")
        elif nk.startswith("trunk_conv."):
            nk = "conv_body." + nk[len("trunk_conv."):]
        elif nk.startswith("upconv1."):
            nk = "conv_up1." + nk[len("upconv1."):]
        elif nk.startswith("upconv2."):
            nk = "conv_up2." + nk[len("upconv2."):]
        elif nk.startswith("HRconv."):
            nk = "conv_hr." + nk[len("HRconv."):]
        remapped[nk] = v
    return remapped


def _load_rrdbnet(
    arch_path: Path,
    weights_path: Path,
    scale: int,
    num_block: int,
) -> nn.Module:
    """
    Instantiate and load a BSRGAN RRDBNet model from a .pth checkpoint.

    BSRGAN checkpoints use the original ESRGAN key naming convention, which
    differs from BasicSR's rrdbnet_arch.py. Keys are remapped before loading.
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
    if isinstance(raw, dict):
        raw = _remap_bsrgan_keys(raw)
    model.load_state_dict(raw, strict=True)
    model.eval()
    return model


def _load_bsrgan_native_rrdbnet(
    arch_path: Path,
    weights_path: Path,
    scale: int,
    num_block: int,
) -> nn.Module:
    """
    Load a BSRGAN model using the project's own architecture file.

    The BSRGAN native arch (network_rrdbnet.py) uses original ESRGAN key names
    (RRDB_trunk, trunk_conv, upconv1/2, HRconv) that match the checkpoint
    directly -- no remapping needed. For scale=2 it omits upconv2, avoiding the
    pixel-unshuffle mismatch present in BasicSR's x2 RRDBNet.
    """
    arch_mod = _load_arch_module(arch_path, "_bsrgan_native_arch")
    with contextlib.redirect_stdout(io.StringIO()):
        model: nn.Module = arch_mod.RRDBNet(
            in_nc=3,
            out_nc=3,
            nf=64,
            nb=num_block,
            gc=32,
            sf=scale,
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
    """Save an ONNX proto as a single self-contained file with all weights inline."""
    for t in proto.graph.initializer:
        t.ClearField("data_location")
        del t.external_data[:]
    onnx.save(proto, str(path), save_as_external_data=False)


def apply_batch_patch(model_proto: onnx.ModelProto) -> onnx.ModelProto:
    """
    Make the batch dimension dynamic and convert size-based Resize nodes to scale-based
    ones so that DirectML computes correct buffer sizes at runtime for any batch size N.
    """
    model_proto.graph.input[0].type.tensor_type.shape.dim[0].ClearField("dim_value")
    model_proto.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "N"
    model_proto.graph.output[0].type.tensor_type.shape.dim[0].ClearField("dim_value")
    model_proto.graph.output[0].type.tensor_type.shape.dim[0].dim_param = "N"

    del model_proto.graph.value_info[:]

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
    native_arch_path: Path,
    simplify: bool,
) -> None:
    """
    Export a single BSRGAN model to ONNX.

    Downloads the .pth weights if not already present, loads the RRDBNet
    architecture, exports to ONNX, applies the dynamic-batch patch, optionally
    simplifies, and runs a CPU sanity check.
    """
    cfg: dict[str, object] = _MODELS[model_key]
    out_path: Path = out_dir / str(cfg["out_name"])

    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists.")
        return

    pth_path: Path = tmp_dir / str(cfg["pth_name"])
    download(str(cfg["url"]), pth_path, str(cfg["pth_name"]))

    scale: int = int(cfg["scale"])
    num_block: int = int(cfg["num_block"])
    use_native: bool = bool(cfg.get("use_native_arch", False))
    print(f"  Loading RRDBNet ({model_key}, {scale}x) ...")
    if use_native:
        model: nn.Module = _load_bsrgan_native_rrdbnet(
            native_arch_path,
            pth_path,
            scale,
            num_block,
        )
    else:
        model = _load_rrdbnet(
            rrdbnet_arch_path,
            pth_path,
            scale,
            num_block,
        )

    dummy: torch.Tensor = torch.zeros(1, 3, _TILE_SIZE, _TILE_SIZE)

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
    expected: int = _TILE_SIZE * scale
    assert out_arr.shape == (1, 3, expected, expected), (
        f"Unexpected output shape {out_arr.shape}, "
        f"expected (1, 3, {expected}, {expected})"
    )
    print(
        f"  Sanity check passed: input={tuple(dummy.shape)} -> output={out_arr.shape}"
    )
    print(f"\nDone. Model written to: {out_path}")


def main() -> None:
    """Parse arguments and export the selected BSRGAN models to ONNX."""
    default_out: Path = Path(__file__).resolve().parent / "bsrgan"

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Export BSRGAN weights to ONNX for EsrganUpscaler."
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
            "x4, x2, or 'all' (default: all)."
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

    tmp_dir: Path = out_dir / "_bsrgan_export_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    _inject_arch_stubs()

    rrdbnet_arch_path: Path = tmp_dir / "rrdbnet_arch.py"
    native_arch_path: Path = tmp_dir / "bsrgan_network_rrdbnet.py"

    print("Step 1: downloading architecture source file(s) ...")
    download(_RRDBNET_ARCH_URL, rrdbnet_arch_path, "rrdbnet_arch.py")
    if any(_MODELS[k].get("use_native_arch") for k in keys):
        download(_BSRGAN_NATIVE_ARCH_URL, native_arch_path, "bsrgan_network_rrdbnet.py")

    for step, key in enumerate(keys, start=2):
        print(f"\nStep {step}: exporting {key} ...")
        export_model(
            key,
            out_dir,
            tmp_dir,
            rrdbnet_arch_path,
            native_arch_path,
            not args.no_simplify,
        )

    if not args.keep_tmp and tmp_dir.exists():
        shutil.rmtree(tmp_dir)
        print(f"Cleaned up temporary directory: {tmp_dir}")


if __name__ == "__main__":
    main()
