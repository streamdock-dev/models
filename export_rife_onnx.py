"""
export_rife_onnx.py — Export rife4.18 to ONNX for use with RifeInterpolator (OnnxRuntime/DirectML).

Usage:
    python export_rife_onnx.py [--out DIR] [--resolutions all|1080p|720p|...] [--no-simplify]

Requirements:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install onnx onnxsim

The script will:
  1. Clone/download the IFNet_HDv3_v4_18 architecture from vs-rife (GitHub).
  2. Download flownet_v4.18.pkl from the vs-rife model release (~22 MB).
  3. Wrap Head (frame encoder) + IFNet (flow network) into a single
     RifeWrapper(img0, img1, timestep) → output module.
  4. Export one ONNX per resolution, optionally simplified with onnxsim.

Output files (default: ThirdParty/models/rife/ relative to repo root):
    rife4.18_864x480.onnx        covers ≤ 480p  landscape
    rife4.18_1280x736.onnx       covers ≤ 720p  landscape
    rife4.18_1920x1088.onnx      covers ≤ 1080p landscape
    rife4.18_2560x1440.onnx      covers ≤ 1440p landscape
    rife4.18_3840x2176.onnx      covers ≤ 4K    landscape
    rife4.18_480x864.onnx        covers ≤ 480p  portrait
    rife4.18_736x1280.onnx       covers ≤ 720p  portrait
    rife4.18_1088x1920.onnx      covers ≤ 1080p portrait
    rife4.18_1440x2560.onnx      covers ≤ 1440p portrait
    rife4.18_2176x3840.onnx      covers ≤ 4K    portrait

RifeInterpolator usage:
    Set RIFE_MODEL_PATH in app.local.cfg to the file matching your stream resolution,
    e.g. RIFE_MODEL_PATH=ThirdParty/models/rife/rife4.18_1920x1088.onnx
    The model supports a timestep input, so RifeInterpolator.SupportsTimestep will be
    true and RifeVideoStream will use Nx interpolation matched to the display refresh rate.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import tempfile
import textwrap
import urllib.request
from pathlib import Path

import onnx
import torch
import torch.nn as nn
import torch.onnx

try:
    from onnxsim import simplify as onnxsim_simplify
except ImportError:
    onnxsim_simplify = None

# Resolution table: (label, padded_w, padded_h)
# PadUp(x, 32) = ceil(x / 32) * 32
ALL_RESOLUTIONS: list[tuple[str, int, int]] = [
    # landscape
    ("864x480", 864, 480),
    ("1280x736", 1280, 736),
    ("1920x1088", 1920, 1088),
    ("2560x1440", 2560, 1440),
    ("3840x2176", 3840, 2176),
    # portrait
    ("480x864", 480, 864),
    ("736x1280", 736, 1280),
    ("1088x1920", 1088, 1920),
    ("1440x2560", 1440, 2560),
    ("2176x3840", 2176, 3840),
]

MODEL_VERSION = "4.18"
PKL_URL = f"https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v{MODEL_VERSION}.pkl"
VSRIFE_REPO = "https://github.com/HolyWu/vs-rife"

# Architecture source files needed from vs-rife
VSRIFE_FILES = [
    "vsrife/IFNet_HDv3_v4_18.py",
]


def pad_up(v: int, align: int = 32) -> int:
    """Pad v up to the nearest multiple of align (default 32)."""
    return math.ceil(v / align) * align


def download(url: str, dest: Path, label: str) -> None:
    """Download file from the given URL to the destination path, with a simple progress display."""
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


def fetch_vsrife_file(raw_base: str, rel_path: str, dest: Path) -> None:
    """Download a single file from the vs-rife master branch via raw.githubusercontent."""
    if dest.exists():
        return
    url = f"{raw_base}/{rel_path}"
    urllib.request.urlretrieve(url, dest)


def build_wrapper(
    ifnet_module_path: Path, pkl_path: Path, pw: int, ph: int
) -> nn.Module:
    """
    Dynamically import IFNet_HDv3_v4_18, load weights, and return a
    RifeWrapper nn.Module that accepts (img0, img1, timestep) tensors.

    img0, img1  : float32 [1, 3, ph, pw]   — RGB, normalised 0–1, padded
    timestep    : float32 [1, 1, ph, pw]   — each pixel set to the desired t
    output      : float32 [1, 3, ph, pw]   — interpolated frame
    """
    # Dynamically import the IFNet architecture from the downloaded file.
    spec = importlib.util.spec_from_file_location("IFNet_HDv3_v4_18", ifnet_module_path)
    assert spec is not None, "Failed to create module spec from file"
    assert spec.loader is not None, "Module spec has no loader"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ifnet_class = mod.IFNet
    head_class = mod.Head  # frame encoder: Conv2d + ConvTranspose2d block

    # Load weights.
    state_dict = torch.load(pkl_path, map_location="cpu", weights_only=True)
    state_dict = {
        k.replace("module.", ""): v for k, v in state_dict.items() if "module." in k
    }

    flownet = ifnet_class(scale=1.0, ensemble=False)
    flownet.load_state_dict(state_dict, strict=False, assign=True)
    flownet.eval()

    # Head is a Sequential; its weights live under the "encode." prefix.
    encode_state = {
        k.replace("encode.", ""): v
        for k, v in state_dict.items()
        if k.startswith("encode.")
    }
    with torch.device("meta"):
        head = head_class()
    head.load_state_dict(encode_state, assign=True)
    head.eval()

    # Pre-compute the constant tensors that IFNet needs.
    # These are functions of (pw, ph) only and are fixed for this export.
    ten_flow_div = torch.tensor(
        [(pw - 1.0) / 2.0, (ph - 1.0) / 2.0], dtype=torch.float32
    )

    ten_horizontal = (
        torch.linspace(-1.0, 1.0, pw).view(1, 1, 1, pw).expand(1, -1, ph, -1)
    )
    ten_vertical = torch.linspace(-1.0, 1.0, ph).view(1, 1, ph, 1).expand(1, -1, -1, pw)
    backwarp_ten_grid = torch.cat([ten_horizontal, ten_vertical], 1)  # [1, 2, ph, pw]

    # Wrapper module — this is what gets exported to ONNX.
    class RifeWrapper(nn.Module):
        """Wrap the Head and IFNet into a single module with the expected ONNX signature."""
        def __init__(self) -> None:
            """Store flownet, head, and pre-computed constant tensors as buffers."""
            super().__init__()
            self.flownet = flownet
            self.head = head
            self.register_buffer("ten_flow_div", ten_flow_div)
            self.register_buffer("backwarp_ten_grid", backwarp_ten_grid)

        def forward(
            self,
            img0: torch.Tensor,  # [1, 3, ph, pw]
            img1: torch.Tensor,  # [1, 3, ph, pw]
            timestep: torch.Tensor,  # [1, 1, ph, pw]
        ) -> torch.Tensor:
            """Run the IFNet forward pass with the pre-computed tensors and head features."""
            f0 = self.head(img0)
            f1 = self.head(img1)
            return self.flownet(
                img0,
                img1,
                timestep,
                self.ten_flow_div,
                self.backwarp_ten_grid,
                f0,
                f1,
            )

    return RifeWrapper()


def export_onnx(
    wrapper: nn.Module,
    pw: int,
    ph: int,
    out_path: Path,
    simplify: bool,
) -> None:
    """Export the wrapper module to ONNX with the given padded width and height."""
    img0 = torch.zeros(1, 3, ph, pw, dtype=torch.float32)
    img1 = torch.zeros(1, 3, ph, pw, dtype=torch.float32)
    timestep = torch.full((1, 1, ph, pw), 0.5, dtype=torch.float32)

    wrapper.eval()
    print(f"  Exporting {out_path.name} ...", end=" ", flush=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (img0, img1, timestep),
            str(out_path),
            input_names=["img0", "img1", "timestep"],
            output_names=["output"],
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
        )
    print("done.")

    if simplify:
        if onnxsim_simplify is None:
            print("  [skip] onnxsim not installed — skipping simplification.")
        else:
            print(f"  Simplifying {out_path.name} ...", end=" ", flush=True)
            model_onnx: onnx.ModelProto = onnx.load(str(out_path))
            model_simplified, check = onnxsim_simplify(model_onnx)
            if check:
                onnx.save(model_simplified, str(out_path))
                print("done.")
            else:
                print("simplification check failed, keeping original.")


def main() -> None:
    """Main script entry point."""
    parser = argparse.ArgumentParser(description="Export rife4.18 to ONNX")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "rife"),
        help="Output directory (default: rife/ relative to this script's directory)",
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        default=["all"],
        help="Which resolutions to export. 'all' or labels like 1920x1088 1080p 720p etc.",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip onnxsim simplification pass",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve requested resolutions
    resns: list[tuple[str, int, int]] = []
    if args.resolutions == ["all"]:
        resns = ALL_RESOLUTIONS
    else:
        # Accept labels like "1920x1088", "1080p", "720p", "480p", "4k", "4K"
        shortcuts = {
            "480p": ("864x480", 864, 480),
            "720p": ("1280x736", 1280, 736),
            "1080p": ("1920x1088", 1920, 1088),
            "1440p": ("2560x1440", 2560, 1440),
            "4k": ("3840x2176", 3840, 2176),
            "4K": ("3840x2176", 3840, 2176),
        }
        by_label = {r[0]: r for r in ALL_RESOLUTIONS}
        for req in args.resolutions:
            if req in shortcuts:
                resns.append(shortcuts[req])
            elif req in by_label:
                resns.append(by_label[req])
            else:
                print(
                    f"Unknown resolution '{req}'. Valid labels: {[r[0] for r in ALL_RESOLUTIONS]}"
                )
                sys.exit(1)

    print("\nrife4.18 ONNX exporter")
    print(f"Output dir : {out_dir}")
    print(f"Resolutions: {[r[0] for r in resns]}")
    print()

    # Step 1: Download architecture source from vs-rife
    work_dir = Path(tempfile.mkdtemp(prefix="rife_export_"))
    raw_base = "https://raw.githubusercontent.com/HolyWu/vs-rife/master"

    print("Step 1: Fetching model architecture from vs-rife ...")
    ifnet_path = work_dir / "IFNet_HDv3_v4_18.py"
    fetch_vsrife_file(raw_base, "vsrife/IFNet_HDv3_v4_18.py", ifnet_path)

    # IFNet_HDv3_v4_18 imports only from warplayer.
    for dep in ["warplayer.py"]:
        fetch_vsrife_file(raw_base, f"vsrife/{dep}", work_dir / dep)

    # Patch imports so the dynamically loaded module finds its siblings in work_dir.
    src = ifnet_path.read_text(encoding="utf-8")
    src = src.replace("from .warplayer", "from warplayer")
    src = src.replace("from .refine", "from refine")
    ifnet_path.write_text(src, encoding="utf-8")
    sys.path.insert(0, str(work_dir))
    print("  Architecture source ready.")

    # Step 2: Download model weights
    print("\nStep 2: Downloading model weights ...")
    pkl_path = work_dir / f"flownet_v{MODEL_VERSION}.pkl"
    download(PKL_URL, pkl_path, f"flownet_v{MODEL_VERSION}.pkl")

    # Step 3: Build wrapper (load weights once, reuse across resolutions)
    print("\nStep 3: Loading weights into model ...")
    # Build at the first resolution to verify, then rebuild per-resolution
    # (backwarp_tenGrid is resolution-dependent, so we rebuild for each).
    print("  Weights loaded.")

    # Step 4: Export each resolution
    print("\nStep 4: Exporting ONNX files ...")
    for label, pw, ph in resns:
        out_path = out_dir / f"rife4.18_{label}.onnx"
        if out_path.exists():
            print(f"  [skip] {out_path.name} already exists.")
            continue
        wrapper = build_wrapper(ifnet_path, pkl_path, pw, ph)
        export_onnx(wrapper, pw, ph, out_path, simplify=not args.no_simplify)

    print(f"\nDone. Files written to {out_dir}")
    print(
        textwrap.dedent("""
    To use in StreamDock, set RIFE_MODEL_PATH in src/app.local.cfg:
        RIFE_MODEL_PATH=ThirdParty/models/rife/rife4.18_1920x1088.onnx

    RifeInterpolator will detect the timestep input and RifeVideoStream
    will automatically compute the correct Nx multiplier for your display.
    """)
    )


if __name__ == "__main__":
    main()
