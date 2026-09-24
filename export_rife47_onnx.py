"""
export_rife47_onnx.py -- Export rife4.7 to a single resolution-independent ONNX.

Usage:
    python export_rife47_onnx.py [--out DIR] [--opset 18]

Requirements:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install onnx onnxruntime onnxscript

The script will:
  1. Download IFNet_HDv3_v4_7.py + warplayer.py from vs-rife (GitHub, MIT).
  2. Download flownet_v4.7.pkl from the vs-rife model release (~21 MB, MIT).
  3. Wrap IFNet so the warp grid is derived from the input shape at run time, and so inputs
     are padded to a multiple of 32 and the result cropped back.
  4. Export with dynamic height/width and fold the weights into one self-contained file.

Output:
    rife/rife47.onnx    any resolution, 2x interpolation, timestep-capable

WHY THIS IS A SEPARATE SCRIPT FROM export_rife_onnx.py
------------------------------------------------------
export_rife_onnx.py emits one fixed-shape ONNX per resolution tier for rife4.18, because it
bakes ten_flow_div and a [1, 2, ph, pw] backwarp grid as buffers computed from the export
resolution. That is fine for 4.18, whose tiers cover every common size, but rife47 exists
precisely to cover the sizes no tier matches -- so it has to accept any resolution.

Two things make that work here, and both are load-bearing:

  * The warp grid is built inside forward() from img0's own shape, using arange rather than
    linspace so the exporter emits a Range over a dynamic dimension instead of folding a
    fixed-length constant into the graph.
  * RIFE's four-scale pyramid requires dimensions that are multiples of 32, so forward() pads
    to the next multiple and crops the output back. Without this the model would reject exactly
    the resolutions it is meant to rescue.

4.7 rather than 4.18 because 4.7's smaller encoder (a 2-layer Sequential under `encode.`, not
4.18's 4-layer Head) traces cleanly at dynamic shapes.

VERIFIED, NOT ASSUMED
---------------------
Checked against the third-party conversion this replaces (yuvraj108c/rife-onnx, which declares
no licence) by interpolating real consecutive video frames and scoring both against the true
middle frame:

    256x448 (aligned)    naive average 22.55 dB | theirs 28.82 dB | this 28.81 dB
    250x440 (unaligned)  naive average 22.57 dB | theirs 28.63 dB | this 28.63 dB

Mean absolute difference between the two models: 0.002.

Do NOT verify this on random noise. An earlier pass did, measured a max-abs difference of ~0.9,
and wrongly concluded the rebuild was broken: RIFE estimates optical flow, which is chaotic on
noise, so the comparison is meaningless there. Use real frames with real motion.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

VSRIFE_RAW = "https://raw.githubusercontent.com/HolyWu/vs-rife/master/vsrife"
PKL_URL = "https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v4.7.pkl"
ALIGN = 32


def download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    print(f"  downloading {dest.name} ...", flush=True)
    urllib.request.urlretrieve(url, dest)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class Rife47(nn.Module):
    """IFNet 4.7 with a shape-derived warp grid and internal padding."""

    def __init__(self, flownet: nn.Module, encode: nn.Module) -> None:
        super().__init__()
        self.flownet = flownet
        self.encode = encode

    def forward(
        self, img0: torch.Tensor, img1: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        h0, w0 = img0.shape[-2], img0.shape[-1]
        h = ((h0 + ALIGN - 1) // ALIGN) * ALIGN
        w = ((w0 + ALIGN - 1) // ALIGN) * ALIGN
        if h != h0 or w != w0:
            pad = (0, w - w0, 0, h - h0)
            img0 = F.pad(img0, pad, mode="replicate")
            img1 = F.pad(img1, pad, mode="replicate")
            timestep = F.pad(timestep, pad, mode="replicate")

        # linspace(-1, 1, n) as arange, so a dynamic dimension stays dynamic.
        xs = torch.arange(w, dtype=torch.float32, device=img0.device) * (2.0 / (w - 1)) - 1.0
        ys = torch.arange(h, dtype=torch.float32, device=img0.device) * (2.0 / (h - 1)) - 1.0
        grid = torch.cat(
            [xs.reshape(1, 1, 1, w).expand(1, 1, h, w), ys.reshape(1, 1, h, 1).expand(1, 1, h, w)],
            1,
        )
        flow_div = torch.stack(
            [
                (torch.tensor(w, dtype=torch.float32, device=img0.device) - 1.0) / 2.0,
                (torch.tensor(h, dtype=torch.float32, device=img0.device) - 1.0) / 2.0,
            ]
        )

        out = self.flownet(
            img0, img1, timestep, flow_div, grid, self.encode(img0), self.encode(img1)
        )
        return out[:, :, :h0, :w0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="rife", help="output directory (default: rife/)")
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_rife47_tmp"
    work.mkdir(exist_ok=True)

    download(f"{VSRIFE_RAW}/warplayer.py", work / "warplayer.py")
    download(f"{VSRIFE_RAW}/IFNet_HDv3_v4_7.py", work / "IFNet_HDv3_v4_7.py")
    download(PKL_URL, work / "flownet_v4.7.pkl")

    # IFNet_HDv3_v4_7 does `from .warplayer import warp`, so it needs a package parent.
    pkg = types.ModuleType("vsrife")
    pkg.__path__ = [str(work.resolve())]
    sys.modules["vsrife"] = pkg
    load_module("vsrife.warplayer", work / "warplayer.py")
    ifnet_mod = load_module("vsrife.IFNet_HDv3_v4_7", work / "IFNet_HDv3_v4_7.py")

    state = torch.load(work / "flownet_v4.7.pkl", map_location="cpu", weights_only=True)
    state = {k.replace("module.", ""): v for k, v in state.items()}

    # ensemble=True matches the conversion this replaces (its filename records the setting) and
    # trades speed for quality by averaging a forward and a reversed pass.
    flownet = ifnet_mod.IFNet(scale=1.0, ensemble=True)
    flownet.load_state_dict({k: v for k, v in state.items() if not k.startswith("encode.")}, strict=False)
    flownet.eval()

    encode = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ConvTranspose2d(16, 4, 4, 2, 1))
    encode.load_state_dict(
        {k.replace("encode.", ""): v for k, v in state.items() if k.startswith("encode.")},
        strict=True,
    )
    encode.eval()

    model = Rife47(flownet, encode).eval()

    # Trace at a size that is NOT a multiple of 32, so the padding branch is exported rather
    # than folded away as a no-op.
    h, w = 250, 440
    sample = (torch.rand(1, 3, h, w), torch.rand(1, 3, h, w), torch.full((1, 1, h, w), 0.5))
    with torch.inference_mode():
        print("  eager forward ->", tuple(model(*sample).shape))

    tmp_onnx = work / "rife47_external.onnx"
    torch.onnx.export(
        model,
        sample,
        str(tmp_onnx),
        input_names=["img0", "img1", "timestep"],
        output_names=["output"],
        dynamic_axes={
            "img0": {2: "height", 3: "width"},
            "img1": {2: "height", 3: "width"},
            "timestep": {2: "height", 3: "width"},
            "output": {2: "height", 3: "width"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )

    # The exporter splits weights into a .data sidecar; the app loads one file per model.
    import onnx

    dest = out_dir / "rife47.onnx"
    onnx.save_model(onnx.load(tmp_onnx), str(dest), save_as_external_data=False)
    print(f"  wrote {dest}  ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
