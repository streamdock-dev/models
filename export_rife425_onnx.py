"""
export_rife425_onnx.py -- Export RIFE 4.25 to a single resolution-independent ONNX.

Usage:
    python -I export_rife425_onnx.py [--out DIR] [--opset 17] [--work DIR] [--keep-tmp]
                                     [--no-check]

Requirements:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install onnx onnxruntime numpy
    (tested with torch 2.12.0+cpu, onnx 1.21, onnxruntime 1.25 CPU and
    onnxruntime-directml 1.24.4)

The script will:
  1. Download IFNet_HDv3_v4_25.py + warplayer.py from vs-rife at a pinned commit (MIT).
  2. Download flownet_v4.25.pkl from the vs-rife model release (~24.6 MB, MIT) and check
     its sha256. That file is byte-identical to train_log/flownet.pkl inside the author's
     own 4.25 release zip on Google Drive (linked from the Practical-RIFE README).
  3. Wrap Head + IFNet so the warp grid and flow divisor are derived from the input shape at
     run time, and so inputs are padded to a multiple of 64 and the result cropped back.
  4. Export with dynamic height/width as one self-contained .onnx (no .data sidecar).
  5. Sanity-check the ONNX against the PyTorch model on CPU at aligned and unaligned sizes.
  6. Delete the default download dir (<out>/_rife425_tmp) unless --keep-tmp is given;
     a directory passed with --work is always kept.

Output:
    <out>/rife425.onnx    any resolution, arbitrary timestep, ~22.8 MB

TENSOR INTERFACE (same names, layout and range as the retired rife4.18 tiers)
-----------------------------------------------------------------------------
    img0      float32 [1, 3, H, W]   RGB, 0..1, first frame
    img1      float32 [1, 3, H, W]   RGB, 0..1, second frame
    timestep  float32 [1, 1, H, W]   every element = t in (0, 1); 0 -> img0, 1 -> img1
    output    float32 [1, 3, H, W]   RGB, 0..1, the frame at time t

    H and W are any positive size (dynamic axes "height" / "width"); the caller need not
    pad. Internally the frames are zero-padded on the right/bottom to a multiple of 64 (what
    upstream Practical-RIFE and vs-rife do for 4.25), the timestep is edge-padded, and the
    output is cropped back to H x W. A caller that already zero-pads to 32 (StreamDock's
    RifeInterpolator) gets bit-for-bit the same frame as one that does not pad at all.
    The model reads only the timestep's value, so the caller may fill it with one constant.

WHY 64, AND WHY THIS IS RESOLUTION-INDEPENDENT WHEN THE 4.18 TIERS WERE NOT
--------------------------------------------------------------------------
4.25 has five flow blocks with scale_list [16, 8, 4, 2, 1]; block0 resizes by 1/16 and then
strides by 4, so the padded size must divide by 64 (4.18 had [8, 4, 2, 1] -> 32).

The retired 4.18 exporter baked ten_flow_div and a [1, 2, ph, pw] backwarp grid as buffers
computed from one export resolution, which is the only reason the 4.18 exports were
fixed-shape. Here both are built in forward() from the padded shape, using arange
rather than linspace so the exporter emits a Range over a dynamic dimension instead of folding
a constant. Every F.interpolate in IFBlock uses scale_factor, which exports as Resize with
constant scales and stays dynamic. GridSample (opset 16+) handles the warp; ONNX Runtime's
CPU, DirectML and CUDA providers all run it at dynamic shapes.

scale is fixed at 1.0 (upstream's default; it suggests 0.5 for 4K). ensemble is not
supported by 4.21+ upstream.

VERIFY ON REAL FRAMES, NOT NOISE
--------------------------------
RIFE estimates optical flow, which is chaotic on noise; ONNX-vs-PyTorch differences there are
meaningless. The built-in check below uses a smooth synthetic pair; check a new export on
consecutive frames of real video as well.

LICENSE
-------
Practical-RIFE (code and, per its README, the trained models it links) is MIT,
Copyright (c) 2021 hzwer: https://github.com/hzwer/Practical-RIFE/blob/main/LICENSE
vs-rife (architecture file, warplayer, weight mirror) is MIT, Copyright (c) 2021 HolyWu:
https://github.com/HolyWu/vs-rife/blob/master/LICENSE
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import shutil
import sys
import types
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

VSRIFE_COMMIT = "3488617283db7c428a83ba4a19382285da698b6a"
VSRIFE_RAW = f"https://raw.githubusercontent.com/HolyWu/vs-rife/{VSRIFE_COMMIT}/vsrife"
PKL_URL = "https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v4.25.pkl"
PKL_SHA256 = "6615790efd627772917205db291f51cd392528a157ecbb2ecaeec3bff8eb6de2"
ALIGN = 64


def download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    print(f"  downloading {dest.name} ...", flush=True)
    urllib.request.urlretrieve(url, dest)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class Rife425(nn.Module):
    """Head + IFNet 4.25 with a shape-derived warp grid and internal padding to 64."""

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
        # Traced unconditionally (the trace size is unaligned), so a zero pad is a no-op.
        pad = (0, w - w0, 0, h - h0)
        img0 = F.pad(img0, pad)  # zeros, as upstream inference and vs-rife do
        img1 = F.pad(img1, pad)
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


def build_model(work: Path) -> Rife425:
    """Fetch upstream sources and weights into work/ and return the eval-mode wrapper."""
    work.mkdir(parents=True, exist_ok=True)
    download(f"{VSRIFE_RAW}/warplayer.py", work / "warplayer.py")
    download(f"{VSRIFE_RAW}/IFNet_HDv3_v4_25.py", work / "IFNet_HDv3_v4_25.py")
    download(PKL_URL, work / "flownet_v4.25.pkl")
    got = sha256(work / "flownet_v4.25.pkl")
    if got != PKL_SHA256:
        raise SystemExit(f"flownet_v4.25.pkl sha256 mismatch: {got}")

    # IFNet_HDv3_v4_25 does `from .warplayer import warp`, so it needs a package parent.
    pkg = types.ModuleType("vsrife")
    pkg.__path__ = [str(work.resolve())]
    sys.modules["vsrife"] = pkg
    load_module("vsrife.warplayer", work / "warplayer.py")
    mod = load_module("vsrife.IFNet_HDv3_v4_25", work / "IFNet_HDv3_v4_25.py")

    state = torch.load(work / "flownet_v4.25.pkl", map_location="cpu", weights_only=True)
    state = {k.replace("module.", ""): v for k, v in state.items() if "module." in k}

    flownet = mod.IFNet(scale=1.0, ensemble=False)
    missing, unexpected = flownet.load_state_dict(state, strict=False)
    # teacher.* and caltime.* are training-only (privileged distillation branch, unused at
    # inference); every inference weight must load.
    unexpected = [k for k in unexpected if not k.startswith(("teacher.", "caltime."))]
    if missing or unexpected:
        raise SystemExit(f"state mismatch: missing={missing} unexpected={unexpected}")
    flownet.eval()

    encode = mod.Head()
    encode.load_state_dict(
        {k[len("encode."):]: v for k, v in state.items() if k.startswith("encode.")}, strict=True
    )
    encode.eval()
    return Rife425(flownet, encode).eval()


def smooth_pair(h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Two smooth frames shifted a few pixels, so flow is well defined (unlike noise)."""
    ys = torch.linspace(0, 6.28, h).view(h, 1)
    xs = torch.linspace(0, 6.28, w).view(1, w)
    def frame(dx: float) -> torch.Tensor:
        r = 0.5 + 0.4 * torch.sin(3 * (xs + dx) + ys)
        g = 0.5 + 0.4 * torch.cos(2 * ys - (xs + dx))
        b = 0.5 + 0.4 * torch.sin(xs + dx) * torch.cos(ys)
        return torch.stack([r, g, b]).unsqueeze(0).expand(1, 3, h, w).contiguous()
    return frame(0.0), frame(0.05)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="rife", help="output directory (default: rife/)")
    ap.add_argument("--work", default=None, help="download dir (default: <out>/_rife425_tmp)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--keep-tmp", action="store_true", help="keep the default download dir")
    ap.add_argument("--no-check", action="store_true", help="skip the ONNX Runtime sanity check")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(args.work) if args.work else out_dir / "_rife425_tmp"
    model = build_model(work)

    # Trace at a size that is NOT a multiple of 64, so the padding is exported rather than
    # folded away as a no-op.
    h, w = 250, 440
    a, b = smooth_pair(h, w)
    sample = (a, b, torch.full((1, 1, h, w), 0.5))
    with torch.inference_mode():
        print("  eager forward ->", tuple(model(*sample).shape))

    import onnx

    tmp_onnx = work / "rife425_external.onnx"
    axes = {2: "height", 3: "width"}
    with torch.no_grad():
        torch.onnx.export(
            model,
            sample,
            str(tmp_onnx),
            input_names=["img0", "img1", "timestep"],
            output_names=["output"],
            dynamic_axes={"img0": axes, "img1": axes, "timestep": axes, "output": axes},
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

    # The app loads one file per model, so fold any weight sidecar back in.
    dest = out_dir / "rife425.onnx"
    proto = onnx.load(str(tmp_onnx))
    # The tracer names the output's batch/channel dims after the crop Slice; pin them to 1, 3.
    out_dims = proto.graph.output[0].type.tensor_type.shape.dim
    for i, v in ((0, 1), (1, 3)):
        out_dims[i].Clear()
        out_dims[i].dim_value = v
    onnx.checker.check_model(proto, full_check=True)
    onnx.save_model(proto, str(dest), save_as_external_data=False)
    print(f"  wrote {dest}  ({dest.stat().st_size / 1e6:.1f} MB, sha256 {sha256(dest)})")

    if not args.no_check:
        check(model, dest)
    if args.work is None and not args.keep_tmp:
        shutil.rmtree(work, ignore_errors=True)


def check(model: Rife425, dest: Path) -> None:
    """Compare ONNX Runtime (CPU) against PyTorch, at two unaligned sizes and one aligned."""
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(dest), providers=["CPUExecutionProvider"])
    for (ch, cw), t in (((250, 440), 0.5), ((368, 642), 0.25), ((512, 512), 0.75)):
        a, b = smooth_pair(ch, cw)
        ts = torch.full((1, 1, ch, cw), t)
        with torch.inference_mode():
            ref = model(a, b, ts).numpy()
        got = sess.run(None, {"img0": a.numpy(), "img1": b.numpy(), "timestep": ts.numpy()})[0]
        diff = float(np.abs(ref - got).max())
        print(f"  check {cw}x{ch} t={t}: shape {got.shape}, max |onnx - torch| = {diff:.2e}")
        if got.shape != ref.shape or diff > 1e-3:
            raise SystemExit("ONNX output does not match PyTorch")
    print("  sanity check passed")


if __name__ == "__main__":
    main()
