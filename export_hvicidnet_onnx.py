"""
export_hvicidnet_onnx.py -- Export HVI-CIDNet (low-light enhancement) to ONNX.

Usage:
    python -I export_hvicidnet_onnx.py [--out DIR] [--checkpoint generalization|sice]
                                       [--gamma G] [--cache DIR] [--no-simplify]

Requirements:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install onnx onnxscript onnxruntime numpy safetensors
    pip install onnxsim        (optional)

    Tested with torch 2.15 (TorchScript exporter, dynamo=False), onnx 1.23, onnxscript,
    onnxruntime 1.24 (also onnxruntime-directml 1.24). No CUDA, einops or huggingface_hub
    needed.

The script will:
  1. Download the four model-definition files of HVI-CIDNet (net/*.py) from GitHub at a
     pinned commit, and the selected checkpoint (model.safetensors) from Hugging Face at a
     pinned revision, into the cache directory (default ~/.cache/hvicidnet). Both are
     verified against pinned SHA-256 digests before anything is loaded.
  2. Build the upstream CIDNet, load the weights, and replace three methods with
     export-friendly equivalents that compute the same values:
       - RGB_HVI.HVIT / RGB_HVI.PHVIT: the upstream code writes through boolean masks
         (hue[mask] = ...), which exports as NonZero/ScatterND with data-dependent
         shapes. Rewritten with torch.where chains in the same precedence order, and the
         density constant k read from the parameter instead of a Python float via .item().
       - CAB.forward: einops.rearrange replaced by reshape (same memory order).
       - IEL.forward: tanh written via exp, so DirectML cannot fuse it into the preceding
         depthwise Conv (its DmlFusedConv+Tanh gives wrong output; see _tanh_unfused).
       - PHVIT also clamps the hue sector index to 0..5: upstream turns pixels whose hue
         rounds to exactly 1.0 black (see the comment in _phvit).
     The patched model is checked against the unpatched upstream model before export.
  3. Wrap it so the ONNX graph takes any H and W: the frame is reflect-padded at the
     bottom/right to a multiple of 8 inside the graph (exactly what upstream eval.py /
     app.py do on the host), run, clamped to [0, 1] and cropped back. Optional input
     gamma (upstream's "gamma curve" slider, default 1.0 = off) is baked in.
  4. Export with opset 17 and dynamic H/W, run onnx.checker, optionally onnxsim, and
     compare ONNX Runtime (CPU) against PyTorch on a random-dark tensor at an odd size.

ONNX tensor interface:
    Input  "input"   [1, 3, H, W]  float32  RGB, normalised [0, 1]   (any H, W >= 8)
    Output "output"  [1, 3, H, W]  float32  enhanced RGB, [0, 1]     (same H, W)

    The output is the finished image (not curve parameters). No host-side padding is
    needed: pad-to-8 and crop happen in the graph, so the runner should NOT pad (a
    zero/black pad added outside would be treated as picture and darken the edges).
    Batch is fixed at 1. The cost is roughly linear in H*W; memory grows with H*W too.

Output file (default --out hvicidnet, relative to the working directory):
    hvicidnet/hvi-cidnet.onnx        (--checkpoint generalization)
    hvicidnet/hvi-cidnet-sice.onnx   (--checkpoint sice)

License:
    Code: MIT, Copyright (c) 2024 Fediory Feng.
        https://github.com/Fediory/HVI-CIDNet/blob/master/LICENSE
    Weights: MIT (license: mit in each Hugging Face model card).
        https://huggingface.co/Fediory/HVI-CIDNet-Generalization
        https://huggingface.co/Fediory/HVI-CIDNet-SICE
    Paper: Yan et al., "HVI: A New Color Space for Low-light Image Enhancement", CVPR 2025.
    The "generalization" weights were trained on LOLv2-Synthetic with random gamma
    augmentation; "sice" on SICE. Those datasets carry their own (research) terms; the
    MIT grant covers the weights as published.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import sys
import types
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import onnx
except ImportError:
    onnx = None  # type: ignore[assignment]

try:
    from onnxsim import simplify as onnxsim_simplify
except ImportError:
    onnxsim_simplify = None

DEFAULT_OUT_DIR = Path("hvicidnet")
DEFAULT_CACHE = Path.home() / ".cache" / "hvicidnet"

# Upstream source, pinned to a commit so the patched methods below keep matching it.
_UPSTREAM_COMMIT = "eb43d7d91e9a336c66856824ff9e4603ae41f408"
_UPSTREAM_RAW = f"https://raw.githubusercontent.com/Fediory/HVI-CIDNet/{_UPSTREAM_COMMIT}/net/"
_UPSTREAM_FILES: dict[str, str | None] = {
    # name -> sha256 of the raw (LF) file at _UPSTREAM_COMMIT
    "CIDNet.py": "dc9b9232f225f5760417175d25f8efc146f51c0acd1ee014df6b9f0a6b2e8d84",
    "HVI_transform.py": "8be0fea6e61c51d6290722f0cf81be1f334b1b998dbfd409435343d9d4bd5bb2",
    "LCA.py": "271f487091bd13ed211a8b0c9cb07036957ba535f5dd3aedc131498500979d26",
    "transformer_utils.py": "7bfb2659e97289f0e8005935e15edbe0af3ae5ef05c50863a074fd745617b13f",
}

_CHECKPOINTS: dict[str, dict[str, str]] = {
    # Recommended upstream for real-world input ("strongest generalization", the demo's
    # default). Identical tensors to weights/generalization.pth in Fediory/HVI-CIDNet.
    "generalization": {
        "repo": "Fediory/HVI-CIDNet-Generalization",
        "revision": "51481ef2546f870060c43eb6d6525399f5b3d2d3",
        "sha256": "2291407125e809cc9c0614cc2d010d21d309a66eb3da33e1ee2386a68fa05894",
        "out_name": "hvi-cidnet.onnx",
    },
    "sice": {
        "repo": "Fediory/HVI-CIDNet-SICE",
        "revision": "dc35250dc973a8efff89ee43c96d791d369d6451",
        "sha256": "77b36ad42df18087e2fd527479a8277f3e3ea2730f97897f9cdc70069925ac10",
        "out_name": "hvi-cidnet-sice.onnx",
    },
}

_HAVE_EINOPS = importlib.util.find_spec("einops") is not None
_PAD_MULTIPLE = 8  # three 0.5x bilinear downsamples; skip connections need exact halves
_PI = math.pi
_EPS = 1e-8


def _sha256(path: Path) -> str:
    """Return the hex SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url: str, dest: Path, sha256: str | None) -> None:
    """Download url to dest unless a verified copy exists; verify sha256 when pinned."""
    if dest.exists() and (sha256 is None or _sha256(dest) == sha256):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    if sha256 is not None and _sha256(tmp) != sha256:
        tmp.unlink()
        sys.exit(f"ERROR: SHA-256 mismatch for {url}")
    tmp.replace(dest)


def _load_upstream(src_dir: Path) -> types.ModuleType:
    """Import upstream net/CIDNet.py as package 'net' without adding it to sys.path.

    CIDNet.py imports huggingface_hub (for a mixin) and LCA.py imports einops; neither is
    needed for inference, so both are stubbed. CAB.forward, the only einops user, is
    replaced by _cab_forward below.
    """
    hub = types.ModuleType("huggingface_hub")
    hub.PyTorchModelHubMixin = type("PyTorchModelHubMixin", (), {})  # type: ignore[attr-defined]
    sys.modules.setdefault("huggingface_hub", hub)
    if not _HAVE_EINOPS:
        ein = types.ModuleType("einops")

        def _no_rearrange(*_a: object, **_k: object) -> None:
            raise RuntimeError("einops stub: CAB.forward should have been patched")

        ein.rearrange = _no_rearrange  # type: ignore[attr-defined]
        sys.modules["einops"] = ein

    pkg = types.ModuleType("net")
    pkg.__path__ = [str(src_dir)]  # type: ignore[attr-defined]
    sys.modules["net"] = pkg
    for name in ("transformer_utils", "HVI_transform", "LCA", "CIDNet"):
        spec = importlib.util.spec_from_file_location(f"net.{name}", src_dir / f"{name}.py")
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"net.{name}"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["net.CIDNet"]


# ---- export-friendly replacements (same maths as upstream, no boolean-mask writes) ----


def _hvit(self: nn.Module, img: torch.Tensor) -> torch.Tensor:
    """RGB -> HVI, equivalent to upstream RGB_HVI.HVIT."""
    r, g, b = img[:, 0], img[:, 1], img[:, 2]
    value = img.max(1)[0]
    img_min = img.min(1)[0]
    d = value - img_min + _EPS
    hue_b = 4.0 + (r - g) / d
    hue_g = 2.0 + (b - r) / d
    hr = (g - b) / d
    hue_r = hr - torch.floor(hr / 6.0) * 6.0  # Python % (sign of divisor)
    # Upstream assigns B, then G, then R, so R wins ties, then G.
    hue = torch.where(r == value, hue_r, torch.where(g == value, hue_g, hue_b))
    hue = torch.where(img_min == value, torch.zeros_like(hue), hue) / 6.0
    sat = (value - img_min) / (value + _EPS)
    sat = torch.where(value == 0, torch.zeros_like(sat), sat)
    hue, sat, value = hue.unsqueeze(1), sat.unsqueeze(1), value.unsqueeze(1)
    k = self.density_k
    cs = (torch.sin(value * 0.5 * _PI) + _EPS).pow(k)
    h_ = cs * sat * torch.cos(2.0 * _PI * hue)
    v_ = cs * sat * torch.sin(2.0 * _PI * hue)
    return torch.cat([h_, v_, value], dim=1)


def _phvit(self: nn.Module, img: torch.Tensor) -> torch.Tensor:
    """HVI -> RGB, upstream RGB_HVI.PHVIT (incl. gated/alpha), minus its black-pixel bug."""
    hh = torch.clamp(img[:, 0], -1, 1)
    vv = torch.clamp(img[:, 1], -1, 1)
    v = torch.clamp(img[:, 2], 0, 1)
    cs = (torch.sin(v * 0.5 * _PI) + _EPS).pow(self.density_k)
    hh = torch.clamp(hh / (cs + _EPS), -1, 1)
    vv = torch.clamp(vv / (cs + _EPS), -1, 1)
    h = torch.atan2(vv + _EPS, hh + _EPS) / (2 * _PI)
    h = h - torch.floor(h)  # h % 1
    s = torch.sqrt(hh**2 + vv**2 + _EPS)
    if self.gated:
        s = s * self.alpha_s
    s = torch.clamp(s, 0, 1)
    # Upstream leaves hi == 6 (h just below 1, so h * 6 rounds to 6.0) matching no
    # sector, and the pixel comes out black: a saturated red pixel (hue ~0) turns into
    # a black speck, and whether it does depends on the EP's rounding (DirectML hit 2
    # pixels on a 1080p frame where CPU hit none). Clamping to sector 5 gives f = 1,
    # which is the continuous value (sector 5 at f = 1 equals sector 0 at f = 0).
    hi = torch.clamp(torch.floor(h * 6.0), 0.0, 5.0)
    f = h * 6.0 - hi
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    zero = torch.zeros_like(v)

    def pick(c0: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor,
             c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor) -> torch.Tensor:
        out = zero
        for idx, c in ((5, c5), (4, c4), (3, c3), (2, c2), (1, c1), (0, c0)):
            out = torch.where(hi == idx, c, out)
        return out

    rgb = torch.stack([pick(v, q, p, p, t, v), pick(t, v, v, q, p, p),
                       pick(p, p, t, v, v, q)], dim=1)
    if self.gated2:
        rgb = rgb * self.alpha
    return rgb


def _cab_forward(self: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Upstream CAB.forward with einops.rearrange expressed as reshape."""
    b, c, h, w = x.shape
    heads = self.num_heads
    q = self.q_dwconv(self.q(x))
    k, v = self.kv_dwconv(self.kv(y)).chunk(2, dim=1)
    q = q.reshape(b, heads, c // heads, h * w)
    k = k.reshape(b, heads, c // heads, h * w)
    v = v.reshape(b, heads, c // heads, h * w)
    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)
    attn = torch.softmax((q @ k.transpose(-2, -1)) * self.temperature, dim=-1)
    out = (attn @ v).reshape(b, c, h, w)
    return self.project_out(out)


def _tanh_unfused(x: torch.Tensor) -> torch.Tensor:
    """tanh written as 1 - 2 / (exp(2x) + 1), so no runtime sees a Conv -> Tanh pair.

    ONNX Runtime's DirectML EP fuses Conv + Tanh into DmlFusedConv (graph level
    EXTENDED and above), and on IEL's depthwise convs that fused kernel is wrong:
    max error ~0.24 on a 960x540 frame on an RTX 5090, ORT-DirectML 1.24.4, while the
    same graph with fusion blocked matches CPU to 3e-5. Exp is not a fusable
    activation, and the form saturates correctly (exp -> inf gives 1, exp -> 0 gives -1).
    """
    return 1.0 - 2.0 / (torch.exp(2.0 * x) + 1.0)


def _iel_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Upstream IEL.forward with the DirectML-safe tanh."""
    x = self.project_in(x)
    x1, x2 = self.dwconv(x).chunk(2, dim=1)
    x1 = _tanh_unfused(self.dwconv1(x1)) + x1
    x2 = _tanh_unfused(self.dwconv2(x2)) + x2
    return self.project_out(x1 * x2)


class HviCidNetWrapper(nn.Module):
    """Pad to a multiple of 8 (reflect), optional input gamma, CIDNet, clamp, crop."""

    def __init__(self, net: nn.Module, gamma: float) -> None:
        """Wrap a loaded CIDNet; gamma 1.0 leaves the input untouched."""
        super().__init__()
        self.net = net
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [1, 3, H, W] RGB in [0, 1] -> [1, 3, H, W] enhanced RGB in [0, 1]."""
        h, w = x.shape[2], x.shape[3]
        pad_h = (_PAD_MULTIPLE - h % _PAD_MULTIPLE) % _PAD_MULTIPLE
        pad_w = (_PAD_MULTIPLE - w % _PAD_MULTIPLE) % _PAD_MULTIPLE
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        if self.gamma != 1.0:
            x = x.clamp_min(0.0) ** self.gamma
        y = torch.clamp(self.net(x), 0.0, 1.0)
        return y[:, :, :h, :w]


def build_model(cache: Path, checkpoint: str) -> tuple[nn.Module, nn.Module]:
    """Return (patched, original) CIDNet instances with the checkpoint loaded."""
    from safetensors.torch import load_file

    src_dir = cache / f"src-{_UPSTREAM_COMMIT[:12]}"
    for name, digest in _UPSTREAM_FILES.items():
        download(_UPSTREAM_RAW + name, src_dir / name, digest)
    ck = _CHECKPOINTS[checkpoint]
    w_path = cache / checkpoint / ck["revision"][:12] / "model.safetensors"
    download(f"https://huggingface.co/{ck['repo']}/resolve/{ck['revision']}/model.safetensors",
             w_path, ck["sha256"])

    mod = _load_upstream(src_dir)
    state = load_file(str(w_path))
    nets = []
    for _ in range(2):
        net = mod.CIDNet()  # config.json: channels [36,36,72,144], heads [1,2,4,8], norm false
        net.load_state_dict(state, strict=True)
        # Upstream inference (eval_hf.py, the HF demo) turns both gates on with alpha 1.0,
        # i.e. a no-op; keep that so the module state matches upstream exactly.
        net.trans.gated = True
        net.trans.gated2 = True
        net.trans.alpha = 1.0
        net.trans.alpha_s = 1.0
        nets.append(net.eval())
    patched, original = nets
    lca = sys.modules["net.LCA"]
    patched.trans.HVIT = types.MethodType(_hvit, patched.trans)
    patched.trans.PHVIT = types.MethodType(_phvit, patched.trans)
    for m in patched.modules():
        if isinstance(m, lca.CAB):
            m.forward = types.MethodType(_cab_forward, m)
        elif isinstance(m, lca.IEL):
            m.forward = types.MethodType(_iel_forward, m)
    # Without einops installed the reference CAB cannot run either; it then uses the
    # reshape version too, and the check covers only the replaced colour maths.
    if not _HAVE_EINOPS:
        for m in original.modules():
            if isinstance(m, lca.CAB):
                m.forward = types.MethodType(_cab_forward, m)
    return patched, original


def _dark_test_tensor(h: int, w: int) -> torch.Tensor:
    """A deterministic dark, textured RGB tensor in [0, 0.3]."""
    g = torch.Generator().manual_seed(0)
    base = torch.rand(1, 3, h // 4 + 1, w // 4 + 1, generator=g)
    img = torch.nn.functional.interpolate(base, size=(h, w), mode="bilinear")
    return (img * 0.3).clamp(0, 1)


def export(wrapper: nn.Module, out_path: Path, simplify: bool) -> None:
    """Export the wrapper to ONNX (opset 17, dynamic H/W) as a single file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = _dark_test_tensor(360, 640)
    # The TorchScript-based exporter: the dynamo one emits opset 18 and its down-converter
    # has no Pad adapter to 17. Shape arithmetic (pad-to-8) is traced as graph ops.
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (dummy,), str(out_path), dynamo=False, opset_version=17,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {2: "height", 3: "width"},
                          "output": {2: "height", 3: "width"}},
            do_constant_folding=True,
        )
    print(f"Saved: {out_path}")
    if onnx is None:
        return
    proto = onnx.load(str(out_path))
    onnx.checker.check_model(proto)
    print("ONNX model check passed.")
    if simplify and onnxsim_simplify is not None:
        simplified, ok = onnxsim_simplify(proto)
        if ok:
            onnx.save(simplified, str(out_path))
            print("onnxsim simplification succeeded.")
        else:
            print("WARNING: onnxsim check failed; keeping the unsimplified graph.")


def sanity_check(wrapper: nn.Module, original: nn.Module, out_path: Path, gamma: float) -> None:
    """Patched vs upstream in PyTorch, then ONNX Runtime CPU vs PyTorch at odd sizes."""
    import onnxruntime as ort

    ref = HviCidNetWrapper(original, gamma).eval()
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    for h, w in ((271, 333), (360, 640)):
        x = _dark_test_tensor(h, w)
        with torch.no_grad():
            a = wrapper(x)
            b = ref(x)
        d_patch = (a - b).abs().max().item()
        y = sess.run(["output"], {"input": x.numpy()})[0]
        d_onnx = float(np.abs(y - a.numpy()).max())
        print(f"  {w}x{h}: patched-vs-upstream {d_patch:.2e}, onnx-vs-torch {d_onnx:.2e}, "
              f"out shape {list(y.shape)}")
        if y.shape != (1, 3, h, w) or d_patch > 1e-4 or d_onnx > 1e-3:
            sys.exit("ERROR: sanity check failed")
    # DirectML at the app's optimisation level (ORT_ENABLE_ALL), when this onnxruntime
    # build has it: catches a fusion that breaks the maths (see _tanh_unfused).
    if "DmlExecutionProvider" in ort.get_available_providers():
        so = ort.SessionOptions()
        so.enable_mem_pattern = False
        so.log_severity_level = 3
        dml = ort.InferenceSession(str(out_path), so, providers=["DmlExecutionProvider"])
        x = _dark_test_tensor(544, 960)
        a = sess.run(["output"], {"input": x.numpy()})[0]
        y = dml.run(["output"], {"input": x.numpy()})[0]
        d_dml = float(np.abs(y - a).max())
        print(f"  960x544: onnx-dml-vs-onnx-cpu {d_dml:.2e}")
        if d_dml > 2e-3:
            sys.exit("ERROR: DirectML output diverges from CPU")
    print("Sanity check passed.")


def main() -> None:
    """Parse arguments and run the export."""
    # torch.onnx prints emoji progress marks; a cp1252 console would raise on them.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser(description="Export HVI-CIDNet (low-light) to ONNX.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, metavar="DIR")
    ap.add_argument("--checkpoint", choices=sorted(_CHECKPOINTS), default="generalization")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="Input gamma baked into the graph (upstream slider; <1 brightens).")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE, metavar="DIR")
    ap.add_argument("--no-simplify", action="store_true")
    args = ap.parse_args()

    patched, original = build_model(args.cache, args.checkpoint)
    wrapper = HviCidNetWrapper(patched, args.gamma).eval()
    out_path = args.out / _CHECKPOINTS[args.checkpoint]["out_name"]
    export(wrapper, out_path, simplify=not args.no_simplify)
    sanity_check(wrapper, original, out_path, args.gamma)
    print(f"{out_path.name}: {out_path.stat().st_size} bytes, sha256 {_sha256(out_path)}")


if __name__ == "__main__":
    main()
