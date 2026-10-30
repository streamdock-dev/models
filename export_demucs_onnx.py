"""
export_demucs_onnx.py -- Export Demucs htdemucs_ft to ONNX.

Usage:
    python export_demucs_onnx.py [--out DIR] [--variant {4stem,vocals,both}] [--no-simplify]

Requirements:
    pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<2.6"
    pip install demucs onnx onnxruntime
    pip install onnxsim  # optional, for graph simplification

The script will:
  1. Load htdemucs_ft from facebookresearch/demucs (MIT licence) via the demucs Python
     package (weights downloaded on first use to ~/.cache/torch/). htdemucs_ft is a "bag" of
     four HTDemucs models, each fine-tuned for one stem: the bag takes stem i from model i.
  2. Replace HTDemucs's complex-valued STFT/iSTFT (torch.stft/istft with return_complex,
     which the ONNX exporter can't handle) with an equivalent real-valued pair built from
     conv1d/conv_transpose1d over a fixed DFT basis. The result matches the original model
     to ~1e-7; the script checks this before exporting.
  3. Trace one chunk at the models' training length -- 7.8 s, 343980 samples. HTDemucs
     refuses anything longer, and pads anything shorter up to it, so this is the only size
     worth exporting.
  4. Export to ONNX at opset 17, check ONNX Runtime against PyTorch, optionally simplify.

Variants (both output all four stems, which is what DemucsFilter.cs reads):
    4stem   all four specialists, each contributing its own stem (htdemucs_ft proper).
            About 4x the size and compute of the vocals variant.
    vocals  the vocals specialist alone. Its vocals stem is htdemucs_ft's; its other three
            stems are that model's own estimates, which are weaker than the specialists'.

ONNX tensor interface (both variants):
    Input  0  audio_in   [1, 2, 343980]     float32  stereo 44.1 kHz PCM, range [-1, 1]
    Output 0  stems_out  [1, 4, 2, 343980]  float32  drums, bass, other, vocals

The chunk length is read from the model's input shape by DemucsFilter.cs, which also does
the overlap-add crossfade between chunks.

Output files:
    <out>/htdemucs-ft-4stem.onnx
    <out>/htdemucs-ft-vocals.onnx

License:
    Demucs (code and weights) is released under the MIT licence.
    See https://github.com/facebookresearch/demucs
"""

from __future__ import annotations

import argparse
import math
import sys
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx

try:
    import onnx
except ImportError:
    onnx = None  # type: ignore[assignment]
try:
    from onnxsim import simplify as onnxsim_simplify
except ImportError:
    onnxsim_simplify = None

MODEL_NAME = "htdemucs_ft"
SAMPLE_RATE = 44_100
STEMS = ["drums", "bass", "other", "vocals"]


class RealStft(nn.Module):
    """torch.stft/istft as HTDemucs calls them (Hann window, normalized, center, reflect
    padding), on real tensors: a complex value is a trailing dimension of 2."""

    def __init__(self, n_fft: int, hop: int) -> None:
        """Build the forward and inverse DFT bases for one FFT size and hop."""
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        n = torch.arange(n_fft, dtype=torch.float64)
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float64)
        window = torch.hann_window(n_fft, dtype=torch.float64)
        angle = 2 * math.pi * k[:, None] * n[None, :] / n_fft
        forward = torch.cat([torch.cos(angle), -torch.sin(angle)], 0) * window / math.sqrt(n_fft)
        # One-sided inverse: the DC and Nyquist bins count once, every other bin twice.
        weight = torch.full((n_fft // 2 + 1, 1), 2.0, dtype=torch.float64)
        weight[0] = weight[-1] = 1.0
        inverse = torch.cat([torch.cos(angle) * weight, -torch.sin(angle) * weight], 0)
        inverse = inverse * window / math.sqrt(n_fft)
        self.register_buffer("forward_basis", forward.float()[:, None, :])
        self.register_buffer("inverse_basis", inverse.float()[:, None, :])
        self.register_buffer("window_sq", (window**2).float().view(1, 1, -1))

    def spectro(self, x: torch.Tensor) -> torch.Tensor:
        """[..., L] -> [..., F, T, 2]."""
        *other, length = x.shape
        x = F.pad(x.reshape(-1, 1, length), (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        y = F.conv1d(x, self.forward_basis, stride=self.hop)
        bins = self.n_fft // 2 + 1
        y = torch.stack([y[:, :bins], y[:, bins:]], -1)
        return y.view(*other, bins, y.shape[-2], 2)

    def ispectro(self, z: torch.Tensor, length: int) -> torch.Tensor:
        """[..., F, T, 2] -> [..., length]."""
        *other, bins, frames, _ = z.shape
        z = z.reshape(-1, bins, frames, 2)
        x = F.conv_transpose1d(
            torch.cat([z[..., 0], z[..., 1]], 1), self.inverse_basis, stride=self.hop
        )
        envelope = F.conv_transpose1d(
            torch.ones(1, 1, frames, dtype=x.dtype), self.window_sq, stride=self.hop
        )
        x = x / torch.clamp(envelope, min=1e-11)
        start = self.n_fft // 2
        return x[..., start : start + length].reshape(*other, length)


_SHARED_STFT: dict[tuple[int, int], RealStft] = {}


def use_real_stft(model: Any) -> None:
    """Swaps an HTDemucs instance's complex spectrogram methods for real-valued ones.

    These mirror HTDemucs._spec/_ispec/_magnitude/_mask (demucs 4.x) line for line, with
    the complex dimension carried as a trailing 2. Only the cac=True path is supported, which
    is what every htdemucs_ft model uses.
    """
    from demucs.hdemucs import pad1d

    if not model.cac:
        sys.exit("ERROR: only cac=True HTDemucs models are supported.")
    # One instance per FFT size, shared by every model: its two DFT bases are ~67 MB each, and
    # a shared module exports as one set of initializers rather than one per model.
    key = (model.nfft, model.hop_length)
    if key not in _SHARED_STFT:
        _SHARED_STFT[key] = RealStft(*key)
    model.add_module("real_stft", _SHARED_STFT[key])

    def _spec(self: Any, x: torch.Tensor) -> torch.Tensor:
        hl = self.hop_length
        le = int(math.ceil(x.shape[-1] / hl))
        pad = hl // 2 * 3
        x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")
        z = self.real_stft.spectro(x)[..., :-1, :, :]
        return z[..., 2 : 2 + le, :]

    def _ispec(self: Any, z: torch.Tensor, length: int | None = None, scale: int = 0):
        hl = self.hop_length // (4**scale)
        z = F.pad(z, (0, 0, 0, 0, 0, 1))
        z = F.pad(z, (0, 0, 2, 2))
        pad = hl // 2 * 3
        le = hl * int(math.ceil(length / hl)) + 2 * pad
        x = self.real_stft.ispectro(z, le)
        return x[..., pad : pad + length]

    def _magnitude(self: Any, z: torch.Tensor) -> torch.Tensor:
        b, c, fr, t, _ = z.shape
        return z.permute(0, 1, 4, 2, 3).reshape(b, c * 2, fr, t)

    def _mask(self: Any, z: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        b, s, _, fr, t = m.shape
        return m.view(b, s, -1, 2, fr, t).permute(0, 1, 2, 4, 5, 3)

    for name, fn in (
        ("_spec", _spec),
        ("_ispec", _ispec),
        ("_magnitude", _magnitude),
        ("_mask", _mask),
    ):
        setattr(model, name, types.MethodType(fn, model))


class DemucsChunkWrapper(nn.Module):
    """One chunk through htdemucs_ft: stem i from model i, as BagOfModels weights them."""

    def __init__(self, models: list[Any], stems: list[int]) -> None:
        """``models[j]`` supplies stem ``stems[j]``; a single model supplies all four."""
        super().__init__()
        self.models = nn.ModuleList(models)
        self.stems = stems

    def forward(self, audio_in: torch.Tensor) -> torch.Tensor:
        """[1, 2, N] -> [1, 4, 2, N] (drums, bass, other, vocals)."""
        if len(self.models) == 1:
            return self.models[0](audio_in)
        return torch.stack(
            [m(audio_in)[:, s] for m, s in zip(self.models, self.stems)], dim=1
        )


def load_bag(device: str = "cpu") -> Any:
    """Load htdemucs_ft from the demucs package."""
    try:
        from demucs.pretrained import get_model
    except ImportError as exc:
        sys.exit(f"ERROR: demucs is not installed (pip install demucs).\nDetails: {exc}")

    print(f"Loading {MODEL_NAME} (this may download weights on first run)...")
    bag = get_model(MODEL_NAME).to(device)
    bag.eval()
    if list(bag.sources) != STEMS:
        sys.exit(f"ERROR: unexpected stem order {bag.sources}; expected {STEMS}.")
    for i, row in enumerate(bag.weights):
        if [float(w) for w in row] != [1.0 if j == i else 0.0 for j in range(len(STEMS))]:
            sys.exit(f"ERROR: {MODEL_NAME}'s weights aren't one stem per model: {bag.weights}")
    return bag


def test_audio(length: int) -> torch.Tensor:
    """A deterministic stereo test signal with tonal, rhythmic and noise content."""
    g = torch.Generator().manual_seed(0)
    t = torch.arange(length) / SAMPLE_RATE
    gate = (torch.sin(2 * math.pi * 2 * t) > 0).float()
    mono = (
        0.3 * torch.sin(2 * math.pi * 110 * t)
        + 0.2 * torch.sin(2 * math.pi * 440 * t) * gate
        + 0.05 * torch.randn(length, generator=g)
    )
    return torch.stack([mono, 0.8 * mono])[None]


def export_variant(bag: Any, out_dir: Path, variant: str, simplify: bool) -> None:
    """Export one variant (4stem or vocals) to ONNX and check it against PyTorch."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vocals = STEMS.index("vocals")
    models = [bag.models[vocals]] if variant == "vocals" else list(bag.models)
    stems = [vocals] if variant == "vocals" else list(range(len(STEMS)))
    length = int(models[0].segment * models[0].samplerate)
    audio = test_audio(length)

    with torch.no_grad():
        reference = DemucsChunkWrapper(models, stems)(audio)
        for m in models:
            if not hasattr(m, "real_stft"):
                use_real_stft(m)
        wrapper = DemucsChunkWrapper(models, stems).eval()
        patched = wrapper(audio)
    drift = (reference - patched).abs().max().item()
    print(f"Real-valued STFT vs torch.stft: max abs difference {drift:.2e}")
    if drift > 1e-4:
        sys.exit("ERROR: the real-valued STFT doesn't reproduce the model; not exporting.")

    out_path = out_dir / f"htdemucs-ft-{variant}.onnx"
    print(
        f"Exporting {variant} variant to ONNX (opset 17)...\n"
        f"  Input  shape: [1, 2, {length}]\n"
        f"  Output shape: [1, {len(STEMS)}, 2, {length}]"
    )
    # nn.MultiheadAttention's inference fast path (aten::_native_multi_head_attention) has no
    # ONNX export; the ordinary path traces to plain MatMul/Softmax.
    torch.backends.mha.set_fastpath_enabled(False)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (torch.zeros(1, 2, length),),
            str(out_path),
            opset_version=17,
            input_names=["audio_in"],
            output_names=["stems_out"],
            do_constant_folding=True,
        )
    print(f"Saved: {out_path}")

    if onnx is not None and simplify and onnxsim_simplify is not None:
        print("Simplifying with onnxsim...")
        simplified, check_ok = onnxsim_simplify(onnx.load(str(out_path)))
        if check_ok:
            onnx.save(simplified, str(out_path))
            print("Simplification succeeded.")
        else:
            print("WARNING: onnxsim check failed -- keeping un-simplified model.")

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed -- skipping the ONNX Runtime check.")
        return
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    result = session.run(None, {"audio_in": audio.numpy()})[0]
    error = (torch.from_numpy(result) - patched).abs().max().item()
    print(f"ONNX Runtime vs PyTorch: max abs difference {error:.2e}")
    if error > 1e-3:
        sys.exit("ERROR: the exported model doesn't match PyTorch.")
    print(f"Done. Output: {out_path}")


def main() -> None:
    """Parse CLI arguments and run the export."""
    parser = argparse.ArgumentParser(
        description="Export Demucs htdemucs_ft to ONNX for use with StreamDock."
    )
    parser.add_argument("--out", type=Path, default=Path("demucs"), help="Output directory")
    parser.add_argument(
        "--variant",
        choices=["4stem", "vocals", "both"],
        default="both",
        help="Which variant to export (default: both)",
    )
    parser.add_argument(
        "--no-simplify", action="store_true", help="Skip the onnxsim simplification pass."
    )
    args = parser.parse_args()

    bag = load_bag(device="cpu")
    variants = ["4stem", "vocals"] if args.variant == "both" else [args.variant]
    for variant in variants:
        print(f"\n--- Exporting {variant} ---")
        export_variant(bag, args.out, variant, simplify=not args.no_simplify)
    print("\nAll done.")


if __name__ == "__main__":
    main()
