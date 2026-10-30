"""
export_deepfilter_onnx.py -- Export DeepFilterNet3 to a single-frame streaming ONNX model.

Usage:
    python export_deepfilter_onnx.py [--torchdf-repo DIR] [--out DIR]

Requirements:
    Python 3.8-3.11 (DeepFilterNet's deepfilterlib has no wheel for 3.12+; building it needs
    Rust), then:
    pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<2.6" "torchaudio>=2.2,<2.6"
    pip install deepfilternet onnx onnxruntime

    (deepfilternet supplies the Rust feature library and the weights download; the Python
    model code comes from the fork below.)

    The streaming model comes from torchDF, a pure-PyTorch streaming port of DeepFilterNet3
    in grazder's DeepFilterNet fork (same MIT/Apache-2.0 dual licence as upstream), which its
    author tested against DeepFilterNet's own enhance():
        git clone https://github.com/grazder/DeepFilterNet
        git -C DeepFilterNet checkout 1097015d53ced78fb234e7d7071a5dd4446e3952
    and is found with --torchdf-repo (default: ./DeepFilterNet).

The script will:
  1. Load DeepFilterNet3's weights through the deepfilternet package (downloaded on first use).
  2. Build torchDF's ExportableStreamingTorchDF around them: STFT analysis, ERB and
     complex-spectrum features with their running normalisation, the encoder's conv
     lookahead buffers, every GRU, the ERB gains, the deep filter's frame history, and ISTFT
     overlap-add -- all carried between calls in one flat state vector.
  3. Wrap it in the interface DeepFilterNetFilter.cs expects (below): the frame as [1, 1, 480],
     the attenuation limit fixed at 0 dB (no limit, as DeepFilterNet's own default), and
     torchDF's per-frame SNR estimate dropped, so the remaining input/output pairs are
     exactly the state.
  4. Export with TorchScript (the model skips stages on silent frames, which is data-dependent
     control flow), check ONNX Runtime against PyTorch frame by frame, and compare the
     streamed result with DeepFilterNet's offline enhance() on a real recording.

ONNX tensor interface (DeepFilterNetFilter.cs pairs input i with output i for i >= 1):
    Input  0  noisy       [1, 1, 480]  float32  mono 48 kHz PCM
    Input  1  states      [S]          float32  zeros on the first frame
    Output 0  enhanced    [1, 1, 480]  float32  denoised mono PCM, 480 samples (10 ms) late
    Output 1  new_states  [S]          float32

Output file:
    <out>/deepfilternet3.onnx

License:
    DeepFilterNet (code and weights) is dual-licensed MIT / Apache-2.0:
    https://github.com/Rikorose/DeepFilterNet
    torchDF is part of grazder's fork of it, under the same licence:
    https://github.com/grazder/DeepFilterNet/tree/torchDF-changes/torchDF
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

FRAME_SIZE = 480
SAMPLE_RATE = 48_000
TORCHDF_COMMIT = "1097015d53ced78fb234e7d7071a5dd4446e3952"
OUT_NAME = "deepfilternet3.onnx"


class StreamingFrame(nn.Module):
    """torchDF's streaming step in DeepFilterNetFilter.cs's shape."""

    def __init__(self, streaming: nn.Module) -> None:
        """Wrap a (scripted) ExportableStreamingTorchDF."""
        super().__init__()
        self.streaming = streaming
        self.register_buffer("atten_lim_db", torch.zeros(()))

    def forward(self, noisy: torch.Tensor, states: torch.Tensor):
        """[1, 1, 480], [S] -> [1, 1, 480], [S]."""
        enhanced, new_states, _ = self.streaming(noisy.reshape(-1), states, self.atten_lim_db)
        return enhanced.reshape(1, 1, -1), new_states


def load_pipeline(repo: Path):
    """Import torchDF from the fork's checkout and build its pipeline."""
    torchdf = repo / "torchDF"
    if not (torchdf / "torch_df_streaming.py").exists():
        sys.exit(
            f"ERROR: {torchdf / 'torch_df_streaming.py'} not found. Clone the fork with:\n"
            "  git clone https://github.com/grazder/DeepFilterNet\n"
            f"  git -C DeepFilterNet checkout {TORCHDF_COMMIT}"
        )
    # torchDF needs the fork's own copy of DeepFilterNet's Python package (its Encoder keeps
    # attributes the published deepfilternet package doesn't), so that goes first on the path.
    sys.path.insert(0, str((repo / "DeepFilterNet").resolve()))
    sys.path.insert(0, str(torchdf.resolve()))
    from torch_df_streaming import TorchDFPipeline  # type: ignore[import-not-found]

    return TorchDFPipeline(always_apply_all_stages=False, device="cpu")


def stream(run, audio: np.ndarray, state_len: int) -> np.ndarray:
    """Run a frame function over ``audio`` the way DeepFilterNetFilter.cs does."""
    states = np.zeros(state_len, dtype=np.float32)
    frames = []
    padded = np.pad(audio, (0, -len(audio) % FRAME_SIZE))
    for i in range(0, len(padded), FRAME_SIZE):
        out, states = run(padded[i : i + FRAME_SIZE].reshape(1, 1, -1), states)
        frames.append(out.reshape(-1))
    return np.concatenate(frames)[: len(audio)]


def test_recording(repo: Path) -> np.ndarray:
    """The noisy speech sample shipped with torchDF, as mono 48 kHz float32."""
    import torchaudio

    path = next((repo / "torchDF" / "examples").glob("*.wav"))
    audio, sr = torchaudio.load(str(path))
    audio = audio.mean(0)
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    return audio.numpy().astype(np.float32)


def export(repo: Path, out_dir: Path) -> None:
    """Build, export and verify the streaming model."""
    pipeline = load_pipeline(repo)
    state_len = int(pipeline.states.numel())
    wrapper = StreamingFrame(torch.jit.script(pipeline.torch_streaming_model)).eval()
    scripted = torch.jit.script(wrapper)
    print(f"State vector: {state_len} floats")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / OUT_NAME
    example = (torch.zeros(1, 1, FRAME_SIZE), torch.zeros(state_len))
    torch.onnx.export(
        scripted,
        example,
        str(out_path),
        input_names=["noisy", "states"],
        output_names=["enhanced", "new_states"],
        opset_version=14,
    )
    print(f"Saved: {out_path}")

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed -- skipping verification.")
        return

    audio = test_recording(repo)
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])

    def run_onnx(frame, states):
        return session.run(None, {"noisy": frame, "states": states})

    def run_torch(frame, states):
        with torch.no_grad():
            out, new = scripted(torch.from_numpy(frame), torch.from_numpy(states))
        # States pass straight through some scripted paths, keeping requires_grad.
        return out.detach().numpy(), new.detach().numpy()

    onnx_out = stream(run_onnx, audio, state_len)
    torch_out = stream(run_torch, audio, state_len)
    error = float(np.abs(onnx_out - torch_out).max())
    print(
        f"ONNX Runtime vs PyTorch, {len(audio) / SAMPLE_RATE:.1f} s streamed: "
        f"max abs difference {error:.2e}"
    )
    if error > 1e-3:
        sys.exit("ERROR: the exported model doesn't match PyTorch.")

    # DeepFilterNet's offline enhance() (the fork's copy, already on the path), aligned by the
    # 480-sample streaming delay.
    from df.enhance import enhance, init_df

    model, df_state, _ = init_df(config_allow_defaults=True)
    reference = enhance(model, df_state, torch.from_numpy(audio)[None]).numpy().reshape(-1)
    delay = FRAME_SIZE
    a, b = onnx_out[delay:], reference[: len(audio) - delay]
    snr = 10 * np.log10(np.sum(b**2) / max(np.sum((a - b) ** 2), 1e-12))
    print(f"Streamed vs DeepFilterNet's enhance(): {snr:.1f} dB agreement")
    if snr < 20:
        sys.exit("ERROR: the streamed output doesn't match DeepFilterNet's offline result.")
    print(f"Done. Output: {out_path}")


def main() -> None:
    """Parse CLI arguments and run the export."""
    parser = argparse.ArgumentParser(
        description="Export DeepFilterNet3 to a streaming ONNX model for StreamDock."
    )
    parser.add_argument(
        "--torchdf-repo",
        type=Path,
        default=Path("DeepFilterNet"),
        help="Checkout of grazder/DeepFilterNet (default: ./DeepFilterNet)",
    )
    parser.add_argument("--out", type=Path, default=Path("deepfilter"), help="Output directory")
    args = parser.parse_args()
    export(args.torchdf_repo, args.out)


if __name__ == "__main__":
    main()
