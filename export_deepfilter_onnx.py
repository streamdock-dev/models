"""
export_deepfilter_onnx.py -- Export DeepFilterNet3 to a streaming ONNX model.

Usage:
    python export_deepfilter_onnx.py [--out DIR] [--no-simplify]

Requirements:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install deepfilter onnx onnxsim

The script will:
  1. Download DeepFilterNet3 weights automatically via the deepfilter package.
  2. Discover and record all nn.GRU modules in the model and their hidden-state shapes.
  3. Wrap the model in a streaming single-frame module that:
       - Accepts raw mono PCM at 48 kHz (480 samples, 10 ms) as the audio input.
       - Includes STFT analysis (Hann window, 960-sample FFT, 480-sample hop) inside
         the ONNX graph using torch.fft.rfft so OnnxRuntime can execute it without
         any pre-processing by the caller.
       - Exposes every GRU hidden state as an explicit ONNX input/output pair so
         OnnxRuntime sessions remain stateless -- the C# caller maintains and passes
         back the state tensors on each Run() call.
       - Includes ISTFT synthesis via torch.fft.irfft and single-frame overlap-add.
  4. Export the wrapper to ONNX at opset 17, optionally simplified with onnxsim.

ONNX tensor interface (DeepFilterNetProcessor.cs detects these automatically):
    Input  0  noisy         [1, 1, 480]       float32  raw mono PCM (48 kHz)
    Input  1  stft_buf      [1, 480]          float32  previous-frame STFT overlap
    Input  2  gru_h_0       [L0, 1, H0]       float32  GRU 0 hidden state
    ...        ...
    Input  n  istft_buf     [1, 480]          float32  ISTFT overlap-add buffer

    Output 0  enhanced      [1, 1, 480]       float32  denoised mono PCM
    Output 1  stft_buf_out  [1, 480]          float32
    Output 2  gru_h_0_out   [L0, 1, H0]       float32
    ...
    Output n  istft_buf_out [1, 480]          float32

The C# DeepFilterNetProcessor.cs pairs inputs[i] with outputs[i] for all i >= 1 as
hidden state pairs, initialising them to zero on first use and threading them back
each frame. This gives correct GRU state continuity across frames without any change
to the C# runtime.

Reconstruction note:
    The Hann window satisfies the COLA condition at 50% overlap:
        w[n] + w[n + 480] = 1  for all n in [0, 480]
    Therefore synthesis uses no additional window (just irfft + OLA), which gives
    perfect reconstruction for an identity model.

Output file (default):
    ThirdParty/models/deepfilter/deepfilternet3.onnx
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.onnx

try:
    import onnx
    from onnxsim import simplify as onnxsim_simplify
except ImportError:
    onnx = None  # type: ignore[assignment]
    onnxsim_simplify = None

# Thread-local context for GRU state injection during tracing.
_tls = threading.local()

# Stash for the original nn.GRU forward (set in main before patching).
_orig_gru_forward: Any = None


def _patched_gru_forward(
    self: nn.GRU,
    input: torch.Tensor,
    hx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Replacement for nn.GRU.forward used during ONNX tracing.

    When a streaming context is active (_tls.ctx is set), this function:
      - Ignores any hidden state the model passes (self.hidden, None, etc.).
      - Injects the explicit state tensor from the wrapper's forward arguments.
      - Captures the output hidden state into _tls.ctx['h_out'].

    When no context is active (probing pass), it falls through to the original
    GRU forward, which records which GRU is called and in what order.
    """
    ctx: dict[str, Any] | None = getattr(_tls, "ctx", None)

    if ctx is None:
        # Probing pass: record this GRU module if not yet seen.
        registry: list[tuple[nn.GRU, tuple[int, ...]]] | None = getattr(
            _tls, "registry", None
        )
        if registry is not None:
            if self not in [r[0] for r in registry]:
                # Determine hidden state shape from GRU attributes.
                # GRU hidden state: [num_layers * num_directions, batch, hidden_size]
                shape = (self.num_layers, 1, self.hidden_size)
                registry.append((self, shape))
        return _orig_gru_forward(self, input, hx)

    # Streaming trace: override hx with explicit state tensor.
    idx: int = ctx["idx"]
    h_in: torch.Tensor = ctx["h_in"][idx]
    out, h_out = _orig_gru_forward(self, input, h_in)
    ctx["h_out"].append(h_out)
    ctx["idx"] += 1
    return out, h_out


def probe_gru_order(
    model: nn.Module,
    dummy_spec: torch.Tensor,
) -> list[tuple[nn.GRU, tuple[int, ...]]]:
    """
    Run a single model forward pass to discover GRU modules in call order.

    Returns a list of (nn.GRU module, hidden_state_shape) tuples in the order
    they are called during a normal forward pass.
    """
    _tls.registry = []
    with torch.no_grad():
        model(dummy_spec)
    result = list(_tls.registry)
    del _tls.registry
    return result


class DeepFilterStreamingWrapper(nn.Module):
    """
    Single-frame streaming wrapper for DeepFilterNet3.

    All GRU hidden states plus the STFT and ISTFT overlap buffers are explicit
    ONNX inputs/outputs so that OnnxRuntime sessions are fully stateless.
    """

    def __init__(
        self,
        df_model: nn.Module,
        gru_order: list[tuple[nn.GRU, tuple[int, ...]]],
        n_fft: int,
        hop: int,
    ) -> None:
        super().__init__()
        self.df_model = df_model
        self.n_fft = n_fft
        self.hop = hop
        self.n_gru = len(gru_order)
        # Hann analysis window stored as a buffer so it is embedded in the ONNX graph.
        self.register_buffer("analysis_win", torch.hann_window(n_fft))

    def forward(
        self,
        noisy: torch.Tensor,     # [1, 1, 480]  raw mono PCM
        stft_buf: torch.Tensor,  # [1, 480]     overlap from previous frame
        gru_states: tuple,       # (h0, h1, ...) one tensor per GRU module
        istft_buf: torch.Tensor, # [1, 480]     ISTFT OLA buffer
    ) -> tuple[torch.Tensor, torch.Tensor, tuple, torch.Tensor]:
        """
        Process one 480-sample frame of mono audio through DeepFilterNet3.

        Analysis:
            Concatenate stft_buf and noisy to form a 960-sample window.
            Apply Hann window then torch.fft.rfft to get the complex spectrum.

        Model:
            Run DeepFilterNet3 on the spectrum, threading GRU hidden states
            through the global patch so they are represented as explicit traced
            tensors in the ONNX graph.

        Synthesis:
            Apply torch.fft.irfft (no additional synthesis window -- Hann COLA
            guarantees sum-of-windows = 1 at 50% overlap).
            Return first 480 samples plus the overlap-add buffer as the enhanced
            output, carry the second 480 samples as the next frame's OLA buffer.

        Returns:
            enhanced      [1, 1, 480]  denoised PCM
            stft_buf_out  [1, 480]     updated STFT overlap (= current noisy frame)
            gru_states_out tuple        updated GRU hidden states
            istft_buf_out [1, 480]     updated ISTFT OLA buffer
        """
        hop = self.hop
        n_fft = self.n_fft

        # Analysis: form the full analysis window from overlap + current frame.
        # noisy: [1, 1, 480] -> squeeze channel dim -> [1, 480]
        noisy_squeezed = noisy.squeeze(1)  # [1, 480]
        stft_input = torch.cat([stft_buf, noisy_squeezed], dim=-1)  # [1, 960]
        windowed = stft_input * self.analysis_win  # [1, 960]

        # rfft: [1, 960] -> [1, 481] complex
        spec_flat = torch.fft.rfft(windowed, n=n_fft)  # [1, 481]

        # Reshape to [B=1, C=1, F=481, T=1] as expected by DfNet.forward.
        spec = spec_flat.unsqueeze(1).unsqueeze(-1)  # [1, 1, 481, 1]

        # Set up GRU state injection context.
        _tls.ctx = {"idx": 0, "h_in": list(gru_states), "h_out": []}

        # Run the DeepFilterNet3 model (GRU states are injected/captured by the patch).
        enhanced_spec = self.df_model(spec)  # [1, 1, 481, 1]

        # Collect updated GRU states.
        gru_states_out: tuple = tuple(_tls.ctx["h_out"])
        _tls.ctx = None

        # Synthesis: ISTFT via irfft + single-frame OLA.
        # enhanced_spec: [1, 1, 481, 1] complex -> [1, 481]
        espec_flat = enhanced_spec.squeeze(1).squeeze(-1)  # [1, 481]
        frame_out = torch.fft.irfft(espec_flat, n=n_fft)  # [1, 960]

        # OLA: output is first 480 samples plus the overlap buffer from the previous frame.
        enhanced = frame_out[:, :hop] + istft_buf  # [1, 480]
        new_istft_buf = frame_out[:, hop:]  # [1, 480]

        # Update STFT overlap: current frame becomes the next frame's overlap input.
        new_stft_buf = noisy_squeezed  # [1, 480]

        # Reshape enhanced to [1, 1, 480] to match the NCT format C# probes for.
        enhanced_out = enhanced.unsqueeze(1)  # [1, 1, 480]

        return enhanced_out, new_stft_buf, gru_states_out, new_istft_buf


def load_df_model() -> tuple[nn.Module, Any]:
    """
    Load DeepFilterNet3 via the deepfilter package.

    Returns (model, df_state).  The model's .forward(spec) accepts a complex
    [B, 1, F, T] spectrogram and returns an enhanced complex spectrogram of
    the same shape.
    """
    try:
        from df import init_df  # type: ignore[import]
    except ImportError:
        print("ERROR: deepfilter package not found.")
        print("Install it with:  pip install deepfilter")
        sys.exit(1)

    print("  Loading DeepFilterNet3 weights (downloads on first run) ...", end=" ", flush=True)
    model, df_state, _ = init_df()
    model.eval()
    print("done.")
    return model, df_state


def build_dummy_spec(n_fft: int) -> torch.Tensor:
    """Return a dummy complex spectrogram [1, 1, n_fft//2+1, 1] for probing."""
    n_bins = n_fft // 2 + 1
    return torch.zeros(1, 1, n_bins, 1, dtype=torch.complex64)


def export(out_path: Path, simplify: bool) -> None:
    """Full export pipeline: load, probe, wrap, export, optionally simplify."""
    global _orig_gru_forward

    n_fft = 960  # 20 ms window at 48 kHz
    hop = 480    # 10 ms hop = frame size

    # Step 1: Load model.
    print("\nStep 1: Loading DeepFilterNet3 ...")
    model, df_state = load_df_model()

    # Step 2: Patch nn.GRU.forward before the probing pass.
    print("\nStep 2: Discovering GRU modules and hidden-state shapes ...")
    _orig_gru_forward = nn.GRU.forward
    nn.GRU.forward = _patched_gru_forward  # type: ignore[method-assign]

    try:
        dummy_spec = build_dummy_spec(n_fft)
        with torch.no_grad():
            gru_order = probe_gru_order(model, dummy_spec)
    finally:
        # Always restore, even on error.
        nn.GRU.forward = _orig_gru_forward  # type: ignore[method-assign]

    if not gru_order:
        print("WARNING: No nn.GRU modules found in the model forward pass.")
        print("         The exported model will have no hidden-state inputs/outputs.")
        print("         GRU state will reset to zero on every frame (poor quality).")
    else:
        print(f"  Found {len(gru_order)} GRU module(s):")
        for i, (gru_mod, shape) in enumerate(gru_order):
            print(f"    GRU {i}: num_layers={gru_mod.num_layers} hidden_size={gru_mod.hidden_size} -> state shape {shape}")

    # Step 3: Build the streaming wrapper.
    print("\nStep 3: Building streaming wrapper ...")
    wrapper = DeepFilterStreamingWrapper(model, gru_order, n_fft, hop)
    wrapper.eval()

    # Step 4: Build dummy inputs for the ONNX export trace.
    print("\nStep 4: Building dummy inputs ...")
    dummy_noisy = torch.zeros(1, 1, hop)
    dummy_stft_buf = torch.zeros(1, hop)
    dummy_gru_states = tuple(torch.zeros(*shape) for _, shape in gru_order)
    dummy_istft_buf = torch.zeros(1, hop)

    # ONNX input/output names.
    n_states = len(gru_order)
    input_names = ["noisy", "stft_buf"] + [f"gru_h_{i}" for i in range(n_states)] + ["istft_buf"]
    output_names = ["enhanced", "stft_buf_out"] + [f"gru_h_{i}_out" for i in range(n_states)] + ["istft_buf_out"]

    print(f"  ONNX inputs : {input_names}")
    print(f"  ONNX outputs: {output_names}")

    # Step 5: Re-patch nn.GRU.forward for the export trace.
    print("\nStep 5: Exporting to ONNX ...")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    nn.GRU.forward = _patched_gru_forward  # type: ignore[method-assign]

    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (dummy_noisy, dummy_stft_buf, dummy_gru_states, dummy_istft_buf),
                str(out_path),
                input_names=input_names,
                output_names=output_names,
                opset_version=17,
                do_constant_folding=True,
                dynamo=False,
            )
    finally:
        nn.GRU.forward = _orig_gru_forward  # type: ignore[method-assign]

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  Written {out_path.name} ({size_mb:.1f} MB)")

    # Step 6: Verify the exported model loads correctly.
    print("\nStep 6: Verifying ONNX model ...")
    if onnx is not None:
        model_proto = onnx.load(str(out_path))
        onnx.checker.check_model(model_proto)
        print("  ONNX checker: OK")
    else:
        print("  onnx package not installed -- skipping check.")

    # Step 7: Optionally simplify.
    if simplify:
        if onnxsim_simplify is None:
            print("\nStep 7: [skip] onnxsim not installed -- skipping simplification.")
        elif onnx is None:
            print("\nStep 7: [skip] onnx not installed -- skipping simplification.")
        else:
            print("\nStep 7: Simplifying with onnxsim ...", end=" ", flush=True)
            model_proto = onnx.load(str(out_path))
            model_simplified, ok = onnxsim_simplify(model_proto)
            if ok:
                onnx.save(model_simplified, str(out_path))
                size_mb = out_path.stat().st_size / (1024 * 1024)
                print(f"done ({size_mb:.1f} MB after simplification).")
            else:
                print("simplification check failed, keeping original.")


def verify_reconstruction(n_fft: int = 960, hop: int = 480, frames: int = 20) -> None:
    """
    Smoke-test the STFT/OLA loop with an identity model to confirm perfect reconstruction.
    Prints the max absolute error; any value < 1e-5 is acceptable.
    """
    win = torch.hann_window(n_fft)
    signal = torch.rand(frames * hop)  # random test signal

    stft_buf = torch.zeros(hop)
    istft_buf = torch.zeros(hop)
    reconstructed = torch.zeros_like(signal)

    for k in range(frames):
        frame_in = signal[k * hop : k * hop + hop]
        combined = torch.cat([stft_buf, frame_in])
        windowed = combined * win

        spec = torch.fft.rfft(windowed, n=n_fft)
        # Identity model: no change to spec.
        frame_out = torch.fft.irfft(spec, n=n_fft)

        # OLA (no synthesis window -- Hann COLA ensures sum-to-1).
        enhanced = frame_out[:hop] + istft_buf
        reconstructed[k * hop : k * hop + hop] = enhanced

        istft_buf = frame_out[hop:].clone()
        stft_buf = frame_in.clone()

    # The first frame has no valid history, so skip it in the error check.
    err = (reconstructed[hop:] - signal[hop:]).abs().max().item()
    status = "OK" if err < 1e-5 else "FAIL"
    print(f"  STFT/OLA identity reconstruction error: {err:.2e}  [{status}]")


def main() -> None:
    """Main entry point: parse args, run export."""
    default_out = (
        Path(__file__).resolve().parent / "deepfilter" / "deepfilternet3.onnx"
    )

    parser = argparse.ArgumentParser(
        description="Export DeepFilterNet3 to streaming ONNX for use with DeepFilterNetProcessor.cs"
    )
    parser.add_argument(
        "--out",
        default=str(default_out),
        help=f"Output ONNX path (default: {default_out})",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip onnxsim simplification pass",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Run STFT/OLA reconstruction smoke test then exit (no model download)",
    )
    args = parser.parse_args()

    print("DeepFilterNet3 ONNX streaming exporter")
    print()

    if args.verify_only:
        print("Running STFT/OLA reconstruction smoke test ...")
        verify_reconstruction()
        return

    # Smoke-test the STFT/OLA implementation before attempting the full export.
    print("Pre-flight: STFT/OLA reconstruction smoke test ...")
    verify_reconstruction()

    export(Path(args.out), simplify=not args.no_simplify)

    print()
    print(f"Done. Model written to: {args.out}")
    print()
    print("To use in StreamDock:")
    print("  Set DEEP_FILTER_MODEL_PATH in app.local.cfg to the path above, or")
    print("  copy the file to ThirdParty/models/deepfilter/deepfilternet3.onnx")
    print("  relative to the StreamDock installation directory.")


if __name__ == "__main__":
    main()
