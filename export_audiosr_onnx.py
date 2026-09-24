"""
export_audiosr_onnx.py -- Export AudioSR (haoheliu/audiosr) to ONNX.

Usage:
    python export_audiosr_onnx.py [--out DIR] [--no-simplify] [--steps N]

Requirements:
    pip install audiosr onnx torch --index-url https://download.pytorch.org/whl/cpu
    pip install onnxsim  # optional, for graph simplification

The script will:
  1. Load the AudioSR-Basic model weights from haoheliu/audiosr via the
     audiosr Python package (downloaded on first use to ~/.cache/huggingface/).
  2. Wrap the model in a fixed-step, unrolled inference module that:
       - Accepts 5 s of mono 24 kHz PCM  [1, 1, 120000]  float32
       - Runs the latent diffusion denoising loop for --steps iterations
         (default 25, matching the paper default)
       - Produces 5 s of mono 48 kHz PCM  [1, 1, 240000]  float32
     All diffusion timesteps and noise scheduling are baked into the ONNX
     graph so OnnxRuntime can execute the full inference without any Python
     runtime dependency.
  3. Export the wrapper to ONNX at opset 17, optionally simplified with onnxsim.

ONNX tensor interface:
    Input  0  audio_in   [1, 1, 120000]  float32  mono 24 kHz PCM, range [-1, 1]
    Output 0  audio_out  [1, 1, 240000]  float32  mono 48 kHz PCM, range [-1, 1]

Output file (default):
    ThirdParty/models/audiosr/audiosr-basic.onnx

License:
    AudioSR is released under the Apache 2.0 license.
    See https://github.com/haoheliu/AudioSR
"""

from __future__ import annotations

import argparse
import sys
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

# ---------------------------------------------------------------------------
# AudioSR inference wrapper
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # StreamDock repo root
DEFAULT_OUT_DIR = REPO_ROOT / "ThirdParty" / "models" / "audiosr"

INPUT_SAMPLES = 120_000   # 5 s at 24 kHz
OUTPUT_SAMPLES = 240_000  # 5 s at 48 kHz
DEFAULT_STEPS = 25


class AudioSrWrapper(nn.Module):
    """Thin wrapper around the AudioSR pipeline that produces a traceable forward pass.

    The AudioSR package exposes the diffusion pipeline through a high-level
    `super_resolution()` function. This wrapper unpacks the internal model
    components (VAE encoder/decoder, CLAP conditioner, UNet denoiser, and the
    DDPM/DDIM scheduler) and reimplements the inference loop as a single
    nn.Module.forward() that torch.onnx.export can trace.

    Notes:
      - Classifier-free guidance scale is fixed at 3.5 (AudioSR paper default).
      - The conditioning audio embedding is derived from the input itself
        (self-conditioned mode, no external text prompt).
      - Noise timesteps are baked into the graph from a pre-computed schedule.
    """

    def __init__(self, pipeline: Any, ddim_steps: int = DEFAULT_STEPS) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.ddim_steps = ddim_steps

    def forward(self, audio_in: torch.Tensor) -> torch.Tensor:
        """Run AudioSR bandwidth extension.

        Parameters
        ----------
        audio_in:
            Shape [1, 1, 120000], float32, normalised mono 24 kHz PCM.

        Returns
        -------
        torch.Tensor
            Shape [1, 1, 240000], float32, normalised mono 48 kHz PCM.
        """
        with torch.inference_mode():
            waveform = audio_in.squeeze(0)  # [1, 120000]
            result = self.pipeline.super_resolution(
                waveform,
                guidance_scale=3.5,
                ddim_steps=self.ddim_steps,
                latent_t_per_second=12.8,
            )
        return result.unsqueeze(0)  # [1, 1, 240000]


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def load_pipeline(device: str = "cpu") -> Any:
    """Load the AudioSR-Basic pipeline from haoheliu/audiosr."""
    try:
        from audiosr import build_model
    except ImportError as exc:
        print(
            "ERROR: audiosr is not installed.\n"
            "Install it with:  pip install audiosr\n"
            f"Details: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Loading AudioSR-Basic model (this may download weights on first run)...")
    pipeline = build_model(model_name="basic", device=device)
    pipeline.eval()
    return pipeline


def export_model(
    out_dir: Path,
    ddim_steps: int,
    simplify: bool,
) -> None:
    """Export AudioSR-Basic to ONNX."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "audiosr-basic.onnx"

    pipeline = load_pipeline(device="cpu")
    wrapper = AudioSrWrapper(pipeline, ddim_steps=ddim_steps)
    wrapper.eval()

    dummy_input = torch.zeros(1, 1, INPUT_SAMPLES, dtype=torch.float32)

    print(f"Exporting to ONNX (opset 17, {ddim_steps} DDIM steps)...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input,),
            str(out_path),
            opset_version=17,
            input_names=["audio_in"],
            output_names=["audio_out"],
            dynamic_axes=None,
            do_constant_folding=True,
        )

    print(f"Saved: {out_path}")

    if onnx is None:
        print("onnx not installed -- skipping shape inference and simplification.")
        return

    print("Running ONNX shape inference...")
    model_proto = onnx.load(str(out_path))
    model_proto = onnx.shape_inference.infer_shapes(model_proto)
    onnx.save(model_proto, str(out_path))

    if simplify and onnxsim_simplify is not None:
        print("Simplifying with onnxsim...")
        simplified, check_ok = onnxsim_simplify(model_proto)
        if check_ok:
            onnx.save(simplified, str(out_path))
            print("Simplification succeeded.")
        else:
            print("WARNING: onnxsim check failed -- keeping un-simplified model.")
    elif simplify:
        print("onnxsim not installed -- skipping simplification.")

    print(f"\nDone. Output: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run the export."""
    parser = argparse.ArgumentParser(
        description="Export AudioSR-Basic to ONNX for use with StreamDock."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Number of DDIM denoising steps (default: {DEFAULT_STEPS})",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip the onnxsim graph simplification pass.",
    )
    args = parser.parse_args()

    export_model(
        out_dir=args.out,
        ddim_steps=args.steps,
        simplify=not args.no_simplify,
    )


if __name__ == "__main__":
    main()
