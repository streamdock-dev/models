"""
export_demucs_onnx.py -- Export Demucs htdemucs_ft to ONNX.

Usage:
    python export_demucs_onnx.py [--out DIR] [--variant {4stem,vocals}] [--no-simplify]

Requirements:
    pip install demucs onnx torch --index-url https://download.pytorch.org/whl/cpu
    pip install onnxsim  # optional, for graph simplification

The script will:
  1. Load the htdemucs_ft model weights from facebookresearch/demucs (MIT licence)
     via the demucs Python package (downloaded on first use to ~/.cache/torch/).
  2. Wrap the model in a fixed-chunk tracing module that:
       - Accepts exactly 10 s of stereo 44.1 kHz PCM  [1, 2, 441000]  float32
       - Produces one chunk of per-stem output:
           4stem variant: [1, 4, 2, 441000]  float32  (drums, bass, other, vocals)
           vocals variant: [1, 1, 2, 441000]  float32  (vocals only)
     The overlap-add crossfade is handled in C# by DemucsProcessor; this script
     exports only the single-chunk forward pass.
  3. Export the wrapper to ONNX at opset 17, optionally simplified with onnxsim.

ONNX tensor interface (both variants):
    Input  0  audio_in    [1, 2, 441000]  float32  stereo 44.1 kHz PCM, range [-1, 1]
    Output 0  stems_out   [1, S, 2, 441000]  float32  where S=4 (4stem) or S=1 (vocals)

Stem index order (4stem variant):
    0 = drums, 1 = bass, 2 = other, 3 = vocals

Output files (default):
    ThirdParty/models/demucs/htdemucs-ft-4stem.onnx
    ThirdParty/models/demucs/htdemucs-ft-vocals.onnx

License:
    Demucs is released under the MIT licence.
    See https://github.com/facebookresearch/demucs
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "ThirdParty" / "models" / "demucs"

CHUNK_SAMPLES = 441_000  # 10 s at 44.1 kHz
SAMPLE_RATE = 44_100
MODEL_NAME = "htdemucs_ft"

ALL_STEMS = ["drums", "bass", "other", "vocals"]


class DemucsChunkWrapper(nn.Module):
    """Thin tracing wrapper around a single Demucs forward chunk.

    Accepts exactly one 10-second stereo chunk and returns per-stem outputs.
    The caller (DemucsProcessor in C#) is responsible for chunking the input
    audio, applying the triangular-window overlap-add crossfade, and writing
    stem files or remixing retained stems.

    Parameters
    ----------
    model:
        A loaded Demucs model (HTDemucs instance).
    stem_indices:
        Indices into the model's source list to retain.  Pass ``None`` to
        keep all stems in their natural order.
    """

    def __init__(self, model: Any, stem_indices: list[int] | None = None) -> None:
        """Initialise the wrapper."""
        super().__init__()
        self.model = model
        self.stem_indices = stem_indices

    def forward(self, audio_in: torch.Tensor) -> torch.Tensor:
        """Run one Demucs inference chunk.

        Parameters
        ----------
        audio_in:
            Shape [1, 2, 441000], float32, stereo 44.1 kHz PCM.

        Returns
        -------
        torch.Tensor
            Shape [1, S, 2, 441000], float32 where S is the number of
            retained stems (4 for 4stem, 1 for vocals).
        """
        out: torch.Tensor = self.model(audio_in)  # [1, num_sources, 2, T]
        if self.stem_indices is not None:
            idx = torch.tensor(self.stem_indices, dtype=torch.long)
            out = out[:, idx, :, :]
        return out


def load_model(device: str = "cpu") -> Any:
    """Load htdemucs_ft from the demucs package."""
    try:
        from demucs.pretrained import get_model
    except ImportError as exc:
        print(
            "ERROR: demucs is not installed.\n"
            "Install it with:  pip install demucs\n"
            f"Details: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading {MODEL_NAME} (this may download weights on first run)...")
    model = get_model(MODEL_NAME)
    model = model.to(device)
    model.eval()
    return model


def export_variant(
    model: Any,
    out_dir: Path,
    variant: str,
    simplify: bool,
) -> None:
    """Export a single variant (4stem or vocals) to ONNX.

    Parameters
    ----------
    model:
        Loaded htdemucs_ft model.
    out_dir:
        Directory to write the .onnx file into.
    variant:
        Either ``"4stem"`` or ``"vocals"``.
    simplify:
        Whether to run onnxsim graph simplification after export.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if variant == "vocals":
        source_list: list[str] = list(model.sources)
        if "vocals" not in source_list:
            print(
                f"ERROR: {MODEL_NAME} does not have a 'vocals' stem. "
                f"Available stems: {source_list}",
                file=sys.stderr,
            )
            sys.exit(1)
        stem_indices: list[int] | None = [source_list.index("vocals")]
        out_path = out_dir / "htdemucs-ft-vocals.onnx"
        output_names = ["stems_out"]
        stem_count = 1
    else:
        stem_indices = None
        out_path = out_dir / "htdemucs-ft-4stem.onnx"
        output_names = ["stems_out"]
        stem_count = 4

    wrapper = DemucsChunkWrapper(model, stem_indices=stem_indices)
    wrapper.eval()

    dummy_input = torch.zeros(1, 2, CHUNK_SAMPLES, dtype=torch.float32)

    print(
        f"Exporting {variant} variant to ONNX (opset 17)...\n"
        f"  Input  shape: [1, 2, {CHUNK_SAMPLES}]\n"
        f"  Output shape: [1, {stem_count}, 2, {CHUNK_SAMPLES}]"
    )

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input,),
            str(out_path),
            opset_version=17,
            input_names=["audio_in"],
            output_names=output_names,
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

    print(f"Done. Output: {out_path}")


def main() -> None:
    """Parse CLI arguments and run the export."""
    parser = argparse.ArgumentParser(
        description="Export Demucs htdemucs_ft to ONNX for use with StreamDock."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--variant",
        choices=["4stem", "vocals", "both"],
        default="both",
        help="Which variant to export: '4stem', 'vocals', or 'both' (default: both)",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip the onnxsim graph simplification pass.",
    )
    args = parser.parse_args()

    model = load_model(device="cpu")

    variants_to_export: list[str]
    if args.variant == "both":
        variants_to_export = ["4stem", "vocals"]
    else:
        variants_to_export = [args.variant]

    for variant in variants_to_export:
        print(f"\n--- Exporting {variant} ---")
        export_variant(
            model=model,
            out_dir=args.out,
            variant=variant,
            simplify=not args.no_simplify,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
