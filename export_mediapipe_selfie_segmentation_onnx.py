"""
export_mediapipe_selfie_segmentation_onnx.py -- Convert Google's MediaPipe "Selfie
Segmentation" TFLite model to ONNX with a stable, agreed-upon tensor-name contract.

Usage:
    python src/Scripts/export_mediapipe_selfie_segmentation_onnx.py [--out DIR] [--landscape]

Requirements:
    pip install tf2onnx onnx tensorflow

The script will:
  1. Download the official MediaPipe Selfie Segmentation TFLite model (general 256x256
     variant by default; --landscape for the 256x144 variant) from Google's model
     hosting.
  2. Convert it to ONNX via tf2onnx's TFLite importer.
  3. Rename the model's single input and single mask output to "input"/"mask" -- the
     raw TFLite -> ONNX conversion produces auto-generated Keras-style names (they can
     also change between model revisions), so this step is what gives
     SelfieSegmentationFilter.cs a name it can rely on regardless of upstream naming.

ONNX tensor interface (agreed with SelfieSegmentationFilter.cs):
    Inputs:
        "input"  [1, H, W, 3]  float32  normalised RGB [0, 1], NHWC (channel-last --
                                         note this differs from RVM's NCHW convention;
                                         it's TFLite's native layout, kept as-is rather
                                         than transposed, since ONNX Runtime handles
                                         NHWC just as well and transposing would only
                                         add a needless per-frame cost).
    Outputs:
        "mask"   [1, H, W, 1]  float32  foreground probability [0, 1]
                 (or [1, H, W, 2] on some model revisions -- a background/foreground
                 softmax pair; SelfieSegmentationFilter.cs handles both shapes, using
                 channel 1 as foreground for the 2-channel case).

H/W are 256/256 (general) or 256/144 (landscape) -- SelfieSegmentationFilter.cs reads
the actual exported shape from the model's own metadata rather than hardcoding either,
so this script doesn't need to record which variant was exported anywhere else.

NOTE ON THE DOWNLOAD URL: MediaPipe's model hosting has moved at least once (from the
legacy "Solutions" API's mediapipe-assets bucket to the newer Tasks API's
mediapipe-models bucket). The URL below is the best known at the time this script was
written -- if it 404s, check https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter
for the current location and update DEFAULT_MODEL_URL/LANDSCAPE_MODEL_URL below.

Models are discovered automatically by ModelCatalog.DiscoverForPipeline(Matting).

License:
    The MediaPipe Selfie Segmentation model is released under the Apache-2.0 licence
    (as part of the MediaPipe project). See https://github.com/google-ai-edge/mediapipe.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

try:
    import onnx
except ImportError:
    onnx = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "ThirdParty" / "models" / "mediapipe-selfie"
TFLITE_CACHE_DIR = Path.home() / ".cache" / "mediapipe-selfie"

# General model (256x256, best all-around quality/speed tradeoff for a webcam-framed subject).
GENERAL_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
# Landscape model (256x144, tuned for wide/landscape framing).
LANDSCAPE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite"

INPUT_NAME = "input"
MASK_NAME = "mask"


def download_model(url: str, dest: Path) -> None:
    """Download the TFLite model to dest, creating parent dirs as needed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model from {url} -> {dest}")
    urllib.request.urlretrieve(url, str(dest))
    print("Download complete.")


def convert_to_onnx(tflite_path: Path, onnx_path: Path) -> None:
    """Convert a TFLite model to ONNX via tf2onnx's command-line TFLite importer."""
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Converting {tflite_path} -> {onnx_path} ...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tf2onnx.convert",
            "--tflite",
            str(tflite_path),
            "--output",
            str(onnx_path),
            "--opset",
            "17",
        ],
        check=True,
    )
    print("Conversion complete.")


def replace_transpose_conv_bias(onnx_path: Path) -> None:
    """Replaces MediaPipe's custom ``Convolution2DTransposeBias`` op with standard ONNX ops.

    The selfie segmenter's final 2x upsample is a MediaPipe custom TFLite op that tf2onnx
    can't map, so it passes through as an unrunnable ``TFL_Convolution2DTransposeBias`` node
    (and onnx.checker rejects the model). The op is a transposed convolution plus bias --
    stride 2, SAME padding, which for a 2x2 kernel means no padding at all. Its input is
    always a Transpose of an NCHW tensor to NHWC (tf2onnx keeps NHWC at a custom op's
    boundary), so this feeds that NCHW tensor straight into an ONNX ConvTranspose and
    transposes the result back to NHWC under the custom op's output name. The weight
    goes from TFLite's OHWI to ConvTranspose's (C_in, C_out, kH, kW).
    """
    if onnx is None:
        sys.exit("ERROR: the onnx package is required: pip install onnx")
    from onnx import helper, numpy_helper

    model = onnx.load(str(onnx_path))
    graph = model.graph
    nodes = [n for n in graph.node if n.op_type == "TFL_Convolution2DTransposeBias"]
    if not nodes:
        return
    init = {i.name: i for i in graph.initializer}
    producers = {o: n for n in graph.node for o in n.output}

    for node in nodes:
        data, weight_name, bias_name = node.input
        to_nhwc = producers[data]
        if to_nhwc.op_type != "Transpose" or list(to_nhwc.attribute[0].ints) != [0, 2, 3, 1]:
            sys.exit(f"ERROR: {node.name}'s input isn't an NCHW->NHWC Transpose; can't patch.")
        nchw_input = to_nhwc.input[0]

        weight = numpy_helper.to_array(init[weight_name]).transpose(3, 0, 1, 2).copy()
        bias = numpy_helper.to_array(init[bias_name]).reshape(-1).copy()
        kh, kw = weight.shape[2], weight.shape[3]
        graph.initializer.append(numpy_helper.from_array(weight, f"{node.name}_W"))
        graph.initializer.append(numpy_helper.from_array(bias, f"{node.name}_B"))

        index = list(graph.node).index(node)
        conv_out = f"{node.name}_nchw"
        graph.node.remove(node)
        graph.node.insert(
            index,
            helper.make_node(
                "ConvTranspose",
                [nchw_input, f"{node.name}_W", f"{node.name}_B"],
                [conv_out],
                name=f"{node.name}_convtranspose",
                kernel_shape=[kh, kw],
                strides=[2, 2],
            ),
        )
        graph.node.insert(
            index + 1,
            helper.make_node(
                "Transpose",
                [conv_out],
                list(node.output),
                name=f"{node.name}_to_nhwc",
                perm=[0, 2, 3, 1],
            ),
        )
        print(f"Replaced {node.name} (Convolution2DTransposeBias) with ConvTranspose.")

    # Drop the now-unused Transpose and the custom op's initializers.
    used = {i for n in graph.node for i in n.input} | {o.name for o in graph.output}
    for n in [n for n in graph.node if not any(o in used for o in n.output)]:
        graph.node.remove(n)
    used = {i for n in graph.node for i in n.input}
    for i in [i for i in graph.initializer if i.name not in used]:
        graph.initializer.remove(i)

    onnx.save(model, str(onnx_path))


def rename_io(onnx_path: Path) -> None:
    """Renames the model's sole input and sole mask output to "input"/"mask".

    tf2onnx's TFLite importer preserves TFLite's own auto-generated (Keras-style)
    tensor names, which aren't stable across MediaPipe model revisions --
    SelfieSegmentationFilter.cs needs a fixed contract to look up, so this renames
    every reference to the graph's first input and first output throughout the graph
    (node inputs/outputs, not just the top-level graph.input/output entries, since ONNX
    graphs reference tensors purely by name string).
    """
    if onnx is None:
        print(
            "WARNING: onnx package not installed -- skipping the input/output rename step. "
            f"SelfieSegmentationFilter.cs falls back to the model's own first input/output name "
            "in that case, so this is non-fatal, but re-run with `pip install onnx` for a "
            "stable, revision-independent contract.",
            file=sys.stderr,
        )
        return

    model = onnx.load(str(onnx_path))
    graph = model.graph

    old_input_name = graph.input[0].name
    old_output_name = graph.output[0].name

    def rename_all(old: str, new: str) -> None:
        if old == new:
            return
        for node in graph.node:
            node.input[:] = [new if n == old else n for n in node.input]
            node.output[:] = [new if n == old else n for n in node.output]
        for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
            if value_info.name == old:
                value_info.name = new

    rename_all(old_input_name, INPUT_NAME)
    rename_all(old_output_name, MASK_NAME)

    onnx.checker.check_model(model)
    onnx.save(model, str(onnx_path))
    print(f"Renamed input '{old_input_name}' -> '{INPUT_NAME}', output '{old_output_name}' -> '{MASK_NAME}'.")


def export(out_dir: Path, landscape: bool) -> None:
    """Run the full pipeline: download TFLite -> convert to ONNX -> rename I/O."""
    variant = "landscape" if landscape else "general"
    url = LANDSCAPE_MODEL_URL if landscape else GENERAL_MODEL_URL
    tflite_path = TFLITE_CACHE_DIR / f"selfie_segmenter_{variant}.tflite"

    if not tflite_path.exists():
        download_model(url, tflite_path)
    else:
        print(f"Using cached TFLite model at {tflite_path}")

    onnx_path = out_dir / "selfie-segmentation.onnx"
    convert_to_onnx(tflite_path, onnx_path)
    replace_transpose_conv_bias(onnx_path)
    rename_io(onnx_path)

    print(f"Done: {onnx_path}")


def main() -> None:
    """Parse command-line arguments and run the export."""
    parser = argparse.ArgumentParser(
        description="Convert MediaPipe Selfie Segmentation (TFLite) to ONNX."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--landscape",
        action="store_true",
        help="Use the 256x144 landscape-framed variant instead of the 256x256 general one.",
    )
    args = parser.parse_args()

    export(args.out, landscape=args.landscape)


if __name__ == "__main__":
    main()
