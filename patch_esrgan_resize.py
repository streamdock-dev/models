"""
Patches an ONNX model to support dynamic batch inference.

Makes the batch dimension dynamic on I/O tensors, then strips all inferred
intermediate shapes from value_info so that DirectML recomputes correct
buffer sizes at runtime rather than using stale batch=1 shapes.
"""

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper, shape_inference

_HERE = Path(__file__).resolve().parent
MODEL_IN = str(_HERE / "esrgan" / "realesr-animevideov3.onnx")
MODEL_OUT = str(_HERE / "esrgan" / "realesr-animevideov3-batched.onnx")


def patch(model_in: str, model_out: str) -> None:
    """Patch an ONNX model for dynamic-batch inference.

    Loads the model at model_in, makes the batch dimension dynamic on all
    I/O tensors, strips all inferred intermediate value_info shapes so
    DirectML recomputes them at runtime, validates the result, saves it to
    model_out, and runs a CPU batch-N=2 inference sanity check.
    """
    model: onnx.ModelProto = onnx.load(model_in)

    # Make batch dimension dynamic on I/O
    model.graph.input[0].type.tensor_type.shape.dim[0].ClearField("dim_value")
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "N"
    model.graph.output[0].type.tensor_type.shape.dim[0].ClearField("dim_value")
    model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = "N"

    # Strip all inferred intermediate tensor shapes. Without this, DirectML
    # uses the stale batch=1 shapes as buffer sizes even when N > 1, producing
    # corrupted output (tiling ghosting). Clearing value_info forces DML to
    # re-infer correct shapes from the actual runtime input dimensions.
    del model.graph.value_info[:]
    print("Stripped value_info (intermediate shapes cleared for dynamic batch).")

    # Run shape inference temporarily just to read initializer shapes for Resize patching
    inferred: onnx.ModelProto = shape_inference.infer_shapes(model)

    shape_map: dict[str, list[int | None]] = {}
    for vi in (
        list(inferred.graph.value_info)
        + list(inferred.graph.input)
        + list(inferred.graph.output)
    ):
        try:
            dims: list[int | None] = []
            for d in vi.type.tensor_type.shape.dim:
                if d.HasField("dim_value") and d.dim_value > 0:
                    dims.append(d.dim_value)
                else:
                    dims.append(None)  # dynamic or unknown
            shape_map[vi.name] = dims
        except Exception:
            pass

    init_arrays: dict[str, np.ndarray] = {
        init.name: numpy_helper.to_array(init) for init in inferred.graph.initializer
    }

    new_inits: list[onnx.TensorProto] = []
    patched: int = 0

    # Iterate nodes on model (what we save), use inferred shapes for scale computation.
    for node in model.graph.node:
        if node.op_type != "Resize":
            continue
        if len(node.input) < 4 or not node.input[3]:
            continue

        sizes_name: str = node.input[3]
        if sizes_name not in init_arrays:
            print(
                f"  SKIP {node.name}: sizes '{sizes_name}' not a constant initializer"
            )
            continue

        input_name: str = node.input[0]
        if input_name not in shape_map:
            print(f"  SKIP {node.name}: no inferred shape for input '{input_name}'")
            continue

        sizes: np.ndarray = init_arrays[sizes_name].flatten().astype(np.float64)
        in_shape: list[int | None] = shape_map[input_name]

        if len(in_shape) != len(sizes):
            print(
                f"  SKIP {node.name}: shape rank {len(in_shape)} != sizes rank {len(sizes)}"
            )
            continue

        scales: list[float] = []
        in_dim: int | None
        out_dim: float
        for in_dim, out_dim in zip(in_shape, sizes):
            if in_dim is None or in_dim == 0:
                scales.append(1.0)  # dynamic (batch) dim -> identity scale
            else:
                scales.append(float(out_dim) / float(in_dim))

        print(
            f"  PATCH {node.name}: in={in_shape} sizes={sizes.tolist()} -> scales={scales}"
        )

        scales_name: str = (
            f"__batched_scales_{node.name.replace('/', '_').strip('_')}__"
        )
        new_inits.append(
            numpy_helper.from_array(
                np.array(scales, dtype=np.float32), name=scales_name
            )
        )

        # Extend input list so indices 0-3 exist
        while len(node.input) < 4:
            node.input.append("")

        node.input[2] = scales_name  # set scales
        node.input[3] = ""  # clear sizes
        patched += 1

    if patched == 0:
        print("No Resize nodes needed patching.")
    else:
        print(f"\nPatched {patched} Resize node(s).")

    # Apply any new initializers to model (stripped), not inferred (has stale shapes).
    model.graph.initializer.extend(new_inits)

    try:
        onnx.checker.check_model(model)
        print("ONNX model check: OK")
    except Exception as e:
        print(f"ONNX model check warning (non-fatal): {e}")

    onnx.save(model, model_out)
    print(f"Saved -> {model_out}")

    # Verify with CPU inference at batch=2
    try:
        sess: ort.InferenceSession = ort.InferenceSession(
            model_out, providers=["CPUExecutionProvider"]
        )
        inp_name: str = sess.get_inputs()[0].name
        dummy: np.ndarray = np.zeros((2, 3, 128, 128), dtype=np.float32)
        out: list[np.ndarray] = sess.run(None, {inp_name: dummy})
        print(f"CPU batch test (N=2): output shape = {out[0].shape}  [PASS]")
    except ImportError:
        print("onnxruntime Python package not installed, skipping CPU batch test.")
    except Exception as e:
        print(f"CPU batch test FAILED: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    patch(MODEL_IN, MODEL_OUT)
