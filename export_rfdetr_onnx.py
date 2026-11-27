"""
export_rfdetr_onnx.py -- Export Roboflow RF-DETR (Nano, Small, Medium) COCO detectors to ONNX.

Usage:
    python export_rfdetr_onnx.py [--out DIR] [--variants nano,small,medium]

Requirements:
    pip install --index-url https://download.pytorch.org/whl/cpu torch
    pip install "rfdetr[onnx]==1.12.1" onnx onnxruntime

    rfdetr downloads each variant's COCO checkpoint from Roboflow on first use
    (repo.roboflow.com/rfdetr/<variant>_coco/checkpoint_best_regular.pth).

ONNX tensor interface (static shape, batch 1, opset 17):
    Input  "input"   [1, 3, S, S]          float32  RGB, stretched (not letterboxed) to SxS,
                                                    /255 then ImageNet mean/std normalised
    Output "dets"    [1, 300, 4]           float32  boxes, normalised cx, cy, w, h
    Output "labels"  [1, 300, 91]          float32  per-class logits (sigmoid, not softmax);
                                                    slot index == COCO category id (1..90),
                                                    slot 0 and the 10 unused ids never fire
    S = 384 (nano), 512 (small), 576 (medium). No NMS needed.

Output files:
    <out>/rfdetr-nano.onnx, rfdetr-small.onnx, rfdetr-medium.onnx

License:
    rfdetr code and the Nano/Small/Medium/Large detection weights are Apache-2.0
    (Copyright 2025 Roboflow, Inc.). See https://github.com/roboflow/rf-detr#license.
    The XL/2XL/Atto/Femto/Pico detectors are under Roboflow's PML 1.0 and are NOT exported here.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

VARIANTS = {
    "nano": ("RFDETRNano", 384),
    "small": ("RFDETRSmall", 512),
    "medium": ("RFDETRMedium", 576),
}


def export_one(name: str, out_dir: Path) -> Path:
    import rfdetr

    cls_name, resolution = VARIANTS[name]
    model = getattr(rfdetr, cls_name)()
    work = out_dir / f"_work_{name}"
    work.mkdir(parents=True, exist_ok=True)
    produced = Path(
        model.export(
            output_dir=str(work), opset_version=17, shape=(resolution, resolution), verbose=False
        )
    )
    if produced.is_dir():
        produced = produced / "inference_model.onnx"
    final = out_dir / f"rfdetr-{name}.onnx"
    shutil.move(str(produced), final)
    shutil.rmtree(work, ignore_errors=True)
    return final


def verify(path: Path) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ins = [(i.name, i.shape) for i in sess.get_inputs()]
    outs = [(o.name, o.shape) for o in sess.get_outputs()]
    print(f"  inputs  {ins}")
    print(f"  outputs {outs}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ThirdParty/models/rfdetr")
    ap.add_argument("--variants", default="nano,small,medium")
    args = ap.parse_args()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.variants.split(","):
        path = export_one(name.strip(), out_dir)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{path.name}: {path.stat().st_size} bytes, sha256 {digest}")
        verify(path)


if __name__ == "__main__":
    main()
