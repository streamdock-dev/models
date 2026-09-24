# StreamDock Models

Open-source ONNX models used by [StreamDock](https://streamdock.net), re-exported from their
upstream PyTorch checkpoints into the fixed input shapes the app's inference pipelines need,
together with the scripts that produce them.

**The model files live in [Releases](../../releases), not in this repository.** Git is a poor
fit for multi-hundred-megabyte binaries; GitHub release assets are not, and they can be
downloaded without authentication.

## What's here

| | |
|---|---|
| [Releases](../../releases) | the models themselves |
| `export_*_onnx.py` | one script per model family, building the ONNX from upstream weights |
| [BUILDING.md](BUILDING.md) | per-model detail: architectures, tiers, upstream checkpoints |
| [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | per-model attribution and licence text |

| Release | Contents |
|---|---|
| `models-1.0.0` | Upscaling, temporal upscaling, face restoration |
| `rife-4.18.0` | RIFE frame-interpolation tiers, one per resolution |

They are split so RIFE can be re-versioned without re-uploading everything else.

StreamDock downloads these on demand through its in-app Model Manager, matching each file
against a SHA-256 recorded in the app's own catalog. Nothing here is bundled into the
installer.

## Why re-export at all

Most upstream projects publish PyTorch checkpoints rather than ONNX, and the ones that do
publish ONNX rarely publish the fixed input shapes a real-time pipeline needs. Each export
script downloads the upstream checkpoint, rebuilds the architecture, traces it at a specific
shape, and simplifies the resulting graph.

```
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<2.6"
pip install -r requirements.txt
python export_esrgan_onnx.py
```

torch is installed separately from the CPU wheel index because the default PyPI package bundles
CUDA (~2 GB) and none of these exports need a GPU. See [BUILDING.md](BUILDING.md) for what each
script produces.

## Licensing

Every model published here is redistributable under its upstream licence -- BSD-3-Clause,
Apache-2.0 or MIT. Full per-model attribution, upstream checkpoint, and licence text are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Check it before redistributing anything from
this repository.

Some models StreamDock supports are **deliberately absent** because their licences forbid
redistribution -- CodeFormer (S-Lab 1.0), RVRT (CC-BY-NC 4.0) and EDVR (CC-BY-NC-SA 4.0) are all
non-commercial. Their export scripts are here, so you can build them yourself from upstream
weights, but the built models will never be published in this repository's releases.

The export scripts themselves are MIT, © StreamDock Authors.
