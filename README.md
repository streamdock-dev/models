# StreamDock Models

Open-source ONNX models used by [StreamDock](https://streamdock.net), re-exported from their
upstream PyTorch checkpoints into the fixed input shapes the app's inference pipelines need.

**The model files live in [Releases](../../releases), not in this repository.** Git is a poor
fit for multi-hundred-megabyte binaries; GitHub release assets are not, and they can be
downloaded without authentication.

## What's here

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
publish ONNX rarely publish the fixed input shapes a real-time pipeline needs. The export
scripts (`export_*_onnx.py`, in the StreamDock repository under
`ThirdParty/av-restoration-models/`) download the upstream checkpoint, rebuild the
architecture, trace it at a specific shape, and simplify the resulting graph.

## Licensing

Every model published here is redistributable under its upstream licence — BSD-3-Clause,
Apache-2.0 or MIT. Full per-model attribution, upstream checkpoint, and licence text are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Check it before redistributing anything from
this repository.

Some models StreamDock supports are **deliberately absent** because their licences forbid
redistribution — CodeFormer (S-Lab 1.0), RVRT (CC-BY-NC 4.0) and EDVR (CC-BY-NC-SA 4.0) are all
non-commercial. Those are built locally by the user from upstream weights instead, and will
never be published here.

The export scripts themselves are MIT, © StreamDock Authors.
