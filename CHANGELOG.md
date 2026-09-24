# Changelog

All notable changes to this release package are documented here.

---

## [Unreleased]

### Added

- `cugan/up2x-latest-conservative.onnx` -- Real-CUGAN UpCunet2x 2x, light in-model
  denoising (noise level 1 -- conservative). Derived from `up2x-latest-conservative.pth`
  (bilibili/ailab updated_weights release). Same architecture as no-denoise; denoising is
  fully encoded in the trained weights.
- `cugan/up2x-latest-denoise2x.onnx` -- Real-CUGAN UpCunet2x 2x, moderate in-model
  denoising (noise level 2). Derived from `up2x-latest-denoise2x.pth`.
- `cugan/up2x-latest-denoise3x.onnx` -- Real-CUGAN UpCunet2x 2x, heavy in-model
  denoising (noise level 3). Derived from `up2x-latest-denoise3x.pth`.
- `hat/hat-real-sr-x4.onnx` -- HAT 4x real-world SR (balanced quality). Derived from
  `Real_HAT_GAN_SRx4.pth` (XPixelGroup/HAT, Apache 2.0). Tile size 64x64 (window_size=16
  constraint). Attention mask and relative position index buffers are baked in at export
  time; forward_features patched to eliminate dynamic control flow.
- `hat/hat-real-sr-x4-sharper.onnx` -- HAT 4x real-world SR (higher perceptual sharpness).
  Derived from `Real_HAT_GAN_sharper.pth` (XPixelGroup/HAT, Apache 2.0).
- `hat/hat-sr-x2.onnx` -- HAT 2x classical SR. Derived from `HAT_SRx2.pth`
  (XPixelGroup/HAT, Apache 2.0).
- `swinir/swinir-real-sr-x2.onnx` -- SwinIR-M 2x real-world SR. Derived from
  `003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth` (JingyunLiang/SwinIR v0.0,
  Apache 2.0). Tile size 128x128.
- `swinir/swinir-real-sr-x4.onnx` -- SwinIR-L 4x real-world SR (larger model, deeper
  depths/wider embedding). Derived from
  `003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth` (JingyunLiang/SwinIR v0.0,
  Apache 2.0). Tile size 128x128.
- `waifu2x/waifu2x-cunet-scale2-noise0.onnx` -- Waifu2x CUNet 2x, no denoising.
  Derived from `pretrained_models/cunet/art/noise0_scale2x.pth`
  (nagadomi/nunif releases, MIT). Tile size 128x128.
- `waifu2x/waifu2x-cunet-scale2-noise1.onnx` -- Waifu2x CUNet 2x, light denoising.
  Derived from `noise1_scale2x.pth`.
- `waifu2x/waifu2x-cunet-scale2-noise2.onnx` -- Waifu2x CUNet 2x, moderate denoising.
  Derived from `noise2_scale2x.pth`.
- `waifu2x/waifu2x-cunet-scale2-noise3.onnx` -- Waifu2x CUNet 2x, heavy denoising.
  Derived from `noise3_scale2x.pth`.
- `export_hat_onnx.py` -- Exports HAT real-world SR models. Downloads architecture from
  GitHub and weights from Google Drive via gdown. Patches forward_features to use
  precomputed attention buffers, eliminating dynamic control flow at ONNX trace time.
- `export_swinir_onnx.py` -- Exports SwinIR-M (x2) and SwinIR-L (x4) real-world SR
  models. Downloads architecture and weights from JingyunLiang/SwinIR GitHub releases.
- `export_waifu2x_onnx.py` -- Exports Waifu2x CUNet scale2 noise0-noise3 models.
  Downloads architecture from nagadomi/nunif and weights from the nunif pretrained
  models zip (437 MB, cached on first run).

### Changed

- All export scripts (`export_hat_onnx.py`, `export_swinir_onnx.py`,
  `export_waifu2x_onnx.py`, `export_cugan_onnx.py`, `export_esrgan_onnx.py`,
  `export_bsrgan_onnx.py`) now delete the `_*_export_tmp/` scratch directory
  (downloaded weights + arch source) automatically after a successful export. Pass
  `--keep-tmp` to retain the directory.

---

## [1.0.0] - 2026-05-12

### Initial public release

**Models included:**

- `realesrgan-x4plus.onnx` -- Real-ESRGAN RRDBNet 4x, general-purpose for live-action
  content. Derived from `RealESRGAN_x4plus.pth` (upstream release v0.1.0).
- `realesrgan-x2plus.onnx` -- Real-ESRGAN RRDBNet 2x, balanced general-purpose upscaler.
  Derived from `RealESRGAN_x2plus.pth` (upstream release v0.2.1).
- `realesr-animevideov3.onnx` -- Real-ESRGAN SRVGGNetCompact 4x, optimised for anime and
  cel animation. Derived from `realesr-animevideov3.pth` (upstream release v0.2.5.0).
  Includes dynamic-batch support; `Resize` nodes use relative scales for DirectML.
- `cugan/up2x-latest-no-denoise.onnx` -- Real-CUGAN UpCunet2x 2x, general-purpose no-denoise
  model. Derived from `up2x-latest-no-denoise.pth` (bilibili/ailab updated_weights release).
- `bsrgan/bsrgan-x4.onnx` -- BSRGAN RRDBNet 4x, blind-degradation upscaler for real-world
  degraded content. Derived from `BSRGAN.pth` (cszn/KAIR v1.0).
- `bsrgan/bsrgan-x2.onnx` -- BSRGAN RRDBNet 2x, blind-degradation upscaler.
  Derived from `BSRGANx2.pth` (cszn/KAIR v1.0).
- `facedetect/face_detection_yunet_2023mar.onnx` -- YuNet face detector, multi-scale anchors.
  Redistributed unmodified from opencv/opencv_zoo (MIT License).
- `facerestore/GFPGANv1.4.onnx` -- GFPGAN v1.4 face restoration, fixed-shape
  `[1, 3, 512, 512]`. Derived from `GFPGANv1.4.pth` (TencentARC/GFPGAN v1.3.4).
- `facerestore/CodeFormer.onnx` -- CodeFormer face restoration at maximum fidelity
  (w=1.0), fixed-shape `[1, 3, 512, 512]`. **Non-commercial use only** (S-Lab License
  1.0). Derived from `codeformer.pth` (sczhou/CodeFormer v0.1.0).
- `rife/rife4.18_864x480.onnx` -- RIFE 4.18 frame interpolator, 864x480 (covers 480p landscape).
- `rife/rife4.18_1280x736.onnx` -- RIFE 4.18 frame interpolator, 1280x736 (covers 720p landscape).
- `rife/rife4.18_1920x1088.onnx` -- RIFE 4.18 frame interpolator, 1920x1088 (covers 1080p landscape).
- `rife/rife4.18_2560x1440.onnx` -- RIFE 4.18 frame interpolator, 2560x1440 (covers 1440p landscape).
- `rife/rife4.18_3840x2176.onnx` -- RIFE 4.18 frame interpolator, 3840x2176 (covers 4K landscape).
- `rife/rife4.18_480x864.onnx` -- RIFE 4.18 frame interpolator, 480x864 (covers 480p portrait).
- `rife/rife4.18_736x1280.onnx` -- RIFE 4.18 frame interpolator, 736x1280 (covers 720p portrait).
- `rife/rife4.18_1088x1920.onnx` -- RIFE 4.18 frame interpolator, 1088x1920 (covers 1080p portrait).
- `rife/rife4.18_1440x2560.onnx` -- RIFE 4.18 frame interpolator, 1440x2560 (covers 1440p portrait).
- `rife/rife4.18_2176x3840.onnx` -- RIFE 4.18 frame interpolator, 2176x3840 (covers 4K portrait).
  RIFE models are derived from `flownet_v4.18.pkl` (HolyWu/vs-rife model release).

**Export scripts included:**

- `export_esrgan_onnx.py` -- Exports all three Real-ESRGAN models with dynamic-batch
  patching and optional onnxsim simplification. Incorporates the Resize-node patching
  logic from `patch_esrgan_resize.py`.
- `export_bsrgan_onnx.py` -- Exports BSRGAN x4 and x2 models using the same 128x128 tiled
  ONNX format and DirectML Resize patch as the ESRGAN export.
- `export_cugan_onnx.py` -- Exports the Real-CUGAN up2x-no-denoise model.
- `export_rife_onnx.py` -- Exports RIFE 4.18 for 10 resolution tiers with timestep input
  support, enabling true N-times frame interpolation.
- `export_facerestore_onnx.py` -- Exports GFPGANv1.4 and CodeFormer face restoration
  models. Stubs out basicsr so neither basicsr nor facexlib need to be installed.
- `patch_esrgan_resize.py` -- Standalone predecessor script for dynamic-batch patching of
  the AnimeVideo V3 model. Retained for reference; logic is integrated into
  `export_esrgan_onnx.py`.

**ONNX conversion details:**

- All models exported with ONNX opset 17 via `torch.onnx.export`.
- Input/output batch dimension is symbolic (`N`), enabling multi-tile batched inference.
- Intermediate shape annotations stripped from the graph so DirectML recomputes buffer
  sizes at runtime rather than using stale batch=1 shapes.
- `Resize` nodes that use static absolute-size tensors are rewritten to use relative
  scale factors to prevent DirectML buffer allocation errors when N > 1.
- All models validated with a CPU `onnxruntime` sanity check at 128x128 input.
