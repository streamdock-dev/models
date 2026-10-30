"""
export_audiosr_onnx.py -- AudioSR (bandwidth extension) to ONNX: not available yet.

There is no working export for AudioSR, and this script only says so. The previous version
never ran: it called a method the audiosr package doesn't have, under torch.inference_mode(),
so there was nothing to trace.

Why it isn't a quick fix: AudioSR is a latent diffusion pipeline, not one network -- resample
and mel features, a VAE encoder for the condition, 25 DDIM steps of a 258 M-parameter UNet run
twice each for classifier-free guidance, a VAE decoder, a vocoder, and SciPy/librosa
post-processing that splices the input's low band back in. Unrolled into one graph it would be
~2.4-2.8 GB (over ONNX's 2 GB single-file limit) and, on CPU, ~13x slower than real time
(66 s per 5.12 s chunk measured on 8 threads).

The plan is three ONNX files -- condition encoder, UNet, decoder plus vocoder -- with the
diffusion loop run by the caller, once StreamDock's Model Manager can download a model made of
several files.

License (for when it's built):
    AudioSR's code is MIT: https://github.com/haoheliu/versatile_audio_super_resolution
    The audiosr_basic weights are Apache-2.0: https://huggingface.co/haoheliu/audiosr_basic
"""

import sys

if __name__ == "__main__":
    sys.exit(__doc__)
