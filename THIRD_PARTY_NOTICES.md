# Third-Party Notices

This release contains ONNX model files derived from upstream PyTorch checkpoints and
architecture source code from the following open-source projects. The original copyright
notices and license texts are reproduced below as required by each license.

---

## Real-ESRGAN

Applies to: `realesrgan-x4plus.onnx`, `realesrgan-x2plus.onnx`,
`realesr-animevideov3.onnx`, `realesr-animevideov3-batched.onnx`

- **Project**: Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data
- **Authors**: Xintao Wang, Liangbin Xie, Chao Dong, Ying Shan
- **Repository**: https://github.com/xinntao/Real-ESRGAN
- **License**: BSD 3-Clause License

The ONNX files are derived from the following upstream checkpoints:

| ONNX file | Upstream checkpoint | Release |
|-----------|---------------------|---------|
| `realesrgan-x4plus.onnx` | `RealESRGAN_x4plus.pth` | [v0.1.0](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.1.0) |
| `realesrgan-x2plus.onnx` | `RealESRGAN_x2plus.pth` | [v0.2.1](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.1) |
| `realesr-animevideov3.onnx` | `realesr-animevideov3.pth` | [v0.2.5.0](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0) |
| `realesr-animevideov3-batched.onnx` | Derived from `realesr-animevideov3.onnx` above | -- |

```
BSD 3-Clause License

Copyright (c) 2021, Xintao Wang
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

## BasicSR

Applies to: `realesrgan-x4plus.onnx`, `realesrgan-x2plus.onnx`

The RRDBNet architecture used by the x4plus and x2plus models originates from the
BasicSR toolkit. The architecture source (`rrdbnet_arch.py`) is fetched at export time
from the BasicSR repository and is not redistributed in this release. It is credited here
because the exported ONNX graph encodes the RRDBNet computation graph.

- **Project**: BasicSR -- Open Source Image and Video Restoration Toolbox
- **Repository**: https://github.com/XPixelGroup/BasicSR
- **License**: Apache License 2.0

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship made available under
      the License, as indicated by a copyright notice that is included in
      or attached to the work (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean, as submitted to the Licensor for inclusion
      in the Work by the copyright owner or by an individual or Legal Entity
      authorized to submit on behalf of the copyright owner. For the purposes
      of this definition, "submitted" means any form of electronic, verbal,
      or written communication sent to the Licensor or its representatives,
      including but not limited to communication on electronic mailing lists,
      source code control systems, and issue tracking systems that are managed
      by, or on behalf of, the Licensor for the purpose of submitting and
      discussing improvements to the Work, but excluding communication that
      is conspicuously marked or otherwise designated in writing by the
      copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any Legal Entity on behalf of
      whom a Contribution has been received by the Licensor and included
      within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by the combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a cross-claim
      or counterclaim in a lawsuit) alleging that the Work or any
      Contributor's Work constitutes direct or contributory patent
      infringement, then any patent licenses granted to You under this
      License for that Work shall terminate as of the date such litigation
      is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, You must include a readable copy of the
          attribution notices contained within such NOTICE file, in
          at least one of the following places: within a NOTICE text
          file distributed as part of the Derivative Works; within
          the Source form or documentation, if provided along with the
          Derivative Works; or, within a display generated by the
          Derivative Works, if and wherever such third-party notices
          normally appear. The contents of the NOTICE file are for
          informational purposes only and do not modify the License.
          You may add Your own attribution notices within Derivative
          Works that You distribute, alongside or as an addendum to
          the NOTICE text from the Work, provided that such additional
          attribution notices cannot be construed as modifying the License.

      You may add Your own license statement for Your modifications and
      may provide additional grant of rights to use, copy, modify, and
      distribute those modifications as governed by the terms of this
      License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or reproducing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or exemplary damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (even if such Contributor has been advised of the possibility
      of such damages).

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may offer only
      conditions that are consistent with this License.

   END OF TERMS AND CONDITIONS

   Copyright 2018-2022 BasicSR Authors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

---

## Real-CUGAN

Applies to: `up2x-latest-no-denoise.onnx`, `up2x-latest-conservative.onnx`,
`up2x-latest-denoise2x.onnx`, `up2x-latest-denoise3x.onnx`

All four are exported by `export_cugan_onnx.py` from bilibili/ailab's
`updated_weights.zip` release. The separately-listed `waifu2x-cunet-*` files are a
different project entirely (see the Waifu2x section below), despite Real-CUGAN's own
architecture also being called UpCunet2x.

- **Project**: Real-CUGAN (Real Cascade U-Nets for Anime Image Super Resolution)
- **Authors**: bilibili / ailab
- **Repository**: https://github.com/bilibili/ailab/tree/main/Real-CUGAN
- **Upstream checkpoint**: `up2x-latest-no-denoise.pth` from the
  [Real-CUGAN updated_weights release](https://github.com/bilibili/ailab/releases/tag/Real-CUGAN)
- **License**: MIT License

```
MIT License

Copyright (c) 2021 bilibili

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## RIFE 4.18 (vs-rife)

Applies to: `rife/rife4.18_*.onnx`

- **Project**: vs-rife -- VapourSynth plugin for RIFE
- **Author**: HolyWu
- **Repository**: https://github.com/HolyWu/vs-rife
- **Upstream checkpoint**: `flownet_v4.18.pkl` from the
  [vs-rife model release](https://github.com/HolyWu/vs-rife/releases/tag/model)
- **License**: MIT License

```
MIT License

Copyright (c) 2021 HolyWu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Waifu2x (nunif)

Applies to: `waifu2x-cunet-scale2-noise0.onnx`, `waifu2x-cunet-scale2-noise1.onnx`,
`waifu2x-cunet-scale2-noise2.onnx`, `waifu2x-cunet-scale2-noise3.onnx`

- **Project**: nunif / waifu2x (CUNet art models)
- **Author**: nagadomi
- **Repository**: https://github.com/nagadomi/nunif
- **Upstream checkpoints**: the `cunet/art` scale2x noise0-noise3 weights from
  `waifu2x_pretrained_models_20250502.zip` in the
  [nunif 0.0.0 release](https://github.com/nagadomi/nunif/releases/tag/0.0.0)
- **Architecture source**: the CUNet architecture, fetched at export time from the nunif
  repository and not redistributed here
- **License**: MIT License

```
The MIT License

Copyright (C) 2019-2023 nagadomi <https://github.com/nagadomi/>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## RIFE 4.7 (Practical-RIFE)

Applies to: `rife47.onnx`

- **Project**: Practical-RIFE / RIFE: Real-Time Intermediate Flow Estimation for Video
  Frame Interpolation
- **Authors**: Zhewei Huang, Tianyuan Zhang, Wen Heng, Boxin Shi, Shuchang Zhou
- **Repository**: https://github.com/hzwer/Practical-RIFE
- **Upstream weights**: RIFE 4.7 flownet weights, released by the authors under the same
  MIT licence as the project itself
- **ONNX conversion**: `rife47_ensemble_True_scale_1_sim.onnx` from the community
  redistribution at https://huggingface.co/yuvraj108c/rife-onnx
- **License**: MIT License (per the upstream RIFE project)

> Unlike every other model published here, this file is a third-party ONNX conversion
> rather than one produced by this project's own `export_*_onnx.py` scripts. The
> conversion repository declares no licence of its own; the MIT grant above comes from
> the upstream RIFE weights it was converted from, which is what governs redistribution.
> It is kept because it is the only RIFE export with dynamic height/width axes, making it
> the resolution-independent fallback when no fixed-resolution rife4.18 tier matches a
> stream. Replacing it with a self-built equivalent requires adding dynamic-axes support
> to `export_rife_onnx.py`, which currently exports fixed shapes only.

```
MIT License

Copyright (c) 2021 hzwer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## GFPGAN

Applies to: `facerestore/GFPGANv1.4.onnx`

- **Project**: GFPGAN: Towards Real-World Blind Face Restoration with Generative Facial Prior
- **Authors**: Xintao Wang, Yu Li, Honglun Zhang, Ying Shan
- **Repository**: https://github.com/TencentARC/GFPGAN
- **Upstream checkpoint**: `GFPGANv1.4.pth` from the
  [GFPGAN v1.3.4 release](https://github.com/TencentARC/GFPGAN/releases/tag/v1.3.4)
- **License**: Apache License 2.0

```
Copyright (c) 2021, Tencent

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## CodeFormer

Applies to: `facerestore/CodeFormer.onnx`

> **Non-commercial use only.** CodeFormer is released under the S-Lab License 1.0,
> which restricts use to non-commercial research purposes. Do not include
> `CodeFormer.onnx` in commercial products or services without explicit permission
> from the authors (sczhou/CodeFormer).

- **Project**: CodeFormer: Towards Robust Blind Face Restoration with Code Dictionary and Contrastive Regularization
- **Authors**: Shangchen Zhou, Kelvin Chan, Chongyi Li, Chen Change Loy
- **Repository**: https://github.com/sczhou/CodeFormer
- **Upstream checkpoint**: `codeformer.pth` from the
  [CodeFormer v0.1.0 release](https://github.com/sczhou/CodeFormer/releases/tag/v0.1.0)
- **License**: S-Lab License 1.0 (non-commercial research use only)

```
S-Lab License 1.0

Copyright 2022 S-Lab

Redistribution and use for non-commercial purposes in source and
binary forms, with or without modification, are permitted provided
that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in
   the documentation and/or other materials provided with the
   distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

In the event that redistribution and/or use for commercial purposes
are required, please contact the author for authorization.
```

---

## YuNet Face Detection Model

Applies to: `facedetect/face_detection_yunet_2023mar.onnx`

- **Project**: YuNet (face_detection_yunet_2023mar.onnx)
- **Copyright**: Copyright (c) 2021 ShiqiYu
- **License**: MIT License
- **Source (model weights)**: https://github.com/ShiqiYu/libfacedetection.train
- **Source (ONNX distribution)**: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- **Paper**: Wu et al., "YuNet: A Tiny Millisecond-Level Face Detector", Machine Intelligence Research, 2023

The file `facedetect/face_detection_yunet_2023mar.onnx` is redistributed unmodified
from the OpenCV model zoo under the MIT License below.

```
MIT License

Copyright (c) 2021 ShiqiYu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## BSRGAN

Applies to: `bsrgan-x4.onnx`, `bsrgan-x2.onnx`

- **Project**: BSRGAN: Designing a Practical Degradation Model for Deep Blind Image Super-Resolution
- **Authors**: Kai Zhang, Jingyun Liang, Luc Van Gool, Radu Timofte
- **Repository**: https://github.com/cszn/BSRGAN
- **Upstream checkpoints**: `BSRGAN.pth` and `BSRGANx2.pth` from the
  [BSRGAN v1.0 release](https://github.com/cszn/BSRGAN/releases/tag/v1.0)
- **Architecture source**: `models/network_rrdbnet.py`, fetched at export time from the
  BSRGAN repository and not redistributed here
- **License**: Apache License 2.0

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## SwinIR

Applies to: `swinir-real-sr-x4.onnx`, `swinir-real-sr-x2.onnx`

- **Project**: SwinIR: Image Restoration Using Swin Transformer
- **Authors**: Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, Radu Timofte
- **Repository**: https://github.com/JingyunLiang/SwinIR
- **Upstream checkpoints**: real-world SR weights from the
  [SwinIR v0.0 release](https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0)
- **License**: Apache License 2.0
- **Note**: SwinIR credits Swin Transformer and KAIR upstream; those projects carry
  their own licenses.

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## HAT

Applies to: `hat-real-sr-x4.onnx`, `hat-real-sr-x4-sharper.onnx`, `hat-sr-x2.onnx`

- **Project**: HAT: Activating More Pixels in Image Super-Resolution Transformer
- **Authors**: Xiangyu Chen, Xintao Wang, Jiantao Zhou, Yu Qiao, Chao Dong
- **Repository**: https://github.com/XPixelGroup/HAT
- **Upstream checkpoints**: `Real_HAT_GAN_SRx4.pth`, `Real_HAT_GAN_sharper.pth` and
  `HAT_SRx2.pth`, downloaded at export time from the authors' Google Drive release
- **License**: Apache License 2.0

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## BasicVSR++

Applies to: `basicvsrpp_x4_t5.onnx`, `basicvsrpp_x4_t10.onnx`

- **Project**: BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment
- **Authors**: Kelvin C.K. Chan, Shangchen Zhou, Xiangyu Xu, Chen Change Loy
- **Repository**: https://github.com/ckkelvinchan/BasicVSR_PlusPlus
- **Upstream checkpoint**: `basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth`
  from the OpenMMLab model zoo (download.openmmlab.com)
- **License**: Apache License 2.0

```
Copyright (c) MMEditing Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## RealBasicVSR

Applies to: `realbasicvsr_x4_t5.onnx`, `realbasicvsr_x4_t10.onnx`

- **Project**: RealBasicVSR: Investigating Tradeoffs in Real-World Video Super-Resolution
- **Authors**: Kelvin C.K. Chan, Shangchen Zhou, Xiangyu Xu, Chen Change Loy
- **Repository**: https://github.com/ckkelvinchan/RealBasicVSR
- **Upstream checkpoint**: `RealBasicVSR.pth`, published by the authors and downloaded
  at export time
- **License**: Apache License 2.0

```
Copyright (c) 2021 Kelvin C.K. Chan

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## AnimeSR

Applies to: `animevsr_x4_t5.onnx`, `animevsr_x4_t10.onnx`

- **Project**: AnimeSR: Learning Real-World Super-Resolution Models for Animation Videos
- **Authors**: Yanze Wu, Xintao Wang, Gen Li, Ying Shan
- **Repository**: https://github.com/TencentARC/AnimeSR
- **Upstream checkpoint**: AnimeSR v1 weights, downloaded at export time from the authors' Google Drive release
- **License**: Apache License 2.0
- **Note**: Exported files use the `animevsr_` prefix and the `RealBasicVsr` pipeline
  because the C# `RealBasicVsrUpscaler` handles both architectures identically
  (same I/O shape).

```
Copyright (C) 2022 THL A29 Limited, a Tencent company.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## Export Scripts

The export scripts (`export_esrgan_onnx.py`, `export_cugan_onnx.py`,
`export_rife_onnx.py`, `export_facerestore_onnx.py`, `patch_esrgan_resize.py`) are
original work by the StreamDock authors and are released under the MIT License.

```
MIT License

Copyright (c) 2026 StreamDock Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


## RVRT

Applies to: `vrt/rvrt_x4_t*.onnx` (user-built via export_rvrt_onnx.py)

> **Non-commercial use only.** RVRT is released under the Creative Commons
> Attribution-NonCommercial 4.0 International License (CC-BY-NC 4.0), which restricts use
> to non-commercial research purposes. Do not use the weights or any derived ONNX files in
> commercial products or services without explicit permission from the authors
> (JingyunLiang/RVRT).

- **Project**: Recurrent Video Restoration Transformer with Guided Deformable Attention (RVRT)
- **Paper**: https://arxiv.org/abs/2206.02146
- **Repository**: https://github.com/JingyunLiang/RVRT
- **Upstream checkpoint**: `01_RVRT_videosr_bi_REDS_30frames.pth` from the
  [RVRT v0.0 release](https://github.com/JingyunLiang/RVRT/releases/tag/v0.0)
- **License**: Creative Commons Attribution-NonCommercial 4.0 International (CC-BY-NC 4.0)
- **License URL**: https://creativecommons.org/licenses/by-nc/4.0/


## EDVR

Applies to: `edvr/edvr_m_x4.onnx`, `edvr/edvr_l_x4.onnx` (user-built via export_edvr_onnx.py)

> **Non-commercial use only.** EDVR is released under the Creative Commons
> Attribution-NonCommercial-ShareAlike 4.0 International License (CC-BY-NC-SA 4.0), which
> restricts use to non-commercial research purposes. Do not use the weights or any derived
> ONNX files in commercial products or services without explicit permission from the authors
> (xinntao/EDVR).

- **Project**: EDVR: Video Restoration with Enhanced Deformable Convolutional Networks
- **Paper**: https://arxiv.org/abs/1905.02716
- **Repository**: https://github.com/xinntao/EDVR
- **Upstream checkpoints**: EDVR_M_x4_SR_REDS.pth, EDVR_L_x4_SR_REDS.pth downloaded at
  export time via gdown from the EDVR Google Drive release
- **License**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC-BY-NC-SA 4.0)
- **License URL**: https://creativecommons.org/licenses/by-nc-sa/4.0/
