# Third-Party Notices

FormulaOCR depends on open-source OCR and formula-recognition components. This file records the main third-party projects used by the application and the notices users should keep when redistributing binaries or offline model packages.

## PaddleOCR

- Project: PaddleOCR
- Repository: https://github.com/PaddlePaddle/PaddleOCR
- Pinned Python package used for builds: `paddleocr==3.6.0`
- Organization: PaddlePaddle / PaddleOCR Authors
- License: Apache License 2.0
- Use in this project: formula recognition through PaddleOCR/PaddleX formula-recognition models.

PaddleOCR source code and model files are not vendored in this repository. If you redistribute a packaged build that includes PaddleOCR, PaddlePaddle, PaddleX, runtime libraries, or model weights, keep the original third-party license files, copyright notices, and any model-specific usage terms with the distributed package.

Suggested citation or acknowledgement:

```text
This software uses PaddleOCR, an open-source OCR toolkit from the PaddlePaddle ecosystem:
https://github.com/PaddlePaddle/PaddleOCR
```

## RapidLaTeXOCR and ONNX Runtime

- Project: RapidAI/RapidLaTeXOCR
- Repository: https://github.com/RapidAI/RapidLaTeXOCR
- License: MIT (the PyPI metadata currently labels it Apache-2.0; the distributed LICENSE file and repository state MIT)
- Use: optional community formula-recognition backend derived from LaTeX-OCR/pix2tex.
- Runtime: Microsoft ONNX Runtime, MIT License.

RapidLaTeXOCR model files are downloaded on demand from the project's official GitHub Release and verified with SHA-256. They are not committed to this repository or included in the default application archive.

## UniMERNet

UniMERNet was developed by Shanghai AI Laboratory and is exposed here through the official PaddleOCR/PaddleX inference model catalog. Users redistributing model weights should retain the upstream model card, license, and attribution applicable to the downloaded release.

## UniMERNet Small ONNX

- Original model: `wanderkid/unimernet_small`
- ONNX conversion repository/model card: `Cooper114/unimernet-onnx`
- Model card: https://huggingface.co/Cooper114/unimernet-onnx
- License: Apache License 2.0 (model card and upstream source attribution)
- Use: optional quantized UniMERNet Small backend using ONNX Runtime.

The `UniMERNetSmallONNX` files are downloaded from the pinned Hugging Face
revision `411ee76221baaad144ffbf996d4deef8df013b54` and verified against the
six-file SHA-256 manifest before use. The model is not committed to this
repository or included in the default application archive. The ONNX export
uses a first decoder plus a `decoder_with_past` KV-cache decoder; retain the
upstream model-card attribution and license when redistributing an offline
copy.

## MathCraft Formula ONNX

- Project: SakuraMathcraft/MathCraft-Models
- Repository: https://github.com/SakuraMathcraft/MathCraft-Models
- Model release: https://github.com/SakuraMathcraft/MathCraft-Models/releases/tag/v1.0.0
- License: MIT (https://github.com/SakuraMathcraft/MathCraft-Models/blob/main/LICENSE)
- Related implementation: SakuraMathcraft/LaTeXSnipper
- Use: optional pure ONNX Runtime formula-recognition backend.

The `MathCraftFormula` archive is downloaded from the official release on demand. The archive SHA-256 is `807dd2d1ac40454424404b31a73d4242c37c76edf176ab544540028da20ec43f`; extracted files are checked against the upstream SHA-256 manifest before atomic installation. The model is not committed to this repository or included in the default application archive. Redistributors should retain the MathCraft-Models license and model-card terms with any offline copy.

## PaddlePaddle LaTeX-OCR Rec

- Project/model: PaddlePaddle `LaTeX_OCR_rec`
- Model card: https://huggingface.co/PaddlePaddle/LaTeX_OCR_rec
- License: Apache License 2.0 (model card)
- Use: optional lightweight Paddle formula-recognition backend.

The `LaTeX_OCR_rec` files are downloaded from the pinned Hugging Face revision
`563fb029dfdf5fc847d0677f3870039960e3a801` and verified with SHA-256 before
installation. They are not committed to this repository or included in the
default application archive. Redistributors should retain the upstream model
card and license terms with any offline copy.

## Pix2Text MFR 1.5

- Project: Breezedeus/Pix2Text
- Repository: https://github.com/breezedeus/Pix2Text
- Model card: https://huggingface.co/breezedeus/pix2text-mfr-1.5
- License: MIT (model card and upstream Pix2Text repository)
- Use: optional Pix2Text mathematical formula recognition backend using ONNX Runtime.

The `Pix2TextMFR15` files are downloaded from the pinned Hugging Face revision `1cef9f0bdcd6a4c63df7de1311fb0894593340cc` on demand. All eight inference files are checked against the recorded SHA-256 manifest before use. They are not committed to this repository or included in the default application archive; redistributors should retain the upstream model-card terms.

## MixTeX

- Project: [RQLuo/MixTeX-Latex-OCR](https://github.com/RQLuo/MixTeX-Latex-OCR)
- Release: `MixTeX-v3.2.4`
- Release asset: `MixTeX.zip`
- Use: optional Chinese/English formula OCR backend using the official merged-decoder ONNX export.

The MixTeX release archive is downloaded on demand from the upstream release and
verified with SHA-256
`734088e8c3ac6d0ebf02b3054ed0cdde7d8be2eb57c33b8f049a66d05e026750`.
FormulaOCR extracts only the ONNX model payload and does not redistribute the
upstream executable. The upstream `User Manual&Terms of Service.md` states that
the software is AGPL-3.0 and that derivatives of the model may not be used for
commercial purposes. FormulaOCR displays this restriction and asks the user to
confirm it before first download/use. Do not include MixTeX weights in a
commercial binary or offline package without written permission from the
upstream rights holder.

The application also uses Python packages listed in `requirements.txt`, including Pillow, paddlepaddle, paddlex, ONNX Runtime, RapidLaTeXOCR, latex2mathml, requests, aiohttp, tokenizers, ftfy, and PyInstaller. Their licenses are controlled by their respective upstream projects.
