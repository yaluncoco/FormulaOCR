from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

try:
    from formula_ocr_app.formula_formats import clean_recognized_latex
    from formula_ocr_app.image_utils import load_rgb_image
    from formula_ocr_app.model_api import DownloadProgressCallback
    from formula_ocr_app.onnx_runtime import (
        create_inference_session,
        repeated_token_suffix_start,
    )
    from formula_ocr_app.rapid_model_downloader import ensure_rapid_model
except ModuleNotFoundError as exc:  # Allows `python formula_ocr_app/app.py`.
    if exc.name != "formula_ocr_app":
        raise
    from formula_formats import clean_recognized_latex
    from image_utils import load_rgb_image
    from model_api import DownloadProgressCallback
    from onnx_runtime import create_inference_session, repeated_token_suffix_start
    from rapid_model_downloader import ensure_rapid_model


class RapidLatexRuntimeError(RuntimeError):
    """Raised when the direct RapidLaTeXOCR runtime cannot be initialized."""


class RapidLatexRecognizer:
    """Pure Pillow/NumPy/ONNX Runtime implementation of RapidLaTeXOCR.

    RapidLaTeXOCR's Python package imports OpenCV for a few grayscale and
    bounding-box operations. Implementing those small operations locally
    removes a roughly 120 MiB OpenCV runtime from FormulaOCR builds while
    retaining the same official model files and preprocessing dimensions.
    """

    def __init__(
        self,
        *,
        device: str = "cpu",
        max_new_tokens: int = 512,
        download_progress_callback: DownloadProgressCallback | None = None,
        model_ensure: Callable[..., Path] = ensure_rapid_model,
    ) -> None:
        self.device = device.strip() if device else "cpu"
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.download_progress_callback = download_progress_callback
        self._model_ensure = model_ensure
        self._resizer_session: Any | None = None
        self._encoder_session: Any | None = None
        self._decoder_session: Any | None = None
        self._tokenizer: Any | None = None

    def predict(self, image_path: str | Path) -> str:
        self._ensure_model()
        assert self._resizer_session is not None
        assert self._encoder_session is not None
        assert self._decoder_session is not None
        assert self._tokenizer is not None

        image = _load_image(Path(image_path).resolve())
        pixel_values = _resize_for_model(image, self._resizer_session)
        encoder_input = self._encoder_session.get_inputs()[0]
        context = self._encoder_session.run(
            None,
            {encoder_input.name: pixel_values},
        )[0]
        tokens = _decode_tokens(
            self._decoder_session,
            context,
            bos_token=1,
            eos_token=2,
            max_new_tokens=self.max_new_tokens,
        )
        decoded = self._tokenizer.decode(tokens, skip_special_tokens=False)
        result = _post_process(
            "".join(decoded.split(" "))
            .replace("Ġ", " ")
            .replace("[EOS]", "")
            .replace("[BOS]", "")
            .replace("[PAD]", "")
            .strip()
        )
        if not result:
            raise RapidLatexRuntimeError("RapidLaTeXOCR 未返回公式。")
        return str(result)

    def close(self) -> None:
        self._resizer_session = None
        self._encoder_session = None
        self._decoder_session = None
        self._tokenizer = None

    def _ensure_model(self) -> None:
        if (
            self._resizer_session is not None
            and self._encoder_session is not None
            and self._decoder_session is not None
            and self._tokenizer is not None
        ):
            return
        model_dir = self._model_ensure(
            progress_callback=self.download_progress_callback
        )
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RapidLatexRuntimeError(
                "程序缺少 RapidLaTeXOCR 的 ONNX Runtime 或 tokenizers 运行组件。"
            ) from exc
        try:
            resizer_session = create_inference_session(
                ort,
                model_dir / "image_resizer.onnx",
                device=self.device,
                error_type=RapidLatexRuntimeError,
            )
            encoder_session = create_inference_session(
                ort,
                model_dir / "encoder.onnx",
                device=self.device,
                error_type=RapidLatexRuntimeError,
            )
            decoder_session = create_inference_session(
                ort,
                model_dir / "decoder.onnx",
                device=self.device,
                error_type=RapidLatexRuntimeError,
            )
            tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        except RapidLatexRuntimeError:
            raise
        except Exception as exc:  # pragma: no cover - native runtime dependent
            raise RapidLatexRuntimeError(
                "RapidLaTeXOCR 模型初始化失败，请检查模型文件和运行时。"
            ) from exc

        self._resizer_session = resizer_session
        self._encoder_session = encoder_session
        self._decoder_session = decoder_session
        self._tokenizer = tokenizer


def _load_image(image_path: Path) -> Any:
    try:
        return load_rgb_image(image_path)
    except (OSError, ValueError) as exc:
        raise RapidLatexRuntimeError(f"无法读取公式图片：{image_path}") from exc


def _resize_for_model(image: Any, resizer_session: Any) -> Any:
    input_image = _minmax_size(_pad_formula(image)).convert("RGB")
    ratio = 1.0
    width, height = input_image.size
    final_image = None
    for _ in range(10):
        height = max(1, int(height * ratio))
        final_image, padded = _preprocess(input_image, ratio, width, height)
        input_name = resizer_session.get_inputs()[0].name
        output = resizer_session.run(None, {input_name: final_image})[0]
        resized_width = (int(output.argmax(axis=-1).reshape(-1)[0]) + 1) * 32
        if resized_width == padded.width:
            break
        ratio = resized_width / max(padded.width, 1)
        width = resized_width
    assert final_image is not None
    return final_image


def _preprocess(image: Any, ratio: float, width: int, height: int) -> tuple[Any, Any]:
    import numpy as np
    from PIL import Image

    resampling = Image.Resampling.BILINEAR if ratio > 1 else Image.Resampling.LANCZOS
    resized = image.resize((max(1, width), max(1, height)), resampling)
    padded = _pad_formula(_minmax_size(resized))
    rgb = np.asarray(padded.convert("RGB"), dtype=np.float32)
    gray = (
        rgb[..., 0] * np.float32(0.299)
        + rgb[..., 1] * np.float32(0.587)
        + rgb[..., 2] * np.float32(0.114)
    )
    normalized = (
        gray - np.float32(0.7931 * 255.0)
    ) / np.float32(0.1738 * 255.0)
    tensor = np.ascontiguousarray(
        normalized[np.newaxis, np.newaxis, ...],
        dtype=np.float32,
    )
    return tensor, padded


def _pad_formula(image: Any, divisor: int = 32) -> Any:
    import numpy as np
    from PIL import Image

    data = np.asarray(image.convert("L"), dtype=np.uint8).astype(np.float32)
    minimum = float(data.min())
    maximum = float(data.max())
    if maximum > minimum:
        data = (data - minimum) / (maximum - minimum) * 255
    if float(data.mean()) > 128:
        foreground = data < 128
    else:
        foreground = data > 128
        data = 255 - data
    coordinates = np.argwhere(foreground)
    if coordinates.size:
        top, left = coordinates.min(axis=0)
        bottom, right = coordinates.max(axis=0)
        data = data[int(top) : int(bottom) + 1, int(left) : int(right) + 1]
    height, width = data.shape
    target_width = max(divisor, math.ceil(width / divisor) * divisor)
    target_height = max(divisor, math.ceil(height / divisor) * divisor)
    padded = Image.new("L", (target_width, target_height), 255)
    padded.paste(
        Image.fromarray(data.clip(0, 255).astype(np.uint8), mode="L"),
        (0, 0),
    )
    return padded


def _minmax_size(
    image: Any,
    *,
    max_dimensions: tuple[int, int] = (672, 192),
    min_dimensions: tuple[int, int] = (32, 32),
) -> Any:
    from PIL import Image

    ratios = [
        image.width / max_dimensions[0],
        image.height / max_dimensions[1],
    ]
    if any(ratio > 1 for ratio in ratios):
        scale = max(ratios)
        image = image.resize(
            (
                max(1, int(image.width // scale)),
                max(1, int(image.height // scale)),
            ),
            Image.Resampling.BILINEAR,
        )
    target = (
        max(image.width, min_dimensions[0]),
        max(image.height, min_dimensions[1]),
    )
    if target != image.size:
        canvas = Image.new("L", target, 255)
        canvas.paste(image, (0, 0))
        image = canvas
    return image


def _decode_tokens(
    decoder_session: Any,
    context: Any,
    *,
    bos_token: int,
    eos_token: int,
    max_new_tokens: int,
) -> list[int]:
    import numpy as np

    inputs = decoder_session.get_inputs()
    if len(inputs) < 3:
        raise RapidLatexRuntimeError(
            "RapidLaTeXOCR decoder 缺少 token、mask 或 encoder context 输入。"
        )
    token_name, mask_name, context_name = (item.name for item in inputs[:3])
    output = np.asarray([[bos_token]], dtype=np.int64)
    mask = np.ones_like(output, dtype=bool)
    generated: list[int] = []
    for _ in range(max(1, int(max_new_tokens))):
        recent_tokens = output[:, -512:]
        recent_mask = mask[:, -512:]
        logits = decoder_session.run(
            None,
            {
                token_name: recent_tokens,
                mask_name: recent_mask,
                context_name: context,
            },
        )[0]
        next_token = int(np.argmax(logits[:, -1, :], axis=-1)[0])
        output = np.concatenate(
            [output, np.asarray([[next_token]], dtype=np.int64)], axis=-1
        )
        mask = np.pad(mask, ((0, 0), (0, 1)), constant_values=True)
        generated.append(next_token)
        if next_token == eos_token:
            break
        repeated_suffix_start = repeated_token_suffix_start(
            generated,
            min_generated_tokens=64,
            min_repeated_tokens=32,
            max_period=4,
            min_repetitions=8,
        )
        if repeated_suffix_start is not None:
            del generated[repeated_suffix_start:]
            break
    return generated


def _post_process(text: str) -> str:
    return clean_recognized_latex(text)
