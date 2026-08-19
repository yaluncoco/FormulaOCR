from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

try:
    from formula_ocr_app.model_downloader import DownloadProgressCallback
    from formula_ocr_app.unimernet_onnx_model_downloader import (
        ensure_unimernet_onnx_model,
    )
except ImportError:  # Allows `python formula_ocr_app/app.py`.
    from model_downloader import DownloadProgressCallback
    from unimernet_onnx_model_downloader import ensure_unimernet_onnx_model


class UniMERNetONNXRuntimeError(RuntimeError):
    """Raised when the quantized UniMERNet ONNX runtime cannot be initialized."""


class UniMERNetSmallFormulaRecognizer:
    """Run Cooper114's quantized UniMERNet Small ONNX export.

    This export has a standalone first decoder and a second decoder accepting
    the first decoder's self/cross-attention KV cache.  It is intentionally a
    separate adapter from the other TrOCR-style models because its cache
    tensors have different value-head dimensions and its image processor uses
    variable-size centered padding.
    """

    def __init__(
        self,
        *,
        device: str = "cpu",
        max_new_tokens: int = 512,
        download_progress_callback: DownloadProgressCallback | None = None,
        model_ensure: Callable[..., Path] = ensure_unimernet_onnx_model,
    ) -> None:
        self.device = device.strip() if device else "cpu"
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.download_progress_callback = download_progress_callback
        self._model_ensure = model_ensure
        self._encoder_session: Any | None = None
        self._decoder_session: Any | None = None
        self._decoder_with_past_session: Any | None = None
        self._tokenizer: Any | None = None
        self._model_dir: Path | None = None

    def close(self) -> None:
        self._encoder_session = None
        self._decoder_session = None
        self._decoder_with_past_session = None
        self._tokenizer = None
        self._model_dir = None

    def predict(self, image_path: str | Path) -> str:
        self._ensure_model()
        assert self._encoder_session is not None
        assert self._decoder_session is not None
        assert self._decoder_with_past_session is not None
        assert self._tokenizer is not None

        pixel_values = self._preprocess_image(Path(image_path))
        encoder_input = self._encoder_session.get_inputs()[0]
        encoder_hidden_states = self._encoder_session.run(
            None,
            {encoder_input.name: pixel_values},
        )[0]
        start_id, eos_id = self._generation_ids()
        token_ids = _generate_tokens(
            self._decoder_session,
            self._decoder_with_past_session,
            encoder_hidden_states,
            decoder_start_id=start_id,
            eos_id=eos_id,
            max_new_tokens=self.max_new_tokens,
        )
        result = self._tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        if not result:
            raise UniMERNetONNXRuntimeError("UniMERNet Small ONNX 未返回公式。")
        return result

    def _ensure_model(self) -> None:
        if (
            self._encoder_session is not None
            and self._decoder_session is not None
            and self._decoder_with_past_session is not None
            and self._tokenizer is not None
        ):
            return
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise UniMERNetONNXRuntimeError(
                "程序缺少 UniMERNet Small ONNX 的 onnxruntime 或 tokenizers 运行组件。"
            ) from exc

        model_dir = self._model_ensure(
            progress_callback=self.download_progress_callback,
        )
        providers = _onnx_providers(ort, self.device)
        try:
            encoder_session = ort.InferenceSession(
                str(model_dir / "encoder_model_quantized.onnx"),
                providers=providers,
            )
            decoder_session = ort.InferenceSession(
                str(model_dir / "decoder_model_quantized.onnx"),
                providers=providers,
            )
            decoder_with_past_session = ort.InferenceSession(
                str(model_dir / "decoder_with_past_model_quantized.onnx"),
                providers=providers,
            )
            tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        except Exception as exc:  # pragma: no cover - depends on local runtime
            raise UniMERNetONNXRuntimeError(
                "UniMERNet Small ONNX 模型初始化失败，请检查模型文件和运行时。"
            ) from exc

        self._model_dir = model_dir
        self._encoder_session = encoder_session
        self._decoder_session = decoder_session
        self._decoder_with_past_session = decoder_with_past_session
        self._tokenizer = tokenizer

    def _preprocess_image(self, image_path: Path):
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise UniMERNetONNXRuntimeError(
                "程序缺少 UniMERNet Small ONNX 的 Pillow 或 numpy 运行组件。"
            ) from exc

        width, height = self._input_size()
        mean, std = self._image_normalization()
        try:
            with Image.open(image_path) as image:
                image = _resize_to_fit(image.convert("RGB"), width, height)
                canvas = Image.new("RGB", (width, height), (255, 255, 255))
                left = (width - image.width) // 2
                top = (height - image.height) // 2
                canvas.paste(image, (left, top))
                pixels = np.asarray(canvas, dtype=np.float32) / 255.0
        except (OSError, ValueError) as exc:
            raise UniMERNetONNXRuntimeError(f"无法读取公式图片：{image_path}") from exc

        pixels = (pixels - np.asarray(mean, dtype=np.float32)) / np.asarray(
            std, dtype=np.float32
        )
        return np.transpose(pixels, (2, 0, 1))[None, ...].astype(np.float32)

    def _input_size(self) -> tuple[int, int]:
        if self._model_dir is not None:
            try:
                data = _read_json(self._model_dir / "preprocessor_config.json")
                size = data.get("size")
                if isinstance(size, (list, tuple)) and len(size) == 2:
                    height, width = int(size[0]), int(size[1])
                elif isinstance(size, dict):
                    height = int(size.get("height", 192))
                    width = int(size.get("width", 672))
                else:
                    raise ValueError
                if 32 <= height <= 2048 and 32 <= width <= 2048:
                    return width, height
            except (OSError, TypeError, ValueError):
                pass
        return 672, 192

    def _image_normalization(self) -> tuple[list[float], list[float]]:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if self._model_dir is None:
            return mean, std
        try:
            data = _read_json(self._model_dir / "preprocessor_config.json")
            candidate_mean = data.get("image_mean")
            candidate_std = data.get("image_std")
            if (
                isinstance(candidate_mean, list)
                and len(candidate_mean) == 3
                and isinstance(candidate_std, list)
                and len(candidate_std) == 3
            ):
                return [float(value) for value in candidate_mean], [
                    float(value) for value in candidate_std
                ]
        except (OSError, TypeError, ValueError):
            pass
        return mean, std

    def _generation_ids(self) -> tuple[int, int | None]:
        assert self._model_dir is not None
        start_id: int | None = None
        eos_id: int | None = None
        try:
            data = _read_json(self._model_dir / "config.json")
        except (OSError, ValueError):
            data = {}
        start_id = data.get("decoder_start_token_id", start_id)
        eos_id = data.get("eos_token_id", eos_id)
        decoder = data.get("decoder")
        if isinstance(decoder, dict):
            start_id = decoder.get("decoder_start_token_id", start_id)
            eos_id = decoder.get("eos_token_id", eos_id)
        return int(start_id if start_id is not None else 0), (
            int(eos_id) if eos_id is not None else None
        )


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return data if isinstance(data, dict) else {}


def _resize_to_fit(image: Any, width: int, height: int) -> Any:
    from PIL import Image

    if image.width <= width and image.height <= height:
        return image
    scale = min(width / max(image.width, 1), height / max(image.height, 1))
    resized_size = (
        max(1, min(width, int(image.width * scale))),
        max(1, min(height, int(image.height * scale))),
    )
    return image.resize(resized_size, Image.Resampling.BILINEAR)


def _generate_tokens(
    first_decoder_session: Any,
    decoder_with_past_session: Any,
    encoder_hidden_states: Any,
    *,
    decoder_start_id: int,
    eos_id: int | None,
    max_new_tokens: int,
) -> list[int]:
    import numpy as np

    first_inputs = first_decoder_session.get_inputs()
    input_ids_input = next(
        (item for item in first_inputs if item.name == "input_ids"), None
    )
    hidden_input = next(
        (item for item in first_inputs if item.name == "encoder_hidden_states"), None
    )
    if input_ids_input is None or hidden_input is None:
        raise UniMERNetONNXRuntimeError(
            "UniMERNet Small ONNX 首次 decoder 输入结构不完整。"
        )
    first_outputs = first_decoder_session.get_outputs()
    output_names = {item.name: index for index, item in enumerate(first_outputs)}
    logits_index = output_names.get("logits", 0)
    cache_outputs = [item for item in first_outputs if item.name.startswith("present.")]
    past_inputs = [
        item
        for item in decoder_with_past_session.get_inputs()
        if item.name.startswith("past_key_values.")
    ]
    if not cache_outputs or len(cache_outputs) != len(past_inputs):
        raise UniMERNetONNXRuntimeError(
            "UniMERNet Small ONNX decoder 缺少完整的首次/缓存输入输出。"
        )
    cache_outputs_by_name = {item.name: item for item in cache_outputs}
    for past_input in past_inputs:
        present_name = past_input.name.replace("past_key_values.", "present.", 1)
        if present_name not in cache_outputs_by_name:
            raise UniMERNetONNXRuntimeError(
                f"UniMERNet Small ONNX 首次 decoder 缺少 cache 输出：{present_name}"
            )
    past_output_names = {
        item.name: index
        for index, item in enumerate(decoder_with_past_session.get_outputs())
    }

    generated: list[int] = []
    first_result = first_decoder_session.run(
        None,
        {
            input_ids_input.name: np.asarray([[decoder_start_id]], dtype=np.int64),
            hidden_input.name: encoder_hidden_states,
        },
    )
    next_id = _next_token(first_result[logits_index])
    if eos_id is not None and next_id == eos_id:
        return generated
    generated.append(next_id)
    if len(generated) >= max(1, int(max_new_tokens)):
        return generated

    cache: dict[str, Any] = {}
    for past_input in past_inputs:
        present_name = past_input.name.replace("past_key_values.", "present.", 1)
        present_index = output_names.get(present_name)
        if present_index is None:
            raise UniMERNetONNXRuntimeError(
                f"UniMERNet Small ONNX 首次 decoder 缺少 cache 输出：{present_name}"
            )
        cache[past_input.name] = first_result[present_index]

    for _ in range(1, max(1, int(max_new_tokens))):
        inputs: dict[str, Any] = {
            "input_ids": np.asarray([[next_id]], dtype=np.int64),
        }
        inputs.update(cache)
        outputs = decoder_with_past_session.run(None, inputs)
        logits_index = past_output_names.get("logits", 0)
        next_id = _next_token(outputs[logits_index])
        if eos_id is not None and next_id == eos_id:
            break
        generated.append(next_id)
        if _repeated_token_suffix(generated) is not None:
            break
        for past_input in past_inputs:
            present_name = past_input.name.replace("past_key_values.", "present.", 1)
            output_index = past_output_names.get(present_name)
            if output_index is None:
                # The with-past export keeps encoder cross-attention KV
                # tensors as inputs but only emits updated decoder self-
                # attention KV tensors.  The encoder cache is static after
                # the first decoder invocation.
                if ".encoder." in present_name:
                    continue
                raise UniMERNetONNXRuntimeError(
                    f"UniMERNet Small ONNX 缺少 cache 输出：{present_name}"
                )
            cache[past_input.name] = outputs[output_index]
    return generated


def _next_token(logits: Any) -> int:
    import numpy as np

    return int(np.argmax(logits[:, -1, :], axis=-1)[0])


def _repeated_token_suffix(
    token_ids: list[int],
    *,
    min_generated_tokens: int = 64,
    min_repeated_tokens: int = 32,
    max_period: int = 4,
    min_repetitions: int = 8,
) -> int | None:
    count = len(token_ids)
    if count < min_generated_tokens:
        return None
    for period in range(1, max_period + 1):
        pattern = token_ids[-period:]
        start = count - period
        repetitions = 1
        while start >= period and token_ids[start - period : start] == pattern:
            start -= period
            repetitions += 1
        if repetitions >= min_repetitions and repetitions * period >= min_repeated_tokens:
            return start + period
    return None


def _onnx_providers(onnxruntime: Any, device: str) -> list[str]:
    available = set(onnxruntime.get_available_providers())
    normalized = device.lower()
    if normalized.startswith("gpu") or normalized.startswith("cuda"):
        if "CUDAExecutionProvider" not in available:
            raise UniMERNetONNXRuntimeError(
                "当前 ONNX Runtime 未提供 CUDAExecutionProvider，请选择 CPU。"
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
