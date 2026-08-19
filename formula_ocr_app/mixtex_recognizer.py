from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from formula_ocr_app.mixtex_model_downloader import ensure_mixtex_model
    from formula_ocr_app.model_downloader import DownloadProgressCallback
except ImportError:  # Allows `python formula_ocr_app/app.py`.
    from mixtex_model_downloader import ensure_mixtex_model
    from model_downloader import DownloadProgressCallback


class MixTexRuntimeError(RuntimeError):
    """Raised when the official MixTeX ONNX runtime cannot be initialized."""


class MixTexFormulaRecognizer:
    """Run the official MixTeX merged-decoder ONNX release.

    MixTeX is not the same export as the MathCraft/Pix2Text models: its
    decoder accepts past key/value tensors and its official preprocessing
    pads each crop to 448x448.  Keeping this adapter separate prevents a
    seemingly compatible TrOCR loop from silently producing bad results.
    """

    def __init__(
        self,
        *,
        device: str = "cpu",
        max_new_tokens: int = 296,
        download_progress_callback: DownloadProgressCallback | None = None,
    ) -> None:
        self.device = device.strip() if device else "cpu"
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.download_progress_callback = download_progress_callback
        self._encoder_session: Any | None = None
        self._decoder_session: Any | None = None
        self._tokenizer: Any | None = None
        self._model_dir: Path | None = None

    def close(self) -> None:
        self._encoder_session = None
        self._decoder_session = None
        self._tokenizer = None
        self._model_dir = None

    def predict(self, image_path: str | Path) -> str:
        self._ensure_model()
        assert self._encoder_session is not None
        assert self._decoder_session is not None
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
            encoder_hidden_states,
            decoder_start_id=start_id,
            eos_id=eos_id,
            max_new_tokens=self.max_new_tokens,
        )
        result = self._tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        if not result:
            raise MixTexRuntimeError("MixTeX 未返回公式。")
        return result

    def _ensure_model(self) -> None:
        if (
            self._encoder_session is not None
            and self._decoder_session is not None
            and self._tokenizer is not None
        ):
            return
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise MixTexRuntimeError(
                "程序缺少 MixTeX 的 onnxruntime 或 tokenizers 运行组件。"
            ) from exc

        model_dir = ensure_mixtex_model(
            progress_callback=self.download_progress_callback,
        )
        providers = _onnx_providers(ort, self.device)
        try:
            encoder_session = ort.InferenceSession(
                str(model_dir / "encoder_model.onnx"),
                providers=providers,
            )
            decoder_session = ort.InferenceSession(
                str(model_dir / "decoder_model_merged.onnx"),
                providers=providers,
            )
            tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        except Exception as exc:  # pragma: no cover - depends on local runtime
            raise MixTexRuntimeError(
                "MixTeX ONNX 模型初始化失败，请检查模型文件和运行时。"
            ) from exc

        self._model_dir = model_dir
        self._encoder_session = encoder_session
        self._decoder_session = decoder_session
        self._tokenizer = tokenizer

    def _preprocess_image(self, image_path: Path):
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise MixTexRuntimeError(
                "程序缺少 MixTeX 的 Pillow 或 numpy 运行组件。"
            ) from exc
        try:
            with Image.open(image_path) as image:
                image = _pad_image(image.convert("RGB"), (448, 448))
                pixels = np.asarray(image, dtype=np.float32) / 255.0
        except (OSError, ValueError) as exc:
            raise MixTexRuntimeError(f"无法读取公式图片：{image_path}") from exc

        # The release's ViTImageProcessor uses mean/std = 0.5 for all RGB
        # channels.  Keep the constants explicit so no Transformers import is
        # required in the desktop build.
        pixels = (pixels - 0.5) / 0.5
        return np.transpose(pixels, (2, 0, 1))[None, ...].astype(np.float32)

    def _generation_ids(self) -> tuple[int, int | None]:
        assert self._model_dir is not None
        start_id: int | None = None
        eos_id: int | None = None
        for filename in ("generation_config.json", "config.json"):
            path = self._model_dir / filename
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                continue
            start_id = data.get("decoder_start_token_id", start_id)
            eos_id = data.get("eos_token_id", eos_id)
            decoder = data.get("decoder")
            if isinstance(decoder, dict):
                start_id = decoder.get("decoder_start_token_id", start_id)
                eos_id = decoder.get("eos_token_id", eos_id)
            if start_id is not None and eos_id is not None:
                break
        return int(start_id if start_id is not None else 0), (
            int(eos_id) if eos_id is not None else None
        )


def _pad_image(image: Any, out_size: tuple[int, int]) -> Any:
    """Match the official MixTeX pad-and-scale preprocessing."""

    from PIL import Image

    out_width, out_height = out_size
    background = Image.new("RGB", (out_width, out_height), (255, 255, 255))
    width, height = image.size
    if width < out_width and height < out_height:
        x = (out_width - width) // 2
        y = (out_height - height) // 2
        background.paste(image, (x, y))
        return background
    scale = min(out_width / max(width, 1), out_height / max(height, 1))
    resized = image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.Resampling.LANCZOS,
    )
    x = (out_width - resized.width) // 2
    y = (out_height - resized.height) // 2
    background.paste(resized, (x, y))
    return background


def _generate_tokens(
    decoder_session: Any,
    encoder_hidden_states: Any,
    *,
    decoder_start_id: int,
    eos_id: int | None,
    max_new_tokens: int,
) -> list[int]:
    import numpy as np

    decoder_inputs = decoder_session.get_inputs()
    input_ids_input = next(
        (item for item in decoder_inputs if item.name == "input_ids"), None
    )
    hidden_input = next(
        (item for item in decoder_inputs if item.name == "encoder_hidden_states"),
        None,
    )
    cache_inputs = [
        item for item in decoder_inputs if item.name.startswith("past_key_values.")
    ]
    cache_inputs.sort(key=lambda item: item.name)
    cache_branch = next(
        (item for item in decoder_inputs if item.name == "use_cache_branch"), None
    )
    if input_ids_input is None or hidden_input is None or not cache_inputs:
        raise MixTexRuntimeError("MixTeX merged decoder 输入结构不完整。")

    # Derive cache shapes from the ONNX signature instead of hard-coding a
    # layer count.  The v3.2.4 release has 3 layers, 12 heads and head size 64.
    first_shape = cache_inputs[0].shape
    try:
        heads = int(first_shape[1])
        head_size = int(first_shape[3])
    except (IndexError, TypeError, ValueError) as exc:
        raise MixTexRuntimeError("MixTeX decoder cache 形状无法识别。") from exc
    if heads <= 0 or head_size <= 0:
        raise MixTexRuntimeError("MixTeX decoder cache 形状无效。")

    present_names = {item.name for item in decoder_session.get_outputs()[1:]}
    if len(present_names) != len(cache_inputs):
        raise MixTexRuntimeError("MixTeX decoder 缺少完整的 present cache 输出。")

    next_input_id = decoder_start_id
    cache: dict[str, Any] = {
        item.name: np.zeros(
            (1, heads, 0, head_size),
            dtype=np.float32,
        )
        for item in cache_inputs
    }
    generated: list[int] = []
    output_specs = decoder_session.get_outputs()
    output_indices = {item.name: index for index, item in enumerate(output_specs)}
    for _ in range(max(1, int(max_new_tokens))):
        inputs: dict[str, Any] = {
            input_ids_input.name: np.asarray([[next_input_id]], dtype=np.int64),
            hidden_input.name: encoder_hidden_states,
        }
        inputs.update(cache)
        if cache_branch is not None:
            inputs[cache_branch.name] = np.asarray([True], dtype=bool)
        outputs = decoder_session.run(None, inputs)
        logits = outputs[0]
        next_id = int(np.argmax(logits[:, -1, :], axis=-1)[0])
        if eos_id is not None and next_id == eos_id:
            break
        generated.append(next_id)
        if _repeated_token_suffix(generated) is not None:
            break
        for cache_input in cache_inputs:
            present_name = cache_input.name.replace(
                "past_key_values.", "present.", 1
            )
            output_index = output_indices.get(present_name)
            if output_index is None:
                raise MixTexRuntimeError(
                    f"MixTeX decoder 缺少 cache 输出：{present_name}"
                )
            cache[cache_input.name] = outputs[output_index]
        next_input_id = next_id
    return generated


def _repeated_token_suffix(
    token_ids: list[int],
    *,
    min_generated_tokens: int = 42,
    min_repeated_tokens: int = 21,
    max_period: int = 4,
    min_repetitions: int = 7,
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
            raise MixTexRuntimeError(
                "当前 ONNX Runtime 未提供 CUDAExecutionProvider，请选择 CPU。"
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
