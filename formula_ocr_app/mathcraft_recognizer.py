from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

try:
    from formula_ocr_app.model_downloader import DownloadProgressCallback
    from formula_ocr_app.mathcraft_model_downloader import ensure_mathcraft_model
except ImportError:  # Allows `python formula_ocr_app/app.py`.
    from model_downloader import DownloadProgressCallback
    from mathcraft_model_downloader import ensure_mathcraft_model


class MathCraftRuntimeError(RuntimeError):
    """Raised when the MathCraft ONNX runtime cannot be initialized."""


class MathCraftFormulaRecognizer:
    """Pure ONNX Runtime wrapper for the MathCraft Formula model.

    The upstream model is a TrOCR-style vision encoder/decoder.  Keeping the
    preprocessing and greedy loop here avoids importing Transformers, PyTorch,
    or torchvision into the desktop application.
    """

    def __init__(
        self,
        *,
        device: str = "cpu",
        max_new_tokens: int = 512,
        download_progress_callback: DownloadProgressCallback | None = None,
        model_ensure: Callable[..., Path] = ensure_mathcraft_model,
    ) -> None:
        self.device = device.strip() if device else "cpu"
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.download_progress_callback = download_progress_callback
        self._model_ensure = model_ensure
        self._encoder_session: Any | None = None
        self._decoder_session: Any | None = None
        self._tokenizer: Any | None = None
        self._model_dir: Path | None = None
        self.last_score = 0.0

    def close(self) -> None:
        self._encoder_session = None
        self._decoder_session = None
        self._tokenizer = None
        self._model_dir = None
        self.last_score = 0.0

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

        decoder_start_id, eos_id = self._generation_ids()
        token_ids, token_scores, repeated = _generate_formula_tokens(
            self._decoder_session,
            encoder_hidden_states,
            decoder_start_id=decoder_start_id,
            eos_id=eos_id,
            max_new_tokens=self.max_new_tokens,
        )
        ids = token_ids[0]
        self.last_score = (
            float(sum(token_scores[0]) / len(token_scores[0]))
            if token_scores[0]
            else 0.0
        )
        if repeated[0]:
            self.last_score = min(self.last_score, 0.5)
        result = self._tokenizer.decode(ids, skip_special_tokens=True).strip()
        if not result:
            raise MathCraftRuntimeError("MathCraft Formula 未返回公式。")
        return result

    def _ensure_model(self) -> None:
        if (
            self._encoder_session is not None
            and self._decoder_session is not None
            and self._tokenizer is not None
        ):
            return

        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise MathCraftRuntimeError(
                "程序缺少 MathCraft Formula 的 numpy、onnxruntime 或 tokenizers 运行组件。"
            ) from exc

        model_dir = self._model_ensure(
            progress_callback=self.download_progress_callback,
        )
        providers = _onnx_providers(ort, self.device)
        try:
            encoder_session = ort.InferenceSession(
                str(model_dir / "encoder_model.onnx"),
                providers=providers,
            )
            decoder_session = ort.InferenceSession(
                str(model_dir / "decoder_model.onnx"),
                providers=providers,
            )
            tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        except Exception as exc:  # pragma: no cover - depends on local runtime
            raise MathCraftRuntimeError(
                "MathCraft Formula ONNX 模型初始化失败，请检查模型文件和运行时。"
            ) from exc

        # Keep the import local: the test suite and the rest of the app can
        # still load the catalog on machines that have not installed ONNX.
        _ = np
        self._model_dir = model_dir
        self._encoder_session = encoder_session
        self._decoder_session = decoder_session
        self._tokenizer = tokenizer

    def _preprocess_image(self, image_path: Path):
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise MathCraftRuntimeError(
                "程序缺少 MathCraft Formula 的 Pillow 或 numpy 运行组件。"
            ) from exc

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image = image.resize(self._input_size(), Image.Resampling.BICUBIC)
                pixels = np.asarray(image, dtype=np.float32) / 255.0
        except (OSError, ValueError) as exc:
            raise MathCraftRuntimeError(f"无法读取公式图片：{image_path}") from exc

        pixels = (pixels - np.asarray([0.5, 0.5, 0.5], dtype=np.float32)) / np.asarray(
            [0.5, 0.5, 0.5], dtype=np.float32
        )
        return np.transpose(pixels, (2, 0, 1))[None, ...].astype(np.float32)

    def _input_size(self) -> tuple[int, int]:
        """Read the model's (width, height) input size from its processor config."""

        if self._model_dir is not None:
            config_path = self._model_dir / "preprocessor_config.json"
            try:
                data = json.loads(config_path.read_text(encoding="utf-8-sig"))
                size = data.get("size")
                if isinstance(size, dict):
                    height = int(size.get("height", 384))
                    width = int(size.get("width", 384))
                    if 32 <= height <= 2048 and 32 <= width <= 2048:
                        return width, height
            except (OSError, TypeError, ValueError):
                pass
        return 384, 384

    def _generation_ids(self) -> tuple[int, int | None]:
        assert self._model_dir is not None
        decoder_start_id: int | None = None
        eos_id: int | None = None
        for filename in ("generation_config.json", "config.json"):
            path = self._model_dir / filename
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                continue
            decoder_start_id = data.get("decoder_start_token_id", decoder_start_id)
            eos_id = data.get("eos_token_id", eos_id)
            decoder = data.get("decoder")
            if isinstance(decoder, dict):
                decoder_start_id = decoder.get(
                    "decoder_start_token_id", decoder_start_id
                )
                eos_id = decoder.get("eos_token_id", eos_id)
            if decoder_start_id is not None and eos_id is not None:
                break

        if decoder_start_id is None:
            decoder_start_id = 2
        return int(decoder_start_id), int(eos_id) if eos_id is not None else None


def _onnx_providers(onnxruntime: Any, device: str) -> list[str]:
    available = set(onnxruntime.get_available_providers())
    normalized = device.lower()
    if normalized.startswith("gpu") or normalized.startswith("cuda"):
        if "CUDAExecutionProvider" not in available:
            raise MathCraftRuntimeError(
                "当前 ONNX Runtime 未提供 CUDAExecutionProvider，请选择 CPU。"
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _softmax(logits: Any):
    import numpy as np

    logits = np.asarray(logits, dtype=np.float32)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=-1, keepdims=True)


def _generate_formula_tokens(
    decoder_session: Any,
    encoder_hidden_states: Any,
    *,
    decoder_start_id: int,
    eos_id: int | None,
    max_new_tokens: int,
) -> tuple[list[list[int]], list[list[float]], Any]:
    """Greedily decode a batch while allowing rows to finish independently."""

    import numpy as np

    batch_size = int(encoder_hidden_states.shape[0])
    active_indices = np.arange(batch_size, dtype=np.int64)
    active_input_ids = np.full(
        (batch_size, 1), decoder_start_id, dtype=np.int64
    )
    active_hidden_states = encoder_hidden_states
    token_ids: list[list[int]] = [[] for _ in range(batch_size)]
    token_scores: list[list[float]] = [[] for _ in range(batch_size)]
    stopped_for_repetition = np.zeros((batch_size,), dtype=bool)
    decoder_inputs = decoder_session.get_inputs()
    if len(decoder_inputs) < 2:
        raise MathCraftRuntimeError("MathCraft decoder 缺少 input_ids 和 encoder_hidden_states 输入。")
    input_ids_name = decoder_inputs[0].name
    hidden_states_name = decoder_inputs[1].name

    for _ in range(max(1, int(max_new_tokens))):
        logits = decoder_session.run(
            None,
            {
                input_ids_name: active_input_ids,
                hidden_states_name: active_hidden_states,
            },
        )[0]
        step_probs = _softmax(logits[:, -1, :])
        next_tokens = np.argmax(step_probs, axis=1).astype(np.int64)
        keep_active_rows: list[int] = []
        for active_row, next_token in enumerate(next_tokens.tolist()):
            result_row = int(active_indices[active_row])
            if eos_id is not None and next_token == eos_id:
                continue
            token_ids[result_row].append(int(next_token))
            token_scores[result_row].append(
                float(step_probs[active_row, next_token])
            )
            repeated_suffix_start = _repeated_token_suffix_start(token_ids[result_row])
            if repeated_suffix_start is not None:
                del token_ids[result_row][repeated_suffix_start:]
                del token_scores[result_row][repeated_suffix_start:]
                stopped_for_repetition[result_row] = True
                continue
            keep_active_rows.append(active_row)
        if not keep_active_rows:
            break
        active_input_ids = np.concatenate(
            [
                active_input_ids[keep_active_rows],
                next_tokens[keep_active_rows].reshape(len(keep_active_rows), 1),
            ],
            axis=1,
        )
        active_hidden_states = active_hidden_states[keep_active_rows]
        active_indices = active_indices[keep_active_rows]

    return token_ids, token_scores, stopped_for_repetition


def _repeated_token_suffix_start(
    token_ids: list[int],
    *,
    min_generated_tokens: int = 64,
    min_repeated_tokens: int = 32,
    max_period: int = 4,
    min_repetitions: int = 8,
) -> int | None:
    token_count = len(token_ids)
    if token_count < min_generated_tokens:
        return None
    for period in range(1, max_period + 1):
        pattern = token_ids[-period:]
        start = token_count - period
        repetitions = 1
        while start >= period and token_ids[start - period : start] == pattern:
            start -= period
            repetitions += 1
        if repetitions < min_repetitions or repetitions * period < min_repeated_tokens:
            continue
        return start + period
    return None
