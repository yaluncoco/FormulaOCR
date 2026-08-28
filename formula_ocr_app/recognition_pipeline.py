from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Callable, Protocol

try:
    from formula_ocr_app.formula_formats import clean_recognized_latex
    from formula_ocr_app.model_api import DownloadProgressCallback
    from formula_ocr_app.model_catalog import DEFAULT_MODEL_ID
    from formula_ocr_app.model_runtime import (
        create_recognizer_backend,
        is_model_cached,
    )
except ModuleNotFoundError as exc:  # Allows `python formula_ocr_app/app.py`.
    if exc.name != "formula_ocr_app":
        raise
    from formula_formats import clean_recognized_latex
    from model_api import DownloadProgressCallback
    from model_catalog import DEFAULT_MODEL_ID
    from model_runtime import create_recognizer_backend, is_model_cached


class FormulaRecognizerBackend(Protocol):
    def predict(self, image_path: str | Path) -> str: ...

    def close(self) -> None: ...


RecognizerFactory = Callable[[str], FormulaRecognizerBackend]
ModelLoadCallback = Callable[[str, bool], None]


@dataclass(frozen=True)
class RecognitionResult:
    """Result from exactly one explicitly selected model."""

    selected_model_name: str
    model_name: str
    latex: str


class FormulaRecognizer:
    """Lazy recognizer that never changes the user's selected model."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_ID,
        model_dir: str | Path | None = None,
        device: str = "cpu",
        recognizer_factory: RecognizerFactory | None = None,
        model_load_callback: ModelLoadCallback | None = None,
        model_download_progress_callback: DownloadProgressCallback | None = None,
    ) -> None:
        self.model_name = model_name.strip() or DEFAULT_MODEL_ID
        self.model_dir = Path(model_dir).expanduser().resolve() if model_dir else None
        self.device = device.strip() if device else "cpu"
        self._recognizer_factory = recognizer_factory
        self._model_load_callback = model_load_callback
        self._model_download_progress_callback = model_download_progress_callback
        self._backend: FormulaRecognizerBackend | None = None
        self._backend_model_name = ""
        self.last_result: RecognitionResult | None = None
        self._predict_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._prediction_active = False
        self._close_requested = False
        self._closed = False

    def predict(self, image_path: str | Path) -> str:
        return self.recognize(image_path).latex

    def recognize(self, image_path: str | Path) -> RecognitionResult:
        backend_to_close: FormulaRecognizerBackend | None = None
        with self._predict_lock:
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("公式识别器已经关闭。")
                self._prediction_active = True
            try:
                self._notify_model_load(self.model_name)
                backend = self._get_backend(self.model_name)
                latex = clean_recognized_latex(backend.predict(image_path))
                result = RecognitionResult(
                    selected_model_name=self.model_name,
                    model_name=self.model_name,
                    latex=latex,
                )
                self.last_result = result
                return result
            finally:
                with self._state_lock:
                    self._prediction_active = False
                    if self._close_requested:
                        backend_to_close = self._detach_backend()
                        self._close_requested = False
                if backend_to_close is not None:
                    _close_backend_safely(backend_to_close)

    def close(self) -> None:
        backend: FormulaRecognizerBackend | None = None
        with self._state_lock:
            self._closed = True
            if self._prediction_active:
                # Native Paddle/ONNX sessions must not be released while their
                # predict call is executing. The worker closes them immediately
                # after inference instead of blocking Tk's shutdown callback.
                self._close_requested = True
                return
            backend = self._detach_backend()
            self._close_requested = False
        if backend is not None:
            _close_backend_safely(backend)

    def _detach_backend(self) -> FormulaRecognizerBackend | None:
        backend = self._backend
        self._backend = None
        self._backend_model_name = ""
        return backend

    def _get_backend(self, model_name: str) -> FormulaRecognizerBackend:
        if self._backend is not None and self._backend_model_name == model_name:
            return self._backend
        if self._backend is not None:
            _close_backend_safely(self._backend)
        self._backend = self._create_backend(model_name)
        self._backend_model_name = model_name
        return self._backend

    def _create_backend(self, model_name: str) -> FormulaRecognizerBackend:
        if self._recognizer_factory is not None:
            return self._recognizer_factory(model_name)
        return create_recognizer_backend(
            model_name,
            model_dir=self.model_dir,
            device=self.device,
            download_progress_callback=self._model_download_progress_callback,
        )

    def _notify_model_load(self, model_name: str) -> None:
        if self._model_load_callback is not None:
            self._model_load_callback(model_name, self._model_is_cached(model_name))

    def _model_is_cached(self, model_name: str) -> bool:
        if self.model_dir is not None:
            return True
        return is_model_cached(model_name)


def _close_backend_safely(backend: FormulaRecognizerBackend) -> None:
    try:
        backend.close()
    except Exception:
        # Releasing a native Paddle/ONNX session is best-effort cleanup. A
        # destructor error must not turn a successful recognition into a GUI
        # failure or crash the application while it is closing.
        pass
