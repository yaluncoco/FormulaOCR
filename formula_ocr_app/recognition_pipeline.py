from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

try:
    from formula_ocr_app.formula_formats import clean_recognized_latex
    from formula_ocr_app.model_catalog import DEFAULT_MODEL_ID
    from formula_ocr_app.model_downloader import DownloadProgressCallback
    from formula_ocr_app.paddle_hf_model_downloader import (
        ensure_paddle_hf_model,
        is_paddle_hf_model_cached,
    )
    from formula_ocr_app.recognizer import PaddleFormulaRecognizer
    from formula_ocr_app.rapid_recognizer import RapidLatexRecognizer
    from formula_ocr_app.mathcraft_recognizer import MathCraftFormulaRecognizer
    from formula_ocr_app.pix2text_recognizer import Pix2TextFormulaRecognizer
    from formula_ocr_app.mixtex_recognizer import MixTexFormulaRecognizer
    from formula_ocr_app.unimernet_onnx_recognizer import (
        UniMERNetSmallFormulaRecognizer,
    )
    from formula_ocr_app.runtime_paths import is_paddle_model_cached
    from formula_ocr_app.mathcraft_model_downloader import is_mathcraft_model_cached
    from formula_ocr_app.pix2text_model_downloader import is_pix2text_model_cached
    from formula_ocr_app.rapid_model_downloader import is_rapid_model_cached
    from formula_ocr_app.mixtex_model_downloader import is_mixtex_model_cached
    from formula_ocr_app.unimernet_onnx_model_downloader import (
        is_unimernet_onnx_model_cached,
    )
except ImportError:  # Allows `python formula_ocr_app/app.py`.
    from formula_formats import clean_recognized_latex
    from model_catalog import DEFAULT_MODEL_ID
    from model_downloader import DownloadProgressCallback
    from paddle_hf_model_downloader import (
        ensure_paddle_hf_model,
        is_paddle_hf_model_cached,
    )
    from recognizer import PaddleFormulaRecognizer
    from rapid_recognizer import RapidLatexRecognizer
    from mathcraft_recognizer import MathCraftFormulaRecognizer
    from pix2text_recognizer import Pix2TextFormulaRecognizer
    from mixtex_recognizer import MixTexFormulaRecognizer
    from unimernet_onnx_recognizer import UniMERNetSmallFormulaRecognizer
    from runtime_paths import is_paddle_model_cached
    from mathcraft_model_downloader import is_mathcraft_model_cached
    from pix2text_model_downloader import is_pix2text_model_cached
    from rapid_model_downloader import is_rapid_model_cached
    from mixtex_model_downloader import is_mixtex_model_cached
    from unimernet_onnx_model_downloader import is_unimernet_onnx_model_cached


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
        paddleocr_repo: str | Path,
        model_name: str = DEFAULT_MODEL_ID,
        model_dir: str | Path | None = None,
        device: str = "cpu",
        recognizer_factory: RecognizerFactory | None = None,
        model_load_callback: ModelLoadCallback | None = None,
        model_download_progress_callback: DownloadProgressCallback | None = None,
    ) -> None:
        self.paddleocr_repo = Path(paddleocr_repo).expanduser().resolve()
        self.model_name = model_name.strip() or DEFAULT_MODEL_ID
        self.model_dir = Path(model_dir).expanduser().resolve() if model_dir else None
        self.device = device.strip() if device else "cpu"
        self._recognizer_factory = recognizer_factory
        self._model_load_callback = model_load_callback
        self._model_download_progress_callback = model_download_progress_callback
        self._backend: FormulaRecognizerBackend | None = None
        self._backend_model_name = ""
        self.last_result: RecognitionResult | None = None

    def predict(self, image_path: str | Path) -> str:
        result = self.recognize(image_path)
        self.last_result = result
        return result.latex

    def recognize(self, image_path: str | Path) -> RecognitionResult:
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

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
        self._backend = None
        self._backend_model_name = ""

    def _get_backend(self, model_name: str) -> FormulaRecognizerBackend:
        if self._backend is not None and self._backend_model_name == model_name:
            return self._backend
        if self._backend is not None:
            self._backend.close()
        self._backend = self._create_backend(model_name)
        self._backend_model_name = model_name
        return self._backend

    def _create_backend(self, model_name: str) -> FormulaRecognizerBackend:
        if self._recognizer_factory is not None:
            return self._recognizer_factory(model_name)
        if model_name == "RapidLaTeXOCR":
            return RapidLatexRecognizer(
                download_progress_callback=self._model_download_progress_callback,
            )
        if model_name == "MathCraftFormula":
            return MathCraftFormulaRecognizer(
                device=self.device,
                download_progress_callback=self._model_download_progress_callback,
            )
        if model_name == "Pix2TextMFR15":
            return Pix2TextFormulaRecognizer(
                device=self.device,
                download_progress_callback=self._model_download_progress_callback,
            )
        if model_name == "MixTexZhEn":
            return MixTexFormulaRecognizer(
                device=self.device,
                download_progress_callback=self._model_download_progress_callback,
            )
        if model_name == "UniMERNetSmallONNX":
            return UniMERNetSmallFormulaRecognizer(
                device=self.device,
                download_progress_callback=self._model_download_progress_callback,
            )
        if model_name == "LaTeX_OCR_rec":
            return PaddleFormulaRecognizer(
                paddleocr_repo=self.paddleocr_repo,
                model_name=model_name,
                model_dir=self.model_dir,
                device=self.device,
                model_ensure=ensure_paddle_hf_model,
                download_progress_callback=self._model_download_progress_callback,
            )
        return PaddleFormulaRecognizer(
            paddleocr_repo=self.paddleocr_repo,
            model_name=model_name,
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
        if model_name == "RapidLaTeXOCR":
            return is_rapid_model_cached()
        if model_name == "MathCraftFormula":
            return is_mathcraft_model_cached()
        if model_name == "Pix2TextMFR15":
            return is_pix2text_model_cached()
        if model_name == "MixTexZhEn":
            return is_mixtex_model_cached()
        if model_name == "UniMERNetSmallONNX":
            return is_unimernet_onnx_model_cached()
        if model_name == "LaTeX_OCR_rec":
            return is_paddle_hf_model_cached(verify_hash=True)
        return is_paddle_model_cached(model_name)
