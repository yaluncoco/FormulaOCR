from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from formula_ocr_app.model_downloader import DownloadProgressCallback
    from formula_ocr_app.rapid_model_downloader import ensure_rapid_model
except ImportError:  # Allows `python formula_ocr_app/app.py`.
    from model_downloader import DownloadProgressCallback
    from rapid_model_downloader import ensure_rapid_model


class RapidLatexRecognizer:
    def __init__(
        self,
        *,
        download_progress_callback: DownloadProgressCallback | None = None,
    ) -> None:
        self.download_progress_callback = download_progress_callback
        self._model: Any | None = None

    def predict(self, image_path: str | Path) -> str:
        self._ensure_model()
        assert self._model is not None
        result, _elapsed = self._model(str(Path(image_path).resolve()))
        if not result:
            raise RuntimeError("RapidLaTeXOCR 未返回公式。")
        return str(result)

    def close(self) -> None:
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        model_dir = ensure_rapid_model(
            progress_callback=self.download_progress_callback
        )
        try:
            from rapid_latex_ocr import LaTeXOCR
        except ImportError as exc:
            raise RuntimeError("程序缺少 RapidLaTeXOCR/ONNX Runtime 运行组件。") from exc
        self._model = LaTeXOCR(
            image_resizer_path=model_dir / "image_resizer.onnx",
            encoder_path=model_dir / "encoder.onnx",
            decoder_path=model_dir / "decoder.onnx",
            tokenizer_json=model_dir / "tokenizer.json",
        )
