"""Compatibility imports for the direct Paddle formula backend."""

try:
    from formula_ocr_app.model_downloader import ensure_official_model
    from formula_ocr_app.paddle_formula_recognizer import (
        PaddleFormulaRecognizer,
        PaddleFormulaRuntimeError,
        PaddleOCRNotReadyError,
    )
except ModuleNotFoundError as exc:  # Allows ``python formula_ocr_app/app.py``.
    if exc.name != "formula_ocr_app":
        raise
    from model_downloader import ensure_official_model
    from paddle_formula_recognizer import (
        PaddleFormulaRecognizer,
        PaddleFormulaRuntimeError,
        PaddleOCRNotReadyError,
    )

__all__ = [
    "PaddleFormulaRecognizer",
    "PaddleFormulaRuntimeError",
    "PaddleOCRNotReadyError",
    "ensure_official_model",
]
