from __future__ import annotations

try:
    from formula_ocr_app.mathcraft_recognizer import MathCraftFormulaRecognizer
    from formula_ocr_app.pix2text_model_downloader import ensure_pix2text_model
except ModuleNotFoundError as exc:  # Allows `python formula_ocr_app/app.py`.
    if exc.name != "formula_ocr_app":
        raise
    from mathcraft_recognizer import MathCraftFormulaRecognizer
    from pix2text_model_downloader import ensure_pix2text_model


class Pix2TextFormulaRecognizer(MathCraftFormulaRecognizer):
    """Pix2Text MFR 1.5 using the shared TrOCR-style ONNX implementation."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("max_new_tokens", 1024)
        super().__init__(model_ensure=ensure_pix2text_model, **kwargs)
