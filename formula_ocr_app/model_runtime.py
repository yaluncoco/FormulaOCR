"""Lazy registry for model download, cache and recognizer operations."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    from formula_ocr_app.interprocess_lock import InterProcessFileLock
    from formula_ocr_app.model_api import DownloadProgressCallback
    from formula_ocr_app.model_catalog import FormulaModelSpec, get_model_spec
    from formula_ocr_app.runtime_paths import (
        external_model_cache_size,
        external_model_dir,
        external_model_has_data,
        is_external_model_bundled,
        is_paddle_model_bundled,
        is_paddle_model_cached,
        paddle_model_cache_size,
        paddle_model_dir,
        paddle_model_has_data,
        remove_external_model,
        remove_paddle_model,
    )
except ModuleNotFoundError as exc:  # Allows ``python formula_ocr_app/app.py``.
    if exc.name != "formula_ocr_app":
        raise
    from interprocess_lock import InterProcessFileLock
    from model_api import DownloadProgressCallback
    from model_catalog import FormulaModelSpec, get_model_spec
    from runtime_paths import (
        external_model_cache_size,
        external_model_dir,
        external_model_has_data,
        is_external_model_bundled,
        is_paddle_model_bundled,
        is_paddle_model_cached,
        paddle_model_cache_size,
        paddle_model_dir,
        paddle_model_has_data,
        remove_external_model,
        remove_paddle_model,
    )


@dataclass(frozen=True)
class _ExternalBackend:
    recognizer_module: str
    recognizer_class: str
    downloader_module: str
    ensure_function: str
    cache_function: str


_EXTERNAL_BACKENDS = {
    "rapid_onnx": _ExternalBackend(
        "rapid_recognizer",
        "RapidLatexRecognizer",
        "rapid_model_downloader",
        "ensure_rapid_model",
        "is_rapid_model_cached",
    ),
    "mathcraft_onnx": _ExternalBackend(
        "mathcraft_recognizer",
        "MathCraftFormulaRecognizer",
        "mathcraft_model_downloader",
        "ensure_mathcraft_model",
        "is_mathcraft_model_cached",
    ),
    "pix2text_onnx": _ExternalBackend(
        "pix2text_recognizer",
        "Pix2TextFormulaRecognizer",
        "pix2text_model_downloader",
        "ensure_pix2text_model",
        "is_pix2text_model_cached",
    ),
    "mixtex_onnx": _ExternalBackend(
        "mixtex_recognizer",
        "MixTexFormulaRecognizer",
        "mixtex_model_downloader",
        "ensure_mixtex_model",
        "is_mixtex_model_cached",
    ),
    "unimernet_onnx": _ExternalBackend(
        "unimernet_onnx_recognizer",
        "UniMERNetSmallFormulaRecognizer",
        "unimernet_onnx_model_downloader",
        "ensure_unimernet_onnx_model",
        "is_unimernet_onnx_model_cached",
    ),
}


def create_recognizer_backend(
    model_id: str,
    *,
    model_dir: str | Path | None,
    device: str,
    download_progress_callback: DownloadProgressCallback | None,
) -> Any:
    spec = get_model_spec(model_id)
    external = _EXTERNAL_BACKENDS.get(spec.backend)
    if external is not None:
        backend_class = getattr(
            _load_module(external.recognizer_module),
            external.recognizer_class,
        )
        return backend_class(
            device=device,
            download_progress_callback=download_progress_callback,
        )

    if not spec.uses_paddle_runtime:
        raise ValueError(f"Unsupported formula recognizer backend: {spec.backend}")
    model_ensure: Callable[..., Path] | None = None
    if spec.backend == "paddle_hf":
        model_ensure = getattr(
            _load_module("paddle_hf_model_downloader"),
            "ensure_paddle_hf_model",
        )
    backend_class = getattr(
        _load_module("paddle_formula_recognizer"),
        "PaddleFormulaRecognizer",
    )
    return backend_class(
        model_name=model_id,
        model_dir=model_dir,
        device=device,
        model_ensure=model_ensure,
        download_progress_callback=download_progress_callback,
    )


def ensure_model(
    model_id: str,
    *,
    progress_callback: DownloadProgressCallback | None = None,
) -> Path:
    spec = get_model_spec(model_id)
    external = _EXTERNAL_BACKENDS.get(spec.backend)
    if external is not None:
        ensure = getattr(_load_module(external.downloader_module), external.ensure_function)
        return Path(ensure(progress_callback=progress_callback))
    if spec.backend == "paddle_hf":
        ensure = getattr(
            _load_module("paddle_hf_model_downloader"),
            "ensure_paddle_hf_model",
        )
        return Path(ensure(model_id, progress_callback=progress_callback))
    ensure = getattr(_load_module("model_downloader"), "ensure_official_model")
    return Path(ensure(model_id, progress_callback=progress_callback))


def is_model_cached(model_id: str, *, verify_hash: bool = True) -> bool:
    spec = get_model_spec(model_id)
    external = _EXTERNAL_BACKENDS.get(spec.backend)
    if external is not None:
        checker = getattr(_load_module(external.downloader_module), external.cache_function)
        return bool(checker(verify_hash=verify_hash))
    if spec.backend == "paddle_hf":
        checker = getattr(
            _load_module("paddle_hf_model_downloader"),
            "is_paddle_hf_model_cached",
        )
        return bool(checker(verify_hash=verify_hash))
    return is_paddle_model_cached(model_id)


def is_model_bundled(model_id: str) -> bool:
    spec = get_model_spec(model_id)
    if spec.uses_paddle_runtime:
        return is_paddle_model_bundled(model_id)
    return is_external_model_bundled(model_id)


def is_model_bundled_only(model_id: str) -> bool:
    """Return whether the only valid copy available to the app is bundled."""

    spec = get_model_spec(model_id)
    if spec.uses_paddle_runtime:
        bundled = is_paddle_model_bundled(model_id)
        if spec.backend == "paddle_hf":
            bundled = bundled and is_model_cached(model_id, verify_hash=True)
        return bundled and not paddle_model_has_data(model_id)
    return is_external_model_bundled(model_id) and not external_model_has_data(
        model_id
    )


def model_cache_size(model_id: str) -> int:
    spec = get_model_spec(model_id)
    if spec.uses_paddle_runtime:
        return paddle_model_cache_size(model_id)
    return external_model_cache_size(model_id)


def model_has_user_cache_data(model_id: str) -> bool:
    spec = get_model_spec(model_id)
    if spec.uses_paddle_runtime:
        return paddle_model_has_data(model_id)
    return external_model_has_data(model_id)


def model_user_cache_path(model_id: str) -> Path:
    spec = get_model_spec(model_id)
    if spec.uses_paddle_runtime:
        return paddle_model_dir(model_id)
    return external_model_dir(model_id)


def model_status_label(
    model_id: str,
    *,
    cached: bool | None = None,
) -> str:
    """Return the user-facing cache state shared by model-selection UIs."""

    spec = get_model_spec(model_id)
    if cached is None:
        # Status labels are queried for every catalog item while opening the
        # picker. Full SHA-256 validation remains part of model installation
        # and recognition, but it must not block Tk while painting a menu.
        cached = is_model_cached(model_id, verify_hash=False)
    if cached and is_model_bundled_only(model_id):
        return "随包内置"
    if not cached and model_has_user_cache_data(model_id):
        return "下载未完成 · 可继续"
    if spec.backend == "paddle_hf" and is_model_bundled(model_id) and not cached:
        return "随包校验失败"
    if not spec.uses_paddle_runtime and is_model_bundled(model_id):
        return "已下载" if cached else "随包校验失败"
    return "已下载" if cached else "待下载"


def remove_model(model_id: str) -> bool:
    spec = get_model_spec(model_id)
    lock_path = _model_mutation_lock_path(spec)
    try:
        with InterProcessFileLock(lock_path, timeout=0):
            if spec.uses_paddle_runtime:
                return remove_paddle_model(model_id)
            return remove_external_model(model_id)
    except TimeoutError as exc:
        raise OSError(
            "该模型正在另一个 FormulaOCR 窗口中下载，暂时不能删除。"
        ) from exc


def _model_mutation_lock_path(spec: FormulaModelSpec) -> Path:
    if spec.backend == "paddle":
        destination = paddle_model_dir(spec.model_id)
        return destination.parent / ".downloads" / f"{spec.model_id}.lock"
    if spec.uses_paddle_runtime:
        return paddle_model_dir(spec.model_id).with_suffix(".lock")
    return external_model_dir(spec.model_id).with_suffix(".lock")


def _load_module(module_name: str):
    if __package__:
        return importlib.import_module(f".{module_name}", package=__package__)
    return importlib.import_module(module_name)
