from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    from formula_ocr_app.download_utils import (
        VerifiedDownloadFailure,
        download_verified_file,
        ensure_safe_directory,
        file_is_valid,
        model_files_are_valid,
        raise_model_download_error,
        sha256_file,
    )
    from formula_ocr_app.interprocess_lock import InterProcessFileLock
    from formula_ocr_app.runtime_paths import (
        bundled_external_model_dir,
        external_model_dir,
    )
except ModuleNotFoundError as exc:  # Allows `python formula_ocr_app/app.py`.
    if exc.name != "formula_ocr_app":
        raise
    from download_utils import (
        VerifiedDownloadFailure,
        download_verified_file,
        ensure_safe_directory,
        file_is_valid,
        model_files_are_valid,
        raise_model_download_error,
        sha256_file,
    )
    from interprocess_lock import InterProcessFileLock
    from runtime_paths import bundled_external_model_dir, external_model_dir


RAPID_RELEASE_BASE = (
    "https://github.com/RapidAI/RapidLaTeXOCR/releases/download/v0.0.0"
)
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class RapidModelFile:
    name: str
    size: int
    sha256: str

    @property
    def url(self) -> str:
        return f"{RAPID_RELEASE_BASE}/{self.name}"


RAPID_MODEL_FILES = (
    RapidModelFile(
        "image_resizer.onnx",
        38_967_751,
        "e0b075c39700f64d50400f39c8fc186bbb3b5d84d31864008313f376603aca9d",
    ),
    RapidModelFile(
        "encoder.onnx",
        89_008_136,
        "01bf5dc25539ca0cd5b1bd29296ea495977a6ba5f629dc4178277809d26e5e7d",
    ),
    RapidModelFile(
        "decoder.onnx",
        50_952_726,
        "bd695497bf1b22279b7626f5916c79226e1e244c84355f8da7edfd2d921d0072",
    ),
    RapidModelFile(
        "tokenizer.json",
        24_174,
        "1dc27b18d6a518d0d5ff3f4bb7bd98521fe80ad39e5b2a246d4109f1bb9d5019",
    ),
)

RapidProgressCallback = Callable[[str, int, int], None]


def rapid_model_dir() -> Path:
    return external_model_dir("RapidLaTeXOCR")


def is_rapid_model_cached(*, verify_hash: bool = False) -> bool:
    roots = [rapid_model_dir()]
    bundled = bundled_external_model_dir("RapidLaTeXOCR")
    if bundled is not None:
        roots.append(bundled)
    return any(_model_files_are_valid(root, verify_hash=verify_hash) for root in roots)


def ensure_rapid_model(
    *, progress_callback: RapidProgressCallback | None = None
) -> Path:
    root = rapid_model_dir()
    total = sum(spec.size for spec in RAPID_MODEL_FILES)
    if _model_files_are_valid(root, verify_hash=True):
        _notify(progress_callback, total, total)
        return root
    bundled = bundled_external_model_dir("RapidLaTeXOCR")
    if bundled is not None and _model_files_are_valid(bundled, verify_hash=True):
        _notify(progress_callback, total, total)
        return bundled

    ensure_safe_directory(root.parent)
    lock = InterProcessFileLock(
        root.with_suffix(".lock"),
        on_wait=lambda: _notify(progress_callback, 0, total),
    )
    with lock:
        if _model_files_are_valid(root, verify_hash=True):
            _notify(progress_callback, total, total)
            return root
        ensure_safe_directory(root)
        bundled = bundled_external_model_dir("RapidLaTeXOCR")
        if bundled is not None and _model_files_are_valid(bundled, verify_hash=True):
            _notify(progress_callback, total, total)
            return bundled
        completed = 0
        for spec in RAPID_MODEL_FILES:
            path = root / spec.name
            if _file_is_valid(path, spec, verify_hash=True):
                completed += spec.size
                _notify(progress_callback, completed, total)
                continue
            _download_file(
                spec,
                path,
                completed=completed,
                total=total,
                callback=progress_callback,
            )
            completed += spec.size
    if not is_rapid_model_cached(verify_hash=True):
        raise RuntimeError("RapidLaTeXOCR 模型下载后校验失败。")
    return root


def _download_file(
    spec: RapidModelFile,
    path: Path,
    *,
    completed: int,
    total: int,
    callback: RapidProgressCallback | None,
) -> None:
    import requests

    partial = path.with_suffix(path.suffix + ".part")
    try:
        download_verified_file(
            spec,
            path,
            partial=partial,
            completed=completed,
            total=total,
            notify=lambda downloaded, expected: _notify(
                callback, downloaded, expected
            ),
            request_get=requests.get,
            request_exception=requests.RequestException,
            timeout=(20, 120),
            chunk_size=CHUNK_SIZE,
        )
    except VerifiedDownloadFailure as failure:
        raise_model_download_error(
            failure,
            error_type=RuntimeError,
            label="RapidLaTeXOCR",
        )


def _file_is_valid(
    path: Path, spec: RapidModelFile, *, verify_hash: bool
) -> bool:
    return file_is_valid(
        path,
        spec.size,
        spec.sha256,
        verify_hash=verify_hash,
    )


def _model_files_are_valid(root: Path, *, verify_hash: bool) -> bool:
    return model_files_are_valid(
        root,
        RAPID_MODEL_FILES,
        verify_hash=verify_hash,
    )


def _sha256(path: Path) -> str:
    return sha256_file(path, chunk_size=CHUNK_SIZE)


def _notify(
    callback: RapidProgressCallback | None, downloaded: int, total: int
) -> None:
    if callback is not None:
        callback("RapidLaTeXOCR", min(downloaded, total), total)
