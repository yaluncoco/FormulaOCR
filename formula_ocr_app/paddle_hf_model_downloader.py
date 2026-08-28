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
    from formula_ocr_app.model_api import ModelDownloadError
    from formula_ocr_app.runtime_paths import (
        bundled_paddle_model_dir,
        paddle_model_dir,
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
    from model_api import ModelDownloadError
    from runtime_paths import bundled_paddle_model_dir, paddle_model_dir


PADDLE_HF_MODEL_ID = "LaTeX_OCR_rec"
PADDLE_HF_REPOSITORY = "PaddlePaddle/LaTeX_OCR_rec"
# Pin the exact Hugging Face commit.  A moving `main` branch must never alter
# the bytes behind an already released application.
PADDLE_HF_REVISION = "563fb029dfdf5fc847d0677f3870039960e3a801"
PADDLE_HF_RELEASE_URL = (
    f"https://huggingface.co/{PADDLE_HF_REPOSITORY}/resolve/{PADDLE_HF_REVISION}"
)
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class PaddleHFModelFile:
    name: str
    size: int
    sha256: str

    @property
    def url(self) -> str:
        return f"{PADDLE_HF_RELEASE_URL}/{self.name}"


PADDLE_HF_MODEL_FILES = (
    PaddleHFModelFile(
        "config.json",
        76_351,
        "61246c3a0675bf57034340f9ea8e79752ce392cc97bc1f936c20b7ba84350336",
    ),
    PaddleHFModelFile(
        "inference.json",
        1_234_034,
        "9684e0125dbc0fa8a3e81a6d7db5ae32759878ea8b73bcd6ee55eba87f811650",
    ),
    PaddleHFModelFile(
        "inference.pdiparams",
        102_384_761,
        "1b2b27fa532e4e687116c9e3a141e4f17420c75ab53a0ef10692b5bb797e67d2",
    ),
    PaddleHFModelFile(
        "inference.yml",
        39_883,
        "56da5be526cc56fd6a6e1eb8720e91a2b60c8f520710503de51259e983d2f3ee",
    ),
)

PADDLE_HF_TOTAL_SIZE = sum(item.size for item in PADDLE_HF_MODEL_FILES)
PaddleHFProgressCallback = Callable[[str, int, int], None]


class PaddleHFModelDownloadError(ModelDownloadError):
    pass


def paddle_hf_model_dir() -> Path:
    return paddle_model_dir(PADDLE_HF_MODEL_ID)


def is_paddle_hf_model_cached(*, verify_hash: bool = False) -> bool:
    roots = [paddle_hf_model_dir()]
    bundled = bundled_paddle_model_dir(PADDLE_HF_MODEL_ID)
    if bundled is not None:
        roots.append(bundled)
    return any(_model_files_are_valid(root, verify_hash=verify_hash) for root in roots)


def ensure_paddle_hf_model(
    model_name: str = PADDLE_HF_MODEL_ID,
    *,
    progress_callback: PaddleHFProgressCallback | None = None,
) -> Path:
    if model_name != PADDLE_HF_MODEL_ID:
        raise PaddleHFModelDownloadError(
            f"不支持的 Hugging Face Paddle 模型：{model_name}"
        )

    root = paddle_hf_model_dir()
    if _model_files_are_valid(root, verify_hash=True):
        _notify(progress_callback, PADDLE_HF_TOTAL_SIZE, PADDLE_HF_TOTAL_SIZE)
        return root
    bundled = bundled_paddle_model_dir(PADDLE_HF_MODEL_ID)
    if bundled is not None and _model_files_are_valid(bundled, verify_hash=True):
        _notify(progress_callback, PADDLE_HF_TOTAL_SIZE, PADDLE_HF_TOTAL_SIZE)
        return bundled

    ensure_safe_directory(root.parent)
    lock = InterProcessFileLock(
        root.with_suffix(".lock"),
        on_wait=lambda: _notify(progress_callback, 0, PADDLE_HF_TOTAL_SIZE),
    )
    with lock:
        if _model_files_are_valid(root, verify_hash=True):
            _notify(progress_callback, PADDLE_HF_TOTAL_SIZE, PADDLE_HF_TOTAL_SIZE)
            return root
        bundled = bundled_paddle_model_dir(PADDLE_HF_MODEL_ID)
        if bundled is not None and _model_files_are_valid(bundled, verify_hash=True):
            _notify(progress_callback, PADDLE_HF_TOTAL_SIZE, PADDLE_HF_TOTAL_SIZE)
            return bundled

        ensure_safe_directory(root)
        downloads_root = root.parent / ".downloads"
        ensure_safe_directory(downloads_root)
        downloads_dir = downloads_root / PADDLE_HF_MODEL_ID
        ensure_safe_directory(downloads_dir)
        completed = 0
        for item in PADDLE_HF_MODEL_FILES:
            destination = root / item.name
            if _file_is_valid(destination, item, verify_hash=True):
                completed += item.size
                _notify(progress_callback, completed, PADDLE_HF_TOTAL_SIZE)
                continue
            _download_file(
                item,
                destination,
                downloads_dir=downloads_dir,
                completed=completed,
                callback=progress_callback,
            )
            completed += item.size

    if not is_paddle_hf_model_cached(verify_hash=True):
        raise PaddleHFModelDownloadError("LaTeX-OCR 模型下载后校验失败。")
    return root


def _download_file(
    item: PaddleHFModelFile,
    destination: Path,
    *,
    downloads_dir: Path,
    completed: int,
    callback: PaddleHFProgressCallback | None,
) -> None:
    import requests

    partial = downloads_dir / f"{item.name}.part"
    try:
        download_verified_file(
            item,
            destination,
            partial=partial,
            completed=completed,
            total=PADDLE_HF_TOTAL_SIZE,
            notify=lambda downloaded, expected: _notify(
                callback, downloaded, expected
            ),
            request_get=requests.get,
            request_exception=requests.RequestException,
            chunk_size=CHUNK_SIZE,
        )
    except VerifiedDownloadFailure as failure:
        raise_model_download_error(
            failure,
            error_type=PaddleHFModelDownloadError,
            label="LaTeX-OCR",
        )


def _file_is_valid(
    path: Path,
    item: PaddleHFModelFile,
    *,
    verify_hash: bool,
) -> bool:
    return file_is_valid(
        path,
        item.size,
        item.sha256,
        verify_hash=verify_hash,
    )


def _model_files_are_valid(root: Path, *, verify_hash: bool) -> bool:
    return model_files_are_valid(
        root,
        PADDLE_HF_MODEL_FILES,
        verify_hash=verify_hash,
    )


def _sha256(path: Path) -> str:
    return sha256_file(path, chunk_size=CHUNK_SIZE)


def _notify(
    callback: PaddleHFProgressCallback | None,
    downloaded: int,
    total: int,
) -> None:
    if callback is not None:
        callback(
            PADDLE_HF_MODEL_ID,
            min(max(downloaded, 0), total),
            total,
        )
