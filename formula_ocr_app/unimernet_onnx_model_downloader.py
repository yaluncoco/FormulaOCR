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
    from formula_ocr_app.model_api import DownloadProgressCallback
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
    from model_api import DownloadProgressCallback
    from runtime_paths import bundled_external_model_dir, external_model_dir


UNIMERNET_ONNX_MODEL_ID = "UniMERNetSmallONNX"
UNIMERNET_ONNX_REPOSITORY = "Cooper114/unimernet-onnx"
# Pin the Hub revision so a future model-card update cannot silently change
# the model behind an existing application build.
UNIMERNET_ONNX_REVISION = "411ee76221baaad144ffbf996d4deef8df013b54"
UNIMERNET_ONNX_RELEASE_URL = (
    f"https://huggingface.co/{UNIMERNET_ONNX_REPOSITORY}/resolve/"
    f"{UNIMERNET_ONNX_REVISION}"
)
UNIMERNET_ONNX_MODEL_SUBDIRECTORY = "small"
UNIMERNET_ONNX_MODEL_PAGE_URL = (
    f"https://huggingface.co/{UNIMERNET_ONNX_REPOSITORY}/tree/"
    f"{UNIMERNET_ONNX_REVISION}/{UNIMERNET_ONNX_MODEL_SUBDIRECTORY}"
)
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class UniMERNetONNXModelFile:
    name: str
    size: int
    sha256: str

    @property
    def url(self) -> str:
        return (
            f"{UNIMERNET_ONNX_RELEASE_URL}/"
            f"{UNIMERNET_ONNX_MODEL_SUBDIRECTORY}/{self.name}"
        )


# This is the complete six-file inference payload from the pinned revision.
# The converter repository identifies the original model as
# wanderkid/unimernet_small.  No model files are committed to this project.
UNIMERNET_ONNX_MODEL_FILES = (
    UniMERNetONNXModelFile(
        "config.json",
        4_916,
        "5e5cecf286ff2cef6c90224fea89c0b9ee3bae932454969b82e6362bcb270599",
    ),
    UniMERNetONNXModelFile(
        "decoder_model_quantized.onnx",
        144_823_893,
        "902f105dba11a1b5ef0b475ff69421ec3cdb88897b9b5350711a4eeb9d86b064",
    ),
    UniMERNetONNXModelFile(
        "decoder_with_past_model_quantized.onnx",
        138_075_003,
        "3593f94e158a006290b267dbe8f4606bf510242b5836d56a8757bce1b5234c00",
    ),
    UniMERNetONNXModelFile(
        "encoder_model_quantized.onnx",
        64_884_402,
        "8ee7108e18fcf46f496b6e0d68a7dc8eb30d2190d76d6f8b0c4bf2c9a8a7db21",
    ),
    UniMERNetONNXModelFile(
        "preprocessor_config.json",
        617,
        "d06c4b87fd6aa74b8b2cdbc4361227d8073ce37e8cfe9e6442802d0bee64da03",
    ),
    UniMERNetONNXModelFile(
        "tokenizer.json",
        2_140_013,
        "02c318d9cfa95bf323371762b8f838a82709530274d36dba6eca880f0add6cc4",
    ),
)
UNIMERNET_ONNX_TOTAL_SIZE = sum(item.size for item in UNIMERNET_ONNX_MODEL_FILES)
UniMERNetONNXProgressCallback = Callable[[str, int, int], None]


class UniMERNetONNXModelDownloadError(RuntimeError):
    pass


def unimernet_onnx_model_dir() -> Path:
    return external_model_dir(UNIMERNET_ONNX_MODEL_ID)


def is_unimernet_onnx_model_cached(*, verify_hash: bool = False) -> bool:
    roots = [unimernet_onnx_model_dir()]
    bundled = bundled_external_model_dir(UNIMERNET_ONNX_MODEL_ID)
    if bundled is not None:
        roots.append(bundled)
    return any(_model_files_are_valid(root, verify_hash=verify_hash) for root in roots)


def ensure_unimernet_onnx_model(
    *, progress_callback: DownloadProgressCallback | None = None,
) -> Path:
    destination = unimernet_onnx_model_dir()
    if _model_files_are_valid(destination, verify_hash=True):
        _notify(progress_callback, UNIMERNET_ONNX_TOTAL_SIZE, UNIMERNET_ONNX_TOTAL_SIZE)
        return destination
    bundled = bundled_external_model_dir(UNIMERNET_ONNX_MODEL_ID)
    if bundled is not None and _model_files_are_valid(bundled, verify_hash=True):
        _notify(progress_callback, UNIMERNET_ONNX_TOTAL_SIZE, UNIMERNET_ONNX_TOTAL_SIZE)
        return bundled

    ensure_safe_directory(destination.parent)
    lock = InterProcessFileLock(
        destination.with_suffix(".lock"),
        on_wait=lambda: _notify(
            progress_callback,
            0,
            UNIMERNET_ONNX_TOTAL_SIZE,
        ),
    )
    with lock:
        if _model_files_are_valid(destination, verify_hash=True):
            _notify(progress_callback, UNIMERNET_ONNX_TOTAL_SIZE, UNIMERNET_ONNX_TOTAL_SIZE)
            return destination
        bundled = bundled_external_model_dir(UNIMERNET_ONNX_MODEL_ID)
        if bundled is not None and _model_files_are_valid(bundled, verify_hash=True):
            _notify(progress_callback, UNIMERNET_ONNX_TOTAL_SIZE, UNIMERNET_ONNX_TOTAL_SIZE)
            return bundled

        ensure_safe_directory(destination)
        downloads_root = destination.parent / ".downloads"
        ensure_safe_directory(downloads_root)
        downloads_dir = downloads_root / UNIMERNET_ONNX_MODEL_ID
        ensure_safe_directory(downloads_dir)
        completed = 0
        for item in UNIMERNET_ONNX_MODEL_FILES:
            target = destination / item.name
            if _file_is_valid(target, item, verify_hash=True):
                completed += item.size
                _notify(progress_callback, completed, UNIMERNET_ONNX_TOTAL_SIZE)
                continue
            _download_file(
                item,
                target,
                downloads_dir=downloads_dir,
                completed=completed,
                total=UNIMERNET_ONNX_TOTAL_SIZE,
                callback=progress_callback,
            )
            completed += item.size

    if not is_unimernet_onnx_model_cached(verify_hash=True):
        raise UniMERNetONNXModelDownloadError(
            f"UniMERNet Small ONNX 模型下载后校验失败：{destination}"
        )
    return destination


def _download_file(
    item: UniMERNetONNXModelFile,
    destination: Path,
    *,
    downloads_dir: Path,
    completed: int,
    total: int,
    callback: DownloadProgressCallback | None,
) -> None:
    import requests

    partial = downloads_dir / f"{item.name}.part"
    try:
        download_verified_file(
            item,
            destination,
            partial=partial,
            completed=completed,
            total=total,
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
            error_type=UniMERNetONNXModelDownloadError,
            label="UniMERNet Small ONNX",
        )


def _file_is_valid(
    path: Path,
    item: UniMERNetONNXModelFile,
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
        UNIMERNET_ONNX_MODEL_FILES,
        verify_hash=verify_hash,
    )


def _sha256(path: Path) -> str:
    return sha256_file(path, chunk_size=CHUNK_SIZE)


def _notify(
    callback: UniMERNetONNXProgressCallback | None,
    downloaded: int,
    total: int,
) -> None:
    if callback is not None:
        callback(
            UNIMERNET_ONNX_MODEL_ID,
            min(max(downloaded, 0), total),
            total,
        )
