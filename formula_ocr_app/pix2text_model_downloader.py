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


PIX2TEXT_MODEL_ID = "Pix2TextMFR15"
PIX2TEXT_REPOSITORY = "breezedeus/pix2text-mfr-1.5"
# Pin the Hub revision so a future upstream update cannot silently change the
# model behind an existing application build.
PIX2TEXT_REVISION = "1cef9f0bdcd6a4c63df7de1311fb0894593340cc"
PIX2TEXT_RELEASE_URL = (
    f"https://huggingface.co/{PIX2TEXT_REPOSITORY}/resolve/{PIX2TEXT_REVISION}"
)
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class Pix2TextModelFile:
    name: str
    size: int
    sha256: str

    @property
    def url(self) -> str:
        return f"{PIX2TEXT_RELEASE_URL}/{self.name}"


# Hashes and sizes were obtained from the pinned public Hugging Face revision.
# The eight files are the complete inference payload; README and Git metadata
# are deliberately not downloaded.
PIX2TEXT_MODEL_FILES = (
    Pix2TextModelFile(
        "config.json",
        1_573,
        "fe4076f08f6ca75940f6af9268d51928b834979a69bc2145dacee62633a5d53d",
    ),
    Pix2TextModelFile(
        "decoder_model.onnx",
        32_026_253,
        "917deb98e91a0453c5f234f58a0f32f9fb037de8527c7eb4ed394daf9e692f2a",
    ),
    Pix2TextModelFile(
        "encoder_model.onnx",
        87_510_770,
        "080a3f660f08bc9ebcacdd96e34be6b6400f8c7e62d7cd0dd8251badc37f610b",
    ),
    Pix2TextModelFile(
        "generation_config.json",
        211,
        "7363c031c6142d35a276815b0e285cc289dfb9f51d0b7c63de8f3ed65cc8d8ad",
    ),
    Pix2TextModelFile(
        "preprocessor_config.json",
        450,
        "36a945a7cc645688b9ef64dabae16979cf5f7c1c448569cc306694edc0598b9b",
    ),
    Pix2TextModelFile(
        "special_tokens_map.json",
        964,
        "8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835",
    ),
    Pix2TextModelFile(
        "tokenizer.json",
        113_168,
        "4ffbeb2143e6a38324bb6111b7a8109530d38a076a8439aa5777535f0a32758a",
    ),
    Pix2TextModelFile(
        "tokenizer_config.json",
        1_244,
        "f5cc321e8545940295fba11e9a59f4f2a208a23af3d63cc8c355c78593645b99",
    ),
)

# Keep this as the catalog-facing download size.  It is the sum of the
# verified files rather than a synthetic ZIP size.
PIX2TEXT_TOTAL_SIZE = sum(item.size for item in PIX2TEXT_MODEL_FILES)
Pix2TextProgressCallback = Callable[[str, int, int], None]


class Pix2TextModelDownloadError(RuntimeError):
    pass


def pix2text_model_dir() -> Path:
    return external_model_dir(PIX2TEXT_MODEL_ID)


def is_pix2text_model_cached(*, verify_hash: bool = False) -> bool:
    roots = [pix2text_model_dir()]
    bundled = bundled_external_model_dir(PIX2TEXT_MODEL_ID)
    if bundled is not None:
        roots.append(bundled)
    return any(_model_files_are_valid(root, verify_hash=verify_hash) for root in roots)


def ensure_pix2text_model(
    *, progress_callback: DownloadProgressCallback | None = None,
) -> Path:
    root = pix2text_model_dir()
    if _model_files_are_valid(root, verify_hash=True):
        _notify(progress_callback, PIX2TEXT_TOTAL_SIZE, PIX2TEXT_TOTAL_SIZE)
        return root
    bundled = bundled_external_model_dir(PIX2TEXT_MODEL_ID)
    if bundled is not None and _model_files_are_valid(bundled, verify_hash=True):
        _notify(progress_callback, PIX2TEXT_TOTAL_SIZE, PIX2TEXT_TOTAL_SIZE)
        return bundled

    ensure_safe_directory(root.parent)
    lock = InterProcessFileLock(
        root.with_suffix(".lock"),
        on_wait=lambda: _notify(progress_callback, 0, PIX2TEXT_TOTAL_SIZE),
    )

    with lock:
        completed = 0
        if _model_files_are_valid(pix2text_model_dir(), verify_hash=True):
            _notify(progress_callback, PIX2TEXT_TOTAL_SIZE, PIX2TEXT_TOTAL_SIZE)
            return pix2text_model_dir()
        bundled = bundled_external_model_dir(PIX2TEXT_MODEL_ID)
        if bundled is not None and _model_files_are_valid(bundled, verify_hash=True):
            _notify(progress_callback, PIX2TEXT_TOTAL_SIZE, PIX2TEXT_TOTAL_SIZE)
            return bundled
        ensure_safe_directory(root)
        downloads_root = root.parent / ".downloads"
        ensure_safe_directory(downloads_root)
        downloads_dir = downloads_root / PIX2TEXT_MODEL_ID
        ensure_safe_directory(downloads_dir)
        for item in PIX2TEXT_MODEL_FILES:
            destination = root / item.name
            if _file_is_valid(destination, item, verify_hash=True):
                completed += item.size
                _notify(progress_callback, completed, PIX2TEXT_TOTAL_SIZE)
                continue
            _download_file(
                item,
                destination,
                downloads_dir=downloads_dir,
                completed=completed,
                total=PIX2TEXT_TOTAL_SIZE,
                callback=progress_callback,
            )
            completed += item.size

    if not is_pix2text_model_cached(verify_hash=True):
        raise Pix2TextModelDownloadError("Pix2Text MFR 模型下载后校验失败。")
    return root


def _download_file(
    item: Pix2TextModelFile,
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
            error_type=Pix2TextModelDownloadError,
            label="Pix2Text MFR",
        )


def _file_is_valid(
    path: Path,
    item: Pix2TextModelFile,
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
        PIX2TEXT_MODEL_FILES,
        verify_hash=verify_hash,
    )


def _sha256(path: Path) -> str:
    return sha256_file(path, chunk_size=CHUNK_SIZE)


def _notify(
    callback: Pix2TextProgressCallback | None,
    downloaded: int,
    total: int,
) -> None:
    if callback is not None:
        callback(
            PIX2TEXT_MODEL_ID,
            min(max(downloaded, 0), total),
            total,
        )
