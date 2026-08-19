from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests
from filelock import FileLock

try:
    from formula_ocr_app.model_downloader import DownloadProgressCallback
    from formula_ocr_app.runtime_paths import (
        bundled_external_model_dir,
        external_model_dir,
    )
except ImportError:  # Allows `python formula_ocr_app/app.py`.
    from model_downloader import DownloadProgressCallback
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

    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(destination.with_suffix(".lock")))
    with lock:
        if _model_files_are_valid(destination, verify_hash=True):
            _notify(progress_callback, UNIMERNET_ONNX_TOTAL_SIZE, UNIMERNET_ONNX_TOTAL_SIZE)
            return destination
        bundled = bundled_external_model_dir(UNIMERNET_ONNX_MODEL_ID)
        if bundled is not None and _model_files_are_valid(bundled, verify_hash=True):
            _notify(progress_callback, UNIMERNET_ONNX_TOTAL_SIZE, UNIMERNET_ONNX_TOTAL_SIZE)
            return bundled

        destination.mkdir(parents=True, exist_ok=True)
        downloads_dir = destination.parent / ".downloads" / UNIMERNET_ONNX_MODEL_ID
        downloads_dir.mkdir(parents=True, exist_ok=True)
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
    partial = downloads_dir / f"{item.name}.part"
    if _file_is_valid(partial, item, verify_hash=True):
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, destination)
        _notify(callback, completed + item.size, total)
        return
    offset = partial.stat().st_size if partial.is_file() else 0
    if offset >= item.size:
        partial.unlink(missing_ok=True)
        offset = 0

    headers = {"User-Agent": "FormulaOCR/2.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    try:
        response = requests.get(
            item.url,
            stream=True,
            timeout=(20, 180),
            headers=headers,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise UniMERNetONNXModelDownloadError(
            f"UniMERNet Small ONNX 模型下载失败：{item.name}\n{exc}"
        ) from exc

    append = offset > 0 and response.status_code == 206
    if not append:
        offset = 0
    mode = "ab" if append else "wb"
    try:
        _notify(callback, completed + offset, total)
        last_report = 0.0
        with partial.open(mode) as stream:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                stream.write(chunk)
                offset += len(chunk)
                now = time.monotonic()
                if now - last_report >= 0.5 or offset >= item.size:
                    _notify(callback, completed + offset, total)
                    last_report = now
    except (OSError, requests.RequestException) as exc:
        raise UniMERNetONNXModelDownloadError(
            f"UniMERNet Small ONNX 下载中断，进度已保留：{partial}\n{exc}"
        ) from exc
    finally:
        response.close()

    if offset != item.size or _sha256(partial) != item.sha256:
        partial.unlink(missing_ok=True)
        raise UniMERNetONNXModelDownloadError(
            f"UniMERNet Small ONNX 文件校验失败：{item.name}"
        )
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, destination)
    except OSError as exc:
        raise UniMERNetONNXModelDownloadError(
            f"无法保存 UniMERNet Small ONNX 文件：{destination}\n{exc}"
        ) from exc


def _file_is_valid(
    path: Path,
    item: UniMERNetONNXModelFile,
    *,
    verify_hash: bool,
) -> bool:
    try:
        if path.is_symlink() or not path.is_file():
            return False
        if path.stat().st_size != item.size:
            return False
        return not verify_hash or _sha256(path) == item.sha256
    except OSError:
        return False


def _model_files_are_valid(root: Path, *, verify_hash: bool) -> bool:
    return all(
        _file_is_valid(root / item.name, item, verify_hash=verify_hash)
        for item in UNIMERNET_ONNX_MODEL_FILES
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


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
