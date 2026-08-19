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

    root.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(root.with_suffix(".lock")))

    with lock:
        completed = 0
        if _model_files_are_valid(pix2text_model_dir(), verify_hash=True):
            _notify(progress_callback, PIX2TEXT_TOTAL_SIZE, PIX2TEXT_TOTAL_SIZE)
            return pix2text_model_dir()
        bundled = bundled_external_model_dir(PIX2TEXT_MODEL_ID)
        if bundled is not None and _model_files_are_valid(bundled, verify_hash=True):
            _notify(progress_callback, PIX2TEXT_TOTAL_SIZE, PIX2TEXT_TOTAL_SIZE)
            return bundled
        root.mkdir(parents=True, exist_ok=True)
        downloads_dir = root.parent / ".downloads" / PIX2TEXT_MODEL_ID
        downloads_dir.mkdir(parents=True, exist_ok=True)
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
    partial = downloads_dir / f"{item.name}.part"
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
        raise Pix2TextModelDownloadError(
            f"Pix2Text MFR 模型下载失败：{item.name}\n{exc}"
        ) from exc

    append = offset > 0 and response.status_code == 206
    if not append:
        offset = 0
    mode = "ab" if append else "wb"
    try:
        _notify(callback, downloaded=completed + offset, total=total)
        last_report = 0.0
        with partial.open(mode) as stream:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                stream.write(chunk)
                offset += len(chunk)
                now = time.monotonic()
                if now - last_report >= 0.5 or offset >= item.size:
                    _notify(
                        callback,
                        completed + offset,
                        total,
                    )
                    last_report = now
    except (OSError, requests.RequestException) as exc:
        raise Pix2TextModelDownloadError(
            f"Pix2Text MFR 下载中断，进度已保留：{partial}\n{exc}"
        ) from exc
    finally:
        response.close()

    if offset != item.size or _sha256(partial) != item.sha256:
        partial.unlink(missing_ok=True)
        raise Pix2TextModelDownloadError(
            f"Pix2Text MFR 文件校验失败：{item.name}"
        )
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, destination)
    except OSError as exc:
        raise Pix2TextModelDownloadError(
            f"无法保存 Pix2Text MFR 文件：{destination}\n{exc}"
        ) from exc


def _file_is_valid(
    path: Path,
    item: Pix2TextModelFile,
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
        for item in PIX2TEXT_MODEL_FILES
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


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
