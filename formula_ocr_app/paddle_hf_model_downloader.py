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
    from formula_ocr_app.model_downloader import ModelDownloadError
    from formula_ocr_app.runtime_paths import (
        bundled_paddle_model_dir,
        paddle_model_dir,
    )
except ImportError:  # Allows `python formula_ocr_app/app.py`.
    from model_downloader import ModelDownloadError
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

    root.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(root.with_suffix(".lock")))
    with lock:
        if _model_files_are_valid(root, verify_hash=True):
            _notify(progress_callback, PADDLE_HF_TOTAL_SIZE, PADDLE_HF_TOTAL_SIZE)
            return root
        bundled = bundled_paddle_model_dir(PADDLE_HF_MODEL_ID)
        if bundled is not None and _model_files_are_valid(bundled, verify_hash=True):
            _notify(progress_callback, PADDLE_HF_TOTAL_SIZE, PADDLE_HF_TOTAL_SIZE)
            return bundled

        root.mkdir(parents=True, exist_ok=True)
        downloads_dir = root.parent / ".downloads" / PADDLE_HF_MODEL_ID
        downloads_dir.mkdir(parents=True, exist_ok=True)
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
        raise PaddleHFModelDownloadError(
            f"LaTeX-OCR 模型下载失败：{item.name}\n{exc}"
        ) from exc

    append = offset > 0 and response.status_code == 206
    if not append:
        offset = 0
    mode = "ab" if append else "wb"
    try:
        _notify(callback, completed + offset, PADDLE_HF_TOTAL_SIZE)
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
                        PADDLE_HF_TOTAL_SIZE,
                    )
                    last_report = now
    except (OSError, requests.RequestException) as exc:
        raise PaddleHFModelDownloadError(
            f"LaTeX-OCR 下载中断，进度已保留：{partial}\n{exc}"
        ) from exc
    finally:
        response.close()

    if offset != item.size or _sha256(partial) != item.sha256:
        partial.unlink(missing_ok=True)
        raise PaddleHFModelDownloadError(
            f"LaTeX-OCR 文件校验失败：{item.name}"
        )
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, destination)
    except OSError as exc:
        raise PaddleHFModelDownloadError(
            f"无法保存 LaTeX-OCR 文件：{destination}\n{exc}"
        ) from exc


def _file_is_valid(
    path: Path,
    item: PaddleHFModelFile,
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
        for item in PADDLE_HF_MODEL_FILES
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


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
