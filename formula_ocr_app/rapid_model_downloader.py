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
    from formula_ocr_app.runtime_paths import (
        bundled_external_model_dir,
        external_model_dir,
    )
except ImportError:  # Allows `python formula_ocr_app/app.py`.
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

    root.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(root.with_suffix(".lock")))
    with lock:
        if _model_files_are_valid(root, verify_hash=True):
            _notify(progress_callback, total, total)
            return root
        root.mkdir(parents=True, exist_ok=True)
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
    partial = path.with_suffix(path.suffix + ".part")
    offset = partial.stat().st_size if partial.is_file() else 0
    if offset >= spec.size:
        partial.unlink(missing_ok=True)
        offset = 0
    headers = {"User-Agent": "FormulaOCR/2.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    try:
        response = requests.get(
            spec.url,
            stream=True,
            timeout=(20, 120),
            headers=headers,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"RapidLaTeXOCR 模型下载失败：{spec.name}\n{exc}") from exc

    append = offset > 0 and response.status_code == 206
    if not append:
        offset = 0
    mode = "ab" if append else "wb"
    try:
        last_report = 0.0
        _notify(callback, completed + offset, total)
        with partial.open(mode) as file:
            for chunk in response.iter_content(CHUNK_SIZE):
                if not chunk:
                    continue
                file.write(chunk)
                offset += len(chunk)
                now = time.monotonic()
                if now - last_report >= 0.5 or offset >= spec.size:
                    _notify(callback, completed + offset, total)
                    last_report = now
    except (OSError, requests.RequestException) as exc:
        raise RuntimeError(
            f"RapidLaTeXOCR 下载中断，进度已保留：{partial}\n{exc}"
        ) from exc
    finally:
        response.close()
    if offset != spec.size or _sha256(partial) != spec.sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"RapidLaTeXOCR 文件校验失败：{spec.name}")
    os.replace(partial, path)


def _file_is_valid(
    path: Path, spec: RapidModelFile, *, verify_hash: bool
) -> bool:
    try:
        if path.is_symlink() or path.stat().st_size != spec.size:
            return False
        return not verify_hash or _sha256(path) == spec.sha256
    except OSError:
        return False


def _model_files_are_valid(root: Path, *, verify_hash: bool) -> bool:
    return all(
        _file_is_valid(root / spec.name, spec, verify_hash=verify_hash)
        for spec in RAPID_MODEL_FILES
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _notify(
    callback: RapidProgressCallback | None, downloaded: int, total: int
) -> None:
    if callback is not None:
        callback("RapidLaTeXOCR", min(downloaded, total), total)
