from __future__ import annotations

import os
import shutil
import tarfile
import time
import zlib
from pathlib import Path
from typing import Callable

import requests
from filelock import FileLock

try:
    from formula_ocr_app.model_catalog import FormulaModelSpec, get_model_spec
    from formula_ocr_app.runtime_paths import (
        is_paddle_model_cached,
        paddle_model_dir,
        resolve_paddle_model_dir,
    )
except ImportError:  # Allows `python formula_ocr_app/app.py`.
    from model_catalog import FormulaModelSpec, get_model_spec
    from runtime_paths import (
        is_paddle_model_cached,
        paddle_model_dir,
        resolve_paddle_model_dir,
    )


DOWNLOAD_CHUNK_SIZE = 1024 * 1024


DownloadProgressCallback = Callable[[str, int, int], None]


class ModelDownloadError(RuntimeError):
    pass


class ModelDownloadCancelled(RuntimeError):
    """Internal signal used to stop a download while preserving its partial file."""


def ensure_official_model(
    model_name: str,
    *,
    progress_callback: DownloadProgressCallback | None = None,
) -> Path:
    """Download, verify and install one supported Paddle formula model."""

    if is_paddle_model_cached(model_name):
        return resolve_paddle_model_dir(model_name)

    try:
        spec = get_model_spec(model_name)
    except ValueError as exc:
        raise ModelDownloadError(f"不支持自动下载模型：{model_name}") from exc
    if spec.backend != "paddle" or spec.archive_crc32 is None:
        raise ModelDownloadError(f"该模型不使用 Paddle 归档下载器：{model_name}")

    destination = paddle_model_dir(model_name)
    models_root = destination.parent
    downloads_dir = models_root / ".downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    lock_path = downloads_dir / f"{model_name}.lock"

    with FileLock(str(lock_path)):
        if is_paddle_model_cached(model_name):
            return resolve_paddle_model_dir(model_name)

        archive_path = downloads_dir / spec.archive_name
        _download_archive(spec, archive_path, progress_callback)
        _install_archive(spec, archive_path, destination, downloads_dir)

    if not is_paddle_model_cached(model_name):
        raise ModelDownloadError(f"模型安装后校验失败：{destination}")
    return destination


def _download_archive(
    spec: FormulaModelSpec,
    archive_path: Path,
    progress_callback: DownloadProgressCallback | None,
) -> None:
    if _archive_is_valid(archive_path, spec):
        _notify(progress_callback, spec, spec.archive_size)
        return
    if archive_path.exists():
        archive_path.unlink()

    partial_path = archive_path.with_suffix(archive_path.suffix + ".part")
    downloaded, checksum = _partial_state(partial_path, spec)
    headers = {"User-Agent": "FormulaOCR/1.0"}
    if downloaded:
        headers["Range"] = f"bytes={downloaded}-"

    try:
        response = requests.get(
            spec.download_url,
            stream=True,
            timeout=(15, 90),
            headers=headers,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ModelDownloadError(f"模型下载失败：{spec.download_url}\n{exc}") from exc

    append = downloaded > 0 and response.status_code == 206
    if not append:
        downloaded = 0
        checksum = 0
    mode = "ab" if append else "wb"
    try:
        last_report = 0.0
        _notify(progress_callback, spec, downloaded)
        with partial_path.open(mode) as file:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if not chunk:
                    continue
                file.write(chunk)
                downloaded += len(chunk)
                checksum = zlib.crc32(chunk, checksum)
                now = time.monotonic()
                if now - last_report >= 0.5 or downloaded >= spec.archive_size:
                    _notify(progress_callback, spec, downloaded)
                    last_report = now
    except requests.RequestException as exc:
        raise ModelDownloadError(
            f"模型下载中断，已保留进度供下次续传：{partial_path}\n{exc}"
        ) from exc
    except OSError as exc:
        raise ModelDownloadError(f"无法写入模型缓存：{partial_path}\n{exc}") from exc
    finally:
        response.close()

    checksum &= 0xFFFFFFFF
    if downloaded != spec.archive_size:
        raise ModelDownloadError(
            f"模型下载不完整：{downloaded} / {spec.archive_size} 字节"
        )
    if checksum != spec.archive_crc32:
        partial_path.unlink(missing_ok=True)
        raise ModelDownloadError(
            f"模型文件校验失败：CRC32 {checksum} != {spec.archive_crc32}"
        )
    try:
        os.replace(partial_path, archive_path)
    except OSError as exc:
        raise ModelDownloadError(f"无法保存模型压缩包：{archive_path}\n{exc}") from exc


def _install_archive(
    spec: FormulaModelSpec,
    archive_path: Path,
    destination: Path,
    downloads_dir: Path,
) -> None:
    extraction_dir = downloads_dir / f".{spec.model_id}.extracting"
    if extraction_dir.exists():
        shutil.rmtree(extraction_dir)
    extraction_dir.mkdir(parents=True)

    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            members = archive.getmembers()
            _validate_tar_members(members)
            archive.extractall(extraction_dir, members=members)

        extracted_model = extraction_dir / spec.archive_root
        if not _model_files_are_complete(extracted_model):
            raise ModelDownloadError(
                f"模型压缩包缺少推理文件：{spec.archive_name}"
            )
        # Keep an already working cache until the new archive has been fully
        # extracted and verified.  This matters when a user updates a model
        # while a previous version is still available.
        backup = destination.with_name(destination.name + ".bak")
        if backup.exists():
            shutil.rmtree(backup)
        if destination.exists():
            destination.replace(backup)
        try:
            shutil.move(str(extracted_model), str(destination))
        except OSError:
            if backup.exists() and not destination.exists():
                backup.replace(destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    except (OSError, tarfile.TarError) as exc:
        raise ModelDownloadError(f"模型解压失败：{archive_path}\n{exc}") from exc
    finally:
        shutil.rmtree(extraction_dir, ignore_errors=True)

    archive_path.unlink(missing_ok=True)


def _archive_is_valid(path: Path, spec: FormulaModelSpec) -> bool:
    if not path.is_file() or path.stat().st_size != spec.archive_size:
        return False
    return _crc32_file(path) == spec.archive_crc32


def _partial_state(path: Path, spec: FormulaModelSpec) -> tuple[int, int]:
    if not path.is_file():
        return 0, 0
    size = path.stat().st_size
    if size <= 0 or size >= spec.archive_size:
        path.unlink(missing_ok=True)
        return 0, 0
    return size, _crc32_file(path)


def _crc32_file(path: Path) -> int:
    checksum = 0
    with path.open("rb") as file:
        while chunk := file.read(DOWNLOAD_CHUNK_SIZE):
            checksum = zlib.crc32(chunk, checksum)
    return checksum & 0xFFFFFFFF


def _validate_tar_members(members: list[tarfile.TarInfo]) -> None:
    for member in members:
        # Tar names are POSIX-like, but a crafted archive can contain
        # backslashes.  Normalize them before checking so the same archive
        # cannot be safe on Linux and escape its destination on Windows.
        normalized_name = member.name.replace("\\", "/")
        member_path = Path(normalized_name)
        has_windows_drive = (
            len(normalized_name) >= 2 and normalized_name[1] == ":"
        )
        if (
            not normalized_name
            or "\x00" in normalized_name
            or member_path.is_absolute()
            or has_windows_drive
            or ".." in member_path.parts
        ):
            raise ModelDownloadError(f"模型压缩包包含不安全路径：{member.name}")
        if not (member.isdir() or member.isfile()):
            raise ModelDownloadError(f"模型压缩包包含不支持的条目：{member.name}")


def _model_files_are_complete(model_dir: Path) -> bool:
    return all(
        (model_dir / name).is_file()
        for name in ("inference.json", "inference.yml", "inference.pdiparams")
    )


def _notify(
    callback: DownloadProgressCallback | None,
    spec: FormulaModelSpec,
    downloaded: int,
) -> None:
    if callback is not None:
        callback(spec.model_id, min(downloaded, spec.archive_size), spec.archive_size)
