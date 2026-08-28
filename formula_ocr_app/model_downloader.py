from __future__ import annotations

import os
import shutil
import tarfile
import time
import zlib
from pathlib import Path

try:
    from formula_ocr_app.download_utils import (
        archive_member_name_is_safe,
        archive_payload_is_within_limits,
        ensure_safe_directory,
        recover_model_directory_backup,
        replace_model_directory,
    )
    from formula_ocr_app.interprocess_lock import InterProcessFileLock
    from formula_ocr_app.model_api import (
        DownloadProgressCallback,
        ModelDownloadError,
    )
    from formula_ocr_app.model_catalog import FormulaModelSpec, get_model_spec
    from formula_ocr_app.runtime_paths import (
        is_paddle_model_cached,
        paddle_model_dir,
        resolve_paddle_model_dir,
    )
except ModuleNotFoundError as exc:  # Allows `python formula_ocr_app/app.py`.
    if exc.name != "formula_ocr_app":
        raise
    from download_utils import (
        archive_member_name_is_safe,
        archive_payload_is_within_limits,
        ensure_safe_directory,
        recover_model_directory_backup,
        replace_model_directory,
    )
    from interprocess_lock import InterProcessFileLock
    from model_api import (
        DownloadProgressCallback,
        ModelDownloadError,
    )
    from model_catalog import FormulaModelSpec, get_model_spec
    from runtime_paths import (
        is_paddle_model_cached,
        paddle_model_dir,
        resolve_paddle_model_dir,
    )


DOWNLOAD_CHUNK_SIZE = 1024 * 1024


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
    try:
        ensure_safe_directory(models_root)
        ensure_safe_directory(downloads_dir)
    except OSError as exc:
        raise ModelDownloadError(f"模型缓存目录不安全：{exc}") from exc
    lock_path = downloads_dir / f"{model_name}.lock"

    with InterProcessFileLock(
        lock_path,
        on_wait=lambda: _notify(progress_callback, spec, 0),
    ):
        recover_model_directory_backup(
            destination,
            is_model_valid=_model_files_are_complete,
            error_type=ModelDownloadError,
        )
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
    import requests

    if _archive_is_valid(archive_path, spec):
        _notify(progress_callback, spec, spec.archive_size)
        return
    if archive_path.is_symlink():
        raise ModelDownloadError(f"拒绝覆盖链接模型压缩包：{archive_path}")
    if archive_path.exists():
        archive_path.unlink()

    partial_path = archive_path.with_suffix(archive_path.suffix + ".part")
    downloaded, checksum = _partial_state(partial_path, spec)
    headers = {"User-Agent": "FormulaOCR/1.0"}
    if downloaded:
        headers["Range"] = f"bytes={downloaded}-"

    # Cancellation and the recognition-time no-download guard must run before
    # requests opens a socket.
    _notify(progress_callback, spec, downloaded)
    response = None
    try:
        response = requests.get(
            spec.download_url,
            stream=True,
            timeout=(15, 90),
            headers=headers,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        _safe_close_response(response)
        raise ModelDownloadError(f"模型下载失败：{spec.download_url}\n{exc}") from exc

    append = downloaded > 0 and response.status_code == 206
    if append:
        response_headers = getattr(response, "headers", None)
        content_range = (
            response_headers.get("Content-Range", "")
            if response_headers is not None
            else ""
        )
        if not _content_range_starts_at(content_range, downloaded):
            _safe_close_response(response)
            partial_path.unlink(missing_ok=True)
            # Proxies and mirrors occasionally return a malformed 206 to a
            # valid Range request. Discard that response and retry from byte
            # zero once; the removed partial means recursion cannot repeat.
            return _download_archive(
                spec,
                archive_path,
                progress_callback,
            )
    if not append:
        downloaded = 0
        checksum = 0
    mode = "ab" if append else "wb"
    oversized = False
    try:
        last_report = 0.0
        _notify(progress_callback, spec, downloaded)
        with partial_path.open(mode) as file:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if not chunk:
                    continue
                if downloaded + len(chunk) > spec.archive_size:
                    oversized = True
                    break
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
        _safe_close_response(response)

    if oversized:
        partial_path.unlink(missing_ok=True)
        raise ModelDownloadError("模型服务器返回的数据超过压缩包声明大小")

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
    if extraction_dir.exists() or extraction_dir.is_symlink():
        if extraction_dir.is_symlink() or not extraction_dir.is_dir():
            raise ModelDownloadError(
                f"模型解压缓存不是安全目录：{extraction_dir}"
            )
        shutil.rmtree(extraction_dir)
    ensure_safe_directory(extraction_dir)

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
        replace_model_directory(
            extracted_model,
            destination,
            is_model_valid=_model_files_are_complete,
            error_type=ModelDownloadError,
        )
    except (OSError, tarfile.TarError) as exc:
        raise ModelDownloadError(f"模型解压失败：{archive_path}\n{exc}") from exc
    finally:
        shutil.rmtree(extraction_dir, ignore_errors=True)

    try:
        archive_path.unlink(missing_ok=True)
    except OSError:
        # A locked archive is harmless after a complete model was installed.
        # It can be reused or removed by a later download attempt.
        pass


def _archive_is_valid(path: Path, spec: FormulaModelSpec) -> bool:
    try:
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != spec.archive_size
        ):
            return False
        return _crc32_file(path) == spec.archive_crc32
    except OSError:
        return False


def _partial_state(path: Path, spec: FormulaModelSpec) -> tuple[int, int]:
    if path.is_symlink():
        raise ModelDownloadError(f"拒绝写入链接断点文件：{path}")
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
    if not archive_payload_is_within_limits(
        len(members),
        sum(max(0, member.size) for member in members),
    ):
        raise ModelDownloadError("模型压缩包解压规模超过安全上限")
    for member in members:
        # Tar names are POSIX-like, but a crafted archive can contain
        # backslashes.  Normalize them before checking so the same archive
        # cannot be safe on Linux and escape its destination on Windows.
        if not archive_member_name_is_safe(member.name):
            raise ModelDownloadError(f"模型压缩包包含不安全路径：{member.name}")
        if not (member.isdir() or member.isfile()):
            raise ModelDownloadError(f"模型压缩包包含不支持的条目：{member.name}")


def _model_files_are_complete(model_dir: Path) -> bool:
    try:
        if model_dir.is_symlink() or not model_dir.is_dir():
            return False
        for name in ("inference.json", "inference.yml", "inference.pdiparams"):
            path = model_dir / name
            if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
                return False
        return True
    except OSError:
        return False


def _notify(
    callback: DownloadProgressCallback | None,
    spec: FormulaModelSpec,
    downloaded: int,
) -> None:
    if callback is not None:
        callback(spec.model_id, min(downloaded, spec.archive_size), spec.archive_size)


def _content_range_starts_at(value: str, offset: int) -> bool:
    prefix = "bytes "
    if not value.lower().startswith(prefix):
        return False
    start_text = value[len(prefix) :].split("-", 1)[0].strip()
    return start_text.isdigit() and int(start_text) == offset


def _safe_close_response(response: object | None) -> None:
    if response is None:
        return
    try:
        response.close()  # type: ignore[attr-defined]
    except Exception:
        pass
