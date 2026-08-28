"""Shared verified, resumable HTTP download primitives."""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, NoReturn, Protocol

DOWNLOAD_CHUNK_SIZE = 1024 * 1024
MAX_ARCHIVE_MEMBERS = 10_000
MAX_EXTRACTED_MODEL_BYTES = 8 * 1024 * 1024 * 1024
_HASH_CACHE_MAX_ENTRIES = 128
_HASH_CACHE: OrderedDict[tuple[str, int, int, int, int, int], str] = OrderedDict()
_HASH_CACHE_LOCK = threading.Lock()
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


class DownloadFileSpec(Protocol):
    name: str
    size: int
    sha256: str
    url: str


@dataclass(frozen=True)
class RemoteFileSpec:
    name: str
    size: int
    sha256: str
    url: str


class VerifiedDownloadFailure(RuntimeError):
    def __init__(
        self,
        phase: str,
        *,
        item_name: str,
        path: Path,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(phase, item_name, str(path))
        self.phase = phase
        self.item_name = item_name
        self.path = path
        self.cause = cause


def download_verified_file(
    item: DownloadFileSpec,
    destination: Path,
    *,
    partial: Path,
    completed: int,
    total: int,
    notify: Callable[[int, int], None],
    request_get: Callable[..., Any],
    request_exception: type[BaseException] | tuple[type[BaseException], ...],
    timeout: tuple[int, int] = (20, 180),
    chunk_size: int = DOWNLOAD_CHUNK_SIZE,
) -> None:
    """Resume one file, verify it, then atomically install it.

    Progress callback exceptions intentionally propagate unchanged. This is
    how the GUI cancels an active transfer while preserving its ``.part``
    file for a later Range request.
    """

    try:
        ensure_safe_directory(destination.parent)
        ensure_safe_directory(partial.parent)
    except OSError as exc:
        raise VerifiedDownloadFailure(
            "install",
            item_name=item.name,
            path=destination,
            cause=exc,
        ) from exc
    if destination.is_symlink() or partial.is_symlink():
        unsafe_path = destination if destination.is_symlink() else partial
        raise VerifiedDownloadFailure(
            "install",
            item_name=item.name,
            path=unsafe_path,
            cause=OSError(f"拒绝写入链接文件：{unsafe_path}"),
        )

    if file_is_valid(partial, item.size, item.sha256, verify_hash=True):
        _replace(partial, destination, item.name)
        notify(completed + item.size, total)
        return

    try:
        offset = partial.stat().st_size if partial.is_file() else 0
    except OSError:
        offset = 0
    if offset >= item.size:
        _remove_file(partial, item.name)
        offset = 0

    headers = {"User-Agent": "FormulaOCR/2.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    # Give cancellation and the no-implicit-download guard a chance to stop
    # the operation before DNS resolution or an HTTP connection is attempted.
    notify(completed + offset, total)
    response = None
    try:
        response = request_get(
            item.url,
            stream=True,
            timeout=timeout,
            headers=headers,
        )
        response.raise_for_status()
    except Exception as exc:
        _safe_close_response(response)
        if isinstance(exc, request_exception):
            raise VerifiedDownloadFailure(
                "request",
                item_name=item.name,
                path=destination,
                cause=exc,
            ) from exc
        raise

    append = offset > 0 and response.status_code == 206
    if append:
        headers = getattr(response, "headers", None)
        content_range = headers.get("Content-Range", "") if headers is not None else ""
        if not _content_range_starts_at(content_range, offset):
            _safe_close_response(response)
            _remove_file(partial, item.name)
            return download_verified_file(
                item,
                destination,
                partial=partial,
                completed=completed,
                total=total,
                notify=notify,
                request_get=request_get,
                request_exception=request_exception,
                timeout=timeout,
                chunk_size=chunk_size,
            )
    if not append:
        offset = 0
    mode = "ab" if append else "wb"
    oversized = False
    try:
        notify(completed + offset, total)
        last_report = 0.0
        with partial.open(mode) as stream:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                if offset + len(chunk) > item.size:
                    oversized = True
                    break
                stream.write(chunk)
                offset += len(chunk)
                now = time.monotonic()
                if now - last_report >= 0.5 or offset >= item.size:
                    notify(completed + offset, total)
                    last_report = now
    except (OSError, request_exception) as exc:
        raise VerifiedDownloadFailure(
            "transfer",
            item_name=item.name,
            path=partial,
            cause=exc,
        ) from exc
    finally:
        _safe_close_response(response)

    if oversized:
        _remove_file(partial, item.name)
        raise VerifiedDownloadFailure(
            "transfer",
            item_name=item.name,
            path=partial,
            cause=RuntimeError("服务器返回的数据超过模型文件声明大小"),
        )

    if offset != item.size:
        raise VerifiedDownloadFailure(
            "transfer",
            item_name=item.name,
            path=partial,
            cause=RuntimeError(
                f"服务器提前结束传输：{offset} / {item.size} 字节"
            ),
        )
    try:
        digest = sha256_file(partial)
    except OSError as exc:
        raise VerifiedDownloadFailure(
            "transfer",
            item_name=item.name,
            path=partial,
            cause=exc,
        ) from exc
    if digest != item.sha256:
        _remove_file(partial, item.name)
        raise VerifiedDownloadFailure(
            "checksum",
            item_name=item.name,
            path=partial,
        )
    _replace(partial, destination, item.name)


def file_is_valid(
    path: Path,
    expected_size: int,
    expected_sha256: str,
    *,
    verify_hash: bool,
) -> bool:
    try:
        if path.is_symlink() or not path.is_file():
            return False
        if path.stat().st_size != expected_size:
            return False
        return not verify_hash or _cached_sha256_file(path) == expected_sha256
    except OSError:
        return False


def model_files_are_valid(
    root: Path,
    items: Iterable[DownloadFileSpec],
    *,
    verify_hash: bool,
) -> bool:
    try:
        if root.is_symlink() or not root.is_dir():
            return False
    except OSError:
        return False
    return all(
        file_is_valid(
            root / item.name,
            item.size,
            item.sha256,
            verify_hash=verify_hash,
        )
        for item in items
    )


def sha256_file(path: Path, *, chunk_size: int = DOWNLOAD_CHUNK_SIZE) -> str:
    with path.open("rb") as stream:
        return _sha256_stream(stream, chunk_size=chunk_size)


def _sha256_stream(stream: Any, *, chunk_size: int = DOWNLOAD_CHUNK_SIZE) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(chunk_size):
        digest.update(chunk)
    return digest.hexdigest()


def _file_signature(stat_result: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(stat_result.st_dev),
        int(stat_result.st_ino),
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def _cached_sha256_file(path: Path) -> str:
    """Hash an unchanged model file once per process.

    Main-window and model-manager refreshes both validate installed ONNX
    models. Caching by path, size and timestamps keeps strong SHA-256 checks
    without repeatedly reading hundreds of megabytes on every UI update.
    """

    for _attempt in range(3):
        normalized_path = os.path.normcase(str(path.resolve()))
        with path.open("rb") as stream:
            before_signature = _file_signature(os.fstat(stream.fileno()))
            key = (normalized_path, *before_signature)
            with _HASH_CACHE_LOCK:
                cached = _HASH_CACHE.get(key)

            if cached is None:
                digest = _sha256_stream(stream)
            else:
                digest = cached

            after_signature = _file_signature(os.fstat(stream.fileno()))
            try:
                path_signature = _file_signature(path.stat())
                path_is_symlink = path.is_symlink()
            except OSError:
                continue
            if (
                path_is_symlink
                or after_signature != before_signature
                or path_signature != before_signature
            ):
                continue

        with _HASH_CACHE_LOCK:
            if cached is not None:
                _HASH_CACHE.move_to_end(key)
                return cached
            stale_keys = [
                entry for entry in _HASH_CACHE if entry[0] == normalized_path
            ]
            for stale_key in stale_keys:
                _HASH_CACHE.pop(stale_key, None)
            _HASH_CACHE[key] = digest
            while len(_HASH_CACHE) > _HASH_CACHE_MAX_ENTRIES:
                _HASH_CACHE.popitem(last=False)
        return digest
    raise OSError(f"校验期间模型文件发生变化：{path}")


def ensure_safe_directory(path: Path) -> None:
    """Create one mutable cache directory without following its final link."""

    if path.is_symlink():
        raise OSError(f"拒绝使用链接缓存目录：{path}")
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"缓存路径不是目录：{path}")
        return
    path.mkdir(parents=True, exist_ok=True)
    # Recheck after creation to narrow the race between the existence check and
    # mkdir on shared/multi-process caches.
    if path.is_symlink() or not path.is_dir():
        raise OSError(f"无法创建安全缓存目录：{path}")


def archive_member_name_is_safe(name: str) -> bool:
    """Apply POSIX traversal and Windows filename rules on every platform."""

    normalized = name.replace("\\", "/")
    if not normalized or "\x00" in normalized:
        return False
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        return False
    for part in path.parts:
        if not part or part == ".":
            continue
        if any(ord(character) < 32 for character in part):
            return False
        # Colons can select NTFS alternate data streams even when they are not
        # a drive prefix. Trailing spaces/dots and device names alias other
        # paths on Windows and are therefore unsafe on all extraction hosts.
        if ":" in part or part.rstrip(" .") != part:
            return False
        base_name = part.split(".", 1)[0].upper()
        if base_name in _WINDOWS_RESERVED_NAMES:
            return False
    return True


def archive_payload_is_within_limits(
    member_count: int,
    uncompressed_bytes: int,
) -> bool:
    return (
        0 <= member_count <= MAX_ARCHIVE_MEMBERS
        and 0 <= uncompressed_bytes <= MAX_EXTRACTED_MODEL_BYTES
    )


def recover_model_directory_backup(
    destination: Path,
    *,
    is_model_valid: Callable[[Path], bool],
    error_type: type[RuntimeError] = RuntimeError,
) -> bool:
    """Recover an interrupted directory swap or discard its redundant backup."""

    backup = destination.with_name(destination.name + ".bak")
    if not backup.exists() and not backup.is_symlink():
        return False
    if backup.is_symlink() or not backup.is_dir():
        raise error_type(f"模型备份路径不是安全目录：{backup}")

    if is_model_valid(destination):
        shutil.rmtree(backup, ignore_errors=True)
        return False
    if not is_model_valid(backup):
        return False

    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir():
            raise error_type(f"模型安装路径不是安全目录：{destination}")
        shutil.rmtree(destination)
    backup.replace(destination)
    return True


def replace_model_directory(
    source: Path,
    destination: Path,
    *,
    is_model_valid: Callable[[Path], bool],
    error_type: type[RuntimeError] = RuntimeError,
) -> None:
    """Atomically swap a verified model directory with crash-safe rollback."""

    ensure_safe_directory(destination.parent)
    if source.is_symlink() or not source.is_dir():
        raise error_type(f"模型安装源不是安全目录：{source}")
    if not is_model_valid(source):
        raise error_type(f"模型安装源校验失败：{source}")
    if destination.is_symlink():
        raise error_type(f"拒绝覆盖链接模型目录：{destination}")

    recover_model_directory_backup(
        destination,
        is_model_valid=is_model_valid,
        error_type=error_type,
    )
    backup = destination.with_name(destination.name + ".bak")
    if backup.exists() or backup.is_symlink():
        if backup.is_symlink() or not backup.is_dir():
            raise error_type(f"模型备份路径不是安全目录：{backup}")
        shutil.rmtree(backup)
    if destination.exists():
        if not destination.is_dir():
            raise error_type(f"模型安装路径不是目录：{destination}")
        destination.replace(backup)
    try:
        source.replace(destination)
    except OSError:
        if backup.exists() and not destination.exists():
            backup.replace(destination)
        raise
    if backup.exists():
        # Antivirus and indexers can briefly hold old files on Windows. The
        # new verified model is already active, so cleanup remains best effort.
        shutil.rmtree(backup, ignore_errors=True)


def raise_model_download_error(
    failure: VerifiedDownloadFailure,
    *,
    error_type: type[RuntimeError],
    label: str,
) -> NoReturn:
    if failure.phase == "request":
        message = f"{label} 模型下载失败：{failure.item_name}"
    elif failure.phase == "transfer":
        message = f"{label} 下载中断，进度已保留：{failure.path}"
    elif failure.phase == "checksum":
        message = f"{label} 文件校验失败：{failure.item_name}"
    else:
        message = f"无法保存 {label} 文件：{failure.path}"
    if failure.cause is not None:
        message += f"\n{failure.cause}"
        raise error_type(message) from failure.cause
    raise error_type(message) from failure


def _replace(partial: Path, destination: Path, item_name: str) -> None:
    try:
        ensure_safe_directory(destination.parent)
        if destination.is_symlink():
            raise OSError(f"拒绝覆盖链接文件：{destination}")
        os.replace(partial, destination)
    except OSError as exc:
        raise VerifiedDownloadFailure(
            "install",
            item_name=item_name,
            path=destination,
            cause=exc,
        ) from exc


def _remove_file(path: Path, item_name: str) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        raise VerifiedDownloadFailure(
            "cleanup",
            item_name=item_name,
            path=path,
            cause=exc,
        ) from exc


def _content_range_starts_at(value: str, offset: int) -> bool:
    prefix = "bytes "
    if not value.lower().startswith(prefix):
        return False
    start_text = value[len(prefix) :].split("-", 1)[0].strip()
    return start_text.isdigit() and int(start_text) == offset


def _safe_close_response(response: Any | None) -> None:
    if response is None:
        return
    try:
        response.close()
    except Exception:
        # Closing an HTTP response is cleanup. It must not replace the actual
        # transfer, checksum, or user-cancellation exception.
        pass
