"""Move older caches into the portable model/app layout without data loss."""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

try:
    from formula_ocr_app.interprocess_lock import InterProcessFileLock
    from formula_ocr_app.runtime_paths import (
        installation_dir, is_frozen, legacy_user_data_dir, model_root_dir,
        runtime_cache_dir, storage_root, user_data_dir,
    )
except ModuleNotFoundError as exc:
    if exc.name != "formula_ocr_app":
        raise
    from interprocess_lock import InterProcessFileLock
    from runtime_paths import (
        installation_dir, is_frozen, legacy_user_data_dir, model_root_dir,
        runtime_cache_dir, storage_root, user_data_dir,
    )


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_path(path: Path) -> None:
    # Refuse junctions as well as symlinks; an old cache must not redirect a
    # migration into an unrelated directory (Windows Python 3.10 included).
    for item in (path, *path.parents):
        if item.is_symlink() or (
            item.exists()
            and getattr(item.lstat(), "st_file_attributes", 0) & 0x400
        ):
            raise OSError(f"拒绝迁移链接路径：{item}")


def _move_file(source: Path, destination: Path) -> None:
    try:
        source.rename(destination)
        return
    except OSError as exc:
        if exc.errno != errno.EXDEV and getattr(exc, "winerror", None) != 17:
            raise
    # Across volumes, publish only a complete verified copy. On interruption
    # the original remains usable and the next launch can retry the move.
    fd, temporary = tempfile.mkstemp(prefix=".migration-", dir=destination.parent)
    os.close(fd)
    staged = Path(temporary)
    try:
        shutil.copy2(source, staged)
        if source.stat().st_size != staged.stat().st_size or _digest(source) != _digest(staged):
            raise OSError(f"迁移校验失败，已保留原文件：{source}")
        staged.replace(destination)
        source.unlink()
    finally:
        staged.unlink(missing_ok=True)


def _merge(source: Path, destination: Path, conflicts: Path) -> None:
    _check_path(source)
    _check_path(destination)
    if not source.exists() or source.resolve() == destination.resolve():
        return
    if source.resolve() in destination.resolve().parents:
        raise OSError(f"迁移目标不能位于原目录内部：{source} -> {destination}")
    if source.is_dir():
        # Check the tree before the fast same-volume directory rename.
        for parent, dirs, files in os.walk(source, followlinks=False):
            for name in dirs + files:
                _check_path(Path(parent) / name)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                source.rename(destination)
                return
            except OSError as exc:
                if exc.errno != errno.EXDEV and getattr(exc, "winerror", None) != 17:
                    raise
        if destination.is_dir() or not destination.exists():
            destination.mkdir(parents=True, exist_ok=True)
            for child in source.iterdir():
                _merge(child, destination / child.name, conflicts / child.name)
            source.rmdir()
            return
    elif destination.is_file():
        if source.stat().st_size == destination.stat().st_size and _digest(source) == _digest(destination):
            source.unlink()
            return
    if destination.exists():
        # Keep the installed copy and preserve a different older copy under
        # app for inspection. A retry never overwrites either version.
        _merge(source, conflicts, conflicts.with_name(conflicts.name + ".older"))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    _move_file(source, destination)


def migrate_legacy_data(*, extra_cache_roots: tuple[Path, ...] = (), force: bool = False) -> None:
    """Idempotent upgrade. Explicit cache roots support moving source caches.

    The installer runs this before removing old dependencies; normal startup
    also runs it once for portable upgrades. An override isolates test runs
    and never consumes the real user's legacy profile.
    """
    root = storage_root()
    data = user_data_dir()
    marker = data / ".storage-v2"
    if marker.is_file() and not extra_cache_roots and not force:
        return
    _check_path(root)
    data.mkdir(parents=True, exist_ok=True)
    with InterProcessFileLock(data / ".migration.lock", timeout=120):
        if marker.is_file() and not extra_cache_roots and not force:
            return
        _check_path(model_root_dir())
        model_root_dir().mkdir(parents=True, exist_ok=True)
        installed = installation_dir()
        isolated = root != installed
        legacy_data = root if isolated else legacy_user_data_dir()
        caches = [legacy_data / "cache", root / "cache",
                  root / "formula_ocr_app" / ".cache", *extra_cache_roots]
        transfers: list[tuple[Path, Path]] = []
        for cache in dict.fromkeys(caches):
            transfers.extend((
                (cache / "cache" / "runtime" / "paddlex" / "official_models", model_root_dir()),
                (cache / "cache" / "models", model_root_dir()),
                (cache / "runtime" / "paddlex" / "official_models", model_root_dir()),
                (cache / "models", model_root_dir()),
                (cache, runtime_cache_dir()),
            ))
        for resources in (root / "_internal", data):
            for backend in ("paddle", "onnx"):
                transfers.append((resources / "models" / backend, model_root_dir()))
        transfers.extend((
            (legacy_data / "settings.json", data / "settings.json"),
            (legacy_data / "logs", data / "logs"),
            (root / "logs", data / "logs"),
        ))
        if is_frozen():
            transfers.append((root / "docs", data / "docs"))
        for source, destination in transfers:
            key = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:12]
            _merge(source, destination, data / "migration-conflicts" / key / source.name)
        # Only remove empty legacy containers, never unrelated profile data.
        for directory in (legacy_data, root / "_internal" / "models", data / "models"):
            if directory != root:
                try:
                    directory.rmdir()
                except OSError:
                    pass
        marker.write_text("2\n", encoding="ascii")
