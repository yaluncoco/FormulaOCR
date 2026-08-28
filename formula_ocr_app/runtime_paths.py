from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


APP_DATA_DIR_NAME = "FormulaOCR"
DATA_DIR_ENV = "FORMULA_OCR_DATA_DIR"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def runtime_cache_dir() -> Path:
    override = _data_dir_override()
    if override is not None:
        return override / "cache"
    if is_frozen():
        return user_data_dir() / "cache"
    return Path(__file__).resolve().parent / ".cache"


def runtime_log_dir() -> Path:
    override = _data_dir_override()
    if override is not None:
        return override / "logs"
    if is_frozen():
        return user_data_dir() / "logs"
    return Path(__file__).resolve().parent.parent / "logs"


def paddle_runtime_cache_dir() -> Path:
    return runtime_cache_dir() / "runtime"


def paddle_model_dir(model_name: str) -> Path:
    return paddle_runtime_cache_dir() / "paddlex" / "official_models" / model_name


def bundled_paddle_model_dir(model_name: str) -> Path | None:
    """Find an optional read-only model shipped with a packaged build."""

    roots: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(Path(meipass))
    if is_frozen():
        executable_root = Path(sys.executable).resolve().parent
        roots.extend((executable_root / "_internal", executable_root))
    for root in roots:
        candidate = root / "models" / "paddle" / model_name
        if candidate.is_dir():
            return candidate
    return None


def resolve_paddle_model_dir(model_name: str) -> Path:
    """Return the complete user cache or optional bundled model directory."""

    user_dir = paddle_model_dir(model_name)
    if _paddle_model_files_exist(user_dir):
        return user_dir
    bundled_dir = bundled_paddle_model_dir(model_name)
    if bundled_dir is not None and _paddle_model_files_exist(bundled_dir):
        return bundled_dir
    return user_dir


def external_model_dir(model_name: str) -> Path:
    return runtime_cache_dir() / "models" / model_name


def bundled_external_model_dir(model_name: str) -> Path | None:
    """Find an optional read-only ONNX model shipped inside a build."""

    roots: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(Path(meipass))
    if is_frozen():
        executable_root = Path(sys.executable).resolve().parent
        roots.extend((executable_root / "_internal", executable_root))
    for root in roots:
        candidate = root / "models" / "onnx" / model_name
        if candidate.is_dir():
            return candidate
    return None


def is_external_model_bundled(model_name: str) -> bool:
    return bundled_external_model_dir(model_name) is not None


def is_paddle_model_cached(model_name: str) -> bool:
    return _paddle_model_files_exist(resolve_paddle_model_dir(model_name))


def is_paddle_model_bundled(model_name: str) -> bool:
    bundled_dir = bundled_paddle_model_dir(model_name)
    return bundled_dir is not None and _paddle_model_files_exist(bundled_dir)


def paddle_model_has_data(model_name: str) -> bool:
    """Return whether a Paddle model or its scoped resume artifacts exist."""

    if not model_name or Path(model_name).name != model_name:
        return False
    return any(
        path.exists() or path.is_symlink()
        for path in _paddle_user_artifact_paths(model_name)
    )


def paddle_model_cache_size(model_name: str) -> int:
    user_size = sum(
        _cache_artifact_size(path)
        for path in _paddle_user_artifact_paths(model_name)
    )
    if user_size:
        return user_size
    return directory_size(resolve_paddle_model_dir(model_name))


def remove_paddle_model(model_name: str) -> bool:
    """Remove one model and its scoped resume artifacts."""

    if not model_name or Path(model_name).name != model_name:
        raise ValueError(f"拒绝删除非法模型缓存名称：{model_name}")
    model_dir = paddle_model_dir(model_name)
    models_root = model_dir.parent.resolve()
    resolved = model_dir.resolve()
    if resolved.parent != models_root or resolved.name != model_name:
        raise ValueError(f"拒绝删除非模型缓存路径：{resolved}")

    removed = False
    if model_dir.exists() or model_dir.is_symlink():
        if model_dir.is_symlink():
            raise ValueError(f"拒绝删除链接模型缓存路径：{model_dir}")
        if not model_dir.is_dir():
            raise ValueError(f"模型缓存不是目录：{model_dir}")
        shutil.rmtree(model_dir)
        removed = True

    backup = model_dir.with_name(model_dir.name + ".bak")
    if backup.exists() or backup.is_symlink():
        if backup.is_symlink() or not backup.is_dir():
            raise ValueError(f"模型备份缓存不是安全目录：{backup}")
        shutil.rmtree(backup)
        removed = True

    downloads_root = model_dir.parent / ".downloads"
    downloads_resolved = downloads_root.resolve()
    artifact_names = (
        f"{model_name}_infer.tar",
        f"{model_name}_infer.tar.part",
        f".{model_name}.extracting",
    )
    # FormulaOCR 1.0 stored official archives beside the model directories.
    # Current builds use `.downloads`; clean both layouts so upgrades do not
    # leave hundreds of megabytes of invisible archives behind.
    for artifact_root in (downloads_root, model_dir.parent):
        expected_parent = artifact_root.resolve()
        for filename in artifact_names:
            artifact = artifact_root / filename
            if not artifact.exists() and not artifact.is_symlink():
                continue
            if artifact.is_symlink() or artifact.resolve().parent != expected_parent:
                raise ValueError(f"拒绝删除非模型下载文件：{artifact}")
            if artifact.is_dir():
                shutil.rmtree(artifact)
            else:
                artifact.unlink()
            removed = True

    partial_dir = downloads_root / model_name
    if partial_dir.exists() or partial_dir.is_symlink():
        if partial_dir.is_symlink() or partial_dir.resolve().parent != downloads_resolved:
            raise ValueError(f"拒绝删除非模型下载目录：{partial_dir}")
        if not partial_dir.is_dir():
            raise ValueError(f"模型下载目录不是目录：{partial_dir}")
        shutil.rmtree(partial_dir)
        removed = True
    return removed


def directory_size(directory: Path) -> int:
    try:
        if directory.is_symlink() or not directory.is_dir():
            return 0
        total = 0
        for path in directory.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            total += path.stat().st_size
        return total
    except OSError:
        return 0


def _cache_artifact_size(path: Path) -> int:
    try:
        if path.is_symlink():
            return 0
        if path.is_file():
            return path.stat().st_size
        if path.is_dir():
            return directory_size(path)
    except OSError:
        pass
    return 0


def _paddle_user_artifact_paths(model_name: str) -> tuple[Path, ...]:
    model_dir = paddle_model_dir(model_name)
    models_root = model_dir.parent
    downloads_root = models_root / ".downloads"
    names = (
        f"{model_name}_infer.tar",
        f"{model_name}_infer.tar.part",
        f".{model_name}.extracting",
    )
    return tuple(
        dict.fromkeys(
            (
                model_dir,
                model_dir.with_name(model_dir.name + ".bak"),
                *(downloads_root / name for name in names),
                downloads_root / model_name,
                *(models_root / name for name in names),
            )
        )
    )


def external_model_has_data(model_name: str) -> bool:
    """Return whether a model or its scoped resumable download exists."""

    if not model_name or Path(model_name).name != model_name:
        return False
    return any(
        path.exists() or path.is_symlink()
        for path in _external_user_artifact_paths(model_name)
    )


def external_model_cache_size(model_name: str) -> int:
    user_size = sum(
        _cache_artifact_size(path)
        for path in _external_user_artifact_paths(model_name)
    )
    if user_size:
        return user_size
    bundled = bundled_external_model_dir(model_name)
    return directory_size(bundled) if bundled is not None else 0


def remove_external_model(model_name: str) -> bool:
    if not model_name or Path(model_name).name != model_name:
        raise ValueError(f"拒绝删除非法模型缓存名称：{model_name}")

    model_dir = external_model_dir(model_name)
    models_root = model_dir.parent.resolve()
    resolved = model_dir.resolve()
    if resolved.parent != models_root or resolved.name != model_name:
        raise ValueError(f"拒绝删除非模型缓存路径：{resolved}")

    removed = False
    if model_dir.exists() or model_dir.is_symlink():
        if model_dir.is_symlink():
            raise ValueError(f"拒绝删除链接模型缓存路径：{model_dir}")
        shutil.rmtree(resolved)
        removed = True

    backup = model_dir.with_name(model_dir.name + ".bak")
    if backup.exists() or backup.is_symlink():
        if backup.is_symlink() or not backup.is_dir():
            raise ValueError(f"模型备份缓存不是安全目录：{backup}")
        shutil.rmtree(backup)
        removed = True

    # Multi-file ONNX downloads keep resumable fragments outside the model
    # directory.  Remove only this model's own fragment directory, never the
    # shared `.downloads` root.
    downloads_root = model_dir.parent / ".downloads"
    partial_dir = downloads_root / model_name
    if partial_dir.exists() or partial_dir.is_symlink():
        downloads_resolved = downloads_root.resolve()
        partial_resolved = partial_dir.resolve()
        if partial_dir.is_symlink():
            raise ValueError(f"拒绝删除链接下载目录：{partial_dir}")
        if partial_resolved.parent != downloads_resolved or partial_resolved.name != model_name:
            raise ValueError(f"拒绝删除非模型下载目录：{partial_resolved}")
        shutil.rmtree(partial_resolved)
        removed = True

    # ZIP downloaders keep their archive at the shared root for backwards
    # compatibility.  Clean only the selected model's two known names.
    if model_name in {"MathCraftFormula", "MixTexZhEn"}:
        filenames = (
            ("mathcraft-formula-rec.zip", "mathcraft-formula-rec.zip.part")
            if model_name == "MathCraftFormula"
            else ("MixTeX.zip", "MixTeX.zip.part")
        )
        for filename in filenames:
            archive_path = downloads_root / filename
            if not archive_path.exists() and not archive_path.is_symlink():
                continue
            if archive_path.is_symlink() or archive_path.resolve().parent != downloads_root.resolve():
                raise ValueError(f"拒绝删除非模型下载文件：{archive_path}")
            archive_path.unlink()
            removed = True
    if model_name == "MathCraftFormula" and downloads_root.is_dir():
        downloads_resolved = downloads_root.resolve()
        for extraction_dir in downloads_root.glob(".MathCraftFormula-*"):
            if (
                extraction_dir.is_symlink()
                or extraction_dir.resolve().parent != downloads_resolved
                or not extraction_dir.is_dir()
            ):
                raise ValueError(f"拒绝删除非模型解压目录：{extraction_dir}")
            shutil.rmtree(extraction_dir)
            removed = True
    return removed


def _external_user_artifact_paths(model_name: str) -> tuple[Path, ...]:
    model_dir = external_model_dir(model_name)
    downloads_root = model_dir.parent / ".downloads"
    paths: list[Path] = [
        model_dir,
        model_dir.with_name(model_dir.name + ".bak"),
        downloads_root / model_name,
    ]
    if model_name == "MathCraftFormula":
        paths.extend(
            downloads_root / filename
            for filename in (
                "mathcraft-formula-rec.zip",
                "mathcraft-formula-rec.zip.part",
            )
        )
        try:
            paths.extend(downloads_root.glob(".MathCraftFormula-*"))
        except OSError:
            pass
    elif model_name == "MixTexZhEn":
        # Current MixTeX archives are inside the model-scoped directory above;
        # these root-level names are retained for FormulaOCR 1.0 upgrades.
        paths.extend(
            downloads_root / filename
            for filename in ("MixTeX.zip", "MixTeX.zip.part")
        )
    return tuple(dict.fromkeys(paths))


def user_data_dir() -> Path:
    override = _data_dir_override()
    if override is not None:
        return override

    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    else:
        xdg_data_home = os.environ.get("XDG_DATA_HOME", "").strip()
        base = Path(xdg_data_home) if xdg_data_home else Path.home() / ".local" / "share"
    return base / APP_DATA_DIR_NAME


def _data_dir_override() -> Path | None:
    value = os.environ.get(DATA_DIR_ENV, "").strip()
    if not value:
        return None
    override = Path(value).expanduser().resolve()
    if is_frozen() and any(
        _path_is_within(override, root) for root in _packaged_runtime_roots()
    ):
        # A frozen build must never use its install directory or PyInstaller's
        # extraction directory for mutable model/cache data.  In particular,
        # do not let an accidental FORMULA_OCR_DATA_DIR override turn
        # _internal into a download target.
        return None
    return override


def _packaged_runtime_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(Path(meipass).resolve())
    if is_frozen():
        executable_root = Path(sys.executable).resolve().parent
        roots.extend(
            (
                (executable_root / "_internal").resolve(),
                executable_root,
            )
        )
    return tuple(dict.fromkeys(roots))


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _paddle_model_files_exist(model_dir: Path) -> bool:
    try:
        if model_dir.is_symlink() or not model_dir.is_dir():
            return False
        for filename in ("inference.json", "inference.yml", "inference.pdiparams"):
            path = model_dir / filename
            if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
                return False
        return True
    except OSError:
        return False
