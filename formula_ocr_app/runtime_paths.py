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
    model_dir = paddle_model_dir(model_name)
    if model_dir.exists() or model_dir.is_symlink():
        return True

    downloads_root = model_dir.parent / ".downloads"
    if any(
        (downloads_root / filename).exists()
        or (downloads_root / filename).is_symlink()
        for filename in (
            f"{model_name}_infer.tar",
            f"{model_name}_infer.tar.part",
            f".{model_name}.extracting",
        )
    ):
        return True
    # Multi-file Paddle models such as LaTeX_OCR_rec keep resumable files in
    # a model-scoped directory, matching the ONNX downloaders.
    partial_dir = downloads_root / model_name
    return partial_dir.exists() or partial_dir.is_symlink()


def paddle_model_cache_size(model_name: str) -> int:
    model_dir = resolve_paddle_model_dir(model_name)
    try:
        return sum(path.stat().st_size for path in model_dir.rglob("*") if path.is_file())
    except OSError:
        return 0


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

    downloads_root = model_dir.parent / ".downloads"
    downloads_resolved = downloads_root.resolve()
    for filename in (
        f"{model_name}_infer.tar",
        f"{model_name}_infer.tar.part",
        f".{model_name}.extracting",
    ):
        artifact = downloads_root / filename
        if not artifact.exists() and not artifact.is_symlink():
            continue
        if artifact.is_symlink() or artifact.resolve().parent != downloads_resolved:
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
        return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    except OSError:
        return 0


def external_model_has_data(model_name: str) -> bool:
    """Return whether a model or its scoped resumable download exists."""

    if not model_name or Path(model_name).name != model_name:
        return False
    model_dir = external_model_dir(model_name)
    if model_dir.exists() or model_dir.is_symlink():
        return True

    downloads_root = model_dir.parent / ".downloads"
    partial_dir = downloads_root / model_name
    if partial_dir.exists() or partial_dir.is_symlink():
        return True
    if model_name in {"MathCraftFormula", "MixTexZhEn"}:
        archive_names = (
            ("mathcraft-formula-rec.zip", "mathcraft-formula-rec.zip.part")
            if model_name == "MathCraftFormula"
            else ("MixTeX.zip", "MixTeX.zip.part")
        )
        return any(
            (downloads_root / filename).exists()
            or (downloads_root / filename).is_symlink()
            for filename in archive_names
        )
    return False


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
    return removed


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
    return all(
        (model_dir / filename).is_file()
        for filename in ("inference.json", "inference.yml", "inference.pdiparams")
    )
