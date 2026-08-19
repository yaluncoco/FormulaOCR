from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
import time
import zipfile
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


MATHCRAFT_MODEL_ID = "MathCraftFormula"
MATHCRAFT_ARCHIVE_SIZE = 108_795_631
MATHCRAFT_ARCHIVE_SHA256 = (
    "807dd2d1ac40454424404b31a73d4242c37c76edf176ab544540028da20ec43f"
)
MATHCRAFT_RELEASE_URL = (
    "https://github.com/SakuraMathcraft/MathCraft-Models/releases/download/"
    "v1.0.0/mathcraft-formula-rec.zip"
)
CHUNK_SIZE = 1024 * 1024


# Hashes are from the upstream MathCraft-Models manifest.  Keeping the file
# manifest in the application lets us validate an extracted archive rather
# than trusting a successful HTTP response or a matching file size.
MATHCRAFT_MODEL_FILES = {
    "config.json": "cc60b5e3fa221c49086147e82583b3bf62a62917bddf38decfccf7ccfdb1e53b",
    "encoder_model.onnx": "bd8d5c322792e9ec45793af5569e9748f82a3d728a9e00213dbfc56c1486f37d",
    "decoder_model.onnx": "fd0f92d7a012f3dae41e1ac79421aea0ea888b5a66cb3f9a004e424f82f3daed",
    "generation_config.json": "cbea88288d5576a9655ad04e2456768544be22273a1c5ca160e0d16384639b4f",
    "preprocessor_config.json": "36a945a7cc645688b9ef64dabae16979cf5f7c1c448569cc306694edc0598b9b",
    "special_tokens_map.json": "8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835",
    "tokenizer.json": "3e2ab757277d22639bec28c9d7972e352d3d1dba223051fa674002dc5ab64df3",
    "tokenizer_config.json": "7ffff31747c73b1a462b766abfc128e03f669e5b8452fe6e175b1430a078ac8d",
}

MathCraftProgressCallback = Callable[[str, int, int], None]


class MathCraftModelDownloadError(RuntimeError):
    pass


def mathcraft_model_dir() -> Path:
    return external_model_dir(MATHCRAFT_MODEL_ID)


def is_mathcraft_model_cached(*, verify_hash: bool = False) -> bool:
    roots = [mathcraft_model_dir()]
    bundled = bundled_external_model_dir(MATHCRAFT_MODEL_ID)
    if bundled is not None:
        roots.append(bundled)
    return any(_model_files_are_valid(root, verify_hash=verify_hash) for root in roots)


def ensure_mathcraft_model(
    *, progress_callback: MathCraftProgressCallback | None = None,
) -> Path:
    destination = mathcraft_model_dir()
    if _model_files_are_valid(destination, verify_hash=True):
        _notify(progress_callback, MATHCRAFT_ARCHIVE_SIZE, MATHCRAFT_ARCHIVE_SIZE)
        return destination
    bundled = bundled_external_model_dir(MATHCRAFT_MODEL_ID)
    if bundled is not None and _model_files_are_valid(bundled, verify_hash=True):
        _notify(progress_callback, MATHCRAFT_ARCHIVE_SIZE, MATHCRAFT_ARCHIVE_SIZE)
        return bundled

    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(destination.with_suffix(".lock")))

    with lock:
        if _model_files_are_valid(destination, verify_hash=True):
            _notify(progress_callback, MATHCRAFT_ARCHIVE_SIZE, MATHCRAFT_ARCHIVE_SIZE)
            return destination
        bundled = bundled_external_model_dir(MATHCRAFT_MODEL_ID)
        if bundled is not None and _model_files_are_valid(bundled, verify_hash=True):
            _notify(progress_callback, MATHCRAFT_ARCHIVE_SIZE, MATHCRAFT_ARCHIVE_SIZE)
            return bundled

        downloads_dir = destination.parent / ".downloads"
        downloads_dir.mkdir(parents=True, exist_ok=True)
        archive_path = downloads_dir / "mathcraft-formula-rec.zip"
        _download_archive(archive_path, progress_callback)
        _install_archive(archive_path, destination, downloads_dir)

    if not is_mathcraft_model_cached(verify_hash=True):
        raise MathCraftModelDownloadError(
            f"MathCraft 模型安装后校验失败：{destination}"
        )
    return destination


def _download_archive(
    archive_path: Path,
    progress_callback: MathCraftProgressCallback | None,
) -> None:
    if _file_is_valid(
        archive_path,
        MATHCRAFT_ARCHIVE_SHA256,
        verify_hash=True,
        expected_size=MATHCRAFT_ARCHIVE_SIZE,
    ):
        _notify(progress_callback, MATHCRAFT_ARCHIVE_SIZE, MATHCRAFT_ARCHIVE_SIZE)
        return
    archive_path.unlink(missing_ok=True)

    partial_path = archive_path.with_suffix(archive_path.suffix + ".part")
    offset = partial_path.stat().st_size if partial_path.is_file() else 0
    if offset >= MATHCRAFT_ARCHIVE_SIZE:
        partial_path.unlink(missing_ok=True)
        offset = 0

    headers = {"User-Agent": "FormulaOCR/2.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    try:
        response = requests.get(
            MATHCRAFT_RELEASE_URL,
            stream=True,
            timeout=(20, 120),
            headers=headers,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise MathCraftModelDownloadError(
            f"MathCraft 模型下载失败：{MATHCRAFT_RELEASE_URL}\n{exc}"
        ) from exc

    append = offset > 0 and response.status_code == 206
    if not append:
        offset = 0
    mode = "ab" if append else "wb"
    try:
        _notify(progress_callback, offset, MATHCRAFT_ARCHIVE_SIZE)
        last_report = 0.0
        with partial_path.open(mode) as stream:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                stream.write(chunk)
                offset += len(chunk)
                now = time.monotonic()
                if now - last_report >= 0.5 or offset >= MATHCRAFT_ARCHIVE_SIZE:
                    _notify(progress_callback, offset, MATHCRAFT_ARCHIVE_SIZE)
                    last_report = now
    except (OSError, requests.RequestException) as exc:
        raise MathCraftModelDownloadError(
            f"MathCraft 下载中断，进度已保留：{partial_path}\n{exc}"
        ) from exc
    finally:
        response.close()

    if offset != MATHCRAFT_ARCHIVE_SIZE:
        partial_path.unlink(missing_ok=True)
        raise MathCraftModelDownloadError(
            f"MathCraft 模型下载不完整：{offset} / {MATHCRAFT_ARCHIVE_SIZE} 字节"
        )
    if _sha256(partial_path) != MATHCRAFT_ARCHIVE_SHA256:
        partial_path.unlink(missing_ok=True)
        raise MathCraftModelDownloadError("MathCraft 模型压缩包 SHA-256 校验失败")
    try:
        os.replace(partial_path, archive_path)
    except OSError as exc:
        raise MathCraftModelDownloadError(
            f"无法保存 MathCraft 模型压缩包：{archive_path}\n{exc}"
        ) from exc


def _install_archive(
    archive_path: Path,
    destination: Path,
    downloads_dir: Path,
) -> None:
    extraction_dir = Path(
        tempfile.mkdtemp(prefix=".MathCraftFormula-", dir=downloads_dir)
    )
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            members = archive.infolist()
            _validate_zip_members(members)
            archive.extractall(extraction_dir, members=members)
        if not _model_files_are_valid(extraction_dir):
            raise MathCraftModelDownloadError(
                "MathCraft 模型压缩包缺少文件或 SHA-256 校验失败"
            )

        backup = destination.with_name(destination.name + ".bak")
        if backup.exists():
            shutil.rmtree(backup)
        if destination.exists():
            destination.replace(backup)
        try:
            shutil.move(str(extraction_dir), str(destination))
        except OSError:
            if backup.exists() and not destination.exists():
                backup.replace(destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    except (OSError, zipfile.BadZipFile) as exc:
        raise MathCraftModelDownloadError(
            f"MathCraft 模型解压失败：{archive_path}\n{exc}"
        ) from exc
    finally:
        shutil.rmtree(extraction_dir, ignore_errors=True)
    archive_path.unlink(missing_ok=True)


def _validate_zip_members(members: list[zipfile.ZipInfo]) -> None:
    for member in members:
        # ZIP names conventionally use '/', but reject Windows separators too
        # so a malicious archive cannot pass validation on one host and escape
        # its extraction directory on another.
        normalized_name = member.filename.replace("\\", "/")
        path = Path(normalized_name)
        has_windows_drive = len(normalized_name) >= 2 and normalized_name[1] == ":"
        if (
            not normalized_name
            or "\x00" in normalized_name
            or path.is_absolute()
            or has_windows_drive
            or ".." in path.parts
        ):
            raise MathCraftModelDownloadError(
                f"MathCraft 压缩包包含不安全路径：{member.filename}"
            )
        mode = (member.external_attr >> 16) & 0xFFFF
        if mode and stat.S_ISLNK(mode):
            raise MathCraftModelDownloadError(
                f"MathCraft 压缩包包含不支持的链接：{member.filename}"
            )


def _model_files_are_valid(root: Path, *, verify_hash: bool = True) -> bool:
    return all(
        _file_is_valid(root / name, digest, verify_hash=verify_hash)
        for name, digest in MATHCRAFT_MODEL_FILES.items()
    )


def _file_is_valid(
    path: Path,
    digest: str,
    *,
    verify_hash: bool,
    expected_size: int | None = None,
) -> bool:
    try:
        if path.is_symlink() or not path.is_file():
            return False
        if expected_size is not None and path.stat().st_size != expected_size:
            return False
        return not verify_hash or _sha256(path) == digest
    except OSError:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _notify(
    callback: MathCraftProgressCallback | None,
    downloaded: int,
    total: int,
) -> None:
    if callback is not None:
        callback(
            MATHCRAFT_MODEL_ID,
            min(max(downloaded, 0), total),
            total,
        )
