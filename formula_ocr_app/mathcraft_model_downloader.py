from __future__ import annotations

import shutil
import stat
import tempfile
import zipfile
from pathlib import Path
from typing import Callable

try:
    from formula_ocr_app.download_utils import (
        RemoteFileSpec,
        VerifiedDownloadFailure,
        archive_member_name_is_safe,
        archive_payload_is_within_limits,
        download_verified_file,
        ensure_safe_directory,
        file_is_valid,
        raise_model_download_error,
        recover_model_directory_backup,
        replace_model_directory,
    )
    from formula_ocr_app.interprocess_lock import InterProcessFileLock
    from formula_ocr_app.runtime_paths import (
        bundled_external_model_dir,
        external_model_dir,
    )
except ModuleNotFoundError as exc:  # Allows `python formula_ocr_app/app.py`.
    if exc.name != "formula_ocr_app":
        raise
    from download_utils import (
        RemoteFileSpec,
        VerifiedDownloadFailure,
        archive_member_name_is_safe,
        archive_payload_is_within_limits,
        download_verified_file,
        ensure_safe_directory,
        file_is_valid,
        raise_model_download_error,
        recover_model_directory_backup,
        replace_model_directory,
    )
    from interprocess_lock import InterProcessFileLock
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

    ensure_safe_directory(destination.parent)
    lock = InterProcessFileLock(
        destination.with_suffix(".lock"),
        on_wait=lambda: _notify(progress_callback, 0, MATHCRAFT_ARCHIVE_SIZE),
    )

    with lock:
        recover_model_directory_backup(
            destination,
            is_model_valid=lambda path: _model_files_are_valid(
                path,
                verify_hash=True,
            ),
            error_type=MathCraftModelDownloadError,
        )
        if _model_files_are_valid(destination, verify_hash=True):
            _notify(progress_callback, MATHCRAFT_ARCHIVE_SIZE, MATHCRAFT_ARCHIVE_SIZE)
            return destination
        bundled = bundled_external_model_dir(MATHCRAFT_MODEL_ID)
        if bundled is not None and _model_files_are_valid(bundled, verify_hash=True):
            _notify(progress_callback, MATHCRAFT_ARCHIVE_SIZE, MATHCRAFT_ARCHIVE_SIZE)
            return bundled

        downloads_dir = destination.parent / ".downloads"
        ensure_safe_directory(downloads_dir)
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
    import requests

    if _file_is_valid(
        archive_path,
        MATHCRAFT_ARCHIVE_SHA256,
        verify_hash=True,
        expected_size=MATHCRAFT_ARCHIVE_SIZE,
    ):
        _notify(progress_callback, MATHCRAFT_ARCHIVE_SIZE, MATHCRAFT_ARCHIVE_SIZE)
        return
    partial_path = archive_path.with_suffix(archive_path.suffix + ".part")
    try:
        download_verified_file(
            RemoteFileSpec(
                name="mathcraft-formula-rec.zip",
                size=MATHCRAFT_ARCHIVE_SIZE,
                sha256=MATHCRAFT_ARCHIVE_SHA256,
                url=MATHCRAFT_RELEASE_URL,
            ),
            archive_path,
            partial=partial_path,
            completed=0,
            total=MATHCRAFT_ARCHIVE_SIZE,
            notify=lambda downloaded, total: _notify(
                progress_callback, downloaded, total
            ),
            request_get=requests.get,
            request_exception=requests.RequestException,
            timeout=(20, 120),
            chunk_size=CHUNK_SIZE,
        )
    except VerifiedDownloadFailure as failure:
        raise_model_download_error(
            failure,
            error_type=MathCraftModelDownloadError,
            label="MathCraft",
        )


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

        replace_model_directory(
            extraction_dir,
            destination,
            is_model_valid=lambda path: _model_files_are_valid(
                path,
                verify_hash=True,
            ),
            error_type=MathCraftModelDownloadError,
        )
    except (OSError, zipfile.BadZipFile) as exc:
        raise MathCraftModelDownloadError(
            f"MathCraft 模型解压失败：{archive_path}\n{exc}"
        ) from exc
    finally:
        shutil.rmtree(extraction_dir, ignore_errors=True)
    try:
        archive_path.unlink(missing_ok=True)
    except OSError:
        pass


def _validate_zip_members(members: list[zipfile.ZipInfo]) -> None:
    if not archive_payload_is_within_limits(
        len(members),
        sum(max(0, member.file_size) for member in members),
    ):
        raise MathCraftModelDownloadError("MathCraft 压缩包解压规模超过安全上限")
    for member in members:
        # ZIP names conventionally use '/', but reject Windows separators too
        # so a malicious archive cannot pass validation on one host and escape
        # its extraction directory on another.
        if not archive_member_name_is_safe(member.filename):
            raise MathCraftModelDownloadError(
                f"MathCraft 压缩包包含不安全路径：{member.filename}"
            )
        mode = (member.external_attr >> 16) & 0xFFFF
        if mode and stat.S_ISLNK(mode):
            raise MathCraftModelDownloadError(
                f"MathCraft 压缩包包含不支持的链接：{member.filename}"
            )


def _model_files_are_valid(root: Path, *, verify_hash: bool = True) -> bool:
    try:
        if root.is_symlink() or not root.is_dir():
            return False
    except OSError:
        return False
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
        return file_is_valid(
            path,
            expected_size if expected_size is not None else path.stat().st_size,
            digest,
            verify_hash=verify_hash,
        )
    except OSError:
        return False


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
