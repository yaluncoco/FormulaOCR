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
    from formula_ocr_app.model_downloader import DownloadProgressCallback
    from formula_ocr_app.runtime_paths import (
        bundled_external_model_dir,
        external_model_dir,
    )
except ImportError:  # Allows `python formula_ocr_app/app.py`.
    from model_downloader import DownloadProgressCallback
    from runtime_paths import bundled_external_model_dir, external_model_dir


MIXTEX_MODEL_ID = "MixTexZhEn"
MIXTEX_RELEASE_URL = (
    "https://github.com/RQLuo/MixTeX-Latex-OCR/releases/download/"
    "MixTeX-v3.2.4/MixTeX.zip"
)
MIXTEX_RELEASE_PAGE_URL = (
    "https://github.com/RQLuo/MixTeX-Latex-OCR/releases/tag/MixTeX-v3.2.4"
)
MIXTEX_TERMS_URL = (
    "https://github.com/RQLuo/MixTeX-Latex-OCR/blob/main/"
    "User%20Manual%26Terms%20of%20Service.md"
)
MIXTEX_ARCHIVE_NAME = "MixTeX.zip"
MIXTEX_ARCHIVE_SIZE = 294_025_378
MIXTEX_ARCHIVE_SHA256 = (
    "734088e8c3ac6d0ebf02b3054ed0cdde7d8be2eb57c33b8f049a66d05e026750"
)
CHUNK_SIZE = 1024 * 1024


# The release ZIP also contains a Windows executable and the upstream terms
# document.  FormulaOCR deliberately extracts only the model payload below;
# the executable is not part of this application's runtime.
MIXTEX_MODEL_FILES = {
    "added_tokens.json": (
        23,
        "4c88db53e3a71727fadb78d1d24dbe1963d8ae1e620e9978baa02e040a45921c",
    ),
    "config.json": (
        5_440,
        "a722a7b6a5d36f25157d6649fa5f91268a5e39b44fabffd5e8618d92a33d8c81",
    ),
    "decoder_model_merged.onnx": (
        208_151_959,
        "1750915b43d54758cc660e928ded3b390716cf96f9cfa225caad7b303147758b",
    ),
    "encoder_model.onnx": (
        200_980_554,
        "076fa63c75f6911c0a24e4a539c1024779434df74be03cba2cfdfa51793e0844",
    ),
    "generation_config.json": (
        201,
        "51d6ecd2e12874de66fe1b6f29fb23a32b72b9638037ffb144b9953867f65096",
    ),
    "merges.txt": (
        335_257,
        "f2b07c3b835c765fc99c6d0afaf2a1a4ab7084ca9d0fd0decd40f83d764bb921",
    ),
    "preprocessor_config.json": (
        228,
        "79afebe7759de552a652cb50564dd3a564ded61f1704141d967428d0af1e6d4d",
    ),
    "special_tokens_map.json": (
        1_008,
        "75cbec71553267dadc77a583ef15122c2aa7422eb6635acc72a184a413f17c1c",
    ),
    "tokenizer.json": (
        1_381_662,
        "73a436686364766d3e12bcaf30f87397ec66d218b566a0046ab60a5d86849a4c",
    ),
    "tokenizer_config.json": (
        1_271,
        "2ad604dead10b94853e6619f9bc6cb626e3ad3c20ca5366e621a968c6a10a5ca",
    ),
    "vocab.json": (
        536_076,
        "3a8206ef9f1b07f1bf28bca08615d12ed373632422ffb57dfb65d9e927947f06",
    ),
}
MIXTEX_TOTAL_MODEL_SIZE = sum(size for size, _digest in MIXTEX_MODEL_FILES.values())
MixTexProgressCallback = Callable[[str, int, int], None]


class MixTexModelDownloadError(RuntimeError):
    pass


def mixtex_model_dir() -> Path:
    return external_model_dir(MIXTEX_MODEL_ID)


def is_mixtex_model_cached(*, verify_hash: bool = False) -> bool:
    roots = [mixtex_model_dir()]
    bundled = bundled_external_model_dir(MIXTEX_MODEL_ID)
    if bundled is not None:
        roots.append(bundled)
    return any(_model_files_are_valid(root, verify_hash=verify_hash) for root in roots)


def ensure_mixtex_model(
    *, progress_callback: DownloadProgressCallback | None = None,
) -> Path:
    destination = mixtex_model_dir()
    if _model_files_are_valid(destination, verify_hash=True):
        _notify(progress_callback, MIXTEX_ARCHIVE_SIZE, MIXTEX_ARCHIVE_SIZE)
        return destination
    bundled = bundled_external_model_dir(MIXTEX_MODEL_ID)
    if bundled is not None and _model_files_are_valid(bundled, verify_hash=True):
        _notify(progress_callback, MIXTEX_ARCHIVE_SIZE, MIXTEX_ARCHIVE_SIZE)
        return bundled

    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(destination.with_suffix(".lock")))
    with lock:
        if _model_files_are_valid(destination, verify_hash=True):
            _notify(progress_callback, MIXTEX_ARCHIVE_SIZE, MIXTEX_ARCHIVE_SIZE)
            return destination
        bundled = bundled_external_model_dir(MIXTEX_MODEL_ID)
        if bundled is not None and _model_files_are_valid(bundled, verify_hash=True):
            _notify(progress_callback, MIXTEX_ARCHIVE_SIZE, MIXTEX_ARCHIVE_SIZE)
            return bundled

        downloads_dir = destination.parent / ".downloads" / MIXTEX_MODEL_ID
        downloads_dir.mkdir(parents=True, exist_ok=True)
        archive_path = downloads_dir / MIXTEX_ARCHIVE_NAME
        _download_archive(archive_path, progress_callback)
        _install_archive(archive_path, destination, downloads_dir)

    if not is_mixtex_model_cached(verify_hash=True):
        raise MixTexModelDownloadError(
            f"MixTeX 模型安装后校验失败：{destination}"
        )
    return destination


def _download_archive(
    archive_path: Path,
    progress_callback: MixTexProgressCallback | None,
) -> None:
    if _file_is_valid(
        archive_path,
        MIXTEX_ARCHIVE_SHA256,
        verify_hash=True,
        expected_size=MIXTEX_ARCHIVE_SIZE,
    ):
        _notify(progress_callback, MIXTEX_ARCHIVE_SIZE, MIXTEX_ARCHIVE_SIZE)
        return
    archive_path.unlink(missing_ok=True)

    partial_path = archive_path.with_suffix(archive_path.suffix + ".part")
    offset = partial_path.stat().st_size if partial_path.is_file() else 0
    if offset >= MIXTEX_ARCHIVE_SIZE:
        partial_path.unlink(missing_ok=True)
        offset = 0

    headers = {"User-Agent": "FormulaOCR/2.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    try:
        response = requests.get(
            MIXTEX_RELEASE_URL,
            stream=True,
            timeout=(20, 180),
            headers=headers,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise MixTexModelDownloadError(
            f"MixTeX 模型下载失败：{MIXTEX_RELEASE_PAGE_URL}\n{exc}"
        ) from exc

    append = offset > 0 and response.status_code == 206
    if not append:
        offset = 0
    mode = "ab" if append else "wb"
    try:
        _notify(progress_callback, offset, MIXTEX_ARCHIVE_SIZE)
        last_report = 0.0
        with partial_path.open(mode) as stream:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                stream.write(chunk)
                offset += len(chunk)
                now = time.monotonic()
                if now - last_report >= 0.5 or offset >= MIXTEX_ARCHIVE_SIZE:
                    _notify(progress_callback, offset, MIXTEX_ARCHIVE_SIZE)
                    last_report = now
    except (OSError, requests.RequestException) as exc:
        raise MixTexModelDownloadError(
            f"MixTeX 下载中断，进度已保留：{partial_path}\n{exc}"
        ) from exc
    finally:
        response.close()

    if offset != MIXTEX_ARCHIVE_SIZE:
        partial_path.unlink(missing_ok=True)
        raise MixTexModelDownloadError(
            f"MixTeX 模型下载不完整：{offset} / {MIXTEX_ARCHIVE_SIZE} 字节"
        )
    if _sha256(partial_path) != MIXTEX_ARCHIVE_SHA256:
        partial_path.unlink(missing_ok=True)
        raise MixTexModelDownloadError("MixTeX 模型压缩包 SHA-256 校验失败")
    try:
        os.replace(partial_path, archive_path)
    except OSError as exc:
        raise MixTexModelDownloadError(
            f"无法保存 MixTeX 模型压缩包：{archive_path}\n{exc}"
        ) from exc


def _install_archive(
    archive_path: Path,
    destination: Path,
    downloads_dir: Path,
) -> None:
    extraction_dir = Path(
        tempfile.mkdtemp(prefix=f".{MIXTEX_MODEL_ID}-", dir=downloads_dir)
    )
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            members = archive.infolist()
            _validate_zip_members(members)
            archive.extractall(extraction_dir, members=members)

        source = extraction_dir / "onnx"
        if not _model_files_are_valid(source, verify_hash=True):
            raise MixTexModelDownloadError(
                "MixTeX 压缩包缺少 onnx 模型文件或 SHA-256 校验失败"
            )
        _replace_model_directory(source, destination)
    except (OSError, zipfile.BadZipFile) as exc:
        raise MixTexModelDownloadError(
            f"MixTeX 模型解压失败：{archive_path}\n{exc}"
        ) from exc
    finally:
        shutil.rmtree(extraction_dir, ignore_errors=True)
    archive_path.unlink(missing_ok=True)


def _replace_model_directory(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise MixTexModelDownloadError(
            f"拒绝覆盖链接模型目录：{destination}"
        )
    backup = destination.with_name(destination.name + ".bak")
    if backup.exists() or backup.is_symlink():
        if backup.is_symlink() or not backup.is_dir():
            raise MixTexModelDownloadError(f"模型备份路径不是安全目录：{backup}")
        shutil.rmtree(backup)
    if destination.exists():
        if not destination.is_dir():
            raise MixTexModelDownloadError(f"模型安装路径不是目录：{destination}")
        destination.replace(backup)
    try:
        shutil.move(str(source), str(destination))
    except OSError:
        if backup.exists() and not destination.exists():
            backup.replace(destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _validate_zip_members(members: list[zipfile.ZipInfo]) -> None:
    for member in members:
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
            raise MixTexModelDownloadError(
                f"MixTeX 压缩包包含不安全路径：{member.filename}"
            )
        mode = (member.external_attr >> 16) & 0xFFFF
        if mode and stat.S_ISLNK(mode):
            raise MixTexModelDownloadError(
                f"MixTeX 压缩包包含不支持的链接：{member.filename}"
            )


def _model_files_are_valid(root: Path, *, verify_hash: bool) -> bool:
    return all(
        _file_is_valid(
            root / filename,
            digest,
            verify_hash=verify_hash,
            expected_size=size,
        )
        for filename, (size, digest) in MIXTEX_MODEL_FILES.items()
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
    callback: MixTexProgressCallback | None,
    downloaded: int,
    total: int,
) -> None:
    if callback is not None:
        callback(MIXTEX_MODEL_ID, min(max(downloaded, 0), total), total)
