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
    from formula_ocr_app.model_api import DownloadProgressCallback
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
    from model_api import DownloadProgressCallback
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

    ensure_safe_directory(destination.parent)
    lock = InterProcessFileLock(
        destination.with_suffix(".lock"),
        on_wait=lambda: _notify(progress_callback, 0, MIXTEX_ARCHIVE_SIZE),
    )
    with lock:
        recover_model_directory_backup(
            destination,
            is_model_valid=lambda path: _model_files_are_valid(
                path,
                verify_hash=True,
            ),
            error_type=MixTexModelDownloadError,
        )
        if _model_files_are_valid(destination, verify_hash=True):
            _notify(progress_callback, MIXTEX_ARCHIVE_SIZE, MIXTEX_ARCHIVE_SIZE)
            return destination
        bundled = bundled_external_model_dir(MIXTEX_MODEL_ID)
        if bundled is not None and _model_files_are_valid(bundled, verify_hash=True):
            _notify(progress_callback, MIXTEX_ARCHIVE_SIZE, MIXTEX_ARCHIVE_SIZE)
            return bundled

        downloads_root = destination.parent / ".downloads"
        ensure_safe_directory(downloads_root)
        downloads_dir = downloads_root / MIXTEX_MODEL_ID
        ensure_safe_directory(downloads_dir)
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
    import requests

    if _file_is_valid(
        archive_path,
        MIXTEX_ARCHIVE_SHA256,
        verify_hash=True,
        expected_size=MIXTEX_ARCHIVE_SIZE,
    ):
        _notify(progress_callback, MIXTEX_ARCHIVE_SIZE, MIXTEX_ARCHIVE_SIZE)
        return
    partial_path = archive_path.with_suffix(archive_path.suffix + ".part")
    try:
        download_verified_file(
            RemoteFileSpec(
                name=MIXTEX_ARCHIVE_NAME,
                size=MIXTEX_ARCHIVE_SIZE,
                sha256=MIXTEX_ARCHIVE_SHA256,
                url=MIXTEX_RELEASE_URL,
            ),
            archive_path,
            partial=partial_path,
            completed=0,
            total=MIXTEX_ARCHIVE_SIZE,
            notify=lambda downloaded, total: _notify(
                progress_callback, downloaded, total
            ),
            request_get=requests.get,
            request_exception=requests.RequestException,
            timeout=(20, 180),
            chunk_size=CHUNK_SIZE,
        )
    except VerifiedDownloadFailure as failure:
        raise_model_download_error(
            failure,
            error_type=MixTexModelDownloadError,
            label="MixTeX",
        )


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
        replace_model_directory(
            source,
            destination,
            is_model_valid=lambda path: _model_files_are_valid(
                path,
                verify_hash=True,
            ),
            error_type=MixTexModelDownloadError,
        )
    except (OSError, zipfile.BadZipFile) as exc:
        raise MixTexModelDownloadError(
            f"MixTeX 模型解压失败：{archive_path}\n{exc}"
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
        raise MixTexModelDownloadError("MixTeX 压缩包解压规模超过安全上限")
    for member in members:
        if not archive_member_name_is_safe(member.filename):
            raise MixTexModelDownloadError(
                f"MixTeX 压缩包包含不安全路径：{member.filename}"
            )
        mode = (member.external_attr >> 16) & 0xFFFF
        if mode and stat.S_ISLNK(mode):
            raise MixTexModelDownloadError(
                f"MixTeX 压缩包包含不支持的链接：{member.filename}"
            )


def _model_files_are_valid(root: Path, *, verify_hash: bool) -> bool:
    try:
        if root.is_symlink() or not root.is_dir():
            return False
    except OSError:
        return False
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
        return file_is_valid(
            path,
            expected_size if expected_size is not None else path.stat().st_size,
            digest,
            verify_hash=verify_hash,
        )
    except OSError:
        return False


def _notify(
    callback: MixTexProgressCallback | None,
    downloaded: int,
    total: int,
) -> None:
    if callback is not None:
        callback(MIXTEX_MODEL_ID, min(max(downloaded, 0), total), total)
