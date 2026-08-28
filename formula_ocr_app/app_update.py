"""Small, non-blocking-friendly GitHub Release update client."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlparse


LATEST_RELEASE_API = (
    "https://api.github.com/repos/yaluncoco/FormulaOCR/releases/latest"
)
RELEASES_URL = "https://github.com/yaluncoco/FormulaOCR/releases"
_REPOSITORY_RELEASE_PATH = "/yaluncoco/FormulaOCR/releases/"
_VERSION_PATTERN = re.compile(
    r"^[vV]?(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$"
)


class UpdateCheckError(RuntimeError):
    """Raised when the latest trustworthy release cannot be determined."""


@dataclass(frozen=True)
class ReleaseInfo:
    current_version: str
    latest_version: str
    tag_name: str
    release_name: str
    release_url: str
    installer_url: str
    notes: str
    published_at: str

    @property
    def update_available(self) -> bool:
        return is_newer_version(self.latest_version, self.current_version)


def normalize_version(value: str) -> str:
    match = _VERSION_PATTERN.fullmatch(str(value).strip())
    if match is None:
        raise UpdateCheckError(f"无法识别版本号：{value!r}")
    core = ".".join(match.group(index) for index in range(1, 4))
    prerelease = match.group(4)
    return core if prerelease is None else f"{core}-{prerelease}"


def is_newer_version(candidate: str, current: str) -> bool:
    return _version_key(candidate) > _version_key(current)


def parse_latest_release(payload: Any, *, current_version: str) -> ReleaseInfo:
    if not isinstance(payload, dict):
        raise UpdateCheckError("GitHub 返回的版本信息格式无效。")
    if payload.get("draft") or payload.get("prerelease"):
        raise UpdateCheckError("GitHub latest release 不是稳定正式版本。")

    tag_name = str(payload.get("tag_name", "")).strip()
    latest_version = normalize_version(tag_name)
    normalized_current = normalize_version(current_version)
    release_url = _trusted_github_url(
        payload.get("html_url"),
        label="Release 页面",
        required_path_prefix=_REPOSITORY_RELEASE_PATH,
    )

    installer_url = ""
    assets = payload.get("assets", ())
    if isinstance(assets, list):
        expected_name = f"FormulaOCRSetup-{latest_version}.exe".lower()
        for asset in assets:
            if not isinstance(asset, dict):
                continue
            name = str(asset.get("name", "")).strip()
            if name.lower() != expected_name:
                continue
            installer_url = _trusted_github_url(
                asset.get("browser_download_url"),
                label="安装包下载地址",
                required_path_prefix=(
                    f"{_REPOSITORY_RELEASE_PATH}download/{tag_name}/"
                ),
            )
            break

    notes = str(payload.get("body") or "").strip()
    if len(notes) > 20_000:
        notes = notes[:20_000].rstrip() + "\n…"
    return ReleaseInfo(
        current_version=normalized_current,
        latest_version=latest_version,
        tag_name=tag_name,
        release_name=str(payload.get("name") or tag_name).strip(),
        release_url=release_url,
        installer_url=installer_url,
        notes=notes,
        published_at=str(payload.get("published_at") or "").strip(),
    )


def fetch_latest_release(
    current_version: str,
    *,
    request_get: Callable[..., Any] | None = None,
) -> ReleaseInfo:
    if request_get is None:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - packaging boundary
            raise UpdateCheckError("程序缺少检查更新所需的网络组件。") from exc
        request_get = requests.get

    try:
        response = request_get(
            LATEST_RELEASE_API,
            timeout=(10, 20),
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": f"FormulaOCR/{normalize_version(current_version)}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        response.raise_for_status()
        payload = response.json()
    except UpdateCheckError:
        raise
    except Exception as exc:
        raise UpdateCheckError(
            "无法连接 GitHub 检查更新，请稍后重试。"
        ) from exc
    return parse_latest_release(payload, current_version=current_version)


def _version_key(value: str) -> tuple[tuple[int, int, int], int, tuple[Any, ...]]:
    normalized = normalize_version(value)
    core_text, separator, prerelease = normalized.partition("-")
    core = tuple(int(item) for item in core_text.split("."))
    if not separator:
        return core, 1, ()
    tokens: list[tuple[int, Any]] = []
    for token in prerelease.split("."):
        if token.isdigit():
            tokens.append((0, int(token)))
        else:
            tokens.append((1, token.lower()))
    return core, 0, tuple(tokens)


def _trusted_github_url(
    value: Any,
    *,
    label: str,
    required_path_prefix: str,
) -> str:
    url = str(value or "").strip()
    parsed = urlparse(url)
    if (
        parsed.scheme.lower() != "https"
        or parsed.hostname not in {"github.com", "www.github.com"}
        or parsed.username is not None
        or parsed.password is not None
        or not parsed.path.startswith(required_path_prefix)
    ):
        raise UpdateCheckError(f"{label}不是可信的 FormulaOCR GitHub 地址。")
    return url
