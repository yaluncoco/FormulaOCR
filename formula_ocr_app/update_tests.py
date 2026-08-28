from __future__ import annotations

import unittest

try:
    from formula_ocr_app.app_update import (
        LATEST_RELEASE_API,
        UpdateCheckError,
        fetch_latest_release,
        is_newer_version,
        normalize_version,
        parse_latest_release,
    )
except ModuleNotFoundError as exc:  # Allows direct script execution.
    if exc.name != "formula_ocr_app":
        raise
    from app_update import (
        LATEST_RELEASE_API,
        UpdateCheckError,
        fetch_latest_release,
        is_newer_version,
        normalize_version,
        parse_latest_release,
    )


def _release_payload(version: str = "1.1.1") -> dict[str, object]:
    tag = f"v{version}"
    return {
        "tag_name": tag,
        "name": f"FormulaOCR {version}",
        "html_url": f"https://github.com/yaluncoco/FormulaOCR/releases/tag/{tag}",
        "body": "修复编辑并加入更新检查。",
        "published_at": "2026-08-28T12:00:00Z",
        "draft": False,
        "prerelease": False,
        "assets": [
            {
                "name": f"FormulaOCRSetup-{version}.exe",
                "browser_download_url": (
                    "https://github.com/yaluncoco/FormulaOCR/releases/"
                    f"download/{tag}/FormulaOCRSetup-{version}.exe"
                ),
            },
            {
                "name": f"FormulaOCRSetup-{version}.exe.sha256",
                "browser_download_url": (
                    "https://github.com/yaluncoco/FormulaOCR/releases/"
                    f"download/{tag}/FormulaOCRSetup-{version}.exe.sha256"
                ),
            },
        ],
    }


class _FakeResponse:
    def __init__(self, payload: object, *, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error

    def raise_for_status(self) -> None:
        if self.error is not None:
            raise self.error

    def json(self) -> object:
        return self.payload


class UpdateTests(unittest.TestCase):
    def test_normalize_version_accepts_release_tag(self) -> None:
        self.assertEqual(normalize_version(" v1.2.3 "), "1.2.3")
        self.assertEqual(normalize_version("1.2.3-rc.1+build.8"), "1.2.3-rc.1")

    def test_normalize_version_rejects_unstructured_tag(self) -> None:
        with self.assertRaises(UpdateCheckError):
            normalize_version("latest")

    def test_version_comparison_handles_patch_and_prerelease(self) -> None:
        self.assertTrue(is_newer_version("1.1.2", "1.1.1"))
        self.assertTrue(is_newer_version("1.2.0", "1.1.99"))
        self.assertTrue(is_newer_version("1.2.0", "1.2.0-rc.1"))
        self.assertFalse(is_newer_version("1.2.0-rc.1", "1.2.0"))
        self.assertFalse(is_newer_version("1.1.1", "1.1.1"))

    def test_parse_latest_release_extracts_trusted_installer(self) -> None:
        release = parse_latest_release(
            _release_payload(),
            current_version="1.1.0",
        )
        self.assertTrue(release.update_available)
        self.assertEqual(release.latest_version, "1.1.1")
        self.assertTrue(release.installer_url.endswith("FormulaOCRSetup-1.1.1.exe"))
        self.assertIn("修复编辑", release.notes)

    def test_parse_latest_release_allows_release_without_installer(self) -> None:
        payload = _release_payload()
        payload["assets"] = []
        release = parse_latest_release(payload, current_version="1.1.1")
        self.assertFalse(release.update_available)
        self.assertEqual(release.installer_url, "")

    def test_parse_latest_release_rejects_untrusted_page(self) -> None:
        payload = _release_payload()
        payload["html_url"] = "https://example.com/FormulaOCR/releases/v1.1.1"
        with self.assertRaises(UpdateCheckError):
            parse_latest_release(payload, current_version="1.1.0")

    def test_parse_latest_release_rejects_prerelease(self) -> None:
        payload = _release_payload("1.2.0-rc.1")
        payload["prerelease"] = True
        with self.assertRaises(UpdateCheckError):
            parse_latest_release(payload, current_version="1.1.0")

    def test_fetch_latest_release_uses_bounded_github_request(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []

        def request_get(url: str, **kwargs):
            calls.append((url, kwargs))
            return _FakeResponse(_release_payload())

        release = fetch_latest_release("1.1.0", request_get=request_get)
        self.assertTrue(release.update_available)
        self.assertEqual(calls[0][0], LATEST_RELEASE_API)
        self.assertEqual(calls[0][1]["timeout"], (10, 20))
        self.assertIn("FormulaOCR/1.1.0", calls[0][1]["headers"]["User-Agent"])

    def test_fetch_latest_release_wraps_transport_error(self) -> None:
        def request_get(_url: str, **_kwargs):
            raise OSError("offline")

        with self.assertRaisesRegex(UpdateCheckError, "无法连接 GitHub"):
            fetch_latest_release("1.1.0", request_get=request_get)


if __name__ == "__main__":
    unittest.main()
