"""Headless-browser renderer for local MathML preview images."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

from PIL import Image, ImageChops


class MathMLPreviewCancelled(RuntimeError):
    """Raised when a superseded or closing preview render is interrupted."""


def render_mathml_to_png(
    token: int,
    mathml: str,
    *,
    cache_dir: Path,
    cancel_event: threading.Event | None = None,
) -> Path:
    if cancel_event is not None and cancel_event.is_set():
        raise MathMLPreviewCancelled("MathML preview render was cancelled.")
    browser = find_browser_executable()
    if browser is None:
        raise RuntimeError("Edge/Chrome was not found for MathML preview.")

    render_dir = cache_dir / "mathml_preview"
    render_dir.mkdir(parents=True, exist_ok=True)
    html_path = render_dir / f"preview_{token}.html"
    png_path = render_dir / f"preview_{token}.png"
    profile_dir = render_dir / f"profile_{token}"
    png_path.unlink(missing_ok=True)
    html_path.write_text(mathml_preview_html(mathml), encoding="utf-8")

    args = [
        str(browser),
        "--headless",
        "--disable-gpu",
        "--hide-scrollbars",
        "--disable-extensions",
        "--disable-background-networking",
        "--no-first-run",
        "--no-default-browser-check",
        f"--user-data-dir={profile_dir}",
        "--default-background-color=fffbfdff",
        "--window-size=2200,620",
        f"--screenshot={png_path}",
        html_path.as_uri(),
    ]
    startupinfo = None
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= getattr(subprocess, "STARTF_USESHOWWINDOW", 1)
        startupinfo.wShowWindow = 0
    process: subprocess.Popen | None = None
    try:
        process = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
            startupinfo=startupinfo,
        )
        wait_for_rendered_png(
            png_path,
            timeout=10.0,
            cancel_event=cancel_event,
        )
        returncode = process.poll()
        if returncode not in (None, 0) and not png_path.exists():
            raise RuntimeError(f"Browser screenshot failed: {returncode}")
        # Headless Chromium normally exits after writing --screenshot. Give it
        # a moment to shut down its child processes cleanly before forcing it;
        # terminating immediately can leave a locked multi-megabyte profile.
        if (
            returncode is None
            and (cancel_event is None or not cancel_event.is_set())
        ):
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
    except Exception:
        png_path.unlink(missing_ok=True)
        raise
    finally:
        if process is not None:
            stop_preview_browser(process)
        html_path.unlink(missing_ok=True)
        for _attempt in range(3):
            shutil.rmtree(profile_dir, ignore_errors=True)
            if not profile_dir.exists():
                break
            time.sleep(0.08)

    trim_mathml_preview_image(png_path)
    return png_path


def wait_for_rendered_png(
    image_path: Path,
    *,
    timeout: float = 3.0,
    cancel_event: threading.Event | None = None,
) -> None:
    deadline = time.monotonic() + timeout
    last_size = -1
    stable_count = 0
    while time.monotonic() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            raise MathMLPreviewCancelled("MathML preview render was cancelled.")
        if image_path.exists():
            size = image_path.stat().st_size
            if size > 0 and size == last_size:
                try:
                    with Image.open(image_path) as image:
                        image.load()
                    stable_count += 1
                    if stable_count >= 2:
                        return
                except Exception:
                    stable_count = 0
            else:
                stable_count = 0
            last_size = size
        time.sleep(0.08)
    raise RuntimeError("Browser screenshot file was not ready.")


def stop_preview_browser(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=4,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000),
            )
            process.wait(timeout=1)
            return
        except Exception:
            pass
    try:
        process.terminate()
        process.wait(timeout=2)
    except Exception:
        try:
            process.kill()
            process.wait(timeout=1)
        except Exception:
            pass


def mathml_preview_html(mathml: str) -> str:
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
html, body {{
  margin: 0;
  width: 2200px;
  height: 620px;
  background: #fbfdff;
  overflow: hidden;
}}
body {{
  display: flex;
  align-items: center;
  justify-content: center;
  color: #172033;
  font-family: "Cambria Math", "Times New Roman", serif;
}}
.formula {{
  box-sizing: border-box;
  width: 2100px;
  min-height: 500px;
  padding: 44px 56px;
  display: flex;
  align-items: center;
  justify-content: center;
}}
math {{
  font-size: 42px;
  line-height: 1.45;
}}
mtd {{
  padding: 3px 8px;
}}
</style>
</head>
<body><div class="formula">{mathml}</div></body>
</html>
"""


def trim_mathml_preview_image(image_path: Path) -> None:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        background = Image.new("RGB", image.size, "#fbfdff")
        diff = ImageChops.difference(image, background)
        bbox = diff.getbbox()
        if bbox is None:
            image.save(image_path)
            return
        left, top, right, bottom = bbox
        margin = 28
        left = max(0, left - margin)
        top = max(0, top - margin)
        right = min(image.width, right + margin)
        bottom = min(image.height, bottom + margin)
        image.crop((left, top, right, bottom)).save(image_path)


def find_browser_executable() -> Path | None:
    candidates: list[Path] = []
    configured = os.environ.get("FORMULA_OCR_BROWSER", "").strip()
    if configured:
        candidates.append(Path(configured))
    for environment_name, relative_path in (
        ("ProgramFiles(x86)", ("Microsoft", "Edge", "Application", "msedge.exe")),
        ("ProgramFiles", ("Microsoft", "Edge", "Application", "msedge.exe")),
        ("ProgramFiles", ("Google", "Chrome", "Application", "chrome.exe")),
        ("LOCALAPPDATA", ("Google", "Chrome", "Application", "chrome.exe")),
    ):
        root = os.environ.get(environment_name, "").strip()
        if root:
            candidates.append(Path(root).joinpath(*relative_path))
    return next((path for path in candidates if path.is_file()), None)
