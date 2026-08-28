from __future__ import annotations

import argparse
import ctypes
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
import traceback
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageGrab, ImageTk

try:
    from formula_ocr_app.app_settings import AppSettings, load_settings, save_settings
    from formula_ocr_app.formula_formats import (
        export_formula_docx,
        latex_to_asciimath,
        latex_to_equation_environment,
        latex_to_html,
        latex_to_markdown_block,
        latex_to_markdown_inline,
        latex_to_mathml,
        latex_to_typst,
        latex_to_word_linear,
        mathml_to_omml,
        mathml_to_word_mathml,
    )
    from formula_ocr_app.image_utils import image_to_rgb, load_rgb_image
    from formula_ocr_app.model_api import ModelDownloadCancelled, ModelDownloadError
    from formula_ocr_app.model_catalog import (
        DEFAULT_MODEL_ID,
        MODEL_BY_ID,
        MODEL_SPECS,
        get_model_spec,
    )
    from formula_ocr_app.model_runtime import (
        ensure_model,
        is_model_bundled_only,
        is_model_cached,
        model_status_label,
        model_user_cache_path,
    )
    from formula_ocr_app.recognition_pipeline import FormulaRecognizer
    from formula_ocr_app.runtime_paths import (
        runtime_cache_dir,
        runtime_log_dir,
    )
    from formula_ocr_app.word_clipboard import (
        FORMAT_HTML,
        FORMAT_MATHML,
        FORMAT_MATHML_PRESENTATION,
        FORMAT_OFFICE_OPEN_XML,
        copy_mathml_for_word_to_clipboard,
        tk_clipboard_text,
        windows_clipboard_formats,
        windows_clipboard_text,
    )
except ModuleNotFoundError as exc:  # Allows `python formula_ocr_app/app.py`.
    if exc.name != "formula_ocr_app":
        raise
    from app_settings import AppSettings, load_settings, save_settings
    from formula_formats import (
        export_formula_docx,
        latex_to_asciimath,
        latex_to_equation_environment,
        latex_to_html,
        latex_to_markdown_block,
        latex_to_markdown_inline,
        latex_to_mathml,
        latex_to_typst,
        latex_to_word_linear,
        mathml_to_omml,
        mathml_to_word_mathml,
    )
    from image_utils import image_to_rgb, load_rgb_image
    from model_api import ModelDownloadCancelled, ModelDownloadError
    from model_catalog import (
        DEFAULT_MODEL_ID,
        MODEL_BY_ID,
        MODEL_SPECS,
        get_model_spec,
    )
    from model_runtime import (
        ensure_model,
        is_model_bundled_only,
        is_model_cached,
        model_status_label,
        model_user_cache_path,
    )
    from recognition_pipeline import FormulaRecognizer
    from runtime_paths import (
        runtime_cache_dir,
        runtime_log_dir,
    )
    from word_clipboard import (
        FORMAT_HTML,
        FORMAT_MATHML,
        FORMAT_MATHML_PRESENTATION,
        FORMAT_OFFICE_OPEN_XML,
        copy_mathml_for_word_to_clipboard,
        tk_clipboard_text,
        windows_clipboard_formats,
        windows_clipboard_text,
    )

try:
    from formula_ocr_app.ui_widgets import (
        ACCENT,
        ACCENT_DARK,
        ACCENT_SOFT,
        APP_BG,
        BORDER,
        PANEL_BG,
        SURFACE_SUBTLE,
        TEXT_PRIMARY,
        TEXT_SECONDARY,
        ModelFilterChips,
        ModelPicker,
        RoundedButton,
        RoundedChoice,
        RoundedPanel,
        SlimScrollbar,
        _anchored_popup_geometry,
        _ScreenArea,
    )
except ModuleNotFoundError as exc:  # Allows `python formula_ocr_app/app.py`.
    if exc.name != "formula_ocr_app":
        raise
    from ui_widgets import (
        ACCENT,
        ACCENT_DARK,
        ACCENT_SOFT,
        APP_BG,
        BORDER,
        PANEL_BG,
        SURFACE_SUBTLE,
        TEXT_PRIMARY,
        TEXT_SECONDARY,
        ModelFilterChips,
        ModelPicker,
        RoundedButton,
        RoundedChoice,
        RoundedPanel,
        SlimScrollbar,
        _anchored_popup_geometry,
        _ScreenArea,
    )


APP_ROOT = Path(__file__).resolve().parent


def _resource_base() -> Path:
    bundled = getattr(sys, "_MEIPASS", None)
    if bundled:
        return Path(bundled)
    return APP_ROOT.parent


CACHE_DIR = runtime_cache_dir()
LOG_DIR = runtime_log_dir()
LOG_FILE = LOG_DIR / "formula_ocr.log"
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3
_LOG_WRITE_LOCK = threading.Lock()
ICON_FILE = _resource_base() / "icon.png"
ICON_ICO_FILE = _resource_base() / "icon.ico"
SESSION_MAX_AGE_SECONDS = 7 * 24 * 60 * 60
LEGACY_PREVIEW_MAX_AGE_SECONDS = 7 * 24 * 60 * 60


def _safe_unlink_temporary_file(path: Path, *, context: str) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        write_log(f"Unable to remove {context} temporary file {path}: {exc}")


def _create_runtime_session_dir() -> Path:
    """Create an instance-private area for input and preview files.

    Multiple FormulaOCR windows must never share ``current_formula.png`` or a
    browser profile. Old crash leftovers are removed only after seven days so
    another currently running instance cannot be disturbed.
    """

    _cleanup_legacy_mathml_preview_cache()
    sessions_root = CACHE_DIR / "sessions"
    sessions_root.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - SESSION_MAX_AGE_SECONDS
    try:
        children = tuple(sessions_root.iterdir())
    except OSError:
        children = ()
    for child in children:
        try:
            if (
                child.is_dir()
                and not child.is_symlink()
                and child.stat().st_mtime < cutoff
                and not _session_owner_is_running(child.name)
            ):
                shutil.rmtree(child, ignore_errors=True)
        except OSError:
            continue
    return Path(tempfile.mkdtemp(prefix=f"{os.getpid()}-", dir=sessions_root))


def _cleanup_legacy_mathml_preview_cache() -> None:
    """Remove only stale files from the pre-session preview cache layout."""

    legacy_root = CACHE_DIR / "mathml_preview"
    try:
        if legacy_root.is_symlink() or not legacy_root.is_dir():
            return
        children = tuple(legacy_root.iterdir())
    except OSError:
        return

    cutoff = time.time() - LEGACY_PREVIEW_MAX_AGE_SECONDS
    for child in children:
        name = child.name
        known_profile = name.startswith("profile_")
        known_preview = name.startswith("preview_") and child.suffix.lower() in {
            ".html",
            ".png",
        }
        if not known_profile and not known_preview:
            continue
        try:
            if child.is_symlink() or _latest_tree_mtime(child) >= cutoff:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            elif child.is_file():
                child.unlink()
        except OSError:
            continue

    try:
        legacy_root.rmdir()
    except OSError:
        # Recent or unknown files intentionally keep the legacy directory.
        pass


def _latest_tree_mtime(path: Path) -> float:
    """Return the newest lstat timestamp without following cache symlinks."""

    latest = path.lstat().st_mtime
    if not path.is_dir() or path.is_symlink():
        return latest
    for root, directory_names, file_names in os.walk(path, followlinks=False):
        root_path = Path(root)
        for name in (*directory_names, *file_names):
            try:
                latest = max(latest, (root_path / name).lstat().st_mtime)
            except OSError:
                # A browser process may remove a file while an old profile is
                # inspected. Treat that as a live/unstable tree and retain it.
                return time.time()
    return latest


def _session_owner_is_running(session_name: str) -> bool:
    pid_text, separator, _random_suffix = session_name.partition("-")
    if not separator or not pid_text.isdigit():
        return False
    pid = int(pid_text)
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if sys.platform == "win32":
        process_query_limited_information = 0x1000
        still_active = 259
        try:
            kernel32 = ctypes.windll.kernel32
            kernel32.OpenProcess.argtypes = (
                ctypes.c_uint,
                ctypes.c_int,
                ctypes.c_uint,
            )
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.GetExitCodeProcess.argtypes = (
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_ulong),
            )
            kernel32.GetExitCodeProcess.restype = ctypes.c_int
            kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
            handle = kernel32.OpenProcess(
                process_query_limited_information,
                False,
                pid,
            )
            if not handle:
                return False
            try:
                exit_code = ctypes.c_ulong()
                return bool(
                    kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
                    and exit_code.value == still_active
                )
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _virtual_screen_area(anchor: tk.Misc) -> _ScreenArea:
    """Return the complete Windows virtual desktop, including negative axes."""

    if sys.platform == "win32":
        try:
            user32 = ctypes.windll.user32
            left = int(user32.GetSystemMetrics(76))  # SM_XVIRTUALSCREEN
            top = int(user32.GetSystemMetrics(77))  # SM_YVIRTUALSCREEN
            width = int(user32.GetSystemMetrics(78))  # SM_CXVIRTUALSCREEN
            height = int(user32.GetSystemMetrics(79))  # SM_CYVIRTUALSCREEN
            if width > 0 and height > 0:
                return _ScreenArea(left, top, left + width, top + height)
        except Exception:
            pass
    return _ScreenArea(0, 0, anchor.winfo_screenwidth(), anchor.winfo_screenheight())


def _set_toplevel_bounds(
    window: tk.Toplevel,
    area: _ScreenArea,
) -> None:
    """Place a toplevel at absolute virtual-desktop coordinates on Windows."""

    width = max(1, area.right - area.left)
    height = max(1, area.bottom - area.top)
    window.geometry(f"{width}x{height}+0+0")
    window.update_idletasks()
    if sys.platform == "win32":
        try:
            user32 = ctypes.windll.user32
            user32.GetParent.argtypes = (ctypes.c_void_p,)
            user32.GetParent.restype = ctypes.c_void_p
            user32.SetWindowPos.argtypes = (
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_uint,
            )
            user32.SetWindowPos.restype = ctypes.c_int
            hwnd = user32.GetParent(window.winfo_id()) or window.winfo_id()
            if user32.SetWindowPos(
                hwnd,
                -1,
                area.left,
                area.top,
                width,
                height,
                0x0050,
            ):
                return
        except Exception:
            pass
    window.geometry(f"{width}x{height}{area.left:+d}{area.top:+d}")


@dataclass(frozen=True)
class RecognizerSettings:
    model_name: str


class FormulaOCRApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("公式识别助手")
        self.geometry("1240x780")
        self.minsize(1040, 650)
        self.configure(bg=APP_BG)
        self.protocol("WM_DELETE_WINDOW", self.destroy)

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.session_dir = _create_runtime_session_dir()
        self.saved_settings = load_settings()
        self.accepted_model_terms = set(self.saved_settings.accepted_model_terms)
        self.current_image: Image.Image | None = None
        self.current_image_path: Path | None = None
        self.image_revision = 0
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.preview_resize_after_id: str | None = None
        self.preview_render_size: tuple[int, int] | None = None
        self.window_icon: tk.PhotoImage | None = None
        self.recognizer: FormulaRecognizer | None = None
        self.recognizer_settings: RecognizerSettings | None = None
        self.recognizer_lock = threading.Lock()
        self.session_cleanup_lock = threading.Lock()
        self.recognition_thread: threading.Thread | None = None
        self.model_download_thread: threading.Thread | None = None
        self.worker_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.mathml_preview_queue: queue.Queue[tuple[int, str, str]] = queue.Queue()
        self.is_busy = False
        self.mathml_update_after_id: str | None = None
        self.worker_poll_after_id: str | None = None
        self.mathml_preview_poll_after_id: str | None = None
        self.mathml_render_token = 0
        self.mathml_render_lock = threading.Lock()
        self.mathml_pending_render: tuple[int, str, str] | None = None
        self.mathml_render_thread: threading.Thread | None = None
        self.mathml_active_cancel_event: threading.Event | None = None
        self.mathml_preview_photo: ImageTk.PhotoImage | None = None
        self.mathml_preview_source: Image.Image | None = None
        self.mathml_preview_resize_after_id: str | None = None
        self.mathml_preview_render_size: tuple[int, int] | None = None
        self.busy_started_at: float | None = None
        self.busy_status_after_id: str | None = None
        self.busy_status_message = "正在加载模型/识别公式..."
        self.download_cancel_event = threading.Event()
        self.is_destroying = False
        self.model_manager_window: tk.Toplevel | None = None
        self.capture_after_id: str | None = None
        self.screenshot_selector: ScreenshotSelector | None = None

        self._configure_styles()
        self._set_window_icon()
        self._build_ui()
        self._bind_shortcuts()
        self._schedule_worker_poll()
        self._schedule_mathml_preview_poll()

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("App.TFrame", background=APP_BG)
        style.configure("Panel.TFrame", background=PANEL_BG)
        style.configure("Toolbar.TFrame", background=PANEL_BG)
        style.configure(
            "Title.TLabel",
            background=APP_BG,
            foreground=TEXT_PRIMARY,
            font=("Microsoft YaHei UI", 18, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=APP_BG,
            foreground=TEXT_SECONDARY,
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "PanelTitle.TLabel",
            background=PANEL_BG,
            foreground=TEXT_PRIMARY,
            font=("Microsoft YaHei UI", 11, "bold"),
        )
        style.configure(
            "Muted.TLabel",
            background=PANEL_BG,
            foreground=TEXT_SECONDARY,
            font=("Microsoft YaHei UI", 9),
        )
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=(12, 7))
        style.configure(
            "Model.Treeview",
            background=PANEL_BG,
            fieldbackground=PANEL_BG,
            foreground=TEXT_PRIMARY,
            rowheight=40,
            borderwidth=0,
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "Model.Treeview.Heading",
            background="#edf2f8",
            foreground=TEXT_SECONDARY,
            relief=tk.FLAT,
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        style.map(
            "Model.Treeview",
            background=[("selected", ACCENT_SOFT)],
            foreground=[("selected", TEXT_PRIMARY)],
        )
        style.configure(
            "Accent.TButton",
            foreground="#ffffff",
            background=ACCENT,
            bordercolor=ACCENT,
            focusthickness=0,
            padding=(18, 8),
        )
        style.map(
            "Accent.TButton",
            background=[("active", ACCENT_DARK), ("disabled", "#9fb7eb")],
            foreground=[("disabled", "#f4f7ff")],
        )
        style.configure(
            "Status.Horizontal.TProgressbar",
            troughcolor="#dfe7f2",
            background=ACCENT,
            lightcolor=ACCENT,
            darkcolor=ACCENT,
            bordercolor=APP_BG,
            thickness=6,
        )

    def _set_window_icon(self) -> None:
        try:
            if ICON_ICO_FILE.exists():
                self.iconbitmap(default=str(ICON_ICO_FILE))
            if ICON_FILE.exists():
                self.window_icon = tk.PhotoImage(file=str(ICON_FILE))
                self.iconphoto(True, self.window_icon)
        except tk.TclError:
            write_log(f"Unable to load window icon: {ICON_ICO_FILE} / {ICON_FILE}")

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = tk.Frame(self, bg=APP_BG)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        header.configure(padx=22, pady=18)

        title_group = tk.Frame(header, bg=APP_BG)
        title_group.grid(row=0, column=0, sticky="w")
        tk.Label(
            title_group,
            text="公式识别助手",
            bg=APP_BG,
            fg=TEXT_PRIMARY,
            font=("Microsoft YaHei UI", 20, "bold"),
        ).grid(
            row=0, column=0, sticky="w"
        )
        tk.Label(
            title_group,
            text="图片公式转 LaTeX，本地多模型识别",
            bg=APP_BG,
            fg=TEXT_SECONDARY,
            font=("Microsoft YaHei UI", 10),
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        model_bar = tk.Frame(header, bg=APP_BG)
        model_bar.grid(row=0, column=1, sticky="e")
        selected_model = (
            self.saved_settings.model_id
            if self.saved_settings.model_id in MODEL_BY_ID
            and is_model_cached(self.saved_settings.model_id, verify_hash=False)
            else ""
        )
        self.model_picker = ModelPicker(
            model_bar,
            specs=MODEL_SPECS,
            model_id=selected_model,
            command=self._on_model_changed,
            status_provider=model_status_label,
            manager_command=self.show_model_manager,
        )
        self.model_picker.pack(side=tk.LEFT)
        self.model_manager_button = RoundedButton(
            model_bar,
            text="模型管理",
            command=self.show_model_manager,
            width=94,
            height=38,
            radius=13,
            bg="#ffffff",
            active_bg=ACCENT_SOFT,
            border=BORDER,
        )
        self.model_manager_button.pack(side=tk.LEFT, padx=(8, 0))
        self.recognize_button = RoundedButton(
            model_bar,
            text="识别",
            command=self.recognize_image,
            width=92,
            height=38,
            radius=13,
            bg=ACCENT,
            active_bg=ACCENT_DARK,
            fg="#ffffff",
            border=ACCENT,
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        self.recognize_button.pack(side=tk.LEFT, padx=(10, 0))

        self.model_info_var = tk.StringVar()
        tk.Label(
            header,
            textvariable=self.model_info_var,
            bg=APP_BG,
            fg=TEXT_SECONDARY,
            anchor=tk.E,
            font=("Microsoft YaHei UI", 9),
        ).grid(row=1, column=1, sticky="e", pady=(7, 0))
        self._update_model_summary()

        content = tk.Frame(self, bg=APP_BG)
        content.grid(row=1, column=0, sticky="nsew", padx=22, pady=(0, 14))
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(0, weight=1)

        left_panel = RoundedPanel(content, radius=22, padding=18)
        right_panel = RoundedPanel(content, radius=22, padding=18)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        left = left_panel.content
        right = right_panel.content

        left.rowconfigure(2, weight=1)
        left.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        image_header = tk.Frame(left, bg=PANEL_BG)
        image_header.grid(row=0, column=0, sticky="ew")
        image_header.columnconfigure(0, weight=1)
        tk.Label(
            image_header,
            text="图片",
            bg=PANEL_BG,
            fg=TEXT_PRIMARY,
            font=("Microsoft YaHei UI", 12, "bold"),
        ).grid(
            row=0, column=0, sticky="w"
        )
        tk.Label(
            image_header,
            text="打开、粘贴或截图",
            bg=PANEL_BG,
            fg=TEXT_SECONDARY,
            font=("Microsoft YaHei UI", 9),
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        image_toolbar = tk.Frame(left, bg=PANEL_BG)
        image_toolbar.grid(row=1, column=0, sticky="ew", pady=(12, 12))
        RoundedButton(
            image_toolbar,
            text="打开图片",
            command=self.open_image,
            width=112,
        ).pack(
            side=tk.LEFT
        )
        RoundedButton(
            image_toolbar,
            text="粘贴图片",
            command=self.paste_image,
            width=112,
        ).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        RoundedButton(
            image_toolbar,
            text="截图",
            command=self.capture_screen,
            width=84,
        ).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        preview_frame = tk.Frame(
            left,
            bg=SURFACE_SUBTLE,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        preview_frame.grid(row=2, column=0, sticky="nsew")
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        self.preview_label = tk.Label(
            preview_frame,
            text="暂无图片",
            anchor=tk.CENTER,
            bg=SURFACE_SUBTLE,
            fg=TEXT_SECONDARY,
            font=("Microsoft YaHei UI", 12),
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        self.preview_label.bind(
            "<Configure>",
            self._schedule_input_preview_resize,
            add="+",
        )

        output_header = tk.Frame(right, bg=PANEL_BG)
        output_header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        output_header.columnconfigure(0, weight=1)
        tk.Label(
            output_header,
            text="识别结果",
            bg=PANEL_BG,
            fg=TEXT_PRIMARY,
            font=("Microsoft YaHei UI", 12, "bold"),
        ).grid(
            row=0, column=0, sticky="w"
        )
        output_actions = tk.Frame(output_header, bg=PANEL_BG)
        output_actions.grid(row=0, column=1, sticky="e")
        RoundedButton(
            output_actions,
            text="复制LaTeX",
            command=self.copy_latex,
            width=104,
            height=34,
            radius=12,
        ).pack(
            side=tk.LEFT
        )
        RoundedButton(
            output_actions,
            text="复制MathML",
            command=self.copy_mathml,
            width=104,
            height=34,
            radius=12,
        ).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        self.format_menu = tk.Menu(
            self,
            tearoff=0,
            bg="#ffffff",
            fg=TEXT_PRIMARY,
            activebackground=ACCENT_SOFT,
            activeforeground=TEXT_PRIMARY,
            relief=tk.FLAT,
            borderwidth=1,
            font=("Microsoft YaHei UI", 10),
        )
        self.format_menu.add_command(
            label="复制MathML(Word)",
            command=lambda: self.copy_format("mathml"),
        )
        self.format_menu.add_command(
            label="复制AsciiMath",
            command=lambda: self.copy_format("asciimath"),
        )
        self.format_menu.add_command(
            label="复制Typst",
            command=lambda: self.copy_format("typst"),
        )
        self.format_menu.add_separator()
        self.format_menu.add_command(
            label="复制Markdown行内公式",
            command=lambda: self.copy_format("markdown_inline"),
        )
        self.format_menu.add_command(
            label="复制Markdown块公式",
            command=lambda: self.copy_format("markdown_block"),
        )
        self.format_menu.add_command(
            label="复制LaTeX equation环境",
            command=lambda: self.copy_format("equation"),
        )
        self.format_menu.add_command(
            label="复制Word线性公式",
            command=lambda: self.copy_format("word_linear"),
        )
        self.format_menu.add_command(
            label="复制Word OMML",
            command=lambda: self.copy_format("omml"),
        )
        self.format_menu.add_command(
            label="复制HTML(MathML)",
            command=lambda: self.copy_format("html"),
        )
        self.format_menu.add_separator()
        self.format_menu.add_command(
            label="导出Docx(Word/WPS)",
            command=self.export_docx,
        )
        self.format_button = RoundedButton(
            output_actions,
            text="更多格式 ▾",
            command=self.show_format_menu,
            width=112,
            height=34,
            radius=12,
            bg=ACCENT_SOFT,
            active_bg="#fbd3e8",
            fg=ACCENT_DARK,
            border="#f6bfd9",
        )
        self.format_button.pack(side=tk.LEFT, padx=(8, 0))
        RoundedButton(
            output_actions,
            text="清空",
            command=self.clear_output,
            width=72,
            height=34,
            radius=12,
        ).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        results_frame = tk.Frame(right, bg=PANEL_BG)
        results_frame.grid(row=1, column=0, sticky="nsew")
        results_frame.rowconfigure(0, weight=1, uniform="result_sections")
        results_frame.rowconfigure(1, weight=1, uniform="result_sections")
        results_frame.columnconfigure(0, weight=1)

        latex_section = tk.Frame(results_frame, bg=PANEL_BG)
        latex_section.grid(row=0, column=0, sticky="nsew", pady=(0, 12))
        latex_section.rowconfigure(1, weight=1)
        latex_section.columnconfigure(0, weight=1)
        latex_header = tk.Frame(latex_section, bg=PANEL_BG)
        latex_header.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        latex_header.columnconfigure(0, weight=1)
        tk.Label(
            latex_header,
            text="LaTeX 结果",
            bg=PANEL_BG,
            fg=TEXT_PRIMARY,
            font=("Microsoft YaHei UI", 10, "bold"),
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            latex_header,
            text="可编辑",
            bg=PANEL_BG,
            fg=TEXT_SECONDARY,
            font=("Microsoft YaHei UI", 9),
        ).grid(row=0, column=1, sticky="e")

        latex_frame = tk.Frame(
            latex_section,
            bg=PANEL_BG,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        latex_frame.grid(row=1, column=0, sticky="nsew")
        latex_frame.rowconfigure(0, weight=1)
        latex_frame.columnconfigure(0, weight=1)

        self.output_text = tk.Text(
            latex_frame,
            wrap=tk.WORD,
            font=("Consolas", 12),
            undo=True,
            height=1,
            bg="#fbfdff",
            fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY,
            relief=tk.FLAT,
            borderwidth=0,
            padx=12,
            pady=12,
        )
        self.output_text.grid(row=0, column=0, sticky="nsew")
        self.latex_scrollbar = SlimScrollbar(
            latex_frame, command=self.output_text.yview
        )
        self.latex_scrollbar.grid(row=0, column=1, sticky="ns", padx=(2, 4), pady=8)
        self.output_text.configure(yscrollcommand=self.latex_scrollbar.set)

        mathml_section = tk.Frame(results_frame, bg=PANEL_BG)
        mathml_section.grid(row=1, column=0, sticky="nsew")
        mathml_section.rowconfigure(1, weight=1)
        mathml_section.columnconfigure(0, weight=1)
        mathml_header = tk.Frame(mathml_section, bg=PANEL_BG)
        mathml_header.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        mathml_header.columnconfigure(0, weight=1)
        tk.Label(
            mathml_header,
            text="MathML 公式展示",
            bg=PANEL_BG,
            fg=TEXT_PRIMARY,
            font=("Microsoft YaHei UI", 10, "bold"),
        ).grid(row=0, column=0, sticky="w")
        RoundedButton(
            mathml_header,
            text="刷新预览",
            command=self.refresh_mathml_preview,
            width=92,
            height=30,
            radius=11,
        ).grid(row=0, column=1, sticky="e", padx=(8, 0))

        mathml_frame = tk.Frame(
            mathml_section,
            bg="#fbfdff",
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        mathml_frame.grid(row=1, column=0, sticky="nsew")
        mathml_frame.rowconfigure(0, weight=1)
        mathml_frame.columnconfigure(0, weight=1)

        self.mathml_preview_label = tk.Label(
            mathml_frame,
            text="暂无公式预览",
            anchor=tk.CENTER,
            justify=tk.CENTER,
            bg="#fbfdff",
            fg=TEXT_SECONDARY,
            font=("Cambria Math", 16),
        )
        self.mathml_preview_label.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        self.mathml_preview_label.bind(
            "<Configure>",
            self._schedule_mathml_preview_resize,
            add="+",
        )
        self.output_text.bind("<<Modified>>", self._on_latex_modified)
        self.output_text.edit_modified(False)

        status_frame = tk.Frame(self, bg=APP_BG)
        status_frame.grid(row=2, column=0, sticky="ew", padx=22, pady=(0, 14))
        status_frame.columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="就绪")
        status_label = tk.Label(
            status_frame,
            textvariable=self.status_var,
            anchor=tk.W,
            bg=APP_BG,
            fg=TEXT_SECONDARY,
            font=("Microsoft YaHei UI", 9),
        )
        status_label.grid(row=0, column=0, sticky="ew")
        self.busy_progress = ttk.Progressbar(
            status_frame,
            mode="indeterminate",
            length=180,
            style="Status.Horizontal.TProgressbar",
        )
        self.cancel_download_button = RoundedButton(
            status_frame,
            text="取消下载",
            command=self._cancel_download,
            width=88,
            height=28,
            radius=10,
            bg="#fff4df",
            active_bg="#ffe8b8",
            fg="#8a5a13",
            border="#f2d39a",
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        self.cancel_download_button.grid(row=0, column=1, sticky="e", padx=(12, 0))
        self.cancel_download_button.grid_remove()
        self.busy_progress.grid(row=0, column=2, sticky="e", padx=(12, 0))
        self.busy_progress.grid_remove()

    def _bind_shortcuts(self) -> None:
        self.bind("<Control-o>", lambda _event: self.open_image())
        self.bind("<Control-v>", self._paste_shortcut)
        # Text's class binding normally handles Ctrl+V before the toplevel
        # binding.  Intercept it at the widget level so a copied image file
        # path is not inserted into the editable LaTeX result first.
        self.output_text.bind("<Control-v>", self._paste_shortcut, add="+")
        self.bind("<Control-Return>", lambda _event: self.recognize_image())
        self.bind("<Control-c>", self._copy_shortcut)

    def _paste_shortcut(self, _event: tk.Event) -> str | None:
        handled, _loaded = self._paste_image_from_clipboard(
            notify_if_missing=False
        )
        return "break" if handled else None

    def _copy_shortcut(self, event: tk.Event) -> str | None:
        if self.focus_get() is self.output_text:
            return None
        self.copy_latex()
        return "break"

    def _on_model_changed(self, _event: tk.Event | None = None) -> None:
        model_id = self._selected_model_id()
        if not model_id or not is_model_cached(model_id, verify_hash=False):
            self.model_picker.set("")
            self._update_model_summary()
            self.status_var.set("请选择一个已下载模型")
            return
        self._reset_recognizer()
        self._save_preferences()
        self._update_model_summary()
        self.status_var.set(f"已选择 {get_model_spec(model_id).display_name}（已下载）")

    def _selected_model_id(self) -> str:
        return self.model_picker.get()

    def _save_preferences(self) -> None:
        try:
            save_settings(
                AppSettings(
                    model_id=self._selected_model_id(),
                    accepted_model_terms=tuple(sorted(self.accepted_model_terms)),
                )
            )
        except OSError as exc:
            write_log(f"Unable to save settings: {exc}")

    @staticmethod
    def _model_terms_key(model_id: str) -> str:
        spec = get_model_spec(model_id)
        return f"{model_id}:{spec.terms_revision or 'current'}"

    def _ensure_model_terms_accepted(self, model_id: str) -> bool:
        """Ask once before downloading/using a model with extra terms."""

        spec = get_model_spec(model_id)
        if not spec.requires_terms_ack:
            return True
        key = self._model_terms_key(model_id)
        if key in self.accepted_model_terms:
            return True
        message = (
            f"{spec.display_name} 的上游使用条款需要单独确认。\n\n"
            f"许可：{spec.license_label}\n"
            f"{spec.usage_restriction}\n\n"
            "FormulaOCR 只会从上游 release 下载模型，不会把它写入 _internal。"
            "继续表示你已阅读并接受按上游条款使用该模型。\n\n"
            f"条款页面：{spec.terms_url}"
        )
        accepted = messagebox.askyesno(
            "确认模型使用条款",
            message,
            parent=self,
        )
        if not accepted:
            self.status_var.set("未确认模型条款，已取消使用")
            return False
        self.accepted_model_terms.add(key)
        self._save_preferences()
        return True

    def _update_model_summary(self) -> None:
        model_id = self._selected_model_id()
        cached = bool(
            model_id
            and model_id in MODEL_BY_ID
            and is_model_cached(model_id, verify_hash=False)
        )
        if not cached:
            if model_id:
                self.model_picker.set("")
            self.model_info_var.set(
                "请先在“模型管理”下载模型，再从主界面下拉框选择"
            )
            self.model_picker.refresh()
            return
        spec = get_model_spec(model_id)
        if cached and is_model_bundled_only(model_id):
            state = "随包内置"
        else:
            state = "已下载"
        terms = " · 使用前需确认上游条款" if spec.requires_terms_ack else ""
        self.model_info_var.set(
            f"{spec.model_id} · {spec.best_for} · {spec.languages} · {state}{terms}"
        )
        self.model_picker.refresh()

    def show_model_manager(self) -> None:
        manager_exists = False
        if self.model_manager_window is not None:
            try:
                manager_exists = bool(self.model_manager_window.winfo_exists())
            except tk.TclError:
                self.model_manager_window = None
        if manager_exists:
            assert self.model_manager_window is not None
            self.model_manager_window.deiconify()
            self.model_manager_window.lift()
            self.model_manager_window.focus_force()
            return
        try:
            from formula_ocr_app.model_manager_dialog import (
                show_model_manager_dialog,
            )
        except ModuleNotFoundError as exc:  # Direct script execution.
            if exc.name != "formula_ocr_app":
                raise
            from model_manager_dialog import show_model_manager_dialog

        window = show_model_manager_dialog(
            self,
            current_model_id=self._selected_model_id,
            can_mutate_models=lambda: not self.is_busy and not self.is_destroying,
            request_download=self._request_model_download,
            before_remove=self._reset_recognizer,
            on_models_changed=self._update_model_summary,
            set_status=self.status_var.set,
            open_model_cache=self._open_model_cache,
            show_runtime_info=self.show_runtime_info,
        )
        self.model_manager_window = window

        def clear_reference(event: tk.Event) -> None:
            if event.widget is window:
                self.model_manager_window = None

        window.bind("<Destroy>", clear_reference, add="+")

    def _request_model_download(self, model_id: str) -> bool:
        if self.is_busy or self.is_destroying:
            return False
        if not self._ensure_model_terms_accepted(model_id):
            return False
        spec = get_model_spec(model_id)
        if is_model_cached(model_id):
            self.status_var.set(f"{spec.display_name} 已下载")
            return False
        self._set_busy(
            True,
            message=f"正在下载 {spec.display_name}...",
            show_cancel=True,
        )
        thread = threading.Thread(
            target=self._prepare_model_worker,
            args=(model_id,),
            daemon=True,
        )
        self.model_download_thread = thread
        thread.start()
        return True

    def _open_model_cache(self) -> None:
        directory = runtime_cache_dir()
        directory.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(directory)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(directory)])
        except OSError as exc:
            messagebox.showerror("无法打开目录", str(exc))

    @staticmethod
    def _bundled_runtime_dir() -> Path:
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        if getattr(sys, "frozen", False):
            return Path(sys.executable).resolve().parent / "_internal"
        return APP_ROOT.parent / "_internal"

    @staticmethod
    def _module_status(module_name: str) -> str:
        try:
            details = _probe_runtime_component(module_name)
        except Exception as exc:
            write_log(
                f"Runtime component probe failed for {module_name}: "
                f"{type(exc).__name__}: {exc}"
            )
            return "不可用（原生组件加载失败）"
        return f"可用（{details}）" if details else "可用"

    def show_runtime_info(self) -> None:
        window = tk.Toplevel(self)
        window.title("运行环境与缓存")
        window.geometry("700x520")
        window.minsize(620, 440)
        window.configure(bg=APP_BG)
        window.transient(self)
        window.columnconfigure(0, weight=1)
        window.rowconfigure(1, weight=1)

        header = tk.Frame(window, bg=APP_BG)
        header.grid(row=0, column=0, sticky="ew", padx=22, pady=(18, 12))
        tk.Label(
            header,
            text="运行环境与缓存边界",
            bg=APP_BG,
            fg=TEXT_PRIMARY,
            font=("Microsoft YaHei UI", 17, "bold"),
        ).pack(anchor=tk.W)
        tk.Label(
            header,
            text="程序运行库和用户模型是两类不同资源，下载器不会覆盖随包运行库。",
            bg=APP_BG,
            fg=TEXT_SECONDARY,
            font=("Microsoft YaHei UI", 9),
        ).pack(anchor=tk.W, pady=(3, 0))

        body = tk.Frame(window, bg=PANEL_BG, highlightbackground=BORDER, highlightthickness=1)
        body.grid(row=1, column=0, sticky="nsew", padx=22, pady=(0, 12))
        body.columnconfigure(1, weight=1)

        runtime_dir = self._bundled_runtime_dir()
        rows = (
            ("运行模式", "Windows 打包版" if getattr(sys, "frozen", False) else "源码运行"),
            ("随包运行库 _internal", str(runtime_dir)),
            ("用户模型缓存", str(runtime_cache_dir())),
            ("模型目录", f"{len(MODEL_SPECS)} 个模型 / {len({item.provider for item in MODEL_SPECS})} 个供应方"),
            (
                "Paddle Inference",
                self._module_status("paddle-native"),
            ),
            ("ONNX Runtime", self._module_status("onnxruntime")),
            (
                "ONNX 模型后端",
                "RapidLaTeXOCR / MathCraft / Pix2Text / MixTeX / UniMERNet Small 已按需加载；"
                + self._module_status("tokenizers"),
            ),
        )
        for row_index, (label, value) in enumerate(rows):
            tk.Label(
                body,
                text=label,
                bg=PANEL_BG,
                fg=TEXT_SECONDARY,
                font=("Microsoft YaHei UI", 9, "bold"),
                anchor=tk.W,
            ).grid(row=row_index, column=0, sticky="nw", padx=(16, 12), pady=(14 if row_index == 0 else 7, 0))
            tk.Label(
                body,
                text=value,
                bg=PANEL_BG,
                fg=TEXT_PRIMARY,
                font=("Microsoft YaHei UI", 9),
                anchor=tk.W,
                justify=tk.LEFT,
                wraplength=450,
            ).grid(row=row_index, column=1, sticky="ew", padx=(0, 16), pady=(14 if row_index == 0 else 7, 0))

        note = tk.Label(
            body,
            text=(
                "说明：_internal 中的 python*.dll、*.pyd、Paddle/ONNX Runtime 和 Tk 资源是\n"
                "PyInstaller 程序本体的一部分，必须与 EXE 一起分发；运行时模型、断点文件和校验后的\n"
                "模型只写入“用户模型缓存”。若需要离线模型，可在打包时显式放入 _internal/models，\n"
                "应用会把它们当作只读随包资源。"
            ),
            bg="#f5f8fc",
            fg=TEXT_SECONDARY,
            font=("Microsoft YaHei UI", 9),
            justify=tk.LEFT,
            anchor=tk.W,
            padx=14,
            pady=12,
        )
        note.grid(row=len(rows), column=0, columnspan=2, sticky="ew", padx=12, pady=(14, 14))

        actions = tk.Frame(window, bg=APP_BG)
        actions.grid(row=2, column=0, sticky="e", padx=22, pady=(0, 18))
        ttk.Button(actions, text="打开用户缓存", command=self._open_model_cache).pack(side=tk.LEFT)
        ttk.Button(actions, text="关闭", command=window.destroy).pack(side=tk.LEFT, padx=(8, 0))

    def _prepare_model_worker(self, model_id: str) -> None:
        try:
            if self.download_cancel_event.is_set():
                raise ModelDownloadCancelled()
            ensure_model(
                model_id,
                progress_callback=self._queue_model_download_progress,
            )
            self.worker_queue.put(("model_ready", model_id))
        except ModelDownloadCancelled:
            # Cancellation can race with the final atomic install. If the
            # selected model is already complete, report success instead of
            # telling the user a finished download was cancelled.
            if is_model_cached(model_id):
                self.worker_queue.put(("model_ready", model_id))
            else:
                self.worker_queue.put(("download_cancelled", model_id))
        except Exception as exc:
            details = "".join(traceback.format_exception(exc)).strip()
            write_log(f"Model download failed for {model_id}\n{details}")
            message = str(exc).strip() or type(exc).__name__
            self.worker_queue.put(("error", message))

    def open_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="选择公式图片",
            filetypes=[
                ("图片文件", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"),
                ("所有文件", "*.*"),
            ],
        )
        if not file_path:
            return
        try:
            image = load_rgb_image(file_path)
            if self._set_image(image):
                self.status_var.set(f"已加载图片：{file_path}")
        except Exception as exc:
            messagebox.showerror("打开失败", str(exc))

    def paste_image(self, *, notify_if_missing: bool = True) -> bool:
        _handled, loaded = self._paste_image_from_clipboard(
            notify_if_missing=notify_if_missing
        )
        return loaded

    def _paste_image_from_clipboard(
        self,
        *,
        notify_if_missing: bool,
    ) -> tuple[bool, bool]:
        """Return ``(handled, loaded)`` for image-oriented clipboard data.

        A copied file list must be consumed even when its temporary image has
        already disappeared. Otherwise Tk's Text binding inserts the stale
        filesystem path into the LaTeX result after this handler returns.
        """

        try:
            data = ImageGrab.grabclipboard()
        except Exception as exc:
            messagebox.showerror("粘贴失败", str(exc), parent=self)
            return True, False

        if isinstance(data, Image.Image):
            if self._set_image(data):
                self.status_var.set("已从剪贴板加载图片")
                return True, True
            return True, False

        # PIL normally returns a list for CF_HDROP, but some clipboard
        # providers expose one dropped file as a bare string. Treat both as
        # image-oriented clipboard data so the path cannot reach Text's
        # default paste binding.
        if isinstance(data, (str, Path)):
            data = [data]
        if isinstance(data, (list, tuple)) and data:
            for item in data:
                try:
                    image = load_rgb_image(item)
                except (OSError, ValueError, TypeError):
                    continue
                if self._set_image(image):
                    self.status_var.set(f"已从剪贴板文件加载图片：{item}")
                    return True, True
                # A valid file may still fail to become the current input
                # (for example, if the image is unreadable while being
                # replaced). Continue looking for another usable clipboard
                # item instead of letting the first bad entry hide later ones.
                continue

            self.status_var.set("剪贴板中的图片文件已失效或无法读取")
            if notify_if_missing:
                messagebox.showinfo(
                    "图片不可用",
                    "剪贴板中的文件不是可读取图片，或临时图片已经被清理。",
                    parent=self,
                )
            return True, False

        if notify_if_missing:
            messagebox.showinfo(
                "没有图片",
                "剪贴板里没有可用的图片。",
                parent=self,
            )
        return False, False

    def capture_screen(self) -> None:
        if self.capture_after_id is not None or self.screenshot_selector is not None:
            return
        self.withdraw()
        self.capture_after_id = self.after(180, self._start_capture_overlay)

    def _start_capture_overlay(self) -> None:
        self.capture_after_id = None
        if self.is_destroying:
            return
        try:
            selector = ScreenshotSelector(self, self._on_screen_captured)
            self.screenshot_selector = selector
            selector.start()
        except Exception as exc:
            self.screenshot_selector = None
            self.deiconify()
            self.lift()
            write_log(f"Unable to start screenshot selector: {exc}")
            messagebox.showerror("截图失败", str(exc), parent=self)

    def _on_screen_captured(self, image: Image.Image | None) -> None:
        self.screenshot_selector = None
        if self.is_destroying:
            return
        self.deiconify()
        self.lift()
        if image is None:
            self.status_var.set("截图已取消")
            return
        if self._set_image(image):
            self.status_var.set("已截取图片")

    def recognize_image(self) -> None:
        if self.is_busy:
            return
        if self.current_image_path is None:
            messagebox.showinfo("没有图片", "请先打开、粘贴或截图一张公式图片。")
            return

        model_id = self._selected_model_id()
        if not model_id or not is_model_cached(model_id, verify_hash=False):
            if model_id:
                self.model_picker.set("")
                self._update_model_summary()
            messagebox.showinfo(
                "没有可用模型",
                "请先在“模型管理”下载模型，再从主界面下拉框选择。",
                parent=self,
            )
            self.status_var.set("尚未选择已下载模型")
            return
        settings = self._current_settings()
        if not self._ensure_model_terms_accepted(settings.model_name):
            return
        self._set_busy(
            True,
            message="正在加载已缓存模型并识别公式...",
            show_cancel=False,
        )
        revision = self.image_revision
        thread = threading.Thread(
            target=self._recognize_worker,
            args=(self.current_image_path, settings, revision),
            daemon=True,
        )
        self.recognition_thread = thread
        thread.start()

    def copy_latex(self) -> None:
        latex = self._current_latex()
        if not latex:
            self.status_var.set("没有可复制的 LaTeX")
            return
        try:
            self._copy_text(latex)
        except tk.TclError as exc:
            write_log(f"Failed to copy LaTeX: {exc}")
            messagebox.showerror("复制失败", "剪贴板暂时不可用，请稍后重试。", parent=self)
            self.status_var.set("复制失败")
            return
        self.status_var.set("LaTeX 已复制到剪贴板")

    def copy_mathml(self) -> None:
        self.copy_format("mathml")

    def show_format_menu(self) -> None:
        self.update_idletasks()
        x, y, _width, _height = _anchored_popup_geometry(
            self.format_button,
            self.format_menu.winfo_reqwidth(),
            self.format_menu.winfo_reqheight(),
            gap=4,
            align="left",
        )
        try:
            self.format_menu.tk_popup(x, y)
        finally:
            self.format_menu.grab_release()

    def copy_format(self, fmt: str) -> None:
        latex = self._current_latex()
        if not latex:
            self.status_var.set("没有可转换的 LaTeX")
            return
        try:
            if fmt == "mathml":
                mathml = latex_to_mathml(latex)
                value = mathml_to_word_mathml(mathml)
                label = "MathML(Word)"
                rich_copied = self._copy_mathml_for_word(mathml, plain_text=value)
                if rich_copied:
                    self.status_var.set(f"{label} 已复制到剪贴板")
                else:
                    self.status_var.set(f"{label} 富格式复制失败，已复制纯文本")
                return
            elif fmt == "asciimath":
                value = latex_to_asciimath(latex)
                label = "AsciiMath"
            elif fmt == "typst":
                value = latex_to_typst(latex)
                label = "Typst"
            elif fmt == "markdown_inline":
                value = latex_to_markdown_inline(latex)
                label = "Markdown 行内公式"
            elif fmt == "markdown_block":
                value = latex_to_markdown_block(latex)
                label = "Markdown 块公式"
            elif fmt == "equation":
                value = latex_to_equation_environment(latex)
                label = "LaTeX equation 环境"
            elif fmt == "word_linear":
                value = latex_to_word_linear(latex)
                label = "Word 线性公式"
            elif fmt == "omml":
                value = mathml_to_omml(latex_to_mathml(latex))
                label = "Word OMML"
            elif fmt == "html":
                value = latex_to_html(latex)
                label = "HTML(MathML)"
            else:
                raise ValueError(f"Unknown format: {fmt}")
        except tk.TclError as exc:
            write_log(f"Failed to copy {fmt}: {exc}")
            messagebox.showerror("复制失败", "剪贴板暂时不可用，请稍后重试。", parent=self)
            self.status_var.set("复制失败")
            return
        except Exception as exc:
            messagebox.showerror("转换失败", str(exc))
            self.status_var.set("转换失败")
            return
        try:
            self._copy_text(value)
        except tk.TclError as exc:
            write_log(f"Failed to copy {label}: {exc}")
            messagebox.showerror("复制失败", "剪贴板暂时不可用，请稍后重试。", parent=self)
            self.status_var.set("复制失败")
            return
        self.status_var.set(f"{label} 已复制到剪贴板")

    def export_docx(self) -> None:
        latex = self._current_latex()
        if not latex:
            self.status_var.set("没有可导出的 LaTeX")
            return
        file_path = filedialog.asksaveasfilename(
            title="导出 Docx",
            defaultextension=".docx",
            initialfile="formula_result.docx",
            filetypes=[("Word 文档", "*.docx"), ("所有文件", "*.*")],
        )
        if not file_path:
            return
        try:
            mathml = latex_to_mathml(latex)
            asciimath = latex_to_asciimath(latex)
            typst = latex_to_typst(latex)
            word_linear = latex_to_word_linear(latex)
            export_formula_docx(
                file_path,
                latex=latex,
                mathml=mathml,
                asciimath=asciimath,
                typst=typst,
                word_linear=word_linear,
                image_path=self.current_image_path,
            )
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))
            self.status_var.set("导出失败")
            return
        self.status_var.set(f"Docx 已导出：{file_path}")

    def _current_latex(self) -> str:
        return self.output_text.get("1.0", tk.END).strip()

    def _copy_text(self, text: str) -> None:
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update_idletasks()

    def _copy_mathml_for_word(self, mathml: str, *, plain_text: str) -> bool:
        if copy_mathml_for_word_to_clipboard(
            mathml,
            plain_text=plain_text,
            clipboard_widget=self,
            owner_hwnd=self.winfo_id(),
        ):
            self.update_idletasks()
            return True
        self._copy_text(plain_text)
        return False

    def refresh_mathml_preview(self) -> None:
        if self.mathml_update_after_id is not None:
            self.after_cancel(self.mathml_update_after_id)
            self.mathml_update_after_id = None
        self._update_mathml_preview()
        self.status_var.set("正在刷新 MathML 预览")

    def _on_latex_modified(self, _event: tk.Event) -> None:
        if not self.output_text.edit_modified():
            return
        self.output_text.edit_modified(False)
        self._schedule_mathml_preview_update()

    def _schedule_mathml_preview_update(self) -> None:
        if self.is_destroying:
            return
        if self.mathml_update_after_id is not None:
            self.after_cancel(self.mathml_update_after_id)
        self.mathml_update_after_id = self.after(450, self._update_mathml_preview)

    def _update_mathml_preview(self) -> None:
        self.mathml_update_after_id = None
        self.mathml_render_token += 1
        token = self.mathml_render_token
        latex = self._current_latex()
        if not latex:
            self._cancel_pending_mathml_render()
            self._set_mathml_preview_text("暂无公式预览")
            return
        try:
            mathml = latex_to_mathml(latex)
        except Exception as exc:
            self._cancel_pending_mathml_render()
            write_log(f"Failed to convert LaTeX to MathML: {exc}")
            self._set_mathml_preview_text("MathML 转换失败")
            return
        self._set_mathml_preview_text("正在渲染 MathML...")
        self._queue_mathml_render(token, latex, mathml)

    def _set_mathml_preview_text(self, text: str) -> None:
        self.mathml_preview_source = None
        self.mathml_preview_render_size = None
        self.mathml_preview_photo = None
        self.mathml_preview_label.configure(image="", text=text, fg=TEXT_SECONDARY)

    def _replace_output_text(self, text: str, *, reset_undo: bool = True) -> None:
        """Programmatically replace LaTeX without scheduling a second preview."""

        self.output_text.edit_modified(False)
        self.output_text.delete("1.0", tk.END)
        if text:
            self.output_text.insert("1.0", text)
        if reset_undo:
            self.output_text.edit_reset()
        self.output_text.edit_modified(False)

    def clear_output(self, *, update_status: bool = True) -> None:
        if self.mathml_update_after_id is not None:
            self.after_cancel(self.mathml_update_after_id)
            self.mathml_update_after_id = None
        self._replace_output_text("")
        self.mathml_render_token += 1
        self._cancel_pending_mathml_render()
        self._set_mathml_preview_text("暂无公式预览")
        if update_status:
            self.status_var.set("结果已清空")

    def _recognize_worker(
        self,
        image_path: Path,
        settings: RecognizerSettings,
        image_revision: int,
    ) -> None:
        start = time.time()
        try:
            recognizer = self._get_recognizer(settings)
            formula = recognizer.predict(image_path)
            elapsed = time.time() - start
            self.worker_queue.put(
                (
                    "success",
                    {
                        "formula": formula,
                        "elapsed": elapsed,
                        "image_revision": image_revision,
                    },
                )
            )
        except ModelDownloadCancelled:
            self.worker_queue.put(
                (
                    "recognition_cancelled",
                    {
                        "model_name": settings.model_name,
                        "image_revision": image_revision,
                    },
                )
            )
        except Exception as exc:
            details = "".join(traceback.format_exception(exc)).strip()
            write_log(
                f"Recognition failed for {settings.model_name} on {image_path}\n"
                f"{details}"
            )
            self.worker_queue.put(
                (
                    "recognition_error",
                    {
                        "message": str(exc).strip() or type(exc).__name__,
                        "image_revision": image_revision,
                    },
                )
            )
        finally:
            self._session_worker_finished("recognition_thread")

    def _schedule_worker_poll(self) -> None:
        if self.is_destroying:
            return
        self.worker_poll_after_id = self.after(100, self._poll_worker_queue)

    def _poll_worker_queue(self) -> None:
        self.worker_poll_after_id = None
        if self.is_destroying:
            return
        try:
            kind, payload = self.worker_queue.get_nowait()
        except queue.Empty:
            self._schedule_worker_poll()
            return
        while kind in {"progress", "download_progress"}:
            try:
                kind, payload = self.worker_queue.get_nowait()
            except queue.Empty:
                break

        if kind == "progress":
            self.busy_status_message = str(payload)
            started_at = self.busy_started_at or time.time()
            elapsed = time.time() - started_at
            self.status_var.set(f"{self.busy_status_message} {elapsed:.1f}s")
            self._schedule_worker_poll()
            return

        if kind == "download_progress":
            if self.download_cancel_event.is_set():
                # Progress messages already queued before cancellation must
                # not re-enable the button or replace the cancellation state.
                self.cancel_download_button.set_disabled(True)
                self.busy_status_message = "正在取消下载，保留已下载进度..."
                self.status_var.set(self.busy_status_message)
                self._schedule_worker_poll()
                return
            if isinstance(payload, dict):
                self.cancel_download_button.grid()
                self.cancel_download_button.set_disabled(False)
                self.busy_status_message = str(payload.get("message", "正在下载模型..."))
                try:
                    percent = float(payload.get("percent", 0.0))
                except (TypeError, ValueError):
                    percent = 0.0
                self.busy_progress.stop()
                self.busy_progress.configure(
                    mode="determinate",
                    maximum=100.0,
                    value=max(0.0, min(100.0, percent)),
                )
            self._schedule_worker_poll()
            return

        if kind == "model_ready":
            self.model_download_thread = None
            self._set_busy(False)
            model_id = str(payload)
            self._update_model_summary()
            self.status_var.set(
                f"{get_model_spec(model_id).display_name} 下载完成；"
                "请在主界面模型下拉框中选择"
            )
            self._schedule_worker_poll()
            return

        if kind == "download_cancelled":
            self.model_download_thread = None
            self._set_busy(False)
            self._update_model_summary()
            self.status_var.set("下载已取消，已保留断点；下次可继续下载")
            self._schedule_worker_poll()
            return

        if kind == "error":
            self.model_download_thread = None
        if kind in {"success", "recognition_cancelled", "recognition_error"}:
            self.recognition_thread = None
            self._cleanup_obsolete_input_images()
        if kind not in {"success", "recognition_cancelled", "recognition_error", "error"}:
            write_log(f"Unknown worker message {kind!r}: {payload!r}")
            self._schedule_worker_poll()
            return
        self._set_busy(False)
        if kind == "recognition_cancelled":
            self.status_var.set("识别已取消")
            self._schedule_worker_poll()
            return

        if kind in {"success", "recognition_error"}:
            if not isinstance(payload, dict):
                raise TypeError("Unexpected recognition worker payload.")
            revision = int(payload.get("image_revision", -1))
            if revision != self.image_revision:
                self.status_var.set("输入图片已更改，已忽略旧图片的识别结果")
                self._schedule_worker_poll()
                return

        if kind == "success":
            formula = str(payload.get("formula", ""))
            elapsed = float(payload.get("elapsed", 0.0))
            self._replace_output_text(formula)
            self._update_mathml_preview()
            self.status_var.set(f"识别完成，用时 {elapsed:.2f} 秒；结果已保留")
        elif kind == "recognition_error":
            messagebox.showerror(
                "识别失败",
                str(payload.get("message", "识别失败")),
                parent=self,
            )
            self.status_var.set("识别失败")
        elif kind == "error":
            self._update_model_summary()
            messagebox.showerror("模型下载失败", str(payload), parent=self)
            self.status_var.set("模型下载失败")

        self._schedule_worker_poll()

    def _schedule_mathml_preview_poll(self) -> None:
        if self.is_destroying:
            return
        self.mathml_preview_poll_after_id = self.after(
            120,
            self._poll_mathml_preview_queue,
        )

    def _poll_mathml_preview_queue(self) -> None:
        self.mathml_preview_poll_after_id = None
        if self.is_destroying:
            return
        try:
            while True:
                token, kind, payload = self.mathml_preview_queue.get_nowait()
                if token != self.mathml_render_token:
                    if kind == "image":
                        _safe_unlink_temporary_file(
                            Path(payload),
                            context="stale MathML preview",
                        )
                    continue
                if kind == "image":
                    self._set_mathml_preview_image(Path(payload))
                elif kind == "text":
                    self._set_mathml_preview_text(payload)
                else:
                    self._set_mathml_preview_text("MathML 预览不可用")
                    write_log(f"MathML preview render failed: {payload}")
        except queue.Empty:
            pass
        self._schedule_mathml_preview_poll()

    def _queue_mathml_render(self, token: int, latex: str, mathml: str) -> None:
        with self.mathml_render_lock:
            if self.mathml_active_cancel_event is not None:
                self.mathml_active_cancel_event.set()
            self.mathml_pending_render = (token, latex, mathml)
            if (
                self.mathml_render_thread is not None
                and self.mathml_render_thread.is_alive()
            ):
                return
            thread = threading.Thread(
                target=self._mathml_render_loop,
                daemon=True,
            )
            self.mathml_render_thread = thread
            thread.start()

    def _cancel_pending_mathml_render(self) -> None:
        with self.mathml_render_lock:
            self.mathml_pending_render = None
            if self.mathml_active_cancel_event is not None:
                self.mathml_active_cancel_event.set()

    def _mathml_render_loop(self) -> None:
        current_thread = threading.current_thread()
        try:
            while True:
                with self.mathml_render_lock:
                    job = self.mathml_pending_render
                    self.mathml_pending_render = None
                    if job is None:
                        if self.mathml_render_thread is current_thread:
                            self.mathml_render_thread = None
                        return
                    cancel_event = threading.Event()
                    self.mathml_active_cancel_event = cancel_event
                try:
                    self._render_mathml_preview_worker(*job, cancel_event)
                finally:
                    with self.mathml_render_lock:
                        if self.mathml_active_cancel_event is cancel_event:
                            self.mathml_active_cancel_event = None
        finally:
            self._session_worker_finished("mathml_render_thread")

    def _render_mathml_preview_worker(
        self,
        token: int,
        latex: str,
        mathml: str,
        cancel_event: threading.Event,
    ) -> None:
        try:
            image_path = self._render_mathml_to_png(token, mathml, cancel_event)
        except Exception as exc:
            if cancel_event.is_set():
                return
            fallback = latex_to_word_linear(latex) or latex
            self.mathml_preview_queue.put((token, "text", fallback))
            write_log(f"MathML browser render fallback: {exc}")
            return
        self.mathml_preview_queue.put((token, "image", str(image_path)))

    def _render_mathml_to_png(
        self,
        token: int,
        mathml: str,
        cancel_event: threading.Event,
    ) -> Path:
        try:
            from formula_ocr_app.mathml_preview import render_mathml_to_png
        except ModuleNotFoundError as exc:  # Direct script execution.
            if exc.name != "formula_ocr_app":
                raise
            from mathml_preview import render_mathml_to_png

        return render_mathml_to_png(
            token,
            mathml,
            cache_dir=self.session_dir,
            cancel_event=cancel_event,
        )

    def _set_mathml_preview_image(self, image_path: Path) -> None:
        try:
            with Image.open(image_path) as image:
                self.mathml_preview_source = image.convert("RGBA")
            self.mathml_preview_render_size = None
            self._update_mathml_preview_image()
        except Exception as exc:
            self._set_mathml_preview_text("MathML 预览加载失败")
            write_log(f"Failed to load MathML preview image: {exc}")
            return
        finally:
            _safe_unlink_temporary_file(
                image_path,
                context="MathML preview",
            )
        self.mathml_preview_label.configure(
            image=self.mathml_preview_photo,
            text="",
            bg="#fbfdff",
        )

    def _schedule_mathml_preview_resize(self, _event: tk.Event | None = None) -> None:
        if (
            self.is_destroying
            or self.mathml_preview_source is None
            or self.mathml_preview_resize_after_id is not None
        ):
            return
        self.mathml_preview_resize_after_id = self.after_idle(
            self._update_mathml_preview_image
        )

    def _update_mathml_preview_image(self) -> None:
        self.mathml_preview_resize_after_id = None
        source = self.mathml_preview_source
        if source is None or self.is_destroying:
            return
        max_width = max(220, self.mathml_preview_label.winfo_width() - 24)
        max_height = max(160, self.mathml_preview_label.winfo_height() - 24)
        render_size = (max_width, max_height)
        if self.mathml_preview_render_size == render_size:
            return
        preview = source.copy()
        preview.thumbnail(render_size, Image.Resampling.LANCZOS)
        self.mathml_preview_photo = ImageTk.PhotoImage(preview)
        self.mathml_preview_render_size = render_size
        self.mathml_preview_label.configure(
            image=self.mathml_preview_photo,
            text="",
            bg="#fbfdff",
        )


    def _set_busy(
        self,
        busy: bool,
        *,
        message: str | None = None,
        show_cancel: bool = False,
    ) -> None:
        self.is_busy = busy
        self.recognize_button.set_disabled(busy)
        self.model_picker.set_disabled(busy)
        self.model_manager_button.set_disabled(busy)
        if self.model_manager_window is not None:
            try:
                if self.model_manager_window.winfo_exists():
                    self.model_manager_window.event_generate(
                        "<<FormulaOCRModelMutationStateChanged>>",
                        when="tail",
                    )
            except tk.TclError:
                self.model_manager_window = None
        if busy:
            self.download_cancel_event.clear()
            if show_cancel:
                self.cancel_download_button.grid()
                self.cancel_download_button.set_disabled(False)
            else:
                self.cancel_download_button.grid_remove()
            if message:
                self.busy_status_message = message
            self._start_busy_feedback()
        else:
            self._stop_busy_feedback()

    def _cancel_download(self) -> None:
        if not self.is_busy:
            return
        self.download_cancel_event.set()
        self.cancel_download_button.set_disabled(True)
        self.busy_status_message = "正在取消下载，保留已下载进度..."
        self.status_var.set(self.busy_status_message)

    def _start_busy_feedback(self) -> None:
        self.busy_started_at = time.time()
        self.busy_progress.grid()
        self.busy_progress.configure(mode="indeterminate", value=0.0)
        self.busy_progress.start(12)
        self._update_busy_status()

    def _stop_busy_feedback(self) -> None:
        if self.busy_status_after_id is not None:
            try:
                self.after_cancel(self.busy_status_after_id)
            except tk.TclError:
                pass
            self.busy_status_after_id = None
        self.busy_started_at = None
        self.busy_progress.stop()
        self.busy_progress.configure(mode="indeterminate", value=0.0)
        self.busy_progress.grid_remove()
        self.cancel_download_button.grid_remove()
        self.cancel_download_button.set_disabled(False)

    def _update_busy_status(self) -> None:
        if not self.is_busy:
            return
        started_at = self.busy_started_at or time.time()
        elapsed = time.time() - started_at
        self.status_var.set(f"{self.busy_status_message} {elapsed:.1f}s")
        self.busy_status_after_id = self.after(300, self._update_busy_status)

    def _get_recognizer(
        self, settings: RecognizerSettings
    ) -> FormulaRecognizer:
        recognizer_to_close: FormulaRecognizer | None = None
        with self.recognizer_lock:
            if self.is_destroying:
                raise RuntimeError("程序正在关闭，已取消本次识别。")
            if self.recognizer is not None and self.recognizer_settings == settings:
                return self.recognizer
            recognizer_to_close = self.recognizer
            recognizer = FormulaRecognizer(
                model_name=settings.model_name,
                device="cpu",
                model_load_callback=self._queue_model_load_status,
                model_download_progress_callback=self._reject_implicit_model_download,
            )
            self.recognizer_settings = settings
            self.recognizer = recognizer
        if recognizer_to_close is not None:
            recognizer_to_close.close()
        return recognizer

    def _queue_model_load_status(self, model_name: str, cached: bool) -> None:
        if cached:
            message = f"正在加载已缓存模型 {get_model_spec(model_name).display_name} 并识别公式..."
            self.worker_queue.put(("progress", message))
        else:
            raise ModelDownloadError(
                f"{get_model_spec(model_name).display_name} 的本地缓存已不存在或不完整。"
                "请在“模型管理”中重新下载后再识别。"
            )

    def _queue_model_download_progress(
        self, model_name: str, downloaded: int, total: int
    ) -> None:
        if self.download_cancel_event.is_set():
            raise ModelDownloadCancelled()
        total = max(total, 1)
        percent = min(100.0, downloaded * 100.0 / total)
        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total / (1024 * 1024)
        message = (
            f"正在下载 {get_model_spec(model_name).display_name}：{percent:.1f}% "
            f"（{downloaded_mb:.1f} / {total_mb:.1f} MB）..."
        )
        self.worker_queue.put(
            (
                "download_progress",
                {"message": message, "percent": percent},
            )
        )

    def _reject_implicit_model_download(
        self,
        model_name: str,
        downloaded: int,
        total: int,
    ) -> None:
        if downloaded >= max(total, 1):
            return
        raise ModelDownloadError(
            f"{get_model_spec(model_name).display_name} 的缓存缺失或校验失败。"
            "识别过程不会自动下载模型，请在“模型管理”中重新下载。"
        )

    def _reset_recognizer(self) -> None:
        with self.recognizer_lock:
            recognizer = self.recognizer
            self.recognizer = None
            self.recognizer_settings = None
        if recognizer is not None:
            recognizer.close()

    def _current_settings(self) -> RecognizerSettings:
        model_id = self._selected_model_id()
        if not model_id:
            raise RuntimeError("尚未选择已下载模型。")
        return RecognizerSettings(model_name=model_id)

    def _set_image(self, image: Image.Image) -> bool:
        # A result belongs to the previous input image.  Clear it immediately
        # so stale LaTeX/MathML cannot be mistaken for the newly pasted image.
        try:
            prepared = image_to_rgb(image)
            revision = self.image_revision + 1
            image_path = self.session_dir / f"input_{revision}.png"
            prepared.save(image_path, format="PNG")
        except Exception as exc:
            write_log(f"Unable to prepare input image: {exc}")
            messagebox.showerror("图片加载失败", str(exc), parent=self)
            self.status_var.set("图片加载失败")
            return False

        self.clear_output(update_status=False)
        self.image_revision = revision
        self.current_image = prepared
        self.current_image_path = image_path
        self.preview_render_size = None
        self._update_preview()
        if self.recognition_thread is None or not self.recognition_thread.is_alive():
            self._cleanup_obsolete_input_images()
        return True

    def _cleanup_obsolete_input_images(self) -> None:
        current = self.current_image_path
        try:
            candidates = tuple(self.session_dir.glob("input_*.png"))
        except OSError:
            return
        for candidate in candidates:
            if current is not None and candidate == current:
                continue
            try:
                candidate.unlink(missing_ok=True)
            except OSError:
                pass

    def _update_preview(self) -> None:
        self.preview_resize_after_id = None
        if self.current_image is None or self.is_destroying:
            return
        max_width = max(200, self.preview_label.winfo_width() - 24)
        max_height = max(160, self.preview_label.winfo_height() - 24)
        render_size = (max_width, max_height)
        if self.preview_render_size == render_size:
            return
        preview = self.current_image.copy()
        preview.thumbnail(render_size, Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(preview)
        self.preview_render_size = render_size
        self.preview_label.configure(image=self.preview_photo, text="", bg=SURFACE_SUBTLE)

    def _schedule_input_preview_resize(self, _event: tk.Event | None = None) -> None:
        if (
            self.is_destroying
            or self.current_image is None
            or self.preview_resize_after_id is not None
        ):
            return
        self.preview_resize_after_id = self.after_idle(self._update_preview)

    def _session_worker_finished(self, attribute: str) -> None:
        current_thread = threading.current_thread()
        with self.session_cleanup_lock:
            if getattr(self, attribute, None) is current_thread:
                setattr(self, attribute, None)
            should_cleanup = self.is_destroying and not any(
                thread is not None and thread.is_alive()
                for thread in (self.recognition_thread, self.mathml_render_thread)
            )
        if should_cleanup:
            shutil.rmtree(self.session_dir, ignore_errors=True)

    def _cleanup_session_if_idle(self) -> None:
        with self.session_cleanup_lock:
            should_cleanup = not any(
                thread is not None and thread.is_alive()
                for thread in (self.recognition_thread, self.mathml_render_thread)
            )
        if should_cleanup:
            shutil.rmtree(self.session_dir, ignore_errors=True)

    def _wait_for_cancelled_workers(self, *, timeout: float = 1.0) -> None:
        deadline = time.monotonic() + max(0.0, timeout)
        current_thread = threading.current_thread()
        for thread in (self.mathml_render_thread, self.model_download_thread):
            if thread is None or thread is current_thread or not thread.is_alive():
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(timeout=remaining)

    def destroy(self) -> None:
        if self.is_destroying:
            return
        self.is_destroying = True
        # Ask an active downloader to stop at its next progress boundary.  The
        # downloader keeps its `.part` file, so closing the window is safe.
        self.download_cancel_event.set()
        self._cancel_pending_mathml_render()
        if self.capture_after_id is not None:
            try:
                self.after_cancel(self.capture_after_id)
            except tk.TclError:
                pass
            self.capture_after_id = None
        if self.screenshot_selector is not None:
            self.screenshot_selector.destroy(notify=False)
            self.screenshot_selector = None
        if self.mathml_update_after_id is not None:
            try:
                self.after_cancel(self.mathml_update_after_id)
            except tk.TclError:
                pass
            self.mathml_update_after_id = None
        if self.worker_poll_after_id is not None:
            try:
                self.after_cancel(self.worker_poll_after_id)
            except tk.TclError:
                pass
            self.worker_poll_after_id = None
        if self.mathml_preview_poll_after_id is not None:
            try:
                self.after_cancel(self.mathml_preview_poll_after_id)
            except tk.TclError:
                pass
            self.mathml_preview_poll_after_id = None
        if self.preview_resize_after_id is not None:
            try:
                self.after_cancel(self.preview_resize_after_id)
            except tk.TclError:
                pass
            self.preview_resize_after_id = None
        if self.mathml_preview_resize_after_id is not None:
            try:
                self.after_cancel(self.mathml_preview_resize_after_id)
            except tk.TclError:
                pass
            self.mathml_preview_resize_after_id = None
        if self.busy_status_after_id is not None:
            try:
                self.after_cancel(self.busy_status_after_id)
            except tk.TclError:
                pass
            self.busy_status_after_id = None
        self._wait_for_cancelled_workers()
        self._reset_recognizer()
        self._cleanup_session_if_idle()
        super().destroy()


class ScreenshotSelector:
    def __init__(self, parent: tk.Tk, callback) -> None:
        self.parent = parent
        self.callback = callback
        self.screen_area = _virtual_screen_area(parent)
        self.start_x = 0
        self.start_y = 0
        self.rect_id: int | None = None
        self.capture_after_id: str | None = None
        self.finished = False
        self.window = tk.Toplevel(parent)
        self.window.withdraw()
        self.window.overrideredirect(True)
        self.window.attributes("-topmost", True)
        self.window.attributes("-alpha", 0.22)

        width = self.screen_area.right - self.screen_area.left
        height = self.screen_area.bottom - self.screen_area.top
        self.window.geometry(f"{width}x{height}+0+0")

        self.canvas = tk.Canvas(
            self.window,
            cursor="crosshair",
            bg="black",
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.window.bind("<Escape>", self._cancel)

    def start(self) -> None:
        self.window.deiconify()
        _set_toplevel_bounds(self.window, self.screen_area)
        self.window.focus_force()

    def _on_press(self, event: tk.Event) -> None:
        self.start_x = int(event.x_root)
        self.start_y = int(event.y_root)
        self.rect_id = self.canvas.create_rectangle(
            event.x,
            event.y,
            event.x,
            event.y,
            outline="#2f8cff",
            width=3,
        )

    def _on_drag(self, event: tk.Event) -> None:
        if self.rect_id is None:
            return
        x0 = self.start_x - self.window.winfo_rootx()
        y0 = self.start_y - self.window.winfo_rooty()
        self.canvas.coords(self.rect_id, x0, y0, event.x, event.y)

    def _on_release(self, event: tk.Event) -> None:
        if self.finished:
            return
        x1, y1 = self.start_x, self.start_y
        x2, y2 = int(event.x_root), int(event.y_root)
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        self.finished = True
        try:
            self.window.destroy()
        except tk.TclError:
            pass

        if right - left < 5 or bottom - top < 5:
            self.callback(None)
            return

        # Let the compositor remove the translucent overlay before grabbing;
        # otherwise its tint and selection border can become part of the OCR
        # input on slower or multi-monitor Windows systems.
        self.capture_after_id = self.parent.after(
            80,
            lambda: self._grab_selection(left, top, right, bottom),
        )

    def _grab_selection(self, left: int, top: int, right: int, bottom: int) -> None:
        self.capture_after_id = None
        try:
            if sys.platform == "win32":
                image = ImageGrab.grab(
                    bbox=(left, top, right, bottom),
                    all_screens=True,
                )
            else:
                image = ImageGrab.grab(bbox=(left, top, right, bottom))
        except TypeError:
            try:
                image = ImageGrab.grab((left, top, right, bottom))
            except Exception as exc:
                self._report_capture_failure(exc)
                return
        except Exception as exc:
            self._report_capture_failure(exc)
            return
        self.callback(image)

    def _report_capture_failure(self, exc: BaseException) -> None:
        write_log(f"Screenshot capture failed: {exc}")
        self.callback(None)
        if not self.parent.is_destroying:
            try:
                messagebox.showerror("截图失败", str(exc), parent=self.parent)
            except tk.TclError:
                pass

    def _cancel(self, _event: tk.Event | None = None) -> None:
        self.destroy(notify=True)

    def destroy(self, *, notify: bool) -> None:
        if self.capture_after_id is not None:
            try:
                self.parent.after_cancel(self.capture_after_id)
            except tk.TclError:
                pass
            self.capture_after_id = None
        try:
            window_exists = bool(self.window.winfo_exists())
        except tk.TclError:
            window_exists = False
        if self.finished and not window_exists:
            return
        self.finished = True
        try:
            self.window.destroy()
        except tk.TclError:
            pass
        if notify:
            self.callback(None)


def main() -> None:
    install_exception_logger()
    enable_windows_dpi_awareness()
    if "--word-mathml-self-test" in sys.argv:
        run_word_mathml_self_test()
        return
    if "--clipboard-self-test" in sys.argv:
        run_clipboard_self_test()
        return
    if "--runtime-self-test" in sys.argv:
        run_runtime_self_test()
        return
    if "--preview-self-test" in sys.argv:
        run_preview_self_test()
        return
    if "--ui-self-test" in sys.argv:
        run_ui_self_test()
        return
    if "--self-test" in sys.argv:
        run_self_test()
        return

    set_windows_app_id()
    app = FormulaOCRApp()
    app.mainloop()


def enable_windows_dpi_awareness() -> None:
    """Keep Tk coordinates aligned with physical screenshot pixels."""

    if sys.platform != "win32":
        return
    try:
        set_context = ctypes.windll.user32.SetProcessDpiAwarenessContext
        set_context.argtypes = (ctypes.c_void_p,)
        set_context.restype = ctypes.c_int
        if set_context(ctypes.c_void_p(-4)):  # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
            return
    except Exception:
        pass
    try:
        if ctypes.windll.shcore.SetProcessDpiAwareness(2) in (0, 0x80070005):
            return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def set_windows_app_id() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "FormulaOCR.Offline.LaTeX.1"
        )
    except Exception:
        pass


def _safe_console_print(message: object, *, error: bool = False) -> None:
    """Print when a console exists; windowed PyInstaller builds may not have one."""

    stream = sys.stderr if error else sys.stdout
    try:
        print(message, file=stream)
    except (OSError, ValueError):
        pass


def install_exception_logger() -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Installing the exception hook is more important than having a log
        # file. Read-only profiles and full disks must not crash startup before
        # the original exception can be reported.
        pass

    def _hook(exc_type, exc_value, exc_traceback) -> None:
        details = "".join(
            traceback.format_exception(exc_type, exc_value, exc_traceback)
        )
        write_log("Unhandled exception\n" + details)
        if any(
            flag in sys.argv
            for flag in (
                "--self-test",
                "--preview-self-test",
                "--clipboard-self-test",
                "--runtime-self-test",
                "--word-mathml-self-test",
                "--ui-self-test",
            )
        ):
            _safe_console_print(details, error=True)
            return
        try:
            messagebox.showerror("程序错误", f"程序启动失败，详情见日志：\n{LOG_FILE}")
        except Exception:
            pass

    sys.excepthook = _hook


def write_log(message: str) -> None:
    try:
        with _LOG_WRITE_LOCK:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            _rotate_log_if_needed()
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with LOG_FILE.open("a", encoding="utf-8") as file:
                file.write(f"[{timestamp}] {message.rstrip()}\n")
    except OSError:
        # Logging is diagnostic only and is frequently called while handling
        # another failure. A full/read-only disk must not crash the program or
        # replace the original exception.
        pass


def _rotate_log_if_needed() -> None:
    try:
        if not LOG_FILE.is_file() or LOG_FILE.stat().st_size < LOG_MAX_BYTES:
            return
        oldest = LOG_FILE.with_name(f"{LOG_FILE.name}.{LOG_BACKUP_COUNT}")
        oldest.unlink(missing_ok=True)
        for index in range(LOG_BACKUP_COUNT - 1, 0, -1):
            source = LOG_FILE.with_name(f"{LOG_FILE.name}.{index}")
            if source.exists():
                os.replace(
                    source,
                    LOG_FILE.with_name(f"{LOG_FILE.name}.{index + 1}"),
                )
        os.replace(LOG_FILE, LOG_FILE.with_name(f"{LOG_FILE.name}.1"))
    except OSError:
        # Another FormulaOCR instance may have rotated the shared log first.
        # Appending to the current file remains safe in that case.
        pass


def run_self_test() -> None:
    from PIL import ImageDraw, ImageFont

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-image", default="")
    parser.add_argument("--self-test-device", default="cpu")
    parser.add_argument("--self-test-model", default="PP-FormulaNet_plus-S")
    args, _unknown = parser.parse_known_args(sys.argv[1:])

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if args.self_test_image:
        test_image = Path(args.self_test_image).expanduser().resolve()
    else:
        test_image = CACHE_DIR / "self_test_formula.png"
        image = Image.new("RGB", (520, 120), "white")
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/times.ttf", 64)
        except Exception:
            font = ImageFont.load_default()
        draw.text((28, 22), "x^2 + y^2 = z^2", fill="black", font=font)
        image.save(test_image)

    recognizer = FormulaRecognizer(
        model_name=args.self_test_model,
        device=args.self_test_device,
    )
    try:
        formula = recognizer.predict(test_image)
        write_log(f"Self-test OK: {formula}")
        _safe_console_print(formula)
    except Exception:
        details = traceback.format_exc()
        write_log("Self-test failed\n" + details)
        _safe_console_print(details, error=True)
        raise
    finally:
        recognizer.close()


def run_preview_self_test() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--preview-self-test", action="store_true")
    parser.add_argument(
        "--preview-formula",
        default=(
            r"e_{i j}^{(s)}=[f_{i}\parallel f_{j}\parallel"
            r"(x_{j}-x_{i})]\in\mathbb{R}^{2C_{i n}+3}"
        ),
    )
    args, _unknown = parser.parse_known_args(sys.argv[1:])

    app = FormulaOCRApp()
    app.withdraw()
    try:
        mathml = latex_to_mathml(args.preview_formula)
        token = int(time.time() * 1000) % 1_000_000
        image_path = app._render_mathml_to_png(token, mathml, threading.Event())
        write_log(f"Preview self-test OK: {image_path}")
        _safe_console_print(image_path)
    except Exception:
        details = traceback.format_exc()
        write_log("Preview self-test failed\n" + details)
        _safe_console_print(details, error=True)
        raise
    finally:
        app.destroy()


def run_ui_self_test() -> None:
    app = FormulaOCRApp()
    app.withdraw()

    def assert_popup_position(
        name: str,
        popup: tk.Toplevel,
        anchor: tk.Misc,
        *,
        gap: int,
        align: str,
    ) -> None:
        app.update()
        popup.update_idletasks()
        expected_x, expected_y, _width, _height = _anchored_popup_geometry(
            anchor,
            popup.winfo_width(),
            popup.winfo_height(),
            gap=gap,
            align=align,
        )
        actual_x = popup.winfo_rootx()
        actual_y = popup.winfo_rooty()
        if abs(actual_x - expected_x) > 8 or abs(actual_y - expected_y) > 8:
            raise RuntimeError(
                f"{name}位置错误：实际 ({actual_x}, {actual_y})，"
                f"期望 ({expected_x}, {expected_y})。"
            )

    try:
        if not app.protocol("WM_DELETE_WINDOW"):
            raise RuntimeError("主窗口关闭事件未绑定统一资源清理流程。")
        sessions_root = CACHE_DIR / "sessions"
        live_session = sessions_root / f"{os.getpid()}-ui-live-test"
        stale_session = sessions_root / "2147483647-ui-stale-test"
        live_session.mkdir(parents=True, exist_ok=True)
        stale_session.mkdir(parents=True, exist_ok=True)
        old_timestamp = time.time() - SESSION_MAX_AGE_SECONDS - 60
        os.utime(live_session, (old_timestamp, old_timestamp))
        os.utime(stale_session, (old_timestamp, old_timestamp))
        cleanup_probe = _create_runtime_session_dir()
        try:
            if not live_session.is_dir():
                raise RuntimeError("会话清理误删了仍在运行实例的临时目录。")
            if stale_session.exists():
                raise RuntimeError("会话清理没有删除已过期的崩溃残留目录。")
        finally:
            shutil.rmtree(cleanup_probe, ignore_errors=True)
            shutil.rmtree(live_session, ignore_errors=True)
            shutil.rmtree(stale_session, ignore_errors=True)
        if app.worker_poll_after_id is not None:
            app.after_cancel(app.worker_poll_after_id)
            app.worker_poll_after_id = None
        app._set_busy(True, message="正在下载测试模型...", show_cancel=True)
        app._cancel_download()
        app.worker_queue.put(
            (
                "download_progress",
                {"message": "不应恢复的旧进度", "percent": 42.0},
            )
        )
        app._poll_worker_queue()
        if not app.cancel_download_button.is_disabled:
            raise RuntimeError("取消后的旧进度消息重新启用了取消按钮。")
        if not app.status_var.get().startswith("正在取消下载"):
            raise RuntimeError("取消后的旧进度消息覆盖了取消状态。")
        app._set_busy(False)
        app.download_cancel_event.clear()
        app._set_busy(True, message="正在下载队列合并测试...", show_cancel=True)
        for percent in range(200):
            app.worker_queue.put(
                (
                    "download_progress",
                    {"message": f"旧下载进度 {percent}", "percent": percent / 2},
                )
            )
        app.worker_queue.put(("download_cancelled", "RapidLaTeXOCR"))
        app._poll_worker_queue()
        if app.is_busy or not app.worker_queue.empty():
            raise RuntimeError("下载进度队列没有合并到终态。")
        if not app.status_var.get().startswith("下载已取消"):
            raise RuntimeError("下载终态被积压的进度消息延迟。")
        app.download_cancel_event.set()
        try:
            app._queue_model_download_progress("RapidLaTeXOCR", 1, 2)
        except ModelDownloadCancelled:
            pass
        else:
            raise RuntimeError("下载取消信号未被进度回调拦截。")
        finally:
            app.download_cancel_event.clear()
        try:
            app._queue_model_load_status("PP-FormulaNet_plus-S", False)
        except ModelDownloadError:
            pass
        else:
            raise RuntimeError("识别流程仍会隐式下载缺失模型。")
        try:
            app._reject_implicit_model_download("PP-FormulaNet_plus-S", 0, 10)
        except ModelDownloadError:
            pass
        else:
            raise RuntimeError("识别后端仍可隐式下载损坏模型。")
        app._reject_implicit_model_download("PP-FormulaNet_plus-S", 10, 10)
        if not hasattr(app, "cancel_download_button"):
            raise RuntimeError("主界面缺少下载取消按钮。")
        selected_model_id = app._selected_model_id()
        if selected_model_id:
            if not is_model_cached(selected_model_id):
                raise RuntimeError("主界面保留了未下载模型作为当前模型。")
            if selected_model_id not in app.model_info_var.get():
                raise RuntimeError("主界面未显示当前模型标识。")
        elif "模型管理" not in app.model_info_var.get():
            raise RuntimeError("没有已下载模型时，主界面未提示用户先下载。")
        if hasattr(app, "auto_copy_toggle") or hasattr(app, "auto_copy_var"):
            raise RuntimeError("主界面仍残留识别后自动复制控件。")
        original_copy_text = app._copy_text
        original_showerror = messagebox.showerror
        copy_errors: list[tuple[str, str]] = []
        try:
            app._replace_output_text(r"x+y")
            app._copy_text = lambda _text: (_ for _ in ()).throw(
                tk.TclError("clipboard busy")
            )
            messagebox.showerror = lambda title, text, **_kwargs: copy_errors.append(
                (str(title), str(text))
            )
            app.copy_latex()
            if app.status_var.get() != "复制失败":
                raise RuntimeError("LaTeX 剪贴板异常没有更新失败状态。")
            app.copy_format("asciimath")
            if app.status_var.get() != "复制失败":
                raise RuntimeError("格式复制剪贴板异常没有更新失败状态。")
            if len(copy_errors) != 2 or any(
                title != "复制失败" or "暂时不可用" not in text
                for title, text in copy_errors
            ):
                raise RuntimeError("剪贴板异常没有显示简洁友好的错误提示。")
        finally:
            app._copy_text = original_copy_text
            messagebox.showerror = original_showerror
            app.clear_output(update_status=False)
        clipboard_test_path = CACHE_DIR / "ui_clipboard_formula.png"
        clipboard_fallback_path = CACHE_DIR / "ui_clipboard_formula_fallback.png"
        Image.new("RGB", (120, 48), "white").save(clipboard_test_path)
        Image.new("RGB", (96, 40), "white").save(clipboard_fallback_path)
        original_grabclipboard = ImageGrab.grabclipboard
        original_set_image = app._set_image
        try:
            app.deiconify()
            set_image_attempts = 0

            def fail_first_clipboard_image(image: Image.Image) -> bool:
                nonlocal set_image_attempts
                set_image_attempts += 1
                if set_image_attempts == 1:
                    return False
                return original_set_image(image)

            app._set_image = fail_first_clipboard_image
            ImageGrab.grabclipboard = lambda: [
                str(clipboard_test_path),
                str(clipboard_fallback_path),
            ]
            if not app.paste_image(notify_if_missing=False):
                raise RuntimeError("首个剪贴板文件处理失败后没有继续尝试后续图片。")
            if str(clipboard_fallback_path) not in app.status_var.get():
                raise RuntimeError("剪贴板多文件回退没有选中后续有效图片。")
            app._set_image = original_set_image
            app.output_text.delete("1.0", tk.END)
            app.output_text.insert("1.0", r"stale_{result}")
            app.clipboard_clear()
            app.clipboard_append(str(clipboard_test_path))
            ImageGrab.grabclipboard = lambda: [str(clipboard_test_path)]
            app.output_text.focus_force()
            app.update()
            app.output_text.event_generate("<Control-v>")
            app.update()
            if app._current_latex():
                raise RuntimeError(
                    "粘贴图片文件时，路径或旧结果进入了 LaTeX 文本框。"
                )
            if app.current_image is None or not app.status_var.get().startswith(
                "已从剪贴板文件加载图片"
            ):
                raise RuntimeError("Ctrl+V 未加载剪贴板中的图片文件。")
            if (
                app.current_image_path is None
                or app.current_image_path.parent != app.session_dir
            ):
                raise RuntimeError("输入图片没有使用当前实例的独立临时目录。")
            if app.mathml_preview_label.cget("text") != "暂无公式预览":
                raise RuntimeError("粘贴新图片后未清理旧 MathML 预览。")

            app.clear_output(update_status=False)
            ImageGrab.grabclipboard = lambda: str(clipboard_fallback_path)
            app.output_text.focus_force()
            app.output_text.event_generate("<Control-v>")
            app.update()
            if app._current_latex():
                raise RuntimeError("单文件字符串剪贴板路径进入了 LaTeX 文本框。")
            if app.current_image_path is None or not app.current_image_path.is_file():
                raise RuntimeError("单文件字符串剪贴板没有加载图片。")

            missing_clipboard_path = CACHE_DIR / "missing_wechat_formula.png"
            missing_clipboard_path.unlink(missing_ok=True)
            app.clear_output(update_status=False)
            app.clipboard_clear()
            app.clipboard_append(str(missing_clipboard_path))
            ImageGrab.grabclipboard = lambda: [str(missing_clipboard_path)]
            app.output_text.focus_force()
            app.output_text.event_generate("<Control-v>")
            app.update()
            if app._current_latex():
                raise RuntimeError(
                    "已失效的剪贴板图片路径进入了 LaTeX 文本框。"
                )
            if "已失效或无法读取" not in app.status_var.get():
                raise RuntimeError("失效剪贴板图片没有给出明确状态。")

            ImageGrab.grabclipboard = lambda: None
            app.clipboard_clear()
            app.clipboard_append(r"x+y")
            app.output_text.focus_force()
            app.output_text.event_generate("<Control-v>")
            app.update()
            if app._current_latex() != "x+y":
                raise RuntimeError("图片粘贴修复破坏了普通 LaTeX 文本粘贴。")
        finally:
            ImageGrab.grabclipboard = original_grabclipboard
            app._set_image = original_set_image
            app.clipboard_clear()
            clipboard_test_path.unlink(missing_ok=True)
            clipboard_fallback_path.unlink(missing_ok=True)
            app.clear_output()
            app.withdraw()

        render_jobs: list[tuple[int, str, str]] = []
        original_queue_mathml_render = app._queue_mathml_render
        try:
            app._queue_mathml_render = lambda token, latex, mathml: render_jobs.append(
                (token, latex, mathml)
            )
            app._replace_output_text(r"x^2+y^2")
            app._update_mathml_preview()
            app.update()
            if len(render_jobs) != 1:
                raise RuntimeError("程序写入识别结果时重复启动了 MathML 渲染。")
            if app.mathml_update_after_id is not None:
                raise RuntimeError("程序写入识别结果后仍残留重复预览定时器。")
        finally:
            app._queue_mathml_render = original_queue_mathml_render
            app.clear_output()

        if app.worker_poll_after_id is not None:
            app.after_cancel(app.worker_poll_after_id)
            app.worker_poll_after_id = None
        app.worker_queue.put(
            (
                "success",
                {
                    "formula": r"stale_{formula}",
                    "elapsed": 0.01,
                    "image_revision": app.image_revision - 1,
                },
            )
        )
        app._poll_worker_queue()
        if app._current_latex():
            raise RuntimeError("旧图片的异步识别结果覆盖了当前输入。")
        if "已忽略旧图片" not in app.status_var.get():
            raise RuntimeError("旧图片识别结果被丢弃时没有给出明确状态。")
        if app.worker_poll_after_id is not None:
            app.after_cancel(app.worker_poll_after_id)
            app.worker_poll_after_id = None
        app.worker_queue.put(
            (
                "recognition_cancelled",
                {
                    "model_name": "PP-FormulaNet_plus-S",
                    "image_revision": app.image_revision,
                },
            )
        )
        app._poll_worker_queue()
        if app.status_var.get() != "识别已取消":
            raise RuntimeError("识别取消状态被错误显示为下载取消。")
        if (
            len(MODEL_SPECS) < 10
            or "MathCraftFormula" not in MODEL_BY_ID
            or "Pix2TextMFR15" not in MODEL_BY_ID
            or "LaTeX_OCR_rec" not in MODEL_BY_ID
            or "MixTexZhEn" not in MODEL_BY_ID
            or "UniMERNetSmallONNX" not in MODEL_BY_ID
        ):
            raise RuntimeError("模型目录未完整加载。")
        original_status_provider = app.model_picker.status_provider
        original_picker_command = app.model_picker.command
        original_picker_model = app.model_picker.get()
        downloaded_for_test = {"MathCraftFormula", "Pix2TextMFR15"}
        try:
            app.model_picker.status_provider = lambda model_id: (
                "已下载" if model_id in downloaded_for_test else "待下载"
            )
            app.model_picker.command = lambda: None
            app.model_picker.refresh()
            app.deiconify()
            app.update()
            app.model_picker.button.event_generate("<Button-1>")
            app.update()
            if app.model_picker.popup is None or not app.model_picker.popup.winfo_exists():
                raise RuntimeError("模型选择弹出卡片被打开点击立即关闭。")
            assert_popup_position(
                "主界面模型下拉框",
                app.model_picker.popup,
                app.model_picker,
                gap=6,
                align="right",
            )
            if set(app.model_picker.visible_model_ids) != downloaded_for_test:
                raise RuntimeError("主界面模型下拉框没有只显示已下载模型。")
            focused_row = app.focus_get()
            if (
                focused_row is None
                or app.model_picker.popup is None
                or focused_row.winfo_toplevel() is not app.model_picker.popup
            ):
                raise RuntimeError("键盘打开模型下拉框后焦点没有进入选项列表。")
            focused_row.event_generate("<Down>")
            app.update()
            next_row = app.focus_get()
            if next_row is None or next_row is focused_row:
                raise RuntimeError("模型下拉框方向键无法移动选项焦点。")
            next_row.event_generate("<Return>")
            app.update()
            if app.model_picker.popup is not None:
                raise RuntimeError("回车选择模型后下拉框没有关闭。")
            app.model_picker._choose("MathCraftFormula")
            if app.model_picker.get() != "MathCraftFormula":
                raise RuntimeError("主界面模型下拉框无法选择已下载模型。")
            app.model_picker._choose("MixTexZhEn")
            if app.model_picker.get() != "MathCraftFormula":
                raise RuntimeError("主界面模型下拉框错误地选择了未下载模型。")
            app.model_picker.set("MixTexZhEn")
            app._update_model_summary()
            if app.model_picker.get():
                raise RuntimeError("主界面没有清除已删除或未下载的旧模型选择。")
        finally:
            app.model_picker._close_popup()
            app.model_picker.status_provider = original_status_provider
            app.model_picker.command = original_picker_command
            app.model_picker.set(original_picker_model)
            app.model_picker.refresh()
            app.withdraw()
        app.show_model_manager()
        app.update_idletasks()
        manager = next(
            child
            for child in app.winfo_children()
            if isinstance(child, tk.Toplevel) and child.title() == "模型管理"
        )
        app.show_model_manager()
        app.update_idletasks()
        if len(
            [
                child
                for child in app.winfo_children()
                if isinstance(child, tk.Toplevel) and child.title() == "模型管理"
            ]
        ) != 1:
            raise RuntimeError("重复点击模型管理创建了多个窗口。")
        manager.update_idletasks()
        model_trees = [
            child
            for child in manager.winfo_children()
            if isinstance(child, ttk.Treeview)
        ]
        if (
            not model_trees
            or not model_trees[0].exists("Pix2TextMFR15")
            or not model_trees[0].exists("LaTeX_OCR_rec")
            or not model_trees[0].exists("MixTexZhEn")
            or not model_trees[0].exists("UniMERNetSmallONNX")
        ):
            raise RuntimeError("模型管理未显示新增模型。")
        manager_entries: list[tk.Entry] = []
        manager_filters: list[RoundedChoice] = []
        manager_quick_filters: list[ModelFilterChips] = []
        manager_buttons: list[RoundedButton] = []
        manager_scrollbars: list[SlimScrollbar] = []

        def collect_manager_controls(widget: tk.Misc) -> None:
            if isinstance(widget, tk.Entry):
                manager_entries.append(widget)
            if isinstance(widget, RoundedChoice):
                manager_filters.append(widget)
            if isinstance(widget, ModelFilterChips):
                manager_quick_filters.append(widget)
            if isinstance(widget, RoundedButton):
                manager_buttons.append(widget)
            if isinstance(widget, SlimScrollbar):
                manager_scrollbars.append(widget)
            for child in widget.winfo_children():
                collect_manager_controls(child)

        collect_manager_controls(manager)
        if not manager_entries or not manager_filters or not manager_quick_filters:
            raise RuntimeError("模型管理缺少搜索、供应商或快捷筛选控件。")
        if not manager_scrollbars:
            raise RuntimeError("模型管理没有使用与主界面一致的细滚动条。")
        mutation_buttons = {
            button.text: button
            for button in manager_buttons
            if button.text in {"下载", "删除缓存"}
        }
        if set(mutation_buttons) != {"下载", "删除缓存"}:
            raise RuntimeError("模型管理缺少下载或删除缓存操作。")
        app._set_busy(True, message="UI 自检忙碌状态")
        app.update()
        if any(not button.is_disabled for button in mutation_buttons.values()):
            raise RuntimeError("识别期间已打开的模型管理仍可修改模型。")
        mutation_buttons["下载"].command()
        if "暂时不能下载或删除模型" not in app.status_var.get():
            raise RuntimeError("模型管理忙碌状态缺少操作入口二次拦截。")
        if app.model_download_thread is not None:
            raise RuntimeError("模型管理忙碌状态仍启动了下载线程。")
        app._set_busy(False)
        app.update()
        manager_filters[0]._toggle_popup()
        manager.update()
        if (
            manager_filters[0].popup is None
            or not manager_filters[0].popup.winfo_exists()
        ):
            raise RuntimeError("模型管理供应商下拉框无法打开。")
        assert_popup_position(
            "模型管理供应商下拉框",
            manager_filters[0].popup,
            manager_filters[0],
            gap=4,
            align="left",
        )
        manager_filters[0]._close_popup()
        manager_button_texts = {button.text for button in manager_buttons}
        if not {"下载", "打开下载源", "上游条款", "刷新"}.issubset(
            manager_button_texts
        ):
            raise RuntimeError("模型管理缺少详情/刷新操作。")
        if "设为当前" in manager_button_texts:
            raise RuntimeError("模型管理不应再承担当前模型选择。")
        manager_entries[0].insert(0, "LaTeX-OCR")
        manager.update_idletasks()
        if not model_trees[0].exists("LaTeX_OCR_rec"):
            raise RuntimeError("模型管理筛选无法找到 LaTeX-OCR Rec。")
        manager_entries[0].delete(0, tk.END)
        manager_quick_filters[0].set("onnx")
        manager.update_idletasks()
        if not model_trees[0].exists("UniMERNetSmallONNX") or any(
            not get_model_spec(model_id).is_onnx
            for model_id in model_trees[0].get_children()
        ):
            raise RuntimeError("模型管理 ONNX 快捷筛选结果不正确。")
        manager_quick_filters[0].set("all")
        manager_filters[0].set("SakuraMathcraft")
        manager.update_idletasks()
        if (
            not model_trees[0].exists("MathCraftFormula")
            or model_trees[0].exists("PP-FormulaNet_plus-S")
        ):
            raise RuntimeError("模型管理供应商筛选结果不正确。")
        manager_filters[0].set("全部供应商")
        manager_trace_bindings = manager._formula_ocr_trace_bindings
        manager.destroy()
        app.update_idletasks()
        if any(variable.trace_info() for variable, _trace_id in manager_trace_bindings):
            raise RuntimeError("关闭模型管理后仍残留 Tcl 变量回调。")
        app.deiconify()
        app.update()
        format_coordinates: list[tuple[int, int]] = []
        original_tk_popup = app.format_menu.tk_popup
        try:
            app.format_menu.tk_popup = lambda x, y: format_coordinates.append((x, y))
            app.show_format_menu()
        finally:
            app.format_menu.tk_popup = original_tk_popup
        expected_format_x, expected_format_y, _width, _height = (
            _anchored_popup_geometry(
                app.format_button,
                app.format_menu.winfo_reqwidth(),
                app.format_menu.winfo_reqheight(),
                gap=4,
                align="left",
            )
        )
        if format_coordinates != [(expected_format_x, expected_format_y)]:
            raise RuntimeError("更多格式菜单没有定位到按钮正下方。")
        app.withdraw()
        app.show_runtime_info()
        app.update_idletasks()
        runtime_windows = [
            child
            for child in app.winfo_children()
            if isinstance(child, tk.Toplevel) and child.title() == "运行环境与缓存"
        ]
        if not runtime_windows:
            raise RuntimeError("运行环境窗口未创建。")
        runtime_windows[0].destroy()
        write_log(f"UI self-test OK: models={len(MODEL_SPECS)}")
        _safe_console_print(f"ui-self-test-ok:{len(MODEL_SPECS)}")
    finally:
        app.destroy()


def _probe_runtime_component(component: str) -> str:
    """Load one native runtime far enough to catch missing DLL/PYD files."""

    if component == "onnxruntime":
        import onnxruntime

        options = onnxruntime.SessionOptions()
        if options is None:
            raise RuntimeError("onnxruntime.SessionOptions() returned no object")
        providers = tuple(onnxruntime.get_available_providers())
        if "CPUExecutionProvider" not in providers:
            raise RuntimeError("ONNX Runtime is missing CPUExecutionProvider")
        return str(getattr(onnxruntime, "__version__", "native runtime"))

    if component == "tokenizers":
        import tokenizers
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel

        tokenizer = Tokenizer(
            WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]")
        )
        if tokenizer.encode("runtime-probe").ids != [0]:
            raise RuntimeError("tokenizers native encode probe returned unexpected data")
        return str(getattr(tokenizers, "__version__", "native runtime"))

    if component == "paddle-native":
        try:
            from formula_ocr_app.paddle_formula_recognizer import (
                _load_paddle_native_runtime,
            )
        except ModuleNotFoundError as exc:  # Direct script execution.
            if exc.name != "formula_ocr_app":
                raise
            from paddle_formula_recognizer import _load_paddle_native_runtime

        paddle_native = _load_paddle_native_runtime()
        missing = [
            name
            for name in ("AnalysisConfig", "create_predictor")
            if not hasattr(paddle_native, name)
        ]
        if missing:
            raise RuntimeError(
                "Paddle native inference API is incomplete: " + ", ".join(missing)
            )
        return "原生推理接口已加载"

    raise ValueError(f"Unknown runtime component: {component}")


def run_runtime_self_test() -> None:
    """Check cache boundaries and load packaged native runtimes without models."""

    cache_dir = runtime_cache_dir().resolve()
    bundled_dir = FormulaOCRApp._bundled_runtime_dir().resolve()
    paddle_dir = model_user_cache_path(DEFAULT_MODEL_ID).resolve()
    external_dir = model_user_cache_path("RapidLaTeXOCR").resolve()
    mixtex_dir = model_user_cache_path("MixTexZhEn").resolve()
    unimernet_dir = model_user_cache_path("UniMERNetSmallONNX").resolve()

    def is_inside(path: Path, parent: Path) -> bool:
        return path == parent or parent in path.parents

    if any(
        is_inside(path, bundled_dir)
        for path in (cache_dir, paddle_dir, external_dir, mixtex_dir, unimernet_dir)
    ):
        raise RuntimeError(
            "运行时模型缓存错误地落在 _internal 中；应使用 LocalAppData\\FormulaOCR\\cache。"
        )
    if len(MODEL_SPECS) < 10 or len({item.provider for item in MODEL_SPECS}) < 6:
        raise RuntimeError("模型目录或供应商目录未完整加载。")
    eager_modules = sorted(
        module_name
        for module_name in (
            "requests",
            "onnxruntime",
            "formula_ocr_app.model_manager_dialog",
            "model_manager_dialog",
            "formula_ocr_app.mathml_preview",
            "mathml_preview",
            "formula_ocr_app.paddle_formula_recognizer",
            "formula_ocr_app.rapid_recognizer",
        )
        if module_name in sys.modules
    )
    if eager_modules:
        raise RuntimeError("启动阶段提前加载了延迟组件：" + ", ".join(eager_modules))

    runtime_details: dict[str, str] = {}
    for component in ("onnxruntime", "tokenizers", "paddle-native"):
        try:
            runtime_details[component] = _probe_runtime_component(component)
        except Exception as exc:
            raise RuntimeError(
                f"原生运行组件自检失败（{component}）：{exc}"
            ) from exc

    message = (
        "runtime-self-test-ok\n"
        f"mode={'frozen' if getattr(sys, 'frozen', False) else 'source'}\n"
        f"internal={bundled_dir}\n"
        f"cache={cache_dir}\n"
        f"paddle_cache={paddle_dir}\n"
        f"onnx_cache={external_dir}\n"
        f"mixtex_cache={mixtex_dir}\n"
        f"unimernet_onnx_cache={unimernet_dir}\n"
        f"onnxruntime={runtime_details['onnxruntime']}\n"
        f"tokenizers={runtime_details['tokenizers']}\n"
        f"paddle_native={runtime_details['paddle-native']}\n"
        f"models={len(MODEL_SPECS)} providers={len({item.provider for item in MODEL_SPECS})}"
    )
    write_log(message)
    _safe_console_print(message)


def run_clipboard_self_test() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--clipboard-self-test", action="store_true")
    parser.add_argument(
        "--clipboard-formula",
        default=(
            r"\sum_{i=1}^{n}\sum_{j=1}^{n}\sum_{k=1}^{n}"
            r" f(x_i,y_j,z_k)+\overline{x}"
        ),
    )
    parser.add_argument("--require-native-clipboard", action="store_true")
    args, _unknown = parser.parse_known_args(sys.argv[1:])

    if sys.platform != "win32":
        print("clipboard-self-test: skipped on non-Windows")
        return

    root = tk.Tk()
    root.withdraw()
    root.update()
    try:
        mathml = latex_to_mathml(args.clipboard_formula)
        copied = copy_mathml_for_word_to_clipboard(
            mathml,
            plain_text=args.clipboard_formula,
            clipboard_widget=root,
            owner_hwnd=root.winfo_id(),
        )
        if not copied:
            raise RuntimeError("failed to write Word HTML MathML clipboard format")
        root.update()
        native_formats: list[str] = []
        native_error = ""
        try:
            native_formats = windows_clipboard_formats()
        except Exception as exc:
            native_error = str(exc)
        native_required = {FORMAT_HTML}
        if native_required.issubset(native_formats):
            html = windows_clipboard_text(FORMAT_HTML)
            formats = native_formats
            clipboard_path = "win32"
        else:
            if args.require_native_clipboard:
                raise RuntimeError(
                    "native clipboard formats unavailable: "
                    + (native_error or "|".join(native_formats))
                )
            html = tk_clipboard_text(root, FORMAT_HTML)
            formats = [FORMAT_HTML, "CF_UNICODETEXT"]
            clipboard_path = "tk-fallback"
        plain_text = root.clipboard_get()
        legacy_formats = {
            FORMAT_OFFICE_OPEN_XML,
            FORMAT_MATHML,
            FORMAT_MATHML_PRESENTATION,
        }
        checks = {
            "html_has_cf_html_header": html.startswith("Version:1.0"),
            "html_has_mathml": "<math" in html and "</math>" in html,
            "html_has_mathml_namespace": (
                'xmlns="http://www.w3.org/1998/Math/MathML"' in html
            ),
            "html_has_overline": _contains_word_overline_mathml(html),
            "html_has_display_limited_large_operator": (
                _contains_display_limited_large_operator(html)
            ),
            "legacy_word_formats_absent": legacy_formats.isdisjoint(formats),
            "unicode_text_matches": plain_text == args.clipboard_formula,
        }
        print(f"clipboard_path:{clipboard_path}")
        print("formats:" + "|".join(formats))
        for name, passed in checks.items():
            print(f"{name}:{passed}")
        if native_formats:
            print("win32_formats:" + "|".join(native_formats))
        elif native_error:
            print(f"win32_formats_unavailable:{native_error}")
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError("clipboard self-test failed: " + ", ".join(failed))
    finally:
        root.destroy()


def run_word_mathml_self_test() -> None:
    try:
        from formula_ocr_app.word_clipboard_tests import (
            CASES,
            run_word_mathml_regression,
        )
    except ImportError:
        from word_clipboard_tests import CASES, run_word_mathml_regression

    failures = run_word_mathml_regression()
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        raise SystemExit(1)
    print(f"word-mathml-regression-ok:{len(CASES)}")


def _contains_word_overline_mathml(text: str) -> bool:
    return '<mover accent="true">' in text and "<mi>―</mi>" in text


def _contains_display_limited_large_operator(text: str) -> bool:
    return "<munderover>" in text and 'largeop="true"' in text


if __name__ == "__main__":
    main()
