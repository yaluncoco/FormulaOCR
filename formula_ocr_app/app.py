from __future__ import annotations

import queue
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import argparse
import ctypes
import importlib.util
import webbrowser
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageChops, ImageGrab, ImageTk

try:
    from formula_ocr_app.app_settings import AppSettings, load_settings, save_settings
    from formula_ocr_app.model_catalog import (
        DEFAULT_MODEL_ID,
        MODEL_BY_ID,
        MODEL_QUICK_FILTERS,
        MODEL_SPECS,
        get_model_spec,
        model_matches_query,
        model_matches_quick_filter,
    )
    from formula_ocr_app.model_downloader import (
        ModelDownloadCancelled,
        ensure_official_model,
    )
    from formula_ocr_app.paddle_hf_model_downloader import (
        ensure_paddle_hf_model,
        is_paddle_hf_model_cached,
    )
    from formula_ocr_app.mathcraft_model_downloader import (
        ensure_mathcraft_model,
        is_mathcraft_model_cached,
    )
    from formula_ocr_app.pix2text_model_downloader import (
        ensure_pix2text_model,
        is_pix2text_model_cached,
    )
    from formula_ocr_app.mixtex_model_downloader import (
        ensure_mixtex_model,
        is_mixtex_model_cached,
    )
    from formula_ocr_app.unimernet_onnx_model_downloader import (
        ensure_unimernet_onnx_model,
        is_unimernet_onnx_model_cached,
    )
    from formula_ocr_app.rapid_model_downloader import (
        ensure_rapid_model,
        is_rapid_model_cached,
    )
    from formula_ocr_app.recognizer import (
        PaddleFormulaRecognizer,
        PaddleOCRNotReadyError,
    )
    from formula_ocr_app.recognition_pipeline import FormulaRecognizer
    from formula_ocr_app.runtime_paths import (
        is_paddle_model_cached,
        is_paddle_model_bundled,
        paddle_model_has_data,
        paddle_model_dir,
        directory_size,
        external_model_has_data,
        external_model_dir,
        bundled_external_model_dir,
        is_external_model_bundled,
        paddle_model_cache_size,
        remove_paddle_model,
        remove_external_model,
        runtime_cache_dir,
        runtime_log_dir,
    )
    from formula_ocr_app.formula_formats import (
        clean_recognized_latex,
        export_formula_docx,
        latex_to_asciimath,
        latex_to_equation_environment,
        latex_to_html,
        latex_to_markdown_block,
        latex_to_markdown_inline,
        latex_to_mathml,
        latex_to_typst,
        latex_to_word_linear,
        mathml_to_word_mathml,
        mathml_to_omml,
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
except ImportError:  # Allows `python formula_ocr_app/app.py`.
    from app_settings import AppSettings, load_settings, save_settings
    from model_catalog import (
        DEFAULT_MODEL_ID,
        MODEL_BY_ID,
        MODEL_QUICK_FILTERS,
        MODEL_SPECS,
        get_model_spec,
        model_matches_query,
        model_matches_quick_filter,
    )
    from model_downloader import ModelDownloadCancelled, ensure_official_model
    from paddle_hf_model_downloader import ensure_paddle_hf_model, is_paddle_hf_model_cached
    from mathcraft_model_downloader import (
        ensure_mathcraft_model,
        is_mathcraft_model_cached,
    )
    from pix2text_model_downloader import (
        ensure_pix2text_model,
        is_pix2text_model_cached,
    )
    from mixtex_model_downloader import ensure_mixtex_model, is_mixtex_model_cached
    from unimernet_onnx_model_downloader import (
        ensure_unimernet_onnx_model,
        is_unimernet_onnx_model_cached,
    )
    from rapid_model_downloader import ensure_rapid_model, is_rapid_model_cached
    from recognizer import PaddleFormulaRecognizer, PaddleOCRNotReadyError
    from recognition_pipeline import FormulaRecognizer
    from runtime_paths import (
        is_paddle_model_cached,
        is_paddle_model_bundled,
        paddle_model_has_data,
        paddle_model_dir,
        directory_size,
        external_model_has_data,
        external_model_dir,
        bundled_external_model_dir,
        is_external_model_bundled,
        paddle_model_cache_size,
        remove_paddle_model,
        remove_external_model,
        runtime_cache_dir,
        runtime_log_dir,
    )
    from formula_formats import (
        clean_recognized_latex,
        export_formula_docx,
        latex_to_asciimath,
        latex_to_equation_environment,
        latex_to_html,
        latex_to_markdown_block,
        latex_to_markdown_inline,
        latex_to_mathml,
        latex_to_typst,
        latex_to_word_linear,
        mathml_to_word_mathml,
        mathml_to_omml,
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


APP_ROOT = Path(__file__).resolve().parent


def _resource_base() -> Path:
    bundled = getattr(sys, "_MEIPASS", None)
    if bundled:
        return Path(bundled)
    return APP_ROOT.parent


CACHE_DIR = runtime_cache_dir()
LOG_DIR = runtime_log_dir()
LOG_FILE = LOG_DIR / "formula_ocr.log"
DEFAULT_PADDLEOCR_REPO = _resource_base() / "PaddleOCR-main"
ICON_FILE = _resource_base() / "icon.png"
ICON_ICO_FILE = _resource_base() / "icon.ico"
APP_BG = "#eef3f8"
PANEL_BG = "#ffffff"
SURFACE_SUBTLE = "#f7f9fc"
TEXT_PRIMARY = "#172033"
TEXT_SECONDARY = "#657086"
ACCENT = "#d4237a"
ACCENT_DARK = "#b71f69"
ACCENT_SOFT = "#fde7f2"
BORDER = "#dce4ef"


@dataclass(frozen=True)
class RecognizerSettings:
    model_name: str


@dataclass(frozen=True)
class _ScreenArea:
    left: int
    top: int
    right: int
    bottom: int


def _monitor_work_area(anchor: tk.Misc) -> _ScreenArea:
    """Return the work area of the monitor containing the anchor widget."""

    if sys.platform == "win32":
        try:
            class POINT(ctypes.Structure):
                _fields_ = (("x", ctypes.c_long), ("y", ctypes.c_long))

            class RECT(ctypes.Structure):
                _fields_ = (
                    ("left", ctypes.c_long),
                    ("top", ctypes.c_long),
                    ("right", ctypes.c_long),
                    ("bottom", ctypes.c_long),
                )

            class MONITORINFO(ctypes.Structure):
                _fields_ = (
                    ("cbSize", ctypes.c_ulong),
                    ("rcMonitor", RECT),
                    ("rcWork", RECT),
                    ("dwFlags", ctypes.c_ulong),
                )

            anchor.update_idletasks()
            point = POINT(
                anchor.winfo_rootx() + anchor.winfo_width() // 2,
                anchor.winfo_rooty() + anchor.winfo_height() // 2,
            )
            user32 = ctypes.windll.user32
            user32.MonitorFromPoint.argtypes = (POINT, ctypes.c_ulong)
            user32.MonitorFromPoint.restype = ctypes.c_void_p
            user32.GetMonitorInfoW.argtypes = (
                ctypes.c_void_p,
                ctypes.POINTER(MONITORINFO),
            )
            user32.GetMonitorInfoW.restype = ctypes.c_int
            monitor = user32.MonitorFromPoint(point, 2)
            info = MONITORINFO()
            info.cbSize = ctypes.sizeof(MONITORINFO)
            if monitor and user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
                work = info.rcWork
                return _ScreenArea(work.left, work.top, work.right, work.bottom)
        except Exception:
            pass
    return _ScreenArea(
        0,
        0,
        anchor.winfo_screenwidth(),
        anchor.winfo_screenheight(),
    )


def _anchored_popup_geometry(
    anchor: tk.Misc,
    width: int,
    height: int,
    *,
    gap: int = 4,
    align: str = "left",
) -> tuple[int, int, int, int]:
    """Place a popup below its control, flipping above only when necessary."""

    anchor.update_idletasks()
    area = _monitor_work_area(anchor)
    margin = 8
    width = min(max(1, int(width)), max(1, area.right - area.left - margin * 2))
    height = min(max(1, int(height)), max(1, area.bottom - area.top - margin * 2))
    if align == "right":
        x = anchor.winfo_rootx() + anchor.winfo_width() - width
    else:
        x = anchor.winfo_rootx()
    x = max(area.left + margin, min(x, area.right - width - margin))

    below = anchor.winfo_rooty() + anchor.winfo_height() + gap
    above = anchor.winfo_rooty() - height - gap
    if below + height <= area.bottom - margin or above < area.top + margin:
        y = below
    else:
        y = above
    y = max(area.top + margin, min(y, area.bottom - height - margin))
    return x, y, width, height


def _show_anchored_popup(
    popup: tk.Toplevel,
    anchor: tk.Misc,
    width: int,
    height: int,
    *,
    gap: int = 4,
    align: str = "left",
) -> tuple[int, int, int, int]:
    """Map an override-redirect popup at an exact monitor-aware position."""

    x, y, width, height = _anchored_popup_geometry(
        anchor,
        width,
        height,
        gap=gap,
        align=align,
    )
    popup.geometry(f"{width}x{height}+0+0")
    popup.update_idletasks()
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
            hwnd = user32.GetParent(popup.winfo_id()) or popup.winfo_id()
            user32.SetWindowPos(hwnd, 0, x, y, width, height, 0x0010)
        except Exception:
            if x >= 0 and y >= 0:
                popup.geometry(f"{width}x{height}+{x}+{y}")
    elif x >= 0 and y >= 0:
        popup.geometry(f"{width}x{height}+{x}+{y}")
    popup.deiconify()
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
            hwnd = user32.GetParent(popup.winfo_id()) or popup.winfo_id()
            user32.SetWindowPos(hwnd, 0, x, y, width, height, 0x0050)
        except Exception:
            pass
    popup.lift()
    return x, y, width, height


def _rounded_rect(
    canvas: tk.Canvas,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    radius: int,
    **kwargs,
) -> int:
    points = [
        x1 + radius,
        y1,
        x2 - radius,
        y1,
        x2,
        y1,
        x2,
        y1 + radius,
        x2,
        y2 - radius,
        x2,
        y2,
        x2 - radius,
        y2,
        x1 + radius,
        y2,
        x1,
        y2,
        x1,
        y2 - radius,
        x1,
        y1 + radius,
        x1,
        y1,
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


class RoundedButton(tk.Canvas):
    def __init__(
        self,
        parent: tk.Widget,
        *,
        text: str,
        command,
        width: int = 112,
        height: int = 38,
        radius: int = 12,
        bg: str = PANEL_BG,
        fg: str = TEXT_PRIMARY,
        active_bg: str = SURFACE_SUBTLE,
        border: str = BORDER,
        selected_bg: str | None = None,
        selected_fg: str = "#ffffff",
        font: tuple[str, int, str] | tuple[str, int] = ("Microsoft YaHei UI", 10),
    ) -> None:
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=parent.cget("bg") if isinstance(parent, tk.Widget) else APP_BG,
            highlightthickness=0,
            bd=0,
            takefocus=True,
        )
        self.command = command
        self.radius = radius
        self.normal_bg = bg
        self.active_bg = active_bg
        self.border = border
        self.fg = fg
        self.selected_bg = selected_bg
        self.selected_fg = selected_fg
        self.text = text
        self.font = font
        self.is_selected = False
        self.is_disabled = False
        self._draw()
        self.bind("<Enter>", lambda _event: self._draw(hover=True))
        self.bind("<Leave>", lambda _event: self._draw())
        self.bind("<Button-1>", self._click)
        self.bind("<Return>", self._keyboard_click)
        self.bind("<space>", self._keyboard_click)

    def set_selected(self, selected: bool) -> None:
        self.is_selected = selected
        self._draw()

    def set_text(self, text: str) -> None:
        self.text = text
        self._draw()

    def set_width(self, width: int) -> None:
        self.configure(width=max(2, int(width)))
        self._draw()

    def set_disabled(self, disabled: bool) -> None:
        self.is_disabled = disabled
        self._draw()

    def _click(self, _event: tk.Event) -> None:
        if not self.is_disabled and self.command:
            self.command()

    def _keyboard_click(self, _event: tk.Event) -> str:
        self._click(_event)
        return "break"

    def _draw(self, hover: bool = False) -> None:
        self.delete("all")
        width = max(2, int(self.winfo_reqwidth()))
        height = max(2, int(self.winfo_reqheight()))
        selected = self.is_selected and self.selected_bg is not None
        fill = self.selected_bg if selected else (self.active_bg if hover else self.normal_bg)
        outline = self.selected_bg if selected else self.border
        text_color = self.selected_fg if selected else self.fg
        if self.is_disabled:
            fill = "#e4eaf3"
            outline = "#d5deeb"
            text_color = "#9aa5b5"
        _rounded_rect(
            self,
            1,
            1,
            width - 1,
            height - 1,
            self.radius,
            fill=fill,
            outline=outline,
            width=1,
        )
        self.create_text(
            width // 2,
            height // 2,
            text=self.text,
            fill=text_color,
            font=self.font,
        )


class ToggleSwitch(tk.Canvas):
    """Compact themed boolean control with a native-looking switch track."""

    def __init__(
        self,
        parent: tk.Widget,
        *,
        text: str,
        variable: tk.BooleanVar,
        command=None,
        width: int = 112,
        height: int = 38,
    ) -> None:
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=parent.cget("bg"),
            highlightthickness=0,
            bd=0,
            cursor="hand2",
            takefocus=True,
        )
        self.text = text
        self.variable = variable
        self.command = command
        self.is_disabled = False
        self.is_hovered = False
        self.has_focus = False
        self._variable_trace = self.variable.trace_add("write", self._sync)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._click)
        self.bind("<Return>", self._keyboard_click)
        self.bind("<space>", self._keyboard_click)
        self.bind("<FocusIn>", self._on_focus_in)
        self.bind("<FocusOut>", self._on_focus_out)
        self.bind("<Destroy>", self._on_destroy, add="+")
        self._draw()

    def set_disabled(self, disabled: bool) -> None:
        self.is_disabled = disabled
        self.configure(cursor="arrow" if disabled else "hand2")
        self._draw()

    def _sync(self, *_args) -> None:
        self._draw()

    def _click(self, _event: tk.Event | None = None) -> None:
        if self.is_disabled:
            return
        self.variable.set(not self.variable.get())
        if self.command:
            self.command()

    def _keyboard_click(self, event: tk.Event) -> str:
        self._click(event)
        return "break"

    def _on_enter(self, _event: tk.Event) -> None:
        self.is_hovered = True
        self._draw()

    def _on_leave(self, _event: tk.Event) -> None:
        self.is_hovered = False
        self._draw()

    def _on_focus_in(self, _event: tk.Event) -> None:
        self.has_focus = True
        self._draw()

    def _on_focus_out(self, _event: tk.Event) -> None:
        self.has_focus = False
        self._draw()

    def _draw(self) -> None:
        self.delete("all")
        width = max(2, int(self.winfo_reqwidth()))
        height = max(2, int(self.winfo_reqheight()))
        enabled = bool(self.variable.get())
        if self.is_disabled:
            surface = "#f1f4f8"
            border = "#dce3ec"
            text_color = "#98a3b3"
            track = "#d3d9e2"
            knob = "#f8fafc"
        else:
            surface = ACCENT_SOFT if self.is_hovered else "#ffffff"
            border = ACCENT if self.has_focus else BORDER
            text_color = TEXT_PRIMARY
            track = ACCENT if enabled else "#c5ceda"
            knob = "#ffffff"
        _rounded_rect(
            self,
            1,
            1,
            width - 1,
            height - 1,
            13,
            fill=surface,
            outline=border,
            width=1,
        )
        self.create_text(
            12,
            height // 2,
            text=self.text,
            fill=text_color,
            font=("Microsoft YaHei UI", 9, "bold" if enabled else "normal"),
            anchor=tk.W,
        )
        track_width = 32
        track_height = 18
        track_right = width - 9
        track_left = track_right - track_width
        track_top = (height - track_height) // 2
        track_bottom = track_top + track_height
        _rounded_rect(
            self,
            track_left,
            track_top,
            track_right,
            track_bottom,
            track_height // 2,
            fill=track,
            outline=track,
            width=0,
        )
        knob_radius = 7
        knob_x = track_right - 9 if enabled else track_left + 9
        knob_y = height // 2
        self.create_oval(
            knob_x - knob_radius,
            knob_y - knob_radius,
            knob_x + knob_radius,
            knob_y + knob_radius,
            fill=knob,
            outline="#ffffff" if enabled else "#eef1f5",
            width=1,
        )

    def _on_destroy(self, event: tk.Event) -> None:
        if event.widget is not self:
            return
        try:
            self.variable.trace_remove("write", self._variable_trace)
        except tk.TclError:
            pass


class RoundedChoice(tk.Frame):
    """Theme-consistent replacement for a small readonly combobox."""

    def __init__(
        self,
        parent: tk.Widget,
        *,
        values,
        variable: tk.StringVar | None = None,
        width: int = 176,
        height: int = 32,
        bg: str = PANEL_BG,
    ) -> None:
        super().__init__(parent, bg=bg, highlightthickness=0, bd=0)
        self.values = tuple(str(value) for value in values)
        if not self.values:
            raise ValueError("RoundedChoice 至少需要一个选项")
        self.variable = variable or tk.StringVar(master=self, value=self.values[0])
        if self.variable.get() not in self.values:
            self.variable.set(self.values[0])
        self.choice_width = width
        self.popup: tk.Toplevel | None = None
        self._popup_root: tk.Misc | None = None
        self._popup_root_binding: str | None = None
        self._variable_trace = self.variable.trace_add("write", self._sync_button)
        self.button = RoundedButton(
            self,
            text="",
            command=self._toggle_popup,
            width=width,
            height=height,
            radius=13,
            bg="#ffffff",
            active_bg=ACCENT_SOFT,
            border=BORDER,
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        self.button.pack(fill=tk.X)
        self.bind("<Destroy>", self._on_destroy, add="+")
        self._sync_button()

    def get(self) -> str:
        return self.variable.get()

    def set(self, value: str) -> None:
        if value in self.values:
            self.variable.set(value)

    def set_disabled(self, disabled: bool) -> None:
        self.button.set_disabled(disabled)
        if disabled:
            self._close_popup()

    def _button_label(self, value: str) -> str:
        font = tkfont.Font(root=self, font=self.button.font)
        available = max(24, self.choice_width - 34)
        if font.measure(value) <= available:
            return f"{value}  ▾"
        suffix = "…"
        low, high = 1, len(value)
        while low < high:
            middle = (low + high + 1) // 2
            if font.measure(value[:middle] + suffix) <= available:
                low = middle
            else:
                high = middle - 1
        return f"{value[:low]}{suffix}  ▾"

    def _sync_button(self, *_args) -> None:
        if not hasattr(self, "button"):
            return
        self.button.set_text(self._button_label(self.variable.get()))

    def _toggle_popup(self) -> None:
        if self.button.is_disabled:
            return
        if self.popup is not None and self.popup.winfo_exists():
            self._close_popup()
            return

        popup = tk.Toplevel(self)
        self.popup = popup
        popup.withdraw()
        popup.overrideredirect(True)
        popup.configure(bg=BORDER)
        popup.transient(self.winfo_toplevel())
        popup.bind("<Escape>", lambda _event: self._close_popup())

        card = tk.Frame(popup, bg=PANEL_BG, padx=9, pady=9)
        card.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        current = self.variable.get()
        for value in self.values:
            selected = value == current
            row_bg = ACCENT_SOFT if selected else PANEL_BG
            row = tk.Frame(
                card,
                bg=row_bg,
                cursor="hand2",
                takefocus=True,
                padx=10,
                pady=7,
            )
            row.pack(fill=tk.X, pady=1)
            label = tk.Label(
                row,
                text=("✓  " if selected else "    ") + value,
                bg=row_bg,
                fg=ACCENT_DARK if selected else TEXT_PRIMARY,
                font=("Microsoft YaHei UI", 10, "bold" if selected else "normal"),
                anchor=tk.W,
            )
            label.pack(fill=tk.X)

            def set_row_color(color: str, targets=(row, label)) -> None:
                for target in targets:
                    target.configure(bg=color)

            for widget in (row, label):
                widget.bind(
                    "<Button-1>",
                    lambda _event, choice=value: self._choose(choice),
                )
                widget.bind(
                    "<Enter>",
                    lambda _event, update=set_row_color: update(ACCENT_SOFT),
                )
                widget.bind(
                    "<Leave>",
                    lambda _event, update=set_row_color, color=row_bg: update(color),
                )
            row.bind("<Return>", lambda _event, choice=value: self._choose(choice))
            row.bind("<space>", lambda _event, choice=value: self._choose(choice))

        popup.update_idletasks()
        width = max(self.winfo_width(), popup.winfo_reqwidth())
        height = popup.winfo_reqheight()
        _show_anchored_popup(popup, self, width, height, gap=4, align="left")
        popup.after_idle(self._install_outside_binding)

    def _install_outside_binding(self) -> None:
        if self.popup is None or not self.popup.winfo_exists():
            return
        root = self.winfo_toplevel()

        def close_from_outside(event: tk.Event) -> None:
            if self.popup is None:
                return
            try:
                if str(event.widget.winfo_toplevel()) == str(self.popup):
                    return
            except tk.TclError:
                return
            self._close_popup()

        self._popup_root = root
        self._popup_root_binding = root.bind(
            "<Button-1>", close_from_outside, add="+"
        )

    def _choose(self, value: str) -> str:
        self.variable.set(value)
        self._close_popup()
        return "break"

    def _close_popup(self) -> None:
        if self._popup_root is not None and self._popup_root_binding is not None:
            try:
                self._popup_root.unbind("<Button-1>", self._popup_root_binding)
            except tk.TclError:
                pass
        self._popup_root = None
        self._popup_root_binding = None
        if self.popup is not None:
            try:
                self.popup.destroy()
            except tk.TclError:
                pass
            self.popup = None

    def _on_destroy(self, event: tk.Event) -> None:
        if event.widget is not self:
            return
        self._close_popup()
        try:
            self.variable.trace_remove("write", self._variable_trace)
        except tk.TclError:
            pass


class ModelFilterChips(tk.Frame):
    """Reusable quick filters shared by the picker and model manager."""

    def __init__(
        self,
        parent: tk.Widget,
        *,
        variable: tk.StringVar | None = None,
        bg: str = PANEL_BG,
    ) -> None:
        super().__init__(parent, bg=bg, highlightthickness=0, bd=0)
        self.variable = variable or tk.StringVar(master=self, value="all")
        valid_keys = {key for key, _label in MODEL_QUICK_FILTERS}
        if self.variable.get() not in valid_keys:
            self.variable.set("all")
        self.buttons: dict[str, RoundedButton] = {}
        font = tkfont.Font(root=self, font=("Microsoft YaHei UI", 8, "bold"))
        for key, label in MODEL_QUICK_FILTERS:
            button = RoundedButton(
                self,
                text=label,
                command=lambda selected=key: self.variable.set(selected),
                width=max(52, font.measure(label) + 24),
                height=28,
                radius=10,
                bg="#ffffff",
                active_bg=ACCENT_SOFT,
                border=BORDER,
                selected_bg=ACCENT,
                selected_fg="#ffffff",
                font=("Microsoft YaHei UI", 8, "bold"),
            )
            button.pack(side=tk.LEFT, padx=(0, 5))
            self.buttons[key] = button
        self._variable_trace = self.variable.trace_add("write", self._sync)
        self.bind("<Destroy>", self._on_destroy, add="+")
        self._sync()

    def get(self) -> str:
        return self.variable.get()

    def set(self, value: str) -> None:
        if value in self.buttons:
            self.variable.set(value)

    def _sync(self, *_args) -> None:
        selected = self.variable.get()
        for key, button in self.buttons.items():
            button.set_selected(key == selected)

    def _on_destroy(self, event: tk.Event) -> None:
        if event.widget is not self:
            return
        try:
            self.variable.trace_remove("write", self._variable_trace)
        except tk.TclError:
            pass


class ModelPicker(tk.Frame):
    """A compact, application-themed model selector.

    ttk's native combobox is useful for forms, but its popup and colors vary
    noticeably between Windows themes.  This picker keeps the compact header
    while showing the supplier, purpose, size and cache state in a consistent
    popup card.
    """

    def __init__(
        self,
        parent: tk.Widget,
        *,
        specs,
        model_id: str,
        command,
        status_provider,
        manager_command,
    ) -> None:
        super().__init__(parent, bg=APP_BG, highlightthickness=0, bd=0)
        self.specs = tuple(specs)
        self.model_id = model_id
        self.command = command
        self.status_provider = status_provider
        self.manager_command = manager_command
        self.popup: tk.Toplevel | None = None
        self._popup_root: tk.Misc | None = None
        self._popup_root_binding: str | None = None
        self.visible_model_ids: tuple[str, ...] = ()
        self.is_disabled = False
        self.button = RoundedButton(
            self,
            text="",
            command=self._toggle_popup,
            width=250,
            height=38,
            radius=13,
            bg="#ffffff",
            active_bg=ACCENT_SOFT,
            border=BORDER,
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        self.button.pack()
        self._sync_button()

    def get(self) -> str:
        return self.model_id

    def set(self, model_id: str, *, notify: bool = False) -> None:
        if model_id not in {spec.model_id for spec in self.specs}:
            return
        changed = self.model_id != model_id
        self.model_id = model_id
        self._sync_button()
        if changed and notify:
            self.command()

    def set_disabled(self, disabled: bool) -> None:
        self.is_disabled = disabled
        self.button.set_disabled(disabled)
        if disabled:
            self._close_popup()

    def refresh(self) -> None:
        self._sync_button()

    def _sync_button(self) -> None:
        spec = next(
            (item for item in self.specs if item.model_id == self.model_id),
            self.specs[0],
        )
        if self._model_is_available(spec.model_id):
            label = f"{spec.compact_name}  ▾"
        else:
            label = "选择已下载模型  ▾"
        font = tkfont.Font(root=self, font=self.button.font)
        self.button.set_width(min(330, max(238, font.measure(label) + 42)))
        self.button.set_text(label)

    @staticmethod
    def _state_is_available(state: str) -> bool:
        return state == "随包内置" or state.startswith("已下载")

    def _model_is_available(self, model_id: str) -> bool:
        return self._state_is_available(self.status_provider(model_id))

    def _toggle_popup(self) -> None:
        if self.is_disabled:
            return
        if self.popup is not None and self.popup.winfo_exists():
            self._close_popup()
            return

        popup = tk.Toplevel(self)
        self.popup = popup
        popup.withdraw()
        popup.overrideredirect(True)
        popup.configure(bg=BORDER)
        popup.transient(self.winfo_toplevel())
        popup.bind("<Escape>", lambda _event: self._close_popup())

        card = tk.Frame(popup, bg=PANEL_BG, padx=10, pady=10)
        card.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        title_bar = tk.Frame(card, bg=PANEL_BG)
        title_bar.pack(fill=tk.X, padx=3, pady=(0, 2))
        tk.Label(
            title_bar,
            text="选择已下载模型",
            bg=PANEL_BG,
            fg=TEXT_PRIMARY,
            font=("Microsoft YaHei UI", 11, "bold"),
            anchor=tk.W,
        ).pack(side=tk.LEFT)
        available_models = []
        for spec in self.specs:
            state = self.status_provider(spec.model_id)
            if self._state_is_available(state):
                available_models.append((spec, state))
        self.visible_model_ids = tuple(spec.model_id for spec, _state in available_models)
        tk.Label(
            title_bar,
            text=f"{len(available_models)} 个",
            bg=PANEL_BG,
            fg=TEXT_SECONDARY,
            font=("Microsoft YaHei UI", 8),
            anchor=tk.E,
        ).pack(side=tk.RIGHT)
        tk.Label(
            card,
            text="需要其他模型时，请先到“模型管理”中下载",
            bg=PANEL_BG,
            fg=TEXT_SECONDARY,
            font=("Microsoft YaHei UI", 9),
            anchor=tk.W,
        ).pack(fill=tk.X, padx=3, pady=(0, 8))

        list_shell = tk.Frame(card, bg=PANEL_BG)
        list_shell.pack(fill=tk.BOTH, expand=True, padx=1)
        list_shell.columnconfigure(0, weight=1)
        list_shell.rowconfigure(0, weight=1)
        list_canvas = tk.Canvas(
            list_shell,
            bg=PANEL_BG,
            highlightthickness=0,
            bd=0,
        )
        list_canvas.grid(row=0, column=0, sticky="nsew")
        list_scrollbar = SlimScrollbar(
            list_shell,
            command=list_canvas.yview,
            width=12,
            bg=PANEL_BG,
        )
        list_scrollbar.grid(row=0, column=1, sticky="ns", padx=(5, 0))
        list_frame = tk.Frame(list_canvas, bg=PANEL_BG)
        list_window = list_canvas.create_window(
            (0, 0),
            window=list_frame,
            anchor="nw",
        )
        list_canvas.configure(yscrollcommand=list_scrollbar.set)

        def sync_scroll_region(_event: tk.Event | None = None) -> None:
            list_canvas.configure(scrollregion=list_canvas.bbox("all"))

        def resize_list_frame(event: tk.Event) -> None:
            list_canvas.itemconfigure(list_window, width=max(1, event.width))

        list_frame.bind("<Configure>", sync_scroll_region)
        list_canvas.bind("<Configure>", resize_list_frame)

        def scroll_list(event: tk.Event) -> str:
            delta = getattr(event, "delta", 0)
            if delta:
                units = -1 if delta > 0 else 1
            else:
                units = -1 if getattr(event, "num", 0) == 4 else 1
            list_canvas.yview_scroll(units, "units")
            return "break"

        list_canvas.bind("<MouseWheel>", scroll_list)
        list_canvas.bind("<Button-4>", scroll_list)
        list_canvas.bind("<Button-5>", scroll_list)

        if not available_models:
            empty_state = tk.Frame(list_frame, bg=PANEL_BG, pady=18)
            empty_state.pack(fill=tk.X)
            tk.Label(
                empty_state,
                text="暂无已下载模型",
                bg=PANEL_BG,
                fg=TEXT_PRIMARY,
                font=("Microsoft YaHei UI", 10, "bold"),
            ).pack()
            tk.Label(
                empty_state,
                text="下载完成后，模型会自动出现在这里",
                bg=PANEL_BG,
                fg=TEXT_SECONDARY,
                font=("Microsoft YaHei UI", 9),
            ).pack(pady=(3, 10))
            RoundedButton(
                empty_state,
                text="打开模型管理",
                command=self._open_model_manager,
                width=116,
                height=32,
                radius=10,
                bg="#ffffff",
                active_bg=ACCENT_SOFT,
                border=BORDER,
            ).pack()
        else:
            for spec, state in available_models:
                selected = spec.model_id == self.model_id
                row_bg = ACCENT_SOFT if selected else PANEL_BG
                row = tk.Frame(
                    list_frame,
                    bg=row_bg,
                    cursor="hand2",
                    padx=8,
                    pady=7,
                )
                row.pack(fill=tk.X, pady=2)
                row.columnconfigure(0, weight=1)
                row.columnconfigure(1, weight=0)
                title = tk.Label(
                    row,
                    text=("✓ " if selected else "    ")
                    + ("★ " if spec.recommended else "")
                    + ("⚠ " if spec.requires_terms_ack else "")
                    + spec.compact_name,
                    bg=row_bg,
                    fg=ACCENT_DARK if selected else TEXT_PRIMARY,
                    font=("Microsoft YaHei UI", 10, "bold"),
                    anchor=tk.W,
                )
                title.grid(row=0, column=0, sticky="ew")
                if state.startswith("随包校验失败"):
                    state_bg, state_fg = "#ffebeb", "#a33a3a"
                elif state.startswith("随包"):
                    state_bg, state_fg = "#e8f7ef", "#187044"
                elif state.startswith("已下载"):
                    state_bg, state_fg = "#e9f1ff", "#2f5f9f"
                elif state.startswith("下载未完成"):
                    state_bg, state_fg = "#fff4df", "#9a6416"
                else:
                    state_bg, state_fg = "#f0f3f7", TEXT_SECONDARY
                state_badge = tk.Label(
                    row,
                    text=state,
                    bg=state_bg,
                    fg=state_fg,
                    font=("Microsoft YaHei UI", 8, "bold"),
                    padx=6,
                    pady=2,
                    anchor=tk.E,
                )
                state_badge.grid(row=0, column=1, sticky="e", padx=(8, 0))
                meta = tk.Label(
                    row,
                    text=(
                        f"{spec.provider}  ·  {spec.size_label}  ·  {spec.languages}"
                        + ("  ·  使用前确认上游条款" if spec.requires_terms_ack else "")
                    ),
                    bg=row_bg,
                    fg=TEXT_SECONDARY,
                    font=("Microsoft YaHei UI", 8),
                    anchor=tk.W,
                )
                meta.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 0))
                detail = tk.Label(
                    row,
                    text=spec.best_for,
                    bg=row_bg,
                    fg=TEXT_SECONDARY,
                    font=("Microsoft YaHei UI", 8),
                    anchor=tk.W,
                )
                detail.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(1, 0))

                def set_row_color(
                    color: str,
                    targets=(row, title, meta, detail),
                ) -> None:
                    for target in targets:
                        target.configure(bg=color)

                for widget in (row, title, state_badge, meta, detail):
                    widget.bind(
                        "<Button-1>",
                        lambda _event, selected_id=spec.model_id: self._choose(
                            selected_id
                        ),
                    )
                    widget.bind(
                        "<MouseWheel>",
                        scroll_list,
                    )
                    widget.bind("<Button-4>", scroll_list)
                    widget.bind("<Button-5>", scroll_list)
                    widget.bind(
                        "<Enter>",
                        lambda _event, update=set_row_color: update(ACCENT_SOFT),
                    )
                    widget.bind(
                        "<Leave>",
                        lambda _event, update=set_row_color, selected=selected: update(
                            ACCENT_SOFT if selected else PANEL_BG
                        ),
                    )

        list_frame.update_idletasks()
        sync_scroll_region()
        list_canvas.yview_moveto(0.0)

        popup.update_idletasks()
        width = max(430, self.winfo_width() + 140)
        work_area = _monitor_work_area(self)
        max_height = max(300, int((work_area.bottom - work_area.top) * 0.78))
        list_height = min(390, max(130, len(available_models) * 72 + 8))
        list_shell.configure(height=list_height)
        list_shell.pack_propagate(False)
        popup.update_idletasks()
        height = min(popup.winfo_reqheight(), max_height)
        _show_anchored_popup(popup, self, width, height, gap=6, align="right")
        # RoundedButton invokes its command on <Button-1>.  Installing the
        # root-level outside-click binding synchronously would let that same
        # event bubble to the root and immediately close the popup again.
        popup.after_idle(self._install_outside_binding)

    def _install_outside_binding(self) -> None:
        if self.popup is None or not self.popup.winfo_exists():
            return
        root = self.winfo_toplevel()

        def close_from_outside(event: tk.Event) -> None:
            if self.popup is None:
                return
            try:
                event_top = event.widget.winfo_toplevel()
                if str(event_top) == str(self.popup):
                    return
            except tk.TclError:
                return
            self._close_popup()

        self._popup_root = root
        self._popup_root_binding = root.bind(
            "<Button-1>", close_from_outside, add="+"
        )

    def _open_model_manager(self) -> None:
        self._close_popup()
        self.manager_command()

    def _choose(self, model_id: str) -> None:
        if not self._model_is_available(model_id):
            return
        self._close_popup()
        if self.model_id == model_id:
            return
        self.model_id = model_id
        self._sync_button()
        self.command()

    def _close_popup(self) -> None:
        if self._popup_root is not None and self._popup_root_binding is not None:
            try:
                self._popup_root.unbind(
                    "<Button-1>", self._popup_root_binding
                )
            except tk.TclError:
                pass
        self._popup_root = None
        self._popup_root_binding = None
        if self.popup is not None:
            try:
                self.popup.destroy()
            except tk.TclError:
                pass
            self.popup = None


class RoundedPanel(tk.Canvas):
    def __init__(self, parent: tk.Widget, *, radius: int = 18, padding: int = 16) -> None:
        super().__init__(
            parent,
            bg=APP_BG,
            highlightthickness=0,
            bd=0,
        )
        self.radius = radius
        self.padding = padding
        self.content = tk.Frame(self, bg=PANEL_BG)
        self.window_id = self.create_window(
            padding,
            padding,
            anchor="nw",
            window=self.content,
        )
        self.bind("<Configure>", self._resize)

    def _resize(self, event: tk.Event) -> None:
        self.delete("panel")
        width = max(2, int(event.width))
        height = max(2, int(event.height))
        _rounded_rect(
            self,
            2,
            2,
            width - 2,
            height - 2,
            self.radius,
            fill=PANEL_BG,
            outline=BORDER,
            width=1,
            tags="panel",
        )
        self.tag_lower("panel")
        inner_width = max(1, width - self.padding * 2)
        inner_height = max(1, height - self.padding * 2)
        self.coords(self.window_id, self.padding, self.padding)
        self.itemconfigure(self.window_id, width=inner_width, height=inner_height)


class SlimScrollbar(tk.Canvas):
    def __init__(
        self,
        parent: tk.Widget,
        *,
        command,
        width: int = 12,
        bg: str = PANEL_BG,
        track: str = "#edf2f8",
        thumb: str = "#aeb7c4",
        active_thumb: str = "#8793a3",
    ) -> None:
        super().__init__(
            parent,
            width=width,
            bg=bg,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        self.command = command
        self.track = track
        self.thumb = thumb
        self.active_thumb = active_thumb
        self.first = 0.0
        self.last = 1.0
        self.drag_start_y = 0
        self.drag_start_first = 0.0
        self.dragging = False
        self.bind("<Configure>", lambda _event: self._draw())
        self.bind("<Button-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Enter>", lambda _event: self._draw(hover=True))
        self.bind("<Leave>", lambda _event: self._draw())

    def set(self, first: float | str, last: float | str) -> None:
        try:
            first_float = float(first)
            last_float = float(last)
        except (TypeError, ValueError):
            return
        self.first = min(max(first_float, 0.0), 1.0)
        self.last = min(max(last_float, self.first), 1.0)
        self._draw()

    def _thumb_bounds(self) -> tuple[int, int]:
        height = max(1, self.winfo_height())
        visible = max(0.0, min(1.0, self.last - self.first))
        if visible >= 0.999:
            return 2, max(3, height - 2)
        thumb_height = min(height - 4, max(34, int(height * visible)))
        movable = max(1, height - 4 - thumb_height)
        max_first = max(0.001, 1.0 - visible)
        top = 2 + int((self.first / max_first) * movable)
        return top, top + thumb_height

    def _draw(self, hover: bool = False) -> None:
        self.delete("all")
        width = max(8, self.winfo_width())
        height = max(8, self.winfo_height())
        bar_width = 5
        x1 = (width - bar_width) // 2
        x2 = x1 + bar_width
        _rounded_rect(
            self,
            x1,
            2,
            x2,
            height - 2,
            3,
            fill=self.track,
            outline=self.track,
            width=0,
        )
        top, bottom = self._thumb_bounds()
        fill = self.active_thumb if hover or self.dragging else self.thumb
        _rounded_rect(
            self,
            x1,
            top,
            x2,
            bottom,
            3,
            fill=fill,
            outline=fill,
            width=0,
        )

    def _on_press(self, event: tk.Event) -> None:
        top, bottom = self._thumb_bounds()
        if top <= event.y <= bottom:
            self.dragging = True
            self.drag_start_y = int(event.y)
            self.drag_start_first = self.first
        else:
            self._move_thumb_to(int(event.y))
            self.dragging = True
            self.drag_start_y = int(event.y)
            self.drag_start_first = self.first
        self._draw(hover=True)

    def _on_drag(self, event: tk.Event) -> None:
        if not self.dragging:
            return
        height = max(1, self.winfo_height())
        visible = max(0.0, min(1.0, self.last - self.first))
        top, bottom = self._thumb_bounds()
        movable = max(1, height - 4 - (bottom - top))
        max_first = max(0.0, 1.0 - visible)
        delta = (int(event.y) - self.drag_start_y) / movable * max_first
        self._moveto(self.drag_start_first + delta)

    def _on_release(self, _event: tk.Event) -> None:
        self.dragging = False
        self._draw()

    def _move_thumb_to(self, y: int) -> None:
        height = max(1, self.winfo_height())
        top, bottom = self._thumb_bounds()
        thumb_height = bottom - top
        movable = max(1, height - 4 - thumb_height)
        visible = max(0.0, min(1.0, self.last - self.first))
        max_first = max(0.0, 1.0 - visible)
        fraction = ((y - 2 - thumb_height / 2) / movable) * max_first
        self._moveto(fraction)

    def _moveto(self, fraction: float) -> None:
        visible = max(0.0, min(1.0, self.last - self.first))
        max_first = max(0.0, 1.0 - visible)
        fraction = min(max(fraction, 0.0), max_first)
        self.command("moveto", fraction)


class FormulaOCRApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("公式识别助手")
        self.geometry("1240x780")
        self.minsize(1040, 650)
        self.configure(bg=APP_BG)

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.saved_settings = load_settings()
        self.accepted_model_terms = set(self.saved_settings.accepted_model_terms)
        self.current_image: Image.Image | None = None
        self.current_image_path: Path | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.window_icon: tk.PhotoImage | None = None
        self.recognizer: FormulaRecognizer | None = None
        self.recognizer_settings: RecognizerSettings | None = None
        self.worker_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.mathml_preview_queue: queue.Queue[tuple[int, str, str]] = queue.Queue()
        self.is_busy = False
        self.mathml_update_after_id: str | None = None
        self.worker_poll_after_id: str | None = None
        self.mathml_preview_poll_after_id: str | None = None
        self.mathml_render_token = 0
        self.mathml_preview_photo: ImageTk.PhotoImage | None = None
        self.busy_started_at: float | None = None
        self.busy_status_after_id: str | None = None
        self.busy_status_message = "正在加载模型/识别公式..."
        self.download_cancel_event = threading.Event()
        self.is_destroying = False

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
            else DEFAULT_MODEL_ID
        )
        self.model_picker = ModelPicker(
            model_bar,
            specs=MODEL_SPECS,
            model_id=selected_model,
            command=self._on_model_changed,
            status_provider=self._model_status_label,
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
        self.bind("<Control-v>", lambda _event: self.paste_image())
        self.bind("<Control-Return>", lambda _event: self.recognize_image())
        self.bind("<Control-c>", self._copy_shortcut)

    def _copy_shortcut(self, event: tk.Event) -> str | None:
        if self.focus_get() is self.output_text:
            return None
        self.copy_latex()
        return "break"

    def _on_model_changed(self, _event: tk.Event | None = None) -> None:
        self._reset_recognizer()
        self._save_preferences()
        self._update_model_summary()
        model_id = self._selected_model_id()
        state = "已下载" if self._model_is_cached(model_id) else "使用时下载"
        self.status_var.set(f"已选择 {get_model_spec(model_id).display_name}（{state}）")

    def _selected_model_id(self) -> str:
        return self.model_picker.get()

    @staticmethod
    def _model_is_cached(model_id: str) -> bool:
        spec = get_model_spec(model_id)
        backend = spec.backend
        if backend == "rapid_onnx":
            return is_rapid_model_cached()
        if backend == "mathcraft_onnx":
            return is_mathcraft_model_cached()
        if backend == "pix2text_onnx":
            return is_pix2text_model_cached()
        if backend == "mixtex_onnx":
            return is_mixtex_model_cached()
        if backend == "unimernet_onnx":
            return is_unimernet_onnx_model_cached()
        if backend == "paddle_hf":
            return is_paddle_hf_model_cached(verify_hash=True)
        return is_paddle_model_cached(model_id)

    @staticmethod
    def _model_is_bundled_only(model_id: str) -> bool:
        """Return whether the valid model available to the app is bundled."""

        spec = get_model_spec(model_id)
        if spec.uses_paddle_runtime:
            bundled = is_paddle_model_bundled(model_id)
            if spec.backend == "paddle_hf":
                bundled = bundled and is_paddle_hf_model_cached(verify_hash=True)
            return bundled and not paddle_model_has_data(model_id)
        return is_external_model_bundled(model_id) and not external_model_has_data(
            model_id
        )

    @staticmethod
    def _model_cache_size(model_id: str) -> int:
        spec = get_model_spec(model_id)
        if spec.uses_paddle_runtime:
            return paddle_model_cache_size(model_id)
        user_size = directory_size(external_model_dir(model_id))
        if user_size:
            return user_size
        bundled_dir = bundled_external_model_dir(model_id)
        return directory_size(bundled_dir) if bundled_dir is not None else 0

    @staticmethod
    def _model_has_user_cache_data(model_id: str) -> bool:
        spec = get_model_spec(model_id)
        if spec.uses_paddle_runtime:
            return paddle_model_has_data(model_id)
        return external_model_has_data(model_id)

    @staticmethod
    def _model_user_cache_path(model_id: str) -> Path:
        spec = get_model_spec(model_id)
        if spec.uses_paddle_runtime:
            return paddle_model_dir(model_id)
        return external_model_dir(model_id)

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
        spec = get_model_spec(model_id)
        cached = self._model_is_cached(model_id)
        if cached and self._model_is_bundled_only(model_id):
            state = "随包内置"
        else:
            state = "已下载" if cached else "未下载 · 首次使用自动下载"
        terms = " · 使用前需确认上游条款" if spec.requires_terms_ack else ""
        self.model_info_var.set(
            f"{spec.model_id} · {spec.best_for} · {spec.languages} · {state}{terms}"
        )
        self.model_picker.refresh()

    def _model_status_label(
        self,
        model_id: str,
        *,
        cached: bool | None = None,
    ) -> str:
        spec = get_model_spec(model_id)
        if cached is None:
            cached = self._model_is_cached(model_id)
        if cached and self._model_is_bundled_only(model_id):
            return "随包内置"
        if not cached:
            has_partial = (
                paddle_model_has_data(model_id)
                if spec.uses_paddle_runtime
                else external_model_has_data(model_id)
            )
            if has_partial:
                return "下载未完成 · 可继续"
        if (
            spec.backend == "paddle_hf"
            and is_paddle_model_bundled(model_id)
            and not cached
        ):
            return "随包校验失败"
        if not spec.uses_paddle_runtime and is_external_model_bundled(model_id):
            return "已下载" if cached else "随包校验失败"
        return "已下载" if cached else "待下载"

    def show_model_manager(self) -> None:
        window = tk.Toplevel(self)
        window.title("模型管理")
        window.geometry("1040x650")
        window.minsize(920, 520)
        window.configure(bg=APP_BG)
        window.transient(self)
        window.columnconfigure(0, weight=1)
        window.columnconfigure(1, weight=0)
        window.rowconfigure(2, weight=1)

        manager_header = tk.Frame(window, bg=APP_BG)
        manager_header.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 13))
        tk.Label(
            manager_header,
            text="本地公式 OCR 模型",
            bg=APP_BG,
            fg=TEXT_PRIMARY,
            font=("Microsoft YaHei UI", 17, "bold"),
        ).pack(anchor=tk.W)
        tk.Label(
            manager_header,
            text="在这里下载或删除模型；下载完成后，在主界面下拉框中选择识别模型。",
            bg=APP_BG,
            fg=TEXT_SECONDARY,
            font=("Microsoft YaHei UI", 9),
        ).pack(anchor=tk.W, pady=(3, 0))
        tk.Label(
            manager_header,
            text=(
                f"运行时下载缓存：{runtime_cache_dir()}  ·  "
                "打包目录 _internal 仅作为只读随包资源，不写入下载文件"
            ),
            bg=APP_BG,
            fg=TEXT_SECONDARY,
            font=("Microsoft YaHei UI", 8),
            anchor=tk.W,
        ).pack(anchor=tk.W, pady=(2, 0))

        filter_bar = tk.Frame(window, bg=APP_BG)
        filter_bar.grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=20,
            pady=(0, 10),
        )
        filter_bar.columnconfigure(1, weight=1)
        tk.Label(
            filter_bar,
            text="筛选模型",
            bg=APP_BG,
            fg=TEXT_SECONDARY,
            font=("Microsoft YaHei UI", 9, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=(0, 8))
        manager_search_var = tk.StringVar()
        manager_search = tk.Entry(
            filter_bar,
            textvariable=manager_search_var,
            relief=tk.FLAT,
            bd=0,
            bg="#ffffff",
            fg=TEXT_PRIMARY,
            insertbackground=ACCENT,
            font=("Microsoft YaHei UI", 9),
            highlightbackground=BORDER,
            highlightcolor=ACCENT,
            highlightthickness=1,
        )
        manager_search.grid(row=0, column=1, sticky="ew", ipady=5)
        manager_provider_var = tk.StringVar(value="全部供应商")
        manager_provider = RoundedChoice(
            filter_bar,
            variable=manager_provider_var,
            values=(
                "全部供应商",
                *dict.fromkeys(spec.provider for spec in MODEL_SPECS),
            ),
            width=218,
            height=38,
            bg=APP_BG,
        )
        manager_provider.grid(row=0, column=2, padx=(8, 0))
        manager_count_var = tk.StringVar()
        tk.Label(
            filter_bar,
            textvariable=manager_count_var,
            bg=APP_BG,
            fg=TEXT_SECONDARY,
            font=("Microsoft YaHei UI", 8),
            anchor=tk.E,
        ).grid(row=0, column=3, sticky="e", padx=(10, 0))

        manager_quick_filter_var = tk.StringVar(value="all")
        tk.Label(
            filter_bar,
            text="快捷筛选",
            bg=APP_BG,
            fg=TEXT_SECONDARY,
            font=("Microsoft YaHei UI", 9, "bold"),
        ).grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        ModelFilterChips(
            filter_bar,
            variable=manager_quick_filter_var,
            bg=APP_BG,
        ).grid(row=1, column=1, columnspan=3, sticky="w", pady=(8, 0))

        columns = ("provider", "model", "size", "scenario", "state")
        tree = ttk.Treeview(
            window,
            columns=columns,
            show="headings",
            height=12,
            style="Model.Treeview",
        )
        headings = {
            "provider": "供应商",
            "model": "模型",
            "size": "下载大小",
            "scenario": "推荐场景",
            "state": "状态",
        }
        widths = {
            "provider": 190,
            "model": 220,
            "size": 90,
            "scenario": 260,
            "state": 120,
        }
        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(column, width=widths[column], anchor=tk.W)
        tree_scrollbar = SlimScrollbar(
            window,
            command=tree.yview,
            width=12,
            bg=APP_BG,
            track="#dfe7f2",
            thumb="#aeb7c4",
            active_thumb=ACCENT,
        )
        tree.configure(yscrollcommand=tree_scrollbar.set)
        tree.grid(row=2, column=0, sticky="nsew", padx=(20, 0))
        tree_scrollbar.grid(row=2, column=1, sticky="ns", padx=(0, 20))

        detail = tk.Frame(
            window,
            bg=PANEL_BG,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        detail.grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=20,
            pady=(10, 0),
        )
        detail.columnconfigure(1, weight=1)
        detail_title_var = tk.StringVar(value="选择一个模型查看详情")
        detail_meta_var = tk.StringVar()
        detail_path_var = tk.StringVar()
        detail_source_var = tk.StringVar()
        detail_description_var = tk.StringVar()
        action_buttons: dict[str, RoundedButton] = {}

        def sync_action_buttons(model_id: str | None) -> None:
            if not action_buttons:
                return
            valid = bool(model_id and model_id in MODEL_BY_ID)
            cached = self._model_is_cached(model_id) if valid and model_id else False
            removable = (
                self._model_has_user_cache_data(model_id)
                if valid and model_id
                else False
            )
            has_terms = bool(
                get_model_spec(model_id).terms_url
                if valid and model_id
                else False
            )
            action_buttons["download"].set_disabled(not valid or cached)
            action_buttons["remove"].set_disabled(not valid or not removable)
            action_buttons["source"].set_disabled(not valid)
            action_buttons["terms"].set_disabled(not valid or not has_terms)
        tk.Label(
            detail,
            textvariable=detail_title_var,
            bg=PANEL_BG,
            fg=TEXT_PRIMARY,
            font=("Microsoft YaHei UI", 10, "bold"),
            anchor=tk.W,
        ).grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=(9, 2))
        tk.Label(
            detail,
            textvariable=detail_meta_var,
            bg=PANEL_BG,
            fg=TEXT_SECONDARY,
            font=("Microsoft YaHei UI", 8),
            anchor=tk.W,
        ).grid(row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 2))
        tk.Label(
            detail,
            textvariable=detail_description_var,
            bg=PANEL_BG,
            fg=TEXT_PRIMARY,
            font=("Microsoft YaHei UI", 8),
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=850,
        ).grid(row=2, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 3))
        tk.Label(
            detail,
            textvariable=detail_path_var,
            bg=PANEL_BG,
            fg=TEXT_SECONDARY,
            font=("Consolas", 8),
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=850,
        ).grid(row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 2))
        tk.Label(
            detail,
            textvariable=detail_source_var,
            bg=PANEL_BG,
            fg=TEXT_SECONDARY,
            font=("Consolas", 8),
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=850,
        ).grid(row=4, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 9))

        def update_detail(model_id: str | None = None) -> None:
            if model_id is None:
                selection = tree.selection()
                model_id = selection[0] if selection else None
            if not model_id or model_id not in MODEL_BY_ID:
                detail_title_var.set("选择一个模型查看详情")
                detail_meta_var.set("")
                detail_description_var.set("")
                detail_path_var.set("")
                detail_source_var.set("")
                sync_action_buttons(None)
                return
            spec = get_model_spec(model_id)
            cached = self._model_is_cached(model_id)
            status = self._model_status_label(model_id, cached=cached)
            detail_title_var.set(
                ("★ 推荐  ·  " if spec.recommended else "")
                + f"{spec.display_name}  ·  {status}"
            )
            terms_state = "需单独确认" if spec.requires_terms_ack else "无附加确认"
            detail_meta_var.set(
                f"供应商：{spec.provider}    模型 ID：{spec.model_id}    "
                f"后端：{spec.backend_label}    语言：{spec.languages}    "
                f"体积：{spec.size_label}    条款：{terms_state}"
            )
            detail_description_var.set(
                f"{spec.description}  推荐场景：{spec.best_for}。"
            )
            if spec.requires_terms_ack:
                detail_description_var.set(
                    detail_description_var.get()
                    + f"  许可/限制：{spec.license_label}；{spec.usage_restriction}"
                )
            detail_path_var.set(
                f"用户缓存：{self._model_user_cache_path(model_id)}"
            )
            source_text = f"下载源：{spec.download_url}"
            if spec.terms_url:
                source_text += f"\n上游条款：{spec.terms_url}"
            detail_source_var.set(source_text)
            sync_action_buttons(model_id)

        def open_source() -> None:
            selection = tree.selection()
            if not selection:
                return
            try:
                webbrowser.open(get_model_spec(selection[0]).download_url, new=2)
            except (OSError, ValueError, webbrowser.Error) as exc:
                messagebox.showerror("无法打开下载源", str(exc), parent=window)

        def open_terms() -> None:
            selection = tree.selection()
            if not selection:
                return
            terms_url = get_model_spec(selection[0]).terms_url
            if not terms_url:
                self.status_var.set("当前模型没有单独的附加条款")
                return
            try:
                webbrowser.open(terms_url, new=2)
            except (OSError, ValueError, webbrowser.Error) as exc:
                messagebox.showerror("无法打开上游条款", str(exc), parent=window)

        def refresh() -> None:
            selected = tree.selection()
            query = manager_search_var.get()
            selected_provider = manager_provider_var.get()
            quick_filter = manager_quick_filter_var.get()
            for item in tree.get_children():
                tree.delete(item)
            matches = []
            for spec in MODEL_SPECS:
                cached = self._model_is_cached(spec.model_id)
                if (
                    (
                        selected_provider == "全部供应商"
                        or spec.provider == selected_provider
                    )
                    and model_matches_query(spec, query)
                    and model_matches_quick_filter(
                        spec,
                        quick_filter,
                        cached=cached,
                    )
                ):
                    matches.append((spec, cached))
            cached_count = 0
            partial_count = 0
            cached_bytes = 0
            for spec, cached in matches:
                if cached:
                    cached_count += 1
                    cached_size = self._model_cache_size(spec.model_id)
                    cached_bytes += cached_size
                    cached_mb = cached_size / (1024 * 1024)
                    state = (
                        f"{self._model_status_label(spec.model_id, cached=cached)} "
                        f"{cached_mb:.0f}MB"
                    )
                else:
                    state = self._model_status_label(spec.model_id, cached=False)
                    if state.startswith("下载未完成"):
                        partial_count += 1
                tree.insert(
                    "",
                    tk.END,
                    iid=spec.model_id,
                    values=(
                        spec.provider,
                        (
                            "★ 当前 · "
                            if spec.model_id == self._selected_model_id()
                            else ""
                        )
                        + spec.display_name,
                        spec.size_label,
                        spec.best_for,
                        state,
                    ),
                )
            manager_count_var.set(
                f"显示 {len(matches)} / {len(MODEL_SPECS)}  ·  "
                f"已缓存 {cached_count}  ·  断点 {partial_count}  ·  "
                f"占用 {cached_bytes / (1024 * 1024):.0f}MB"
            )
            target = (
                selected[0]
                if selected and tree.exists(selected[0])
                else self._selected_model_id()
            )
            if tree.exists(target):
                tree.selection_set(target)
                tree.focus(target)
                update_detail(target)
            else:
                update_detail(None)

        def selected_id() -> str | None:
            selection = tree.selection()
            return selection[0] if selection else None

        def download() -> None:
            model_id = selected_id()
            if model_id is None:
                return
            if not self._ensure_model_terms_accepted(model_id):
                return
            if self._model_is_cached(model_id):
                self.status_var.set(f"{get_model_spec(model_id).display_name} 已下载")
                return
            window.destroy()
            self._set_busy(
                True,
                message=f"正在下载 {get_model_spec(model_id).display_name}...",
                show_cancel=True,
            )
            threading.Thread(
                target=self._prepare_model_worker,
                args=(model_id,),
                daemon=True,
            ).start()

        def remove() -> None:
            model_id = selected_id()
            if model_id is None:
                return
            spec = get_model_spec(model_id)
            if not self._model_has_user_cache_data(model_id):
                if self._model_is_bundled_only(model_id):
                    self.status_var.set(
                        f"{spec.display_name} 为随包内置模型，不能删除"
                    )
                else:
                    self.status_var.set(
                        f"{spec.display_name} 没有可删除的用户缓存"
                    )
                return
            if not messagebox.askyesno(
                "删除模型",
                f"删除 {spec.display_name} 的本地缓存？以后使用时可以重新下载。",
                parent=window,
            ):
                return
            try:
                self._reset_recognizer()
                if spec.uses_paddle_runtime:
                    remove_paddle_model(model_id)
                else:
                    remove_external_model(model_id)
            except (OSError, ValueError) as exc:
                messagebox.showerror("删除失败", str(exc), parent=window)
            refresh()
            self._update_model_summary()

        actions = tk.Frame(window, bg=APP_BG)
        actions.grid(row=4, column=0, columnspan=2, sticky="e", padx=20, pady=18)

        def add_action(
            key: str,
            text: str,
            command,
            width: int,
            *,
            primary: bool = False,
        ) -> None:
            button = RoundedButton(
                actions,
                text=text,
                command=command,
                width=width,
                height=34,
                radius=11,
                bg=ACCENT if primary else "#ffffff",
                active_bg=ACCENT_DARK if primary else ACCENT_SOFT,
                fg="#ffffff" if primary else TEXT_PRIMARY,
                border=ACCENT if primary else BORDER,
                font=("Microsoft YaHei UI", 9, "bold" if primary else "normal"),
            )
            button.pack(side=tk.LEFT, padx=(0 if not action_buttons else 7, 0))
            action_buttons[key] = button

        add_action("download", "下载", download, 72, primary=True)
        add_action("remove", "删除缓存", remove, 84)
        add_action("source", "打开下载源", open_source, 96)
        add_action("terms", "上游条款", open_terms, 84)
        add_action("cache", "打开缓存目录", self._open_model_cache, 108)
        add_action("runtime", "运行环境", self.show_runtime_info, 84)
        add_action("refresh", "刷新", refresh, 64)
        add_action("close", "关闭", window.destroy, 64)
        tree.bind("<<TreeviewSelect>>", lambda _event: update_detail())
        manager_search_var.trace_add("write", lambda *_args: refresh())
        manager_provider_var.trace_add("write", lambda *_args: refresh())
        manager_quick_filter_var.trace_add("write", lambda *_args: refresh())
        manager_search.bind("<Escape>", lambda _event: manager_search_var.set(""))
        refresh()

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
            return "可用" if importlib.util.find_spec(module_name) else "未找到"
        except (ImportError, ModuleNotFoundError, ValueError):
            return "未找到"

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
            ("Paddle / PaddleX", self._module_status("paddle") + " / " + self._module_status("paddlex")),
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
            backend = get_model_spec(model_id).backend
            if backend == "rapid_onnx":
                ensure_rapid_model(
                    progress_callback=self._queue_model_download_progress,
                )
            elif backend == "mathcraft_onnx":
                ensure_mathcraft_model(
                    progress_callback=self._queue_model_download_progress,
                )
            elif backend == "pix2text_onnx":
                ensure_pix2text_model(
                    progress_callback=self._queue_model_download_progress,
                )
            elif backend == "mixtex_onnx":
                ensure_mixtex_model(
                    progress_callback=self._queue_model_download_progress,
                )
            elif backend == "unimernet_onnx":
                ensure_unimernet_onnx_model(
                    progress_callback=self._queue_model_download_progress,
                )
            elif backend == "paddle_hf":
                ensure_paddle_hf_model(
                    model_id,
                    progress_callback=self._queue_model_download_progress,
                )
            else:
                ensure_official_model(
                    model_id,
                    progress_callback=self._queue_model_download_progress,
                )
            self.worker_queue.put(("model_ready", model_id))
        except ModelDownloadCancelled:
            self.worker_queue.put(("download_cancelled", model_id))
        except Exception as exc:
            self.worker_queue.put(("error", str(exc)))

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
            image = Image.open(file_path)
            self._set_image(image)
            self.status_var.set(f"已加载图片：{file_path}")
        except Exception as exc:
            messagebox.showerror("打开失败", str(exc))

    def paste_image(self) -> None:
        try:
            data = ImageGrab.grabclipboard()
        except Exception as exc:
            messagebox.showerror("粘贴失败", str(exc))
            return

        if isinstance(data, Image.Image):
            self._set_image(data)
            self.status_var.set("已从剪贴板加载图片")
            return

        if isinstance(data, list) and data:
            try:
                image = Image.open(data[0])
                self._set_image(image)
                self.status_var.set(f"已从剪贴板文件加载图片：{data[0]}")
                return
            except Exception as exc:
                messagebox.showerror("粘贴失败", str(exc))
                return

        messagebox.showinfo("没有图片", "剪贴板里没有可用的图片。")

    def capture_screen(self) -> None:
        self.withdraw()
        self.after(180, self._start_capture_overlay)

    def _start_capture_overlay(self) -> None:
        selector = ScreenshotSelector(self, self._on_screen_captured)
        selector.start()

    def _on_screen_captured(self, image: Image.Image | None) -> None:
        self.deiconify()
        self.lift()
        if image is None:
            self.status_var.set("截图已取消")
            return
        self._set_image(image)
        self.status_var.set("已截取图片")

    def recognize_image(self) -> None:
        if self.is_busy:
            return
        if self.current_image_path is None:
            messagebox.showinfo("没有图片", "请先打开、粘贴或截图一张公式图片。")
            return

        settings = self._current_settings()
        if not self._ensure_model_terms_accepted(settings.model_name):
            return
        cached = self._model_is_cached(settings.model_name)
        if cached:
            busy_message = "正在加载已缓存模型并识别公式..."
        else:
            spec = get_model_spec(settings.model_name)
            busy_message = (
                f"首次使用，正在下载 {spec.display_name}（{spec.size_label}）并初始化..."
            )
        self._set_busy(True, message=busy_message, show_cancel=not cached)
        thread = threading.Thread(
            target=self._recognize_worker,
            args=(self.current_image_path, settings),
            daemon=True,
        )
        thread.start()

    def copy_latex(self) -> None:
        latex = self._current_latex()
        if not latex:
            self.status_var.set("没有可复制的 LaTeX")
            return
        self._copy_text(latex)
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
        except Exception as exc:
            messagebox.showerror("转换失败", str(exc))
            self.status_var.set("转换失败")
            return
        self._copy_text(value)
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

    def _copy_formula_result(self, latex: str) -> bool:
        try:
            mathml = latex_to_mathml(latex)
        except Exception as exc:
            write_log(f"Failed to prepare Word MathML clipboard: {exc}")
            self._copy_text(latex)
            return False
        return self._copy_mathml_for_word(mathml, plain_text=latex)

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
        self.status_var.set("MathML 预览已刷新")

    def _on_latex_modified(self, _event: tk.Event) -> None:
        if not self.output_text.edit_modified():
            return
        self.output_text.edit_modified(False)
        self._schedule_mathml_preview_update()

    def _schedule_mathml_preview_update(self) -> None:
        if self.mathml_update_after_id is not None:
            self.after_cancel(self.mathml_update_after_id)
        self.mathml_update_after_id = self.after(450, self._update_mathml_preview)

    def _update_mathml_preview(self) -> None:
        self.mathml_update_after_id = None
        latex = self._current_latex()
        if not latex:
            self._set_mathml_preview_text("暂无公式预览")
            return
        try:
            mathml = latex_to_mathml(latex)
        except Exception as exc:
            write_log(f"Failed to convert LaTeX to MathML: {exc}")
            self._set_mathml_preview_text("MathML 转换失败")
            return
        self.mathml_render_token += 1
        token = self.mathml_render_token
        self._set_mathml_preview_text("正在渲染 MathML...")
        thread = threading.Thread(
            target=self._render_mathml_preview_worker,
            args=(token, latex, mathml),
            daemon=True,
        )
        thread.start()

    def _set_mathml_preview_text(self, text: str) -> None:
        self.mathml_preview_photo = None
        self.mathml_preview_label.configure(image="", text=text, fg=TEXT_SECONDARY)

    def clear_output(self) -> None:
        if self.mathml_update_after_id is not None:
            self.after_cancel(self.mathml_update_after_id)
            self.mathml_update_after_id = None
        self.output_text.delete("1.0", tk.END)
        self.mathml_render_token += 1
        self._set_mathml_preview_text("暂无公式预览")
        self.status_var.set("结果已清空")

    def _recognize_worker(
        self, image_path: Path, settings: RecognizerSettings
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
                    },
                )
            )
        except ModelDownloadCancelled:
            self.worker_queue.put(("download_cancelled", settings.model_name))
        except PaddleOCRNotReadyError as exc:
            details = "".join(traceback.format_exception(exc)).strip()
            self.worker_queue.put(("error", f"{exc}\n\n{details}"))
        except Exception as exc:
            details = "".join(traceback.format_exception(exc)).strip()
            self.worker_queue.put(("error", f"{exc}\n\n{details}"))

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

        if kind == "progress":
            self.busy_status_message = str(payload)
            started_at = self.busy_started_at or time.time()
            elapsed = time.time() - started_at
            self.status_var.set(f"{self.busy_status_message} {elapsed:.1f}s")
            self._schedule_worker_poll()
            return

        if kind == "download_progress":
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
            self._set_busy(False)
            self._update_model_summary()
            self.status_var.set("下载已取消，已保留断点；下次可继续下载")
            self._schedule_worker_poll()
            return

        self._set_busy(False)
        if kind == "success":
            if not isinstance(payload, dict):
                raise TypeError("Unexpected recognition worker payload.")
            formula = str(payload.get("formula", ""))
            elapsed = float(payload.get("elapsed", 0.0))
            formula = clean_recognized_latex(formula)
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert("1.0", formula)
            self._update_mathml_preview()
            self.status_var.set(f"识别完成，用时 {elapsed:.2f} 秒；结果已保留")
        else:
            messagebox.showerror("识别失败", str(payload))
            self.status_var.set("识别失败")

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

    def _render_mathml_preview_worker(
        self,
        token: int,
        latex: str,
        mathml: str,
    ) -> None:
        try:
            image_path = self._render_mathml_to_png(token, mathml)
        except Exception as exc:
            fallback = latex_to_word_linear(latex) or latex
            self.mathml_preview_queue.put((token, "text", fallback))
            write_log(f"MathML browser render fallback: {exc}")
            return
        self.mathml_preview_queue.put((token, "image", str(image_path)))

    def _render_mathml_to_png(self, token: int, mathml: str) -> Path:
        browser = self._find_browser_executable()
        if browser is None:
            raise RuntimeError("Edge/Chrome was not found for MathML preview.")

        render_dir = CACHE_DIR / "mathml_preview"
        render_dir.mkdir(parents=True, exist_ok=True)
        html_path = render_dir / f"preview_{token}.html"
        png_path = render_dir / f"preview_{token}.png"
        profile_dir = render_dir / f"profile_{token}"
        try:
            png_path.unlink()
        except FileNotFoundError:
            pass
        html_path.write_text(self._mathml_preview_html(mathml), encoding="utf-8")

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
        process = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
            startupinfo=startupinfo,
        )
        try:
            self._wait_for_rendered_png(png_path, timeout=10.0)
            returncode = process.poll()
            if returncode not in (None, 0) and not png_path.exists():
                raise RuntimeError(f"Browser screenshot failed: {returncode}")
        finally:
            self._stop_preview_browser(process)
            shutil.rmtree(profile_dir, ignore_errors=True)

        self._trim_mathml_preview_image(png_path)
        return png_path

    def _wait_for_rendered_png(self, image_path: Path, *, timeout: float = 3.0) -> None:
        deadline = time.time() + timeout
        last_size = -1
        stable_count = 0
        while time.time() < deadline:
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

    def _stop_preview_browser(self, process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=2)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def _mathml_preview_html(self, mathml: str) -> str:
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

    def _trim_mathml_preview_image(self, image_path: Path) -> None:
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
            cropped = image.crop((left, top, right, bottom))
            cropped.save(image_path)

    def _set_mathml_preview_image(self, image_path: Path) -> None:
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGBA")
                self.update_idletasks()
                max_width = max(220, self.mathml_preview_label.winfo_width() - 24)
                max_height = max(160, self.mathml_preview_label.winfo_height() - 24)
                image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                self.mathml_preview_photo = ImageTk.PhotoImage(image)
        except Exception as exc:
            self._set_mathml_preview_text("MathML 预览加载失败")
            write_log(f"Failed to load MathML preview image: {exc}")
            return
        self.mathml_preview_label.configure(
            image=self.mathml_preview_photo,
            text="",
            bg="#fbfdff",
        )

    def _find_browser_executable(self) -> Path | None:
        candidates = [
            os.environ.get("FORMULA_OCR_BROWSER", ""),
            str(Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft" / "Edge" / "Application" / "msedge.exe"),
            str(Path(os.environ.get("ProgramFiles", "")) / "Microsoft" / "Edge" / "Application" / "msedge.exe"),
            str(Path(os.environ.get("ProgramFiles", "")) / "Google" / "Chrome" / "Application" / "chrome.exe"),
            str(Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "Application" / "chrome.exe"),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate)
            if path.exists():
                return path
        return None

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
        if self.recognizer is not None and self.recognizer_settings == settings:
            return self.recognizer
        self._reset_recognizer()
        self.recognizer_settings = settings
        self.recognizer = FormulaRecognizer(
            paddleocr_repo=DEFAULT_PADDLEOCR_REPO,
            model_name=settings.model_name,
            device="cpu",
            model_load_callback=self._queue_model_load_status,
            model_download_progress_callback=self._queue_model_download_progress,
        )
        return self.recognizer

    def _queue_model_load_status(self, model_name: str, cached: bool) -> None:
        if cached:
            message = f"正在加载已缓存模型 {get_model_spec(model_name).display_name} 并识别公式..."
            self.worker_queue.put(("progress", message))
        else:
            spec = get_model_spec(model_name)
            message = (
                f"首次使用 {spec.display_name}，正在下载模型（{spec.size_label}）；"
                "下载完成后将自动继续识别..."
            )
            # Expose cancellation before the downloader receives its first
            # network progress callback.
            self.worker_queue.put(
                (
                    "download_progress",
                    {"message": message, "percent": 0.0},
                )
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

    def _reset_recognizer(self) -> None:
        if self.recognizer is not None:
            self.recognizer.close()
        self.recognizer = None
        self.recognizer_settings = None

    def _current_settings(self) -> RecognizerSettings:
        return RecognizerSettings(model_name=self._selected_model_id())

    def _set_image(self, image: Image.Image) -> None:
        image = image.convert("RGB")
        self.current_image = image
        self.current_image_path = CACHE_DIR / "current_formula.png"
        image.save(self.current_image_path)
        self._update_preview()

    def _update_preview(self) -> None:
        if self.current_image is None:
            return
        self.update_idletasks()
        max_width = max(200, self.preview_label.winfo_width() - 24)
        max_height = max(160, self.preview_label.winfo_height() - 24)
        preview = self.current_image.copy()
        preview.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(preview)
        self.preview_label.configure(image=self.preview_photo, text="", bg=SURFACE_SUBTLE)

    def destroy(self) -> None:
        self.is_destroying = True
        # Ask an active downloader to stop at its next progress boundary.  The
        # downloader keeps its `.part` file, so closing the window is safe.
        self.download_cancel_event.set()
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
        if self.busy_status_after_id is not None:
            try:
                self.after_cancel(self.busy_status_after_id)
            except tk.TclError:
                pass
            self.busy_status_after_id = None
        self._reset_recognizer()
        super().destroy()


class ScreenshotSelector:
    def __init__(self, parent: tk.Tk, callback) -> None:
        self.parent = parent
        self.callback = callback
        self.start_x = 0
        self.start_y = 0
        self.rect_id: int | None = None
        self.window = tk.Toplevel(parent)
        self.window.withdraw()
        self.window.overrideredirect(True)
        self.window.attributes("-topmost", True)
        self.window.attributes("-alpha", 0.22)

        width = self.window.winfo_screenwidth()
        height = self.window.winfo_screenheight()
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
        x1, y1 = self.start_x, self.start_y
        x2, y2 = int(event.x_root), int(event.y_root)
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        self.window.destroy()

        if right - left < 5 or bottom - top < 5:
            self.callback(None)
            return

        try:
            image = ImageGrab.grab(bbox=(left, top, right, bottom))
        except TypeError:
            image = ImageGrab.grab((left, top, right, bottom))
        self.callback(image)

    def _cancel(self, _event: tk.Event | None = None) -> None:
        self.window.destroy()
        self.callback(None)


def main() -> None:
    install_exception_logger()
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
    LOG_DIR.mkdir(parents=True, exist_ok=True)

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
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(f"[{timestamp}] {message.rstrip()}\n")


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
        paddleocr_repo=DEFAULT_PADDLEOCR_REPO,
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
        image_path = app._render_mathml_to_png(token, mathml)
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
        app.download_cancel_event.set()
        try:
            app._queue_model_download_progress("RapidLaTeXOCR", 1, 2)
        except ModelDownloadCancelled:
            pass
        else:
            raise RuntimeError("下载取消信号未被进度回调拦截。")
        finally:
            app.download_cancel_event.clear()
        if not hasattr(app, "cancel_download_button"):
            raise RuntimeError("主界面缺少下载取消按钮。")
        if app.model_info_var.get().find(app._selected_model_id()) < 0:
            raise RuntimeError("主界面未显示当前模型标识。")
        if hasattr(app, "auto_copy_toggle") or hasattr(app, "auto_copy_var"):
            raise RuntimeError("主界面仍残留识别后自动复制控件。")
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
            app.model_picker._choose("MathCraftFormula")
            if app.model_picker.get() != "MathCraftFormula":
                raise RuntimeError("主界面模型下拉框无法选择已下载模型。")
            app.model_picker._choose("MixTexZhEn")
            if app.model_picker.get() != "MathCraftFormula":
                raise RuntimeError("主界面模型下拉框错误地选择了未下载模型。")
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
        manager.destroy()
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


def run_runtime_self_test() -> None:
    """Check the packaged-runtime/cache boundary without loading OCR models."""

    cache_dir = runtime_cache_dir().resolve()
    bundled_dir = FormulaOCRApp._bundled_runtime_dir().resolve()
    paddle_dir = paddle_model_dir(DEFAULT_MODEL_ID).resolve()
    external_dir = external_model_dir("RapidLaTeXOCR").resolve()
    mixtex_dir = external_model_dir("MixTexZhEn").resolve()
    unimernet_dir = external_model_dir("UniMERNetSmallONNX").resolve()

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

    message = (
        "runtime-self-test-ok\n"
        f"mode={'frozen' if getattr(sys, 'frozen', False) else 'source'}\n"
        f"internal={bundled_dir}\n"
        f"cache={cache_dir}\n"
        f"paddle_cache={paddle_dir}\n"
        f"onnx_cache={external_dir}\n"
        f"mixtex_cache={mixtex_dir}\n"
        f"unimernet_onnx_cache={unimernet_dir}\n"
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
