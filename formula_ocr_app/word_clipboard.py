from __future__ import annotations

import ctypes
import sys
import time
import tkinter as tk
from dataclasses import dataclass

try:
    from formula_ocr_app.formula_formats import mathml_to_word_mathml
except ModuleNotFoundError as exc:  # Allows `python formula_ocr_app/app.py`.
    if exc.name != "formula_ocr_app":
        raise
    from formula_formats import mathml_to_word_mathml


# Kept as named constants so diagnostics can prove these legacy formats are not
# present on the clipboard.
FORMAT_OFFICE_OPEN_XML = "Office Open XML"
FORMAT_MATHML = "MathML"
FORMAT_MATHML_PRESENTATION = "MathML Presentation"
FORMAT_HTML = "HTML Format"
CF_UNICODETEXT = 13


@dataclass(frozen=True)
class WordClipboardPayload:
    plain_text: str
    mathml: str
    html: bytes


def build_word_clipboard_payload(mathml: str, *, plain_text: str) -> WordClipboardPayload:
    word_mathml = _strip_xml_declaration(mathml_to_word_mathml(mathml))
    return WordClipboardPayload(
        plain_text=plain_text,
        mathml=word_mathml,
        html=_cf_html(_mathml_clipboard_html(word_mathml)),
    )


def copy_mathml_for_word_to_clipboard(
    mathml: str,
    *,
    plain_text: str,
    clipboard_widget: tk.Misc | None = None,
    owner_hwnd: int | None = None,
) -> bool:
    payload = build_word_clipboard_payload(mathml, plain_text=plain_text)

    if sys.platform == "win32" and owner_hwnd:
        if write_windows_clipboard(payload, owner_hwnd=owner_hwnd):
            if clipboard_widget is not None:
                try:
                    clipboard_widget.update()
                except Exception:
                    pass
            return True
        return False

    if sys.platform != "win32" and clipboard_widget is not None:
        write_tk_clipboard(clipboard_widget, payload)
        return True
    return False


def write_windows_clipboard(
    payload: WordClipboardPayload,
    *,
    owner_hwnd: int,
) -> bool:
    if sys.platform != "win32":
        return False
    user32 = ctypes.windll.user32
    user32.OpenClipboard.argtypes = [ctypes.c_void_p]
    user32.OpenClipboard.restype = ctypes.c_int
    user32.EmptyClipboard.restype = ctypes.c_int
    user32.CloseClipboard.restype = ctypes.c_int
    user32.RegisterClipboardFormatW.argtypes = [ctypes.c_wchar_p]
    user32.RegisterClipboardFormatW.restype = ctypes.c_uint
    user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
    user32.SetClipboardData.restype = ctypes.c_void_p
    opened = False
    try:
        try:
            _open_windows_clipboard(owner_hwnd)
        except Exception:
            _open_windows_clipboard(None)
        opened = True
        if not user32.EmptyClipboard():
            raise ctypes.WinError()
        for name, data in _custom_format_bytes(payload):
            format_id = user32.RegisterClipboardFormatW(name)
            if not format_id:
                raise ctypes.WinError()
            _set_clipboard_bytes(format_id, data)
        _set_clipboard_bytes(
            CF_UNICODETEXT,
            (payload.plain_text + "\0").encode("utf-16le"),
        )
        return True
    except Exception:
        return False
    finally:
        if opened:
            user32.CloseClipboard()


def write_tk_clipboard(widget: tk.Misc, payload: WordClipboardPayload) -> None:
    widget.clipboard_clear()
    widget.clipboard_append(payload.html.decode("utf-8"), type=FORMAT_HTML)
    widget.clipboard_append(payload.plain_text)
    widget.update()


def windows_clipboard_formats() -> list[str]:
    user32 = ctypes.windll.user32
    user32.OpenClipboard.argtypes = [ctypes.c_void_p]
    user32.OpenClipboard.restype = ctypes.c_int
    user32.EnumClipboardFormats.argtypes = [ctypes.c_uint]
    user32.EnumClipboardFormats.restype = ctypes.c_uint
    user32.GetClipboardFormatNameW.argtypes = [
        ctypes.c_uint,
        ctypes.c_wchar_p,
        ctypes.c_int,
    ]
    _open_windows_clipboard(None)
    try:
        names: list[str] = []
        format_id = 0
        while True:
            format_id = user32.EnumClipboardFormats(format_id)
            if not format_id:
                return names
            names.append(_windows_clipboard_format_name(format_id))
    finally:
        user32.CloseClipboard()


def windows_clipboard_text(format_name: str) -> str:
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    user32.OpenClipboard.argtypes = [ctypes.c_void_p]
    user32.OpenClipboard.restype = ctypes.c_int
    user32.RegisterClipboardFormatW.argtypes = [ctypes.c_wchar_p]
    user32.RegisterClipboardFormatW.restype = ctypes.c_uint
    user32.GetClipboardData.argtypes = [ctypes.c_uint]
    user32.GetClipboardData.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalSize.argtypes = [ctypes.c_void_p]
    kernel32.GlobalSize.restype = ctypes.c_size_t
    format_id = user32.RegisterClipboardFormatW(format_name)
    if not format_id:
        return ""
    _open_windows_clipboard(None)
    try:
        handle = user32.GetClipboardData(format_id)
        if not handle:
            return ""
        size = kernel32.GlobalSize(handle)
        if not size:
            return ""
        pointer = kernel32.GlobalLock(handle)
        if not pointer:
            return ""
        try:
            raw = ctypes.string_at(pointer, size)
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()
    return raw.split(b"\0", 1)[0].decode("utf-8", "replace")


def tk_clipboard_text(root: tk.Misc, format_name: str) -> str:
    return root.selection_get(selection="CLIPBOARD", type=format_name)


def _custom_format_bytes(payload: WordClipboardPayload) -> tuple[tuple[str, bytes], ...]:
    return ((FORMAT_HTML, payload.html),)


def _mathml_clipboard_html(mathml: str) -> str:
    mathml = _strip_xml_declaration(mathml)
    return (
        "<!doctype html>"
        '<html><head><meta charset="utf-8"></head><body>'
        "<!--StartFragment-->"
        f"{mathml}"
        "<!--EndFragment-->"
        "</body></html>"
    )


def _strip_xml_declaration(mathml: str) -> str:
    mathml = mathml.strip()
    if mathml.startswith("<?xml"):
        _, _, mathml = mathml.partition("?>")
    return mathml.strip()


def _cf_html(html: str) -> bytes:
    prefix = (
        "Version:1.0\r\n"
        "StartHTML:{start_html:010d}\r\n"
        "EndHTML:{end_html:010d}\r\n"
        "StartFragment:{start_fragment:010d}\r\n"
        "EndFragment:{end_fragment:010d}\r\n"
        "StartSelection:{start_fragment:010d}\r\n"
        "EndSelection:{end_fragment:010d}\r\n"
        "SourceURL:about:blank\r\n"
    )
    empty_header = prefix.format(
        start_html=0,
        end_html=0,
        start_fragment=0,
        end_fragment=0,
    )
    start_html = len(empty_header.encode("utf-8"))
    start_marker = "<!--StartFragment-->"
    end_marker = "<!--EndFragment-->"
    start_fragment = start_html + len(html[: html.index(start_marker)].encode("utf-8"))
    start_fragment += len(start_marker.encode("utf-8"))
    end_fragment = start_html + len(html[: html.index(end_marker)].encode("utf-8"))
    end_html = start_html + len(html.encode("utf-8"))
    header = prefix.format(
        start_html=start_html,
        end_html=end_html,
        start_fragment=start_fragment,
        end_fragment=end_fragment,
    )
    return (header + html).encode("utf-8")


def _open_windows_clipboard(
    owner_hwnd: int | None,
    *,
    retries: int = 12,
    delay_seconds: float = 0.025,
) -> None:
    user32 = ctypes.windll.user32
    hwnd = ctypes.c_void_p(owner_hwnd) if owner_hwnd else None
    last_error = 0
    for attempt in range(retries):
        if user32.OpenClipboard(hwnd):
            return
        try:
            last_error = ctypes.GetLastError()
        except AttributeError:
            last_error = 0
        if attempt + 1 < retries:
            time.sleep(delay_seconds)
    raise ctypes.WinError(last_error)


def _set_clipboard_bytes(format_id: int, payload: bytes) -> None:
    kernel32 = ctypes.windll.kernel32
    kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalFree.argtypes = [ctypes.c_void_p]
    kernel32.GlobalFree.restype = ctypes.c_void_p
    gmem_moveable = 0x0002
    handle = kernel32.GlobalAlloc(gmem_moveable, len(payload))
    if not handle:
        raise ctypes.WinError()
    locked = kernel32.GlobalLock(handle)
    if not locked:
        kernel32.GlobalFree(handle)
        raise ctypes.WinError()
    try:
        ctypes.memmove(locked, payload, len(payload))
    finally:
        kernel32.GlobalUnlock(handle)
    if not ctypes.windll.user32.SetClipboardData(format_id, handle):
        kernel32.GlobalFree(handle)
        raise ctypes.WinError()


def _windows_clipboard_format_name(format_id: int) -> str:
    standard_formats = {
        1: "CF_TEXT",
        7: "CF_OEMTEXT",
        13: "CF_UNICODETEXT",
        16: "CF_LOCALE",
    }
    if format_id in standard_formats:
        return standard_formats[format_id]
    buffer = ctypes.create_unicode_buffer(256)
    ctypes.windll.user32.GetClipboardFormatNameW(format_id, buffer, 256)
    return buffer.value or str(format_id)
