from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass
import tkinter as tk
import tkinter.font as tkfont

try:
    from formula_ocr_app.model_catalog import MODEL_QUICK_FILTERS
except ModuleNotFoundError as exc:  # Allows `python formula_ocr_app/app.py`.
    if exc.name != "formula_ocr_app":
        raise
    from model_catalog import MODEL_QUICK_FILTERS


APP_BG = "#eef3f8"
PANEL_BG = "#ffffff"
SURFACE_SUBTLE = "#f7f9fc"
TEXT_PRIMARY = "#172033"
TEXT_SECONDARY = "#657086"
ACCENT = "#d4237a"
ACCENT_DARK = "#b71f69"
ACCENT_SOFT = "#fde7f2"
BORDER = "#dce4ef"


__all__ = [
    "ACCENT",
    "ACCENT_DARK",
    "ACCENT_SOFT",
    "APP_BG",
    "BORDER",
    "FlowFrame",
    "ModelFilterChips",
    "ModelPicker",
    "PANEL_BG",
    "RoundedButton",
    "RoundedChoice",
    "RoundedPanel",
    "ResponsiveRow",
    "SURFACE_SUBTLE",
    "SlimScrollbar",
    "TEXT_PRIMARY",
    "TEXT_SECONDARY",
    "Toast",
    "WrappingLabel",
    "fit_window_to_screen",
    "ui_pixels",
    "_ScreenArea",
    "_anchored_popup_geometry",
    "_enable_popup_row_keyboard_navigation",
    "_monitor_work_area",
    "_rounded_rect",
    "_show_anchored_popup",
]


def ui_pixels(widget: tk.Misc, value: float) -> int:
    """Scale a 96-DPI pixel dimension alongside Tk's point-sized fonts."""

    return max(1, round(value * float(widget.tk.call("tk", "scaling")) * 0.75))


def fit_window_to_screen(
    window: tk.Toplevel | tk.Tk,
    size: tuple[int, int],
    minimum: tuple[int, int],
) -> None:
    area = _monitor_work_area(window.master or window)
    available_width = max(1, area.right - area.left - ui_pixels(window, 32))
    available_height = max(1, area.bottom - area.top - ui_pixels(window, 64))
    width = min(ui_pixels(window, size[0]), available_width)
    height = min(ui_pixels(window, size[1]), available_height)
    window.geometry(f"{width}x{height}")
    window.minsize(
        min(ui_pixels(window, minimum[0]), width),
        min(ui_pixels(window, minimum[1]), height),
    )


class WrappingLabel(tk.Label):
    """Wrap to the allocated width without forcing the window to grow."""

    def __init__(self, parent: tk.Misc, **kwargs) -> None:
        kwargs.setdefault("width", 1)
        kwargs.setdefault("wraplength", ui_pixels(parent, 600))
        kwargs.setdefault("justify", tk.LEFT)
        kwargs.setdefault("anchor", tk.W)
        super().__init__(parent, **kwargs)
        self.bind("<Configure>", self._resize, add="+")

    def _resize(self, event: tk.Event) -> None:
        width = max(1, event.width - 2 * int(self.cget("padx")) - 4)
        if int(self.cget("wraplength")) != width:
            self.configure(wraplength=width)


class FlowFrame(tk.Frame):
    """A toolbar that keeps controls at their natural size and wraps rows."""

    def __init__(self, parent: tk.Misc, *, gap: int = 8, align: str = "left", **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self.gap = ui_pixels(self, gap)
        self.align = align
        self.items: list[tk.Widget] = []
        self._layout_after_id: str | None = None
        self.bind("<Configure>", self._schedule_layout, add="+")
        self.bind("<Destroy>", self._cancel_layout, add="+")

    @property
    def preferred_width(self) -> int:
        return sum(item.winfo_reqwidth() for item in self.items) + self.gap * max(0, len(self.items) - 1)

    def add(self, widget: tk.Widget) -> None:
        self.items.append(widget)
        widget.bind("<Configure>", self._schedule_layout, add="+")
        self._schedule_layout()

    def _schedule_layout(self, _event: tk.Event | None = None) -> None:
        if self._layout_after_id is None:
            self._layout_after_id = self.after_idle(self._layout)

    def _layout(self) -> None:
        self._layout_after_id = None
        if not self.items:
            return
        minimum_width = max(item.winfo_reqwidth() for item in self.items)
        available = max(minimum_width, self.winfo_width())
        rows: list[list[tk.Widget]] = [[]]
        row_width = 0
        for item in self.items:
            width = item.winfo_reqwidth()
            if rows[-1] and row_width + self.gap + width > available:
                rows.append([])
                row_width = 0
            row_width += (self.gap if rows[-1] else 0) + width
            rows[-1].append(item)
        y = 0
        for row in rows:
            height = max(item.winfo_reqheight() for item in row)
            width = sum(item.winfo_reqwidth() for item in row) + self.gap * (len(row) - 1)
            x = max(0, available - width) if self.align == "right" else 0
            for item in row:
                item.place(x=x, y=y + (height - item.winfo_reqheight()) // 2)
                x += item.winfo_reqwidth() + self.gap
            y += height + self.gap
        self.configure(width=minimum_width, height=y - self.gap)

    def _cancel_layout(self, event: tk.Event) -> None:
        if event.widget is self and self._layout_after_id is not None:
            self.after_cancel(self._layout_after_id)
            self._layout_after_id = None


class ResponsiveRow(tk.Frame):
    """Keep a heading beside its actions when they fit, otherwise stack them."""

    def __init__(self, parent: tk.Misc, *, gap: int = 8, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self.gap = ui_pixels(self, gap)
        self.leading: tk.Widget | None = None
        self.trailing: tk.Widget | None = None
        self._stacked: bool | None = None
        self.columnconfigure(1, weight=1)
        self.bind("<Configure>", self._layout, add="+")

    def set_widgets(self, leading: tk.Widget, trailing: tk.Widget) -> None:
        self.leading, self.trailing = leading, trailing
        leading.bind("<Configure>", self._layout, add="+")
        trailing.bind("<Configure>", self._layout, add="+")
        self._layout()

    def _layout(self, _event: tk.Event | None = None) -> None:
        if self.leading is None or self.trailing is None:
            return
        trailing_width = self.trailing.preferred_width if isinstance(self.trailing, FlowFrame) else self.trailing.winfo_reqwidth()
        stacked = self.leading.winfo_reqwidth() + trailing_width + self.gap > self.winfo_width()
        if self._stacked == stacked:
            return
        self._stacked = stacked
        if stacked:
            self.leading.grid(row=0, column=0, columnspan=2, sticky="w")
            self.trailing.grid(row=1, column=0, columnspan=2, sticky="ew", padx=0, pady=(self.gap, 0))
        else:
            self.leading.grid(row=0, column=0, columnspan=1, sticky="w")
            self.trailing.grid(row=0, column=1, columnspan=1, sticky="e" if not isinstance(self.trailing, FlowFrame) else "ew", padx=(self.gap, 0), pady=0)


class Toast(tk.Frame):
    """One reusable, non-modal notice; repeated actions restart its timer."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, bd=0, highlightthickness=1)
        self.label = tk.Label(
            self, font=("Microsoft YaHei UI", 10, "bold"),
            padx=ui_pixels(self, 14), pady=ui_pixels(self, 9), justify=tk.LEFT,
        )
        self.label.pack()
        self._after_id: str | None = None
        self.bind("<Destroy>", self._on_destroy, add="+")

    def show(self, text: str, *, anchor: tk.Widget, warning: bool = False, duration: int = 2400) -> None:
        self.hide()
        bg, fg, border = ("#fff4df", "#8a5a13", "#f2d39a") if warning else ("#e8f7ef", "#187044", "#b7dec8")
        self.configure(bg=bg, highlightbackground=border)
        margin = ui_pixels(self, 12)
        available = max(1, self.master.winfo_width() - 2 * margin)
        self.label.configure(
            text=text, bg=bg, fg=fg,
            wraplength=max(1, min(ui_pixels(self, 400), available - ui_pixels(self, 30))),
        )
        self.update_idletasks()
        width, height = min(available, self.winfo_reqwidth()), self.winfo_reqheight()
        x = anchor.winfo_rootx() - self.master.winfo_rootx() + anchor.winfo_width() - width
        y = anchor.winfo_rooty() - self.master.winfo_rooty() + anchor.winfo_height() + margin
        x = max(margin, min(x, self.master.winfo_width() - width - margin))
        y = max(margin, min(y, self.master.winfo_height() - height - margin))
        self.place(x=x, y=y, width=width, height=height)
        self.lift()
        self._after_id = self.after(duration, self.hide)

    def hide(self) -> None:
        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None
        self.place_forget()

    def _on_destroy(self, event: tk.Event) -> None:
        if event.widget is self and self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None


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
            popup.geometry(f"{width}x{height}{x:+d}{y:+d}")
    else:
        popup.geometry(f"{width}x{height}{x:+d}{y:+d}")
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


def _enable_popup_row_keyboard_navigation(
    popup: tk.Toplevel,
    rows: list[tk.Widget],
    *,
    initial_index: int = 0,
    see_row=None,
) -> None:
    if not rows:
        return

    def focus_row(index: int) -> str:
        if popup.winfo_exists():
            row = rows[index % len(rows)]
            row.focus_set()
            if see_row is not None:
                see_row(row)
        return "break"

    for index, row in enumerate(rows):
        row.bind("<Up>", lambda _event, item=index: focus_row(item - 1))
        row.bind("<Down>", lambda _event, item=index: focus_row(item + 1))
        row.bind("<Home>", lambda _event: focus_row(0))
        row.bind("<End>", lambda _event: focus_row(len(rows) - 1))
    popup.after_idle(lambda: focus_row(initial_index))


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
            cursor="hand2",
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
        self._font = tkfont.Font(root=self, font=font)
        self._minimum_width = width
        self._minimum_height = height
        self._hover = False
        self.is_selected = False
        self.is_disabled = False
        self._update_requested_size()
        self._draw()
        self.bind("<Configure>", lambda _event: self._draw())
        self.bind("<Enter>", lambda _event: self._set_hover(True))
        self.bind("<Leave>", lambda _event: self._set_hover(False))
        self.bind("<FocusIn>", lambda _event: self._draw())
        self.bind("<FocusOut>", lambda _event: self._draw())
        self.bind("<Button-1>", self._click)
        self.bind("<Return>", self._keyboard_click)
        self.bind("<space>", self._keyboard_click)

    def set_selected(self, selected: bool) -> None:
        self.is_selected = selected
        self._draw()

    def set_text(self, text: str) -> None:
        self.text = text
        self._update_requested_size()
        self._draw()

    def set_width(self, width: int) -> None:
        self._minimum_width = width
        self._update_requested_size()
        self._draw()

    def set_disabled(self, disabled: bool) -> None:
        self.is_disabled = disabled
        self.configure(cursor="arrow" if disabled else "hand2", takefocus=not disabled)
        self._draw()

    def _update_requested_size(self) -> None:
        # Canvas dimensions are pixels, while positive Tk font sizes are points.
        # Both the minimum size and the measured caption must fit at this DPI.
        self.configure(
            width=max(ui_pixels(self, self._minimum_width), self._font.measure(self.text) + ui_pixels(self, 24)),
            height=max(ui_pixels(self, self._minimum_height), self._font.metrics("linespace") + ui_pixels(self, 12)),
        )

    def _set_hover(self, hover: bool) -> None:
        self._hover = hover
        self._draw()

    def _click(self, _event: tk.Event) -> None:
        if not self.is_disabled and self.command:
            self.command()

    def _keyboard_click(self, _event: tk.Event) -> str:
        self._click(_event)
        return "break"

    def _draw(self) -> None:
        self.delete("all")
        width = max(2, self.winfo_width() if self.winfo_width() > 1 else self.winfo_reqwidth())
        height = max(2, self.winfo_height() if self.winfo_height() > 1 else self.winfo_reqheight())
        radius = min(ui_pixels(self, self.radius), (width - 2) // 2, (height - 2) // 2)
        selected = self.is_selected and self.selected_bg is not None
        fill = self.selected_bg if selected else (self.active_bg if self._hover else self.normal_bg)
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
            radius,
            fill=fill,
            outline=outline,
            width=1,
        )
        self.create_text(
            width // 2,
            height // 2,
            text=self.text,
            fill=text_color,
            font=self._font,
            tags="caption",
        )
        if not self.is_disabled and self.focus_get() is self:
            _rounded_rect(
                self, 3, 3, width - 3, height - 3, max(1, radius - 2),
                fill="", outline="#ffffff" if fill == ACCENT else ACCENT,
                width=2, tags="focus",
            )


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
        font = self.button._font
        available = max(1, ui_pixels(self, self.choice_width - 24) - font.measure("  ▾"))
        if font.measure(value) <= available:
            return f"{value}  ▾"
        suffix = "…"
        low, high = 0, len(value)
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
        keyboard_rows: list[tk.Widget] = []
        for value in self.values:
            selected = value == current
            row_bg = ACCENT_SOFT if selected else PANEL_BG
            row = tk.Frame(
                card,
                bg=row_bg,
                cursor="hand2",
                takefocus=True,
                highlightthickness=1,
                highlightbackground=row_bg,
                highlightcolor=ACCENT,
                padx=10,
                pady=7,
            )
            row.pack(fill=tk.X, pady=1)
            keyboard_rows.append(row)
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

        try:
            initial_index = self.values.index(current)
        except ValueError:
            initial_index = 0
        _enable_popup_row_keyboard_navigation(
            popup,
            keyboard_rows,
            initial_index=initial_index,
        )

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


class ModelFilterChips(FlowFrame):
    """Reusable quick filters shared by the picker and model manager."""

    def __init__(
        self,
        parent: tk.Widget,
        *,
        variable: tk.StringVar | None = None,
        bg: str = PANEL_BG,
    ) -> None:
        super().__init__(parent, bg=bg, highlightthickness=0, bd=0, gap=5)
        self.variable = variable or tk.StringVar(master=self, value="all")
        valid_keys = {key for key, _label in MODEL_QUICK_FILTERS}
        if self.variable.get() not in valid_keys:
            self.variable.set("all")
        self.buttons: dict[str, RoundedButton] = {}
        for key, label in MODEL_QUICK_FILTERS:
            button = RoundedButton(
                self,
                text=label,
                command=lambda selected=key: self.variable.set(selected),
                width=52,
                height=28,
                radius=10,
                bg="#ffffff",
                active_bg=ACCENT_SOFT,
                border=BORDER,
                selected_bg=ACCENT,
                selected_fg="#ffffff",
                font=("Microsoft YaHei UI", 8, "bold"),
            )
            self.add(button)
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
        self.bind("<Destroy>", self._on_destroy, add="+")
        self._sync_button()

    def get(self) -> str:
        return self.model_id

    def set(self, model_id: str, *, notify: bool = False) -> None:
        if not model_id:
            changed = bool(self.model_id)
            self.model_id = ""
            self._sync_button()
            if changed and notify:
                self.command()
            return
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
            None,
        )
        if spec is not None and self._model_is_available(spec.model_id):
            label = f"{spec.compact_name}  ▾"
        else:
            label = "选择已下载模型  ▾"
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

        keyboard_rows: list[tk.Widget] = []
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
                    takefocus=True,
                    highlightthickness=1,
                    highlightbackground=row_bg,
                    highlightcolor=ACCENT,
                    padx=8,
                    pady=7,
                )
                row.pack(fill=tk.X, pady=2)
                keyboard_rows.append(row)
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
                meta = WrappingLabel(
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
                detail = WrappingLabel(
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
                row.bind(
                    "<Return>",
                    lambda _event, selected_id=spec.model_id: self._choose(
                        selected_id
                    ),
                )
                row.bind(
                    "<space>",
                    lambda _event, selected_id=spec.model_id: self._choose(
                        selected_id
                    ),
                )

        selected_index = next(
            (
                index
                for index, (spec, _state) in enumerate(available_models)
                if spec.model_id == self.model_id
            ),
            0,
        )
        def see_row(row: tk.Widget) -> None:
            list_canvas.update_idletasks()
            total = max(1, list_frame.winfo_height())
            top, bottom = row.winfo_y(), row.winfo_y() + row.winfo_height()
            view_top = list_canvas.yview()[0] * total
            view_bottom = view_top + list_canvas.winfo_height()
            if top < view_top:
                list_canvas.yview_moveto(top / total)
            elif bottom > view_bottom:
                list_canvas.yview_moveto((bottom - list_canvas.winfo_height()) / total)

        list_frame.update_idletasks()
        sync_scroll_region()
        list_canvas.yview_moveto(0.0)

        popup.update_idletasks()
        width = max(ui_pixels(self, 430), self.winfo_width() + ui_pixels(self, 140))
        work_area = _monitor_work_area(self)
        max_height = max(300, int((work_area.bottom - work_area.top) * 0.78))
        list_height = min(ui_pixels(self, 390), max(ui_pixels(self, 100), list_frame.winfo_reqheight()))
        list_shell.configure(height=list_height, width=width - ui_pixels(self, 24))
        list_shell.grid_propagate(False)
        popup.update_idletasks()
        height = min(popup.winfo_reqheight(), max_height)
        _show_anchored_popup(popup, self, width, height, gap=6, align="right")
        _enable_popup_row_keyboard_navigation(
            popup,
            keyboard_rows,
            initial_index=selected_index,
            see_row=see_row,
        )
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

    def _on_destroy(self, event: tk.Event) -> None:
        if event.widget is self:
            self._close_popup()


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
            width=ui_pixels(parent, width),
            height=1,
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
