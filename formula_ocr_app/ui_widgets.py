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
    "ModelFilterChips",
    "ModelPicker",
    "PANEL_BG",
    "RoundedButton",
    "RoundedChoice",
    "RoundedPanel",
    "SURFACE_SUBTLE",
    "SlimScrollbar",
    "TEXT_PRIMARY",
    "TEXT_SECONDARY",
    "_ScreenArea",
    "_anchored_popup_geometry",
    "_enable_popup_row_keyboard_navigation",
    "_monitor_work_area",
    "_rounded_rect",
    "_show_anchored_popup",
]


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
) -> None:
    if not rows:
        return

    def focus_row(index: int) -> str:
        if popup.winfo_exists():
            rows[index % len(rows)].focus_set()
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
        keyboard_rows: list[tk.Widget] = []
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
        _enable_popup_row_keyboard_navigation(
            popup,
            keyboard_rows,
            initial_index=selected_index,
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
