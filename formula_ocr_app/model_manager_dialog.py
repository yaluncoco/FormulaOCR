"""Model catalog browser and cache-management dialog."""

from __future__ import annotations

from collections.abc import Callable
import tkinter as tk
from tkinter import messagebox, ttk
import webbrowser

try:
    from formula_ocr_app.model_catalog import (
        MODEL_BY_ID,
        MODEL_SPECS,
        get_model_spec,
        model_matches_query,
        model_matches_quick_filter,
    )
    from formula_ocr_app.model_runtime import (
        is_model_bundled_only,
        is_model_cached,
        model_cache_size,
        model_has_user_cache_data,
        model_status_label,
        model_user_cache_path,
        remove_model,
    )
    from formula_ocr_app.runtime_paths import runtime_cache_dir
    from formula_ocr_app.ui_widgets import (
        ACCENT,
        ACCENT_DARK,
        ACCENT_SOFT,
        APP_BG,
        BORDER,
        PANEL_BG,
        TEXT_PRIMARY,
        TEXT_SECONDARY,
        ModelFilterChips,
        RoundedButton,
        RoundedChoice,
        SlimScrollbar,
    )
except ModuleNotFoundError as exc:  # Allows `python formula_ocr_app/app.py`.
    if exc.name != "formula_ocr_app":
        raise
    from model_catalog import (
        MODEL_BY_ID,
        MODEL_SPECS,
        get_model_spec,
        model_matches_query,
        model_matches_quick_filter,
    )
    from model_runtime import (
        is_model_bundled_only,
        is_model_cached,
        model_cache_size,
        model_has_user_cache_data,
        model_status_label,
        model_user_cache_path,
        remove_model,
    )
    from runtime_paths import runtime_cache_dir
    from ui_widgets import (
        ACCENT,
        ACCENT_DARK,
        ACCENT_SOFT,
        APP_BG,
        BORDER,
        PANEL_BG,
        TEXT_PRIMARY,
        TEXT_SECONDARY,
        ModelFilterChips,
        RoundedButton,
        RoundedChoice,
        SlimScrollbar,
    )


def show_model_manager_dialog(
    parent: tk.Misc,
    *,
    current_model_id: Callable[[], str],
    can_mutate_models: Callable[[], bool],
    request_download: Callable[[str], bool],
    before_remove: Callable[[], None],
    on_models_changed: Callable[[], None],
    set_status: Callable[[str], None],
    open_model_cache: Callable[[], None],
    show_runtime_info: Callable[[], None],
) -> tk.Toplevel:
    """Create the model manager while keeping app-level workflow in callbacks."""

    window = tk.Toplevel(parent)
    window.title("模型管理")
    window.geometry("1040x650")
    window.minsize(920, 520)
    window.configure(bg=APP_BG)
    window.transient(parent)
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
        mutable = can_mutate_models()
        cached = (
            is_model_cached(model_id, verify_hash=False)
            if valid and model_id
            else False
        )
        removable = (
            model_has_user_cache_data(model_id)
            if valid and model_id
            else False
        )
        has_terms = bool(
            get_model_spec(model_id).terms_url
            if valid and model_id
            else False
        )
        action_buttons["download"].set_disabled(not mutable or not valid or cached)
        action_buttons["remove"].set_disabled(
            not mutable or not valid or not removable
        )
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
        cached = is_model_cached(model_id, verify_hash=False)
        status = model_status_label(model_id, cached=cached)
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
        detail_path_var.set(f"用户缓存：{model_user_cache_path(model_id)}")
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
            set_status("当前模型没有单独的附加条款")
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
            # Rendering the complete catalog should only inspect manifests and
            # file sizes. Selecting/using a model still performs full hashes.
            cached = is_model_cached(spec.model_id, verify_hash=False)
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
            cached_size = model_cache_size(spec.model_id)
            cached_bytes += cached_size
            if cached:
                cached_count += 1
                cached_mb = cached_size / (1024 * 1024)
                state = (
                    f"{model_status_label(spec.model_id, cached=cached)} "
                    f"{cached_mb:.0f}MB"
                )
            else:
                state = model_status_label(spec.model_id, cached=False)
                if state.startswith("下载未完成"):
                    partial_count += 1
                    if cached_size:
                        state += f" {cached_size / (1024 * 1024):.0f}MB"
            tree.insert(
                "",
                tk.END,
                iid=spec.model_id,
                values=(
                    spec.provider,
                    (
                        "★ 当前 · "
                        if spec.model_id == current_model_id()
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
            else current_model_id()
        )
        if target and tree.exists(target):
            tree.selection_set(target)
            tree.focus(target)
            update_detail(target)
        else:
            update_detail(None)

    def selected_id() -> str | None:
        selection = tree.selection()
        return selection[0] if selection else None

    def mutation_allowed() -> bool:
        if can_mutate_models():
            return True
        set_status("正在识别或下载，暂时不能下载或删除模型")
        sync_action_buttons(selected_id())
        return False

    def download() -> None:
        if not mutation_allowed():
            return
        model_id = selected_id()
        if model_id is not None and request_download(model_id):
            window.destroy()

    def remove() -> None:
        if not mutation_allowed():
            return
        model_id = selected_id()
        if model_id is None:
            return
        spec = get_model_spec(model_id)
        if not model_has_user_cache_data(model_id):
            if is_model_bundled_only(model_id):
                set_status(f"{spec.display_name} 为随包内置模型，不能删除")
            else:
                set_status(f"{spec.display_name} 没有可删除的用户缓存")
            return
        if not messagebox.askyesno(
            "删除模型",
            f"删除 {spec.display_name} 的本地缓存？以后使用时可以重新下载。",
            parent=window,
        ):
            return
        if not mutation_allowed():
            return
        try:
            before_remove()
            removed = remove_model(model_id)
        except (OSError, ValueError) as exc:
            messagebox.showerror("删除失败", str(exc), parent=window)
            on_models_changed()
            refresh()
            return
        on_models_changed()
        refresh()
        if removed:
            set_status(f"已删除 {spec.display_name} 的用户缓存")
        else:
            set_status(f"{spec.display_name} 没有可删除的用户缓存")

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
    add_action("cache", "打开缓存目录", open_model_cache, 108)
    add_action("runtime", "运行环境", show_runtime_info, 84)
    add_action("refresh", "刷新", refresh, 64)
    add_action("close", "关闭", window.destroy, 64)
    tree.bind("<<TreeviewSelect>>", lambda _event: update_detail())
    window.bind(
        "<<FormulaOCRModelMutationStateChanged>>",
        lambda _event: sync_action_buttons(selected_id()),
        add="+",
    )
    trace_bindings = (
        (
            manager_search_var,
            manager_search_var.trace_add("write", lambda *_args: refresh()),
        ),
        (
            manager_provider_var,
            manager_provider_var.trace_add("write", lambda *_args: refresh()),
        ),
        (
            manager_quick_filter_var,
            manager_quick_filter_var.trace_add("write", lambda *_args: refresh()),
        ),
    )

    def remove_traces(event: tk.Event) -> None:
        if event.widget is not window:
            return
        for variable, trace_id in trace_bindings:
            try:
                variable.trace_remove("write", trace_id)
            except tk.TclError:
                pass

    # Retain the variables until window destruction and make the lifecycle
    # observable to the UI self-test. Without explicit trace removal, repeatedly
    # opening the manager leaves Tcl callback commands behind.
    window._formula_ocr_trace_bindings = trace_bindings  # type: ignore[attr-defined]
    window.bind("<Destroy>", remove_traces, add="+")
    manager_search.bind("<Escape>", lambda _event: manager_search_var.set(""))
    refresh()
    return window
