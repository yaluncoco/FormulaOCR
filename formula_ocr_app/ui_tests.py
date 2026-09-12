"""Desktop regressions for scaled layouts and action feedback (no OCR models)."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
import tkinter as tk
from tkinter import ttk
import unittest
from unittest import mock

from PIL import Image

from formula_ocr_app import app as app_module
from formula_ocr_app import ui_widgets
from formula_ocr_app.app_settings import AppSettings
from formula_ocr_app.ui_widgets import RoundedButton, RoundedChoice, ScrollableFrame, ui_pixels


def descendants(widget: tk.Misc):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


def update_release(version: str = "99.0.0"):
    return app_module.ReleaseInfo(
        current_version=app_module.__version__, latest_version=version,
        tag_name=f"v{version}", release_name=f"FormulaOCR {version}",
        release_url=f"https://github.com/yaluncoco/FormulaOCR/releases/tag/v{version}",
        installer_url=f"https://github.com/yaluncoco/FormulaOCR/releases/download/v{version}/FormulaOCRSetup-{version}.exe",
        notes="改进更新提醒与界面交互。", published_at="2026-09-12T00:00:00Z",
    )


@unittest.skipUnless(sys.platform == "win32" or os.environ.get("DISPLAY"), "requires a desktop Tk display")
class DesktopInteractionTests(unittest.TestCase):
    @contextmanager
    def window(
        self, scale: float = 1.5, screen: tuple[int, int] | None = None,
        *, auto_check_updates: bool = False, update_fetcher=None,
    ):
        app_module.enable_windows_dpi_awareness()
        callbacks = []

        class ScaledApp(app_module.FormulaOCRApp):
            def _configure_styles(self):
                self.tk.call("tk", "scaling", scale * 96 / 72)
                super()._configure_styles()

            def report_callback_exception(self, exc, value, tb):
                callbacks.append("".join(traceback.format_exception(exc, value, tb)))

        with (
            tempfile.TemporaryDirectory(prefix="formulaocr-ui-test-") as scratch,
            mock.patch.object(app_module, "CACHE_DIR", Path(scratch)),
            mock.patch.object(app_module, "load_settings", return_value=AppSettings()),
            mock.patch.object(app_module, "is_model_cached", return_value=False),
            mock.patch.object(app_module, "model_status_label", return_value="待下载"),
            mock.patch.object(app_module, "write_log"),
            mock.patch.object(app_module.messagebox, "showerror") as errors,
            mock.patch.object(ui_widgets, "_monitor_work_area", return_value=ui_widgets._ScreenArea(0, 0, *screen)) if screen else nullcontext(),
        ):
            app = ScaledApp(auto_check_updates=auto_check_updates, update_fetcher=update_fetcher)
            try:
                app.update()
                yield app, errors
                self.assertEqual(callbacks, [], "Tk callback failed")
            finally:
                app.destroy()

    def assert_buttons_fit(self, window: tk.Misc) -> None:
        window.update()
        buttons = [item for item in descendants(window) if isinstance(item, RoundedButton) and item.winfo_ismapped()]
        self.assertTrue(buttons)
        for button in buttons:
            with self.subTest(caption=button.text):
                ancestor = button.master
                while ancestor is not None:
                    if isinstance(ancestor, ScrollableFrame):
                        ancestor.see(button)
                    ancestor = ancestor.master
                window.update()
                box = button.bbox("content")
                self.assertIsNotNone(box)
                self.assertGreaterEqual(box[0], 2)
                self.assertGreaterEqual(box[1], 2)
                self.assertLessEqual(box[2], button.winfo_width() - 2)
                self.assertLessEqual(box[3], button.winfo_height() - 2)
                left, top = button.winfo_rootx(), button.winfo_rooty()
                right, bottom = left + button.winfo_width(), top + button.winfo_height()
                ancestor = button.master
                while ancestor is not None:
                    self.assertGreaterEqual(left, ancestor.winfo_rootx(), str(ancestor))
                    self.assertGreaterEqual(top, ancestor.winfo_rooty(), str(ancestor))
                    self.assertLessEqual(right, ancestor.winfo_rootx() + ancestor.winfo_width(), str(ancestor))
                    self.assertLessEqual(bottom, ancestor.winfo_rooty() + ancestor.winfo_height(), str(ancestor))
                    if isinstance(ancestor, (tk.Toplevel, tk.Tk)):
                        break
                    ancestor = ancestor.master

    def assert_window_on_screen(self, window: tk.Misc) -> None:
        area = ui_widgets._monitor_work_area(window)
        self.assertGreaterEqual(window.winfo_rootx(), area.left)
        self.assertGreaterEqual(window.winfo_rooty(), area.top)
        self.assertLessEqual(window.winfo_rootx() + window.winfo_width(), area.right)
        self.assertLessEqual(window.winfo_rooty() + window.winfo_height(), area.bottom)

    @staticmethod
    def pump(app: tk.Tk, milliseconds: int) -> None:
        until = time.monotonic() + milliseconds / 1000
        while time.monotonic() < until:
            app.update()
            time.sleep(0.01)
        app.update()

    def test_main_window_at_five_scales_and_narrow_widths(self):
        for scale in (1.0, 1.25, 1.5, 1.75, 2.0):
            with self.subTest(scale=scale), self.window(scale) as (app, _errors):
                app.update_button.set_available_version("99.0.0")
                large = f"{app.winfo_width()}x{app.winfo_height()}"
                self.assert_window_on_screen(app)
                self.assert_buttons_fit(app)
                width, height = app.minsize()
                app.geometry(f"{width}x{height}")
                self.assert_buttons_fit(app)
                self.assertGreater(app.output_text.winfo_height(), ui_pixels(app, 45))
                app.geometry(large)
                self.assert_buttons_fit(app)

    def test_startup_update_badge_is_quiet_and_click_reuses_release_details(self):
        release = update_release()
        fetch = mock.Mock(return_value=release)
        with (
            self.window(auto_check_updates=True, update_fetcher=fetch) as (app, errors),
            mock.patch.object(app_module.messagebox, "askyesno", return_value=False) as prompt,
            mock.patch.object(app_module.messagebox, "showinfo") as info,
            mock.patch.object(app_module.messagebox, "showwarning") as warning,
            mock.patch.object(app, "_open_update_url") as open_url,
        ):
            app.focus_force()
            app.update()
            app.focus_latex_editor()
            app.status_var.set("正在编辑公式")
            self.pump(app, 2500)
            fetch.assert_called_once_with(app_module.__version__)
            self.assertTrue(app.update_button.find_withtag("badge"))
            self.assertEqual(app.status_var.get(), "正在编辑公式")
            self.assertIs(app.focus_get(), app.output_text)
            prompt.assert_not_called()
            info.assert_not_called()
            warning.assert_not_called()
            open_url.assert_not_called()
            app.update_button.event_generate("<Button-1>")
            prompt.assert_called_once()
            self.assertIn(release.latest_version, prompt.call_args.args[1])
            self.assertIn(release.notes, prompt.call_args.args[1])
            self.assertTrue(app.update_button.find_withtag("badge"))
            open_url.assert_not_called()
            prompt.return_value = True
            app.update_button.event_generate("<Button-1>")
            open_url.assert_called_once_with(release.installer_url)
            fetch.assert_called_once()
            self.assertTrue(app.update_button.find_withtag("badge"))
            errors.assert_not_called()

    def test_update_badge_survives_offline_check_and_retry_can_clear_it(self):
        with (
            self.window() as (app, errors),
            mock.patch.object(app_module.messagebox, "showwarning") as warning,
            mock.patch.object(app_module.messagebox, "showinfo") as info,
        ):
            release = update_release()
            app._handle_update_ready(release, interactive=False)
            app._update_fetcher = mock.Mock(side_effect=app_module.UpdateCheckError("offline"))
            app.check_for_updates(interactive=False)
            app.update_check_thread.join(timeout=2)
            app._poll_worker_queue()
            self.assertIs(app.available_update, release)
            self.assertTrue(app.update_button.find_withtag("badge"))
            self.assertFalse(app.update_button.is_checking)
            self.assertFalse(app.update_button.is_disabled)
            warning.assert_not_called()
            app._handle_update_ready(update_release(app_module.__version__), interactive=False)
            self.assertIsNone(app.available_update)
            self.assertFalse(app.update_button.find_withtag("badge"))
            self.assertIn("已是最新版本", app.update_button.tooltip_text)
            app._handle_update_error("offline", interactive=False)
            self.assertIn("点击重试", app.update_button.tooltip_text)
            app._update_fetcher = mock.Mock(return_value=update_release(app_module.__version__))
            app.update_button.event_generate("<Button-1>")
            app.update_check_thread.join(timeout=2)
            app._poll_worker_queue()
            info.assert_called_once()
            self.assertIn("已是最新版本", app.update_button.tooltip_text)
            self.assertFalse(app.update_button.find_withtag("badge"))
            errors.assert_not_called()

    def test_update_icon_tooltip_preserves_focus_and_keyboard_click_works(self):
        for scale in (1.0, 2.0):
            with self.subTest(scale=scale), self.window(scale) as (app, _errors):
                button = app.update_button
                button.set_available_version("99.0.0")
                self.assertEqual(button.winfo_reqwidth(), button.winfo_reqheight())
                app.focus_force()
                app.update()
                app.focus_latex_editor()
                button.event_generate("<Enter>")
                self.pump(app, 550)
                self.assertIsNotNone(button.tooltip)
                self.assert_window_on_screen(button.tooltip)
                self.assertIs(app.focus_get(), app.output_text)
                self.assertIn("99.0.0", button.tooltip_text)
                button.event_generate("<Leave>")
                self.assertIsNone(button.tooltip)
                button.focus_set()
                self.pump(app, 550)
                self.assertIsNotNone(button.tooltip)
                self.assertIs(app.focus_get(), button)
                with mock.patch.object(button, "command") as activate:
                    button.event_generate("<Return>")
                    activate.assert_called_once()
                    self.assertIsNone(button.tooltip)

    def test_closing_during_update_cancels_animation_and_tooltip(self):
        with self.window() as (app, _errors):
            button = app.update_button
            button.event_generate("<Enter>")
            button.set_checking(True)
            self.pump(app, 100)
            spin_timer = button._spin_after_id
            tooltip_timer = button._tooltip_after_id
            self.assertIsNotNone(spin_timer)
            self.assertIsNotNone(tooltip_timer)
            app.destroy()
            remaining = app.tk.call("after", "info")
            self.assertNotIn(spin_timer, remaining)
            self.assertNotIn(tooltip_timer, remaining)

    def test_small_screen_keeps_controls_reachable_at_large_scale(self):
        for screen, scale in (((1024, 728), 2.0), ((1366, 728), 1.5)):
            with self.subTest(screen=screen, scale=scale), self.window(scale, screen) as (app, _errors):
                self.assert_window_on_screen(app)
                self.assert_buttons_fit(app)
                self.assertTrue(app.workspace.scrollbar.winfo_ismapped())
                app.workspace.viewport.yview_moveto(0)
                app.focus_force()
                app.update()
                app.latex_edit_button.focus_set()
                app.update()
                self.assertGreater(app.workspace.viewport.yview()[0], 0)
                self.assertGreaterEqual(app.latex_edit_button.winfo_rooty(), app.workspace.viewport.winfo_rooty())
                self.assertLessEqual(
                    app.latex_edit_button.winfo_rooty() + app.latex_edit_button.winfo_height(),
                    app.workspace.viewport.winfo_rooty() + app.workspace.viewport.winfo_height(),
                )
                app.workspace.viewport.yview_moveto(0)
                with mock.patch.object(app, "_queue_mathml_render"):
                    app.worker_queue.put(("success", {
                        "formula": r"[-45.67\%, 19.38\%]", "elapsed": 0.8,
                        "image_revision": app.image_revision,
                    }))
                    app._poll_worker_queue()
                app.update()
                self.assertGreater(app.workspace.viewport.yview()[0], 0)
                self.assertGreaterEqual(app.result_panel.winfo_rooty(), app.workspace.viewport.winfo_rooty())
                app.show_model_manager()
                app.update()
                self.assert_window_on_screen(app.model_manager_window)
                self.assert_buttons_fit(app.model_manager_window)

    def test_canvas_caption_uses_allocated_size_and_disabled_buttons_skip_focus(self):
        with self.window() as (app, _errors):
            frame = tk.Toplevel(app)
            frame.geometry("560x180")
            calls = []
            button = RoundedButton(frame, text="复制 MathML", command=lambda: calls.append(True))
            button.pack(fill=tk.BOTH, expand=True)
            app.update()
            x, y = button.coords(button.find_withtag("caption")[0])
            self.assertAlmostEqual(x, button.winfo_width() / 2, delta=1)
            self.assertAlmostEqual(y, button.winfo_height() / 2, delta=1)
            button.set_disabled(True)
            button.event_generate("<Button-1>")
            self.assertEqual(calls, [])
            self.assertEqual(str(button.cget("takefocus")), "0")
            button.set_disabled(False)
            button.focus_force()
            app.update()
            self.assertTrue(button.find_withtag("focus"))
            button.event_generate("<Return>")
            self.assertEqual(calls, [True])
            frame.destroy()

    def test_copy_confirms_real_clipboard_without_taking_editor_focus(self):
        with self.window() as (app, errors):
            app._replace_output_text(r"\frac{x}{y}")
            app.focus_force()
            app.update()
            app.focus_latex_editor()
            app.update()
            app.copy_latex_button.event_generate("<Button-1>")
            app.update()
            self.assertEqual(app.clipboard_get(), r"\frac{x}{y}")
            self.assertTrue(app.feedback_toast.winfo_ismapped())
            self.assertIn("LaTeX 已复制", app.feedback_toast.label.cget("text"))
            self.assertIs(app.focus_get(), app.output_text)
            errors.assert_not_called()

    def test_incomplete_interval_reports_review_status_and_next_success_clears_it(self):
        with self.window() as (app, errors), mock.patch.object(app, "_queue_mathml_render"):
            for formula, incomplete in (
                ("[left- 4 5.6 7 " + "\\", True),
                (r"[-45.67\%, 19.38\%]", False),
            ):
                app.worker_queue.put(("success", {
                    "formula": formula, "elapsed": 0.8,
                    "image_revision": app.image_revision,
                }))
                app.after_cancel(app.worker_poll_after_id)
                app.worker_poll_after_id = None
                app._poll_worker_queue()
                app.update()
                self.assertEqual(app.output_text.get("1.0", "end-1c"), formula)
                if incomplete:
                    self.assertIn("可能不完整", app.status_var.get())
                    self.assertTrue(app.feedback_toast.winfo_ismapped())
                else:
                    self.assertIn("识别完成", app.status_var.get())
                    self.assertFalse(app.feedback_toast.winfo_ismapped())
            errors.assert_not_called()

    def test_selection_copy_and_more_formats_report_the_actual_copy(self):
        with self.window() as (app, errors):
            app._replace_output_text("x + y")
            app.output_text.tag_add(tk.SEL, "1.0", "1.1")
            app.output_text.event_generate("<<Copy>>")
            app.update()
            self.assertEqual(app.clipboard_get(), "x")
            self.assertIn("选中文本 已复制", app.feedback_toast.label.cget("text"))
            app.copy_format("markdown_inline")
            self.assertEqual(app.clipboard_get(), "$x + y$")
            self.assertIn("Markdown 行内公式 已复制", app.feedback_toast.label.cget("text"))
            errors.assert_not_called()

    def test_repeated_feedback_replaces_timer_and_auto_hides(self):
        with self.window() as (app, _errors):
            toast = app.feedback_toast
            toast.show("第一条", anchor=app.copy_latex_button, duration=80)
            first_timer = toast._after_id
            toast.show("第二条", anchor=app.copy_mathml_button, duration=400)
            self.assertNotIn(first_timer, app.tk.call("after", "info"))
            self.pump(app, 120)
            self.assertTrue(toast.winfo_ismapped())
            self.assertEqual(toast.label.cget("text"), "第二条")
            self.pump(app, 320)
            self.assertFalse(toast.winfo_ismapped())
            self.assertIsNone(toast._after_id)

    def test_failed_copy_clears_previous_success(self):
        with self.window() as (app, errors):
            app._replace_output_text("x")
            app.copy_latex()
            with mock.patch.object(app, "_copy_text", side_effect=tk.TclError("clipboard busy")):
                app.copy_latex()
            self.assertEqual(app.status_var.get(), "复制失败")
            self.assertFalse(app.feedback_toast.winfo_ismapped())
            errors.assert_called_once()

    def test_word_plain_text_fallback_is_distinct_from_rich_copy_success(self):
        with self.window() as (app, errors):
            app._replace_output_text("x")
            for rich in (True, False):
                with mock.patch.object(app, "_copy_mathml_for_word", return_value=rich):
                    app.copy_mathml()
                self.assertEqual("已复制纯文本" in app.feedback_toast.label.cget("text"), not rich)
                self.assertEqual(app.feedback_toast.label.cget("bg"), "#e8f7ef" if rich else "#fff4df")
            errors.assert_not_called()

    def test_empty_copy_and_preview_give_feedback_without_starting_work(self):
        with self.window() as (app, errors):
            with mock.patch.object(app, "_copy_text") as copy, mock.patch.object(app, "_queue_mathml_render") as render:
                app.copy_latex()
                self.assertIn("没有可复制", app.feedback_toast.label.cget("text"))
                app.refresh_mathml_preview()
                self.assertIn("暂无可预览", app.feedback_toast.label.cget("text"))
                copy.assert_not_called()
                render.assert_not_called()
            errors.assert_not_called()

    def test_manager_remains_usable_with_long_details_at_large_scale(self):
        with self.window(2.0) as (app, _errors):
            app.show_model_manager()
            manager = app.model_manager_window
            width, height = manager.minsize()
            manager.geometry(f"{width}x{height}")
            tree = next(item for item in descendants(manager) if isinstance(item, ttk.Treeview))
            tree.selection_set("MixTexZhEn")
            app.update()
            self.assert_buttons_fit(manager)
            self.assertGreater(tree.winfo_height(), ui_pixels(app, 100))
            details = next(item for item in descendants(manager) if isinstance(item, tk.Text))
            self.assertIn("MixTexZhEn", details.get("1.0", tk.END))
            details.yview_moveto(1.0)
            self.assertAlmostEqual(details.yview()[1], 1.0, delta=0.01)
            horizontal = next(item for item in descendants(manager) if isinstance(item, ttk.Scrollbar))
            self.assertTrue(horizontal.winfo_ismapped())
            provider = next(item for item in descendants(manager) if isinstance(item, RoundedChoice))
            provider.set("OpenDataLab / Cooper114")
            self.assert_buttons_fit(manager)

    def test_manual_preview_finishes_and_does_not_overwrite_new_copy_feedback(self):
        with self.window() as (app, errors):
            app._replace_output_text("x")
            app.after_cancel(app.mathml_preview_poll_after_id)
            app.mathml_preview_poll_after_id = None
            with mock.patch.object(app, "_queue_mathml_render"):
                app.refresh_mathml_preview()
                old_token = app.mathml_render_token
                app._replace_output_text("x + 1")
                app._update_mathml_preview()
            preview = app.session_dir / "preview-test.png"
            Image.new("RGB", (60, 30), "white").save(preview)
            app.mathml_preview_queue.put((old_token, "error", "obsolete render"))
            app.mathml_preview_queue.put((app.mathml_render_token, "image", str(preview)))
            app._poll_mathml_preview_queue()
            self.assertEqual(app.status_var.get(), "预览已刷新")
            self.assertIsNone(app.manual_preview_token)
            with mock.patch.object(app, "_queue_mathml_render"):
                app.refresh_mathml_preview()
            app.copy_latex()
            app.after_cancel(app.mathml_preview_poll_after_id)
            app.mathml_preview_queue.put((app.mathml_render_token, "error", "browser failed"))
            app._poll_mathml_preview_queue()
            self.assertEqual(app.status_var.get(), "LaTeX 已复制到剪贴板")
            self.assertIn("LaTeX 已复制", app.feedback_toast.label.cget("text"))
            errors.assert_not_called()

    def test_model_picker_keyboard_keeps_last_row_visible(self):
        with self.window(2.0) as (app, _errors):
            app.model_picker.status_provider = lambda _model: "已下载"
            app.focus_force()
            app.model_picker._toggle_popup()
            app.update()
            popup = app.model_picker.popup
            row = app.focus_get()
            self.assertIsNotNone(row)
            row.event_generate("<End>")
            app.update()
            last = app.focus_get()
            canvas = last.master.master
            self.assertIsInstance(canvas, tk.Canvas)
            self.assertGreater(canvas.yview()[0], 0)
            self.assertGreaterEqual(last.winfo_rooty(), canvas.winfo_rooty())
            self.assertLessEqual(last.winfo_rooty() + last.winfo_height(), canvas.winfo_rooty() + canvas.winfo_height() + 1)
            for label in descendants(last):
                if isinstance(label, tk.Label):
                    self.assertLessEqual(label.winfo_reqheight(), label.winfo_height())
            self.assertLessEqual(popup.winfo_width(), app.winfo_screenwidth())
            app.model_picker._close_popup()


if __name__ == "__main__":
    unittest.main()
