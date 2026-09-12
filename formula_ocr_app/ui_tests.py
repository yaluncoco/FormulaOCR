"""Desktop regressions for scaled layouts and action feedback (no OCR models)."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sys
import tempfile
import time
import tkinter as tk
from tkinter import ttk
import unittest
from unittest import mock

from PIL import Image

from formula_ocr_app import app as app_module
from formula_ocr_app.app_settings import AppSettings
from formula_ocr_app.ui_widgets import RoundedButton, RoundedChoice, ui_pixels


def descendants(widget: tk.Misc):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


@unittest.skipUnless(sys.platform == "win32" or os.environ.get("DISPLAY"), "requires a desktop Tk display")
class DesktopInteractionTests(unittest.TestCase):
    @contextmanager
    def window(self, scale: float = 1.5):
        app_module.enable_windows_dpi_awareness()

        class ScaledApp(app_module.FormulaOCRApp):
            def _configure_styles(self):
                self.tk.call("tk", "scaling", scale * 96 / 72)
                super()._configure_styles()

        with (
            tempfile.TemporaryDirectory(prefix="formulaocr-ui-test-") as scratch,
            mock.patch.object(app_module, "CACHE_DIR", Path(scratch)),
            mock.patch.object(app_module, "load_settings", return_value=AppSettings()),
            mock.patch.object(app_module, "is_model_cached", return_value=False),
            mock.patch.object(app_module, "model_status_label", return_value="待下载"),
            mock.patch.object(app_module, "write_log"),
            mock.patch.object(app_module.messagebox, "showerror") as errors,
        ):
            app = ScaledApp(auto_check_updates=False)
            callbacks = []
            app.report_callback_exception = lambda _type, value, _tb: callbacks.append(str(value))
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
                box = button.bbox("caption")
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
                large = f"{app.winfo_width()}x{app.winfo_height()}"
                self.assert_buttons_fit(app)
                width, height = app.minsize()
                app.geometry(f"{width}x{height}")
                self.assert_buttons_fit(app)
                self.assertNotEqual(app.copy_latex_button.winfo_y(), app.clear_button.winfo_y())
                self.assertGreater(app.output_text.winfo_height(), ui_pixels(app, 45))
                app.geometry(large)
                self.assert_buttons_fit(app)

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
            tree = next(item for item in manager.winfo_children() if isinstance(item, ttk.Treeview))
            tree.selection_set("MixTexZhEn")
            app.update()
            self.assert_buttons_fit(manager)
            self.assertGreater(tree.winfo_height(), ui_pixels(app, 100))
            details = next(item for item in descendants(manager) if isinstance(item, tk.Text))
            self.assertIn("MixTexZhEn", details.get("1.0", tk.END))
            details.yview_moveto(1.0)
            self.assertAlmostEqual(details.yview()[1], 1.0, delta=0.01)
            horizontal = next(item for item in manager.winfo_children() if isinstance(item, ttk.Scrollbar))
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
