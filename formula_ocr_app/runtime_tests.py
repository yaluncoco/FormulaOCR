"""Regression checks for shared runtime state and inference hot paths."""

from __future__ import annotations

import hashlib
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from formula_ocr_app import download_utils, model_runtime
from formula_ocr_app.image_utils import foreground_bbox
from formula_ocr_app.mathcraft_recognizer import _generate_formula_tokens
from formula_ocr_app.model_api import ModelDownloadCancelled
from formula_ocr_app.rapid_recognizer import _decode_tokens
from formula_ocr_app.recognition_pipeline import FormulaRecognizer


class RuntimeStateTests(unittest.TestCase):
    def test_failed_model_switch_does_not_reuse_closed_backend(self) -> None:
        original = mock.Mock()
        original.predict.return_value = "x"
        replacement = mock.Mock()
        replacement.predict.return_value = "y"
        factory = mock.Mock(
            side_effect=[original, RuntimeError("load failed"), replacement]
        )
        recognizer = FormulaRecognizer(model_name="first", recognizer_factory=factory)
        self.addCleanup(recognizer.close)

        self.assertEqual(recognizer.predict("unused.png"), "x")
        recognizer.model_name = "second"
        with self.assertRaisesRegex(RuntimeError, "load failed"):
            recognizer.predict("unused.png")
        recognizer.model_name = "first"

        self.assertEqual(recognizer.predict("unused.png"), "y")
        original.close.assert_called_once_with()
        self.assertEqual(factory.call_count, 3)

    def test_result_records_model_that_started_recognition(self) -> None:
        backend = mock.Mock()
        recognizer = FormulaRecognizer(
            model_name="first", recognizer_factory=lambda _name: backend
        )
        self.addCleanup(recognizer.close)

        def predict(_path):
            recognizer.model_name = "second"
            return "x"

        backend.predict.side_effect = predict
        result = recognizer.recognize("unused.png")
        self.assertEqual(result.selected_model_name, "first")
        self.assertEqual(result.model_name, "first")

    def test_bundled_status_does_not_hash_models_on_ui_thread(self) -> None:
        with (
            mock.patch.object(
                model_runtime, "is_paddle_model_bundled", return_value=True
            ),
            mock.patch.object(
                model_runtime, "paddle_model_has_data", return_value=False
            ),
            mock.patch.object(
                model_runtime, "is_model_cached", return_value=True
            ) as checker,
        ):
            self.assertEqual(
                model_runtime.model_status_label("LaTeX_OCR_rec", cached=True),
                "随包内置",
            )
        self.assertTrue(
            all(
                call.kwargs.get("verify_hash") is False
                for call in checker.call_args_list
            )
        )

    def test_bundled_validity_check_still_verifies_hash_by_default(self) -> None:
        with (
            mock.patch.object(
                model_runtime, "is_paddle_model_bundled", return_value=True
            ),
            mock.patch.object(
                model_runtime, "paddle_model_has_data", return_value=False
            ),
            mock.patch.object(
                model_runtime, "is_model_cached", return_value=False
            ) as checker,
        ):
            self.assertFalse(model_runtime.is_model_bundled_only("LaTeX_OCR_rec"))
        checker.assert_called_once_with("LaTeX_OCR_rec", verify_hash=True)

    def test_download_request_defers_validation_to_worker(self) -> None:
        from formula_ocr_app import app

        window = mock.Mock(is_busy=False, is_destroying=False)
        window._ensure_model_terms_accepted.return_value = True
        with (
            mock.patch.object(app, "is_model_cached") as checker,
            mock.patch.object(app.threading, "Thread") as thread,
        ):
            self.assertTrue(
                app.FormulaOCRApp._request_model_download(window, "RapidLaTeXOCR")
            )
        checker.assert_not_called()
        thread.return_value.start.assert_called_once_with()


class DownloadRuntimeTests(unittest.TestCase):
    def test_hash_cache_eviction_during_validation_is_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.bin"
            payload = b"verified model"
            path.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            with download_utils._HASH_CACHE_LOCK:
                download_utils._HASH_CACHE.clear()
            self.addCleanup(download_utils._HASH_CACHE.clear)
            self.assertTrue(
                download_utils.file_is_valid(
                    path, len(payload), digest, verify_hash=True
                )
            )
            cache_read = threading.Event()
            resume = threading.Event()
            results: queue.Queue = queue.Queue()
            original_signature = download_utils._file_signature
            calls = 0

            def signature(stat):
                nonlocal calls
                calls += 1
                if calls == 2:
                    cache_read.set()
                    if not resume.wait(5):
                        raise TimeoutError("cache eviction test did not resume")
                return original_signature(stat)

            def validate():
                try:
                    results.put(
                        download_utils.file_is_valid(
                            path, len(payload), digest, verify_hash=True
                        )
                    )
                except BaseException as exc:
                    results.put(exc)

            with mock.patch.object(
                download_utils, "_file_signature", side_effect=signature
            ):
                thread = threading.Thread(target=validate, daemon=True)
                thread.start()
                try:
                    self.assertTrue(
                        cache_read.wait(5), "validation did not read the cached digest"
                    )
                    with download_utils._HASH_CACHE_LOCK:
                        download_utils._HASH_CACHE.clear()
                finally:
                    resume.set()
                    thread.join(5)
                self.assertFalse(thread.is_alive())
            self.assertIs(results.get_nowait(), True)

    def test_tuple_of_transfer_errors_preserves_resume_and_cause(self) -> None:
        class ConnectionLost(Exception):
            pass

        failure = ConnectionLost("connection lost")

        def chunks(**_kwargs):
            yield b"abc"
            raise failure

        response = mock.Mock(status_code=200, headers={})
        response.iter_content.side_effect = chunks
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "model.bin"
            partial = destination.with_suffix(".part")
            with self.assertRaises(download_utils.VerifiedDownloadFailure) as caught:
                download_utils.download_verified_file(
                    download_utils.RemoteFileSpec(
                        "model.bin",
                        6,
                        hashlib.sha256(b"abcdef").hexdigest(),
                        "https://example.invalid/model",
                    ),
                    destination,
                    partial=partial,
                    completed=0,
                    total=6,
                    notify=lambda *_args: None,
                    request_get=lambda *_args, **_kwargs: response,
                    request_exception=(ConnectionLost, TimeoutError),
                )
            self.assertEqual(caught.exception.phase, "transfer")
            self.assertIs(caught.exception.cause, failure)
            self.assertEqual(partial.read_bytes(), b"abc")
            self.assertFalse(destination.exists())
        response.close.assert_called_once_with()

    def test_tuple_of_transfer_errors_does_not_mask_cancellation(self) -> None:
        response = mock.Mock(status_code=200, headers={})
        response.iter_content.return_value = [b"abc"]
        notifications = 0

        def notify(*_args):
            nonlocal notifications
            notifications += 1
            if notifications == 3:
                raise ModelDownloadCancelled()

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "model.bin"
            partial = destination.with_suffix(".part")
            with self.assertRaises(ModelDownloadCancelled):
                download_utils.download_verified_file(
                    download_utils.RemoteFileSpec(
                        "model.bin", 6, "unused", "https://example.invalid/model"
                    ),
                    destination,
                    partial=partial,
                    completed=0,
                    total=6,
                    notify=notify,
                    request_get=lambda *_args, **_kwargs: response,
                    request_exception=(ConnectionError, TimeoutError),
                )
            self.assertEqual(partial.read_bytes(), b"abc")
        response.close.assert_called_once_with()


class DecoderRuntimeTests(unittest.TestCase):
    def test_foreground_bounds_match_coordinates_for_empty_and_sparse_masks(
        self,
    ) -> None:
        rng = np.random.default_rng(42)
        for shape in ((0, 0), (1, 1), (1, 37), (29, 1), (53, 71)):
            for density in (0.0, 0.05, 0.5, 1.0):
                with self.subTest(shape=shape, density=density):
                    mask = rng.random(shape) < density
                    coordinates = np.argwhere(mask)
                    expected = None
                    if coordinates.size:
                        top, left = coordinates.min(axis=0)
                        bottom, right = coordinates.max(axis=0)
                        expected = (left, top, right + 1, bottom + 1)
                    self.assertEqual(foreground_bbox(mask), expected)

    def test_foreground_bounds_use_exclusive_right_and_bottom_edges(self) -> None:
        mask = np.zeros((11, 19), dtype=bool)
        mask[3, 5] = True
        self.assertEqual(foreground_bbox(mask), (5, 3, 6, 4))
        mask[-1, -1] = True
        self.assertEqual(foreground_bbox(mask), (5, 3, 19, 11))

    def test_mathcraft_batch_rows_finish_independently(self) -> None:
        expected = {0: [], 1: [3, 4], 2: [5, 6, 7, 8]}
        prefixes = []

        class Decoder:
            def get_inputs(self):
                return [
                    SimpleNamespace(name="input_ids"),
                    SimpleNamespace(name="encoder_hidden_states"),
                ]

            def run(self, _outputs, inputs):
                ids = inputs["input_ids"]
                hidden = inputs["encoder_hidden_states"]
                rows = hidden[:, 0, 0].astype(int).tolist()
                prefixes.append((rows, ids.copy()))
                logits = np.full((len(rows), 1, 12), -10.0, dtype=np.float32)
                for index, row in enumerate(rows):
                    next_index = ids.shape[1] - 1
                    values = expected[row]
                    next_id = values[next_index] if next_index < len(values) else 2
                    logits[index, 0, next_id] = 10
                return [logits]

        hidden = np.arange(3, dtype=np.float32).reshape(3, 1, 1)
        tokens, scores, repeated = _generate_formula_tokens(
            Decoder(), hidden, decoder_start_id=2, eos_id=2, max_new_tokens=10
        )
        self.assertEqual(tokens, list(expected.values()))
        self.assertEqual([len(row) for row in scores], [0, 2, 4])
        self.assertFalse(repeated.any())
        for rows, ids in prefixes:
            for index, row in enumerate(rows):
                self.assertEqual(
                    ids[index].tolist(), [2, *expected[row][: ids.shape[1] - 1]]
                )

    def test_decoders_respect_short_token_limits_without_eos(self) -> None:
        class Decoder:
            def __init__(self, names):
                self.names = names
                self.calls = 0

            def get_inputs(self):
                return [SimpleNamespace(name=name) for name in self.names]

            def run(self, _outputs, inputs):
                self.calls += 1
                logits = np.full((1, 1, 8), -10, dtype=np.float32)
                logits[0, 0, 3] = 10
                return [logits]

        for limit in (0, 1, 5):
            with self.subTest(limit=limit):
                expected = [3] * max(1, limit)
                mathcraft = Decoder(("input_ids", "encoder_hidden_states"))
                tokens, scores, repeated = _generate_formula_tokens(
                    mathcraft,
                    np.zeros((1, 1, 1)),
                    decoder_start_id=2,
                    eos_id=None,
                    max_new_tokens=limit,
                )
                self.assertEqual(tokens, [expected])
                self.assertEqual(len(scores[0]), len(expected))
                self.assertFalse(repeated.any())
                rapid = Decoder(("tokens", "mask", "context"))
                self.assertEqual(
                    _decode_tokens(
                        rapid,
                        np.zeros((1, 1, 1)),
                        bos_token=1,
                        eos_token=2,
                        max_new_tokens=limit,
                    ),
                    expected,
                )

    def test_rapid_long_formula_keeps_sliding_context_and_eos(self) -> None:
        generated = [3 + index % 17 for index in range(540)]
        calls = []

        class Decoder:
            def get_inputs(self):
                return [
                    SimpleNamespace(name=name) for name in ("tokens", "mask", "context")
                ]

            def run(self, _outputs, inputs):
                step = len(calls)
                calls.append(inputs["tokens"].copy())
                np.testing.assert_array_equal(
                    inputs["tokens"], [[1, *generated[:step]][-512:]]
                )
                self_test.assertEqual(inputs["mask"].shape, inputs["tokens"].shape)
                self_test.assertTrue(inputs["mask"].all())
                logits = np.full((1, 1, 20), -10, dtype=np.float32)
                logits[0, 0, generated[step] if step < len(generated) else 2] = 10
                return [logits]

        self_test = self
        result = _decode_tokens(
            Decoder(), np.zeros((1, 1, 1)), bos_token=1, eos_token=2, max_new_tokens=600
        )
        self.assertEqual(result, [*generated, 2])
        self.assertEqual(len(calls), 541)


if __name__ == "__main__":
    unittest.main()
