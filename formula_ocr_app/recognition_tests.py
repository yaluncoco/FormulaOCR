from __future__ import annotations

import hashlib
import os
import queue
import shutil
import tarfile
import threading
import time
import unittest
import zipfile
import zlib
from pathlib import Path
from unittest import mock

import formula_ocr_app.mixtex_model_downloader as mixtex_downloader
import formula_ocr_app.model_runtime as model_runtime
import formula_ocr_app.paddle_hf_model_downloader as paddle_hf_downloader
import formula_ocr_app.pix2text_model_downloader as pix2text_downloader
import formula_ocr_app.unimernet_onnx_model_downloader as unimernet_onnx_downloader
from formula_ocr_app.app_settings import AppSettings, load_settings, save_settings
from formula_ocr_app.download_utils import (
    RemoteFileSpec,
    VerifiedDownloadFailure,
    download_verified_file,
    model_files_are_valid,
    replace_model_directory,
)
from formula_ocr_app.formula_formats import (
    clean_recognized_latex,
    export_formula_docx,
    latex_to_equation_environment,
    latex_to_html,
    latex_to_markdown_block,
    latex_to_markdown_inline,
)
from formula_ocr_app.image_utils import image_to_rgb
from formula_ocr_app.interprocess_lock import InterProcessFileLock
from formula_ocr_app.mathcraft_model_downloader import (
    MATHCRAFT_ARCHIVE_SHA256,
    MATHCRAFT_MODEL_FILES,
    MathCraftModelDownloadError,
    _file_is_valid,
    _install_archive,
    _validate_zip_members,
)
from formula_ocr_app.mathcraft_recognizer import MathCraftFormulaRecognizer
from formula_ocr_app.mathml_preview import (
    MathMLPreviewCancelled,
    wait_for_rendered_png,
)
from formula_ocr_app.mixtex_model_downloader import (
    MIXTEX_ARCHIVE_SHA256,
    MIXTEX_ARCHIVE_SIZE,
    MIXTEX_MODEL_FILES,
    MixTexModelDownloadError,
)
from formula_ocr_app.mixtex_model_downloader import (
    _install_archive as _install_mixtex_archive,
)
from formula_ocr_app.mixtex_model_downloader import (
    _validate_zip_members as _validate_mixtex_zip_members,
)
from formula_ocr_app.mixtex_recognizer import (
    _generate_tokens as _generate_mixtex_tokens,
)
from formula_ocr_app.model_api import ModelDownloadCancelled
from formula_ocr_app.model_catalog import (
    MODEL_BY_ID,
    MODEL_QUICK_FILTERS,
    MODEL_SPECS,
    model_matches_query,
    model_matches_quick_filter,
)
from formula_ocr_app.model_downloader import (
    ModelDownloadError,
    _validate_tar_members,
)
from formula_ocr_app.model_downloader import (
    _download_archive as _download_official_archive,
)
from formula_ocr_app.model_downloader import (
    _install_archive as _install_official_archive,
)
from formula_ocr_app.model_runtime import create_recognizer_backend, remove_model
from formula_ocr_app.onnx_runtime import (
    execution_providers,
    repeated_token_suffix_start,
)
from formula_ocr_app.paddle_formula_recognizer import (
    PaddleFormulaRecognizer,
    _load_model_configuration,
    _normalize_latex_ocr_text,
    _normalize_unimernet_text,
    _opencv_gray_from_rgb,
    _preprocess_image,
    _PreprocessSpec,
    _UniMERNetDecoder,
)
from formula_ocr_app.paddle_hf_model_downloader import (
    PADDLE_HF_MODEL_FILES,
    PaddleHFModelFile,
)
from formula_ocr_app.paddle_hf_model_downloader import (
    _download_file as _download_paddle_hf_file,
)
from formula_ocr_app.pix2text_model_downloader import (
    PIX2TEXT_MODEL_FILES,
    Pix2TextModelFile,
    _download_file,
)
from formula_ocr_app.pix2text_recognizer import Pix2TextFormulaRecognizer
from formula_ocr_app.rapid_model_downloader import RAPID_MODEL_FILES
from formula_ocr_app.rapid_recognizer import (
    RapidLatexRecognizer,
    RapidLatexRuntimeError,
    _decode_tokens as _decode_rapid_tokens,
)
from formula_ocr_app.rapid_recognizer import (
    _post_process as _post_process_rapid,
)
from formula_ocr_app.recognition_pipeline import FormulaRecognizer
from formula_ocr_app.runtime_paths import (
    _paddle_model_files_exist,
    bundled_external_model_dir,
    external_model_cache_size,
    external_model_dir,
    external_model_has_data,
    is_external_model_bundled,
    is_paddle_model_cached,
    paddle_model_cache_size,
    paddle_model_dir,
    paddle_model_has_data,
    remove_external_model,
    remove_paddle_model,
    runtime_cache_dir,
    runtime_log_dir,
)
from formula_ocr_app.unimernet_onnx_model_downloader import (
    UNIMERNET_ONNX_MODEL_FILES,
    UNIMERNET_ONNX_TOTAL_SIZE,
    UniMERNetONNXModelFile,
)
from formula_ocr_app.unimernet_onnx_model_downloader import (
    _download_file as _download_unimernet_onnx_file,
)
from formula_ocr_app.unimernet_onnx_model_downloader import (
    _model_files_are_valid as _unimernet_model_files_are_valid,
)
from formula_ocr_app.unimernet_onnx_recognizer import (
    UniMERNetSmallFormulaRecognizer,
)
from formula_ocr_app.unimernet_onnx_recognizer import (
    _generate_tokens as _generate_unimernet_tokens,
)
from formula_ocr_app.unimernet_onnx_recognizer import (
    _resize_to_fit as _resize_unimernet_to_fit,
)

M_ARGMAX_OUTPUT = r"k^{*}=\arg\operatorname*{m a x}_{k}J(k)"
EXPECTED_ARGMAX = r"k^{*}=\arg\max\limits_{k}J(k)"


class _FakeRecognizer:
    def __init__(
        self,
        model_name: str,
        outputs: dict[str, str | Exception],
        calls: list[str],
    ) -> None:
        self.model_name = model_name
        self.outputs = outputs
        self.calls = calls

    def predict(self, _image_path: str | Path) -> str:
        self.calls.append(self.model_name)
        output = self.outputs[self.model_name]
        if isinstance(output, Exception):
            raise output
        return output

    def close(self) -> None:
        return None


class RecognitionPostprocessTests(unittest.TestCase):
    def test_tensor_separator_quote_is_repaired_without_corrupting_primes(self) -> None:
        self.assertEqual(
            clean_recognized_latex(
                r"F_{ctx}=Concat(F_{local}'F_{global})"
            ),
            r"F_{ctx}=Concat(F_{local},F_{global})",
        )
        for formula in (r"f'(x)", r"{f}'x", r"x_{i}'y_{j}", r"F_{i}'F_{j}"):
            with self.subTest(formula=formula):
                self.assertEqual(clean_recognized_latex(formula), formula)

    def test_spaced_named_operator_is_canonicalized(self) -> None:
        self.assertEqual(clean_recognized_latex(M_ARGMAX_OUTPUT), EXPECTED_ARGMAX)

    def test_argmax_subscript_is_made_an_explicit_display_limit(self) -> None:
        self.assertEqual(
            clean_recognized_latex(r"k^{*}=\arg\max_{k}J(k)"),
            EXPECTED_ARGMAX,
        )

    def test_existing_limits_are_not_duplicated(self) -> None:
        self.assertEqual(clean_recognized_latex(EXPECTED_ARGMAX), EXPECTED_ARGMAX)

    def test_text_command_keeps_meaningful_spaces(self) -> None:
        self.assertEqual(
            clean_recognized_latex(r"x=\text{if x is valid}"),
            r"x=\text{if x is valid}",
        )

    def test_paddle_decoders_keep_spaces_inside_text_like_commands(self) -> None:
        samples = (
            r"x + \text {if x \alpha y} + z",
            r"A = \mathrm {speed of \sin x} + B",
        )
        for formula in samples:
            with self.subTest(formula=formula, decoder="latex_ocr"):
                self.assertEqual(_normalize_latex_ocr_text(formula), formula)
            with self.subTest(formula=formula, decoder="unimernet"):
                self.assertEqual(_normalize_unimernet_text(formula), formula)
            with self.subTest(formula=formula, decoder="rapid"):
                self.assertEqual(_post_process_rapid(formula), formula)

    def test_spaced_roman_ocr_words_are_joined_without_touching_prose(self) -> None:
        self.assertEqual(
            clean_recognized_latex(
                r"\mathrm{i f~}x,\quad\mathrm{o t h e r w i\;s e}"
            ),
            r"\mathrm{if~}x,\quad\mathrm{otherwise}",
        )
        self.assertEqual(
            clean_recognized_latex(r"\mathrm{speed of light}"),
            r"\mathrm{speed of light}",
        )
        self.assertEqual(
            clean_recognized_latex(r"\mathrm{otherwi \; se}"),
            r"\mathrm{otherwise}",
        )
        self.assertEqual(
            clean_recognized_latex(r"\mathrm{speed \; of light}"),
            r"\mathrm{speed \; of light}",
        )

    def test_spurious_control_space_before_math_command_is_repaired(self) -> None:
        self.assertEqual(
            clean_recognized_latex(r"{\ mathcal{M}}"),
            r"{\mathcal{M}}",
        )

    def test_decimal_digits_are_kept_as_one_number(self) -> None:
        self.assertEqual(
            clean_recognized_latex(r"M(i,j) > 0 . 5"),
            r"M(i,j) > 0.5",
        )

    def test_cases_does_not_keep_a_second_ocr_added_left_brace(self) -> None:
        self.assertEqual(
            clean_recognized_latex(
                r"x=\Big \{ \begin{cases}0&a\\1&b\end{cases}"
            ),
            r"x=\begin{cases}0&a\\1&b\end{cases}",
        )
        self.assertEqual(
            clean_recognized_latex(
                r"x=\left\{\begin{cases}0&a\\1&b\end{cases}\right."
            ),
            r"x=\begin{cases}0&a\\1&b\end{cases}",
        )
        aligned = r"x=\left\{\begin{aligned}0&a\\1&b\end{aligned}\right."
        self.assertEqual(clean_recognized_latex(aligned), aligned)


class ImageInputTests(unittest.TestCase):
    def test_transparent_formula_background_becomes_white(self) -> None:
        from PIL import Image, ImageDraw

        source = Image.new("RGBA", (20, 12), (0, 0, 0, 0))
        ImageDraw.Draw(source).rectangle((6, 3, 13, 8), fill=(0, 0, 0, 255))

        converted = image_to_rgb(source)

        self.assertEqual(converted.mode, "RGB")
        self.assertEqual(converted.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(converted.getpixel((8, 5)), (0, 0, 0))


class DirectPaddleRecognizerTests(unittest.TestCase):
    def test_pp_formulanet_configuration_builds_direct_backend(self) -> None:
        config_path = Path(
            "formula_ocr_app/.cache/runtime/paddlex/official_models/"
            "PP-FormulaNet_plus-S/inference.yml"
        )
        if not config_path.is_file():
            self.skipTest("local PP-FormulaNet+ S configuration is not present")

        preprocess, decoder = _load_model_configuration(config_path)

        self.assertEqual(preprocess.family, "unimernet")
        self.assertEqual(preprocess.input_size, (384, 384))
        self.assertEqual(preprocess.output_divisor, 16)
        self.assertIsInstance(decoder, _UniMERNetDecoder)

    def test_unimernet_preprocess_crops_and_formats_without_opencv(self) -> None:
        import numpy as np
        from PIL import Image, ImageDraw

        root = Path.cwd() / "_direct_paddle_preprocess"
        self.addCleanup(shutil.rmtree, root, True)
        root.mkdir()
        image_path = root / "formula.png"
        image = Image.new("RGB", (160, 80), "white")
        ImageDraw.Draw(image).rectangle((45, 25, 115, 55), outline="black", width=5)
        image.save(image_path)

        tensor = _preprocess_image(
            image_path,
            _PreprocessSpec(
                family="unimernet",
                input_size=(384, 384),
                output_divisor=16,
            ),
        )

        self.assertEqual(tensor.shape, (1, 1, 384, 384))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(np.isfinite(tensor).all())
        self.assertLess(float(tensor.min()), float(tensor.max()))

    def test_rgb_grayscale_weights_match_opencv_rgb_conversion(self) -> None:
        import numpy as np

        pixels = np.asarray([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8)
        gray = _opencv_gray_from_rgb(pixels)

        self.assertEqual(int(gray[0, 0]), 76)
        self.assertEqual(int(gray[0, 1]), 29)

    def test_direct_predictor_receives_tensor_and_decodes_one_output(self) -> None:
        import numpy as np

        class _InputHandle:
            def reshape(self, shape):
                self.shape = tuple(shape)

            def copy_from_cpu(self, tensor):
                self.tensor = tensor

        class _OutputHandle:
            def copy_to_cpu(self):
                return np.array([[0, 4, 2]], dtype=np.int64)

        class _Predictor:
            def __init__(self):
                self.input = _InputHandle()
                self.runs = 0

            def get_input_names(self):
                return ["x"]

            def get_input_handle(self, _name):
                return self.input

            def run(self):
                self.runs += 1

            def get_output_names(self):
                return ["fetch_name_0"]

            def get_output_handle(self, _name):
                return _OutputHandle()

        class _Decoder:
            def decode(self, tokens):
                self.tokens = tokens.copy()
                return r"x^2"

        recognizer = PaddleFormulaRecognizer(model_name="PP-FormulaNet_plus-S")
        predictor = _Predictor()
        decoder = _Decoder()
        recognizer._predictor = predictor
        recognizer._preprocess_spec = _PreprocessSpec(
            family="unimernet",
            input_size=(32, 32),
        )
        recognizer._decoder = decoder
        with mock.patch(
            "formula_ocr_app.paddle_formula_recognizer._preprocess_image",
            return_value=np.zeros((1, 1, 32, 32), dtype=np.float32),
        ):
            result = recognizer.predict("unused.png")

        self.assertEqual(result, r"x^2")
        self.assertEqual(predictor.input.shape, (1, 1, 32, 32))
        self.assertEqual(predictor.runs, 1)
        np.testing.assert_array_equal(decoder.tokens, np.array([[0, 4, 2]]))


class SelectedModelRecognizerTests(unittest.TestCase):
    def _pipeline(
        self,
        outputs: dict[str, str | Exception],
        calls: list[str],
        *,
        model_name: str = "PP-FormulaNet_plus-S",
    ) -> FormulaRecognizer:
        return FormulaRecognizer(
            model_name=model_name,
            recognizer_factory=lambda name: _FakeRecognizer(name, outputs, calls),
        )

    def test_only_the_explicitly_selected_model_is_called(self) -> None:
        calls: list[str] = []
        pipeline = self._pipeline(
            {
                "PP-FormulaNet_plus-S": r"k^{*}=\arg\max_{k}J(k)",
                "PP-FormulaNet_plus-M": M_ARGMAX_OUTPUT,
                "PP-FormulaNet_plus-L": EXPECTED_ARGMAX,
            },
            calls,
        )

        result = pipeline.recognize("unused.png")

        self.assertEqual(result.latex, EXPECTED_ARGMAX)
        self.assertEqual(result.model_name, "PP-FormulaNet_plus-S")
        self.assertEqual(calls, ["PP-FormulaNet_plus-S"])

    def test_model_load_callback_reports_uncached_model(self) -> None:
        calls: list[str] = []
        events: list[tuple[str, bool]] = []
        data_dir = str(Path.cwd() / "_nonexistent_formula_ocr_test_data")
        with mock.patch.dict(
            os.environ,
            {"FORMULA_OCR_DATA_DIR": data_dir},
        ):
            pipeline = FormulaRecognizer(
                model_name="PP-FormulaNet_plus-S",
                recognizer_factory=lambda name: _FakeRecognizer(
                    name,
                    {"PP-FormulaNet_plus-S": EXPECTED_ARGMAX},
                    calls,
                ),
                model_load_callback=lambda name, cached: events.append(
                    (name, cached)
                ),
            )
            pipeline.recognize("unused.png")

        self.assertEqual(events, [("PP-FormulaNet_plus-S", False)])

    def test_runtime_registry_builds_only_the_selected_external_backend(self) -> None:
        fake_backend = object()

        class _BackendClass:
            def __new__(cls, **kwargs):
                self.assertEqual(kwargs["device"], "cpu")
                self.assertIsNone(kwargs["download_progress_callback"])
                return fake_backend

        with mock.patch(
            "formula_ocr_app.model_runtime._load_module"
        ) as load_module:
            load_module.return_value = mock.Mock(
                MathCraftFormulaRecognizer=_BackendClass
            )
            created = create_recognizer_backend(
                "MathCraftFormula",
                model_dir=None,
                device="cpu",
                download_progress_callback=None,
            )

        self.assertIs(created, fake_backend)
        load_module.assert_called_once_with("mathcraft_recognizer")

    def test_close_during_native_predict_is_deferred_until_worker_finishes(self) -> None:
        started = threading.Event()
        release = threading.Event()
        closed = threading.Event()

        class _BlockingRecognizer:
            def predict(self, _image_path: str | Path) -> str:
                started.set()
                if not release.wait(timeout=5):
                    raise RuntimeError("test prediction did not resume")
                return "x"

            def close(self) -> None:
                closed.set()

        pipeline = FormulaRecognizer(
            model_name="PP-FormulaNet_plus-S",
            recognizer_factory=lambda _name: _BlockingRecognizer(),
        )
        result: list[str] = []
        worker = threading.Thread(
            target=lambda: result.append(pipeline.predict("unused.png"))
        )
        worker.start()
        self.assertTrue(started.wait(timeout=2))

        pipeline.close()

        self.assertFalse(closed.is_set())
        release.set()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(result, ["x"])
        self.assertTrue(closed.is_set())

    def test_predict_after_close_does_not_create_a_backend(self) -> None:
        created: list[str] = []
        pipeline = FormulaRecognizer(
            model_name="PP-FormulaNet_plus-S",
            recognizer_factory=lambda name: created.append(name) or mock.Mock(),
        )

        pipeline.close()

        with self.assertRaisesRegex(RuntimeError, "已经关闭"):
            pipeline.predict("unused.png")
        self.assertEqual(created, [])

    def test_deferred_backend_close_error_does_not_replace_success(self) -> None:
        started = threading.Event()
        release = threading.Event()

        class _CloseFailureRecognizer:
            def predict(self, _image_path: str | Path) -> str:
                started.set()
                release.wait(timeout=5)
                return "x"

            def close(self) -> None:
                raise RuntimeError("native close failure")

        pipeline = FormulaRecognizer(
            model_name="PP-FormulaNet_plus-S",
            recognizer_factory=lambda _name: _CloseFailureRecognizer(),
        )
        results: list[str] = []
        worker = threading.Thread(
            target=lambda: results.append(pipeline.predict("unused.png"))
        )
        worker.start()
        self.assertTrue(started.wait(timeout=2))
        pipeline.close()
        release.set()
        worker.join(timeout=5)

        self.assertFalse(worker.is_alive())
        self.assertEqual(results, ["x"])


class RuntimePathTests(unittest.TestCase):
    def test_paddle_cache_requires_weights_and_metadata(self) -> None:
        data_dir = Path.cwd() / "_formula_ocr_incomplete_model_data"
        self.addCleanup(shutil.rmtree, data_dir, True)
        with mock.patch.dict(os.environ, {"FORMULA_OCR_DATA_DIR": str(data_dir)}):
            model_dir = paddle_model_dir("PP-FormulaNet_plus-S")
            model_dir.mkdir(parents=True)
            (model_dir / "inference.json").write_text("{}", encoding="utf-8")
            (model_dir / "inference.yml").write_text("model", encoding="utf-8")
            self.assertFalse(is_paddle_model_cached("PP-FormulaNet_plus-S"))
            (model_dir / "inference.pdiparams").write_bytes(b"weights")
            self.assertTrue(is_paddle_model_cached("PP-FormulaNet_plus-S"))

    def test_paddle_cleanup_removes_scoped_resume_artifacts(self) -> None:
        data_dir = Path.cwd() / "_formula_ocr_paddle_cleanup"
        self.addCleanup(shutil.rmtree, data_dir, True)
        with mock.patch.dict(os.environ, {"FORMULA_OCR_DATA_DIR": str(data_dir)}):
            model_dir = paddle_model_dir("PP-FormulaNet_plus-S")
            downloads_root = model_dir.parent / ".downloads"
            downloads_root.mkdir(parents=True)
            partial = downloads_root / "PP-FormulaNet_plus-S_infer.tar.part"
            extracting = downloads_root / ".PP-FormulaNet_plus-S.extracting"
            partial.write_bytes(b"partial")
            extracting.mkdir()
            shared_marker = downloads_root / "keep.txt"
            shared_marker.write_text("keep", encoding="utf-8")

            self.assertTrue(paddle_model_has_data("PP-FormulaNet_plus-S"))
            self.assertTrue(remove_paddle_model("PP-FormulaNet_plus-S"))
            self.assertFalse(partial.exists())
            self.assertFalse(extracting.exists())
            self.assertTrue(shared_marker.exists())
            self.assertFalse(paddle_model_has_data("PP-FormulaNet_plus-S"))

    def test_paddle_hf_cleanup_removes_multifile_resume_artifacts(self) -> None:
        data_dir = Path.cwd() / "_formula_ocr_paddle_hf_cleanup"
        self.addCleanup(shutil.rmtree, data_dir, True)
        with mock.patch.dict(os.environ, {"FORMULA_OCR_DATA_DIR": str(data_dir)}):
            model_dir = paddle_model_dir("LaTeX_OCR_rec")
            partial_dir = model_dir.parent / ".downloads" / "LaTeX_OCR_rec"
            partial_dir.mkdir(parents=True)
            (partial_dir / "inference.pdiparams.part").write_bytes(b"partial")
            shared_marker = model_dir.parent / ".downloads" / "keep.txt"
            shared_marker.write_text("keep", encoding="utf-8")

            self.assertTrue(paddle_model_has_data("LaTeX_OCR_rec"))
            self.assertTrue(remove_paddle_model("LaTeX_OCR_rec"))
            self.assertFalse(partial_dir.exists())
            self.assertTrue(shared_marker.exists())
            self.assertFalse(paddle_model_has_data("LaTeX_OCR_rec"))

    def test_paddle_cleanup_includes_legacy_archives_and_failed_backup_cleanup(self) -> None:
        data_dir = Path.cwd() / "_formula_ocr_paddle_legacy_cleanup"
        self.addCleanup(shutil.rmtree, data_dir, True)
        with mock.patch.dict(os.environ, {"FORMULA_OCR_DATA_DIR": str(data_dir)}):
            model_dir = paddle_model_dir("PP-FormulaNet_plus-L")
            backup = model_dir.with_name(model_dir.name + ".bak")
            backup.mkdir(parents=True)
            backup_payload = backup / "old.bin"
            backup_payload.write_bytes(b"backup")
            legacy_archive = model_dir.parent / "PP-FormulaNet_plus-L_infer.tar"
            legacy_partial = model_dir.parent / "PP-FormulaNet_plus-L_infer.tar.part"
            legacy_archive.write_bytes(b"archive")
            legacy_partial.write_bytes(b"partial")

            expected_size = sum(
                path.stat().st_size
                for path in (backup_payload, legacy_archive, legacy_partial)
            )
            self.assertTrue(paddle_model_has_data("PP-FormulaNet_plus-L"))
            self.assertEqual(
                paddle_model_cache_size("PP-FormulaNet_plus-L"),
                expected_size,
            )
            self.assertTrue(remove_paddle_model("PP-FormulaNet_plus-L"))
            self.assertFalse(backup.exists())
            self.assertFalse(legacy_archive.exists())
            self.assertFalse(legacy_partial.exists())
            self.assertFalse(paddle_model_has_data("PP-FormulaNet_plus-L"))

    def test_data_directory_override_controls_cache_logs_and_models(self) -> None:
        data_dir = Path.cwd() / "_formula_ocr_data_override"
        with mock.patch.dict(
            os.environ,
            {"FORMULA_OCR_DATA_DIR": str(data_dir)},
        ):
            resolved = data_dir.resolve()
            self.assertEqual(runtime_cache_dir(), resolved / "cache")
            self.assertEqual(runtime_log_dir(), resolved / "logs")
            self.assertEqual(
                paddle_model_dir("PP-FormulaNet_plus-S"),
                resolved
                / "cache"
                / "runtime"
                / "paddlex"
                / "official_models"
                / "PP-FormulaNet_plus-S",
            )

    def test_external_model_cache_and_partial_downloads_are_scoped(self) -> None:
        data_dir = Path.cwd() / "_formula_ocr_external_cleanup"
        self.addCleanup(shutil.rmtree, data_dir, True)
        with mock.patch.dict(os.environ, {"FORMULA_OCR_DATA_DIR": str(data_dir)}):
            model_dir = external_model_dir("Pix2TextMFR15")
            partial_dir = model_dir.parent / ".downloads" / "Pix2TextMFR15"
            model_dir.mkdir(parents=True)
            partial_dir.mkdir(parents=True)
            (model_dir / "stale.onnx").write_bytes(b"stale")
            (partial_dir / "encoder_model.onnx.part").write_bytes(b"partial")
            shared_marker = model_dir.parent / ".downloads" / "keep.txt"
            shared_marker.write_text("keep", encoding="utf-8")

            self.assertTrue(external_model_has_data("Pix2TextMFR15"))
            self.assertTrue(remove_external_model("Pix2TextMFR15"))
            self.assertFalse(model_dir.exists())
            self.assertFalse(partial_dir.exists())
            self.assertTrue(shared_marker.exists())
            self.assertFalse(external_model_has_data("Pix2TextMFR15"))

    def test_mixtex_partial_download_cleanup_is_scoped(self) -> None:
        data_dir = Path.cwd() / "_formula_ocr_mixtex_cleanup"
        self.addCleanup(shutil.rmtree, data_dir, True)
        with mock.patch.dict(os.environ, {"FORMULA_OCR_DATA_DIR": str(data_dir)}):
            model_dir = external_model_dir("MixTexZhEn")
            partial_dir = model_dir.parent / ".downloads" / "MixTexZhEn"
            partial_dir.mkdir(parents=True)
            (partial_dir / "MixTeX.zip.part").write_bytes(b"partial")
            shared_marker = model_dir.parent / ".downloads" / "keep.txt"
            shared_marker.write_text("keep", encoding="utf-8")

            self.assertTrue(external_model_has_data("MixTexZhEn"))
            self.assertTrue(remove_external_model("MixTexZhEn"))
            self.assertFalse(partial_dir.exists())
            self.assertTrue(shared_marker.exists())
            self.assertFalse(external_model_has_data("MixTexZhEn"))

    def test_external_cleanup_includes_failed_backup_cleanup(self) -> None:
        data_dir = Path.cwd() / "_formula_ocr_external_backup_cleanup"
        self.addCleanup(shutil.rmtree, data_dir, True)
        with mock.patch.dict(os.environ, {"FORMULA_OCR_DATA_DIR": str(data_dir)}):
            model_dir = external_model_dir("MathCraftFormula")
            backup = model_dir.with_name(model_dir.name + ".bak")
            backup.mkdir(parents=True)
            payload = backup / "old.onnx"
            payload.write_bytes(b"old-model")
            extraction = model_dir.parent / ".downloads" / ".MathCraftFormula-crash"
            extraction.mkdir(parents=True)
            extraction_payload = extraction / "encoder_model.onnx"
            extraction_payload.write_bytes(b"crash-leftover")

            self.assertTrue(external_model_has_data("MathCraftFormula"))
            self.assertEqual(
                external_model_cache_size("MathCraftFormula"),
                payload.stat().st_size + extraction_payload.stat().st_size,
            )
            self.assertTrue(remove_external_model("MathCraftFormula"))
            self.assertFalse(backup.exists())
            self.assertFalse(extraction.exists())
            self.assertFalse(external_model_has_data("MathCraftFormula"))

    def test_frozen_build_allows_missing_model_to_download(self) -> None:
        data_dir = Path.cwd() / "_nonexistent_formula_ocr_frozen_data"
        recognizer = PaddleFormulaRecognizer(
            model_name="PP-FormulaNet_plus-S",
        )
        with mock.patch.dict(
            os.environ,
            {"FORMULA_OCR_DATA_DIR": str(data_dir)},
        ), mock.patch("sys.frozen", True, create=True):
            self.assertIsNone(recognizer._cached_model_dir())

    def test_frozen_external_models_use_user_data_not_internal(self) -> None:
        local_app_data = Path.cwd() / "_formula_ocr_local_app_data"
        self.addCleanup(shutil.rmtree, local_app_data, True)
        with mock.patch.dict(
            os.environ,
            {
                "FORMULA_OCR_DATA_DIR": "",
                "LOCALAPPDATA": str(local_app_data),
                "XDG_DATA_HOME": str(local_app_data),
            },
        ), mock.patch("sys.frozen", True, create=True):
            self.assertEqual(
                external_model_dir("Pix2TextMFR15"),
                local_app_data.resolve()
                / "FormulaOCR"
                / "cache"
                / "models"
                / "Pix2TextMFR15",
            )
            self.assertNotIn("_internal", str(external_model_dir("Pix2TextMFR15")))

    def test_frozen_internal_data_override_is_rejected(self) -> None:
        root = Path.cwd() / "_formula_ocr_frozen_override"
        self.addCleanup(shutil.rmtree, root, True)
        install_root = root / "FormulaOCR"
        internal = install_root / "_internal"
        executable = install_root / "FormulaOCR.exe"
        local_app_data = root / "LocalAppData"
        with mock.patch.dict(
            os.environ,
            {
                "FORMULA_OCR_DATA_DIR": str(internal),
                "LOCALAPPDATA": str(local_app_data),
                "XDG_DATA_HOME": str(local_app_data),
            },
        ), mock.patch("sys.frozen", True, create=True), mock.patch(
            "sys.executable", str(executable)
        ):
            cache = runtime_cache_dir()
            self.assertEqual(
                cache,
                local_app_data.resolve() / "FormulaOCR" / "cache",
            )
            self.assertNotIn("_internal", str(cache))

    def test_bundled_external_model_is_discovered_as_read_only_resource(self) -> None:
        bundle_root = Path.cwd() / "_formula_ocr_bundle_root"
        self.addCleanup(shutil.rmtree, bundle_root, True)
        bundled = bundle_root / "models" / "onnx" / "Pix2TextMFR15"
        bundled.mkdir(parents=True)
        with mock.patch("sys._MEIPASS", str(bundle_root), create=True):
            self.assertEqual(bundled_external_model_dir("Pix2TextMFR15"), bundled)
            self.assertTrue(is_external_model_bundled("Pix2TextMFR15"))

    def test_pix2text_can_use_a_valid_bundled_model_without_downloading(self) -> None:
        root = Path.cwd() / "_pix2text_bundle_test"
        self.addCleanup(shutil.rmtree, root, True)
        bundle = root / "bundle"
        user = root / "user"
        bundle.mkdir(parents=True)
        payload = b"bundled-model"
        item = Pix2TextModelFile(
            "tiny.bin",
            len(payload),
            hashlib.sha256(payload).hexdigest(),
        )
        (bundle / item.name).write_bytes(payload)
        with mock.patch.object(
            pix2text_downloader,
            "PIX2TEXT_MODEL_FILES",
            (item,),
        ), mock.patch.object(
            pix2text_downloader,
            "bundled_external_model_dir",
            return_value=bundle,
        ), mock.patch.object(
            pix2text_downloader,
            "pix2text_model_dir",
            return_value=user,
        ):
            self.assertTrue(
                pix2text_downloader.is_pix2text_model_cached(verify_hash=True)
            )
            resolved = pix2text_downloader.ensure_pix2text_model()
        self.assertEqual(resolved, bundle)
        self.assertFalse((user / item.name).exists())
        self.assertFalse(user.exists())

    def test_pix2text_prefers_valid_user_cache_over_bundled_model(self) -> None:
        root = Path.cwd() / "_pix2text_user_priority_test"
        self.addCleanup(shutil.rmtree, root, True)
        bundle = root / "bundle"
        user = root / "user"
        bundle.mkdir(parents=True)
        user.mkdir(parents=True)
        bundle_payload = b"bundled-model"
        user_payload = b"user-model!!!"
        item = Pix2TextModelFile(
            "tiny.bin",
            len(bundle_payload),
            hashlib.sha256(user_payload).hexdigest(),
        )
        # Keep the declared size/hash pair valid for the user file while
        # making the bundled copy a different, invalid revision.
        (bundle / item.name).write_bytes(bundle_payload)
        (user / item.name).write_bytes(user_payload)
        with mock.patch.object(
            pix2text_downloader,
            "PIX2TEXT_MODEL_FILES",
            (item,),
        ), mock.patch.object(
            pix2text_downloader,
            "bundled_external_model_dir",
            return_value=bundle,
        ), mock.patch.object(
            pix2text_downloader,
            "pix2text_model_dir",
            return_value=user,
        ):
            resolved = pix2text_downloader.ensure_pix2text_model()
        self.assertEqual(resolved, user)

    def test_latex_ocr_can_use_a_valid_bundled_model_without_downloading(self) -> None:
        root = Path.cwd() / "_latex_ocr_bundle_test"
        self.addCleanup(shutil.rmtree, root, True)
        bundle = root / "bundle"
        user = root / "user"
        bundle.mkdir(parents=True)
        payload = b"bundled-paddle-model"
        item = PaddleHFModelFile(
            "inference.pdiparams",
            len(payload),
            hashlib.sha256(payload).hexdigest(),
        )
        (bundle / item.name).write_bytes(payload)
        with mock.patch.object(
            paddle_hf_downloader,
            "PADDLE_HF_MODEL_FILES",
            (item,),
        ), mock.patch.object(
            paddle_hf_downloader,
            "PADDLE_HF_TOTAL_SIZE",
            len(payload),
        ), mock.patch.object(
            paddle_hf_downloader,
            "bundled_paddle_model_dir",
            return_value=bundle,
        ), mock.patch.object(
            paddle_hf_downloader,
            "paddle_hf_model_dir",
            return_value=user,
        ):
            self.assertTrue(
                paddle_hf_downloader.is_paddle_hf_model_cached(verify_hash=True)
            )
            resolved = paddle_hf_downloader.ensure_paddle_hf_model()
        self.assertEqual(resolved, bundle)
        self.assertFalse(user.exists())

    def test_missing_model_is_downloaded_before_direct_runtime_initialization(self) -> None:
        recognizer = PaddleFormulaRecognizer(
            model_name="PP-FormulaNet_plus-S",
        )
        recognizer._configure_runtime_cache = lambda: None  # type: ignore[method-assign]
        initialized: list[Path] = []
        recognizer._initialize_runtime = initialized.append  # type: ignore[method-assign]

        data_dir = Path.cwd() / "_nonexistent_formula_ocr_download_data"
        shutil.rmtree(data_dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, data_dir, True)
        downloaded_model = data_dir / "downloaded-model"
        with mock.patch.dict(
            os.environ,
            {"FORMULA_OCR_DATA_DIR": str(data_dir)},
        ), mock.patch("sys.frozen", True, create=True), mock.patch(
            "formula_ocr_app.model_downloader.ensure_official_model",
            return_value=downloaded_model,
        ) as download:
            recognizer._ensure_model()

        download.assert_called_once_with(
            "PP-FormulaNet_plus-S",
            progress_callback=None,
        )
        self.assertEqual(initialized, [downloaded_model.resolve()])


class InterProcessFileLockTests(unittest.TestCase):
    def test_lock_blocks_a_second_owner_and_can_be_reacquired(self) -> None:
        lock_path = Path.cwd() / "_formula_ocr_interprocess_lock_test.lock"
        self.addCleanup(lock_path.unlink, missing_ok=True)
        first = InterProcessFileLock(lock_path)
        second = InterProcessFileLock(
            lock_path,
            timeout=0.05,
            poll_interval=0.01,
        )

        first.acquire()
        try:
            with self.assertRaises(TimeoutError):
                second.acquire()
        finally:
            first.release()

        with second:
            self.assertTrue(lock_path.is_file())

    def test_wait_callback_can_cancel_lock_acquisition(self) -> None:
        lock_path = Path.cwd() / "_formula_ocr_cancel_lock_test.lock"
        self.addCleanup(lock_path.unlink, missing_ok=True)
        first = InterProcessFileLock(lock_path)
        callbacks: list[str] = []

        def cancel_wait() -> None:
            callbacks.append("wait")
            raise ModelDownloadCancelled()

        first.acquire()
        try:
            second = InterProcessFileLock(
                lock_path,
                timeout=2,
                poll_interval=0.01,
                on_wait=cancel_wait,
                on_wait_interval=0.01,
            )
            with self.assertRaises(ModelDownloadCancelled):
                second.acquire()
        finally:
            first.release()

        self.assertEqual(callbacks, ["wait"])

    def test_remove_model_refuses_while_download_lock_is_held(self) -> None:
        data_dir = Path.cwd() / "_formula_ocr_locked_model_remove"
        self.addCleanup(shutil.rmtree, data_dir, True)
        with mock.patch.dict(os.environ, {"FORMULA_OCR_DATA_DIR": str(data_dir)}):
            model_dir = external_model_dir("RapidLaTeXOCR")
            model_dir.mkdir(parents=True)
            (model_dir / "partial.bin").write_bytes(b"partial")
            lock = InterProcessFileLock(model_dir.with_suffix(".lock"))
            lock.acquire()
            try:
                with self.assertRaisesRegex(OSError, "正在另一个 FormulaOCR"):
                    remove_model("RapidLaTeXOCR")
                self.assertTrue(model_dir.is_dir())
            finally:
                lock.release()

            self.assertTrue(remove_model("RapidLaTeXOCR"))
            self.assertFalse(model_dir.exists())


class ModelDownloaderTests(unittest.TestCase):
    def test_directory_install_restores_valid_backup_when_swap_fails(self) -> None:
        root = Path.cwd() / "_model_directory_swap_rollback"
        self.addCleanup(shutil.rmtree, root, True)
        source = root / "extracting"
        destination = root / "model"
        backup = root / "model.bak"
        source.mkdir(parents=True)
        backup.mkdir()
        (source / "model.bin").write_bytes(b"new")
        (backup / "model.bin").write_bytes(b"old")

        original_replace = Path.replace

        def fail_new_model_swap(path: Path, target: Path):
            if path == source:
                raise OSError("simulated interrupted model swap")
            return original_replace(path, target)

        with mock.patch("pathlib.Path.replace", new=fail_new_model_swap):
            with self.assertRaisesRegex(OSError, "interrupted model swap"):
                replace_model_directory(
                    source,
                    destination,
                    is_model_valid=lambda path: (path / "model.bin").is_file(),
                )

        self.assertEqual((destination / "model.bin").read_bytes(), b"old")
        self.assertFalse(backup.exists())
        self.assertEqual((source / "model.bin").read_bytes(), b"new")

    def test_directory_install_recovers_stale_backup_before_replacing_it(self) -> None:
        root = Path.cwd() / "_model_directory_swap_recovery"
        self.addCleanup(shutil.rmtree, root, True)
        source = root / "extracting"
        destination = root / "model"
        backup = root / "model.bak"
        source.mkdir(parents=True)
        backup.mkdir()
        (source / "model.bin").write_bytes(b"new")
        (backup / "model.bin").write_bytes(b"old")

        replace_model_directory(
            source,
            destination,
            is_model_valid=lambda path: (path / "model.bin").is_file(),
        )

        self.assertEqual((destination / "model.bin").read_bytes(), b"new")
        self.assertFalse(source.exists())
        self.assertFalse(backup.exists())

    def test_runtime_cache_checks_external_models_with_sha256_by_default(self) -> None:
        checker = mock.Mock(return_value=False)
        with mock.patch.object(
            model_runtime,
            "_load_module",
            return_value=mock.Mock(is_rapid_model_cached=checker),
        ):
            self.assertFalse(model_runtime.is_model_cached("RapidLaTeXOCR"))
        checker.assert_called_once_with(verify_hash=True)

    def test_model_status_label_uses_fast_cache_check(self) -> None:
        checker = mock.Mock(return_value=False)
        with mock.patch.object(model_runtime, "is_model_cached", checker), mock.patch.object(
            model_runtime,
            "model_has_user_cache_data",
            return_value=False,
        ), mock.patch.object(model_runtime, "is_model_bundled", return_value=False):
            self.assertEqual(
                model_runtime.model_status_label("RapidLaTeXOCR"),
                "待下载",
            )
        checker.assert_called_once_with("RapidLaTeXOCR", verify_hash=False)

    def test_repeated_sha256_validation_uses_file_signature_cache(self) -> None:
        from formula_ocr_app import download_utils

        root = Path.cwd() / "_formula_ocr_hash_cache"
        self.addCleanup(shutil.rmtree, root, True)
        root.mkdir()
        path = root / "model.bin"
        payload = b"verified-model"
        path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        download_utils._HASH_CACHE.clear()
        self.addCleanup(download_utils._HASH_CACHE.clear)

        with mock.patch.object(
            download_utils,
            "_sha256_stream",
            wraps=download_utils._sha256_stream,
        ) as hasher:
            self.assertTrue(
                download_utils.file_is_valid(
                    path,
                    len(payload),
                    digest,
                    verify_hash=True,
                )
            )
            self.assertTrue(
                download_utils.file_is_valid(
                    path,
                    len(payload),
                    digest,
                    verify_hash=True,
                )
            )
        self.assertEqual(hasher.call_count, 1)

    def test_sha256_cache_rejects_atomic_same_size_replacement(self) -> None:
        from formula_ocr_app import download_utils

        root = Path.cwd() / "_formula_ocr_hash_replacement"
        self.addCleanup(shutil.rmtree, root, True)
        root.mkdir()
        path = root / "model.bin"
        replacement = root / "replacement.bin"
        original = b"original-model-data"
        changed = b"modified-model-data"
        self.assertEqual(len(original), len(changed))
        path.write_bytes(original)
        original_stat = path.stat()
        original_digest = hashlib.sha256(original).hexdigest()
        changed_digest = hashlib.sha256(changed).hexdigest()
        download_utils._HASH_CACHE.clear()
        self.addCleanup(download_utils._HASH_CACHE.clear)

        self.assertTrue(
            download_utils.file_is_valid(
                path,
                len(original),
                original_digest,
                verify_hash=True,
            )
        )
        replacement.write_bytes(changed)
        os.utime(
            replacement,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
        )
        os.replace(replacement, path)
        # Size and mtime deliberately match the cached file. File identity and
        # the live handle/path comparison must still invalidate the digest.
        self.assertFalse(
            download_utils.file_is_valid(
                path,
                len(original),
                original_digest,
                verify_hash=True,
            )
        )
        self.assertTrue(
            download_utils.file_is_valid(
                path,
                len(changed),
                changed_digest,
                verify_hash=True,
            )
        )

    def test_legacy_preview_cleanup_is_scoped_and_age_guarded(self) -> None:
        from formula_ocr_app import app as app_module

        root = Path.cwd() / "_formula_ocr_legacy_preview_cleanup"
        self.addCleanup(shutil.rmtree, root, True)
        legacy_root = root / "mathml_preview"
        stale_profile = legacy_root / "profile_stale"
        recent_profile = legacy_root / "profile_recent"
        stale_preview = legacy_root / "preview_stale.png"
        unknown_file = legacy_root / "user-note.txt"
        stale_profile.mkdir(parents=True)
        recent_profile.mkdir()
        (stale_profile / "Cache.bin").write_bytes(b"stale")
        (recent_profile / "Cache.bin").write_bytes(b"recent")
        stale_preview.write_bytes(b"stale")
        unknown_file.write_text("keep", encoding="utf-8")
        old_timestamp = time.time() - app_module.LEGACY_PREVIEW_MAX_AGE_SECONDS - 60
        for stale_path in (
            stale_profile / "Cache.bin",
            stale_profile,
            stale_preview,
            unknown_file,
        ):
            os.utime(stale_path, (old_timestamp, old_timestamp))

        with mock.patch.object(app_module, "CACHE_DIR", root):
            app_module._cleanup_legacy_mathml_preview_cache()

        self.assertFalse(stale_profile.exists())
        self.assertFalse(stale_preview.exists())
        self.assertTrue(recent_profile.is_dir())
        self.assertTrue(unknown_file.is_file())

    def test_download_cancellation_after_atomic_install_reports_model_ready(self) -> None:
        from formula_ocr_app import app as app_module

        fake_app = mock.Mock()
        fake_app.download_cancel_event = threading.Event()
        fake_app.worker_queue = queue.Queue()
        with mock.patch.object(
            app_module,
            "ensure_model",
            side_effect=ModelDownloadCancelled(),
        ), mock.patch.object(app_module, "is_model_cached", return_value=True):
            app_module.FormulaOCRApp._prepare_model_worker(
                fake_app,
                "PP-FormulaNet_plus-S",
            )

        self.assertEqual(
            fake_app.worker_queue.get_nowait(),
            ("model_ready", "PP-FormulaNet_plus-S"),
        )

    def test_resumed_download_restarts_when_content_range_is_missing(self) -> None:
        root = Path.cwd() / "_formula_ocr_bad_content_range"
        self.addCleanup(shutil.rmtree, root, True)
        payload = b"abc"
        partial = root / "file.bin.part"
        partial.parent.mkdir(parents=True)
        partial.write_bytes(payload[:1])
        destination = root / "file.bin"
        bad_response = mock.Mock()
        bad_response.status_code = 206
        bad_response.headers = {}
        bad_response.raise_for_status.return_value = None

        class _FullResponse:
            status_code = 200
            headers = {}

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int):
                return [payload]

            def close(self) -> None:
                return None

        request_get = mock.Mock(side_effect=(bad_response, _FullResponse()))
        download_verified_file(
            RemoteFileSpec(
                "file.bin",
                len(payload),
                hashlib.sha256(payload).hexdigest(),
                "https://example.invalid/file.bin",
            ),
            destination,
            partial=partial,
            completed=0,
            total=len(payload),
            notify=lambda *_args: None,
            request_get=request_get,
            request_exception=OSError,
        )

        bad_response.close.assert_called_once_with()
        self.assertEqual(request_get.call_count, 2)
        self.assertEqual(
            request_get.call_args_list[0].kwargs["headers"]["Range"],
            "bytes=1-",
        )
        self.assertNotIn("Range", request_get.call_args_list[1].kwargs["headers"])
        self.assertEqual(destination.read_bytes(), payload)
        self.assertFalse(partial.exists())

    def test_preflight_cancellation_prevents_network_request(self) -> None:
        root = Path.cwd() / "_formula_ocr_close_error_cancel"
        self.addCleanup(shutil.rmtree, root, True)
        payload = b"abc"
        request_get = mock.Mock()

        with self.assertRaises(ModelDownloadCancelled):
            download_verified_file(
                RemoteFileSpec(
                    "file.bin",
                    len(payload),
                    hashlib.sha256(payload).hexdigest(),
                    "https://example.invalid/file.bin",
                ),
                root / "file.bin",
                partial=root / "file.bin.part",
                completed=0,
                total=len(payload),
                notify=lambda *_args: (_ for _ in ()).throw(
                    ModelDownloadCancelled()
                ),
                request_get=request_get,
                request_exception=OSError,
            )

        request_get.assert_not_called()

    def test_response_close_error_does_not_mask_download_cancellation(self) -> None:
        root = Path.cwd() / "_formula_ocr_close_error_cancel_after_request"
        self.addCleanup(shutil.rmtree, root, True)
        payload = b"abc"
        response = mock.Mock()
        response.status_code = 200
        response.headers = {}
        response.raise_for_status.return_value = None
        response.iter_content.return_value = [payload]
        response.close.side_effect = RuntimeError("close failed")
        notification_count = 0

        def cancel_after_request(*_args) -> None:
            nonlocal notification_count
            notification_count += 1
            if notification_count >= 2:
                raise ModelDownloadCancelled()

        with self.assertRaises(ModelDownloadCancelled):
            download_verified_file(
                RemoteFileSpec(
                    "file.bin",
                    len(payload),
                    hashlib.sha256(payload).hexdigest(),
                    "https://example.invalid/file.bin",
                ),
                root / "file.bin",
                partial=root / "file.bin.part",
                completed=0,
                total=len(payload),
                notify=cancel_after_request,
                request_get=lambda *_args, **_kwargs: response,
                request_exception=OSError,
            )

        response.close.assert_called_once_with()

    def test_truncated_response_keeps_partial_for_resume(self) -> None:
        root = Path.cwd() / "_formula_ocr_truncated_download"
        self.addCleanup(shutil.rmtree, root, True)
        payload = b"complete-payload"
        response = mock.Mock()
        response.status_code = 200
        response.headers = {}
        response.raise_for_status.return_value = None
        response.iter_content.return_value = [payload[:7]]

        with self.assertRaises(VerifiedDownloadFailure) as raised:
            download_verified_file(
                RemoteFileSpec(
                    "file.bin",
                    len(payload),
                    hashlib.sha256(payload).hexdigest(),
                    "https://example.invalid/file.bin",
                ),
                root / "file.bin",
                partial=root / "file.bin.part",
                completed=0,
                total=len(payload),
                notify=lambda *_args: None,
                request_get=lambda *_args, **_kwargs: response,
                request_exception=OSError,
            )

        self.assertEqual(raised.exception.phase, "transfer")
        self.assertEqual((root / "file.bin.part").read_bytes(), payload[:7])
        self.assertFalse((root / "file.bin").exists())

    def test_complete_file_with_wrong_hash_discards_partial(self) -> None:
        root = Path.cwd() / "_formula_ocr_bad_checksum"
        self.addCleanup(shutil.rmtree, root, True)
        expected = b"abc"
        response = mock.Mock()
        response.status_code = 200
        response.headers = {}
        response.raise_for_status.return_value = None
        response.iter_content.return_value = [b"xyz"]

        with self.assertRaises(VerifiedDownloadFailure) as raised:
            download_verified_file(
                RemoteFileSpec(
                    "file.bin",
                    len(expected),
                    hashlib.sha256(expected).hexdigest(),
                    "https://example.invalid/file.bin",
                ),
                root / "file.bin",
                partial=root / "file.bin.part",
                completed=0,
                total=len(expected),
                notify=lambda *_args: None,
                request_get=lambda *_args, **_kwargs: response,
                request_exception=OSError,
            )

        self.assertEqual(raised.exception.phase, "checksum")
        self.assertFalse((root / "file.bin.part").exists())

    def test_official_paddle_cancellation_prevents_network_request(self) -> None:
        root = Path.cwd() / "_formula_ocr_paddle_preflight_cancel"
        self.addCleanup(shutil.rmtree, root, True)
        request_get = mock.Mock()
        spec = MODEL_BY_ID["PP-FormulaNet_plus-S"]

        with mock.patch("requests.get", request_get), self.assertRaises(
            ModelDownloadCancelled
        ):
            _download_official_archive(
                spec,
                root / spec.archive_name,
                lambda *_args: (_ for _ in ()).throw(ModelDownloadCancelled()),
            )

        request_get.assert_not_called()

    def test_official_paddle_bad_content_range_restarts_full_download(self) -> None:
        root = Path.cwd() / "_formula_ocr_official_bad_content_range"
        self.addCleanup(shutil.rmtree, root, True)
        root.mkdir()
        payload = b"official-model"
        archive = root / "model.tar"
        partial = archive.with_suffix(".tar.part")
        partial.write_bytes(payload[:4])
        spec = mock.Mock(
            model_id="test-model",
            archive_size=len(payload),
            archive_crc32=zlib.crc32(payload) & 0xFFFFFFFF,
            download_url="https://example.invalid/model.tar",
        )

        class _Response:
            def __init__(self, status_code: int, chunks: list[bytes], headers=None):
                self.status_code = status_code
                self.headers = headers or {}
                self.chunks = chunks
                self.closed = False

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int):
                return self.chunks

            def close(self) -> None:
                self.closed = True

        bad_response = _Response(206, [], headers={})
        full_response = _Response(200, [payload])
        request_get = mock.Mock(side_effect=(bad_response, full_response))

        with mock.patch("requests.get", request_get):
            _download_official_archive(spec, archive, None)

        self.assertTrue(bad_response.closed)
        self.assertTrue(full_response.closed)
        self.assertEqual(request_get.call_count, 2)
        self.assertEqual(
            request_get.call_args_list[0].kwargs["headers"]["Range"],
            "bytes=4-",
        )
        self.assertNotIn("Range", request_get.call_args_list[1].kwargs["headers"])
        self.assertEqual(archive.read_bytes(), payload)
        self.assertFalse(partial.exists())

    def test_model_root_symlink_is_not_accepted_as_cache(self) -> None:
        root = Path.cwd() / "_formula_ocr_model_symlink"
        self.addCleanup(shutil.rmtree, root, True)
        payload = b"model"
        real_root = root / "real"
        real_root.mkdir(parents=True)
        (real_root / "model.bin").write_bytes(payload)
        linked_root = root / "linked"
        try:
            linked_root.symlink_to(real_root, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"symbolic links are unavailable: {exc}")
        item = RemoteFileSpec(
            "model.bin",
            len(payload),
            hashlib.sha256(payload).hexdigest(),
            "https://example.invalid/model.bin",
        )

        self.assertFalse(
            model_files_are_valid(linked_root, (item,), verify_hash=False)
        )

    def test_paddle_cache_rejects_empty_files_and_linked_root(self) -> None:
        root = Path.cwd() / "_formula_ocr_paddle_cache_validation"
        self.addCleanup(shutil.rmtree, root, True)
        real_root = root / "real"
        real_root.mkdir(parents=True)
        for filename in ("inference.json", "inference.yml", "inference.pdiparams"):
            (real_root / filename).write_bytes(b"")
        self.assertFalse(_paddle_model_files_exist(real_root))
        for filename in ("inference.json", "inference.yml", "inference.pdiparams"):
            (real_root / filename).write_bytes(b"x")
        self.assertTrue(_paddle_model_files_exist(real_root))

        linked_root = root / "linked"
        try:
            linked_root.symlink_to(real_root, target_is_directory=True)
        except OSError:
            return
        self.assertFalse(_paddle_model_files_exist(linked_root))

    def test_official_model_specs_have_expected_download_sizes(self) -> None:
        self.assertEqual(
            MODEL_BY_ID["PP-FormulaNet_plus-S"].archive_size,
            259_604_480,
        )
        self.assertEqual(
            MODEL_BY_ID["PP-FormulaNet_plus-L"].archive_size,
            731_504_640,
        )

    def test_safe_tar_members_are_accepted(self) -> None:
        directory = tarfile.TarInfo("./PP-FormulaNet_plus-S_infer")
        directory.type = tarfile.DIRTYPE
        model = tarfile.TarInfo(
            "./PP-FormulaNet_plus-S_infer/inference.pdiparams"
        )
        _validate_tar_members([directory, model])

    def test_tar_path_traversal_is_rejected(self) -> None:
        with self.assertRaises(ModelDownloadError):
            _validate_tar_members([tarfile.TarInfo("../outside.txt")])

    def test_tar_links_are_rejected(self) -> None:
        link = tarfile.TarInfo("model/link")
        link.type = tarfile.SYMTYPE
        link.linkname = "../outside.txt"
        with self.assertRaises(ModelDownloadError):
            _validate_tar_members([link])

    def test_tar_windows_path_traversal_is_rejected(self) -> None:
        with self.assertRaises(ModelDownloadError):
            _validate_tar_members([tarfile.TarInfo(r"nested\..\outside.txt")])
        with self.assertRaises(ModelDownloadError):
            _validate_tar_members([tarfile.TarInfo(r"C:\outside.txt")])

    def test_tar_windows_aliases_and_extraction_bombs_are_rejected(self) -> None:
        for name in ("model/file.bin:stream", "model/CON", "model/name. "):
            with self.subTest(name=name):
                with self.assertRaises(ModelDownloadError):
                    _validate_tar_members([tarfile.TarInfo(name)])
        oversized = tarfile.TarInfo("model/huge.bin")
        oversized.size = 8 * 1024 * 1024 * 1024 + 1
        with self.assertRaises(ModelDownloadError):
            _validate_tar_members([oversized])

    def test_mathcraft_manifest_has_sha256_entries(self) -> None:
        self.assertEqual(len(MATHCRAFT_ARCHIVE_SHA256), 64)
        self.assertRegex(MATHCRAFT_ARCHIVE_SHA256, r"^[0-9a-f]{64}$")
        for name, digest in MATHCRAFT_MODEL_FILES.items():
            with self.subTest(name=name):
                self.assertTrue(name)
                self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_mathcraft_zip_path_traversal_is_rejected(self) -> None:
        with self.assertRaises(MathCraftModelDownloadError):
            _validate_zip_members([zipfile.ZipInfo("../outside.txt")])
        with self.assertRaises(MathCraftModelDownloadError):
            _validate_zip_members([zipfile.ZipInfo("nested\\..\\outside.txt")])
        with self.assertRaises(MathCraftModelDownloadError):
            _validate_zip_members([zipfile.ZipInfo("C:\\outside.txt")])

    def test_mathcraft_zip_windows_aliases_and_bombs_are_rejected(self) -> None:
        for name in ("model/file.bin:stream", "model/NUL", "model/name. "):
            with self.subTest(name=name):
                with self.assertRaises(MathCraftModelDownloadError):
                    _validate_zip_members([zipfile.ZipInfo(name)])
        oversized = zipfile.ZipInfo("model/huge.bin")
        oversized.file_size = 8 * 1024 * 1024 * 1024 + 1
        with self.assertRaises(MathCraftModelDownloadError):
            _validate_zip_members([oversized])

    def test_mathcraft_file_with_wrong_hash_is_rejected(self) -> None:
        path = Path.cwd() / "_mathcraft_wrong_hash.bin"
        self.addCleanup(path.unlink, missing_ok=True)
        path.write_bytes(b"not-the-official-model")
        self.assertFalse(
            _file_is_valid(path, "0" * 64, verify_hash=True)
        )

    def test_mathcraft_fake_zip_cannot_install_unsafe_member(self) -> None:
        root = Path.cwd() / "_mathcraft_zip_install_test"
        self.addCleanup(shutil.rmtree, root, True)
        downloads = root / "downloads"
        downloads.mkdir(parents=True)
        archive_path = downloads / "model.zip"
        destination = root / "MathCraftFormula"
        manifest = {"config.json": hashlib.sha256(b"{}").hexdigest()}
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("config.json", b"{}")
            archive.writestr("../outside.txt", b"must not escape")
        with mock.patch(
            "formula_ocr_app.mathcraft_model_downloader.MATHCRAFT_MODEL_FILES",
            manifest,
        ):
            with self.assertRaises(MathCraftModelDownloadError):
                _install_archive(archive_path, destination, downloads)
        self.assertFalse((root / "outside.txt").exists())
        self.assertFalse(destination.exists())

    def test_successful_archive_installs_ignore_locked_cleanup_artifacts(self) -> None:
        root = Path.cwd() / "_locked_archive_cleanup_test"
        self.addCleanup(shutil.rmtree, root, True)
        root.mkdir()
        original_unlink = Path.unlink
        original_rmtree = shutil.rmtree

        def install_with_locked_cleanup(
            archive_path: Path,
            destination: Path,
            installer,
            *installer_args,
        ) -> None:
            backup = destination.with_name(destination.name + ".bak")

            def guarded_unlink(path: Path, *args, **kwargs) -> None:
                if path == archive_path:
                    raise PermissionError("archive is temporarily locked")
                original_unlink(path, *args, **kwargs)

            def guarded_rmtree(path, *args, **kwargs) -> None:
                if Path(path) == backup:
                    if kwargs.get("ignore_errors"):
                        return
                    raise PermissionError("backup is temporarily locked")
                original_rmtree(path, *args, **kwargs)

            with mock.patch("pathlib.Path.unlink", new=guarded_unlink), mock.patch(
                "shutil.rmtree", new=guarded_rmtree
            ):
                installer(archive_path, destination, *installer_args)

            self.assertTrue(destination.is_dir())
            self.assertTrue(archive_path.is_file())

        official_root = root / "official"
        official_downloads = official_root / "downloads"
        official_downloads.mkdir(parents=True)
        official_archive = official_downloads / "model.tar"
        official_destination = official_root / "model"
        official_destination.mkdir()
        (official_destination / "old.bin").write_bytes(b"old")
        official_spec = MODEL_BY_ID["PP-FormulaNet_plus-S"]
        payload_root = root / official_spec.archive_root
        payload_root.mkdir()
        for name in ("inference.json", "inference.yml", "inference.pdiparams"):
            (payload_root / name).write_bytes(b"x")
        with tarfile.open(official_archive, "w") as archive:
            archive.add(payload_root, arcname=official_spec.archive_root)
        install_with_locked_cleanup(
            official_archive,
            official_destination,
            lambda archive, destination, spec, downloads: _install_official_archive(
                spec,
                archive,
                destination,
                downloads,
            ),
            official_spec,
            official_downloads,
        )

        mathcraft_root = root / "mathcraft"
        mathcraft_downloads = mathcraft_root / "downloads"
        mathcraft_downloads.mkdir(parents=True)
        mathcraft_archive = mathcraft_downloads / "model.zip"
        mathcraft_destination = mathcraft_root / "model"
        mathcraft_destination.mkdir()
        (mathcraft_destination / "old.bin").write_bytes(b"old")
        mathcraft_payload = b"{}"
        with zipfile.ZipFile(mathcraft_archive, "w") as archive:
            archive.writestr("config.json", mathcraft_payload)
        with mock.patch(
            "formula_ocr_app.mathcraft_model_downloader.MATHCRAFT_MODEL_FILES",
            {"config.json": hashlib.sha256(mathcraft_payload).hexdigest()},
        ):
            install_with_locked_cleanup(
                mathcraft_archive,
                mathcraft_destination,
                _install_archive,
                mathcraft_downloads,
            )

        mixtex_root = root / "mixtex"
        mixtex_downloads = mixtex_root / "downloads"
        mixtex_downloads.mkdir(parents=True)
        mixtex_archive = mixtex_downloads / "model.zip"
        mixtex_destination = mixtex_root / "model"
        mixtex_destination.mkdir()
        (mixtex_destination / "old.bin").write_bytes(b"old")
        mixtex_payload = b"{}"
        with zipfile.ZipFile(mixtex_archive, "w") as archive:
            archive.writestr("onnx/config.json", mixtex_payload)
        with mock.patch(
            "formula_ocr_app.mixtex_model_downloader.MIXTEX_MODEL_FILES",
            {
                "config.json": (
                    len(mixtex_payload),
                    hashlib.sha256(mixtex_payload).hexdigest(),
                )
            },
        ):
            install_with_locked_cleanup(
                mixtex_archive,
                mixtex_destination,
                _install_mixtex_archive,
                mixtex_downloads,
            )

    def test_pix2text_manifest_matches_catalog_size(self) -> None:
        self.assertEqual(
            sum(item.size for item in PIX2TEXT_MODEL_FILES),
            MODEL_BY_ID["Pix2TextMFR15"].archive_size,
        )
        for item in PIX2TEXT_MODEL_FILES:
            with self.subTest(name=item.name):
                self.assertEqual(len(item.sha256), 64)
                self.assertRegex(item.sha256, r"^[0-9a-f]{64}$")

    def test_pix2text_download_installs_verified_file_atomically(self) -> None:
        root = Path.cwd() / "_pix2text_download_test"
        self.addCleanup(shutil.rmtree, root, True)
        downloads = root / "downloads"
        downloads.mkdir(parents=True)
        payload = b"verified-pix2text-file"
        item = Pix2TextModelFile(
            "tiny.bin",
            len(payload),
            hashlib.sha256(payload).hexdigest(),
        )
        destination = root / "model" / item.name
        events: list[tuple[str, int, int]] = []

        class _Response:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int):
                self.chunk_size = chunk_size
                return [payload]

            def close(self) -> None:
                return None

        with mock.patch(
            "requests.get",
            return_value=_Response(),
        ):
            _download_file(
                item,
                destination,
                downloads_dir=downloads,
                completed=0,
                total=item.size,
                callback=lambda *event: events.append(event),
            )
        self.assertEqual(destination.read_bytes(), payload)
        self.assertFalse((downloads / "tiny.bin.part").exists())
        self.assertEqual(events[-1], ("Pix2TextMFR15", item.size, item.size))

    def test_unimernet_onnx_manifest_matches_catalog_size(self) -> None:
        self.assertEqual(UNIMERNET_ONNX_TOTAL_SIZE, 349_928_844)
        self.assertEqual(
            sum(item.size for item in UNIMERNET_ONNX_MODEL_FILES),
            MODEL_BY_ID["UniMERNetSmallONNX"].archive_size,
        )
        self.assertEqual(
            unimernet_onnx_downloader.UNIMERNET_ONNX_REVISION,
            "411ee76221baaad144ffbf996d4deef8df013b54",
        )
        self.assertTrue(
            UNIMERNET_ONNX_MODEL_FILES[0].url.endswith("/small/config.json")
        )
        for item in UNIMERNET_ONNX_MODEL_FILES:
            with self.subTest(name=item.name):
                self.assertEqual(len(item.sha256), 64)
                self.assertRegex(item.sha256, r"^[0-9a-f]{64}$")

    def test_unimernet_onnx_manifest_validates_local_smoke_payload(self) -> None:
        smoke_root = os.environ.get("FORMULA_OCR_UNIMERNET_SMOKE_DIR", "").strip()
        if not smoke_root:
            self.skipTest("set FORMULA_OCR_UNIMERNET_SMOKE_DIR for the real payload check")
        root = Path(smoke_root).expanduser()
        if not root.is_dir():
            self.skipTest("real UniMERNet Small smoke payload is not present")
        self.assertTrue(_unimernet_model_files_are_valid(root, verify_hash=True))

    def test_unimernet_onnx_download_installs_verified_file_atomically(self) -> None:
        root = Path.cwd() / "_unimernet_onnx_download_test"
        self.addCleanup(shutil.rmtree, root, True)
        downloads = root / "downloads"
        downloads.mkdir(parents=True)
        payload = b"verified-unimernet-file"
        item = UniMERNetONNXModelFile(
            "tiny.bin",
            len(payload),
            hashlib.sha256(payload).hexdigest(),
        )
        destination = root / "model" / item.name
        events: list[tuple[str, int, int]] = []

        class _Response:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int):
                self.chunk_size = chunk_size
                return [payload]

            def close(self) -> None:
                return None

        with mock.patch(
            "requests.get",
            return_value=_Response(),
        ):
            _download_unimernet_onnx_file(
                item,
                destination,
                downloads_dir=downloads,
                completed=0,
                total=item.size,
                callback=lambda *event: events.append(event),
            )
        self.assertEqual(destination.read_bytes(), payload)
        self.assertFalse((downloads / "tiny.bin.part").exists())
        self.assertEqual(events[-1], ("UniMERNetSmallONNX", item.size, item.size))

    def test_mixtex_archive_download_resumes_and_verifies(self) -> None:
        root = Path.cwd() / "_mixtex_archive_resume_test"
        self.addCleanup(shutil.rmtree, root, True)
        downloads = root / "downloads"
        downloads.mkdir(parents=True)
        payload = b"verified-mixtex-release-payload"
        split_at = 11
        archive = downloads / "MixTeX.zip"
        partial = downloads / "MixTeX.zip.part"
        partial.write_bytes(payload[:split_at])
        requested_headers: dict[str, str] = {}

        class _Response:
            status_code = 206
            headers = {"Content-Range": f"bytes {split_at}-{len(payload) - 1}/{len(payload)}"}

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int):
                self.chunk_size = chunk_size
                return [payload[split_at:]]

            def close(self) -> None:
                return None

        def fake_get(_url: str, **kwargs):
            requested_headers.update(kwargs["headers"])
            return _Response()

        with mock.patch.object(
            mixtex_downloader,
            "MIXTEX_ARCHIVE_SIZE",
            len(payload),
        ), mock.patch.object(
            mixtex_downloader,
            "MIXTEX_ARCHIVE_SHA256",
            hashlib.sha256(payload).hexdigest(),
        ), mock.patch(
            "requests.get",
            side_effect=fake_get,
        ):
            mixtex_downloader._download_archive(archive, None)

        self.assertEqual(requested_headers["Range"], f"bytes={split_at}-")
        self.assertEqual(archive.read_bytes(), payload)
        self.assertFalse(partial.exists())

    def test_paddle_hf_download_resumes_a_partial_file(self) -> None:
        root = Path.cwd() / "_paddle_hf_resume_test"
        self.addCleanup(shutil.rmtree, root, True)
        downloads = root / "downloads"
        downloads.mkdir(parents=True)
        payload = b"paddle-hf-resumable-payload"
        split_at = 9
        item = PaddleHFModelFile(
            "tiny.bin",
            len(payload),
            hashlib.sha256(payload).hexdigest(),
        )
        partial = downloads / "tiny.bin.part"
        partial.write_bytes(payload[:split_at])
        destination = root / "model" / item.name
        requested_headers: dict[str, str] = {}

        class _Response:
            status_code = 206
            headers = {"Content-Range": f"bytes {split_at}-{len(payload) - 1}/{len(payload)}"}

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int):
                self.chunk_size = chunk_size
                return [payload[split_at:]]

            def close(self) -> None:
                return None

        def fake_get(_url: str, **kwargs):
            requested_headers.update(kwargs["headers"])
            return _Response()

        with mock.patch(
            "requests.get",
            side_effect=fake_get,
        ):
            _download_paddle_hf_file(
                item,
                destination,
                downloads_dir=downloads,
                completed=0,
                callback=None,
            )

        self.assertEqual(requested_headers["Range"], f"bytes={split_at}-")
        self.assertEqual(destination.read_bytes(), payload)
        self.assertFalse(partial.exists())

    def test_paddle_hf_download_cancellation_keeps_partial_for_resume(self) -> None:
        root = Path.cwd() / "_paddle_hf_cancel_test"
        self.addCleanup(shutil.rmtree, root, True)
        downloads = root / "downloads"
        downloads.mkdir(parents=True)
        payload = b"cancel-and-resume-payload"
        split_at = 8
        item = PaddleHFModelFile(
            "tiny.bin",
            len(payload),
            hashlib.sha256(payload).hexdigest(),
        )
        partial = downloads / "tiny.bin.part"
        destination = root / "model" / item.name

        class _Response:
            def __init__(self, status_code: int, chunks: list[bytes]) -> None:
                self.status_code = status_code
                self.chunks = chunks
                self.closed = False
                self.headers = (
                    {
                        "Content-Range": (
                            f"bytes {split_at}-{len(payload) - 1}/{len(payload)}"
                        )
                    }
                    if status_code == 206
                    else {}
                )

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int):
                self.chunk_size = chunk_size
                return self.chunks

            def close(self) -> None:
                self.closed = True

        first_response = _Response(200, [payload[:split_at], payload[split_at:]])
        resumed_response = _Response(206, [payload[split_at:]])
        requested_headers: list[dict[str, str]] = []

        def fake_get(_url: str, **kwargs):
            requested_headers.append(kwargs["headers"])
            return first_response if len(requested_headers) == 1 else resumed_response

        def cancel_after_first_chunk(_name: str, downloaded: int, _total: int) -> None:
            if downloaded >= split_at:
                raise RuntimeError("simulated download cancellation")

        with mock.patch(
            "requests.get",
            side_effect=fake_get,
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated download cancellation"):
                _download_paddle_hf_file(
                    item,
                    destination,
                    downloads_dir=downloads,
                    completed=0,
                    callback=cancel_after_first_chunk,
                )

            self.assertEqual(partial.read_bytes(), payload[:split_at])
            self.assertFalse(destination.exists())

            _download_paddle_hf_file(
                item,
                destination,
                downloads_dir=downloads,
                completed=0,
                callback=None,
            )

        self.assertEqual(destination.read_bytes(), payload)
        self.assertFalse(partial.exists())
        self.assertEqual(requested_headers[1].get("Range"), f"bytes={split_at}-")
        self.assertTrue(first_response.closed)
        self.assertTrue(resumed_response.closed)


class MathCraftRecognizerTests(unittest.TestCase):
    def test_mathml_preview_wait_can_be_cancelled_immediately(self) -> None:
        cancel_event = threading.Event()
        cancel_event.set()

        with self.assertRaises(MathMLPreviewCancelled):
            wait_for_rendered_png(
                Path.cwd() / "_preview_that_must_not_exist.png",
                timeout=10.0,
                cancel_event=cancel_event,
            )

    def test_repeated_token_suffix_retains_one_period(self) -> None:
        token_ids = [9, 8, *([1, 2] * 8)]

        trim_start = repeated_token_suffix_start(
            token_ids,
            min_generated_tokens=16,
            min_repeated_tokens=12,
            max_period=2,
            min_repetitions=6,
        )

        self.assertEqual(trim_start, 4)
        self.assertEqual(token_ids[:trim_start], [9, 8, 1, 2])

    def test_repeated_token_suffix_ignores_normal_formula_tokens(self) -> None:
        self.assertIsNone(
            repeated_token_suffix_start(
                list(range(64)),
                min_generated_tokens=64,
                min_repeated_tokens=32,
                max_period=4,
                min_repetitions=8,
            )
        )

    def test_mock_onnx_sessions_decode_formula(self) -> None:
        import numpy as np

        class _Input:
            def __init__(self, name: str) -> None:
                self.name = name

        class _Encoder:
            def get_inputs(self):
                return [_Input("pixel_values")]

            def run(self, _outputs, inputs):
                self.inputs = inputs
                return [np.zeros((1, 2, 3), dtype=np.float32)]

        class _Decoder:
            def __init__(self) -> None:
                self.calls = 0

            def get_inputs(self):
                return [_Input("input_ids"), _Input("encoder_hidden_states")]

            def run(self, _outputs, inputs):
                self.calls += 1
                logits = np.full((1, inputs["input_ids"].shape[1], 5), -10.0)
                logits[0, -1, 3 if self.calls == 1 else 2] = 10.0
                return [logits]

        class _Tokenizer:
            def decode(self, ids, skip_special_tokens=True):
                self.last_ids = list(ids)
                return "x" if ids == [3] else ""

        recognizer = MathCraftFormulaRecognizer(max_new_tokens=8)
        encoder = _Encoder()
        decoder = _Decoder()
        tokenizer = _Tokenizer()
        recognizer._encoder_session = encoder
        recognizer._decoder_session = decoder
        recognizer._tokenizer = tokenizer
        recognizer._preprocess_image = lambda _path: np.zeros(  # type: ignore[method-assign]
            (1, 3, 384, 384), dtype=np.float32
        )
        recognizer._generation_ids = lambda: (2, 2)  # type: ignore[method-assign]

        self.assertEqual(recognizer.predict("unused.png"), "x")
        self.assertEqual(tokenizer.last_ids, [3])
        self.assertEqual(decoder.calls, 2)
        self.assertGreater(recognizer.last_score, 0.9)

    def test_pix2text_recognizer_uses_long_formula_limit(self) -> None:
        recognizer = Pix2TextFormulaRecognizer()
        self.addCleanup(recognizer.close)
        self.assertEqual(recognizer.max_new_tokens, 1024)


class MixTexRecognizerTests(unittest.TestCase):
    def test_merged_decoder_switches_cache_branch_and_trims_repeated_tail(
        self,
    ) -> None:
        import numpy as np

        class _Value:
            def __init__(self, name: str, shape=None) -> None:
                self.name = name
                self.shape = shape

        cache_names = ("past_key_values.0.key", "past_key_values.0.value")
        present_names = ("present.0.key", "present.0.value")

        class _Decoder:
            def __init__(self) -> None:
                self.calls = 0
                self.cache_branches: list[bool] = []

            def get_inputs(self):
                return [
                    _Value("input_ids"),
                    _Value("encoder_hidden_states"),
                    *(_Value(name, [1, 2, "past", 3]) for name in cache_names),
                    _Value("use_cache_branch"),
                ]

            def get_outputs(self):
                return [_Value("logits"), *(_Value(name) for name in present_names)]

            def run(self, _outputs, inputs):
                self.calls += 1
                self.cache_branches.append(bool(inputs["use_cache_branch"][0]))
                next_id = (3, 4)[self.calls - 1] if self.calls <= 2 else 7
                logits = np.full((1, 1, 8), -10.0, dtype=np.float32)
                logits[0, 0, next_id] = 10.0
                cache = np.ones((1, 2, self.calls, 3), dtype=np.float32)
                return [logits, cache, cache]

        decoder = _Decoder()
        generated = _generate_mixtex_tokens(
            decoder,
            np.zeros((1, 2, 3), dtype=np.float32),
            decoder_start_id=0,
            eos_id=None,
            max_new_tokens=50,
        )

        self.assertEqual(generated, [3, 4, 7])
        self.assertEqual(decoder.cache_branches[0], False)
        self.assertTrue(all(decoder.cache_branches[1:]))


class RapidRecognizerTests(unittest.TestCase):
    def test_decoder_trims_repeated_tail(self) -> None:
        import numpy as np

        class _Input:
            def __init__(self, name: str) -> None:
                self.name = name

        class _RepeatingDecoder:
            def __init__(self) -> None:
                self.calls = 0

            def get_inputs(self):
                return [_Input("tokens"), _Input("mask"), _Input("context")]

            def run(self, _outputs, inputs):
                self.calls += 1
                logits = np.full(
                    (1, inputs["tokens"].shape[1], 6),
                    -10.0,
                    dtype=np.float32,
                )
                logits[0, -1, 4] = 10.0
                return [logits]

        decoder = _RepeatingDecoder()
        generated = _decode_rapid_tokens(
            decoder,
            np.zeros((1, 2, 3), dtype=np.float32),
            bos_token=1,
            eos_token=2,
            max_new_tokens=200,
        )

        self.assertEqual(generated, [4])
        self.assertEqual(decoder.calls, 64)

    def test_direct_onnx_sessions_decode_formula_without_wrapper_package(self) -> None:
        import numpy as np

        class _Input:
            def __init__(self, name: str) -> None:
                self.name = name

        class _Resizer:
            def get_inputs(self):
                return [_Input("image")]

            def run(self, _outputs, _inputs):
                logits = np.full((1, 21), -10.0, dtype=np.float32)
                logits[0, 1] = 10.0
                return [logits]

        class _Encoder:
            def get_inputs(self):
                return [_Input("image")]

            def run(self, _outputs, _inputs):
                return [np.zeros((1, 2, 3), dtype=np.float32)]

        class _Decoder:
            def __init__(self) -> None:
                self.calls = 0

            def get_inputs(self):
                return [_Input("tokens"), _Input("mask"), _Input("context")]

            def run(self, _outputs, inputs):
                self.calls += 1
                logits = np.full(
                    (1, inputs["tokens"].shape[1], 6),
                    -10.0,
                    dtype=np.float32,
                )
                logits[0, -1, 4 if self.calls == 1 else 2] = 10.0
                return [logits]

        class _Tokenizer:
            def decode(self, ids, skip_special_tokens=False):
                self.ids = list(ids)
                return "x" if ids == [4, 2] else ""

        recognizer = RapidLatexRecognizer(max_new_tokens=8)
        recognizer._resizer_session = _Resizer()
        recognizer._encoder_session = _Encoder()
        recognizer._decoder_session = _Decoder()
        tokenizer = _Tokenizer()
        recognizer._tokenizer = tokenizer
        with mock.patch(
            "formula_ocr_app.rapid_recognizer._load_image",
            return_value=object(),
        ), mock.patch(
            "formula_ocr_app.rapid_recognizer._resize_for_model",
            return_value=np.zeros((1, 1, 32, 64), dtype=np.float32),
        ):
            result = recognizer.predict("unused.png")

        self.assertEqual(result, "x")
        self.assertEqual(tokenizer.ids, [4, 2])
        self.assertEqual(recognizer._decoder_session.calls, 2)

    def test_onnx_provider_policy_rejects_unavailable_cuda(self) -> None:
        runtime = mock.Mock()
        runtime.get_available_providers.return_value = ["CPUExecutionProvider"]

        with self.assertRaises(RapidLatexRuntimeError):
            execution_providers(
                runtime,
                "gpu",
                error_type=RapidLatexRuntimeError,
            )


class UniMERNetONNXRecognizerTests(unittest.TestCase):
    def test_cached_decoder_trims_repeated_tail(self) -> None:
        import numpy as np

        class _Value:
            def __init__(self, name: str) -> None:
                self.name = name

        class _FirstDecoder:
            def get_inputs(self):
                return [_Value("input_ids"), _Value("encoder_hidden_states")]

            def get_outputs(self):
                return [_Value("logits"), _Value("present.0.decoder.key")]

            def run(self, _outputs, _inputs):
                logits = np.full((1, 1, 8), -10.0, dtype=np.float32)
                logits[0, 0, 3] = 10.0
                cache = np.ones((1, 1, 1, 1), dtype=np.float32)
                return [logits, cache]

        class _PastDecoder:
            def __init__(self) -> None:
                self.calls = 0

            def get_inputs(self):
                return [
                    _Value("input_ids"),
                    _Value("past_key_values.0.decoder.key"),
                ]

            def get_outputs(self):
                return [_Value("logits"), _Value("present.0.decoder.key")]

            def run(self, _outputs, _inputs):
                self.calls += 1
                logits = np.full((1, 1, 8), -10.0, dtype=np.float32)
                logits[0, 0, 7] = 10.0
                cache = np.ones((1, 1, self.calls + 1, 1), dtype=np.float32)
                return [logits, cache]

        generated = _generate_unimernet_tokens(
            _FirstDecoder(),
            _PastDecoder(),
            np.zeros((1, 2, 3), dtype=np.float32),
            decoder_start_id=0,
            eos_id=None,
            max_new_tokens=70,
        )

        self.assertEqual(generated, [3, 7])

    def test_small_input_is_scaled_up_to_use_the_encoder_canvas(self) -> None:
        from PIL import Image

        resized = _resize_unimernet_to_fit(
            Image.new("RGB", (100, 50), "white"),
            672,
            192,
        )

        self.assertEqual(resized.size, (384, 192))

    def test_mock_first_and_past_decoders_keep_static_encoder_cache(self) -> None:
        import numpy as np

        class _Input:
            def __init__(self, name: str) -> None:
                self.name = name

        class _Output:
            def __init__(self, name: str) -> None:
                self.name = name

        cache_names = (
            "past_key_values.0.decoder.key",
            "past_key_values.0.decoder.value",
            "past_key_values.0.encoder.key",
            "past_key_values.0.encoder.value",
        )
        present_names = tuple(
            name.replace("past_key_values.", "present.", 1)
            for name in cache_names
        )

        class _Encoder:
            def get_inputs(self):
                return [_Input("pixel_values")]

            def run(self, _outputs, _inputs):
                return [np.zeros((1, 2, 3), dtype=np.float32)]

        class _FirstDecoder:
            def get_inputs(self):
                return [_Input("input_ids"), _Input("encoder_hidden_states")]

            def get_outputs(self):
                return [_Output("logits"), *(_Output(name) for name in present_names)]

            def run(self, _outputs, inputs):
                self.inputs = inputs
                logits = np.full((1, 1, 8), -10.0, dtype=np.float32)
                logits[0, 0, 3] = 10.0
                return [logits, *[np.ones((1, 1, 1, 1), dtype=np.float32) for _ in present_names]]

        class _PastDecoder:
            def __init__(self) -> None:
                self.calls = 0

            def get_inputs(self):
                return [_Input("input_ids"), *(_Input(name) for name in cache_names)]

            def get_outputs(self):
                return [
                    _Output("logits"),
                    _Output("present.0.decoder.key"),
                    _Output("present.0.decoder.value"),
                ]

            def run(self, _outputs, inputs):
                self.calls += 1
                self.inputs = inputs
                logits = np.full((1, 1, 8), -10.0, dtype=np.float32)
                logits[0, 0, 2] = 10.0
                return [
                    logits,
                    np.ones((1, 1, 2, 1), dtype=np.float32),
                    np.ones((1, 1, 2, 1), dtype=np.float32),
                ]

        class _Tokenizer:
            def decode(self, ids, skip_special_tokens=True):
                self.last_ids = list(ids)
                return "x" if ids == [3] else ""

        recognizer = UniMERNetSmallFormulaRecognizer(max_new_tokens=8)
        encoder = _Encoder()
        first_decoder = _FirstDecoder()
        past_decoder = _PastDecoder()
        tokenizer = _Tokenizer()
        recognizer._encoder_session = encoder
        recognizer._decoder_session = first_decoder
        recognizer._decoder_with_past_session = past_decoder
        recognizer._tokenizer = tokenizer
        recognizer._preprocess_image = lambda _path: np.zeros(  # type: ignore[method-assign]
            (1, 3, 192, 672), dtype=np.float32
        )
        recognizer._generation_ids = lambda: (0, 2)  # type: ignore[method-assign]

        self.assertEqual(recognizer.predict("unused.png"), "x")
        self.assertEqual(tokenizer.last_ids, [3])
        self.assertEqual(past_decoder.calls, 1)
        self.assertIn("past_key_values.0.encoder.key", past_decoder.inputs)
        self.assertIn("past_key_values.0.encoder.value", past_decoder.inputs)


class ModelCatalogTests(unittest.TestCase):
    def test_catalog_contains_multiple_model_suppliers(self) -> None:
        self.assertGreaterEqual(len(MODEL_SPECS), 10)
        self.assertGreaterEqual(len({spec.provider for spec in MODEL_SPECS}), 6)

    def test_every_model_has_verified_archive_metadata(self) -> None:
        for spec in MODEL_SPECS:
            with self.subTest(model=spec.model_id):
                self.assertGreater(spec.archive_size, 1_000_000)
                if spec.backend == "paddle":
                    self.assertGreater(spec.archive_crc32, 0)
                elif spec.backend == "paddle_hf":
                    self.assertIsNone(spec.archive_crc32)
                self.assertTrue(spec.download_url.startswith("https://"))

    def test_quick_filters_cover_recommended_cached_language_and_backend(self) -> None:
        filter_keys = {key for key, _label in MODEL_QUICK_FILTERS}
        self.assertEqual(
            filter_keys,
            {"all", "recommended", "downloaded", "lightweight", "chinese", "onnx"},
        )
        default = MODEL_BY_ID["PP-FormulaNet_plus-S"]
        pix2text = MODEL_BY_ID["Pix2TextMFR15"]
        self.assertTrue(model_matches_quick_filter(default, "recommended"))
        self.assertTrue(model_matches_quick_filter(default, "chinese"))
        self.assertTrue(model_matches_quick_filter(pix2text, "onnx"))
        self.assertTrue(model_matches_quick_filter(pix2text, "lightweight"))
        self.assertFalse(model_matches_quick_filter(pix2text, "downloaded"))
        self.assertTrue(
            model_matches_quick_filter(pix2text, "downloaded", cached=True)
        )
        with self.assertRaises(ValueError):
            model_matches_quick_filter(default, "unknown")

    def test_model_search_includes_backend_and_usage_metadata(self) -> None:
        self.assertTrue(
            model_matches_query(MODEL_BY_ID["UniMERNetSmallONNX"], "onnx runtime")
        )
        self.assertTrue(
            model_matches_query(MODEL_BY_ID["MixTexZhEn"], "商业用途")
        )
        self.assertFalse(model_matches_query(MODEL_BY_ID["Pix2TextMFR15"], "百度飞桨"))

    def test_latex_ocr_paddle_manifest_is_pinned_and_verified(self) -> None:
        spec = MODEL_BY_ID["LaTeX_OCR_rec"]
        self.assertEqual(spec.backend, "paddle_hf")
        self.assertEqual(spec.archive_size, 103_735_029)
        self.assertEqual(
            sum(item.size for item in PADDLE_HF_MODEL_FILES),
            spec.archive_size,
        )
        self.assertEqual(
            paddle_hf_downloader.PADDLE_HF_REVISION,
            "563fb029dfdf5fc847d0677f3870039960e3a801",
        )
        for item in PADDLE_HF_MODEL_FILES:
            self.assertEqual(len(item.sha256), 64)

    def test_rapid_onnx_manifest_matches_catalog_size(self) -> None:
        self.assertEqual(
            sum(item.size for item in RAPID_MODEL_FILES),
            MODEL_BY_ID["RapidLaTeXOCR"].archive_size,
        )
        self.assertEqual(
            MODEL_BY_ID["RapidLaTeXOCR"].download_url,
            "https://github.com/RapidAI/RapidLaTeXOCR/releases/tag/v0.0.0",
        )
        for item in RAPID_MODEL_FILES:
            self.assertEqual(len(item.sha256), 64)

    def test_mathcraft_catalog_entry_is_verified(self) -> None:
        spec = MODEL_BY_ID["MathCraftFormula"]
        self.assertEqual(spec.backend, "mathcraft_onnx")
        self.assertEqual(spec.provider, "SakuraMathcraft")
        self.assertEqual(spec.archive_size, 108_795_631)
        self.assertEqual(
            spec.download_url,
            "https://github.com/SakuraMathcraft/MathCraft-Models/releases/"
            "download/v1.0.0/mathcraft-formula-rec.zip",
        )
        self.assertEqual(
            spec.archive_size,
            108_795_631,
        )
        self.assertEqual(spec.license_label, "MIT")
        self.assertIn("MathCraft-Models/blob/main/LICENSE", spec.terms_url)
        self.assertFalse(spec.requires_terms_ack)

    def test_pix2text_catalog_entry_is_verified(self) -> None:
        spec = MODEL_BY_ID["Pix2TextMFR15"]
        self.assertEqual(spec.backend, "pix2text_onnx")
        self.assertEqual(spec.provider, "Breezedeus / Pix2Text")
        self.assertEqual(spec.archive_size, 119_654_633)
        self.assertEqual(
            spec.download_url,
            "https://huggingface.co/breezedeus/pix2text-mfr-1.5",
        )

    def test_mixtex_catalog_entry_is_verified(self) -> None:
        spec = MODEL_BY_ID["MixTexZhEn"]
        self.assertEqual(spec.backend, "mixtex_onnx")
        self.assertEqual(spec.provider, "MixTeX / RQLuo")
        self.assertEqual(spec.archive_size, MIXTEX_ARCHIVE_SIZE)
        self.assertEqual(spec.archive_size, 294_025_378)
        self.assertEqual(
            spec.download_url,
            "https://github.com/RQLuo/MixTeX-Latex-OCR/releases/"
            "tag/MixTeX-v3.2.4",
        )
        self.assertTrue(spec.requires_terms_ack)
        self.assertIn("商业用途", spec.usage_restriction)
        self.assertEqual(len(MIXTEX_ARCHIVE_SHA256), 64)
        self.assertEqual(
            sum(size for size, _digest in MIXTEX_MODEL_FILES.values()),
            411_393_679,
        )

    def test_unimernet_small_onnx_catalog_entry_is_verified(self) -> None:
        spec = MODEL_BY_ID["UniMERNetSmallONNX"]
        self.assertEqual(spec.backend, "unimernet_onnx")
        self.assertEqual(spec.provider, "OpenDataLab / Cooper114")
        self.assertEqual(spec.archive_size, UNIMERNET_ONNX_TOTAL_SIZE)
        self.assertIn("Cooper114/unimernet-onnx", spec.download_url)

    def test_mixtex_zip_path_validation_rejects_traversal(self) -> None:
        member = zipfile.ZipInfo("onnx/../escape.onnx")
        with self.assertRaises(MixTexModelDownloadError):
            _validate_mixtex_zip_members([member])

    def test_mixtex_zip_path_validation_rejects_windows_aliases(self) -> None:
        for name in ("onnx/model.onnx:stream", "onnx/COM1", "onnx/name. "):
            with self.subTest(name=name):
                with self.assertRaises(MixTexModelDownloadError):
                    _validate_mixtex_zip_members([zipfile.ZipInfo(name)])


class SettingsAndFormatTests(unittest.TestCase):
    def test_settings_round_trip_and_invalid_model_recovery(self) -> None:
        data_dir = Path.cwd() / "_settings_roundtrip_data"
        self.addCleanup(shutil.rmtree, data_dir, True)
        with mock.patch.dict(os.environ, {"FORMULA_OCR_DATA_DIR": str(data_dir)}):
            save_settings(
                AppSettings(
                    model_id="UniMERNet",
                    accepted_model_terms=("MixTexZhEn:MixTeX-v3.2.4",),
                )
            )
            self.assertEqual(
                load_settings(),
                AppSettings(
                    model_id="UniMERNet",
                    accepted_model_terms=("MixTexZhEn:MixTeX-v3.2.4",),
                ),
            )

    def test_non_object_settings_json_falls_back_to_defaults(self) -> None:
        data_dir = Path.cwd() / "_settings_non_object_data"
        self.addCleanup(shutil.rmtree, data_dir, True)
        with mock.patch.dict(os.environ, {"FORMULA_OCR_DATA_DIR": str(data_dir)}):
            settings_file = data_dir / "settings.json"
            data_dir.mkdir(parents=True)
            for payload in ("[]", "null", '"text"', "42"):
                with self.subTest(payload=payload):
                    settings_file.write_text(payload, encoding="utf-8")
                    self.assertEqual(load_settings(), AppSettings())

    def test_concurrent_settings_saves_remain_valid_and_leave_no_temp_files(self) -> None:
        data_dir = Path.cwd() / "_settings_concurrent_data"
        self.addCleanup(shutil.rmtree, data_dir, True)
        errors: list[BaseException] = []
        model_ids = tuple(MODEL_BY_ID)[:8]

        def writer(index: int) -> None:
            try:
                save_settings(
                    AppSettings(
                        model_id=model_ids[index % len(model_ids)],
                        accepted_model_terms=(f"terms-{index}",),
                    )
                )
            except BaseException as exc:  # Preserve all worker failures for assertion.
                errors.append(exc)

        with mock.patch.dict(os.environ, {"FORMULA_OCR_DATA_DIR": str(data_dir)}):
            threads = [threading.Thread(target=writer, args=(index,)) for index in range(24)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)
            self.assertFalse(any(thread.is_alive() for thread in threads))
            self.assertEqual(errors, [])
            loaded = load_settings()
            self.assertIn(loaded.model_id, model_ids)
            self.assertEqual(len(loaded.accepted_model_terms), 1)
            self.assertEqual(list(data_dir.glob(".settings.json.*.tmp")), [])

    def test_additional_export_wrappers(self) -> None:
        latex = r"x^2+y^2=z^2"
        self.assertEqual(latex_to_markdown_inline(latex), f"${latex}$")
        self.assertEqual(latex_to_markdown_block(latex), f"$$\n{latex}\n$$")
        self.assertIn(r"\begin{equation}", latex_to_equation_environment(latex))
        self.assertIn("<math", latex_to_html(latex))

    def test_docx_export_is_valid_and_replaces_target_atomically(self) -> None:
        root = Path.cwd() / "_formula_ocr_docx_export"
        self.addCleanup(shutil.rmtree, root, True)
        root.mkdir()
        output = root / "formula.docx"
        latex = r"x^2+y^2=z^2"
        mathml = latex_to_html(latex)

        export_formula_docx(
            output,
            latex=latex,
            mathml=mathml,
            asciimath="x^2+y^2=z^2",
            typst="x^2 + y^2 = z^2",
        )

        with zipfile.ZipFile(output) as document:
            self.assertIn("word/document.xml", document.namelist())
            self.assertIn(latex.encode("utf-8"), document.read("word/document.xml"))
        self.assertEqual(list(root.glob(".formula.docx.*.tmp")), [])

        original = output.read_bytes()
        with mock.patch(
            "formula_ocr_app.formula_formats.zipfile.ZipFile",
            side_effect=OSError("disk full"),
        ):
            with self.assertRaisesRegex(OSError, "disk full"):
                export_formula_docx(
                    output,
                    latex=latex,
                    mathml=mathml,
                    asciimath="x",
                    typst="x",
                )
        self.assertEqual(output.read_bytes(), original)
        self.assertEqual(list(root.glob(".formula.docx.*.tmp")), [])


def main() -> None:
    unittest.main(module=__name__)


if __name__ == "__main__":
    main()
