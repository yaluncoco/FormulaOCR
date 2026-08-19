from __future__ import annotations

import os
import hashlib
import shutil
import tarfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from formula_ocr_app.app_settings import AppSettings, load_settings, save_settings
from formula_ocr_app.formula_formats import (
    clean_recognized_latex,
    latex_to_equation_environment,
    latex_to_html,
    latex_to_markdown_block,
    latex_to_markdown_inline,
)
from formula_ocr_app.model_catalog import (
    MODEL_BY_ID,
    MODEL_QUICK_FILTERS,
    MODEL_SPECS,
    model_matches_query,
    model_matches_quick_filter,
)
from formula_ocr_app.mathcraft_model_downloader import (
    MATHCRAFT_ARCHIVE_SHA256,
    MATHCRAFT_MODEL_FILES,
    MathCraftModelDownloadError,
    _file_is_valid,
    _install_archive,
    _validate_zip_members,
)
from formula_ocr_app.mathcraft_recognizer import MathCraftFormulaRecognizer
from formula_ocr_app.pix2text_recognizer import Pix2TextFormulaRecognizer
from formula_ocr_app.unimernet_onnx_recognizer import (
    UniMERNetSmallFormulaRecognizer,
)
from formula_ocr_app.mixtex_model_downloader import (
    MIXTEX_ARCHIVE_SHA256,
    MIXTEX_ARCHIVE_SIZE,
    MIXTEX_MODEL_FILES,
    MixTexModelDownloadError,
    _validate_zip_members as _validate_mixtex_zip_members,
)
import formula_ocr_app.paddle_hf_model_downloader as paddle_hf_downloader
import formula_ocr_app.pix2text_model_downloader as pix2text_downloader
import formula_ocr_app.mixtex_model_downloader as mixtex_downloader
import formula_ocr_app.unimernet_onnx_model_downloader as unimernet_onnx_downloader
from formula_ocr_app.model_downloader import (
    ModelDownloadError,
    _validate_tar_members,
)
from formula_ocr_app.recognition_pipeline import FormulaRecognizer
from formula_ocr_app.recognizer import PaddleFormulaRecognizer
from formula_ocr_app.runtime_paths import (
    bundled_external_model_dir,
    external_model_dir,
    external_model_has_data,
    is_external_model_bundled,
    is_paddle_model_cached,
    paddle_model_has_data,
    paddle_model_dir,
    remove_paddle_model,
    remove_external_model,
    runtime_cache_dir,
    runtime_log_dir,
)
from formula_ocr_app.rapid_model_downloader import RAPID_MODEL_FILES
from formula_ocr_app.paddle_hf_model_downloader import (
    PADDLE_HF_MODEL_FILES,
    PaddleHFModelFile,
    _download_file as _download_paddle_hf_file,
)
from formula_ocr_app.pix2text_model_downloader import (
    PIX2TEXT_MODEL_FILES,
    Pix2TextModelFile,
    _download_file,
)
from formula_ocr_app.unimernet_onnx_model_downloader import (
    UNIMERNET_ONNX_MODEL_FILES,
    UNIMERNET_ONNX_TOTAL_SIZE,
    UniMERNetONNXModelFile,
    _model_files_are_valid as _unimernet_model_files_are_valid,
    _download_file as _download_unimernet_onnx_file,
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
    def test_spaced_named_operator_is_canonicalized(self) -> None:
        self.assertEqual(clean_recognized_latex(M_ARGMAX_OUTPUT), EXPECTED_ARGMAX)

    def test_argmax_subscript_is_made_an_explicit_display_limit(self) -> None:
        self.assertEqual(
            clean_recognized_latex(r"k^{*}=\arg\max_{k}J(k)"),
            EXPECTED_ARGMAX,
        )

    def test_existing_limits_are_not_duplicated(self) -> None:
        self.assertEqual(clean_recognized_latex(EXPECTED_ARGMAX), EXPECTED_ARGMAX)


class SelectedModelRecognizerTests(unittest.TestCase):
    def _pipeline(
        self,
        outputs: dict[str, str | Exception],
        calls: list[str],
        *,
        model_name: str = "PP-FormulaNet_plus-S",
    ) -> FormulaRecognizer:
        return FormulaRecognizer(
            paddleocr_repo=Path.cwd(),
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
                paddleocr_repo=Path.cwd(),
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

    def test_frozen_build_allows_missing_model_to_download(self) -> None:
        data_dir = Path.cwd() / "_nonexistent_formula_ocr_frozen_data"
        recognizer = PaddleFormulaRecognizer(
            paddleocr_repo=Path.cwd(),
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

    def test_missing_model_is_downloaded_before_paddlex_initialization(self) -> None:
        captured_kwargs: dict[str, object] = {}

        class _FakeFormulaRecognition:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.update(kwargs)

        recognizer = PaddleFormulaRecognizer(
            paddleocr_repo=Path.cwd(),
            model_name="PP-FormulaNet_plus-S",
        )
        recognizer._configure_runtime_cache = lambda: None  # type: ignore[method-assign]
        recognizer._patch_subprocess_no_window = lambda: None  # type: ignore[method-assign]
        recognizer._install_optional_download_stubs = lambda: None  # type: ignore[method-assign]
        recognizer._load_formula_recognition_class = (  # type: ignore[method-assign]
            lambda: _FakeFormulaRecognition
        )
        recognizer._validate_device = lambda: None  # type: ignore[method-assign]

        data_dir = Path.cwd() / "_nonexistent_formula_ocr_download_data"
        downloaded_model = data_dir / "downloaded-model"
        with mock.patch.dict(
            os.environ,
            {"FORMULA_OCR_DATA_DIR": str(data_dir)},
        ), mock.patch("sys.frozen", True, create=True), mock.patch(
            "formula_ocr_app.recognizer.ensure_official_model",
            return_value=downloaded_model,
        ) as download:
            recognizer._ensure_model()

        download.assert_called_once_with(
            "PP-FormulaNet_plus-S",
            progress_callback=None,
        )
        self.assertEqual(captured_kwargs["model_name"], "PP-FormulaNet_plus-S")
        self.assertEqual(captured_kwargs["model_dir"], str(downloaded_model))


class ModelDownloaderTests(unittest.TestCase):
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
            "formula_ocr_app.pix2text_model_downloader.requests.get",
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
            "formula_ocr_app.unimernet_onnx_model_downloader.requests.get",
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
        ), mock.patch.object(
            mixtex_downloader.requests,
            "get",
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
            "formula_ocr_app.paddle_hf_model_downloader.requests.get",
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
            "formula_ocr_app.paddle_hf_model_downloader.requests.get",
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


class UniMERNetONNXRecognizerTests(unittest.TestCase):
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

    def test_additional_export_wrappers(self) -> None:
        latex = r"x^2+y^2=z^2"
        self.assertEqual(latex_to_markdown_inline(latex), f"${latex}$")
        self.assertEqual(latex_to_markdown_block(latex), f"$$\n{latex}\n$$")
        self.assertIn(r"\begin{equation}", latex_to_equation_environment(latex))
        self.assertIn("<math", latex_to_html(latex))


def main() -> None:
    unittest.main(module=__name__)


if __name__ == "__main__":
    main()
