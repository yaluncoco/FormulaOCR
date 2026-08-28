"""Small Paddle Inference backend for formula-recognition models.

The application only needs a narrow slice of PaddleOCR/PaddleX at runtime:
image preprocessing, ``paddle.inference`` and tokenizer based decoding. This
module implements that slice directly so packaged builds do not need the full
PaddleOCR/PaddleX/OpenCV platform.

The preprocessing and decoding behavior is compatible with PaddleX's formula
recognition processors. Those processors are distributed under Apache-2.0;
see NOTICE.md for attribution.
"""

from __future__ import annotations

import json
import importlib.util
import math
import os
import re
import sys
import threading
import types
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

try:
    from formula_ocr_app.formula_formats import clean_recognized_latex
    from formula_ocr_app.image_utils import load_rgb_image
    from formula_ocr_app.model_api import (
        DownloadProgressCallback,
        ModelDownloadError,
    )
    from formula_ocr_app.model_catalog import DEFAULT_MODEL_ID
    from formula_ocr_app.runtime_paths import (
        is_paddle_model_cached,
        paddle_runtime_cache_dir,
        resolve_paddle_model_dir,
    )
except ModuleNotFoundError as exc:  # Allows ``python formula_ocr_app/app.py``.
    if exc.name != "formula_ocr_app":
        raise
    from formula_formats import clean_recognized_latex
    from image_utils import load_rgb_image
    from model_api import DownloadProgressCallback, ModelDownloadError
    from model_catalog import DEFAULT_MODEL_ID
    from runtime_paths import (
        is_paddle_model_cached,
        paddle_runtime_cache_dir,
        resolve_paddle_model_dir,
    )


class PaddleFormulaRuntimeError(RuntimeError):
    """Raised when a Paddle formula model cannot be prepared or executed."""


# Compatibility name retained for callers built against FormulaOCR 1.0.
PaddleOCRNotReadyError = PaddleFormulaRuntimeError


_PADDLE_NATIVE: Any | None = None
_PADDLE_NATIVE_LOCK = threading.Lock()
_PADDLE_DLL_HANDLES: list[Any] = []


class _FormulaDecoder(Protocol):
    def decode(self, token_ids: Any) -> str: ...


@dataclass(frozen=True)
class _PreprocessSpec:
    family: str
    input_size: tuple[int, int] | None = None
    min_dimensions: tuple[int, int] = (32, 32)
    max_dimensions: tuple[int, int] = (672, 192)
    mean: float = 0.7931
    std: float = 0.1738
    output_divisor: int = 16


class PaddleFormulaRecognizer:
    """Run Paddle formula-recognition exports without PaddleOCR or PaddleX."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_ID,
        model_dir: str | Path | None = None,
        device: str = "cpu",
        download_progress_callback: DownloadProgressCallback | None = None,
        model_ensure: Callable[..., Path] | None = None,
    ) -> None:
        self.model_name = model_name.strip() or DEFAULT_MODEL_ID
        self.model_dir = Path(model_dir).expanduser().resolve() if model_dir else None
        self.device = device.strip() if device else "cpu"
        self.download_progress_callback = download_progress_callback
        self.model_ensure = model_ensure
        self._predictor: Any | None = None
        self._preprocess_spec: _PreprocessSpec | None = None
        self._decoder: _FormulaDecoder | None = None

    def close(self) -> None:
        # Paddle's Python predictor has no public close method. Dropping all
        # references releases its native resources on model switch/shutdown.
        self._predictor = None
        self._preprocess_spec = None
        self._decoder = None

    def predict(self, image_path: str | Path) -> str:
        self._ensure_model()
        assert self._predictor is not None
        assert self._preprocess_spec is not None
        assert self._decoder is not None

        tensor = _preprocess_image(Path(image_path).resolve(), self._preprocess_spec)
        try:
            input_names = self._predictor.get_input_names()
            if len(input_names) != 1:
                raise PaddleOCRNotReadyError(
                    f"公式模型输入数量异常：期望 1 个，实际 {len(input_names)} 个。"
                )
            input_handle = self._predictor.get_input_handle(input_names[0])
            input_handle.reshape(tensor.shape)
            copy_from_cpu = getattr(input_handle, "copy_from_cpu", None)
            if copy_from_cpu is not None:
                copy_from_cpu(tensor)
            else:
                input_handle._copy_from_cpu_bind(tensor)
            self._predictor.run()

            output_names = self._predictor.get_output_names()
            if not output_names:
                raise PaddleOCRNotReadyError("公式模型没有可读取的输出。")
            token_ids = self._predictor.get_output_handle(output_names[0]).copy_to_cpu()
        except PaddleOCRNotReadyError:
            raise
        except Exception as exc:
            raise PaddleOCRNotReadyError(
                "Paddle 公式模型推理失败。请检查图片、模型文件和计算设备。"
            ) from exc

        formula = self._decoder.decode(token_ids).strip()
        if not formula:
            raise PaddleOCRNotReadyError("Paddle 公式模型未返回可用公式。")
        return formula

    def _ensure_model(self) -> None:
        if self._predictor is not None:
            return

        self._configure_runtime_cache()
        model_dir = self._resolve_or_download_model()
        self._initialize_runtime(model_dir)

    def _resolve_or_download_model(self) -> Path:
        if self.model_dir is not None:
            return self.model_dir

        if self.model_ensure is not None:
            ensure = self.model_ensure
        elif is_paddle_model_cached(self.model_name):
            return resolve_paddle_model_dir(self.model_name)
        else:
            try:
                from formula_ocr_app.model_downloader import ensure_official_model
            except ImportError:  # Allows ``python formula_ocr_app/app.py``.
                from model_downloader import ensure_official_model

            ensure = ensure_official_model

        try:
            return Path(
                ensure(
                    self.model_name,
                    progress_callback=self.download_progress_callback,
                )
            ).resolve()
        except ModelDownloadError as exc:
            raise PaddleOCRNotReadyError(
                "公式模型下载或校验失败。请检查网络连接、磁盘空间，"
                "然后重新识别；未完成的下载会在下次继续。"
            ) from exc

    def _initialize_runtime(self, model_dir: Path) -> None:
        required = {
            "inference.json": model_dir / "inference.json",
            "inference.yml": model_dir / "inference.yml",
            "inference.pdiparams": model_dir / "inference.pdiparams",
        }
        missing = [name for name, path in required.items() if not path.is_file()]
        if missing:
            raise PaddleOCRNotReadyError(
                f"公式模型文件不完整（缺少 {', '.join(missing)}）：{model_dir}"
            )

        preprocess_spec, decoder = _load_model_configuration(required["inference.yml"])
        predictor = self._create_predictor(
            required["inference.json"], required["inference.pdiparams"]
        )
        # Publish state only after every initialization step has succeeded.
        self._preprocess_spec = preprocess_spec
        self._decoder = decoder
        self._predictor = predictor

    def _create_predictor(self, model_file: Path, params_file: Path) -> Any:
        try:
            paddle_native = _load_paddle_native_runtime()
        except Exception as exc:  # pragma: no cover - depends on local runtime
            raise PaddleOCRNotReadyError(
                "程序缺少 Paddle Inference 运行组件，请重新安装或重新打包。"
            ) from exc

        try:
            config = paddle_native.AnalysisConfig(str(model_file), str(params_file))
            device = self.device.lower()
            if device.startswith("gpu"):
                device_id = _device_index(device)
                cuda_check = getattr(paddle_native, "is_compiled_with_cuda", None)
                cuda_ready = bool(cuda_check and cuda_check())
                if not cuda_ready:
                    raise PaddleOCRNotReadyError(
                        "当前 Paddle 运行时不支持 CUDA，请选择 CPU 或安装 GPU 版。"
                    )
                config.enable_use_gpu(512, device_id)
            else:
                config.disable_gpu()
                config.set_cpu_math_library_num_threads(self._cpu_threads())
                # The direct Windows predictor has crashed natively with
                # MKLDNN enabled. Keep the verified stable path by default.
                config.disable_mkldnn()

            config.disable_glog_info()
            # ``enable_memory_optim`` is not compatible with every Paddle 3.x
            # Windows inference graph (it can fail before predictor creation
            # with a missing ``memory_optimize_pass`` registration). The
            # inference graph already owns its tensors, so this optional pass
            # is not worth making startup fragile.
            return paddle_native.create_predictor(config)
        except PaddleOCRNotReadyError:
            raise
        except Exception as exc:  # pragma: no cover - native runtime failure
            raise PaddleOCRNotReadyError(
                "Paddle 公式模型初始化失败。请检查模型文件和运行时版本。"
            ) from exc

    def _configure_runtime_cache(self) -> None:
        cache_root = paddle_runtime_cache_dir()
        cache_root.mkdir(parents=True, exist_ok=True)
        # Direct inference does not need PaddleX's model hub and therefore
        # must not globally replace HOME or USERPROFILE.
        os.environ.setdefault("PADDLE_HOME", str(cache_root / "paddle"))
        os.environ.setdefault(
            "PADDLE_EXTENSION_DIR", str(cache_root / "paddle_extension")
        )

    def _cached_model_dir(self) -> Path | None:
        if is_paddle_model_cached(self.model_name):
            return resolve_paddle_model_dir(self.model_name)
        return None

    @staticmethod
    def _cpu_threads() -> int:
        raw_value = os.environ.get("FORMULA_OCR_CPU_THREADS", "").strip()
        if raw_value.isdigit():
            return max(1, int(raw_value))
        return max(2, min(os.cpu_count() or 4, 10))


def _load_paddle_native_runtime() -> Any:
    """Load only Paddle's native inference extension, not ``paddle`` itself.

    Importing ``paddle.inference`` first executes the broad top-level Paddle
    package and pulls training, distributed, data and visualization helpers
    into PyInstaller's graph. The inference API used here is exported directly
    by ``libpaddle``; loading that extension behind two tiny namespace stubs
    keeps both startup and the packaged dependency graph focused.
    """

    global _PADDLE_NATIVE
    if _PADDLE_NATIVE is not None:
        return _PADDLE_NATIVE
    with _PADDLE_NATIVE_LOCK:
        if _PADDLE_NATIVE is not None:
            return _PADDLE_NATIVE
        existing = sys.modules.get("paddle.base.libpaddle")
        if existing is not None:
            _PADDLE_NATIVE = existing
            return existing

        extension_path, paddle_root = _find_libpaddle_extension()
        libs_dir = paddle_root / "libs"
        _configure_paddle_library_path(libs_dir)
        _install_paddle_namespace_stubs(paddle_root)

        spec = importlib.util.spec_from_file_location(
            "paddle.base.libpaddle",
            extension_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"无法创建 Paddle 原生扩展加载器：{extension_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(spec.name, None)
            raise
        base_package = sys.modules.get("paddle.base")
        if base_package is not None:
            setattr(base_package, "libpaddle", module)
        _PADDLE_NATIVE = module
        return module


def _find_libpaddle_extension() -> tuple[Path, Path]:
    roots: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(Path(meipass) / "paddle")
    if getattr(sys, "frozen", False):
        executable_root = Path(sys.executable).resolve().parent
        roots.extend((executable_root / "_internal" / "paddle", executable_root / "paddle"))
    roots.extend(Path(entry) / "paddle" for entry in sys.path if entry)

    seen: set[Path] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        base_dir = resolved / "base"
        candidates = sorted(
            path
            for path in base_dir.glob("libpaddle*")
            if path.is_file() and path.suffix.lower() in {".pyd", ".so", ".dylib"}
        )
        if candidates:
            return candidates[0], resolved
    raise ImportError("未找到 paddle/base/libpaddle 原生推理扩展。")


def _configure_paddle_library_path(libs_dir: Path) -> None:
    if not libs_dir.is_dir():
        return
    libs_text = str(libs_dir)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if libs_text not in path_parts:
        os.environ["PATH"] = libs_text + os.pathsep + os.environ.get("PATH", "")
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None:
        try:
            _PADDLE_DLL_HANDLES.append(add_dll_directory(libs_text))
        except OSError:
            pass


def _install_paddle_namespace_stubs(paddle_root: Path) -> None:
    paddle_package = sys.modules.get("paddle")
    if paddle_package is None:
        paddle_package = types.ModuleType("paddle")
        paddle_package.__file__ = str(paddle_root / "__init__.py")
        paddle_package.__path__ = [str(paddle_root)]  # type: ignore[attr-defined]
        paddle_package.__package__ = "paddle"
        sys.modules["paddle"] = paddle_package

    base_package = sys.modules.get("paddle.base")
    if base_package is None:
        base_package = types.ModuleType("paddle.base")
        base_package.__file__ = str(paddle_root / "base" / "__init__.py")
        base_package.__path__ = [str(paddle_root / "base")]  # type: ignore[attr-defined]
        base_package.__package__ = "paddle.base"
        sys.modules["paddle.base"] = base_package
    setattr(paddle_package, "base", base_package)


def _device_index(device: str) -> int:
    _prefix, separator, raw_index = device.partition(":")
    if separator and raw_index.isdigit():
        return int(raw_index)
    return 0


def _load_model_configuration(config_path: Path) -> tuple[_PreprocessSpec, _FormulaDecoder]:
    try:
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PaddleOCRNotReadyError(f"无法读取公式模型配置：{config_path}") from exc
    if not isinstance(config, dict):
        raise PaddleOCRNotReadyError(f"公式模型配置格式无效：{config_path}")

    transforms: dict[str, dict[str, Any]] = {}
    try:
        transform_ops = config["PreProcess"]["transform_ops"]
        for entry in transform_ops:
            name, arguments = next(iter(entry.items()))
            transforms[name] = arguments or {}
        postprocess = config["PostProcess"]
        postprocess_name = postprocess["name"]
        character_dict = postprocess["character_dict"]
    except (KeyError, TypeError, ValueError, StopIteration) as exc:
        raise PaddleOCRNotReadyError(
            f"公式模型配置缺少预处理或解码信息：{config_path}"
        ) from exc

    if postprocess_name == "UniMERNetDecode":
        input_size = transforms.get("UniMERNetImgDecode", {}).get("input_size")
        if not (
            isinstance(input_size, (list, tuple))
            and len(input_size) == 2
            and all(isinstance(value, int) and value > 0 for value in input_size)
        ):
            raise PaddleOCRNotReadyError(
                f"公式模型配置中的 input_size 无效：{config_path}"
            )
        divisor = 32 if "UniMERNetImageFormat" in transforms else 16
        return (
            _PreprocessSpec(
                family="unimernet",
                input_size=(int(input_size[0]), int(input_size[1])),
                output_divisor=divisor,
            ),
            _UniMERNetDecoder(character_dict),
        )

    if postprocess_name == "LaTeXOCRDecode":
        resize = transforms.get("MinMaxResize", {})
        normalize = transforms.get("NormalizeImage", {})
        return (
            _PreprocessSpec(
                family="latex_ocr",
                min_dimensions=_dimension_pair(
                    resize.get("min_dimensions"), (32, 32)
                ),
                max_dimensions=_dimension_pair(
                    resize.get("max_dimensions"), (672, 192)
                ),
                mean=_single_channel_value(normalize.get("mean"), 0.7931),
                std=_single_channel_value(normalize.get("std"), 0.1738),
                output_divisor=16,
            ),
            _LaTeXOCRDecoder(character_dict),
        )

    raise PaddleOCRNotReadyError(
        f"暂不支持公式模型解码器 {postprocess_name!r}：{config_path}"
    )


def _dimension_pair(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, int) and item > 0 for item in value)
    ):
        return int(value[0]), int(value[1])
    return default


def _single_channel_value(value: Any, default: float) -> float:
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _preprocess_image(image_path: Path, spec: _PreprocessSpec) -> Any:
    try:
        import numpy as np
        image = load_rgb_image(image_path)
    except Exception as exc:
        raise PaddleOCRNotReadyError(f"无法读取公式图片：{image_path}") from exc

    if spec.family == "unimernet":
        assert spec.input_size is not None
        prepared = _prepare_unimernet_image(image, spec.input_size)
        array = np.asarray(prepared, dtype=np.float32)
        # Albumentations receives RGB and applies its RGB-to-gray conversion
        # before normalization, matching PaddleOCR's UniMERNetTestTransform.
        normalized = (array / np.float32(255.0) - np.float32(spec.mean)) / np.float32(
            spec.std
        )
        gray = (
            normalized[..., 0] * np.float32(0.299)
            + normalized[..., 1] * np.float32(0.587)
            + normalized[..., 2] * np.float32(0.114)
        )
    elif spec.family == "latex_ocr":
        prepared = _prepare_latex_ocr_image(
            image,
            min_dimensions=spec.min_dimensions,
            max_dimensions=spec.max_dimensions,
        )
        array = np.asarray(prepared)
        if array.ndim == 2:
            gray_u8 = array.astype(np.uint8, copy=False)
        else:
            gray_u8 = _opencv_gray_from_rgb(array)
        gray = (
            gray_u8.astype(np.float32) / np.float32(255.0) - np.float32(spec.mean)
        ) / np.float32(spec.std)
    else:  # pragma: no cover - guarded while parsing configuration
        raise PaddleOCRNotReadyError(f"未知公式图片预处理类型：{spec.family}")

    height, width = gray.shape
    padded_height = math.ceil(height / spec.output_divisor) * spec.output_divisor
    padded_width = math.ceil(width / spec.output_divisor) * spec.output_divisor
    if (padded_height, padded_width) != (height, width):
        gray = np.pad(
            gray,
            ((0, padded_height - height), (0, padded_width - width)),
            constant_values=1,
        )
    return np.ascontiguousarray(gray[np.newaxis, np.newaxis, ...], dtype=np.float32)


def _prepare_unimernet_image(image: Any, input_size: tuple[int, int]) -> Any:
    import numpy as np
    from PIL import ImageOps

    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    minimum = int(gray.min())
    maximum = int(gray.max())
    cropped = image
    if maximum != minimum:
        normalized = (gray.astype(np.float32) - minimum) / (maximum - minimum) * 255
        coordinates = np.argwhere(normalized < 200)
        if coordinates.size:
            top, left = coordinates.min(axis=0)
            bottom, right = coordinates.max(axis=0)
            crop_width = int(right - left + 1)
            crop_height = int(bottom - top + 1)
            if min(crop_width, crop_height) > 0 and (
                max(crop_width, crop_height) / min(crop_width, crop_height) <= 200
            ):
                cropped = image.crop(
                    (int(left), int(top), int(right) + 1, int(bottom) + 1)
                )

    target_height, target_width = input_size
    width, height = cropped.size
    if min(width, height) <= 0:
        raise PaddleOCRNotReadyError("公式图片裁剪后尺寸为 0。")
    requested = min(input_size)
    if width <= height:
        resized_width = requested
        resized_height = int(requested * height / width)
    else:
        resized_height = requested
        resized_width = int(requested * width / height)
    cropped = cropped.resize((resized_width, resized_height), resample=2)
    cropped.thumbnail((target_width, target_height))

    delta_width = target_width - cropped.width
    delta_height = target_height - cropped.height
    left = delta_width // 2
    top = delta_height // 2
    return ImageOps.expand(
        cropped,
        (left, top, delta_width - left, delta_height - top),
        fill=0,
    )


def _prepare_latex_ocr_image(
    image: Any,
    *,
    min_dimensions: tuple[int, int],
    max_dimensions: tuple[int, int],
) -> Any:
    import numpy as np
    from PIL import Image

    width, height = image.size
    if (
        min_dimensions[0] <= width <= max_dimensions[0]
        and min_dimensions[1] <= height <= max_dimensions[1]
    ):
        return image

    data = np.asarray(image.convert("L"), dtype=np.uint8).astype(np.float32)
    minimum = float(data.min())
    maximum = float(data.max())
    if maximum > minimum:
        data = (data - minimum) / (maximum - minimum) * 255
    if float(data.mean()) > 128:
        mask = data < 128
    else:
        mask = data > 128
        data = 255 - data
    coordinates = np.argwhere(mask)
    if coordinates.size:
        top, left = coordinates.min(axis=0)
        bottom, right = coordinates.max(axis=0)
        data = data[int(top) : int(bottom) + 1, int(left) : int(right) + 1]

    cropped = Image.fromarray(np.clip(data, 0, 255).astype(np.uint8), mode="L")
    padded_width = math.ceil(cropped.width / 32) * 32
    padded_height = math.ceil(cropped.height / 32) * 32
    padded = Image.new("L", (padded_width, padded_height), 255)
    padded.paste(cropped, (0, 0))

    ratios = [
        padded.width / max_dimensions[0],
        padded.height / max_dimensions[1],
    ]
    if any(ratio > 1 for ratio in ratios):
        scale = max(ratios)
        resized = (
            max(1, int(padded.width // scale)),
            max(1, int(padded.height // scale)),
        )
        padded = padded.resize(resized, resample=2)

    target_width = max(padded.width, min_dimensions[0])
    target_height = max(padded.height, min_dimensions[1])
    if (target_width, target_height) != padded.size:
        minimum_size = Image.new("L", (target_width, target_height), 255)
        minimum_size.paste(padded, (0, 0))
        padded = minimum_size
    return padded


def _opencv_gray_from_rgb(array: Any) -> Any:
    import numpy as np

    weighted = (
        array[..., 0].astype(np.float32) * np.float32(0.299)
        + array[..., 1].astype(np.float32) * np.float32(0.587)
        + array[..., 2].astype(np.float32) * np.float32(0.114)
    )
    return np.clip(np.rint(weighted), 0, 255).astype(np.uint8)


def _load_tokenizer(serialized: Any) -> Any:
    try:
        from tokenizers import Tokenizer

        return Tokenizer.from_buffer(json.dumps(serialized).encode("utf-8"))
    except Exception as exc:
        raise PaddleOCRNotReadyError(
            "公式模型 tokenizer 初始化失败，请检查 tokenizers 运行组件。"
        ) from exc


class _UniMERNetDecoder:
    def __init__(self, character_dict: Any) -> None:
        try:
            serialized = character_dict["fast_tokenizer_file"]
        except (KeyError, TypeError) as exc:
            raise PaddleOCRNotReadyError("UniMERNet tokenizer 配置不完整。") from exc
        self.tokenizer = _load_tokenizer(serialized)

    def decode(self, token_ids: Any) -> str:
        import numpy as np

        rows = np.asarray(token_ids)
        if rows.ndim == 1:
            rows = rows[np.newaxis, :]
        if rows.ndim != 2 or rows.shape[0] == 0:
            return ""
        row = rows[0].astype(np.int64, copy=False)
        eos = np.flatnonzero(row == 2)
        if eos.size:
            row = row[: int(eos[0]) + 1]
        text = self.tokenizer.decode(row.tolist(), skip_special_tokens=True)
        return _normalize_unimernet_text(text)


class _LaTeXOCRDecoder:
    def __init__(self, character_dict: Any) -> None:
        self.tokenizer = _load_tokenizer(character_dict)

    def decode(self, token_ids: Any) -> str:
        import numpy as np

        rows = np.asarray(token_ids)
        if rows.ndim == 1:
            rows = rows[np.newaxis, :]
        if rows.ndim != 2 or rows.shape[0] == 0:
            return ""
        decoded = self.tokenizer.decode(rows[0].astype(np.int64).tolist())
        text = (
            "".join(decoded.split(" "))
            .replace("Ġ", " ")
            .replace("[EOS]", "")
            .replace("[BOS]", "")
            .replace("[PAD]", "")
            .strip()
        )
        return _normalize_latex_ocr_text(text)


def _normalize_latex_ocr_text(text: str) -> str:
    return clean_recognized_latex(text)


def _normalize_unimernet_text(text: str) -> str:
    text = _remove_chinese_text_wrapping(text)
    try:
        from ftfy import fix_text

        text = fix_text(text)
    except ImportError:
        text = unicodedata.normalize("NFC", text)
    return clean_recognized_latex(text)


def _remove_chinese_text_wrapping(formula: str) -> str:
    pattern = re.compile(r"\\text\s*{([^{}]*[\u4e00-\u9fff]+[^{}]*)}")
    return pattern.sub(lambda match: match.group(1), formula).replace('"', "")
