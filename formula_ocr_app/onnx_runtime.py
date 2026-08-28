"""Shared ONNX Runtime session policy for formula-recognition backends."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypeVar

RuntimeErrorType = TypeVar("RuntimeErrorType", bound=RuntimeError)


def create_inference_session(
    onnxruntime: Any,
    model_path: str | Path,
    *,
    device: str = "cpu",
    error_type: type[RuntimeErrorType] = RuntimeError,
) -> Any:
    """Create one consistently configured inference session.

    Model adapters retain responsibility for interpreting model inputs and
    outputs. This function only centralizes provider selection, thread limits,
    logging and graph optimization so every backend behaves the same way.
    """

    path = Path(model_path)
    if not path.is_file():
        raise error_type(f"ONNX 模型文件不存在：{path}")

    options = onnxruntime.SessionOptions()
    options.log_severity_level = 3
    options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    threads = _thread_count()
    if threads is not None:
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1

    return onnxruntime.InferenceSession(
        str(path),
        sess_options=options,
        providers=execution_providers(
            onnxruntime,
            device,
            error_type=error_type,
        ),
    )


def execution_providers(
    onnxruntime: Any,
    device: str,
    *,
    error_type: type[RuntimeErrorType] = RuntimeError,
) -> list[str]:
    available = set(onnxruntime.get_available_providers())
    normalized = (device or "cpu").strip().lower()
    if normalized.startswith(("gpu", "cuda")):
        if "CUDAExecutionProvider" not in available:
            raise error_type(
                "当前 ONNX Runtime 未提供 CUDAExecutionProvider，请选择 CPU。"
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def repeated_token_suffix_start(
    token_ids: list[int],
    *,
    min_generated_tokens: int,
    min_repeated_tokens: int,
    max_period: int,
    min_repetitions: int,
) -> int | None:
    """Return where a looping suffix should be trimmed, retaining one period."""

    token_count = len(token_ids)
    if token_count < min_generated_tokens:
        return None
    for period in range(1, max_period + 1):
        pattern = token_ids[-period:]
        start = token_count - period
        repetitions = 1
        while start >= period and token_ids[start - period : start] == pattern:
            start -= period
            repetitions += 1
        if (
            repetitions >= min_repetitions
            and repetitions * period >= min_repeated_tokens
        ):
            return start + period
    return None


def _thread_count() -> int | None:
    raw_value = os.environ.get("FORMULA_OCR_ONNX_THREADS", "").strip()
    if raw_value.isdigit():
        return max(1, int(raw_value))
    # Leave ONNX Runtime's own scheduler in control by default. Users with
    # constrained systems can set the environment variable explicitly.
    return None
