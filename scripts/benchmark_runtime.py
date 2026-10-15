"""Compare legacy and bounded ONNX scheduling on the same local model/image.

Run each policy in a fresh process, e.g.:
  python scripts/benchmark_runtime.py --policy bounded --model-dir ... --image ...
No downloads or changes to model files are performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("legacy", "bounded"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    from formula_ocr_app import rapid_recognizer

    if args.policy == "legacy":
        def legacy_session(ort, model_path, **_kwargs):
            options = ort.SessionOptions()
            options.log_severity_level = 3
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            return ort.InferenceSession(str(model_path), sess_options=options,
                                        providers=["CPUExecutionProvider"])
        rapid_recognizer.create_inference_session = legacy_session
    recognizer = rapid_recognizer.RapidLatexRecognizer()
    recognizer._model_ensure = lambda **_kwargs: args.model_dir.resolve()
    started = time.perf_counter()
    recognizer._ensure_model()
    loaded = time.perf_counter() - started
    predictions = []
    for _ in range(2):
        started = time.perf_counter()
        formula = recognizer.predict(args.image)
        predictions.append({"seconds": time.perf_counter() - started, "latex": formula})
    cpu = time.process_time()
    time.sleep(1)
    idle_cpu = time.process_time() - cpu
    recognizer.close()
    print(json.dumps({"policy": args.policy, "load_seconds": loaded,
                      "predictions": predictions, "idle_cpu_seconds": idle_cpu}, ensure_ascii=False))


if __name__ == "__main__":
    main()
