from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from formula_ocr_app.interprocess_lock import InterProcessFileLock
    from formula_ocr_app.model_catalog import DEFAULT_MODEL_ID, MODEL_BY_ID
    from formula_ocr_app.runtime_paths import user_data_dir
except ModuleNotFoundError as exc:  # Allows `python formula_ocr_app/app.py`.
    if exc.name != "formula_ocr_app":
        raise
    from interprocess_lock import InterProcessFileLock
    from model_catalog import DEFAULT_MODEL_ID, MODEL_BY_ID
    from runtime_paths import user_data_dir


@dataclass(frozen=True)
class AppSettings:
    model_id: str = DEFAULT_MODEL_ID
    accepted_model_terms: tuple[str, ...] = ()


def settings_path() -> Path:
    return user_data_dir() / "settings.json"


def load_settings() -> AppSettings:
    path = settings_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return AppSettings()
    if not isinstance(data, dict):
        return AppSettings()
    model_id = str(data.get("model_id", DEFAULT_MODEL_ID))
    if model_id not in MODEL_BY_ID:
        model_id = DEFAULT_MODEL_ID
    raw_terms = data.get("accepted_model_terms", ())
    if isinstance(raw_terms, (list, tuple, set)):
        accepted_model_terms = tuple(
            sorted({str(item) for item in raw_terms if str(item).strip()})
        )
    else:
        accepted_model_terms = ()
    return AppSettings(
        model_id=model_id,
        accepted_model_terms=accepted_model_terms,
    )


def save_settings(settings: AppSettings) -> None:
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with InterProcessFileLock(lock_path, poll_interval=0.01):
        temporary_name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
                json.dump(
                    asdict(settings),
                    temporary,
                    ensure_ascii=False,
                    indent=2,
                )
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, path)
            temporary_name = None
        finally:
            if temporary_name is not None:
                try:
                    Path(temporary_name).unlink(missing_ok=True)
                except OSError:
                    pass
