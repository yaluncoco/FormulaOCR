from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from formula_ocr_app.model_catalog import DEFAULT_MODEL_ID, MODEL_BY_ID
    from formula_ocr_app.runtime_paths import user_data_dir
except ImportError:  # Allows `python formula_ocr_app/app.py`.
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
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(asdict(settings), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)
