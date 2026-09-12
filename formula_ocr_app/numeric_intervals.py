"""Conservative recovery of single-line numeric intervals with the same model.

Some parallel formula decoders stop at the first percent sign in an interval,
although they recognize either endpoint on its own. Recovery requires visible
delimiters and one comma, two numeric predictions, and agreement with every
number already read. Ambiguous images and general formulas keep their output.
"""

from __future__ import annotations

import re
from typing import Any, Callable

try:
    from formula_ocr_app.image_utils import foreground_bbox
except ModuleNotFoundError as exc:
    if exc.name != "formula_ocr_app":
        raise
    from image_utils import foreground_bbox


_VALUE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_NUMBER = re.compile(rf"({_VALUE})(\\?%)?")


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text).replace("−", "-")


def _number(text: str) -> str | None:
    match = _NUMBER.fullmatch(_compact(text))
    if match is None:
        return None
    return match[1] + (r"\%" if match[2] else "")


def normalize_numeric_formula(latex: str) -> str | None:
    """Normalize a standalone number or a complete numeric interval only."""

    return _number(latex) or normalize_numeric_interval(latex)


def normalize_numeric_interval(latex: str) -> str | None:
    """Join OCR-spaced digits only in a complete, purely numeric interval."""

    text = re.sub(r"\\(?:left|right)\b", "", latex)
    match = re.fullmatch(r"([\[(])([^,]+),([^,]+)([\])])", _compact(text))
    if match is None:
        return None
    left, right = _number(match[2]), _number(match[3])
    if left is None or right is None:
        return None
    return f"{match[1]}{left}, {right}{match[4]}"


def numeric_interval_needs_retry(latex: str) -> bool:
    if len(latex) > 160 or not re.search(r"[\[(]|\\left\b", latex):
        return False
    if normalize_numeric_interval(latex) is not None:
        return False
    # A bare "left" is a known decoder artifact, as in "[left- 4 5.6 7 \\".
    text = re.sub(r"\\?(?:left|right)\b", "", latex)
    return bool(
        re.search(r"\d", text)
        and re.fullmatch(r"[\d\s+−.,%\\()\[\]-]+", text)
    )


def recover_numeric_interval(
    image: Any,
    latex: str,
    recognize: Callable[[Any], str],
) -> str:
    """Make at most two extra predictions; never recursively retry crops."""

    if not numeric_interval_needs_retry(latex):
        return latex
    parts = split_numeric_interval_image(image)
    if parts is None:
        return latex
    opening, left_image, right_image, closing = parts
    anchors = _NUMBER.findall(_compact(latex))
    if not 1 <= len(anchors) <= 2:
        return latex
    values: list[str] = []
    for index, endpoint in enumerate((left_image, right_image)):
        # Large screenshot scaling can make the small parallel decoder repeat
        # a narrow digit. Normalize only these isolated single-line crops;
        # ordinary formula preprocessing and high-resolution originals stay intact.
        prepared = endpoint
        if endpoint.height > 48:
            from PIL import Image

            prepared = endpoint.resize(
                (max(1, round(endpoint.width * 48 / endpoint.height)), 48),
                Image.Resampling.LANCZOS,
            )
        value = _number(recognize(prepared))
        if value is None or not _numeric_glyph_count_matches(endpoint, value):
            return latex
        if index < len(anchors):
            number, percent = anchors[index]
            if value.removesuffix(r"\%") != number:
                return latex
            if percent and not value.endswith(r"\%"):
                return latex
        values.append(value)
    return f"{opening}{values[0]}, {values[1]}{closing}"


def _numeric_glyph_count_matches(image: Any, value: str) -> bool:
    import numpy as np

    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    low, high = int(gray.min()), int(gray.max())
    occupied = np.any(gray < low + (high - low) * 0.7, axis=0)
    glyph_count = np.count_nonzero(np.diff(np.r_[False, occupied, False])) // 2
    # Digits, decimal points, signs and percent glyphs must all be separated.
    # Touching or unusually disconnected glyphs are ambiguous; do not guess.
    return glyph_count == len(value.replace(r"\%", "%"))


def split_numeric_interval_image(image: Any) -> tuple[str, Any, Any, str] | None:
    """Locate separated brackets and a descending comma on a light background.

    This is a layout check, not digit recognition. Projection gaps must fully
    separate the delimiters and comma from both endpoints. Small, touching,
    dark-background, noisy and multiline inputs are deliberately declined.
    """

    import numpy as np

    if image.width * image.height > 4_000_000:
        return None
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    low, high = int(gray.min()), int(gray.max())
    if high - low < 32:
        return None
    mask = gray < low + (high - low) * 0.7
    if float(mask.mean()) > 0.4:
        return None
    bounds = foreground_bbox(mask)
    if bounds is None:
        return None
    x, y, end_x, end_y = bounds
    mask = mask[y:end_y, x:end_x]
    height, width = mask.shape
    if height < 12 or not 2.5 <= width / height <= 30:
        return None
    occupied = np.any(mask, axis=0)
    runs = np.flatnonzero(np.diff(np.r_[False, occupied, False])).reshape(-1, 2)
    if not 5 <= len(runs) <= 70:
        return None
    boxes = []
    for start, end in runs:
        rows = np.flatnonzero(np.any(mask[:, start:end], axis=1))
        boxes.append((int(start), int(rows[0]), int(end), int(rows[-1]) + 1))
    first, last = boxes[0], boxes[-1]
    opening = _delimiter(mask[:, first[0]:first[2]], left=True)
    closing = _delimiter(mask[:, last[0]:last[2]], left=False)
    if opening is None or closing is None:
        return None

    interior = mask[:, first[2]:last[0]]
    occupied_rows = np.flatnonzero(np.any(interior, axis=1))
    if occupied_rows.size < 2 or np.diff(occupied_rows).max() > height * 0.18:
        return None
    tall = [box for box in boxes[1:-1] if box[3] - box[1] >= height * 0.5]
    if len(tall) < 2:
        return None
    baseline = float(np.median([box[3] for box in tall]))
    commas = [
        index for index, (left, top, right, bottom) in enumerate(boxes[1:-1], 1)
        if right - left <= height * 0.3
        and height * 0.15 <= bottom - top <= height * 0.45
        and top >= height * 0.5
        and bottom >= baseline + max(1, height * 0.07)
    ]
    if len(commas) != 1:
        return None
    comma_index = commas[0]
    if comma_index < 2 or comma_index > len(boxes) - 3:
        return None
    comma = boxes[comma_index]
    left_crop = image.crop((x + first[2], y, x + comma[0], end_y))
    right_crop = image.crop((x + comma[2], y, x + last[0], end_y))
    return opening, left_crop, right_crop, closing


def _delimiter(mask: Any, *, left: bool) -> str | None:
    """Require a straight bracket spine or a symmetric curved parenthesis."""

    import numpy as np

    height, width = mask.shape
    if not 3 <= width <= height * 0.38 or not np.any(mask, axis=1).all():
        return None
    if not left:
        mask = mask[:, ::-1]
    band = max(1, height // 6)
    middle = mask[height // 2 - band:height // 2 + band]
    middle_x = np.nonzero(middle)[1]
    top_x = np.nonzero(mask[:band])[1]
    bottom_x = np.nonzero(mask[-band:])[1]
    if not all(part.size for part in (middle_x, top_x, bottom_x)):
        return None
    if float(middle_x.mean()) >= width * 0.4:
        return None
    coverage = mask.sum(axis=0) / height
    if float(coverage.max()) >= 0.85:
        if np.argmax(coverage) >= width * 0.4:
            return None
        if min(int(top_x.max()), int(bottom_x.max())) + 1 < width * 0.75:
            return None
        return "[" if left else "]"
    if (
        float(coverage.max()) <= 0.8
        and min(float(top_x.mean()), float(bottom_x.mean())) >= width * 0.55
        and abs(float(top_x.mean()) - float(bottom_x.mean())) <= width * 0.2
    ):
        return "(" if left else ")"
    return None
