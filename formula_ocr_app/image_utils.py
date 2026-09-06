"""Shared image loading rules for every recognition backend and the GUI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


def image_to_rgb(image: Image.Image) -> Image.Image:
    """Return a detached, orientation-correct RGB image on a white background.

    Direct ``RGBA -> RGB`` conversion turns transparent pixels black, which is
    especially harmful for copied formula images with a transparent canvas.
    Formula recognition models are trained predominantly on white paper, so
    alpha is composited onto white before conversion.
    """

    oriented = ImageOps.exif_transpose(image)
    oriented.load()
    if oriented.mode in {"RGBA", "LA"} or "transparency" in oriented.info:
        rgba = oriented.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return oriented.convert("RGB")


def load_rgb_image(image_path: str | Path) -> Image.Image:
    """Open an image without retaining a handle to the source file."""

    with Image.open(image_path) as source:
        return image_to_rgb(source)


def foreground_bbox(mask: Any) -> tuple[int, int, int, int] | None:
    """Return a Pillow bounding box for a 2-D foreground mask.

    Reduce each axis instead of allocating an (N, 2) coordinate array for all
    foreground pixels. Temporary storage then scales with width + height,
    which keeps large screenshots from allocating hundreds of megabytes.
    """

    import numpy as np

    rows = np.flatnonzero(np.any(mask, axis=1))
    if rows.size == 0:
        return None
    columns = np.flatnonzero(np.any(mask, axis=0))
    return int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1
