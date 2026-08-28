"""Lightweight contracts shared by model download and inference layers."""

from __future__ import annotations

from typing import Callable


DownloadProgressCallback = Callable[[str, int, int], None]


class ModelDownloadError(RuntimeError):
    """Raised when a model cannot be downloaded, verified, or installed."""


class ModelDownloadCancelled(RuntimeError):
    """Raised when the user cancels an active model download."""
