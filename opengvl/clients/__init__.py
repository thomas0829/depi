"""Minimal clients package exposing QwenClient."""

from opengvl.clients.base import BaseModelClient  # noqa: F401
from opengvl.clients.qwen import QwenClient  # noqa: F401

__all__ = ["BaseModelClient", "QwenClient"]
