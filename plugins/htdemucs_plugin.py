"""
HTDemucs Plugin Facade — Routes to DemucsV4Plugin (htdemucs_6s.onnx).

Provides ``get_htdemucs_plugin()`` as bridge-compatible accessor (§9.7.4).
Delegates to ``plugins.demucs_v4_plugin.get_demucs_plugin()``.

Author: Aurik Development Team
Version: 9.10.57
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plugins.demucs_v4_plugin import DemucsV4Plugin

logger = logging.getLogger(__name__)

try:
    from plugins.demucs_v4_plugin import get_demucs_plugin as _get_demucs
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False
    logger.debug("htdemucs_plugin: DemucsV4Plugin not available")


def get_htdemucs_plugin() -> "DemucsV4Plugin | None":
    """Return the HTDemucs/Demucs V4 singleton, or None if unavailable."""
    if not _AVAILABLE:
        return None
    return _get_demucs()
