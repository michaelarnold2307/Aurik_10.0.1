"""
§2.59 QualityMode-Validierung (2026-07-09)

Zentrale Validierung aller Quality-Mode-Strings.
Verhindert stille Fallbacks durch Tippfehler wie "restoraton".

Usage:
  from backend.core.quality_mode import validate_mode, QUALITY_MODES
  mode = validate_mode(user_input)  # "restoration" → "restoration"
                                     # "restoraton" → WARNING + Fallback
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Kanonische Modi ────────────────────────────────────────────────────────

QUALITY_MODES: frozenset[str] = frozenset({
    "restoration",
    "quality",
    "maximum",
    "studio_2026",
    "balanced",
    "fast",
})

MODE_ALIASES: dict[str, str] = {
    "restoration": "quality",
    "studio_2026": "maximum",
    "quality": "quality",
    "maximum": "maximum",
    "balanced": "balanced",
    "fast": "fast",
}

MODE_FALLBACK = "quality"


def validate_mode(mode: Any, fallback: str = MODE_FALLBACK) -> str:
    """Validiert und normalisiert einen Quality-Mode-String.

    Args:
        mode: Roher Mode-String vom User/API
        fallback: Fallback-Mode bei ungültiger Eingabe

    Returns:
        Kanonischer Mode-String
    """
    if mode is None or not isinstance(mode, str):
        logger.warning(
            "QualityMode: invalid type %s, fallback to '%s'",
            type(mode).__name__ if mode is not None else "None",
            fallback,
        )
        return fallback

    mode_lower = mode.strip().lower()

    if mode_lower in MODE_ALIASES:
        canonical = MODE_ALIASES[mode_lower]
        if canonical != mode_lower:
            logger.debug("QualityMode: alias '%s' → '%s'", mode_lower, canonical)
        return canonical

    # Check partial matches for common typos
    for known in QUALITY_MODES:
        if mode_lower in known or known in mode_lower:
            logger.warning(
                "QualityMode: '%s' is not a valid mode. Did you mean '%s'? "
                "Falling back to '%s'.",
                mode,
                known,
                fallback,
            )
            return fallback

    logger.warning(
        "QualityMode: '%s' is not a valid mode. Valid: %s. Fallback to '%s'.",
        mode,
        sorted(QUALITY_MODES),
        fallback,
    )
    return fallback
