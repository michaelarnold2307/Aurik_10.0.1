"""Phase result summary — human-readable Fazit-Meldung for every Aurik phase.

Usage inside a phase's process():
    from backend.core.phase_fazit import log_phase_fazit
    log_phase_fazit(
        phase="03",
        name="Entrauschen",
        score=8.5,
        summary="Rauschen um 12.3 dB reduziert, Sprachverständlichkeit erhalten",
        details={"snr_before_db": 18.2, "snr_after_db": 30.5},
    )

Output in logs:
    ┌─ Phase 03 (Entrauschen) ──────────────────────────────────────────┐
    │ ✅ Rauschen um 12.3 dB reduziert, Sprachverständlichkeit erhalten  │
    │ 📊 Score: 8.5 / 10.0  (SNR: 18.2 → 30.5 dB)                      │
    └────────────────────────────────────────────────────────────────────┘

Score meaning:
    10.0 = Phase zu 100% wie geplant umgesetzt, alle Defekte unhörbar
     7.0 = Deutliche Verbesserung, leichte Restdefekte hörbar
     5.0 = Moderate Verbesserung, Defekte teilweise reduziert
     3.0 = Geringe Verbesserung, Defekte noch deutlich hörbar
     0.0 = Keine Verbesserung / Phase wirkungslos
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Width of the box
_WIDTH = 72


def log_phase_fazit(
    phase: str,
    name: str,
    score: float,
    summary: str,
    details: dict[str, Any] | None = None,
) -> None:
    """Log a human-readable phase summary with score.

    Args:
        phase: Phase number (e.g. "03", "09", "24")
        name:  Human-readable phase name (e.g. "Entrauschen", "Knackser-Entfernung")
        score: 0.0–10.0 rating of phase success
        summary: One-sentence summary of what was achieved
        details: Optional dict of key metrics (e.g. {"SNR": "18→30 dB"})
    """
    score = float(max(0.0, min(10.0, score)))
    score_emoji = _score_emoji(score)

    # Build header
    header = f" Phase {phase} ({name}) "
    pad_total = _WIDTH - len(header) - 2  # 2 for ┌─ and ─┐
    if pad_total > 0:
        header = "┌─" + header + "─" * pad_total + "┐"
    else:
        header = "┌─" + header[: _WIDTH - 4] + "─┐"

    # Build score line
    score_text = f" {score_emoji} Score: {score:.1f} / 10.0"
    if details:
        detail_parts = []
        for k, v in list(details.items())[:3]:
            detail_parts.append(f"{k}: {v}")
        score_text += "  (" + ", ".join(detail_parts) + ")"
    score_text = score_text.ljust(_WIDTH - 2)[: _WIDTH - 2] + " │"

    # Build summary line (may wrap)
    summary_lines = _wrap_text(f" {summary}", _WIDTH - 4)
    summary_formatted = []
    for sl in summary_lines:
        summary_formatted.append("│ " + sl.ljust(_WIDTH - 4) + " │")

    # Build footer
    footer = "└" + "─" * (_WIDTH - 2) + "┘"

    # Log as single multi-line INFO message
    lines = [header] + summary_formatted + [score_text, footer]
    logger.info("\n".join(lines))


def _score_emoji(score: float) -> str:
    """Return an emoji for the score bracket."""
    if score >= 9.0:
        return "🏆"
    elif score >= 7.5:
        return "✅"
    elif score >= 5.0:
        return "👍"
    elif score >= 2.5:
        return "⚠️"
    else:
        return "❌"


def _wrap_text(text: str, width: int) -> list[str]:
    """Wrap text to width, returning list of lines."""
    if len(text) <= width:
        return [text]
    lines = []
    while len(text) > width:
        # Find last space within width
        split = text.rfind(" ", 0, width)
        if split < 0:
            split = width
        lines.append(text[:split])
        text = " " + text[split:].lstrip()
    if text.strip():
        lines.append(text)
    return lines
