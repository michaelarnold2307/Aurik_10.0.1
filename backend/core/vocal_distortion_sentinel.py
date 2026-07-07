"""
§2.59 Vocal Distortion Sentinel (2026-07-09)

Erkennt Gesangsverzerrung durch MESSUNG, nicht durch Pauschal-Annahmen.
Signalisiert an die Pipeline: Schutz-Phasen fehlen, HNR verschlechtert sich.
Die tatsächliche Stärke-Anpassung erfolgt in den Phasen selbst
(§2.46b tilt-cap, §0p VocalNoHarmGate, PMGG).

Prinzip: Messen → Signalisieren → Phase handelt selbstständig.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class VocalDistortionSentinel:
    """Misst Gesangsqualität und signalisiert Handlungsbedarf."""

    def __init__(self, singing_confidence: float = 0.0) -> None:
        self._singing_conf = singing_confidence
        self._hnr_before: float | None = None
        self._hnr_after: float | None = None
        self._harmonic_restoration_applied: bool = False
        self._deesser_applied: bool = False
        self._warnings: list[str] = []
        self._injected_phases: list[str] = []

    def set_baseline_hnr(self, hnr_db: float) -> None:
        self._hnr_before = hnr_db

    def record_phase(self, phase_id: str) -> None:
        if "harmonic_restoration" in phase_id:
            self._harmonic_restoration_applied = True
        if "de_esser" in phase_id or "deesser" in phase_id:
            self._deesser_applied = True

    def check(self, post_hnr_db: float | None = None) -> dict[str, Any]:
        """Misst und signalisiert. KEINE pauschalen Strength-Overrides."""
        self._warnings = []
        self._injected_phases = []

        # Messung 1: HNR-Veränderung
        hnr_delta = None
        if post_hnr_db is not None and self._hnr_before is not None:
            hnr_delta = post_hnr_db - self._hnr_before
            if hnr_delta < -3.0:
                self._warnings.append(
                    f"HNR-Abfall gemessen: {hnr_delta:+.1f} dB "
                    f"({self._hnr_before:.1f} → {post_hnr_db:.1f}) — "
                    f"VocalNoHarmGate sollte Harmonic Restoration zurücknehmen"
                )

        # Messung 2: Fehlende Schutz-Phasen
        if self._harmonic_restoration_applied and not self._deesser_applied:
            if self._singing_conf >= 0.25:
                self._warnings.append(
                    f"Harmonic Restoration aktiv (singing={self._singing_conf:.2f}) "
                    f"aber KEIN De-Esser im Plan — wird jetzt injiziert"
                )
                self._injected_phases.extend([
                    "phase_19_de_esser",
                    "phase_43_ml_deesser",
                ])

        for w in self._warnings:
            logger.warning("🎤 VocalSentinel: %s", w)
        if self._injected_phases:
            logger.info(
                "🎤 VocalSentinel injiziert: %s", ", ".join(self._injected_phases)
            )

        return {
            "warnings": self._warnings,
            "injected_phases": self._injected_phases,
            "hnr_delta_db": hnr_delta,
            "has_actions": bool(self._injected_phases),
        }
