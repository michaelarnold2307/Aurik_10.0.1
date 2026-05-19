"""§Gap7 SectionGoalAdapter — Sektionsweise Musical-Goal-Anpassung (v9.12.8).

Segmentiert einen Song in musikalische Sektionen (Intro/Verse/Chorus/Bridge/Outro)
anhand des Energie- und Onsetprofils. Gibt pro Sektion angepasste `SectionTarget`-
Gewichte zurück, die feingranularere Qualitätsziele ermöglichen als ein globales
Durchschnittsziel.

Typische Anpassungen:
  Intro/Outro (leise):    Raumtiefe+Wärme wichtiger, Brillanz/NR weniger aggressiv.
  Chorus (laut, emotional): Frisson-Schutz aktiv, Timbral-Fidelity erhöht.
  Verse (gesang-zentral):  Vokal-Qualität und Artikulation priorisiert.
  Bridge (dynamisch):      Mikrodynamik und emotionale Kontinuität betont.

Verwendung in UV3 (nach VocalFocusAnalyzer, vor GoalApplicabilityFilter):
    from backend.core.section_goal_adapter import get_section_goal_adapter
    _section_targets = get_section_goal_adapter().compute_section_targets(audio, sr)
    _restoration_context["section_targets"] = _section_targets

Phasen greifen auf per-Section-Gewichte zu:
    _secs = kwargs.get("_restoration_context", {}).get("section_targets", [])
    _t = t_sample / sr
    weight = next((s.nr_strength_scale for s in _secs if s.start_s <= _t < s.end_s), 1.0)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------

_MIN_SECTION_DURATION_S: float = 2.0  # Minimale Sektionslänge (verhindert Micro-Splitting)
_CHORUS_ENERGY_PERCENTILE: float = 75.0  # Frames oberhalb dieses RMS-Perzentils → Chorus-Kandidat
_INTRO_OUTRO_MAX_FRAC: float = 0.15  # Max. 15% des Songs als Intro/Outro
_RMS_FRAME_DURATION_S: float = 0.050  # 50 ms RMS-Frames
_ONSET_DENSITY_THRESHOLD: float = 5.0  # Onsets/s oberhalb → aktive Sektion


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class SectionTarget:
    """Qualitätsziel-Gewichte für eine musikalische Sektion."""

    start_s: float = 0.0
    end_s: float = 0.0
    section_type: str = "verse"  # "intro" | "verse" | "chorus" | "bridge" | "outro"
    nr_strength_scale: float = 1.0  # NR-Stärke in dieser Sektion (1.0 = Standard)
    vq_weight: float = 1.0  # Vokal-Qualitäts-Gewichtung (1.0 = Standard)
    frisson_protection: bool = False  # Frisson-Schutz aktiv?
    extra: dict[str, float] = field(default_factory=dict)

    def duration_s(self) -> float:
        """Gibt die Dauer dieser Sektion in Sekunden zurück."""
        return max(0.0, self.end_s - self.start_s)

    def to_dict(self) -> dict[str, Any]:
        """Serialisiert die Sektion als Dictionary."""
        return {
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
            "section_type": self.section_type,
            "nr_strength_scale": round(self.nr_strength_scale, 3),
            "vq_weight": round(self.vq_weight, 3),
            "frisson_protection": self.frisson_protection,
            **{f"extra_{k}": round(v, 3) for k, v in self.extra.items()},
        }


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class SectionGoalAdapter:
    """Singleton — DSP-basierte Sektionssegmentierung ohne ML-Abhängigkeit."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def compute_section_targets(
        self,
        audio: np.ndarray,
        sr: int,
        frisson_zones: list | None = None,
    ) -> list[SectionTarget]:
        """Berechnet pro-Sektion-Gewichte aus Energie- und Onset-Profil.

        Args:
            audio:         Mono/Stereo audio (float32), bereits bei 48000 Hz.
            sr:            Abtastrate (48000 Hz).
            frisson_zones: Optional vorberechnete Frisson-Zonen (Liste von (start_s, end_s)).

        Returns:
            Liste von SectionTarget-Objekten (chronologisch sortiert).
            Leer bei Fehler (non-blocking).
        """
        try:
            return self._compute(audio, sr, frisson_zones or [])
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("SectionGoalAdapter non-blocking error: %s", exc)
            return []

    def _compute(
        self,
        audio: np.ndarray,
        sr: int,
        frisson_zones: list,
    ) -> list[SectionTarget]:
        """Kern-Berechnung der Sektionssegmentierung."""
        # Mono-Downmix für Analyse
        if audio.ndim == 2:
            mono = audio.mean(axis=0)
        else:
            mono = np.asarray(audio, dtype=np.float32)

        if len(mono) < int(sr * _MIN_SECTION_DURATION_S * 2):
            # Zu kurz für Segmentierung → eine globale Sektion
            duration_s = float(len(mono)) / max(1, sr)
            return [SectionTarget(start_s=0.0, end_s=duration_s, section_type="verse")]

        # --- RMS-Energie-Profil ---
        frame_size = int(_RMS_FRAME_DURATION_S * sr)
        n_frames = len(mono) // frame_size
        rms_frames = np.array(
            [float(np.sqrt(np.mean(mono[k * frame_size : (k + 1) * frame_size] ** 2))) for k in range(n_frames)]
        )
        rms_frames = np.clip(rms_frames, 1e-8, None)
        total_duration_s = float(len(mono)) / max(1, sr)
        frame_times_s = np.arange(n_frames) * _RMS_FRAME_DURATION_S

        # --- Chorus-Erkennung: hohe Energie + Wiederholung ---
        chorus_threshold = float(np.percentile(rms_frames, _CHORUS_ENERGY_PERCENTILE))
        is_chorus_frame = rms_frames >= chorus_threshold

        # --- Intro/Outro: erste/letzte 15% mit niedriger Energie ---
        intro_end_frame = max(1, int(n_frames * _INTRO_OUTRO_MAX_FRAC))
        outro_start_frame = max(intro_end_frame + 1, n_frames - intro_end_frame)

        # --- Sektionen zusammenstellen ---
        sections: list[SectionTarget] = []

        # Intro
        intro_end_s = float(frame_times_s[intro_end_frame - 1] + _RMS_FRAME_DURATION_S)
        sections.append(
            SectionTarget(
                start_s=0.0,
                end_s=intro_end_s,
                section_type="intro",
                nr_strength_scale=0.75,  # Sanftere NR im Intro (mehr Raumcharakter)
                vq_weight=0.90,
                frisson_protection=False,
            )
        )

        # Mittelteil: Verse/Chorus-Abwechslung
        prev_end_s = intro_end_s
        _chunk_dur_s = max(4.0, (outro_start_frame - intro_end_frame) * _RMS_FRAME_DURATION_S / 4)
        _chunk_frames = max(1, int(_chunk_dur_s / _RMS_FRAME_DURATION_S))
        for chunk_start in range(intro_end_frame, outro_start_frame, _chunk_frames):
            chunk_end = min(chunk_start + _chunk_frames, outro_start_frame)
            chunk_frames = is_chorus_frame[chunk_start:chunk_end]
            _chorus_ratio = float(np.mean(chunk_frames)) if len(chunk_frames) > 0 else 0.0
            _chunk_start_s = float(frame_times_s[chunk_start])
            _chunk_end_s = float(frame_times_s[min(chunk_end - 1, n_frames - 1)] + _RMS_FRAME_DURATION_S)
            _chunk_end_s = min(_chunk_end_s, total_duration_s)

            if _chunk_end_s - _chunk_start_s < _MIN_SECTION_DURATION_S:
                # Zu kurz → zur letzten Sektion zusammenführen
                if sections:
                    sections[-1].end_s = _chunk_end_s
                continue

            if _chorus_ratio >= 0.60:
                # Chorus — hohe Energie, Frisson-Kandidat
                _is_frisson = self._check_frisson_overlap(_chunk_start_s, _chunk_end_s, frisson_zones)
                sections.append(
                    SectionTarget(
                        start_s=_chunk_start_s,
                        end_s=_chunk_end_s,
                        section_type="chorus",
                        nr_strength_scale=0.80,  # NR zurückhalten (Energie erhalten)
                        vq_weight=1.20,  # Vokal-Qualität priorisiert
                        frisson_protection=_is_frisson,
                        extra={"chorus_energy_ratio": round(_chorus_ratio, 2)},
                    )
                )
            else:
                # Verse oder Bridge
                _stype = "verse" if len(sections) % 2 == 1 else "bridge"
                sections.append(
                    SectionTarget(
                        start_s=_chunk_start_s,
                        end_s=_chunk_end_s,
                        section_type=_stype,
                        nr_strength_scale=1.00,
                        vq_weight=1.10,
                        frisson_protection=False,
                    )
                )
            prev_end_s = _chunk_end_s

        # Outro
        outro_start_s = (
            float(frame_times_s[outro_start_frame]) if outro_start_frame < n_frames else (total_duration_s - 2.0)
        )
        outro_start_s = max(prev_end_s, outro_start_s)
        if total_duration_s - outro_start_s >= _MIN_SECTION_DURATION_S:
            sections.append(
                SectionTarget(
                    start_s=outro_start_s,
                    end_s=total_duration_s,
                    section_type="outro",
                    nr_strength_scale=0.75,
                    vq_weight=0.90,
                    frisson_protection=False,
                )
            )

        # Lücken schließen (sicherstellen dass sections nahtlos sind)
        sections = sorted(sections, key=lambda s: s.start_s)
        for i in range(1, len(sections)):
            if sections[i].start_s < sections[i - 1].end_s:
                sections[i].start_s = sections[i - 1].end_s

        logger.debug(
            "SectionGoalAdapter: %d Sektionen für %.1f s Audio",
            len(sections),
            total_duration_s,
        )
        return sections

    @staticmethod
    def _check_frisson_overlap(
        start_s: float,
        end_s: float,
        frisson_zones: list,
    ) -> bool:
        """True wenn eine Frisson-Zone sich mit [start_s, end_s) überschneidet."""
        for zone in frisson_zones:
            try:
                z_start = float(zone[0] if isinstance(zone, (list, tuple)) else getattr(zone, "start_s", 0))
                z_end = float(zone[1] if isinstance(zone, (list, tuple)) else getattr(zone, "end_s", 0))
                if z_start < end_s and z_end > start_s:
                    return True
            except Exception:
                pass
        return False


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: SectionGoalAdapter | None = None
_lock = threading.Lock()


def get_section_goal_adapter() -> SectionGoalAdapter:
    """Thread-safe Singleton-Accessor."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = SectionGoalAdapter()
    return _instance
