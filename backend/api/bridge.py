"""Aurik 9 — API Bridge (§11 Spec 08)
=====================================
Einziger Eintrittspunkt für Frontend/CLI → Backend-Core.

Das Frontend darf ``backend/core/``, ``dsp/`` oder ``plugins/`` **nicht**
direkt importieren. Alle Core-Zugriffe laufen über diese Datei.

Verwendung im Frontend::

    from backend.api.bridge import get_quality_mode, get_restorer_classes, get_defect_scanner

Referenz: Spec 08 §11 Softwareschichten-Architektur.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from backend.core.performance_guard import QualityMode as _QualityMode
    from backend.core.unified_restorer_v3 import RestorationConfig as _RestorationConfig
    from backend.core.unified_restorer_v3 import UnifiedRestorerV3 as _UnifiedRestorerV3
    from backend.core.defect_scanner import DefectScanner as _DefectScanner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defect-Scan-Cache  (Thread-sicher, Prozess-Lebensdauer, RAM-only)
# Key: file_path (str), Value: ScanResult-Objekt
# Limit: 64 Einträge (FIFO-Trim)
# ---------------------------------------------------------------------------

_defect_cache: dict[str, object] = {}
_defect_cache_lock = threading.Lock()
_DEFECT_CACHE_MAX = 64


def cache_defect_result(file_path: str, result: object) -> None:
    """Speichert einen DefectScanner-Befund für *file_path* im Cache.

    Thread-sicher. Trimmt den Cache auf _DEFECT_CACHE_MAX Einträge (FIFO).
    """
    with _defect_cache_lock:
        _defect_cache[file_path] = result
        # FIFO-Trim
        if len(_defect_cache) > _DEFECT_CACHE_MAX:
            oldest = next(iter(_defect_cache))
            del _defect_cache[oldest]
    logger.debug("bridge: DefectScan cached for '%s'", file_path)


def get_cached_defect_result(file_path: str) -> Optional[object]:
    """Gibt einen gecachten DefectScanner-Befund zurück oder ``None``."""
    with _defect_cache_lock:
        return _defect_cache.get(file_path)


def clear_defect_cache(file_path: Optional[str] = None) -> None:
    """Löscht einen oder alle Einträge aus dem DefectScan-Cache."""
    with _defect_cache_lock:
        if file_path is not None:
            _defect_cache.pop(file_path, None)
        else:
            _defect_cache.clear()


# ---------------------------------------------------------------------------
# Lazy-Import-Wrappers  (Core-Module werden erst bei Bedarf geladen)
# ---------------------------------------------------------------------------


def get_quality_mode() -> type:
    """Gibt die ``QualityMode``-Enum zurück (lazy import)."""
    from backend.core.performance_guard import QualityMode  # type: ignore[import]
    return QualityMode


def get_medium_type_enum() -> type:
    """Gibt die ``MediumType``-Enum zurück (lazy import)."""
    from backend.core.enums import MediumType  # type: ignore[import]
    return MediumType


def get_processing_mode_enum() -> type:
    """Gibt die ``ProcessingMode``-Enum zurück (lazy import)."""
    from backend.core.enums import ProcessingMode  # type: ignore[import]
    return ProcessingMode


def get_restorer_classes() -> tuple[type, type]:
    """Gibt ``(RestorationConfig, UnifiedRestorerV3)`` zurück (lazy import)."""
    from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3  # type: ignore[import]
    return RestorationConfig, UnifiedRestorerV3


def get_aurik_denker_class() -> type:
    """Gibt ``AurikDenker``-Klasse zurück (lazy import, §2.2 Spec 08).

    Primary entry point for the full 8-stage restoration with carrier analysis,
    DefektDenker, MusikalischerGlobalplan, VERSA MOS scoring and ExzellenzDenker.
    Use this instead of UnifiedRestorerV3 for production pipelines.
    """
    from denker.aurik_denker import AurikDenker  # type: ignore[import]
    return AurikDenker


def get_aurik_denker_instance():
    """Gibt den thread-sicheren AurikDenker-Prozess-Singleton zurück (lazy, §2.2 Spec 08).

    Primary production accessor for BatchProcessingThread.
    Ensures Single-Orchestrator Ownership per process (No-Competing-Instances-Protokoll).
    Use ``get_aurik_denker_class()`` only for testing / mocking scenarios.
    """
    from denker.aurik_denker import get_aurik_denker  # type: ignore[import]
    return get_aurik_denker()


def get_defect_scanner() -> type:
    """Gibt die ``DefectScanner``-Klasse zurück (lazy import)."""
    from backend.core.defect_scanner import DefectScanner  # type: ignore[import]
    return DefectScanner


def get_audio_file_validator():
    """Gibt den ``AudioFileValidator``-Singleton zurück (lazy import, §10.5).

    Pflicht-Gate vor jedem ``_bg_load``-Thread-Start.  Wirf
    ``AudioLoadError`` (mit ``.message_user`` auf Deutsch) bei ungültiger Datei.
    """
    from backend.core.audio_file_validator import get_audio_file_validator as _get  # type: ignore[import]
    return _get()


def get_defect_type() -> type:
    """Gibt die ``DefectType``-Enum-Klasse zurück (lazy import).

    Wird von ``_defect_analysis_to_display`` und ``_result_scores_to_display``
    im Frontend benötigt, um DefectScanner-Scores zu indizieren.
    """
    from backend.core.defect_scanner import DefectType  # type: ignore[import]
    return DefectType


def get_medium_classifier_fn():
    """Gibt ``classify_medium``-Funktion zurück (lazy import, §2.5).

    Signatur: ``classify_medium(mono_audio: np.ndarray, sr: int) -> MediumResult``
    """
    from backend.core.medium_classifier import classify_medium  # type: ignore[import]
    return classify_medium


def get_era_classifier_fn():
    """Gibt ``classify_era``-Funktion zurück (lazy import, §2.4).

    Signatur: ``classify_era(audio: np.ndarray, sr: int) -> EraResult``
    """
    from backend.core.era_classifier import classify_era  # type: ignore[import]
    return classify_era


def get_genre_classifier_fn():
    """Gibt ``classify_genre``-Funktion zurück (lazy import).

    Signatur: ``classify_genre(audio: np.ndarray, sr: int) -> GenreResult``
    """
    from backend.core.genre_classifier import classify_genre  # type: ignore[import]
    return classify_genre


def get_restorability_estimator_class() -> type:
    """Gibt ``RestorabilityEstimator``-Klasse zurück (lazy import, §2.3).

    Verwendung: ``get_restorability_estimator_class()().estimate(audio, sr)``
    """
    from backend.core.restorability_estimator import RestorabilityEstimator  # type: ignore[import]
    return RestorabilityEstimator


def get_carrier_forensics_fn():
    """Gibt ``analyze_carrier_forensics``-Funktion zurück (lazy import).

    Signatur: ``analyze_carrier_forensics(mono: np.ndarray, sr: int) -> dict``
    Rückgabe-Keys: ``"carrier_forensic"`` (str), ``"score"`` (float).

    Intern wird ``classify_medium`` aus ``backend.core.medium_classifier``
    genutzt (``backend.carrier_forensics`` ist ein veralteter Shim).
    """
    from backend.core.medium_classifier import classify_medium as _cm  # type: ignore[import]

    def _analyze_carrier_forensics(mono: np.ndarray, sr: int) -> dict:
        result = _cm(mono, sr)
        return {"carrier_forensic": result.material_type, "score": float(result.confidence)}

    return _analyze_carrier_forensics


def get_audio_exporter_class() -> type:
    """Gibt ``AudioExporter``-Klasse zurück (lazy import).

    Fallback falls nicht verfügbar: ``None`` — Code muss ``sf.write`` nutzen.
    """
    from backend.core.audio_exporter import AudioExporter  # type: ignore[import]
    return AudioExporter


def get_lyrics_guided_enhancement_fn():
    """Gibt ``LyricsGuidedEnhancement``-Singleton zurück (lazy import, §2.36).

    Rückgabe: ``LyricsGuidedEnhancement``-Instanz mit ``.enhance(audio, sr)``
    und ``.get_timeline()``.

    Pflicht ab 9.10.x (§2.36): Wird im Frontend für L-Shortcut-Overlay und
    im BatchProcessingThread für ContentAwareProcessor-Integration verwendet.
    """
    from backend.core.lyrics_guided_enhancement import get_lyrics_guided_enhancement  # type: ignore[import]
    return get_lyrics_guided_enhancement()


def get_cleanup_after_file_fn():
    """Gibt ``cleanup_after_file``-Funktion zurück (lazy import)."""
    from backend.core.plugin_lifecycle_manager import cleanup_after_file  # type: ignore[import]
    return cleanup_after_file


def get_stem_remix_balancer_fn():
    """Gibt ``StemRemixBalancer.balance_remix``-Funktion zurück (lazy import, §1.4).

    Signatur: ``balance_remix(vocals, instruments, original, sr, vocal_weight) -> np.ndarray``
    Verwendet ITU-R BS.1770-5 K-gewichtete LUFS-Messung für Gain-Korrektur.
    LUFS-Differenz nach Re-Mix ≤ 0.3 LU gegenüber Original (§1.4 Spec).
    """
    from backend.core.stem_remix_balancer import StemRemixBalancer  # type: ignore[import]
    return StemRemixBalancer().balance_remix


def get_clipping_classifier():
    """Gibt ``ClippingClassifier``-Singleton zurück (lazy import, §6.3).

    Rückgabe: ``ClippingClassifier``-Instanz.
    Verwende ``classify_clipping(audio, sr)`` (Convenience-Funktion) für
    direkten Aufruf ohne Singleton-Handle.

    §6.3 CLIPPING vs SOFT_SATURATION: THD-basierte Diskriminierung.
    SOFT_SATURATION (gerade Harmonische — Röhre/Tape) → bewahren.
    CLIPPING (ungerade Harmonische + flat_tops > 0.1 %) → reparieren.
    """
    from backend.core.clipping_detection import get_clipping_classifier as _get  # type: ignore[import]
    return _get()


# ---------------------------------------------------------------------------
# Export-Guard  (PFLICHT vor jedem sf.write / AudioExporter.export)
# ---------------------------------------------------------------------------


def export_guard(audio: np.ndarray) -> np.ndarray:
    """Stellt sicher, dass Audio NaN/Inf-frei und auf [-1, 1] geclippt ist.

    Muss vor jedem ``sf.write()`` oder ``AudioExporter.export()`` aufgerufen
    werden. Entspricht der Numerischen Robustheit-Pflicht (§3.1 Spec 08).

    Args:
        audio: Audio-Array (float32 oder float64).

    Returns:
        Bereinigtes Audio (float32, kein NaN/Inf, Werte ∈ [-1, 1]).
    """
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = np.clip(audio, -1.0, 1.0)
    return audio


# ---------------------------------------------------------------------------
# Warmup  (Modell-Vorinitialisierung im Hintergrund, §9.7.4)
# ---------------------------------------------------------------------------


def warmup_models_background() -> None:
    """Initialisiert häufig genutzte ML-Modelle im Hintergrund vor.

    Kanonische Warmup-Funktion (§9.7.4). Wird 2 Sekunden nach App-Start
    als Daemon-Thread gestartet — aus ``ModernMainWindow.__init__`` via
    ``QTimer.singleShot(2000, ...)``. Fehler werden nur geloggt, kein Absturz.

    Plugin-Reihenfolge spiegelt §4.4-Priorisierung:
    Tier-1-Primär-Plugins zuerst, Fallbacks danach.
    """
    import importlib
    import time

    time.sleep(2)  # App-Fenster soll sichtbar sein bevor Last beginnt
    _plugins = [
        # Tier-1 Primär-Plugins (§9.7.4 — Pflicht-Vorwärmen)
        ("plugins.fcpe_plugin",                 "get_fcpe_plugin"),    # Pitch-Tracking Primär (§4.4)
        ("plugins.beats_plugin",                "get_beats_plugin"),   # Audio-Tagging Primär
        ("plugins.sgmse_plugin",                "get_sgmse_plugin"),   # Dereverb/Denoising Primär
        ("plugins.silero_plugin",               "get_silero_vad"),     # VAD (~1 MB, ultraschnell)
        ("backend.core.noise_reduction",        "get_noise_reducer"),  # DeepFilterNet v3.II Breitrauschen
        # Fallback-Plugins (nach Bedarf)
        ("plugins.panns_plugin",                "get_panns_plugin"),   # Audio-Tagging Fallback
        ("plugins.crepe_plugin",                "get_crepe_plugin"),   # Pitch-Tracking Fallback
        ("plugins.rmvpe_plugin",                "get_rmvpe_plugin"),   # Pitch-Tracking Backup
    ]
    logger.info("bridge: Warmup gestartet (%d Plugins) …", len(_plugins))
    for _mod, _accessor in _plugins:
        try:
            m = importlib.import_module(_mod)
            fn = getattr(m, _accessor, None)
            if fn is not None:
                fn()
                logger.debug("bridge: %s.%s vorgeladen", _mod.split(".")[-1], _accessor)
        except Exception as _e:
            logger.debug("bridge: %s.%s übersprungen: %s", _mod, _accessor, _e)
    logger.info("bridge: Warmup abgeschlossen")
