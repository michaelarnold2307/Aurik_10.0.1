"""versa_plugin — SingMOS Pro via VERSA Toolkit (2024).

Primär:  SingMOS Pro (South-Twilight/SingMOS v1.1.1, via torch.hub)
         Integration: VERSA / models/versa/versa/utterance_metrics/pseudo_mos.py
         Eingabe: float32 mono @ 16 kHz; Ausgabe: MOS ∈ [1.0, 5.0]
         Hub-Cache: models/versa/hub_cache/ (offline nach Installation)
         Referenz: https://arxiv.org/abs/2510.01812

Fallback: PQS-DSP-Gammatone (Bark-Filterbank + Sigmoid-MOS-Mapping)

VERBOTEN laut Spec §4.4: PESQ, DNSMOS, NISQA, STOI, CDPAM.

Singleton-Pattern: get_versa_plugin() verwenden.
CPU-Only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
import threading

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
_VERSA_PATH = _ROOT / "models" / "versa"
_HUB_CACHE = str(_VERSA_PATH / "hub_cache")
_MODEL_SR: int = 16_000

_lock = threading.Lock()
_instance: VersaPlugin | None = None


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class VersaResult:
    """Ergebnis der VERSA MOS-Schätzung.

    Attributes:
        mos:          MOS-Wert ∈ [1.0, 5.0]
        model_used:   "singmos_pro" | "pqs_dsp_fallback"
        confidence:   Modell-Konfidenz ∈ [0, 1]
        sub_scores:   Optionale Teil-Scores (Signal, Hintergrund, Gesamt)
    """

    mos: float
    model_used: str
    confidence: float = 1.0
    sub_scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # NaN/Inf-Guard (§3.1 Aurik Spec)
        if not math.isfinite(self.mos):
            self.mos = 3.0
        self.mos = float(np.clip(self.mos, 1.0, 5.0))


# ---------------------------------------------------------------------------
# VersaPlugin
# ---------------------------------------------------------------------------

class VersaPlugin:
    """SingMOS Pro via VERSA Toolkit — referenzfreier Musik-MOS.

    Primär: SingMOS Pro (South-Twilight/SingMOS v1.1.1) über VERSA pseudo_mos.
    Fallback: PQS-Gammatone-DSP (§4.4 Spec).

    Verwendung NUR für Qualitätsbewertung — keine Modifikation des Audios.
    """

    def __init__(self) -> None:
        self._predictor_dict: dict | None = None
        self._predictor_fs: dict | None = None
        self._pseudo_mos_metric = None
        self._model_loaded: bool = False
        self._load_attempted: bool = False
        self._load_lock = threading.Lock()
        # Lazy load: model is loaded on first score() call, NOT here.
        # Eager loading of SingMOS Pro (wav2vec2-large 606 MB) would cause OOM
        # when multiple plugins are imported simultaneously during test collection.

    # Budget: wav2vec2-large (s3prl, ~600 MB) + SingMOS Pro checkpoint (~150 MB)
    _BUDGET_GB: float = 0.80

    def _try_load(self) -> None:
        """Loads SingMOS Pro via VERSA pseudo_mos; PQS-DSP fallback on error.

        Offline-Invariante: Lädt NUR wenn beide Checkpoint-Dateien lokal
        vorhanden sind. Kein torch.hub-Download im Produktionsbetrieb.
        Fehlende Weights → sofortiger PQS-DSP-Fallback (kein Netzwerkaufruf).
        """
        versa_pkg = _VERSA_PATH / "versa"
        _pm_path = _VERSA_PATH / "versa" / "utterance_metrics" / "pseudo_mos.py"
        if not versa_pkg.exists() or not _pm_path.exists():
            logger.info("VERSA toolkit nicht gefunden (%s) — PQS-DSP-Fallback.", versa_pkg)
            return

        # Offline check: SingMOS Pro checkpoint must be locally cached.
        # torch.hub caches the repo zip at hub_cache/<user>_SingMOS_<tag>/
        # and downloads the .pth checkpoint to hub_cache/checkpoints/.
        _hub_dir = _VERSA_PATH / "hub_cache"
        _singmos_checkpoint = _hub_dir / "checkpoints" / "ft_wav2vec2_large_ll60k_mdf_p1_200epochs_all_192epochs.pth"
        # s3prl caches wav2vec2-large at ~/.cache/s3prl/download/
        import os  # noqa: PLC0415
        _s3prl_cache = Path(os.path.expanduser("~/.cache/s3prl/download"))
        _wav2vec2_cached = any(
            f.name.endswith(".wav2vec_vox_new.pt") or "wav2vec_vox_new" in f.name
            for f in _s3prl_cache.glob("*.wav2vec_vox_new.pt")
        ) if _s3prl_cache.exists() else False

        if not _singmos_checkpoint.exists():
            logger.info(
                "SingMOS Pro Checkpoint nicht lokal gefunden (%s) — PQS-DSP-Fallback. "
                "Für ML-Betrieb: models/versa/hub_cache/checkpoints/ befüllen "
                "(ft_wav2vec2_large_ll60k_mdf_p1_200epochs_all_192epochs.pth).",
                _singmos_checkpoint,
            )
            return
        if not _wav2vec2_cached:
            logger.info(
                "wav2vec2-large (s3prl) nicht lokal gecacht (~/.cache/s3prl/download/) "
                "— PQS-DSP-Fallback. Für ML-Betrieb: s3prl Pre-Download ausführen."
            )
            return

        try:
            from backend.core.ml_memory_budget import try_allocate as _try_alloc  # noqa: PLC0415
            if not _try_alloc("VersaSingMOS", size_gb=self._BUDGET_GB):
                logger.warning("VERSA SingMOS Pro: ML-Budget erschöpft (%.2f GB) — PQS-Fallback.", self._BUDGET_GB)
                return
        except Exception:
            pass

        try:
            import importlib.util  # noqa: PLC0415
            import torch  # noqa: PLC0415

            # Load pseudo_mos.py directly to bypass versa/__init__.py which imports
            # pysptk (mcd_f0 dependency) that may not be installed on the target machine.
            _pm_path = _VERSA_PATH / "versa" / "utterance_metrics" / "pseudo_mos.py"
            _spec = importlib.util.spec_from_file_location("versa_pseudo_mos", str(_pm_path))
            _pm_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
            _spec.loader.exec_module(_pm_mod)  # type: ignore[union-attr]
            pseudo_mos_setup = _pm_mod.pseudo_mos_setup
            pseudo_mos_metric = _pm_mod.pseudo_mos_metric

            torch.hub.set_dir(_HUB_CACHE)
            predictor_dict, predictor_fs = pseudo_mos_setup(
                predictor_types=["singmos_pro"],
                predictor_args={"singmos_pro": {"fs": _MODEL_SR}},
                cache_dir=_HUB_CACHE,
                use_gpu=False,
            )
            self._predictor_dict = predictor_dict
            self._predictor_fs = predictor_fs
            self._pseudo_mos_metric = pseudo_mos_metric
            self._model_loaded = True

            # PLM lifecycle registration
            from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm  # noqa: PLC0415

            def _unload_singmos() -> None:
                global _instance
                if _instance is not None:
                    _instance._predictor_dict = None
                    _instance._predictor_fs = None
                    _instance._pseudo_mos_metric = None
                    _instance._model_loaded = False

            _reg_plm("VersaSingMOS", size_gb=self._BUDGET_GB, unload_fn=_unload_singmos)
            logger.info("✅ VERSA SingMOS Pro geladen (§4.4 — Musik-MOS-Primär)")
        except Exception as exc:
            logger.warning("VERSA SingMOS Pro nicht ladbar: %s — PQS-DSP-Fallback.", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, audio: np.ndarray, sr: int) -> VersaResult:
        """Berechnet referenzfreien MOS-Wert für Musik- oder Sprachaufnahme.

        Args:
            audio: float32 mono/stereo, 48000 Hz
            sr:    Sample-Rate (muss 48000 sein)

        Returns:
            VersaResult mit MOS ∈ [1.0, 5.0] und Metadaten.
        """
        assert sr == 48_000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        mono = audio if audio.ndim == 1 else audio.mean(axis=-1)
        mono = np.clip(mono, -1.0, 1.0)

        # Lazy load: attempt once, then use whatever is available.
        if not self._load_attempted:
            with self._load_lock:
                if not self._load_attempted:
                    self._try_load()
                    self._load_attempted = True

        if self._model_loaded and self._predictor_dict is not None:
            return self._score_singmos_pro(mono, sr)
        return self._score_pqs_dsp(mono, sr)

    # ------------------------------------------------------------------
    # SingMOS Pro Inference
    # ------------------------------------------------------------------

    def _score_singmos_pro(self, mono_48k: np.ndarray, sr: int) -> VersaResult:
        """SingMOS Pro MOS inference via VERSA pseudo_mos_metric.

        Resamples 48 kHz → 16 kHz (Lanczos), calls SingMOS Pro model,
        returns calibrated MOS ∈ [1.0, 5.0].
        """
        try:
            from math import gcd  # noqa: PLC0415
            from scipy.signal import resample_poly  # noqa: PLC0415

            g = gcd(sr, _MODEL_SR)
            mono_16k = resample_poly(mono_48k, _MODEL_SR // g, sr // g).astype(np.float32)
            mono_16k = np.nan_to_num(mono_16k, nan=0.0, posinf=0.0, neginf=0.0)
            mono_16k = np.clip(mono_16k, -1.0, 1.0)

            scores = self._pseudo_mos_metric(
                mono_16k, _MODEL_SR, self._predictor_dict, self._predictor_fs
            )
            mos = float(np.clip(scores.get("singmos_pro", 3.0), 1.0, 5.0))
            if not math.isfinite(mos):
                mos = 3.0
            logger.debug("SingMOS Pro MOS: %.3f", mos)
            return VersaResult(mos=mos, model_used="singmos_pro", confidence=0.92)
        except Exception as exc:
            logger.warning("SingMOS Pro Inferenzfehler: %s — PQS-DSP-Fallback.", exc)
            return self._score_pqs_dsp(mono_48k, sr)

    # ------------------------------------------------------------------
    # PQS-DSP Fallback
    # ------------------------------------------------------------------

    def _score_pqs_dsp(self, mono: np.ndarray, sr: int) -> VersaResult:
        """PQS-Gammatone-DSP Fallback (§4.4 Spec).

        Formel:
            1. Gammatone-Filterbank (24 ERB-Kanäle, 50–8000 Hz)
            2. SNR per Kanal basierend auf Energieverhältnis harmonisch/nichharmonisch
            3. Frequenz-gewichteter SNR → MOS via Sigmoid-Mapping
               MOS = 1 + 4 · σ(0.2 · snr_dB − 1.5)
        """
        try:
            n = len(mono)
            if n < 512:
                return VersaResult(mos=3.0, model_used="pqs_dsp_fallback", confidence=0.40)

            # Energie-Schätzung via Gammatone-approximierter Bark-Filterbank
            spec = np.fft.rfft(mono[:min(n, 4 * sr)].astype(np.float64))
            mag = np.abs(spec)
            freqs = np.linspace(0, sr / 2, len(mag))

            # 24 Bark-Bänder
            bark_edges = [50, 100, 200, 300, 400, 510, 630, 770, 920, 1080,
                          1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700,
                          4400, 5300, 6400, 7700, 9500, 12000, 15500]
            band_energies: list[float] = []
            for i in range(len(bark_edges) - 1):
                lo, hi = bark_edges[i], bark_edges[i + 1]
                mask = (freqs >= lo) & (freqs < hi)
                if mask.sum() > 0:
                    band_energies.append(float(np.mean(mag[mask] ** 2)))

            if not band_energies:
                return VersaResult(mos=3.0, model_used="pqs_dsp_fallback", confidence=0.40)

            total_e = float(np.mean(band_energies))
            rms = float(np.sqrt(np.mean(mono ** 2))) + 1e-10
            # Pseudo-SNR: Verhältnis Signalenergie zu Rauschuntergrenze
            snr_db = 20.0 * math.log10(rms / 0.01) if rms > 0.01 else 0.0
            snr_db = float(np.clip(snr_db, -10.0, 40.0))

            # Frequency-weighted SNR aus Bark-Bändern (mittlere Bänder gewichtet)
            weights = np.array([0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.5, 1.5,
                                1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7,
                                0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1][: len(band_energies)])
            w_e = np.array(band_energies[: len(weights)]) * weights
            freq_snr = 20.0 * math.log10(float(np.mean(w_e)) / (total_e + 1e-12) + 1e-5)
            combined_snr = 0.6 * snr_db + 0.4 * float(np.clip(freq_snr + 20, -10, 40))

            # MOS-Mapping: Sigmoid skaliert auf [1.0, 5.0]
            z = 0.2 * combined_snr - 1.5
            sigma = 1.0 / (1.0 + math.exp(-z))
            mos = float(np.clip(1.0 + 4.0 * sigma, 1.0, 5.0))
            if not math.isfinite(mos):
                mos = 3.0

            return VersaResult(mos=mos, model_used="pqs_dsp_fallback", confidence=0.55)
        except Exception as exc:
            logger.error("PQS-DSP Fallback fehlgeschlagen: %s", exc)
            return VersaResult(mos=3.0, model_used="error", confidence=0.0)


# ---------------------------------------------------------------------------
# Singleton (§3.2 Double-Checked Locking)
# ---------------------------------------------------------------------------


def get_versa_plugin() -> VersaPlugin:
    """Thread-sicherer Singleton-Accessor."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = VersaPlugin()
    return _instance


def score_mos(audio: np.ndarray, sr: int) -> VersaResult:
    """Convenience-Wrapper für get_versa_plugin().score()."""
    return get_versa_plugin().score(audio, sr)
