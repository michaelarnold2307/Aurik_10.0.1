"""
Aurik 9 — forensics/medium_detector.py  (§6.7, bindend ab v9.10.45)
=====================================================================
Tonträgerketten-Erkennung: bestimmt den vollständigen Degradationspfad
einer Aufnahme (z. B. cassette_tape → mp3_low) und liefert einen
MaterialType-Prior für den DefectScanner.

Pflicht-Spektralfingerabdruck (§6.7.1):
    1. Rolloff 95 %  — diagnostiziert Bandbreitenbegrenzung
    2. Wow/Flutter-Index — Pitch-Instabilität via pYIN-Ableitung
    3. HF-Energie > 16 kHz — MP3/Kassettenkette
    4. Rauschpegel (Percentile-5 PSD)  — Bandrauschen
    5. Effektive Bandbreite — physikalische Signalbandbreite

Kettenerkennung (§6.7.2):
    - Primär-Träger               = letzte Analogstufe
    - Sekundäre Stufen            = digital/komprimiert
    - is_multi_generation=True    → kombinierte Phasen beider Materialien
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class SpectralFingerprint:
    """Pflicht-Spektralfingerabdruck (§6.7.1) aus Rohsignal-Vorabanalyse."""

    rolloff_95_hz: float = 0.0  # Spectral Rolloff 95 % — Median
    wow_flutter_index: float = 0.0  # Pitch-Varianz [Hz std] über 100-ms-Fenster
    hf_energy_above_16k: float = 0.0  # Anteil Energie > 16 kHz an Gesamt
    noise_floor_db: float = -60.0  # 5. Perzentil der Frame-Energien [dBFS]
    effective_bandwidth_hz: float = 0.0  # HF-Rolloff −60 dBFS

    # --- Alias-Properties für Test-Kompatibilität (§6.7.1) ---
    @property
    def rolloff_95_percent_hz(self) -> float:
        """Alias für rolloff_95_hz — Rückwärtskompatibilität."""
        return self.rolloff_95_hz

    @property
    def hf_energy_above_16khz_percent(self) -> float:
        """Alias für hf_energy_above_16k als Prozentwert (0–100)."""
        return float(self.hf_energy_above_16k * 100.0)

    def __contains__(self, item: object) -> bool:
        """Unterstützt 'key in fingerprint'-Syntax für Tests."""
        return item in (
            "rolloff_95_hz", "rolloff_95_percent_hz",
            "wow_flutter_index",
            "hf_energy_above_16k", "hf_energy_above_16khz_percent",
            "noise_floor_db", "effective_bandwidth_hz",
        )

    def as_dict(self) -> dict:
        return {
            "rolloff_95_hz": self.rolloff_95_hz,
            "rolloff_95_percent_hz": self.rolloff_95_hz,
            "wow_flutter_index": self.wow_flutter_index,
            "hf_energy_above_16k_fraction": self.hf_energy_above_16k,
            "hf_energy_above_16khz_percent": self.hf_energy_above_16k * 100.0,
            "noise_floor_db": self.noise_floor_db,
            "effective_bandwidth_hz": self.effective_bandwidth_hz,
        }


@dataclass
class TransferChain:
    """Erkannte Medien-Transferkette."""

    chain: list[str] = field(default_factory=list)
    """Kette von MediaType-Strings, z. B. ['tape', 'mp3_low']."""

    is_multi_generation: bool = False
    """True wenn ≥ 2 verschiedene Medienstufen erkannt wurden."""

    primary_material: str = "unknown"
    """Letzter analoger Träger = primärer MaterialType-Prior."""

    confidence: float = 0.0
    """Gesamtkonfidenz der Ketten-Schätzung ∈ [0, 1]."""

    reasoning: str = ""

    def __len__(self) -> int:
        return len(self.chain)


@dataclass
class MediumDetectionResult:
    """Vollständiges Ergebnis der Tonträgerketten-Erkennung."""

    transfer_chain: list[str]
    """Kette wie ['tape', 'mp3_low'] — primärer Träger zuerst."""

    is_multi_generation: bool
    primary_material: str
    confidence: float
    spectral_fingerprint: SpectralFingerprint
    evidence: list[str] = field(default_factory=list)
    """Laienverständliche Diagnose-Begründungen."""
    medium_confidences: list[float] = field(default_factory=list)
    """Per-Link-Konfidenz — gleiche Länge wie transfer_chain."""

    @property
    def chain_label(self) -> str:
        return " → ".join(self.transfer_chain) if self.transfer_chain else "unknown"

    def as_dict(self) -> dict:
        return {
            "transfer_chain": self.transfer_chain,
            "medium_confidences": self.medium_confidences,
            "is_multi_generation": self.is_multi_generation,
            "primary_material": self.primary_material,
            "confidence": self.confidence,
            "chain_label": self.chain_label,
            "spectral_fingerprint": self.spectral_fingerprint.as_dict(),
            "evidence": self.evidence,
        }


# ---------------------------------------------------------------------------
# Haupt-Klasse
# ---------------------------------------------------------------------------


class MediumDetector:
    """Erkennt Tonträgerketten forensisch (§6.7).

    Laufreihenfolge je Import:
        1.  Pflicht-Spektralfingerabdruck (5 Merkmale)
        2a. Kassetten-Erkennung  (Rolloff, Wow/Flutter, eff. BW < 14.5 kHz, Rauschpegel)
        2b. Shellac/Schellack    (Rolloff ≤ 4 kHz + eff. BW ≤ 7 kHz + Rauschboden > −40 dBFS)
        2c. MP3/Codec-Kette      (HF-Anteil 0 %, Frequenz-Kerbmuster)
        2d. Digitaler Träger     (Rolloff > 18 kHz, niedriger Rauschboden)
        3.  Kettenzusammenführung (primär + sekundär)

    Singleton-Zugang: ``get_medium_detector()``
    Convenience:      ``detect_medium_chain(audio, sr)``
    """

    # ── Diagnostik-Schwellen (§6.7.1) ──────────────────────────────────
    SHELLAC_ROLLOFF_MAX_HZ: float = 4_500.0
    # Shellac requires narrow ACTUAL bandwidth AND high noise floor (not just rolloff_95).
    # rolloff_95 alone triggers false positives for bass-heavy digital music where 95 % of
    # spectral energy lies below 4.5 kHz even in a modern MP3.
    SHELLAC_EFFECTIVE_BW_MAX_HZ: float = 7_000.0   # real shellac: physical BW ≤ 7 kHz
    SHELLAC_NOISE_FLOOR_MIN_DB: float = -40.0       # shellac is always very noisy (> −40 dBFS)
    TAPE_ROLLOFF_MAX_HZ: float = 10_000.0
    # Calibrated to current wow/flutter proxy scale: real tape-digitized music often lands
    # around 0.03–0.08 in this metric. 0.4 was too strict and suppressed tape→mp3 detection.
    TAPE_SPEED_VARIATION_MIN: float = 0.02  # Hz std
    TAPE_NOISE_FLOOR_MAX_DB: float = -36.0  # lauter = Bandrauschen
    TAPE_EFFECTIVE_BW_MAX_HZ: float = 14_500.0      # tape: BW ≤ ~14 kHz; MP3 1990s: ≥ 15 kHz
    TAPE_NOISE_FLOOR_MIN_DB: float = -52.0           # tape always has some analog noise
    TAPE_WOW_FLUTTER_MAX: float = 25.0               # reject unrealistically unstable pseudo-wow values
    # Tape-digitised recordings have physical BW ≤ 10–11 kHz.  A music track's 5th-percentile
    # frame energy is often > -45 dBFS even in a clean MP3 (quiet musical passages ≠ tape hiss).
    # Guard against this false "tape+mp3" inference by also requiring narrow actual bandwidth.
    TAPE_DIGITAL_BW_MAX_HZ: float = 13_500.0   # raised: 1970s cassette tape up to ~13.5 kHz         # tape+mp3 heuristic: eff_bw must be ≤ 11 kHz
    HF_ENERGY_THRESHOLD_FRACTION: float = 0.001  # < 0.1 % → kein HF
    MP3_KERBMUSTER_THZ: float = 16_000.0  # typischer MP3-Rolloff

    @staticmethod
    def _is_benign_codec_source(audio: np.ndarray, sr: int, fp: SpectralFingerprint) -> bool:
        """Heuristic guard to prevent false analog-chain inference on clean digital sources."""
        mono = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if mono.ndim == 2:
            mono = mono.mean(axis=0) if mono.shape[0] <= mono.shape[1] else mono.mean(axis=1)
        if mono.size < 4096 or sr <= 0:
            return False

        abs_mono = np.abs(mono)
        hard_clip_ratio = float(np.mean(abs_mono >= 0.999))
        near_clip_ratio = float(np.mean(abs_mono >= 0.98))

        dyn_window = max(1, int(sr * 0.4))
        dyn_frames = mono.size // dyn_window
        if dyn_frames >= 2:
            dyn_blocks = mono[: dyn_frames * dyn_window].reshape(dyn_frames, dyn_window)
            dyn_rms = np.sqrt(np.mean(dyn_blocks**2, axis=1) + 1e-12)
            dyn_std_db = float(np.std(20.0 * np.log10(dyn_rms + 1e-12)))
        else:
            dyn_std_db = 0.0

        n_fft = 4096
        hop = 1024
        window = np.hanning(n_fft).astype(np.float32)
        flatness_values: list[float] = []
        for start in range(0, mono.size - n_fft + 1, hop):
            frame = mono[start : start + n_fft] * window
            mag = np.abs(np.fft.rfft(frame)).astype(np.float64) + 1e-12
            flatness_values.append(float(np.exp(np.mean(np.log(mag))) / np.mean(mag)))

        if not flatness_values:
            return False

        flatness_median = float(np.median(flatness_values))
        return (
            hard_clip_ratio <= 1e-5
            and near_clip_ratio <= 1e-4
            and dyn_std_db >= 3.5
            and flatness_median <= 1e-2
            and fp.noise_floor_db <= -38.0
            and fp.wow_flutter_index < 0.02
            and fp.effective_bandwidth_hz >= 12_000.0
        )

    def _compute_fingerprint(self, audio: np.ndarray, sr: int) -> SpectralFingerprint:
        """Berechnet den Pflicht-Spektralfingerabdruck (§6.7.1).

        NaN/Inf-sicher; alle Felder werden immer befüllt.
        """
        mono = self._to_mono(audio)
        n = len(mono)
        if n == 0:
            return SpectralFingerprint()

        hop = max(1, n // 200)
        win = min(2048, n)

        # ── 1. Rolloff 95 % ────────────────────────────────────────────
        try:
            frames = [mono[i : i + win] for i in range(0, n - win, hop)]
            rolloffs = []
            for frame in frames[:100]:
                spec = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
                freqs = np.fft.rfftfreq(len(frame), 1.0 / sr)
                cum = np.cumsum(spec**2)
                total = cum[-1]
                if total > 0:
                    idx = int(np.searchsorted(cum, 0.95 * total))
                    rolloffs.append(float(freqs[min(idx, len(freqs) - 1)]))
            rolloff_95 = float(np.median(rolloffs)) if rolloffs else 0.0
        except Exception:
            rolloff_95 = 0.0

        # ── 2. Wow/Flutter-Index ────────────────────────────────────────
        try:
            from scipy.signal import hilbert

            frame_size = int(0.1 * sr)  # 100 ms
            pitches = []
            for start in range(0, n - frame_size, frame_size):
                frame = mono[start : start + frame_size].astype(np.float64)
                analytic: np.ndarray = hilbert(frame)  # type: ignore[assignment]  # scipy stub returns Dispatchable
                env = np.abs(analytic)
                mean_e = float(np.mean(env))
                if mean_e > 1e-6:
                    pitches.append(mean_e)
            wow_flutter = float(np.std(np.diff(pitches))) if len(pitches) > 2 else 0.0
        except Exception:
            wow_flutter = 0.0

        # ── 3. HF-Energie > 16 kHz ─────────────────────────────────────
        try:
            spec_full = np.abs(np.fft.rfft(mono[: min(n, 65536)], n=65536))
            freqs_full = np.fft.rfftfreq(65536, 1.0 / sr)
            mask_hf = freqs_full > 16_000
            total_e = float(np.sum(spec_full**2))
            hf_e = float(np.sum(spec_full[mask_hf] ** 2))
            hf_fraction = hf_e / max(total_e, 1e-12)
        except Exception:
            hf_fraction = 0.0

        # ── 4. Rauschpegel (5. Perzentil PSD) ──────────────────────────
        try:
            frame_energies = []
            for start in range(0, n - win, hop):
                e = float(np.mean(mono[start : start + win] ** 2))
                if e > 0:
                    frame_energies.append(10 * math.log10(e))
            noise_floor = float(np.percentile(frame_energies, 5)) if frame_energies else -60.0
            noise_floor = max(-120.0, min(0.0, noise_floor))
        except Exception:
            noise_floor = -60.0

        # ── 5. Effektive Bandbreite (Rolloff −60 dBFS) ──────────────────
        try:
            spec_bw = np.abs(np.fft.rfft(mono[: min(n, 65536)], n=65536))
            freqs_bw = np.fft.rfftfreq(65536, 1.0 / sr)
            spec_db = 20 * np.log10(np.clip(spec_bw / max(spec_bw.max(), 1e-12), 1e-15, np.inf))
            above_thresh = freqs_bw[spec_db > -60.0]
            eff_bw = float(above_thresh.max()) if len(above_thresh) > 0 else 0.0
        except Exception:
            eff_bw = 0.0

        return SpectralFingerprint(
            rolloff_95_hz=float(np.nan_to_num(rolloff_95)),
            wow_flutter_index=float(np.nan_to_num(wow_flutter)),
            hf_energy_above_16k=float(np.nan_to_num(hf_fraction)),
            noise_floor_db=float(np.nan_to_num(noise_floor, nan=-60.0)),
            effective_bandwidth_hz=float(np.nan_to_num(eff_bw)),
        )

    def detect(self, audio: np.ndarray, sr: int) -> MediumDetectionResult:
        """Erkennt die Tonträgerkette forensisch.

        Laufreihenfolge MUSS VOR classify_medium() sein (§6.7.2).

        Returns:
            MediumDetectionResult mit transfer_chain, is_multi_generation,
            primary_material, confidence, spectral_fingerprint.
        """
        if sr != 48000:
            logger.debug("MediumDetector: SR=%d (erwartet 48000), arbeite trotzdem weiter", sr)

        fp = self._compute_fingerprint(audio, sr)
        benign_codec_source = self._is_benign_codec_source(audio, sr, fp)
        chain: list[str] = []
        evidence: list[str] = []
        confidence_parts: list[float] = []
        # chain_confidences tracks per-link confidence (same length as chain).
        # Bonus confidence increments (e.g. tape noise) are folded into the last entry.
        chain_confidences: list[float] = []

        # ── Shellac/Wachswalze (extremste Bandbreitenbegrenzung) ─────────
        # rolloff_95 alone is misleading: bass-heavy digital music can have 95 % of spectral
        # energy below 4.5 kHz. We require BOTH narrow effective bandwidth AND high noise floor.
        if (
            not benign_codec_source
            and
            fp.rolloff_95_hz < self.SHELLAC_ROLLOFF_MAX_HZ
            and fp.rolloff_95_hz > 0
            and fp.effective_bandwidth_hz < self.SHELLAC_EFFECTIVE_BW_MAX_HZ
            and fp.noise_floor_db > self.SHELLAC_NOISE_FLOOR_MIN_DB
        ):
            chain.append("shellac")
            confidence_parts.append(0.80)
            chain_confidences.append(0.80)
            evidence.append(
                f"Shellac-Signatur: Rolloff {fp.rolloff_95_hz:.0f} Hz, "
                f"eff. BW {fp.effective_bandwidth_hz:.0f} Hz, "
                f"Rauschboden {fp.noise_floor_db:.1f} dBFS"
            )

        # ── Kassetten-Magnetband (Tape) ───────────────────────────────────
        # Guard against 1990s/2000s MP3 files: digital files have effective BW ≥ 15 kHz
        # and a very quiet noise floor. Tape has limited BW and always carries analog noise.
        elif (
            not benign_codec_source
            and
            fp.rolloff_95_hz < self.TAPE_ROLLOFF_MAX_HZ
            and fp.wow_flutter_index > self.TAPE_SPEED_VARIATION_MIN
            and fp.effective_bandwidth_hz < self.TAPE_EFFECTIVE_BW_MAX_HZ
            and fp.noise_floor_db > self.TAPE_NOISE_FLOOR_MIN_DB
        ):
            chain.append("tape")
            _tape_conf = 0.75
            confidence_parts.append(_tape_conf)
            chain_confidences.append(_tape_conf)
            evidence.append(
                f"Kassetten-Signatur: Rolloff {fp.rolloff_95_hz:.0f} Hz, "
                f"Wow/Flutter {fp.wow_flutter_index:.2f}, "
                f"eff. BW {fp.effective_bandwidth_hz:.0f} Hz"
            )
            if fp.noise_floor_db > self.TAPE_NOISE_FLOOR_MAX_DB:
                evidence.append(f"Starkes Bandrauschen ({fp.noise_floor_db:.1f} dBFS)")
                confidence_parts.append(0.10)
                # Fold bonus into the existing tape entry (no new chain link)
                chain_confidences[-1] = min(1.0, chain_confidences[-1] + 0.10)

        # ── Vinyl (Crackle-Profil, mittlerer Rolloff, niedriger Rauschboden) ─
        elif (
            not benign_codec_source
            and
            self.TAPE_ROLLOFF_MAX_HZ <= fp.rolloff_95_hz < 18_000
            and fp.wow_flutter_index < 0.3
            and fp.noise_floor_db < -38.0
        ):
            chain.append("vinyl")
            confidence_parts.append(0.65)
            chain_confidences.append(0.65)
            evidence.append(f"Vinyl-Profil: Rolloff {fp.rolloff_95_hz:.0f} Hz, ruhiger Rauschboden")

        # ── Digitaler Träger (CD/WAV) ────────────────────────────────────
        elif fp.rolloff_95_hz >= 18_000 and fp.hf_energy_above_16k > 0.01:
            chain.append("cd_digital")
            confidence_parts.append(0.70)
            chain_confidences.append(0.70)
            evidence.append(f"Digitaler Träger: HF-Energie vorhanden ({fp.hf_energy_above_16k*100:.1f} %)")

        # ── Sekundäre MP3/Codec-Kette erkennen ──────────────────────────
        has_mp3_signature = (
            fp.hf_energy_above_16k < self.HF_ENERGY_THRESHOLD_FRACTION and fp.effective_bandwidth_hz < 17_500
        )

        if has_mp3_signature:
            # Wenn kein primärer Träger erkannt → mp3 ist primär
            if not chain:
                # Kassette+MP3-Kette: hohes Bandrauschen UND schmale Bandbreite.
                # noise_floor_db > -45 dBFS allein reicht nicht: bei musikvollem Material
                # liegen die leisen Frames (5. Perzentil) oft bei -30 bis -35 dBFS, obwohl
                # kein Bandrauschen vorliegt. Nur echte Kassetten-Digitalisierungen haben
                # zusätzlich eff_bw ≤ 11 kHz (physikalische Bandbreite der Kassette).
                tape_plus_codec = (
                    not benign_codec_source
                    and fp.noise_floor_db > -45.0
                    and fp.effective_bandwidth_hz < self.TAPE_DIGITAL_BW_MAX_HZ
                    and fp.rolloff_95_hz < self.TAPE_ROLLOFF_MAX_HZ
                    and self.TAPE_SPEED_VARIATION_MIN < fp.wow_flutter_index < self.TAPE_WOW_FLUTTER_MAX
                )
                if tape_plus_codec:
                    chain = ["tape", "mp3_low"]
                    confidence_parts.extend([0.55, 0.20])
                    chain_confidences = [0.55, 0.20]
                    evidence.append(
                        f"Kassette+MP3-Kette: kein HF ({fp.hf_energy_above_16k*100:.2f} %), "
                        f"Rauschboden {fp.noise_floor_db:.1f} dBFS, "
                        f"eff. BW {fp.effective_bandwidth_hz:.0f} Hz"
                    )
                else:
                    bitrate_estimate = "mp3_low" if fp.effective_bandwidth_hz < 14_000 else "mp3_high"
                    chain = [bitrate_estimate]
                    confidence_parts.append(0.60)
                    chain_confidences = [0.60]
                    evidence.append(f"MP3-Kette: kein HF, Bandbreite {fp.effective_bandwidth_hz:.0f} Hz")
            else:
                # Sekundäre Codec-Stufe
                bitrate = "mp3_low" if fp.effective_bandwidth_hz < 14_000 else "mp3_high"
                chain.append(bitrate)
                confidence_parts.append(0.20)
                chain_confidences.append(0.20)
                evidence.append(f"Sekundäre MP3-Kodierung erkannt (BW {fp.effective_bandwidth_hz:.0f} Hz)")

        # ── Fallback ─────────────────────────────────────────────────────
        if not chain:
            chain = ["unknown"]
            confidence_parts = [0.30]
            chain_confidences = [0.30]
            evidence.append("Träger unbekannt — Standard-Prior wird verwendet")

        primary = chain[0]
        is_multi = len(chain) > 1
        confidence = float(np.clip(sum(confidence_parts), 0.0, 1.0))

        logger.info(
            "MediumDetector: Kette=%s, primär=%s, multi=%s, Konfidenz=%.2f",
            " → ".join(chain),
            primary,
            is_multi,
            confidence,
        )

        return MediumDetectionResult(
            transfer_chain=chain,
            is_multi_generation=is_multi,
            primary_material=primary,
            confidence=confidence,
            spectral_fingerprint=fp,
            evidence=evidence,
            medium_confidences=chain_confidences,
        )

    # ── Hilfsmethode ─────────────────────────────────────────────────────

    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        """Wandelt beliebiges Audio in mono float32 um."""
        if audio.ndim == 2:
            audio = audio.mean(axis=0) if audio.shape[0] <= audio.shape[1] else audio.mean(axis=1)
        mono = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(mono, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Singleton + Convenience
# ---------------------------------------------------------------------------

_instance: Optional[MediumDetector] = None
_lock = threading.Lock()


def get_medium_detector() -> MediumDetector:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking, §3.2)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MediumDetector()
    return _instance


def detect_medium_chain(audio: np.ndarray, sr: int) -> MediumDetectionResult:
    """Convenience-Wrapper: erkennt die Tonträgerkette eines Audio-Signals."""
    return get_medium_detector().detect(audio, sr)
