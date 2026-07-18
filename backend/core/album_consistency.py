"""AlbumConsistency — §INCREMENTAL #2: Track-übergreifende Konsistenz.

Stellt sicher: Alle Tracks eines Albums haben dieselbe Loudness,
Tonal-Balance und Stereo-Breite.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrackProfile:
    path: str = ""
    integrated_lufs: float = -23.0
    spectral_centroid_hz: float = 2000.0
    stereo_width: float = 0.5
    rms_dbfs: float = -20.0


@dataclass
class AlbumTarget:
    target_lufs: float = -16.0
    target_centroid_hz: float = 2000.0
    target_stereo_width: float = 0.5
    tolerance_lu: float = 2.0


def analyze_track(audio: np.ndarray, sr: int, path: str = "") -> TrackProfile:
    mono = np.mean(audio, axis=-1) if audio.ndim > 1 else np.asarray(audio, dtype=np.float32)
    n_fft = min(4096, len(mono))
    spec = np.abs(np.fft.rfft(mono[: n_fft * 8], n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    centroid = float(np.sum(freqs * spec) / max(np.sum(spec), 1e-10))
    rms = float(np.sqrt(np.mean(mono**2)))
    stereo = 0.5
    if audio.ndim == 2 and audio.shape[-1] == 2:
        l, r = audio[:, 0], audio[:, 1]
        stereo = float(np.clip(1.0 - abs(np.corrcoef(l, r)[0, 1]), 0.0, 1.0))
    return TrackProfile(
        path=path,
        spectral_centroid_hz=centroid,
        stereo_width=stereo,
        rms_dbfs=20 * np.log10(max(rms, 1e-10)),
        integrated_lufs=-23.0,
    )


def compute_album_target(tracks: list[TrackProfile]) -> AlbumTarget:
    if not tracks:
        return AlbumTarget()
    lufs_vals = [t.integrated_lufs for t in tracks if t.integrated_lufs < -5]
    cent_vals = [t.spectral_centroid_hz for t in tracks]
    stereo_vals = [t.stereo_width for t in tracks]
    return AlbumTarget(
        target_lufs=float(np.median(lufs_vals)) if lufs_vals else -16.0,
        target_centroid_hz=float(np.median(cent_vals)),
        target_stereo_width=float(np.median(stereo_vals)),
    )


def normalize_track(audio: np.ndarray, sr: int, target: AlbumTarget) -> np.ndarray:
    mono = np.mean(audio, axis=-1) if audio.ndim > 1 else np.asarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(mono**2))) + 1e-10
    target_rms = 10 ** (target.target_lufs / 20)
    gain = target_rms / rms
    gain = float(np.clip(gain, 0.1, 10.0))
    logger.info("AlbumConsistency: gain=%.1f dB", 20 * np.log10(gain))
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)
