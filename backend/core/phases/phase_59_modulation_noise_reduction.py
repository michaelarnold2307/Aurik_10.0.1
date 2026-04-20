"""
Phase 59 — Modulation Noise Reduction.

Signal-dependent modulation noise (unique to analog tape) where noise level
tracks the signal envelope.  Unlike stationary tape hiss (phase_29), modulation
noise requires signal-adaptive noise gating that varies its threshold with the
signal level.

Algorithm (Esquef & Biscainho 2006):
1. Estimate signal envelope (10 ms RMS)
2. Estimate noise floor as a function of signal level
3. Apply signal-dependent spectral gating: G(f) = max(G_floor, 1 - alpha * N(f)/S(f))
   where N(f) is estimated noise at frequency f given current signal level
4. Wet/dry blend with strength parameter

Scientific basis: Esquef & Biscainho (2006), Czyzewski et al. (2020).
"""

from __future__ import annotations

import logging
import time as _time

import numpy as np
import scipy.signal as sps

logger = logging.getLogger(__name__)


def _rms_dbfs_gated(sig: np.ndarray) -> float:
    """§2.45a-I: Frame-basierter RMS in dBFS, ignoriert Frames < −50 dBFS (Stille).

    Stereo → Mono-Downmix vor Framing. Gibt -96.0 zurück wenn kein aktiver Frame.
    """
    if sig.ndim == 2:
        _mono = sig.mean(axis=0).astype(np.float64) if sig.shape[0] <= 2 else sig.mean(axis=1).astype(np.float64)
    else:
        _mono = sig.astype(np.float64)
    _frame = 480  # 10 ms @ 48 kHz
    _active = [
        _mono[i : i + _frame]
        for i in range(0, len(_mono) - _frame, _frame)
        if 20.0 * np.log10(np.sqrt(np.mean(_mono[i : i + _frame] ** 2)) + 1e-10) > -50.0
    ]
    if not _active:
        return -96.0
    return float(20.0 * np.log10(np.sqrt(np.mean(np.concatenate(_active) ** 2)) + 1e-10))


_MIN_MODULATION_NOISE_SCORE: float = 0.10
_G_FLOOR: float = 0.08  # Minimum spectral gain to avoid musical noise


def apply(
    audio: np.ndarray,
    sample_rate: int,
    strength: float = 0.7,
    defect_scores: dict | None = None,
    min_modulation_noise_score: float = _MIN_MODULATION_NOISE_SCORE,
    g_floor: float = _G_FLOOR,
) -> np.ndarray:
    """Main entry point for Phase 59.

    Args:
        audio:        Input audio float32, mono or stereo
        sample_rate:  Must be 48000 Hz
        strength:     Processing strength 0–1
        defect_scores: Defect scan scores (optional, for gating)

    Returns:
        Processed audio, same shape as input
    """
    assert sample_rate == 48000, f"SR must be 48000 Hz, got: {sample_rate}"
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    # Gate: only process if modulation noise is detected
    if defect_scores is not None:
        mn_score = float(defect_scores.get("modulation_noise", 0.0))
        if mn_score < min_modulation_noise_score:
            logger.debug(
                "Phase 59: modulation_noise score %.3f < %.3f — skipped",
                mn_score,
                min_modulation_noise_score,
            )
            return np.clip(audio, -1.0, 1.0)

    stereo = audio.ndim == 2
    if stereo:
        # §2.51 Linked-Stereo: Noise-Modell aus Mid, identischer STFT-Gain auf L+R
        mono_mix = (audio[0] + audio[1]) / 2.0
        mono_denoised = apply(mono_mix, sample_rate, strength=strength, defect_scores=defect_scores)
        _eps_mn = 1e-10
        _gain_mn = np.where(
            np.abs(mono_mix) > _eps_mn,
            mono_denoised / (mono_mix + _eps_mn * np.sign(mono_mix + _eps_mn)),
            1.0,
        )
        _gain_mn = np.clip(_gain_mn, 0.0, 10.0)
        return np.clip(np.stack([audio[0] * _gain_mn, audio[1] * _gain_mn], axis=0), -1.0, 1.0).astype(np.float32)

    x = audio.astype(np.float64)
    n = len(x)

    # STFT parameters
    n_fft = 2048
    hop = n_fft // 4
    window = sps.windows.hann(n_fft, sym=False)

    # Compute STFT
    n_frames = max(1, (n - n_fft) // hop + 1)
    stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)
    for i in range(n_frames):
        start = i * hop
        end = start + n_fft
        if end > n:
            break
        frame = x[start:end] * window
        stft[:, i] = np.fft.rfft(frame)

    mag = np.abs(stft)
    phase = np.angle(stft)

    # Signal envelope (per-frame RMS)
    frame_rms = np.sqrt(np.mean(mag**2, axis=0) + 1e-12)

    # Estimate noise model: noise_level(f) = alpha * signal_level
    # Use low-energy frames to calibrate the noise/signal ratio
    noise_floor = np.percentile(mag, 10, axis=1, keepdims=True)
    np.median(mag, axis=1, keepdims=True) + 1e-12

    # Signal-dependent noise estimate per frame
    alpha = float(np.clip(strength * 0.8, 0.1, 0.9))
    noise_estimate = noise_floor * (frame_rms / (np.median(frame_rms) + 1e-12))[np.newaxis, :]

    # Spectral gating: reduce noise proportional to signal level
    gain = np.maximum(g_floor, 1.0 - alpha * noise_estimate / (mag + 1e-12))
    gain = np.clip(gain, g_floor, 1.0)

    # Apply gain and reconstruct
    mag_clean = mag * gain
    stft_clean = mag_clean * np.exp(1j * phase)

    # Inverse STFT (overlap-add)
    out = np.zeros(n, dtype=np.float64)
    win_sum = np.zeros(n, dtype=np.float64)
    for i in range(min(n_frames, stft_clean.shape[1])):
        start = i * hop
        end = start + n_fft
        if end > n:
            break
        frame = np.fft.irfft(stft_clean[:, i], n=n_fft) * window
        out[start:end] += frame
        win_sum[start:end] += window**2

    win_sum = np.maximum(win_sum, 1e-8)
    out /= win_sum

    # Wet/dry blend
    result = x * (1.0 - strength) + out * strength
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(result, -1.0, 1.0).astype(np.float32)


# ─── PhaseInterface ────────────────────────────────────────────────────────────

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult


class ModulationNoiseReductionPhase(PhaseInterface):
    """Phase 59: Signal-dependent modulation noise reduction for analog tape."""

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_59_modulation_noise_reduction",
            name="Modulation Noise Reduction",
            category=PhaseCategory.RESTORATION,
            priority=6,
            dependencies=["phase_03"],
            estimated_time_factor=0.05,
            version="1.0.0",
            memory_requirement_mb=32,
            is_cpu_intensive=False,
            quality_impact=0.65,
            description=(
                "Signal-dependent modulation noise reduction using adaptive spectral "
                "gating that tracks signal envelope. Targets noise that varies with "
                "signal level (unique to analog tape recordings)."
            ),
        )

    @staticmethod
    def _compute_modulation_noise_profile(
        material_key: str,
        quality_mode: str | None,
        restorability_score: float,
    ) -> dict[str, float]:
        """§2.54 Adaptive modulation-noise profile."""
        _material = str(material_key or "unknown").strip().lower()
        _aliases = {"restoration": "balanced", "studio_2026": "maximum"}
        _mode = _aliases.get(
            str(quality_mode or "balanced").strip().lower(), str(quality_mode or "balanced").strip().lower()
        )

        if any(token in _material for token in ("tape", "reel_tape", "cassette")):
            min_modulation_noise_score = 0.10
            g_floor = 0.08
        elif any(token in _material for token in ("cd_digital", "dat", "streaming", "flac")):
            min_modulation_noise_score = 0.18
            g_floor = 0.12
        else:
            min_modulation_noise_score = 0.14
            g_floor = 0.10

        _rest = float(np.clip(float(restorability_score or 50.0), 0.0, 100.0))
        _rest_norm = _rest / 100.0
        min_modulation_noise_score += (_rest_norm - 0.5) * 0.10
        g_floor += (0.5 - _rest_norm) * 0.08

        _mode_offsets = {
            "fast": (0.04, 0.03),
            "balanced": (0.0, 0.0),
            "quality": (-0.03, -0.02),
            "maximum": (-0.05, -0.03),
        }
        _score_off, _g_off = _mode_offsets.get(_mode, (0.0, 0.0))
        min_modulation_noise_score += _score_off
        g_floor += _g_off

        return {
            "min_modulation_noise_score": float(np.clip(min_modulation_noise_score, 0.05, 0.25)),
            "g_floor": float(np.clip(g_floor, 0.02, 0.30)),
        }

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        strength: float = 0.7,
        defect_scores: dict | None = None,
        **kwargs,
    ) -> PhaseResult:
        sample_rate = kwargs.get("sample_rate", sample_rate)
        t0 = _time.perf_counter()
        assert sample_rate == 48000, f"SR must be 48000 Hz, got: {sample_rate}"

        _defect_scores = defect_scores or kwargs.get("defect_analysis", {})
        _profile_59 = self._compute_modulation_noise_profile(
            str(kwargs.get("material_type", kwargs.get("material", "unknown"))).lower(),
            kwargs.get("quality_mode"),
            float(kwargs.get("restorability_score", 50.0)),
        )
        phase_locality_factor = float(np.clip(float(kwargs.get("phase_locality_factor", 1.0)), 0.35, 1.0))
        _pmgg_strength = float(kwargs.get("strength", strength))
        _effective_strength = float(np.clip(_pmgg_strength * phase_locality_factor, 0.0, 1.0))
        if _effective_strength <= 0.0:
            passthrough = np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)
            passthrough = np.clip(passthrough, -1.0, 1.0)
            return PhaseResult(
                audio=passthrough,
                success=True,
                execution_time_seconds=_time.perf_counter() - t0,
                metrics={
                    "modulation_noise_score": float((_defect_scores or {}).get("modulation_noise", 0.0)),
                    "strength": strength,
                    "effective_strength": 0.0,
                },
                metadata={
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                warnings=["Modulation noise reduction skipped due to zero effective strength"],
            )
        _rms_in_db = _rms_dbfs_gated(audio)
        result_audio = apply(
            audio,
            sample_rate,
            strength=_effective_strength,
            defect_scores=_defect_scores,
            min_modulation_noise_score=_profile_59["min_modulation_noise_score"],
            g_floor=_profile_59["g_floor"],
        )
        elapsed = _time.perf_counter() - t0

        # §4.5 Psychoacoustic Masking Clamp — only reduce audible modulation noise
        try:
            from backend.core.dsp.psychoacoustics import apply_psychoacoustic_masking_clamp

            result_audio = apply_psychoacoustic_masking_clamp(
                audio,
                result_audio,
                sample_rate,
                strength=_effective_strength,
                mode="subtractive",
            )
        except Exception as _pm_exc:
            import logging as _log59

            _log59.getLogger(__name__).debug("Phase59 masking clamp non-blocking: %s", _pm_exc)

        _rms_out_db = _rms_dbfs_gated(result_audio)
        _rms_drop = (_rms_out_db - _rms_in_db) if _rms_in_db > -80.0 else 0.0
        return PhaseResult(
            audio=result_audio,
            success=True,
            execution_time_seconds=elapsed,
            metrics={
                "modulation_noise_score": float((_defect_scores or {}).get("modulation_noise", 0.0)),
                "strength": strength,
                "effective_strength": _effective_strength,
            },
            metadata={
                "modulation_noise_profile": dict(_profile_59),
                "min_modulation_noise_score": float(_profile_59["min_modulation_noise_score"]),
                "g_floor": float(_profile_59["g_floor"]),
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": _effective_strength,
                "rms_drop_db": round(float(min(0.0, _rms_drop)), 3),
                "loudness_makeup_db": 0.0,
            },
        )
