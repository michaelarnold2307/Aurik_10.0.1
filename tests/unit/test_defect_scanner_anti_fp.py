"""Anti-False-Positive-Tests für DefectScanner (§6.3 — Aurik v9.10.57).

Validates that the three hardened detectors (_detect_clicks,
_detect_crackle, _detect_compression_artifacts) do NOT produce false
positives on clean/musical signals while still detecting real defects.
"""

from __future__ import annotations

import numpy as np
import pytest

SR = 48_000


def _scanner(sr: int = SR):
    from backend.core.defect_scanner import DefectScanner
    return DefectScanner(sample_rate=sr)


def _sine(freq: float = 440.0, amp: float = 0.5, duration: float = 3.0) -> np.ndarray:
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)


def _complex_tone(duration: float = 3.0) -> np.ndarray:
    """Harmonically rich tone (fundamental + 5 harmonics) — no defects."""
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    sig = np.zeros_like(t, dtype=np.float32)
    for k in range(1, 7):
        sig += (0.3 / k) * np.sin(2 * np.pi * 440 * k * t).astype(np.float32)
    return sig


# ============================================================
# CLICKS – Anti-False-Positive
# ============================================================

class TestClicksAntiFP:
    """Clean signals must NOT trigger click detection."""

    def test_pure_sine_no_clicks(self):
        """Pure 440 Hz sine → severity 0, no locations."""
        sc = _scanner()
        score = sc._detect_clicks(_sine(440, 0.5))
        assert score.severity == 0.0
        assert len(score.locations) == 0

    def test_complex_tone_no_clicks(self):
        """Harmonically rich tone → negligible click severity."""
        sc = _scanner()
        score = sc._detect_clicks(_complex_tone())
        # Allow marginal severity from harmonic peak transitions
        assert score.severity < 0.05

    def test_loud_sine_no_clicks(self):
        """Full-scale sine → no clicks (high diff values, but periodic)."""
        sc = _scanner()
        score = sc._detect_clicks(_sine(440, 0.95))
        assert score.severity == 0.0

    def test_low_freq_sine_no_clicks(self):
        """50 Hz sine → steeper diff per sample, still no clicks."""
        sc = _scanner()
        score = sc._detect_clicks(_sine(50, 0.5))
        assert score.severity == 0.0

    def test_real_clicks_still_detected(self):
        """Injected clicks on sine must still be found."""
        sc = _scanner()
        audio = _sine(440, 0.3)
        for pos in [0.5, 1.5, 2.5]:
            idx = int(pos * SR)
            audio[idx] = 0.99
        score = sc._detect_clicks(audio)
        assert score.severity > 0.0
        assert len(score.locations) >= 3

    def test_many_clicks_capped_at_50(self):
        """200 injected clicks → locations list ≤ 50."""
        sc = _scanner()
        audio = _sine(440, 0.3)
        rng = np.random.default_rng(42)
        positions = rng.integers(1000, len(audio) - 1000, size=200)
        for p in positions:
            audio[int(p)] = 0.99
        score = sc._detect_clicks(audio)
        assert len(score.locations) <= 50
        assert score.metadata["total_clicks"] > 50  # severity uses full count

    def test_drum_transient_not_click(self):
        """Realistic drum transient (~2 ms attack) must not be detected as click."""
        sc = _scanner()
        audio = _sine(440, 0.2)
        # Simulate a realistic drum hit: smooth 2 ms attack + 20 ms decay
        # Added on TOP of existing signal (like real drums) to avoid
        # boundary discontinuities that create artificial click-like edges.
        idx = int(1.0 * SR)
        attack_samples = int(0.002 * SR)   # 2 ms attack (96 samples @ 48 kHz)
        decay_samples = int(0.020 * SR)    # 20 ms decay
        total = attack_samples + decay_samples
        envelope = np.concatenate([
            np.linspace(0.0, 0.6, attack_samples),
            np.linspace(0.6, 0.0, decay_samples),
        ]).astype(np.float32)
        audio[idx:idx + total] += envelope
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
        score = sc._detect_clicks(audio)
        # Wide musical transient should not be flagged as click
        click_at_1s = [loc for loc in score.locations
                       if abs(loc[0] - 1.0) < 0.03]
        assert len(click_at_1s) == 0


# ============================================================
# CRACKLE – Anti-False-Positive
# ============================================================

class TestCrackleAntiFP:
    """Brilliant / HF-rich signals must NOT trigger crackle detection."""

    def test_bright_harmonic_no_crackle(self):
        """Harmonically rich signal (cymbals-like) → no crackle FP."""
        sc = _scanner()
        score = sc._detect_crackle(_complex_tone())
        assert score.severity < 0.15  # tolerance for edge cases

    def test_pure_hf_sine_no_crackle(self):
        """12 kHz sine → high HP energy, but tonal → no crackle."""
        sc = _scanner()
        score = sc._detect_crackle(_sine(12000, 0.3))
        assert score.severity < 0.15

    def test_real_crackle_detected(self):
        """Injected impulsive noise (crackle) must still be found."""
        sc = _scanner()
        rng = np.random.default_rng(7)
        audio = _sine(440, 0.2)
        # Inject sparse high-frequency impulses (crackle)
        n_crackle = 500
        positions = rng.integers(0, len(audio), size=n_crackle)
        amplitudes = rng.uniform(0.05, 0.2, size=n_crackle).astype(np.float32)
        for p, a in zip(positions, amplitudes):
            audio[int(p)] += a * (1 if rng.random() > 0.5 else -1)
        score = sc._detect_crackle(audio)
        assert score.severity > 0.0


# ============================================================
# COMPRESSION ARTIFACTS – Anti-False-Positive
# ============================================================

class TestCompressionArtifactsAntiFP:
    """Tonal / full-bandwidth signals must NOT trigger codec artifact detection."""

    def test_pure_sine_no_compression(self):
        """440 Hz sine → no compression artifacts."""
        sc = _scanner()
        score = sc._detect_compression_artifacts(_sine(440, 0.5))
        assert score.severity < 0.15

    def test_complex_fullband_no_compression(self):
        """Harmonically rich tone with HF content → no codec FP."""
        sc = _scanner()
        score = sc._detect_compression_artifacts(_complex_tone())
        assert score.severity < 0.15

    def test_white_noise_no_compression(self):
        """Broadband white noise has low SFM-variance → could be edge case."""
        sc = _scanner()
        rng = np.random.default_rng(42)
        audio = (rng.standard_normal(3 * SR) * 0.1).astype(np.float32)
        score = sc._detect_compression_artifacts(audio)
        # White noise has uniform SFM → low temporal variance → not compression
        assert score.severity < 0.3

    def test_bandwidth_limited_detected(self):
        """Signal hard-cut at 16 kHz (like low-bitrate MP3) → detected."""
        sc = _scanner()
        rng = np.random.default_rng(42)
        audio = (rng.standard_normal(3 * SR) * 0.1).astype(np.float32)
        # Hard low-pass at 16 kHz via FFT
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1 / SR)
        fft[freqs > 16000] = 0
        audio = np.fft.irfft(fft, n=len(audio)).astype(np.float32)
        score = sc._detect_compression_artifacts(audio)
        # Should detect the bandwidth limitation
        assert score.severity > 0.0
