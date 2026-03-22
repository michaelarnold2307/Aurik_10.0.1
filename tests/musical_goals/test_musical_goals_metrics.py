"""
AURIK v8 Musical Goals Metrics - Automated Test Suite
======================================================

Comprehensive test suite for all 7 musical goals metrics:
1. Bass-Kraft (20-250 Hz)
2. Brillanz (8-20 kHz)
3. Wärme (200-2000 Hz)
4. Natürlichkeit (Spectral properties)
5. Authentizität (Voice/Spectral fingerprint)
6. Emotionalität (Dynamics)
7. Transparenz (Clarity/Separation)

Test Categories:
- Unit Tests: Individual metric correctness
- Range Tests: Score bounds (0.0-1.0)
- Stability Tests: Consistent results
- Regression Tests: Prevent degradation
- Golden Sample Tests: Real-world validation

Quelle: Finalisierungs_Roadmap.md - Component 0.9.1
Autor: AI Team
Datum: 8. Februar 2026
"""

import numpy as np
import pytest

from backend.core.musical_goals import (
    AuthentizitaetMetric,
    BassKraftMetric,
    BrillanzMetric,
    EmotionalitaetMetric,
    MusicalGoalsChecker,
    NatuerlichkeitMetric,
    TransparenzMetric,
    WaermeMetric,
)


class TestBassKraftMetric:
    """Test suite for Bass-Kraft metric (20-250 Hz)."""

    @pytest.fixture
    def metric(self):
        return BassKraftMetric(threshold=0.85)

    @pytest.fixture
    def bass_heavy_audio(self):
        """Audio with strong bass (100 Hz)."""
        sr = 48000
        t = np.linspace(0, 1.0, sr)
        # Heavy bass at 100 Hz
        audio = 0.8 * np.sin(2 * np.pi * 100 * t) + 0.2 * np.sin(2 * np.pi * 1000 * t)
        return audio, sr

    @pytest.fixture
    def bass_light_audio(self):
        """Audio with weak bass (mostly high frequencies)."""
        sr = 48000
        t = np.linspace(0, 1.0, sr)
        # Mostly high frequencies
        audio = 0.1 * np.sin(2 * np.pi * 100 * t) + 0.9 * np.sin(2 * np.pi * 5000 * t)
        return audio, sr

    def test_bass_kraft_score_range(self, metric, bass_heavy_audio):
        """Test that bass kraft score is in valid range [0.0, 1.0]."""
        audio, sr = bass_heavy_audio
        score = metric.measure(audio, sr)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_bass_heavy_high_score(self, metric, bass_heavy_audio):
        """Test that bass-heavy audio gets high score."""
        audio, sr = bass_heavy_audio
        score = metric.measure(audio, sr)
        assert score > 0.7, f"Bass-heavy audio should score >0.7, got {score}"

    def test_bass_light_low_score(self, metric, bass_light_audio):
        """Test that bass-light audio gets low score."""
        audio, sr = bass_light_audio
        score = metric.measure(audio, sr)
        assert score < 0.5, f"Bass-light audio should score <0.5, got {score}"

    def test_bass_preservation_check(self, metric, bass_heavy_audio):
        """Test bass preservation check."""
        audio, sr = bass_heavy_audio
        # Simulate processing that reduces bass
        processed = audio * np.array([0.5 if i < len(audio) // 4 else 1.0 for i in range(len(audio))])

        passed, loss, details = metric.check_preservation(audio, processed, sr)
        assert 0.0 <= loss <= 1.0, "Loss should be in [0.0, 1.0]"
        assert "original_score" in details
        assert "processed_score" in details

    def test_measurement_stability(self, metric, bass_heavy_audio):
        """Test that multiple measurements are consistent."""
        audio, sr = bass_heavy_audio
        scores = [metric.measure(audio, sr) for _ in range(5)]
        std = np.std(scores)
        assert std < 0.05, f"Measurements unstable, std={std}"


class TestBrillanzMetric:
    """Test suite for Brillanz metric (8-20 kHz)."""

    @pytest.fixture
    def metric(self):
        return BrillanzMetric(threshold=0.85)

    @pytest.fixture
    def bright_audio(self):
        """Audio with strong high frequencies."""
        sr = 48000
        t = np.linspace(0, 1.0, sr)
        audio = 0.2 * np.sin(2 * np.pi * 100 * t) + 0.8 * np.sin(2 * np.pi * 10000 * t)
        return audio, sr

    @pytest.fixture
    def dull_audio(self):
        """Audio with weak high frequencies."""
        sr = 48000
        t = np.linspace(0, 1.0, sr)
        audio = 0.9 * np.sin(2 * np.pi * 500 * t) + 0.1 * np.sin(2 * np.pi * 10000 * t)
        return audio, sr

    def test_brillanz_score_range(self, metric, bright_audio):
        """Test that brillanz score is in valid range."""
        audio, sr = bright_audio
        score = metric.measure(audio, sr)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_bright_audio_high_score(self, metric, bright_audio):
        """Test that bright audio gets high score."""
        audio, sr = bright_audio
        score = metric.measure(audio, sr)
        assert score > 0.6, f"Bright audio should score >0.6, got {score}"

    def test_dull_audio_low_score(self, metric, dull_audio):
        """Test that dull audio gets low score."""
        audio, sr = dull_audio
        score = metric.measure(audio, sr)
        assert score < 0.5, f"Dull audio should score <0.5, got {score}"

    def test_measurement_stability(self, metric, bright_audio):
        """Test measurement consistency."""
        audio, sr = bright_audio
        scores = [metric.measure(audio, sr) for _ in range(5)]
        std = np.std(scores)
        assert std < 0.05, f"Measurements unstable, std={std}"


class TestWaermeMetric:
    """Test suite for Wärme metric (200-2000 Hz)."""

    @pytest.fixture
    def metric(self):
        return WaermeMetric(threshold=0.80)

    @pytest.fixture
    def warm_audio(self):
        """Audio with strong mid-range (warm)."""
        sr = 48000
        t = np.linspace(0, 1.0, sr)
        audio = 0.8 * np.sin(2 * np.pi * 500 * t) + 0.2 * np.sin(2 * np.pi * 5000 * t)
        return audio, sr

    def test_waerme_score_range(self, metric, warm_audio):
        """Test that wärme score is in valid range."""
        audio, sr = warm_audio
        score = metric.measure(audio, sr)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_warm_audio_high_score(self, metric, warm_audio):
        """Test that warm audio gets high score."""
        audio, sr = warm_audio
        score = metric.measure(audio, sr)
        assert score > 0.6, f"Warm audio should score >0.6, got {score}"


class TestNatuerlichkeitMetric:
    """Test suite for Natürlichkeit metric."""

    @pytest.fixture
    def metric(self):
        return NatuerlichkeitMetric(threshold=0.90)

    @pytest.fixture
    def natural_audio(self):
        """Natural audio with harmonics."""
        sr = 48000
        t = np.linspace(0, 1.0, sr)
        # Fundamental + harmonics (natural sound)
        audio = (
            0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t) + 0.2 * np.sin(2 * np.pi * 1320 * t)
        )
        return audio, sr

    @pytest.fixture
    def unnatural_audio(self):
        """Unnatural audio (white noise)."""
        sr = 48000
        audio = np.random.randn(sr)
        return audio, sr

    def test_natuerlichkeit_score_range(self, metric, natural_audio):
        """Test that natürlichkeit score is in valid range."""
        audio, sr = natural_audio
        score = metric.measure(audio, sr)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_natural_audio_high_score(self, metric, natural_audio):
        """Test that natural audio gets high score."""
        audio, sr = natural_audio
        score = metric.measure(audio, sr)
        assert score > 0.7, f"Natural audio should score >0.7, got {score}"

    def test_unnatural_audio_low_score(self, metric, unnatural_audio):
        """Test that unnatural audio gets low score."""
        audio, sr = unnatural_audio
        score = metric.measure(audio, sr)
        assert score < 0.6, f"Unnatural audio should score <0.6, got {score}"


class TestAuthentizitaetMetric:
    """Test suite for Authentizität metric."""

    @pytest.fixture
    def metric(self):
        return AuthentizitaetMetric(threshold=0.88)

    @pytest.fixture
    def test_audio(self):
        sr = 48000
        t = np.linspace(0, 1.0, sr)
        audio = np.sin(2 * np.pi * 440 * t)
        return audio, sr

    def test_authentizitaet_score_range(self, metric, test_audio):
        """Test that authentizität score is in valid range."""
        audio, sr = test_audio
        score = metric.measure(audio, sr)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_with_reference_audio(self, metric, test_audio):
        """Test authentizität with reference audio."""
        audio, sr = test_audio
        reference = audio.copy()
        score = metric.measure(audio, sr, reference=reference)
        # Same audio should have high authenticity (>= 0.75 wegen möglichem DSP-Fallback ohne skimage)
        assert score >= 0.75, f"Identical audio should score >=0.75, got {score}"


class TestEmotionalitaetMetric:
    """Test suite for Emotionalität metric."""

    @pytest.fixture
    def metric(self):
        return EmotionalitaetMetric(threshold=0.87)

    @pytest.fixture
    def dynamic_audio(self):
        """Audio with high dynamics."""
        sr = 48000
        t = np.linspace(0, 1.0, sr)
        # Varying amplitude (emotional)
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)
        audio = envelope * np.sin(2 * np.pi * 440 * t)
        return audio, sr

    @pytest.fixture
    def flat_audio(self):
        """Audio with low dynamics."""
        sr = 48000
        t = np.linspace(0, 1.0, sr)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # Constant amplitude
        return audio, sr

    def test_emotionalitaet_score_range(self, metric, dynamic_audio):
        """Test that emotionalität score is in valid range."""
        audio, sr = dynamic_audio
        score = metric.measure(audio, sr)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_dynamic_audio_high_score(self, metric, dynamic_audio):
        """Test that dynamic audio gets high score."""
        audio, sr = dynamic_audio
        score = metric.measure(audio, sr)
        assert score > 0.5, f"Dynamic audio should score >0.5, got {score}"

    def test_flat_audio_low_score(self, metric, flat_audio):
        """Test that flat audio gets low score."""
        audio, sr = flat_audio
        score = metric.measure(audio, sr)
        assert score < 0.5, f"Flat audio should score <0.5, got {score}"


class TestTransparenzMetric:
    """Test suite for Transparenz metric."""

    @pytest.fixture
    def metric(self):
        return TransparenzMetric(threshold=0.89)

    @pytest.fixture
    def clear_audio(self):
        """Clear audio with good separation."""
        sr = 48000
        t = np.linspace(0, 1.0, sr)
        # Clear frequencies, well-separated
        audio = (
            0.3 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 2000 * t) + 0.3 * np.sin(2 * np.pi * 8000 * t)
        )
        return audio, sr

    def test_transparenz_score_range(self, metric, clear_audio):
        """Test that transparenz score is in valid range."""
        audio, sr = clear_audio
        score = metric.measure(audio, sr)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range"


class TestMusicalGoalsChecker:
    """Integration tests for MusicalGoalsChecker."""

    @pytest.fixture
    def checker(self):
        return MusicalGoalsChecker()

    @pytest.fixture
    def test_audio(self):
        """Multi-frequency test audio."""
        sr = 48000
        t = np.linspace(0, 2.0, int(sr * 2))
        audio = (
            0.3 * np.sin(2 * np.pi * 100 * t)
            + 0.3 * np.sin(2 * np.pi * 500 * t)
            + 0.2 * np.sin(2 * np.pi * 2000 * t)
            + 0.2 * np.sin(2 * np.pi * 8000 * t)
        )
        return audio, sr

    def test_measure_all_returns_all_goals(self, checker, test_audio):
        """Test that measure_all returns all 14 goals (v9.9.9 Spec)."""
        audio, sr = test_audio
        scores = checker.measure_all(audio, sr)

        # 14 Musical Goals gemäß Spec §1.2 / §8.1 — deutsche Schlüssel
        expected_goals = {
            "bass_kraft",
            "brillanz",
            "waerme",
            "natuerlichkeit",
            "authentizitaet",
            "emotionalitaet",
            "transparenz",
            "groove",  # v9.9 Groove-Metrik
            "spatial_depth",  # v9.9 Raumtiefe
            "timbre_authentizitaet",  # v9.9 Timbre-Authentizität (deutsch)
            "tonal_center",  # v9.9.5 Tonales Zentrum
            "micro_dynamics",  # v9.9.5 Mikro-Dynamik
            "separation_fidelity",  # v9.9.9 Separation-Treue
            "artikulation",  # v9.9.9 Artikulation
        }
        assert set(scores.keys()) == expected_goals, (
            f"Missing or extra goals. \nGot: {sorted(scores.keys())}\nExpected: {sorted(expected_goals)}"
        )

    def test_all_scores_in_valid_range(self, checker, test_audio):
        """Test that all scores are in [0.0, 1.0]."""
        audio, sr = test_audio
        scores = checker.measure_all(audio, sr)

        for goal, score in scores.items():
            assert 0.0 <= score <= 1.0, f"{goal} score {score} out of range"

    def test_check_all_preserved(self, checker, test_audio):
        """Test check_all_preserved with minimal degradation."""
        audio, sr = test_audio

        # Slightly degraded audio (98% of original)
        degraded = audio * 0.98

        passed, violations = checker.check_all_preserved(audio, degraded, sr)
        # Should have some violations but not catastrophic
        assert isinstance(passed, bool)
        assert isinstance(violations, dict)

    def test_measure_single_goal(self, checker, test_audio):
        """Test measuring single goal."""
        audio, sr = test_audio
        result = checker.measure_single("brillanz", audio, sr)

        assert result.goal_name == "brillanz"
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.passed, bool)
        assert result.threshold == checker.thresholds["brillanz"]


class TestRegressionPrevention:
    """Regression tests to prevent metric degradation."""

    @pytest.fixture
    def checker(self):
        return MusicalGoalsChecker()

    def test_reference_scores_stability(self, checker):
        """Test that reference audio has consistent scores over time."""
        # Reference audio (stored baseline scores)
        sr = 48000
        t = np.linspace(0, 2.0, int(sr * 2))
        audio = (
            0.3 * np.sin(2 * np.pi * 100 * t)
            + 0.3 * np.sin(2 * np.pi * 500 * t)
            + 0.2 * np.sin(2 * np.pi * 2000 * t)
            + 0.2 * np.sin(2 * np.pi * 8000 * t)
        )

        # Expected baseline scores — UPDATED v9.10 after formula recalibration
        # (BrillanzMetric ceiling fix, EmotionalitaetMetric dB crest, TransparenzMetric rolloff)
        # Signal: 100+500+2000+8000 Hz tones, amplitudes 0.3/0.3/0.2/0.2
        baseline_scores = {
            "bass_kraft": (0.90, 1.05),  # Bass-heavy signal always near 1.0
            "brillanz": (0.75, 0.92),  # 8000 Hz = ~15% energy → hf_score=1.0, centroid~2180 Hz
            "waerme": (0.90, 1.05),  # Mid-heavy signal always near 1.0
            "natuerlichkeit": (0.89, 1.00),  # Low flatness (pure tones) → high naturalness
            "authentizitaet": (0.63, 0.79),  # No-reference heuristic: moderate for 4-tone mix
            "emotionalitaet": (0.22, 0.32),  # dB crest fix: 4-tone mix crest ~8.9 dB → 0.27
            "transparenz": (0.56, 0.71),  # 75% rolloff, bandwidth ≥ 4000 Hz after fix
        }

        scores = checker.measure_all(audio, sr)

        for goal, (min_score, max_score) in baseline_scores.items():
            assert min_score <= scores[goal] <= max_score, (
                f"Regression detected in {goal}: {scores[goal]} not in [{min_score}, {max_score}]"
            )


class TestISO226WeightingAndVirtualPitch:
    """Tests für ISO 226:2023 Equal-Loudness-Gewichtung (Brillanz/Wärme) und
    Virtual Pitch / Missing Fundamental (BassKraft) — Spec §8.1."""

    SR = 48000

    def _band_energy_signal(self, freq_hz: float, duration: float = 1.5) -> np.ndarray:
        """Sinuston bei freq_hz als float32-Mono."""
        t = np.linspace(0, duration, int(self.SR * duration), endpoint=False)
        return (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)

    def _harmonic_bass_signal(self, f0_hz: float = 55.0, duration: float = 2.0) -> np.ndarray:
        """Sinussumme: schwacher F0 + starke Obertöne 2F0, 3F0, 4F0 (Missing Fundamental)."""
        t = np.linspace(0, duration, int(self.SR * duration), endpoint=False)
        audio = (
            0.05 * np.sin(2 * np.pi * f0_hz * t)  # weak fundamental
            + 0.40 * np.sin(2 * np.pi * 2 * f0_hz * t)  # strong 2nd harmonic
            + 0.35 * np.sin(2 * np.pi * 3 * f0_hz * t)  # strong 3rd harmonic
            + 0.25 * np.sin(2 * np.pi * 4 * f0_hz * t)  # strong 4th harmonic
        )
        return audio.astype(np.float32)

    # --- ISO 226 helper --------------------------------------------------

    def test_iso226_weights_shape_and_finite(self):
        """`_iso226_weights` gibt float32-Array korrekter Länge ohne NaN/Inf zurück."""
        from backend.core.musical_goals.musical_goals_metrics import _iso226_weights

        freqs = np.linspace(0, 24000, 1025, dtype=np.float32)
        w = _iso226_weights(freqs)
        assert w.shape == freqs.shape
        assert w.dtype == np.float32
        assert np.isfinite(w).all()

    def test_iso226_weights_reference_1khz(self):
        """1 kHz muss Gewicht 1.0 ergeben (ISO 226-Referenz)."""
        from backend.core.musical_goals.musical_goals_metrics import _iso226_weights

        w = _iso226_weights(np.array([1000.0], dtype=np.float32))
        assert abs(float(w[0]) - 1.0) < 0.02, f"1 kHz weight = {w[0]:.4f} (expected ~1.0)"

    def test_iso226_sensitivity_peak_3to4khz(self):
        """3\u20134 kHz muss Gewicht > 1.5 haben (Ohr am empfindlichsten dort)."""
        from backend.core.musical_goals.musical_goals_metrics import _iso226_weights

        w = _iso226_weights(np.array([3150.0, 4000.0], dtype=np.float32))
        assert float(w[0]) > 1.5, f"3150 Hz weight = {w[0]:.3f} (expected >1.5)"
        assert float(w[1]) > 1.5, f"4000 Hz weight = {w[1]:.3f} (expected >1.5)"

    def test_iso226_hf_weight_less_than_midrange(self):
        """16 kHz-Gewicht muss deutlich unter 1 kHz-Gewicht liegen (HF-Rolloff)."""
        from backend.core.musical_goals.musical_goals_metrics import _iso226_weights

        w = _iso226_weights(np.array([1000.0, 16000.0], dtype=np.float32))
        assert float(w[1]) < 0.15, f"16 kHz weight = {w[1]:.4f} (expected <0.15)"

    # --- BrillanzMetric --------------------------------------------------

    def test_brillanz_hf_rich_scores_higher_than_muffled(self):
        """HF-reiches Signal muss perceptuell h\u00f6her als gedämpftes Signal bewertet werden."""
        from backend.core.musical_goals.musical_goals_metrics import BrillanzMetric

        m = BrillanzMetric()
        hf_signal = self._band_energy_signal(10000.0)
        muffled = self._band_energy_signal(300.0)
        score_hf = m.measure(hf_signal, self.SR)
        score_muf = m.measure(muffled, self.SR)
        assert score_hf > score_muf, f"HF {score_hf:.3f} should > muffled {score_muf:.3f}"

    def test_brillanz_score_in_range(self):
        """BrillanzMetric gibt Score in [0, 1]."""
        from backend.core.musical_goals.musical_goals_metrics import BrillanzMetric

        m = BrillanzMetric()
        for freq in [200.0, 1000.0, 8000.0, 14000.0]:
            s = m.measure(self._band_energy_signal(freq), self.SR)
            assert 0.0 <= s <= 1.0, f"Score out of range at {freq} Hz: {s}"

    # --- WaermeMetric ----------------------------------------------------

    def test_waerme_presence_zone_weighted_above_body(self):
        """1\u20132 kHz (Präsenz-Zone, ISO-226-sensitiv) muss höher bewertet werden als reine
        200-400 Hz Energie (Körper-Zone, ISO-226-untergewichtet)."""
        from backend.core.musical_goals.musical_goals_metrics import WaermeMetric

        m = WaermeMetric()
        # 1500 Hz: ISO 226 weight ~1.5 (sensitive); 250 Hz: weight ~0.3 (less sensitive)
        score_presence = m.measure(self._band_energy_signal(1500.0), self.SR)
        score_body = m.measure(self._band_energy_signal(250.0), self.SR)
        assert score_presence >= score_body, f"Presence {score_presence:.3f} should >= body {score_body:.3f} (ISO 226)"

    def test_waerme_score_in_range(self):
        """WaermeMetric gibt Score in [0, 1]."""
        from backend.core.musical_goals.musical_goals_metrics import WaermeMetric

        m = WaermeMetric()
        for freq in [300.0, 700.0, 1500.0]:
            s = m.measure(self._band_energy_signal(freq), self.SR)
            assert 0.0 <= s <= 1.0, f"Score out of range at {freq} Hz: {s}"

    # --- BassKraftMetric / Virtual Pitch ---------------------------------

    def test_virtual_pitch_score_harmonic_signal_high(self):
        """Signal mit starkem Obertonsignal bei 120\u2013500 Hz muss hohen VP-Score liefern."""
        import librosa

        from backend.core.musical_goals.musical_goals_metrics import BassKraftMetric

        audio = self._harmonic_bass_signal(f0_hz=55.0)
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        mag = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=self.SR, n_fft=2048)
        score = BassKraftMetric._virtual_pitch_score(mag, freqs)
        assert score > 0.3, f"Harmonic bass: VP score too low: {score:.3f}"

    def test_virtual_pitch_score_noise_midrange(self):
        """Weißes Rauschen (kein harmonischer Zusammenhang) muss VP-Score < 0.6 liefern."""
        import librosa

        from backend.core.musical_goals.musical_goals_metrics import BassKraftMetric

        rng = np.random.default_rng(99)
        audio = rng.standard_normal(self.SR * 2).astype(np.float32) * 0.3
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        mag = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=self.SR, n_fft=2048)
        score = BassKraftMetric._virtual_pitch_score(mag, freqs)
        assert score < 0.6, f"Noise VP score unexpectedly high: {score:.3f}"

    def test_virtual_pitch_score_in_range(self):
        """VP-Score muss in [0, 1] liegen."""
        import librosa

        from backend.core.musical_goals.musical_goals_metrics import BassKraftMetric

        audio = self._harmonic_bass_signal()
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        mag = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=self.SR, n_fft=2048)
        score = BassKraftMetric._virtual_pitch_score(mag, freqs)
        assert 0.0 <= score <= 1.0

    def test_basskraft_measure_returns_valid_score(self):
        """BassKraftMetric.measure() integriert VP ohne Absturz, Score in [0,1]."""
        from backend.core.musical_goals.musical_goals_metrics import BassKraftMetric

        m = BassKraftMetric()
        audio = self._harmonic_bass_signal()
        score = m.measure(audio, self.SR)
        assert 0.0 <= score <= 1.0, f"BassKraft score out of range: {score}"


class TestTonalCenterMetricKeyShift:
    """Tests für die Key-Shift-Invariante in TonalCenterMetric (Spec §1.2).

    Spec: Chroma-Korrelation >= 0.95 UND kein Key-Shift > 0 Cent.
    Penalty-Tabelle: 0 Halbtöne → 1.0, 1 Halbton → ≤ 0.50, ≥ 2 → 0.0.
    """

    SR = 48000
    DUR = 2.0

    def _sine_for_key(self, root_hz: float, sr: int = SR, dur: float = DUR) -> np.ndarray:
        """Pure-tone chord rooted at root_hz (root + major third + fifth)."""
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        audio = (
            0.4 * np.sin(2 * np.pi * root_hz * t)
            + 0.3 * np.sin(2 * np.pi * root_hz * 1.2599 * t)  # major third
            + 0.3 * np.sin(2 * np.pi * root_hz * 1.4983 * t)  # perfect fifth
        )
        return audio.astype(np.float32)

    @pytest.fixture
    def metric(self):
        from backend.core.musical_goals.musical_goals_metrics import TonalCenterMetric

        return TonalCenterMetric()

    def test_no_key_shift_high_score(self, metric):
        """Identische Tonart → Score nahe 1.0 (kein Abzug)."""
        ref = self._sine_for_key(440.0)  # A4
        result = metric.measure(ref, self.SR, reference=ref)
        assert result >= 0.90, f"Same-key score too low: {result}"

    def test_one_semitone_shift_penalised(self, metric):
        """1-Halbton-Verschiebung → Score klar unter 0.50 (schwere Strafe)."""
        ref = self._sine_for_key(440.0)  # A4
        shifted = self._sine_for_key(466.16)  # A#4 / Bb4 — 1 semitone up
        result = metric.measure(shifted, self.SR, reference=ref)
        assert result <= 0.55, f"1-semitone shift not adequately penalised: {result}"

    def test_two_semitone_shift_catastrophic(self, metric):
        """2-Halbton-Verschiebung → Score = 0.0 (absolut inakzeptabel)."""
        ref = self._sine_for_key(440.0)  # A4
        shifted = self._sine_for_key(493.88)  # B4 — 2 semitones up
        result = metric.measure(shifted, self.SR, reference=ref)
        assert result == 0.0, f"2-semitone shift must yield 0.0, got: {result}"

    def test_dominant_chroma_class_helper(self, metric):
        """_dominant_chroma_class gibt gültigen Pitch-Class zurück (0..11)."""
        import numpy as _np

        chroma = _np.random.rand(12, 50).astype(_np.float32)
        pc = metric._dominant_chroma_class(chroma)
        assert 0 <= pc <= 11

    def test_key_shift_semitones_symmetry(self, metric):
        """_key_shift_semitones ist symmetrisch und in [0,6]."""
        for a in range(12):
            for b in range(12):
                shift = metric._key_shift_semitones(a, b)
                assert 0 <= shift <= 6, f"shift({a},{b}) = {shift} out of [0,6]"
                assert shift == metric._key_shift_semitones(b, a)

    def test_no_reference_mode_returns_valid_score(self, metric):
        """Referenz-freier Modus gibt Score in [0,1]."""
        audio = self._sine_for_key(440.0)
        result = metric.measure(audio, self.SR, reference=None)
        assert 0.0 <= result <= 1.0

    def test_rms_profile_vectorised_matches_loop(self):
        """Vektorisierter _rms_profile liefert identische Werte wie die Schleifen-Variante."""
        import numpy as _np

        from backend.core.musical_goals.musical_goals_metrics import MicroDynamicsMetric

        m = MicroDynamicsMetric()
        sr = 48000
        audio = _np.random.randn(sr * 3).astype(_np.float32)
        win = int(sr * 0.4)
        result = m._rms_profile(audio, win)
        # Sanity: matches naive loop
        n_frames = len(audio) // win
        expected = _np.array(
            [float(_np.sqrt(_np.mean(audio[i * win : (i + 1) * win] ** 2) + 1e-10)) for i in range(n_frames)],
            dtype=_np.float32,
        )
        _np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestH2H4WarmthOvertone:
    """Tests für WaermeMetric._h2h4_warmth — Even-Harmonic-Bias als Röhren/Tape-Wärme-Proxy."""

    SR = 48000

    def _tube_warm_signal(self, dur: float = 1.5) -> np.ndarray:
        """Signal with strong even harmonics: H2=0.30, H4=0.15 vs H3=0.01, H5=0.005."""
        n = int(self.SR * dur)
        t = np.linspace(0, dur, n, endpoint=False)
        sig = (
            np.sin(2 * np.pi * 200 * t)  # H1 = 1.0
            + 0.30 * np.sin(2 * np.pi * 400 * t)  # H2 = 0.30
            + 0.01 * np.sin(2 * np.pi * 600 * t)  # H3 = 0.01
            + 0.15 * np.sin(2 * np.pi * 800 * t)  # H4 = 0.15
            + 0.005 * np.sin(2 * np.pi * 1000 * t)  # H5 = 0.005
        )
        return (sig / (np.max(np.abs(sig)) + 1e-10)).astype(np.float32)

    def test_method_exists(self):
        """WaermeMetric._h2h4_warmth ist als statische Methode vorhanden."""
        from backend.core.musical_goals.musical_goals_metrics import WaermeMetric

        assert callable(WaermeMetric._h2h4_warmth)

    def test_tube_warm_signal_high_score(self):
        """Signal mit starkem H2/H4-Even-Harmonic-Bias erzielt Score ≥ 0.5."""
        from backend.core.musical_goals.musical_goals_metrics import WaermeMetric

        audio = self._tube_warm_signal()
        score = WaermeMetric._h2h4_warmth(audio, self.SR)
        assert score >= 0.5, f"Tube-warm signal: expected ≥ 0.5, got {score:.3f}"

    def test_white_noise_low_score(self):
        """Weißes Rauschen (even ≈ odd Harmonics) erzielt Score ≤ 0.15."""
        from backend.core.musical_goals.musical_goals_metrics import WaermeMetric

        rng = np.random.default_rng(seed=42)
        noise = rng.standard_normal(self.SR * 2).astype(np.float32)
        score = WaermeMetric._h2h4_warmth(noise, self.SR)
        assert score <= 0.15, f"White noise: expected ≤ 0.15, got {score:.3f}"

    def test_score_in_range(self):
        """Ausgabe liegt immer in [0, 1]."""
        from backend.core.musical_goals.musical_goals_metrics import WaermeMetric

        for seed in range(5):
            rng = np.random.default_rng(seed)
            audio = rng.standard_normal(self.SR).astype(np.float32)
            s = WaermeMetric._h2h4_warmth(audio, self.SR)
            assert 0.0 <= s <= 1.0, f"Score {s:.3f} outside [0,1] for seed={seed}"

    def test_short_signal_returns_neutral(self):
        """Zu kurzes Signal (< 512 Samples) liefert neutralen Prior 0.5."""
        from backend.core.musical_goals.musical_goals_metrics import WaermeMetric

        score = WaermeMetric._h2h4_warmth(np.zeros(100, dtype=np.float32), self.SR)
        assert score == 0.5

    def test_tube_warm_beats_noise(self):
        """Röhren-warmes Signal erzielt höheren Score als weißes Rauschen."""
        from backend.core.musical_goals.musical_goals_metrics import WaermeMetric

        warm = self._tube_warm_signal()
        rng = np.random.default_rng(seed=99)
        noise = rng.standard_normal(len(warm)).astype(np.float32)
        assert WaermeMetric._h2h4_warmth(warm, self.SR) > WaermeMetric._h2h4_warmth(noise, self.SR)

    def test_waerme_measure_integrates_h2h4(self):
        """WaermeMetric.measure() integriert H2/H4 ohne Absturz, Score in [0, 1]."""
        from backend.core.musical_goals.musical_goals_metrics import WaermeMetric

        m = WaermeMetric()
        audio = self._tube_warm_signal()
        score = m.measure(audio, self.SR)
        assert 0.0 <= score <= 1.0, f"WaermeMetric.measure out of range: {score:.3f}"


class TestSeparationFidelitySIRProxy:
    """Tests für den SIR-Proxy in SeparationFidelityMetric._reference_based."""

    SR = 48000

    def _sine(self, freq: float, dur: float = 1.0) -> np.ndarray:
        t = np.linspace(0, dur, int(self.SR * dur), endpoint=False)
        return np.sin(2 * np.pi * freq * t).astype(np.float32)

    def test_perfect_restoration_score_near_1(self):
        """Identische restored/reference → Score ≥ 0.95."""
        from backend.core.musical_goals.musical_goals_metrics import SeparationFidelityMetric

        m = SeparationFidelityMetric()
        ref = self._sine(200.0)
        score = m._reference_based(ref.copy(), ref)
        assert score >= 0.95, f"Perfect restoration: expected ≥ 0.95, got {score:.3f}"

    def test_periodic_interference_reduces_score(self):
        """Periodische Interferenz (Frequenz-Leakage) senkt den Score vs. perfekter Restaurierung."""
        from backend.core.musical_goals.musical_goals_metrics import SeparationFidelityMetric

        m = SeparationFidelityMetric()
        ref = self._sine(200.0)
        restored_int = (ref + 0.5 * self._sine(440.0)).astype(np.float32)
        score_perfect = m._reference_based(ref.copy(), ref)
        score_interference = m._reference_based(restored_int, ref)
        assert score_perfect > score_interference, (
            f"Interference should reduce score: perfect={score_perfect:.3f} interference={score_interference:.3f}"
        )

    def test_score_range(self):
        """Score liegt immer in [0, 1]."""
        from backend.core.musical_goals.musical_goals_metrics import SeparationFidelityMetric

        m = SeparationFidelityMetric()
        rng = np.random.default_rng(seed=7)
        for _ in range(5):
            ref = rng.standard_normal(self.SR).astype(np.float32)
            restored = rng.standard_normal(self.SR).astype(np.float32)
            assert 0.0 <= m._reference_based(restored, ref) <= 1.0

    def test_formula_weights_perfect_case(self):
        """Gewichtete Summe (0.40+0.35+0.25) ergibt ≈ 1.0 bei perfekter Restaurierung."""
        from backend.core.musical_goals.musical_goals_metrics import SeparationFidelityMetric

        m = SeparationFidelityMetric()
        ref = self._sine(200.0)
        score = m._reference_based(ref.copy(), ref)
        assert abs(score - 1.0) < 0.05, f"Weighted sum for perfect restoration should be ≈ 1.0, got {score:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
