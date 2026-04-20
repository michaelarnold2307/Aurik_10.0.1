import numpy as np

from backend.core.phases.phase_31_speed_pitch_correction import SpeedPitchCorrectionPhase


def _test_audio(sr: int = 48000) -> np.ndarray:
    duration_s = 1.0
    t = np.linspace(0.0, duration_s, int(sr * duration_s), endpoint=False)
    return 0.2 * np.sin(2.0 * np.pi * 440.0 * t)


def test_studio_2026_alias_routes_to_maximum_hybrid(monkeypatch):
    phase = SpeedPitchCorrectionPhase()
    called = {"mode": None}

    def _fake_detect_pitch_ml_hybrid(audio, sample_rate, quality_mode):
        called["mode"] = quality_mode
        return 440.0, 0.95, {"strategy": "polyphonic_speed_curve"}

    monkeypatch.setattr(phase, "_detect_pitch_ml_hybrid", _fake_detect_pitch_ml_hybrid)

    result = phase.process(
        _test_audio(),
        material_type="tape",
        reference_pitch=440.0,
        sample_rate=48000,
        quality_mode="studio_2026",
    )

    assert called["mode"] == "maximum"
    assert result.metadata.get("quality_mode") == "maximum"


def test_restoration_alias_routes_to_balanced_hybrid(monkeypatch):
    phase = SpeedPitchCorrectionPhase()
    called = {"mode": None}

    def _fake_detect_pitch_ml_hybrid(audio, sample_rate, quality_mode):
        called["mode"] = quality_mode
        return 440.0, 0.95, {"strategy": "adaptive"}

    monkeypatch.setattr(phase, "_detect_pitch_ml_hybrid", _fake_detect_pitch_ml_hybrid)

    result = phase.process(
        _test_audio(),
        material_type="tape",
        reference_pitch=440.0,
        sample_rate=48000,
        quality_mode="restoration",
    )

    assert called["mode"] == "balanced"
    assert result.metadata.get("quality_mode") == "balanced"
