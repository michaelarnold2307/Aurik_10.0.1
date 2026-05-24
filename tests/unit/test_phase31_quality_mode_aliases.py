import numpy as np

from backend.core.defect_scanner import MaterialType
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


def test_phase31_accepts_material_enum_inputs(monkeypatch):
    phase = SpeedPitchCorrectionPhase()

    def _fake_detect_pitch_pyin(_audio, _params):
        return 440.0, 0.95

    def _fake_tuning_offset(_audio, _sample_rate, _reference_pitch, _detected_pitch):
        return 24.0, 1.02

    monkeypatch.setattr(phase, "_detect_pitch_pyin", _fake_detect_pitch_pyin)
    monkeypatch.setattr(phase, "_compute_tuning_offset", _fake_tuning_offset)
    monkeypatch.setattr(phase, "_correct_wsola", lambda audio, ratio, params: np.asarray(audio, dtype=np.float64))

    result = phase.process(
        _test_audio(),
        material_type=MaterialType.TAPE,
        reference_pitch=440.0,
        sample_rate=48000,
        quality_mode="fast",
    )

    assert result.metadata.get("material_type") == "tape"
    assert result.success is True


def test_phase31_maps_cassette_alias_to_tape_profile(monkeypatch):
    phase = SpeedPitchCorrectionPhase()
    captured = {"params": None}

    def _fake_detect_pitch_pyin(_audio, _params):
        captured["params"] = dict(_params)
        return 440.0, 0.95

    def _fake_tuning_offset(_audio, _sample_rate, _reference_pitch, _detected_pitch):
        return 24.0, 1.02

    monkeypatch.setattr(phase, "_detect_pitch_pyin", _fake_detect_pitch_pyin)
    monkeypatch.setattr(phase, "_compute_tuning_offset", _fake_tuning_offset)
    monkeypatch.setattr(phase, "_correct_wsola", lambda audio, ratio, params: np.asarray(audio, dtype=np.float64))

    result = phase.process(
        _test_audio(),
        material_type="cassette",
        reference_pitch=440.0,
        sample_rate=48000,
        quality_mode="fast",
    )

    assert captured["params"] is not None
    assert float(captured["params"]["max_speed_error"]) == float(phase.MATERIAL_PARAMS["tape"]["max_speed_error"])
    assert result.metadata.get("material_type") == "tape"
