"""Unit-Tests: ModulationNoiseReductionPhase._compute_modulation_noise_profile() (§2.56)."""

import numpy as np

from backend.core.phases.phase_59_modulation_noise_reduction import ModulationNoiseReductionPhase


def _profile(material: str, qm: str = "balanced", rest: float = 50.0) -> dict:
    return ModulationNoiseReductionPhase._compute_modulation_noise_profile(material, qm, rest)


def test_tape_more_sensitive_than_cd():
    tape = _profile("tape")
    cd = _profile("cd_digital")
    assert tape["min_modulation_noise_score"] < cd["min_modulation_noise_score"]


def test_quality_mode_adjustments():
    base = _profile("tape", "balanced", 60.0)
    q = _profile("tape", "quality", 60.0)
    assert q["min_modulation_noise_score"] < base["min_modulation_noise_score"]
    assert q["g_floor"] < base["g_floor"]


def test_fast_mode_adjustments():
    base = _profile("tape", "balanced", 60.0)
    fast = _profile("tape", "fast", 60.0)
    assert fast["min_modulation_noise_score"] > base["min_modulation_noise_score"]
    assert fast["g_floor"] > base["g_floor"]


def test_low_restorability_adjustments():
    high_rest = _profile("tape", "balanced", 80.0)
    low_rest = _profile("tape", "balanced", 20.0)
    assert low_rest["min_modulation_noise_score"] < high_rest["min_modulation_noise_score"]
    assert low_rest["g_floor"] >= high_rest["g_floor"]


def test_profile_bounds():
    for material in ["tape", "cassette", "cd_digital", "unknown"]:
        for qm in ["balanced", "quality", "maximum", "fast", None]:
            for rest in [5.0, 50.0, 95.0]:
                p = _profile(material, qm, rest)
                assert 0.05 <= p["min_modulation_noise_score"] <= 0.25
                assert 0.02 <= p["g_floor"] <= 0.30


def test_process_metadata_contains_profile():
    phase = ModulationNoiseReductionPhase()
    audio = np.random.uniform(-0.2, 0.2, 48000).astype(np.float32)

    result = phase.process(
        audio,
        sample_rate=48000,
        quality_mode="quality",
        restorability_score=35.0,
        material_type="tape",
        defect_scores={"modulation_noise": 0.2},
        strength=0.5,
    )

    assert result.success
    assert "modulation_noise_profile" in result.metadata
    assert "min_modulation_noise_score" in result.metadata
    assert "g_floor" in result.metadata
