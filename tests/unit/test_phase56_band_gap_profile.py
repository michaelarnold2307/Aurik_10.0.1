"""Unit-Tests: SpectralBandGapRepairPhase._compute_band_gap_profile() (§2.56)."""

import numpy as np

from backend.core.phases.phase_56_spectral_band_gap_repair import SpectralBandGapRepairPhase


def _profile(material: str, qm: str = "balanced", rest: float = 50.0) -> dict:
    return SpectralBandGapRepairPhase._compute_band_gap_profile(material, qm, rest)


def test_tape_has_lower_confidence_gate_than_cd():
    tape = _profile("tape")
    cd = _profile("cd_digital")
    assert tape["min_head_wear_confidence"] < cd["min_head_wear_confidence"]


def test_quality_mode_more_sensitive_than_balanced():
    base = _profile("tape", "balanced", 60.0)
    quality = _profile("tape", "quality", 60.0)
    assert quality["min_head_wear_confidence"] < base["min_head_wear_confidence"]
    assert quality["mid_gap_fraction_min"] < base["mid_gap_fraction_min"]


def test_fast_mode_more_conservative_than_balanced():
    base = _profile("tape", "balanced", 60.0)
    fast = _profile("tape", "fast", 60.0)
    assert fast["min_head_wear_confidence"] > base["min_head_wear_confidence"]
    assert fast["mid_gap_fraction_min"] > base["mid_gap_fraction_min"]


def test_profile_bounds():
    p = _profile("unknown", "maximum", 10.0)
    assert 0.40 <= p["min_head_wear_confidence"] <= 0.85
    assert 0.70 <= p["mid_gap_fraction_min"] <= 0.97
    assert 0.85 <= p["side_gap_fraction_min"] <= 0.995


def test_process_metadata_contains_band_gap_profile():
    phase = SpectralBandGapRepairPhase()
    audio = np.random.uniform(-0.1, 0.1, 4096).astype(np.float32)

    result = phase.process(
        audio,
        sample_rate=48000,
        confidence=1.0,
        quality_mode="quality",
        restorability_score=35.0,
        material_type="tape",
        strength=0.5,
    )

    assert result.success
    assert "band_gap_profile" in result.metadata
    assert "min_head_wear_confidence" in result.metadata
    assert "mid_gap_fraction_min" in result.metadata
    assert "side_gap_fraction_min" in result.metadata
