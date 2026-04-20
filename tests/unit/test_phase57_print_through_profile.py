"""Unit-Tests: PrintThroughReductionPhase._compute_print_through_profile() (§2.56)."""

import numpy as np

from backend.core.phases.phase_57_print_through_reduction import PrintThroughReductionPhase


def _profile(material: str, qm: str = "balanced", rest: float = 50.0) -> dict:
    return PrintThroughReductionPhase._compute_print_through_profile(material, qm, rest)


def test_tape_more_sensitive_than_cd():
    tape = _profile("reel_tape")
    cd = _profile("cd_digital")
    assert tape["min_print_through_score"] < cd["min_print_through_score"]


def test_quality_adjustment():
    base = _profile("reel_tape", "balanced", 60.0)
    q = _profile("reel_tape", "quality", 60.0)
    assert q["min_print_through_score"] < base["min_print_through_score"]
    assert q["coherence_floor"] > base["coherence_floor"]


def test_fast_adjustment():
    base = _profile("reel_tape", "balanced", 60.0)
    fast = _profile("reel_tape", "fast", 60.0)
    assert fast["min_print_through_score"] > base["min_print_through_score"]


def test_low_restorability_adjustment():
    high_rest = _profile("reel_tape", "balanced", 80.0)
    low_rest = _profile("reel_tape", "balanced", 20.0)
    assert low_rest["min_print_through_score"] < high_rest["min_print_through_score"]


def test_profile_bounds():
    for material in ["reel_tape", "tape", "cd_digital", "unknown"]:
        p = _profile(material, "maximum", 10.0)
        assert 0.05 <= p["min_print_through_score"] <= 0.30
        assert 0.90 <= p["coherence_floor"] <= 0.99


def test_process_metadata_contains_profile():
    phase = PrintThroughReductionPhase()
    audio = np.random.uniform(-0.2, 0.2, 48000).astype(np.float32)

    result = phase.process(
        audio,
        sample_rate=48000,
        quality_mode="quality",
        restorability_score=35.0,
        material_type="reel_tape",
        defect_scores={"print_through": 0.25},
        strength=0.5,
    )

    assert result.success
    assert "print_through_profile" in result.metadata
    assert "min_print_through_score" in result.metadata
    assert "coherence_floor" in result.metadata
