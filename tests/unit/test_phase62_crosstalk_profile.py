"""Unit-Tests: CrosstalkCancellationPhase._compute_crosstalk_profile() (§2.56)."""

import numpy as np

from backend.core.phases.phase_62_crosstalk_cancellation import CrosstalkCancellationPhase


def _profile(material: str, qm: str = "balanced", rest: float = 50.0) -> dict:
    return CrosstalkCancellationPhase._compute_crosstalk_profile(material, qm, rest)


def test_vinyl_more_sensitive_than_cd():
    vinyl = _profile("vinyl")
    cd = _profile("cd_digital")
    assert vinyl["min_crosstalk_score"] < cd["min_crosstalk_score"]


def test_quality_adjustment():
    base = _profile("vinyl", "balanced", 60.0)
    q = _profile("vinyl", "quality", 60.0)
    assert q["min_crosstalk_score"] < base["min_crosstalk_score"]
    assert q["alpha_max"] < base["alpha_max"]


def test_fast_adjustment():
    base = _profile("vinyl", "balanced", 60.0)
    fast = _profile("vinyl", "fast", 60.0)
    assert fast["min_crosstalk_score"] > base["min_crosstalk_score"]


def test_low_restorability_adjustment():
    high_rest = _profile("vinyl", "balanced", 80.0)
    low_rest = _profile("vinyl", "balanced", 20.0)
    assert low_rest["min_crosstalk_score"] < high_rest["min_crosstalk_score"]
    assert low_rest["alpha_max"] <= high_rest["alpha_max"]


def test_profile_bounds():
    for material in ["vinyl", "shellac", "cd_digital", "unknown"]:
        p = _profile(material, "maximum", 10.0)
        assert 0.05 <= p["min_crosstalk_score"] <= 0.25
        assert 0.50 <= p["alpha_max"] <= 0.70


def test_process_metadata_contains_profile():
    phase = CrosstalkCancellationPhase()
    audio = np.random.uniform(-0.2, 0.2, (2, 48000)).astype(np.float32)

    result = phase.process(
        audio,
        sample_rate=48000,
        quality_mode="quality",
        restorability_score=35.0,
        material_type="vinyl",
        defect_scores={"crosstalk": 0.2},
        strength=0.5,
    )

    assert result.success
    assert "crosstalk_profile" in result.metadata
    assert "min_crosstalk_score" in result.metadata
    assert "alpha_max" in result.metadata
