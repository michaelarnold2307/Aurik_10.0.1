"""Unit-Tests: TapeSpliceRepairPhase._compute_splice_profile() (§2.56)."""

import numpy as np

from backend.core.phases.phase_64_tape_splice_repair import TapeSpliceRepairPhase


def _profile(material: str, qm: str = "balanced", rest: float = 50.0) -> dict:
    return TapeSpliceRepairPhase._compute_splice_profile(material, qm, rest)


def test_tape_more_sensitive_than_cd():
    tape = _profile("tape")
    cd = _profile("cd_digital")
    assert tape["min_splice_score"] < cd["min_splice_score"]


def test_quality_adjustment():
    base = _profile("tape", "balanced", 60.0)
    q = _profile("tape", "quality", 60.0)
    assert q["min_splice_score"] < base["min_splice_score"]
    assert q["crossfade_ms"] > base["crossfade_ms"]


def test_fast_adjustment():
    base = _profile("tape", "balanced", 60.0)
    fast = _profile("tape", "fast", 60.0)
    assert fast["min_splice_score"] > base["min_splice_score"]
    assert fast["crossfade_ms"] < base["crossfade_ms"]


def test_low_restorability_adjustment():
    high_rest = _profile("tape", "balanced", 80.0)
    low_rest = _profile("tape", "balanced", 20.0)
    assert low_rest["min_splice_score"] < high_rest["min_splice_score"]
    assert low_rest["crossfade_ms"] >= high_rest["crossfade_ms"]


def test_profile_bounds():
    for material in ["tape", "reel_tape", "cd_digital", "unknown"]:
        for qm in ["balanced", "quality", "maximum", "fast", None]:
            p = _profile(material, qm, 30.0)
            assert 0.05 <= p["min_splice_score"] <= 0.25
            assert 6.0 <= p["crossfade_ms"] <= 30.0


def test_process_metadata_contains_profile():
    phase = TapeSpliceRepairPhase()
    audio = np.random.uniform(-0.2, 0.2, 48000).astype(np.float32)

    result = phase.process(
        audio,
        sample_rate=48000,
        quality_mode="quality",
        restorability_score=35.0,
        material_type="tape",
        defect_scores={"tape_splice_artifact": 0.2},
        strength=0.5,
    )

    assert result.success
    assert "splice_profile" in result.metadata
    assert "min_splice_score" in result.metadata
    assert "crossfade_ms" in result.metadata
