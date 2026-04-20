from __future__ import annotations

from backend.core.phases.phase_50_spectral_repair import SpectralRepairPhase


def test_runtime_profile_keys_and_bounds() -> None:
    p = SpectralRepairPhase._compute_threshold_runtime_profile(
        material_key="vinyl",
        quality_mode="quality",
        restorability_score=55.0,
    )

    assert set(p.keys()) == {"strength_floor", "side_multiplier"}
    assert 0.06 <= p["strength_floor"] <= 0.18
    assert 1.60 <= p["side_multiplier"] <= 2.40


def test_fast_mode_more_conservative_than_quality() -> None:
    fast = SpectralRepairPhase._compute_threshold_runtime_profile(
        material_key="cd_digital",
        quality_mode="fast",
        restorability_score=60.0,
    )
    quality = SpectralRepairPhase._compute_threshold_runtime_profile(
        material_key="cd_digital",
        quality_mode="quality",
        restorability_score=60.0,
    )

    assert fast["strength_floor"] > quality["strength_floor"]
    assert fast["side_multiplier"] > quality["side_multiplier"]


def test_low_restorability_relaxes_floor_and_side_multiplier() -> None:
    low = SpectralRepairPhase._compute_threshold_runtime_profile(
        material_key="shellac",
        quality_mode="balanced",
        restorability_score=10.0,
    )
    high = SpectralRepairPhase._compute_threshold_runtime_profile(
        material_key="shellac",
        quality_mode="balanced",
        restorability_score=90.0,
    )

    assert low["strength_floor"] < high["strength_floor"]
    assert low["side_multiplier"] < high["side_multiplier"]
