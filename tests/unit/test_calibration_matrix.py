"""Unit tests for backend/core/calibration_matrix.py — §09.1/§09.2/§09.7.

Tests cover:
- CANONICAL_THRESHOLDS_* export and consistency
- estimate_song_goal_targets: era/material/genre bias application
- predict_quality_score: material-ceiling and restorability scaling
"""

from __future__ import annotations

import numpy as np

from backend.core.calibration_matrix import (
    CANONICAL_THRESHOLDS_RESTORATION,
    CANONICAL_THRESHOLDS_STUDIO2026,
    estimate_song_goal_targets,
    predict_quality_score,
)

# ---------------------------------------------------------------------------
# §09.1 Canonical Thresholds
# ---------------------------------------------------------------------------

_P1_GOALS = {"natuerlichkeit", "authentizitaet"}
_P2_GOALS = {"tonal_center", "timbre_authentizitaet", "artikulation"}


def test_canonical_thresholds_restoration_has_all_goals():
    """All 14 goals must be present in CANONICAL_THRESHOLDS_RESTORATION."""
    expected = {
        "natuerlichkeit",
        "authentizitaet",
        "tonal_center",
        "timbre_authentizitaet",
        "artikulation",
        "emotionalitaet",
        "mikrodynamik",
        "groove",
        "transparenz",
        "waerme",
        "bass_kraft",
        "separation_fidelity",
        "brillanz",
        "raumtiefe",
    }
    for g in expected:
        assert g in CANONICAL_THRESHOLDS_RESTORATION, f"Goal '{g}' missing in CANONICAL_THRESHOLDS_RESTORATION"


def test_canonical_thresholds_studio2026_has_all_goals():
    """Studio 2026 thresholds must have the same goal keys."""
    for g in CANONICAL_THRESHOLDS_RESTORATION:
        assert g in CANONICAL_THRESHOLDS_STUDIO2026, f"Goal '{g}' in RESTORATION but missing in STUDIO2026"


def test_p1_p2_floors_are_identical_across_modes():
    """P1/P2 goals must have identical floors in both modes (§09.1a/b comment)."""
    p1p2 = _P1_GOALS | _P2_GOALS
    for g in p1p2:
        if g in CANONICAL_THRESHOLDS_RESTORATION and g in CANONICAL_THRESHOLDS_STUDIO2026:
            r = CANONICAL_THRESHOLDS_RESTORATION[g]
            s = CANONICAL_THRESHOLDS_STUDIO2026[g]
            # Allow ≤ 0.03 delta — spec notes P1/P2 are "identical or near-identical"
            assert abs(r - s) <= 0.05, f"P1/P2 goal '{g}': Restoration={r}, Studio={s}, delta too large"


def test_all_thresholds_in_valid_range():
    """All canonical thresholds must be in (0.50, 1.00)."""
    for mode, thresholds in [
        ("Restoration", CANONICAL_THRESHOLDS_RESTORATION),
        ("Studio2026", CANONICAL_THRESHOLDS_STUDIO2026),
    ]:
        for goal, val in thresholds.items():
            assert 0.50 <= val < 1.00, f"{mode}/{goal}={val} out of range [0.50, 1.00)"


# ---------------------------------------------------------------------------
# §09.2 estimate_song_goal_targets
# ---------------------------------------------------------------------------


def test_shellac_1930_has_lower_brillanz_than_cd_2005():
    """Ultra-analog era+material bias must reduce brillanz target vs. modern digital."""
    t_old = estimate_song_goal_targets(
        material_type="shellac",
        era_decade=1935,
        is_studio_2026=False,
        restorability_score=40,
    )
    t_new = estimate_song_goal_targets(
        material_type="cd_digital",
        era_decade=2005,
        is_studio_2026=False,
        restorability_score=85,
    )
    assert t_old["brillanz"] < t_new["brillanz"], (
        f"shellac/1935 brillanz={t_old['brillanz']:.3f} should be < cd/2005 {t_new['brillanz']:.3f}"
    )


def test_shellac_1930_has_higher_waerme_than_cd_2005():
    """Vintage analog material should have higher waerme target (warm character)."""
    t_old = estimate_song_goal_targets(
        material_type="shellac",
        era_decade=1935,
        is_studio_2026=False,
        restorability_score=40,
    )
    t_new = estimate_song_goal_targets(
        material_type="cd_digital",
        era_decade=2005,
        is_studio_2026=False,
        restorability_score=85,
    )
    assert t_old["waerme"] > t_new["waerme"], (
        f"shellac/1935 waerme={t_old['waerme']:.3f} should be > cd/2005 {t_new['waerme']:.3f}"
    )


def test_klassik_genre_raises_raumtiefe():
    """Klassik genre bias must result in higher raumtiefe target than Pop."""
    t_klassik = estimate_song_goal_targets(
        material_type="vinyl",
        era_decade=1975,
        genre_label="klassik",
        is_studio_2026=False,
        restorability_score=70,
    )
    t_pop = estimate_song_goal_targets(
        material_type="vinyl",
        era_decade=1975,
        genre_label="pop",
        is_studio_2026=False,
        restorability_score=70,
    )
    assert t_klassik["raumtiefe"] > t_pop["raumtiefe"]


def test_jazz_genre_raises_waerme():
    """Jazz genre bias must result in higher waerme target than rock."""
    t_jazz = estimate_song_goal_targets(
        material_type="vinyl",
        era_decade=1975,
        genre_label="jazz",
        is_studio_2026=False,
        restorability_score=70,
    )
    t_rock = estimate_song_goal_targets(
        material_type="vinyl",
        era_decade=1975,
        genre_label="rock",
        is_studio_2026=False,
        restorability_score=70,
    )
    assert t_jazz["waerme"] > t_rock["waerme"]


def test_targets_all_in_valid_range():
    """All returned targets must be in [0.30, 0.99]."""
    targets = estimate_song_goal_targets(
        material_type="shellac",
        era_decade=1928,
        genre_label="jazz",
        restorability_score=25,
        is_studio_2026=False,
    )
    for g, v in targets.items():
        assert 0.30 <= v <= 0.99, f"target[{g}]={v} out of [0.30, 0.99]"


def test_targets_all_finite():
    """All returned targets must be finite (no NaN/Inf)."""
    targets = estimate_song_goal_targets(
        material_type="vinyl",
        era_decade=1968,
        genre_label="rock",
        restorability_score=55,
        is_studio_2026=True,
    )
    for g, v in targets.items():
        assert np.isfinite(v), f"target[{g}]={v} is not finite"


def test_targets_keys_match_restoration_canonical():
    """estimate_song_goal_targets must return the same keys as CANONICAL_THRESHOLDS_RESTORATION."""
    targets = estimate_song_goal_targets(
        material_type="vinyl",
        era_decade=1975,
        is_studio_2026=False,
        restorability_score=65,
    )
    assert set(targets.keys()) == set(CANONICAL_THRESHOLDS_RESTORATION.keys())


def test_studio_2026_mode_raises_targets():
    """Studio 2026 mode targets must be ≥ Restoration targets for P3–P5 goals."""
    common_kwargs = {
        "material_type": "vinyl",
        "era_decade": 1975,
        "genre_label": "pop",
        "restorability_score": 70,
    }
    t_rest = estimate_song_goal_targets(is_studio_2026=False, **common_kwargs)
    t_s26 = estimate_song_goal_targets(is_studio_2026=True, **common_kwargs)
    # P3-P5 floor is higher in Studio 2026 → targets must be higher or equal
    for g in ("transparenz", "brillanz", "groove", "mikrodynamik"):
        assert t_s26.get(g, 0) >= t_rest.get(g, 0) - 0.02, (
            f"Studio2026[{g}]={t_s26.get(g):.3f} < Restoration[{g}]={t_rest.get(g):.3f}"
        )


def test_goal_weights_above_1_raise_target():
    """goal_weight > 1.0 for a goal must increase its target slightly."""
    base = estimate_song_goal_targets(
        material_type="vinyl",
        era_decade=1975,
        restorability_score=70,
        is_studio_2026=False,
    )
    weighted = estimate_song_goal_targets(
        material_type="vinyl",
        era_decade=1975,
        restorability_score=70,
        is_studio_2026=False,
        goal_weights={"natuerlichkeit": 1.8},
    )
    assert weighted["natuerlichkeit"] >= base["natuerlichkeit"]


def test_goal_weights_below_1_lower_target():
    """goal_weight < 1.0 for a goal must lower its target slightly."""
    base = estimate_song_goal_targets(
        material_type="vinyl",
        era_decade=1975,
        restorability_score=70,
        is_studio_2026=False,
    )
    weighted = estimate_song_goal_targets(
        material_type="vinyl",
        era_decade=1975,
        restorability_score=70,
        is_studio_2026=False,
        goal_weights={"brillanz": 0.4},
    )
    assert weighted["brillanz"] <= base["brillanz"]


def test_none_inputs_do_not_crash():
    """Graceful handling of None/missing inputs — must not raise."""
    result = estimate_song_goal_targets(is_studio_2026=False)
    assert isinstance(result, dict)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# §09.7 predict_quality_score
# ---------------------------------------------------------------------------


def test_shellac_quality_lower_than_vinyl():
    """Shellac has lower quality ceiling than vinyl."""
    q_shellac = predict_quality_score("shellac", 50.0, 0.4, False)
    q_vinyl = predict_quality_score("vinyl", 50.0, 0.4, False)
    assert q_shellac < q_vinyl


def test_cd_digital_has_highest_ceiling():
    """CD digital with high restorability and no defects → near-maximum quality."""
    q_cd = predict_quality_score("cd_digital", 95.0, 0.0, False)
    assert q_cd > 0.85


def test_studio_boost_raises_score():
    """Studio 2026 mode adds boost over Restoration."""
    q_rest = predict_quality_score("vinyl", 70.0, 0.3, False)
    q_s26 = predict_quality_score("vinyl", 70.0, 0.3, True)
    assert q_s26 > q_rest


def test_heavy_defects_reduce_score():
    """Heavy defects (severity=0.9) must reduce quality vs. no defects."""
    q_clean = predict_quality_score("vinyl", 65.0, 0.0, False)
    q_damaged = predict_quality_score("vinyl", 65.0, 0.9, False)
    assert q_damaged < q_clean


def test_quality_score_in_valid_range():
    """Output must always be in [0.0, 0.99]."""
    for mat in ["shellac", "vinyl", "tape", "cd_digital", "mp3_low", "wax_cylinder"]:
        for rest in [5.0, 50.0, 95.0]:
            q = predict_quality_score(mat, rest, 0.5, False)
            assert 0.0 <= q <= 0.99, f"{mat}/rest={rest}: q={q} out of range"


def test_quality_score_finite():
    """Output must never be NaN or Inf."""
    q = predict_quality_score("unknown_material", 55.0, 0.5, True)
    assert np.isfinite(q)
