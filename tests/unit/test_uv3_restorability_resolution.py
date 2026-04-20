from types import SimpleNamespace

import numpy as np

from backend.core.unified_restorer_v3 import UnifiedRestorerV3


def test_cached_restorability_score_wins_over_estimator():
    score, source, result = UnifiedRestorerV3._resolve_pmgg_restorability_score(
        cached_result=SimpleNamespace(restorability_score=42.0),
        analysis_audio=np.zeros(64, dtype=np.float32),
        analysis_sample_rate=48000,
        material_key="vinyl",
        fallback_score=65.0,
        estimator_fn=lambda *args, **kwargs: SimpleNamespace(restorability_score=99.0),
    )

    assert score == 42.0
    assert source == "cached"
    assert getattr(result, "restorability_score", None) == 42.0


def test_incomplete_cached_result_falls_back_to_estimator():
    score, source, result = UnifiedRestorerV3._resolve_pmgg_restorability_score(
        cached_result=SimpleNamespace(),
        analysis_audio=np.zeros(64, dtype=np.float32),
        analysis_sample_rate=48000,
        material_key="shellac",
        fallback_score=65.0,
        estimator_fn=lambda *args, **kwargs: SimpleNamespace(restorability_score=37.5),
    )

    assert score == 37.5
    assert source == "estimated"
    assert getattr(result, "restorability_score", None) == 37.5


def test_estimator_failure_uses_neutral_fallback_score():
    def _raise_estimator(*args, **kwargs):
        raise RuntimeError("estimator unavailable")

    score, source, result = UnifiedRestorerV3._resolve_pmgg_restorability_score(
        cached_result=None,
        analysis_audio=np.zeros(64, dtype=np.float32),
        analysis_sample_rate=48000,
        material_key="tape",
        fallback_score=61.0,
        estimator_fn=_raise_estimator,
    )

    assert score == 61.0
    assert source == "fallback"
    assert result is None
