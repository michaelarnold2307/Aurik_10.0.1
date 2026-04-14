"""Unit-Tests fuer deterministische Fallback-Quality-Floor-Entscheidung (Paket 8).

Scope:
- Determinismus der UV3-Endentscheidung bei identischer Heavy-Fallback-Lage.
- Keine Seiteneffekte auf Eingangsdaten.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.core.unified_restorer_v3 import UnifiedRestorerV3


@pytest.mark.unit
@pytest.mark.timeout(20)
def test_fallback_quality_floor_is_deterministic_for_same_inputs() -> None:
    fallbacks = [
        {"phase": "phase_06_frequency_restoration", "fallback": "SBR"},
        {"phase": "phase_43_ml_deesser", "fallback": "OMLSA"},
    ]
    hpi = SimpleNamespace(passed=True, hpi=0.213)

    first = UnifiedRestorerV3._evaluate_fallback_quality_floor(
        ml_fallbacks_used=fallbacks,
        hpi_result=hpi,
        artifact_freedom=0.97,
    )
    second = UnifiedRestorerV3._evaluate_fallback_quality_floor(
        ml_fallbacks_used=fallbacks,
        hpi_result=hpi,
        artifact_freedom=0.97,
    )

    assert first == second
    assert first["triggered"] is True
    assert first["passed"] is True
    assert first["status"] == "passed"
    assert first["fallback_count"] == 2


@pytest.mark.unit
@pytest.mark.timeout(20)
def test_fallback_quality_floor_does_not_mutate_input_list() -> None:
    fallbacks = [{"phase": "phase_23_spectral_repair", "fallback": "harmonic"}]
    snapshot = list(fallbacks)

    _ = UnifiedRestorerV3._evaluate_fallback_quality_floor(
        ml_fallbacks_used=fallbacks,
        hpi_result=SimpleNamespace(passed=True, hpi=0.2),
        artifact_freedom=0.99,
    )

    assert fallbacks == snapshot


@pytest.mark.unit
@pytest.mark.timeout(20)
def test_fallback_quality_floor_requires_both_artifact_and_hpi() -> None:
    fallbacks = [{"phase": "phase_06_frequency_restoration", "fallback": "SBR"}]

    fail_artifact = UnifiedRestorerV3._evaluate_fallback_quality_floor(
        ml_fallbacks_used=fallbacks,
        hpi_result=SimpleNamespace(passed=True, hpi=0.33),
        artifact_freedom=0.94,
    )
    fail_hpi = UnifiedRestorerV3._evaluate_fallback_quality_floor(
        ml_fallbacks_used=fallbacks,
        hpi_result=SimpleNamespace(passed=False, hpi=-0.01),
        artifact_freedom=0.99,
    )

    assert fail_artifact["status"] == "degraded"
    assert fail_hpi["status"] == "degraded"
    assert fail_artifact["reason"] == "hpi_or_artifact_floor_failed"
    assert fail_hpi["reason"] == "hpi_or_artifact_floor_failed"


@pytest.mark.unit
@pytest.mark.timeout(20)
def test_fallback_quality_floor_not_triggered_without_ml_fallbacks() -> None:
    result = UnifiedRestorerV3._evaluate_fallback_quality_floor(
        ml_fallbacks_used=[],
        hpi_result=SimpleNamespace(passed=False, hpi=-1.0),
        artifact_freedom=0.10,
    )

    assert result["triggered"] is False
    assert result["passed"] is True
    assert result["status"] == "passed"
    assert result["fallback_count"] == 0


@pytest.mark.unit
@pytest.mark.timeout(20)
def test_fallback_quality_floor_handles_non_finite_hpi_stably() -> None:
    result = UnifiedRestorerV3._evaluate_fallback_quality_floor(
        ml_fallbacks_used=[{"phase": "phase_43_ml_deesser", "fallback": "bypass"}],
        hpi_result=SimpleNamespace(passed=False, hpi=float("nan")),
        artifact_freedom=0.98,
    )

    assert result["hpi"] is None
    assert result["status"] == "degraded"
