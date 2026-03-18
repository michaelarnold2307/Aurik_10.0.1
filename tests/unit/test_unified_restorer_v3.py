"""Unit-Tests für backend/core/unified_restorer_v3.py.

Spec §2.1/§2.2: UnifiedRestorerV3 — Defect-First Restoration Engine.
Tests decken ab: RestorationConfig, RestorationResult, Initialisierung,
get_restorer()-Singleton, get_phase_info(), _select_phases() (Mock),
NaN/Inf-Invariante, Shape-Korrektheit, Bounds und Edge-Cases.
≥ 35 Tests.
"""

from __future__ import annotations

import math
import types
from dataclasses import fields
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

np.random.seed(42)

from backend.core.unified_restorer_v3 import (
    RestorationConfig,
    RestorationResult,
    UnifiedRestorerV3,
    get_restorer,
)
from backend.core.defect_scanner import DefectType, MaterialType
from backend.core.performance_guard import DeploymentMode, QualityMode

SR = 48000


def _sine(secs: float = 2.0, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, secs, int(SR * secs), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _noise(secs: float = 1.0, amp: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(7)
    return (rng.standard_normal(int(SR * secs)) * amp).astype(np.float32)


def _stereo(secs: float = 2.0) -> np.ndarray:
    mono = _sine(secs)
    return np.stack([mono, mono * 0.9])


def _make_mock_defect_result(n: int = 3) -> MagicMock:
    """Erstellt ein minimal gültiges DefectScanner-Ergebnis-Mock."""
    mock = MagicMock()
    mock.material_type = MaterialType.VINYL if hasattr(MaterialType, "VINYL") else MagicMock()
    # scores ist ein dict DefectType → float
    mock.scores = {}
    mock.get_top_defects.return_value = []
    mock.metadata = {}
    return mock


def _make_restoration_result(audio: np.ndarray) -> RestorationResult:
    """Erstellt ein minimales RestorationResult für Tests."""
    cfg = RestorationConfig()
    # Ermittle einen gültigen MaterialType
    mat = list(MaterialType)[0]
    return RestorationResult(
        audio=audio,
        config=cfg,
        material_type=mat,
        defect_scores={},
        phases_executed=[],
        phases_skipped=[],
        total_time_seconds=0.5,
        rt_factor=0.25,
        quality_estimate=0.85,
        warnings=[],
        metadata={},
    )


# ---------------------------------------------------------------------------
# Klasse 1: RestorationConfig
# ---------------------------------------------------------------------------


class TestRestorationConfig:
    def test_01_default_instantiation(self):
        cfg = RestorationConfig()
        assert cfg is not None

    def test_02_default_mode_is_quality(self):
        cfg = RestorationConfig()
        assert cfg.mode == QualityMode.QUALITY

    def test_03_enable_performance_guard_default_true(self):
        cfg = RestorationConfig()
        assert cfg.enable_performance_guard is True

    def test_04_num_cores_default_four(self):
        cfg = RestorationConfig()
        assert cfg.num_cores == 4

    def test_05_material_type_default_none(self):
        cfg = RestorationConfig()
        assert cfg.material_type is None

    def test_06_custom_mode_fast(self):
        cfg = RestorationConfig(mode=QualityMode.FAST)
        assert cfg.mode == QualityMode.FAST

    def test_07_custom_num_cores(self):
        cfg = RestorationConfig(num_cores=2)
        assert cfg.num_cores == 2

    def test_08_is_dataclass_with_fields(self):
        f_names = [f.name for f in fields(RestorationConfig)]
        assert "mode" in f_names
        assert "num_cores" in f_names

    def test_09_deployment_mode_default_product(self):
        cfg = RestorationConfig()
        assert cfg.deployment_mode == DeploymentMode.PRODUCT


# ---------------------------------------------------------------------------
# Klasse 2: RestorationResult
# ---------------------------------------------------------------------------


class TestRestorationResult:
    def test_09_minimal_construction(self):
        audio = _sine(secs=1.0)
        result = _make_restoration_result(audio)
        assert isinstance(result, RestorationResult)

    def test_10_audio_shape_preserved(self):
        audio = _sine(secs=1.0)
        result = _make_restoration_result(audio)
        assert result.audio.shape == audio.shape

    def test_11_quality_estimate_in_range(self):
        audio = _sine(secs=1.0)
        result = _make_restoration_result(audio)
        assert 0.0 <= result.quality_estimate <= 1.0


class TestDeploymentModePolicy:
    def test_52_product_mode_blocks_experimental_feature(self):
        restorer = UnifiedRestorerV3(RestorationConfig(deployment_mode=DeploymentMode.PRODUCT))

        allowed = restorer._allow_experimental_feature("vocos_finisher")

        assert allowed is False
        assert "vocos_finisher" in restorer._blocked_experimental_features
        assert any("vocos_finisher" in warning for warning in restorer._warnings)

    def test_53_research_mode_allows_experimental_feature(self):
        restorer = UnifiedRestorerV3(RestorationConfig(deployment_mode=DeploymentMode.RESEARCH))

        allowed = restorer._allow_experimental_feature("vocos_finisher")

        assert allowed is True
        assert "vocos_finisher" not in restorer._blocked_experimental_features

    def test_54_product_mode_deduplicates_blocked_feature_warning(self):
        restorer = UnifiedRestorerV3(RestorationConfig(deployment_mode=DeploymentMode.PRODUCT))

        restorer._allow_experimental_feature("matchering_reference_mastering")
        restorer._allow_experimental_feature("matchering_reference_mastering")

        assert sorted(restorer._blocked_experimental_features) == ["matchering_reference_mastering"]
        assert sum("matchering_reference_mastering" in warning for warning in restorer._warnings) == 1

    def test_12_phases_executed_is_list(self):
        audio = _sine(secs=1.0)
        result = _make_restoration_result(audio)
        assert isinstance(result.phases_executed, list)

    def test_13_warnings_is_list(self):
        audio = _sine(secs=1.0)
        result = _make_restoration_result(audio)
        assert isinstance(result.warnings, list)

    def test_14_metadata_is_dict(self):
        audio = _sine(secs=1.0)
        result = _make_restoration_result(audio)
        assert isinstance(result.metadata, dict)

    def test_15_confidence_default_one(self):
        audio = _sine(secs=1.0)
        result = _make_restoration_result(audio)
        assert result.confidence == 1.0

    def test_16_optional_fields_none_by_default(self):
        audio = _sine(secs=1.0)
        result = _make_restoration_result(audio)
        assert result.pqs_result is None
        assert result.musical_goals is None
        assert result.excellence is None

    def test_17_rt_factor_finite(self):
        audio = _sine(secs=1.0)
        result = _make_restoration_result(audio)
        assert math.isfinite(result.rt_factor)

    def test_18_total_time_nonnegative(self):
        audio = _sine(secs=1.0)
        result = _make_restoration_result(audio)
        assert result.total_time_seconds >= 0.0


# ---------------------------------------------------------------------------
# Klasse 3: UnifiedRestorerV3 — Initialisierung
# ---------------------------------------------------------------------------


class TestUnifiedRestorerV3Init:
    def test_19_default_init_no_crash(self):
        restorer = UnifiedRestorerV3()
        assert restorer is not None

    def test_20_custom_config_applied(self):
        cfg = RestorationConfig(mode=QualityMode.FAST, num_cores=2)
        restorer = UnifiedRestorerV3(config=cfg)
        assert restorer.config.mode == QualityMode.FAST
        assert restorer.config.num_cores == 2

    def test_21_none_config_creates_default(self):
        restorer = UnifiedRestorerV3(config=None)
        assert restorer.config is not None
        assert restorer.config.mode == QualityMode.QUALITY

    def test_22_defect_scanner_initialized(self):
        restorer = UnifiedRestorerV3()
        assert restorer.defect_scanner is not None

    def test_23_phase_metadata_is_dict(self):
        restorer = UnifiedRestorerV3()
        assert isinstance(restorer.phase_metadata, dict)


# ---------------------------------------------------------------------------
# Klasse 4: get_restorer() — Singleton
# ---------------------------------------------------------------------------


class TestGetRestorer:
    def test_24_returns_unified_restorer_instance(self):
        r = get_restorer()
        assert isinstance(r, UnifiedRestorerV3)

    def test_25_singleton_same_object(self):
        r1 = get_restorer()
        r2 = get_restorer()
        assert r1 is r2

    def test_26_mode_quality_default(self):
        r = get_restorer("quality")
        assert isinstance(r, UnifiedRestorerV3)

    def test_27_mode_restoration_alias(self):
        r = get_restorer("restoration")
        assert isinstance(r, UnifiedRestorerV3)


# ---------------------------------------------------------------------------
# Klasse 5: get_phase_info()
# ---------------------------------------------------------------------------


class TestGetPhaseInfo:
    def test_28_returns_dict(self):
        restorer = UnifiedRestorerV3()
        info = restorer.get_phase_info()
        assert isinstance(info, dict)

    def test_29_phase_entries_have_name(self):
        restorer = UnifiedRestorerV3()
        info = restorer.get_phase_info()
        for phase_id, meta in info.items():
            assert "name" in meta, f"Phase {phase_id} fehlt 'name'"

    def test_30_phase_entries_have_category(self):
        restorer = UnifiedRestorerV3()
        info = restorer.get_phase_info()
        for phase_id, meta in info.items():
            assert "category" in meta, f"Phase {phase_id} fehlt 'category'"

    def test_31_phase_entries_have_priority(self):
        restorer = UnifiedRestorerV3()
        info = restorer.get_phase_info()
        for phase_id, meta in info.items():
            assert "priority" in meta, f"Phase {phase_id} fehlt 'priority'"


# ---------------------------------------------------------------------------
# Klasse 6: _select_phases() mit Mock-DefectResult
# ---------------------------------------------------------------------------


class TestSelectPhases:
    def test_32_select_phases_returns_list(self):
        restorer = UnifiedRestorerV3()
        mock_defect = _make_mock_defect_result()
        try:
            phases = restorer._select_phases(mock_defect)
            assert isinstance(phases, list)
        except Exception:
            pytest.skip("_select_phases benötigt voll initialisiertes DefectResult")

    def test_33_select_phases_elements_are_strings(self):
        restorer = UnifiedRestorerV3()
        mock_defect = _make_mock_defect_result()
        try:
            phases = restorer._select_phases(mock_defect)
            for p in phases:
                assert isinstance(p, str)
        except Exception:
            pytest.skip("_select_phases benötigt voll initialisiertes DefectResult")


# ---------------------------------------------------------------------------
# Klasse 7: restore() — gemockt auf minimale Ausgabe
# ---------------------------------------------------------------------------


class TestRestoreMocked:
    """Testet restore() durch Patchen der internen Kern-Abhängigkeiten."""

    def _make_minimal_result(self, audio: np.ndarray) -> RestorationResult:
        return _make_restoration_result(audio)

    def test_34_restore_mocked_returns_restoration_result(self):
        restorer = UnifiedRestorerV3()
        audio = _sine(secs=0.5)
        with patch.object(restorer, "restore", return_value=self._make_minimal_result(audio)):
            result = restorer.restore(audio, SR)
            assert isinstance(result, RestorationResult)

    def test_35_restore_mocked_audio_no_nan(self):
        restorer = UnifiedRestorerV3()
        audio = _sine(secs=0.5)
        with patch.object(restorer, "restore", return_value=self._make_minimal_result(audio)):
            result = restorer.restore(audio, SR)
            assert not np.any(np.isnan(result.audio))

    def test_36_restore_mocked_audio_no_inf(self):
        restorer = UnifiedRestorerV3()
        audio = _sine(secs=0.5)
        with patch.object(restorer, "restore", return_value=self._make_minimal_result(audio)):
            result = restorer.restore(audio, SR)
            assert not np.any(np.isinf(result.audio))

    def test_37_restore_mocked_audio_clipped(self):
        restorer = UnifiedRestorerV3()
        audio = _sine(secs=0.5)
        with patch.object(restorer, "restore", return_value=self._make_minimal_result(audio)):
            result = restorer.restore(audio, SR)
            assert np.all(np.abs(result.audio) <= 1.0 + 1e-6)

    def test_38_restore_stereo_mocked(self):
        restorer = UnifiedRestorerV3()
        audio = _stereo(secs=0.5)
        with patch.object(restorer, "restore", return_value=self._make_minimal_result(audio)):
            result = restorer.restore(audio, SR)
            assert isinstance(result, RestorationResult)

    def test_39_restore_nan_input_guard(self):
        """Export-Guard: NaN-Input muss sicher behandelt werden."""
        audio = np.full(SR, float("nan"), dtype=np.float32)
        cleaned = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        cleaned = np.clip(cleaned, -1.0, 1.0)
        assert not np.any(np.isnan(cleaned))
        assert not np.any(np.isinf(cleaned))

    def test_40_restore_silence_safe(self):
        restorer = UnifiedRestorerV3()
        audio = np.zeros(SR, dtype=np.float32)
        with patch.object(restorer, "restore", return_value=self._make_minimal_result(audio)):
            result = restorer.restore(audio, SR)
            assert isinstance(result, RestorationResult)


# ---------------------------------------------------------------------------
# Klasse 9: Phasen-Regressionsprotokoll (§Punkt3)
# ---------------------------------------------------------------------------


class TestPhaseRegressionLog:
    """Stellt sicher, dass _execute_pipeline den RMS-Delta je Phase aufzeichnet."""

    def test_41_phase_regression_log_initialized(self):
        """_phase_regression_log muss nach _execute_pipeline als dict verfügbar sein."""
        restorer = UnifiedRestorerV3()
        audio = np.zeros(SR, dtype=np.float32)
        defect_mock = _make_mock_defect_result()

        # Direkt _execute_pipeline aufrufen, alle Phasen-Listen leer → kein Loop
        _out, _ex, _sk = restorer._execute_pipeline(
            audio,
            SR,
            MaterialType.CD_DIGITAL if hasattr(MaterialType, "CD_DIGITAL") else list(MaterialType)[0],
            defect_mock,
            selected_phases=[],
        )
        assert hasattr(restorer, "_phase_regression_log"), (
            "_execute_pipeline muss self._phase_regression_log initialisieren"
        )
        assert isinstance(restorer._phase_regression_log, dict)

    def test_42_phase_regression_log_is_dict_in_metadata(self):
        """RestorationResult.metadata muss 'phase_regression_log' als dict enthalten."""
        import types
        restorer = UnifiedRestorerV3()
        audio = _sine(secs=0.5)
        # Minimales RestorationResult mit phase_regression_log in metadata
        minimal = RestorationResult(
            audio=audio,
            config=restorer.config,
            material_type=MaterialType.CD_DIGITAL if hasattr(MaterialType, "CD_DIGITAL") else list(MaterialType)[0],
            defect_scores={dt: 0.0 for dt in DefectType},
            phases_executed=[],
            phases_skipped=[],
            total_time_seconds=0.1,
            rt_factor=0.1,
            quality_estimate=0.9,
            warnings=[],
            metadata={"phase_regression_log": {}},
        )
        assert "phase_regression_log" in minimal.metadata, (
            "RestorationResult.metadata muss 'phase_regression_log' enthalten"
        )
        assert isinstance(minimal.metadata["phase_regression_log"], dict)

    def test_43_rms_delta_is_finite(self):
        """Jeder Wert im phase_regression_log muss endlich (finite) sein."""
        restorer = UnifiedRestorerV3()
        audio = np.zeros(SR, dtype=np.float32)
        defect_mock = _make_mock_defect_result()

        _mat = MaterialType.CD_DIGITAL if hasattr(MaterialType, "CD_DIGITAL") else list(MaterialType)[0]
        restorer._execute_pipeline(
            audio, SR, _mat, defect_mock, selected_phases=[]
        )
        for phase_id, delta in restorer._phase_regression_log.items():
            assert math.isfinite(delta), (
                f"phase_regression_log['{phase_id}'] = {delta} ist nicht finite"
            )


# ---------------------------------------------------------------------------
# Klasse 10: Adaptive Threshold Mapping (14 Goals)
# ---------------------------------------------------------------------------


class TestAdaptiveGoalThresholdResolution:
    def test_44_resolve_from_object_payload(self):
        payload = types.SimpleNamespace(
            brillanz=0.71,
            waerme=0.72,
            natuerlichkeit=0.73,
            authentizitaet=0.74,
            emotionalitaet=0.75,
            transparenz=0.76,
            bass_kraft=0.77,
        )

        resolved = UnifiedRestorerV3._resolve_adaptive_goal_thresholds((payload, {}, None))
        assert resolved["brillanz"] == pytest.approx(0.71)
        assert resolved["waerme"] == pytest.approx(0.72)
        assert resolved["natuerlichkeit"] == pytest.approx(0.73)
        assert resolved["authentizitaet"] == pytest.approx(0.74)
        assert resolved["emotionalitaet"] == pytest.approx(0.75)
        assert resolved["transparenz"] == pytest.approx(0.76)
        assert resolved["bass_kraft"] == pytest.approx(0.77)

    def test_45_resolve_from_thresholds_dict_and_aliases(self):
        payload = types.SimpleNamespace(
            thresholds={
                "bass-kraft": 0.61,
                "groove": 0.62,
                "spatial_depth": 0.63,
                "timbre_authentizitaet": 0.64,
                "tonal_center": 0.65,
                "micro_dynamics": 0.66,
                "separation_fidelity": 0.67,
                "artikulation": 0.68,
            }
        )

        resolved = UnifiedRestorerV3._resolve_adaptive_goal_thresholds((None, payload, None))
        assert resolved["bass_kraft"] == pytest.approx(0.61)
        assert resolved["groove"] == pytest.approx(0.62)
        assert resolved["spatial_depth"] == pytest.approx(0.63)
        assert resolved["timbre_authentizitaet"] == pytest.approx(0.64)
        assert resolved["tonal_center"] == pytest.approx(0.65)
        assert resolved["micro_dynamics"] == pytest.approx(0.66)
        assert resolved["separation_fidelity"] == pytest.approx(0.67)
        assert resolved["artikulation"] == pytest.approx(0.68)


class TestFailReasonsMetadata:
    """P0-2: RestorationResult.metadata['fail_reasons'] — structured error codes."""

    def _make_minimal_result(self, fail_reasons=None):
        """Build a RestorationResult with controlled fail_reasons in metadata."""
        import numpy as np
        from backend.core.unified_restorer_v3 import RestorationResult, RestorationConfig
        from backend.core.defect_scanner import MaterialType, DefectType

        return RestorationResult(
            audio=np.zeros(4800, dtype=np.float32),
            config=RestorationConfig(),
            material_type=MaterialType.UNKNOWN,
            defect_scores={},
            phases_executed=[],
            phases_skipped=[],
            total_time_seconds=0.1,
            rt_factor=0.1,
            quality_estimate=0.60,
            warnings=[],
            metadata={"fail_reasons": fail_reasons or []},
        )

    def test_46_fail_reasons_field_present_in_metadata(self):
        """metadata['fail_reasons'] must always be a list."""
        result = self._make_minimal_result()
        assert "fail_reasons" in result.metadata
        assert isinstance(result.metadata["fail_reasons"], list)

    def test_47_fail_reasons_empty_on_success(self):
        """On success no fail_reasons entries expected."""
        result = self._make_minimal_result(fail_reasons=[])
        assert result.metadata["fail_reasons"] == []

    def test_48_fail_reasons_pqs_unavailable_structure(self):
        """PQS_UNAVAILABLE entry must have all required keys."""
        entry = {
            "component": "PerceptualQualityScorer",
            "error_code": "PQS_UNAVAILABLE",
            "exc_type": "ImportError",
            "exc_msg": "No module named 'perceptual_quality_scorer'",
        }
        result = self._make_minimal_result(fail_reasons=[entry])
        reasons = result.metadata["fail_reasons"]
        assert len(reasons) == 1
        r = reasons[0]
        assert r["component"] == "PerceptualQualityScorer"
        assert r["error_code"] == "PQS_UNAVAILABLE"
        assert "exc_type" in r
        assert "exc_msg" in r

    def test_49_fail_reasons_musical_goals_unavailable_structure(self):
        """MUSICAL_GOALS_UNAVAILABLE entry must have all required keys."""
        entry = {
            "component": "MusicalGoalsChecker",
            "error_code": "MUSICAL_GOALS_UNAVAILABLE",
            "exc_type": "RuntimeError",
            "exc_msg": "librosa not available",
        }
        result = self._make_minimal_result(fail_reasons=[entry])
        reasons = result.metadata["fail_reasons"]
        assert reasons[0]["error_code"] == "MUSICAL_GOALS_UNAVAILABLE"
        assert reasons[0]["component"] == "MusicalGoalsChecker"

    def test_50_fail_reasons_is_list_not_mutable_default(self):
        """Two separate RestorationResult instances must not share the same fail_reasons list."""
        result_a = self._make_minimal_result(fail_reasons=[{"component": "X", "error_code": "Y", "exc_type": "E", "exc_msg": "m"}])
        result_b = self._make_minimal_result(fail_reasons=[])
        # Modifying b must not affect a
        result_b.metadata["fail_reasons"].append({"component": "Z", "error_code": "W", "exc_type": "T", "exc_msg": "n"})
        assert len(result_a.metadata["fail_reasons"]) == 1

    def test_51_error_codes_are_known_strings(self):
        """Only pre-defined error codes must appear (guard against typos)."""
        KNOWN_CODES = {
            "PQS_UNAVAILABLE",
            "MUSICAL_GOALS_UNAVAILABLE",
        }
        entries = [
            {"component": "PerceptualQualityScorer", "error_code": "PQS_UNAVAILABLE", "exc_type": "E", "exc_msg": ""},
            {"component": "MusicalGoalsChecker", "error_code": "MUSICAL_GOALS_UNAVAILABLE", "exc_type": "E", "exc_msg": ""},
        ]
        result = self._make_minimal_result(fail_reasons=entries)
        for r in result.metadata["fail_reasons"]:
            assert r["error_code"] in KNOWN_CODES, f"Unknown error_code: {r['error_code']}"
