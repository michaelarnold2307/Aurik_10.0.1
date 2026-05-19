from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from backend.core.real_audio_execution_golden_gate import (
    ExecutionCaseResult,
    ExecutionGateThresholds,
    _evaluate_execution_gate,
    run_real_audio_execution_golden_gate,
)
from backend.core.real_audio_strategy_golden_gate import StrategyCaseResult


def _case(**overrides):
    base = {
        "case_id": "case",
        "required_phases": ("phase_01_click_removal",),
        "planned_phases": ("phase_01_click_removal",),
        "phases_executed": ("phase_01_click_removal",),
        "phases_skipped": (),
        "missing_required_executions": (),
        "forbidden_executed": (),
        "phase_delta_phases": ("phase_01_click_removal",),
        "missing_phase_deltas": (),
        "artifact_freedom": 0.99,
        "artifact_contract_passed": True,
        "hpi": 0.55,
        "hpi_contract_passed": True,
        "vqi": None,
        "vocal_required": False,
        "vocal_contract_passed": True,
        "export_contract_passed": True,
        "export_strategy": "success",
        "export_blocked": False,
        "degradation_status": "ok",
        "fail_reasons": (),
        "runtime_seconds": 1.0,
        "duration_seconds": 2.0,
        "metadata": {},
    }
    base.update(overrides)
    return ExecutionCaseResult(**base)


def test_execution_gate_aggregates_contract_failures() -> None:
    gate = _evaluate_execution_gate(
        [
            _case(
                missing_required_executions=("phase_01_click_removal",),
                missing_phase_deltas=("phase_01_click_removal",),
                artifact_contract_passed=False,
                hpi_contract_passed=False,
                export_contract_passed=False,
            )
        ],
        ExecutionGateThresholds(),
    )

    assert gate.passed is False
    assert gate.phase_execution_recall == 0.0
    assert gate.phase_delta_coverage == 0.0
    assert gate.artifact_contract_rate == 0.0
    assert gate.hpi_contract_rate == 0.0
    assert gate.export_contract_rate == 0.0
    assert gate.fail_reasons


def test_real_audio_execution_gate_with_mocked_uv3(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path
    audio_path = repo_root / "audio.wav"
    audio_path.write_bytes(b"placeholder")
    manifest = repo_root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "target_sample_rate": 48000,
                "execution_thresholds": {
                    "min_phase_execution_recall": 1.0,
                    "min_phase_delta_coverage": 1.0,
                    "min_artifact_contract_rate": 1.0,
                    "min_hpi_contract_rate": 1.0,
                    "min_vocal_contract_rate": 1.0,
                    "min_export_contract_rate": 1.0,
                    "max_runtime_factor": 10.0,
                },
                "cases": [
                    {
                        "case_id": "mock_case",
                        "path": "audio.wav",
                        "material_type": "vinyl",
                        "requires_vocal_gate": True,
                        "description": "Choir restoration case",
                        "required_phases": ["phase_01_click_removal"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    strategy = StrategyCaseResult(
        case_id="mock_case",
        accepted_causes=(),
        cause_top_k=3,
        primary_cause="vinyl_crackle",
        top_causes=("vinyl_crackle",),
        cause_hit=True,
        required_phases=("phase_01_click_removal",),
        missing_required_phases=(),
        forbidden_phases=(),
        forbidden_present=(),
        ordered_before=(),
        order_violations=(),
        reasoner_phases=("phase_01_click_removal",),
        mapper_phases=(),
        combined_phases=("phase_01_click_removal",),
        runtime_seconds=0.1,
        duration_seconds=2.0,
        metadata={},
    )
    monkeypatch.setattr("backend.core.real_audio_execution_golden_gate._scan_strategy_case", lambda *args: strategy)
    monkeypatch.setattr(
        "backend.core.real_audio_execution_golden_gate._load_audio_for_execution",
        lambda *args, **kwargs: (np.zeros(2048, dtype=np.float32), 48000, 2.0),
    )

    class FakeRestorer:
        def __init__(self, config):
            self.config = config

        def restore(self, audio, sample_rate, **kwargs):
            assert kwargs["vocal_material_prior"] is True
            assert kwargs["multi_singer_prior"] is True
            return SimpleNamespace(
                audio=np.zeros(2048, dtype=np.float32),
                phases_executed=["phase_01_click_removal"],
                phases_skipped=[],
                quality_estimate=0.9,
                metadata={
                    "fail_reasons": [],
                    "degradation_status": "ok",
                    "phase_deltas": {"phase_01_click_removal": {"delta": {}}},
                    "artifact_freedom": {"score": 0.99, "passed": True},
                    "holistic_perceptual_gate": {"hpi": 0.55, "passed": True},
                    "vqi": 0.82,
                },
            )

    monkeypatch.setattr("backend.core.real_audio_execution_golden_gate.UnifiedRestorerV3", FakeRestorer)

    report = run_real_audio_execution_golden_gate(
        manifest_path=manifest,
        repo_root=repo_root,
        output_dir=tmp_path / "exports",
    )

    assert report.scanned_cases == 1
    assert report.gate.passed is True
    assert report.gate.phase_execution_recall == 1.0
    assert report.gate.export_contract_rate == 1.0
