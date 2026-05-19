from __future__ import annotations

from pathlib import Path

from backend.core.real_audio_strategy_golden_gate import run_real_audio_strategy_golden_gate


def test_real_audio_strategy_golden_manifest_passes_gate() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest = repo_root / "audit" / "real_audio_strategy_golden_manifest.json"

    report = run_real_audio_strategy_golden_gate(manifest_path=manifest, repo_root=repo_root)

    assert report.scanned_cases >= 8
    assert report.skipped_cases == []
    assert report.gate.passed is True
    assert report.gate.cause_topk_accuracy == 1.0
    assert report.gate.phase_recall == 1.0
    assert report.gate.phase_precision == 1.0
    assert report.gate.forbidden_phase_violations == 0
    assert report.gate.order_violations == 0
    assert report.gate.fail_reasons == ()
