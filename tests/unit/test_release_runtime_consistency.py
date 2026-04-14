from __future__ import annotations

import json
from pathlib import Path

from audit.release_runtime_consistency import consolidate


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_consolidate_detects_contradiction(tmp_path: Path) -> None:
    release = tmp_path / "release_report.json"
    runtime = tmp_path / "runtime_spec_report.json"
    output = tmp_path / "consolidated.json"

    _write_json(
        release,
        {
            "timestamp": "2026-04-12T09:04:50.198941",
            "release_ready": True,
            "compliance_ok": True,
        },
    )
    _write_json(
        runtime,
        {
            "timestamp": "2026-04-14T17:48:50.472080",
            "compliance_ok": False,
            "required_passed": 8,
            "required_total": 9,
        },
    )

    report = consolidate(str(release), str(runtime), str(output))

    assert report["contradiction"] is True
    assert report["final_ready"] is False
    assert "release_runtime_contradiction" in report["reasons"]
    assert report["latest_source"] == "runtime"


def test_consolidate_green_only_when_both_green(tmp_path: Path) -> None:
    release = tmp_path / "release_report.json"
    runtime = tmp_path / "runtime_spec_report.json"
    output = tmp_path / "consolidated.json"

    _write_json(
        release,
        {
            "timestamp": "2026-04-14T18:00:00",
            "release_ready": True,
            "compliance_ok": True,
        },
    )
    _write_json(
        runtime,
        {
            "timestamp": "2026-04-14T18:00:01",
            "compliance_ok": True,
            "required_passed": 9,
            "required_total": 9,
        },
    )

    report = consolidate(str(release), str(runtime), str(output))

    assert report["contradiction"] is False
    assert report["final_ready"] is True
    assert report["latest_source"] == "runtime"


def test_consolidate_handles_missing_runtime_report(tmp_path: Path) -> None:
    release = tmp_path / "release_report.json"
    runtime = tmp_path / "runtime_spec_report.json"
    output = tmp_path / "consolidated.json"

    _write_json(
        release,
        {
            "timestamp": "2026-04-14T18:00:00",
            "release_ready": True,
            "compliance_ok": True,
        },
    )

    report = consolidate(str(release), str(runtime), str(output))

    assert report["final_ready"] is False
    assert "runtime_report_missing_or_invalid" in report["reasons"]
