"""Consolidate release and runtime audit reports into one final status.

This module resolves contradictory top-level signals by enforcing a strict policy:
- final_ready is True only if BOTH release and runtime required checks are green
- contradictions are explicit in output metadata
- newest-run policy based on report timestamps
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ConsolidatedStatus:
    timestamp: str
    release_report_path: str
    runtime_report_path: str
    release_timestamp: str | None
    runtime_timestamp: str | None
    latest_source: str
    release_ready: bool
    runtime_compliance_ok: bool
    required_passed: int
    required_total: int
    contradiction: bool
    final_ready: bool
    reasons: list[str]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _parse_iso(ts: Any) -> datetime | None:
    if not isinstance(ts, str) or not ts.strip():
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def consolidate(
    release_report_path: str = "audit/release_report.json",
    runtime_report_path: str = "audit/runtime_spec_report.json",
    output_path: str = "audit/consolidated_release_status.json",
) -> dict[str, Any]:
    release_path = Path(release_report_path)
    runtime_path = Path(runtime_report_path)

    release = _load_json(release_path)
    runtime = _load_json(runtime_path)

    release_ready = bool(release.get("release_ready", False))
    runtime_ok = bool(runtime.get("compliance_ok", False))

    required_passed = int(runtime.get("required_passed", 0) or 0)
    required_total = int(runtime.get("required_total", 0) or 0)

    release_ts_raw = release.get("timestamp")
    runtime_ts_raw = runtime.get("timestamp")
    release_ts = _parse_iso(release_ts_raw)
    runtime_ts = _parse_iso(runtime_ts_raw)

    if release_ts and runtime_ts:
        latest_source = "runtime" if runtime_ts >= release_ts else "release"
    elif runtime_ts:
        latest_source = "runtime"
    elif release_ts:
        latest_source = "release"
    else:
        latest_source = "unknown"

    contradiction = release_ready != runtime_ok
    reasons: list[str] = []

    if not release:
        reasons.append("release_report_missing_or_invalid")
    if not runtime:
        reasons.append("runtime_report_missing_or_invalid")
    if release and not release_ready:
        reasons.append("release_not_ready")
    if runtime and not runtime_ok:
        reasons.append("runtime_compliance_failed")
    if runtime and required_total > 0 and required_passed < required_total:
        reasons.append(f"runtime_required_failed:{required_passed}/{required_total}")
    if contradiction:
        reasons.append("release_runtime_contradiction")

    final_ready = release_ready and runtime_ok and bool(release) and bool(runtime)

    payload = ConsolidatedStatus(
        timestamp=datetime.now().isoformat(),
        release_report_path=str(release_path),
        runtime_report_path=str(runtime_path),
        release_timestamp=release_ts_raw if isinstance(release_ts_raw, str) else None,
        runtime_timestamp=runtime_ts_raw if isinstance(runtime_ts_raw, str) else None,
        latest_source=latest_source,
        release_ready=release_ready,
        runtime_compliance_ok=runtime_ok,
        required_passed=required_passed,
        required_total=required_total,
        contradiction=contradiction,
        final_ready=final_ready,
        reasons=reasons,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return asdict(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Consolidate release + runtime audit status")
    parser.add_argument("--release-report", default="audit/release_report.json")
    parser.add_argument("--runtime-report", default="audit/runtime_spec_report.json")
    parser.add_argument("--output", default="audit/consolidated_release_status.json")
    args = parser.parse_args(argv)

    report = consolidate(
        release_report_path=args.release_report,
        runtime_report_path=args.runtime_report,
        output_path=args.output,
    )

    print(
        f"Consolidated status: final_ready={report.get('final_ready')} | release_ready={report.get('release_ready')} | runtime_compliance_ok={report.get('runtime_compliance_ok')}"
    )
    if report.get("reasons"):
        print("Reasons:")
        for r in report["reasons"]:
            print(f"- {r}")

    return 0 if report.get("final_ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
