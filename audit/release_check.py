"""Release readiness check with deterministic 0-10 scoring.

The script validates documented quality gates against audit trail entries,
calculates a release readiness score (0..10), writes a JSON report and returns
an exit code suitable for CI usage.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

SCORE_MAX: float = 10.0
RELEASE_READY_THRESHOLD: float = 9.5


def load_audit_log(audit_path: str = "audit/audit_trail.json") -> list[dict[str, Any]]:
    """Load audit entries from JSON, returning an empty list on missing/invalid files."""
    path = Path(audit_path)
    if not path.exists():
        print("Audit-Log nicht gefunden.")
        return []
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (json.JSONDecodeError, OSError):
        print("Audit-Log konnte nicht gelesen werden.")
        return []

    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]
    return []


def check_compliance(
    audit_data: list[dict[str, Any]],
    doc_gates_path: str = "docs/audit/QUALITY_GATES.md",
    doc_policy_path: str = "policy/policy_engine.py",
) -> tuple[bool, list[str]]:
    """Verify gate documentation and failing gates against audit entries."""
    compliance_ok = True
    changes: list[str] = []

    doc_gates = ""
    gates_path = Path(doc_gates_path)
    if gates_path.exists():
        with gates_path.open("r", encoding="utf-8") as file:
            doc_gates = file.read()

    # Keep a strict policy-file existence check as baseline safety signal.
    if not Path(doc_policy_path).exists():
        compliance_ok = False
        changes.append("Policy-Datei fehlt: policy/policy_engine.py")

    for entry in audit_data:
        results = entry.get("results", {})
        if not isinstance(results, dict):
            continue
        for gate, value in results.items():
            if isinstance(gate, str) and gate not in doc_gates:
                compliance_ok = False
                changes.append(f"Quality-Gate '{gate}' nicht in Dokumentation.")
            if value is False:
                compliance_ok = False
                changes.append(f"Quality-Gate '{gate}' nicht bestanden.")

    return compliance_ok, changes


def _gate_stats(audit_data: list[dict[str, Any]]) -> tuple[int, int]:
    """Return total gate count and number of passed gates from audit entries."""
    total = 0
    passed = 0
    for entry in audit_data:
        results = entry.get("results", {})
        if not isinstance(results, dict):
            continue
        for value in results.values():
            total += 1
            if value is not False:
                passed += 1
    return total, passed


def calculate_release_score(compliance_ok: bool, changes: list[str], audit_data: list[dict[str, Any]]) -> float:
    """Compute release score in [0, 10].

    Rule set:
    - Base score is gate pass-rate scaled to 0..10.
    - Missing documentation and failed gates apply additional penalties.
    - Empty audit log is penalized to avoid false 10/10 reports.
    """
    total_gates, passed_gates = _gate_stats(audit_data)
    if total_gates == 0:
        score = 6.0
    else:
        score = SCORE_MAX * (passed_gates / total_gates)

    undocumented_count = sum(1 for c in changes if "nicht in Dokumentation" in c)
    failed_gate_count = sum(1 for c in changes if "nicht bestanden" in c)
    score -= undocumented_count * 0.2
    score -= failed_gate_count * 0.5

    if not compliance_ok:
        score -= 0.5

    return round(max(0.0, min(SCORE_MAX, score)), 2)


def generate_release_report(
    compliance_ok: bool,
    changes: list[str],
    audit_data: list[dict[str, Any]],
    output_path: str = "audit/release_report.json",
) -> dict[str, Any]:
    """Generate and persist release report JSON."""
    score = calculate_release_score(compliance_ok, changes, audit_data)
    release_ready = compliance_ok and score >= RELEASE_READY_THRESHOLD

    report: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "compliance_ok": compliance_ok,
        "release_ready": release_ready,
        "score": score,
        "score_max": SCORE_MAX,
        "changes": changes,
        "audit_summary": audit_data[-5:] if audit_data else [],
        "gate_stats": {
            "total": _gate_stats(audit_data)[0],
            "passed": _gate_stats(audit_data)[1],
        },
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print(f"Release-Report generiert: {output_path}")
    print(f"Release-Score: {score:.2f}/{SCORE_MAX:.0f}")
    if not release_ready:
        print("WARNUNG: Release nicht freigegeben. Bitte Änderungen prüfen.")
        for change in changes:
            print(f"- {change}")

    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns 0 for release-ready reports, else 1."""
    parser = argparse.ArgumentParser(description="Aurik Release-Check und Score-Berechnung")
    parser.add_argument("--audit-path", default="audit/audit_trail.json")
    parser.add_argument("--gates-doc", default="docs/audit/QUALITY_GATES.md")
    parser.add_argument("--policy-path", default="policy/policy_engine.py")
    parser.add_argument("--output", default="audit/release_report.json")
    args = parser.parse_args(argv)

    audit_data = load_audit_log(args.audit_path)
    compliance_ok, changes = check_compliance(audit_data, args.gates_doc, args.policy_path)
    report = generate_release_report(compliance_ok, changes, audit_data, args.output)
    return 0 if report.get("release_ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
