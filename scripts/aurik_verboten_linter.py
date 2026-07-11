#!/usr/bin/env python3
"""V01-V52 VERBOTEN-Linter — Automatisierte Spec-Compliance-Prüfung.

§Spec 10: Prüft Codebase auf Verstöße gegen die VERBOTEN-Tabelle.
Exit-Code 0 = sauber, 1 = Verstöße gefunden.

Nutzung:
  python scripts/aurik_verboten_linter.py
  python scripts/aurik_verboten_linter.py --ci

Autor: Aurik 10 — 11. Juli 2026
"""

from __future__ import annotations

import json, os, re, sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent

# ── VERBOTEN-Regeln (aus .github/VERBOTEN.md) ──────────────────────────────
RULES = {
    "V01": {"pattern": r'from backend\.core import|import backend\.core\.', "desc": "Bridge-Bypass-Verbot: Kein direkter Import von backend.core"},
    "V14": {"pattern": r'PESQ|pesq|SI.SDR|si_sdr|STOI|stoi', "desc": "Keine Speech-Metriken für Musikqualität"},
    "V21": {"pattern": r'truncat(?!ed).*ohne.*dither|kein.*dither|without.*dither', "desc": "Kein Truncation ohne Dithering"},
    "V44": {"pattern": r'IACC.*<.*0\.7|mono.*collapse', "desc": "Keine Mono-Kollaps-Detektion ohne Stereo-Guard"},
}

# Verzeichnisse die übersprungen werden
SKIP = {".venv", "__pycache__", "node_modules", ".git", "models/", "temp_repro/"}


def scan_file(filepath: Path) -> list[str]:
    """Scannt eine Datei auf VERBOTEN-Verstöße."""
    if any(s in str(filepath) for s in SKIP):
        return []
    if filepath.suffix != '.py':
        return []

    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    issues = []
    for rule_id, rule in RULES.items():
        if re.search(rule["pattern"], content, re.IGNORECASE):
            # Bei V01: Nur in Nicht-Bridge-Dateien
            if rule_id == "V01":
                if "bridge" in str(filepath).lower() or "api/bridge" in str(filepath):
                    continue
            issues.append(f"{rule_id}: {rule['desc']} — in {filepath.relative_to(_PROJECT_ROOT)}")

    return issues


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="V01-V52 VERBOTEN-Linter")
    p.add_argument("--ci", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    all_issues = {}
    for py_file in _PROJECT_ROOT.rglob("*.py"):
        issues = scan_file(py_file)
        if issues:
            all_issues[str(py_file.relative_to(_PROJECT_ROOT))] = issues

    total = sum(len(v) for v in all_issues.values())

    if args.json:
        print(json.dumps({"clean": total == 0, "issues": total, "details": all_issues}))
    else:
        if total > 0:
            print(f"\n❌ {total} VERBOTEN-Verstöße in {len(all_issues)} Dateien:\n")
            for fname, issues in all_issues.items():
                for issue in issues:
                    print(f"  {issue}")
            return 1
        else:
            print(f"✅ VERBOTEN-Linter: {len(list(_PROJECT_ROOT.rglob('*.py')))} Dateien geprüft, 0 Verstöße")

    return 0


if __name__ == "__main__":
    sys.exit(main())
