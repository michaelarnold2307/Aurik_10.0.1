from __future__ import annotations

import json
from pathlib import Path

from audit.runtime_spec_check import run_check


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _get_mode_check(report: dict) -> dict:
    for chk in report["checks"]:
        if chk["id"] == "mode_contract":
            return chk
    raise AssertionError("mode_contract-Check nicht gefunden")


def test_mode_contract_accepts_internal_studio2026(tmp_path: Path) -> None:
    backend = tmp_path / "backend.log"
    frontend = tmp_path / "frontend.log"
    output = tmp_path / "report.json"

    _write(
        backend,
        "\n".join(
            [
                "AurikDenker.denke() gestartet",
                "run context mode=studio2026",
                "AurikDenker.denke() abgeschlossen",
            ]
        ),
    )
    _write(frontend, "")

    report = run_check(backend, frontend, output)
    mode_chk = _get_mode_check(report)

    assert mode_chk["passed"] is True
    assert "vorhanden" in mode_chk["evidence"]

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["checks"]


def test_mode_contract_accepts_ui_studio_2026(tmp_path: Path) -> None:
    backend = tmp_path / "backend.log"
    frontend = tmp_path / "frontend.log"
    output = tmp_path / "report.json"

    _write(
        backend,
        "\n".join(
            [
                "AurikDenker.denke() gestartet",
                'payload: {"mode":"STUDIO_2026"}',
                "AurikDenker.denke() abgeschlossen",
            ]
        ),
    )
    _write(frontend, "")

    report = run_check(backend, frontend, output)
    mode_chk = _get_mode_check(report)

    assert mode_chk["passed"] is True
