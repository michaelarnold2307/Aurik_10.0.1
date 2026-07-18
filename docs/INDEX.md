# Aurik 10.0.8 - Dokumentationsindex

Offizielle Dokumentation fuer Aurik 10.0.0.

## Normativer Vorrang

Bei Abweichungen zwischen Einzel-Dokumenten und Spezifikation gilt immer:

1. `.github/specs/01-14`
2. `docs/CHANGELOG_HISTORY.md`

## Kernfakten

- Phasen: 68 (Phase 01–66 + Vocal Repair + Glue Stage)
- Musical Goals: 14
- DetectionTypes: 62
- Kausal-Ursachen: 62
- Tests: ~18.400

## Release-Must-Leitplanken

- Desktop-only (Linux AppImage, Windows 10/11)
- 100 % offline nach Installation
- Endnutzer-Workflow: One-Button mit Moduswahl `Restoration` oder `Studio 2026`
- Kanonischer Vertrag: Bridge -> AurikDenker.denke -> export_guard

## Startpunkte

### Fuer Anwender

- [Installations-Guide](guides/INSTALLATION.md)
- [Benutzerhandbuch](guides/USER_GUIDE.md)
- [Konfigurations-Guide](guides/CONFIGURATION.md)
- [Troubleshooting](guides/TROUBLESHOOTING.md)

### Fuer Entwicklung und Audit

- [KI-Agent Integration Guide](KI-AGENT-INTEGRATION-GUIDE.md)
- [Python API](api/PYTHON_API.md)
- [Architektur-Ueberblick](architecture/ARCHITECTURE.md)
- [Phasen-Ueberblick](architecture/PHASES_OVERVIEW.md)
- [Pipeline-Analyse](architecture/PIPELINE_FLOW_ANALYSIS.md)
- [CI/CD](CI_CD.md)
- [Testing Guide (inkl. Gate-Runbook)](development/TESTING.md)
- [Worldclass Change Acceptance Protocol](WORLDCLASS_CHANGE_ACCEPTANCE_PROTOCOL.md)
- [Spec-Evidenzberichte](reports/spec_evidence/README.md)

## Kanonischer Vertragsfluss (Kurz)

```text
Import (Bridge) -> Voranalyse -> AurikDenker.denke -> Holistic Gates -> export_guard
```

## Legacy-Regel

Historische Dokumente mit v2-/Server-/Docker-Produktpfaden sind nur als
`LEGACY_NON_RELEASE` zu betrachten, sofern sie nicht auf den kanonischen Vertrag
aktualisiert wurden.
