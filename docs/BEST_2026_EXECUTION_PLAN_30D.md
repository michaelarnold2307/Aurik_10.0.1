# Aurik 2026 Best-in-Class Ausfuehrungsplan (30 Tage)

Stand: 14.04.2026

Geltungsbereich: Umsetzungsprogramm fuer die priorisierten Punkte 2 bis 10.
Ausgenommen: Punkt 1 (externe Hoer-Validierung) auf Wunsch deaktiviert.

## Fortschritt (Live)

- Paket 2: abgeschlossen
- Paket 3: abgeschlossen
- Paket 4: abgeschlossen
- Paket 5: abgeschlossen
- Paket 6: abgeschlossen
- Paket 7: abgeschlossen
- Paket 8: abgeschlossen
- Paket 9: abgeschlossen
- Paket 10: abgeschlossen

## Zielbild

- Ein konsistentes, auditierbares, reproduzierbares Qualitäts- und Runtime-System ohne widerspruechliche Normwerte.
- Ein eindeutiger Release-Status, bei dem Runtime-Compliance und Release-Report synchron sind.
- Fruehere Fehlererkennung bei Stereo-Drift, Mode-Vertragsbruch und Gate-Regressionen.

## Priorisierte Pakete (2 bis 10)

2. Normkonflikte zwischen Haupt-Instructions und Specs eliminieren.
3. Audit-Pipeline konsolidieren (einheitlicher, aktueller Wahrheitsstand).
4. Mode-Contract-Verletzung im Runtime-Spec-Check beheben.
5. Quality-Gate-Kalibrierung auf reale Transferketten haerten.
6. End-to-end Carrier-Recovery-Beweisfuehrung automatisieren.
7. Stereo-Integritaet als globales No-Regress-Gate ausbauen.
8. Determinismus in Heavy-Pfaden verschaerfen.
9. UX-Propagation der Gate-Entscheide standardisieren.
10. Continuous Real-Audio-Gate taeglich betreiben.

## 30-Tage-Plan

### Woche 1 (Tag 1-7) - Governance und Messbarkeit

- Paket 2: Norm-Sync-Check als Test einbauen.
- Paket 3: Audit-Report-Schema vereinheitlichen (Timestamp, Run-ID, Mode, Pflichtchecks, Fail-Reasons).
- Paket 4: Mode-Contract im Runtime-Check deterministisch machen.

Deliverables:

- Neuer Konsistenztest fuer widerspruechliche Grenzwerte.
- Audit-Parser mit klarer Prioritaet auf neuesten Lauf.
- Runtime-Spec-Report ohne false negative auf mode_contract.

Akzeptanzkriterien:

- Kein Konflikt zwischen [/.github/copilot-instructions.md](../.github/copilot-instructions.md) und [/.github/specs/07_quality_and_tests.md](../.github/specs/07_quality_and_tests.md) fuer gemeinsame Budget-Felder.
- [audit/runtime_spec_report.json](../audit/runtime_spec_report.json) zeigt required_passed = required_total in Referenzlauf.
- Ein eindeutiger Release-Finalstatus ohne gegenlaeufige Gesamtampel zwischen [audit/release_report.json](../audit/release_report.json) und [audit/runtime_spec_report.json](../audit/runtime_spec_report.json).

### Woche 2 (Tag 8-14) - Klanggates haerten

- Paket 5: Material- und transferkettenbezogene Regressionstests fuer Gates erweitern.
- Paket 6: Carrier-Recovery 3-Ebenen-Invariante in einem integrierten E2E-Test absichern.
- Paket 7: Stereo-Health-Checkpoints vor und nach kritischen Additive- und Dynamics-Bloecken einziehen.

Deliverables:

- Neue Tests fuer Gate-Robustheit bei Multi-Generation-Material.
- Neuer E2E-Test fuer PMGG + End-Goals + HPI Referenzmodell.
- Stereo-No-Regress-Test fuer kumulative Drift.

Akzeptanzkriterien:

- Keine Pipeline-Blockade bei legitimer Carrier-Inversion.
- Kein kumulativer Stereo-Kollaps ohne fruehes Gate-Ereignis.
- Alle neuen Tests laufen gruen in Unit und Integration.

### Woche 3 (Tag 15-21) - Reproduzierbarkeit und Runtime-Stabilitaet

- Paket 8: Determinismus fuer Heavy-Pfade und Fallback-Wege absichern.
- Paket 3 Vertiefung: Audit-Delta-Reporter mit Change-Erklaerung pro Lauf.

Deliverables:

- Determinismus-Testset fuer Heavy-ML/Fallback-Szenarien.
- Audit-Diff-Report zwischen zwei aufeinanderfolgenden Runs.

Akzeptanzkriterien:

- Wiederholte Runs mit gleichen Inputs liefern stabile Gate-Entscheide.
- Unterschiede zwischen Runs sind begruendet und im Report sichtbar.

### Woche 4 (Tag 22-30) - UX-Transparenz und taegliches Real-Audio-Gate

- Paket 9: Einheitliche UI-Ausgabe fuer recovered/degraded plus Top-Tradeoffs/Fails.
- Paket 10: Taeglicher Real-Audio-Gate-Lauf mit Trend-Tracking.

Deliverables:

- Standardisierter Insight-Block fuer UI ueber Bridge-Daten.
- Automatisierter Daily-Gate-Lauf mit Verlaufsgrafik und Schwellenwarnungen.

Akzeptanzkriterien:

- Nutzer sieht immer klare Begruendung fuer Endstatus.
- Daily-Run produziert reproduzierbaren Verlauf fuer HPI, Artifact-Freedom, OQS, Fail-Reasons.

## Reihenfolge fuer direkte Umsetzung

1. Paket 2
2. Paket 4
3. Paket 3
4. Paket 6
5. Paket 7
6. Paket 5
7. Paket 8
8. Paket 9
9. Paket 10

Begruendung: Erst Normklarheit und mode_contract, dann Gate-Kernlogik, danach UX und Betrieb.

## Risiko-Register

- Risiko: Alte Audit-Artefakte verfaelschen Gesamtstatus.
Gegenmassnahme: Neuester-Run-Policy + harte Run-ID-Korrelation.

- Risiko: Zusatztets erhoehen Laufzeit deutlich.
Gegenmassnahme: Trennung in fast gates und nightly heavy gates.

- Risiko: Striktere Stereo-Gates erzeugen kurzfristig mehr Rollbacks.
Gegenmassnahme: Adaptive Toleranz und transparente Fail-Reason-Telemetrie.

## Abschlusskriterien nach 30 Tagen

- Normkonflikte im Kernbereich auf 0.
- Runtime-Compliance stabil 100 Prozent bei Pflichtchecks.
- Kein ungeklaerter Widerspruch zwischen Release- und Runtime-Report.
- Sichtbar stabilerer Real-Audio-Qualitaetsverlauf im Daily-Gate.

## Erkenntnisdokumentation

- Konsolidierte Erkenntnisse und Rest-Risiken: [BEST_2026_ERKENNTNISSE.md](BEST_2026_ERKENNTNISSE.md)
