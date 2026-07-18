# Aurik 10 — Spec 11: Entscheidungsintelligenz | §v10 Pleasantness-First

> **Normative Quelle** für alle Entscheidungsmodule.
> **Invarianten** sind mit `§INV` markiert und MÜSSEN bei allen Änderungen erhalten bleiben.
> **Roadmap**-Einträge sind mit `§ROADMAP` markiert — spezifiziert, noch nicht implementiert.

---

## IMPLEMENTIERT (v10.0.0) [RELEASE_MUST]

## §INV-1 [RELEASE_MUST]: Zentrale Entscheidungsintelligenz im Denker

Die **Entscheidungsintelligenz liegt zentral bei den Denkern** (`denker/`).
Der Denker berechnet den `global_scalar` aus allen verfügbaren Informationen.
Einzelne Phasen treffen **keine** eigenständigen „sei konservativer"-Entscheidungen.

**Phasen dürfen nur binäre Capability-Checks durchführen:**

- „Kann ich auf diesem Material überhaupt etwas Sinnvolles tun?" → Ja/Nein
- **Nicht erlaubt:** „Ich reduziere meine Stärke um 30% weil bw_loss hoch ist."

## §INV-2 [RELEASE_MUST]: SongCalibration — Multi-Faktor global_scalar

| Faktor | Formel | Wirkung |
|---|---|---|
| Defekt-Diversität | `0.90 + 0.20 × defect_diversity` | Mehr Defekte → vorsichtiger |
| Restorability | `0.88 + 0.24 × restorability_score` | Schlechter restaurierbar → vorsichtiger |
| SNR | `0.90 + 0.10 × (1.0 − snr_norm)` | Mehr Rauschen → vorsichtiger |
| Confidence | `0.92 + 0.16 × pipeline_confidence` | Unsicher → vorsichtiger |
| **Bandwidth-Loss** | `1.0 − 0.25 × bw_loss` | **§2.12** Kein HF-Inhalt → max. −25% |
| **Detektor-Dissens** | `×0.90` bei >20 Jahre Era-Differenz | **§2.13** Unsicherheit → −10% |
| **Fragile-Material** | Cap bei 0.70 wenn bw_loss≥0.90 & SNR<16dB | **§2.15** Extrem degradiert |
| **Preservation Mode** | Flag wenn bw_loss≥0.90 & SNR<16dB | **§2.16** Transparenz |

Wertebereich: `[0.50, 1.50]`.

## §INV-3 [RELEASE_MUST]: SectionStrengthEnvelope — Kontinuierliche per-Segment-Anpassung

```
SectionGoalAdapter (6 Sektionen)
         │
         ▼
SectionStrengthEnvelope.build()
  • Cosine-Crossfade 200ms, max 1dB/100ms
  • Frisson-Zonen: ≤ 0.30
  • float32[n_samples], Bereich [0.10, 1.50]
         │
         ▼
_profiled_phase_call() → kwargs["strength_envelope"]
         │
         ▼
Phase: strength = base × envelope[frame].mean()
```

**Garantien:** Räumlichkeit/Rauschflor/LUFS bleiben song-global. Keine hörbaren Sprünge.

## §INV-4 [RELEASE_MUST]: Effektive Tonträgerkette

```
reel_tape → vinyl → cassette → mp3_low
    │          │         │          │
    │          │         │          └─ Codec-Guards
    │          │         └─ Transport-Defekte  
    │          └─ Primär-Material, Defekt-Profil
    └─ Bandbreiten-Ziel (18-20 kHz)
```

Physical-Detektion schlägt statistischen Prior. Era-Information bleibt als Precursor erhalten.

## §INV-5 [RELEASE_MUST]: Defekt-Differenzierung pro Tonträger

| Defekt | Cassette | Reel_Tape | Begründung |
|---|---|---|---|
| Transport-Bump | 0.15 | 0.95 | Pinch-Roller nur bei Kassette |
| Print-Through | 0.40 | 0.10 | Spulentonband lagert gewickelt |
| Tape-Head-Level-Dip | 0.15 | 0.65 | Kleine Köpfe → schneller Verschleiß |
| Wow | 0.22 | 0.40 | Billiger Capstan-Motor |
| Flutter | 0.25 | 0.35 | Schmale Bandführung |

## §INV-6 [RELEASE_MUST]: Qualitätsmetriken mit Hörbarkeits-Gate

- **GrooveMetric Onset-Guard:** DTW=0 ∧ onset≥90% → Score≥0.85
- **PQS-MOS < 2.5:** → `quality_gate_rollback` an ExzellenzDenker
- **MUSHRA = Similarity** zum degradierten Original — kein absolutes Qualitätsmaß

## §INV-7–10

- Phase 40: analog+vokal → ±8dB, uniformer Gain, keine Entfernung
- Phase 19: pYIN + Contralto + Spectral Dynamic EQ + Phonem-adaptiv
- Stages 2–6: Breath, Formant, Presence, Inpainting, Dynamics aktiv

---

## ROADMAP (v10.0.0+)

## §ROADMAP-1 [RELEASE_MUST]: Cross-Phase Consensus (§3.0)

**Status:** ✅ **Implementiert.** `denker/cross_phase_coordinator.py` (754 Zeilen).
Aktiv in `_profiled_phase_call` mit Overlap-Matrix, Budget-Verteilung (≤1.0 pro Band),
Material-adaptiven Caps und Naturalness-Guard (Musical Noise, Metallic Ringing, Roughness).

**Problem:** Phase 19 (De-Esser) und Phase 38 (Presence Boost) bearbeiten beide
den Frequenzbereich 2–8 kHz — unabhängig voneinander. Ihre Effekte können sich
addieren und den Präsenzbereich überbetonen.

**Lösung:** Ein `CrossPhaseCoordinator` im Denker, der VOR der Pipeline-Ausführung
alle Phasen-Überlappungen im Frequenzbereich identifiziert und die Stärken so
verteilt, dass die kumulative Wirkung ≤ 100% der gewünschten Bearbeitung bleibt.

**Architektur:**

```
CrossPhaseCoordinator.analyze(phase_plan)
    → Overlap-Matrix [phase_i × phase_j × freq_band]
    → Budget-Verteilung: sum(strength_band) ≤ 1.0 pro Frequenzband
    → Injiziert capped_strength pro Phase in kwargs
```

**Priorität:** Hoch — direkter Einfluss auf Klangqualität.

---

## §ROADMAP-2 [RELEASE_MUST]: SectionStrengthEnvelope aktivieren (§2.17.1)

**Status:** ✅ **Implementiert.** Aktiv in Phase 18 (Noise Gate), 19 (De-Esser) und 38 (Presence Boost).
Alle drei lesen `kwargs.get("strength_envelope")` und modulieren `_effective_strength`.
Cosine-Crossfades (200 ms) verhindern hörbare Übergänge zwischen Sektionen.

**Nächster Schritt:** Exemplarische Integration in Phase 19 (De-Esser) und Phase 38
(Presence Boost). Diese beiden Phasen haben den größten per-Segment-Variationsbedarf.

**Implementierung pro Phase:**

```python
envelope = kwargs.get("strength_envelope")
if envelope is not None:
    seg_strength = get_section_strength_at(envelope, frame_start, frame_end)
    effective_strength = base_strength * seg_strength
```

**Validierung:** A/B-Test Strophe vs. Refrain — hörbare, aber fließende Unterschiede.

**Priorität:** Hoch — Infrastruktur ungenutzt, höchster Impact.

---

## §ROADMAP-3 [RELEASE_MUST]: Artist/Track-Fingerprint-Persistenz (§4.0)

**Status:** ✅ **Implementiert.** `SingerVoiceModel`-Ergebnisse werden via
`_batch_intelligence.store()` im `BatchSessionLearner` persistiert (song_id-basiert).
Folgende Songs derselben Session laden Stimmparameter als Prior.

**Problem:** Aurik analysiert Elke Bests Stimme jedes Mal neu — Vibrato, Formanten,
Register — obwohl der Song-ID `eb49f1d4` bekannt ist. Kein Transfer zwischen
Restaurierungen desselben Künstlers.

**Lösung:** `SingerVoiceModel`-Ergebnisse pro `song_id` in `BatchSessionLearner`
persistieren. Beim nächsten Song derselben Künstlerin die gespeicherten
Stimmparameter als Prior laden, nicht von Null rechnen.

**Architektur:**

```
BatchSessionLearner.store(song_id, "singer_voice_model", svm_result)
    ↓
Nächster Song: BatchSessionLearner.load(song_id) → svm_prior
    ↓
VocalFocusAnalyzer: startet mit Prior statt blank
```

**Priorität:** Mittel — spart Analysezeit, verbessert Konsistenz über Alben.

---

## §ROADMAP-4 [RELEASE_MUST]: Dynamic Phase Ordering (§5.0)

**Status:** ✅ **Implementiert.** `PhaseInteractionDenker._dag_reorder()` sortiert
Phasen material-abhängig: Shellac→additive vor subtractive, Tape→subtractive vor additive.
Nutzt `PHASE_FREQ_PROFILES` aus dem `CrossPhaseCoordinator`.

**Problem:** Die Phasen-Reihenfolge wird vor der Pipeline einmalig festgelegt.
Aber die optimale Reihenfolge hängt vom Material ab: EQ vor Denoise bei
bandbreitenbegrenztem Material, Denoise vor EQ bei rausch-dominiertem Material.

**Lösung:** `PhaseInteractionDenker` um volles DAG (Directed Acyclic Graph)
erweitern. Statt fester Sequenz: Phasen deklarieren ihre Input/Output-Frequenz-
bänder. Der Denker topologisch sortiert für minimale kumulative Artefakte.

**DAG-Knoten:** Jede Phase deklariert `{affects: [freq_bands], requires: [freq_bands], conflicts: [phase_ids]}`

**Priorität:** Mittel — großer Architektur-Umbau, relevanter Qualitätsgewinn.

---

## §ROADMAP-5 [RELEASE_MUST]: Real-Time Preview (§6.0)

**Problem:** Der Nutzer wartet 30 Minuten auf das Ergebnis und kann erst danach
beurteilen, ob die Restaurierung gelungen ist. Zu spät für Korrekturen.

**Lösung:** 10-Sekunden-Preview nach der Pre-Analyse-Phase. Aurik restauriert
die ersten 30 Sekunden des Songs (oder einen repräsentativen 30s-Ausschnitt)
mit voller Qualität, aber zeitlich begrenzt. Der Nutzer hört, validiert,
und startet dann die vollständige Restaurierung.

**Architektur:**

```
restore(audio, mode="preview", preview_duration_s=30)
    → volle Pre-Analyse auf voller Länge
    → Pipeline nur auf ersten 30s
    → Export als 30s-FLAC
    → Nutzer hört → bestätigt oder passt Parameter an
    → restore(audio, mode="restoration") auf voller Länge
```

**Priorität:** Mittel — UX-Verbesserung, kein Qualitätsgewinn.

---

> **Letzte Änderung:** v10.1.0 — ROADMAP 1–4 implementiert, 5–7 spezifiziert
