---
applyTo: "backend/core/musical_goals/*.py"
---

# Musical Goals — Regeln (normativ, Aurik 9.12.x)

## 14 Goals — Prioritäten und kanonische Böden

| Prio | Goal | Restoration-Boden | Studio-2026-Boden |
|---|---|---|---|
| **P1** | natuerlichkeit | ≥ 0.90 | ≥ 0.92 |
| **P1** | authentizitaet | ≥ 0.88 | ≥ 0.90 |
| **P2** | tonal_center | ≥ 0.95 | ≥ 0.96 |
| **P2** | timbre | ≥ 0.87 | ≥ 0.89 |
| **P2** | artikulation | ≥ 0.85 | ≥ 0.87 |
| **P3** | emotionalitaet | ≥ 0.82 | ≥ 0.84 |
| **P3** | mikrodynamik | ≥ 0.88 | ≥ 0.90 |
| **P3** | groove | ≥ 0.83 | ≥ 0.85 |
| **P4** | transparenz | ≥ 0.82 | ≥ 0.85 |
| **P4** | waerme | ≥ 0.75 | ≥ 0.78 |
| **P4** | bass_kraft | ≥ 0.78 | ≥ 0.80 |
| **P4** | sep_fidelity | ≥ 0.78 | ≥ 0.80 |
| **P5** | brillanz | ≥ 0.78 | ≥ 0.82 |
| **P5** | raumtiefe | ≥ 0.70 | ≥ 0.74 |

> **VERBOTEN**: Böden hardcoden. **RICHTIG**: `calibration_matrix.get_material_floor(material_type, goal)` — material-adaptive Böden: Shellac ~0.72, Vinyl ~0.82, CD ~0.90.

## `measure_all()` — Rückgabe-Invariante

```python
def measure_all(audio: np.ndarray, sr: int, **kwargs) -> dict[str, float]:
    """Misst alle 14 Goals. MUSS immer dict[str, float] zurückgeben — niemals None."""
    results: dict[str, float] = {}
    for goal_name, metric in self._metrics.items():
        try:
            results[goal_name] = float(metric.measure(audio, sr, **kwargs))
        except Exception:
            results[goal_name] = 0.0  # nie None, nie KeyError
    return results
```

## Material-adaptive Böden — Warum korrekt

```
Shellac (1920-1950): SNR ~15 dB, BW ~7 kHz, Mono
→ natuerlichkeit 0.90 ist physikalisch UNMÖGLICH auf diesem Medium
→ material_floor: natuerlichkeit ≈ 0.72

Vinyl (1950-1990): SNR ~60 dB, BW ~16 kHz
→ material_floor: natuerlichkeit ≈ 0.82

CD (1980+): SNR ~96 dB, BW ~22 kHz
→ material_floor: natuerlichkeit ≈ 0.90

VERBOTEN: Alle Böden auf CD-Wert anheben
→ Shellac-Restaurierungen wären permanenter Fail → Recovery-Kaskade sinnlos aktiv
```

## Per-Song Studio-Day-Target (§0k)

```python
# KANONISCH — VOR Pipeline:
from backend.core.studio_goal_targets import estimate_song_goal_targets

studio_targets = estimate_song_goal_targets(
    era_decade=era_decade,          # z.B. 1970
    genre_label=genre_label,        # z.B. "schlager"
    material_chain=material_chain,  # z.B. ["vinyl", "mp3_low"]
    restorability=restorability,    # 0-100
)
# Beispiele:
# 1920er Shellac: brillanz≈0.52, raumtiefe≈0.30 (Mono)
# 1970er Schlager: brillanz≈0.80, waerme≈0.85
# 1990er CD-Pop:   brillanz≈0.88, transparenz≈0.90

# PhaseConductor stoppt Enhancement sobald goal ≈ studio_targets[goal]
# VERBOTEN: Phasen über studio_targets[goal] optimieren ohne neue Signal-Evidenz
```

## §2.56 Per-Song-Gewichtung

```python
from backend.core.song_goal_importance import estimate_goal_importance

goal_weights = estimate_goal_importance(audio, sr, metadata)
# 5-stufige Kaskade: Label/Audio/Psychoakustik/Vokal-Harmonik/Interactions
# Bereich: [0.30, 2.00]
# P1/P2-Floor ≥ 0.70 (darf nie auf 0 gesetzt werden)
```

## §2.56a Harmonik-Adaptation (advisory)

```python
# _compute_harmonic_adaptation_scalar() → Bereich [0.72, 1.18]
# advisory-only in UV3 _profiled_phase_call
# Explizite PMGG-Strength hat VORRANG vor diesem Scalar
```

## §C10 Bayesian-EMA-Blend

```python
# 15% Nudge aus SongGoalFeedbackStore nach Stufe 7:
nudges = SongGoalFeedbackStore.get_nudges(song_fingerprint)
for goal, nudge in nudges.items():
    effective_target[goal] = 0.85 * studio_targets[goal] + 0.15 * nudge
```

## Regressions-Regime (P1/P2 vs. P3-P5)

```python
# P1/P2 (natuerlichkeit, authentizitaet, tonal_center, timbre, artikulation):
# → Pipeline-Ende-PFLICHT: am Ende MÜSSEN alle ≥ Schwellwert sein
# → Einzelphasen dürfen vorübergehend senken wenn Carrier-Repair Grund ist
# → §2.29c Baseline-Capping gilt für restorative Phasen

# P3-P5 (emotionalitaet, mikrodynamik, groove, transparenz, ...):
# → Pipeline-Netto-Budget: Einzelphasen dürfen vorübergehend sinken
# → PMGG loggt Zwischenregressionen, blockiert aber nicht
```

## Vocal Quality Index (VQI, §2.35c)

```python
from backend.core.musical_goals.vocal_quality_index import compute_vqi

# PFLICHT bei panns_singing_confidence >= 0.35:
vqi = compute_vqi(audio, sr)
metadata["vqi"] = vqi
# Schwellwert: 0.72 → darunter Recovery-Kaskade (kein Veto)

# singer_identity_cosine (Resemblyzer):
# Vor Pipeline + nach Pipeline → cos_sim < 0.92 → Rollback letzter Vokal-Phase
# DSP-Fallback bei Resemblyzer-Ausfall Pflicht
```

## Frisson-Schutz in Goal-Messungen

```python
from backend.core.frisson_candidate_detector import get_frisson_detector

# VOR MDEM-Aufruf:
try:
    frisson_zones = get_frisson_detector().detect(original_audio, sr)
except Exception:
    frisson_zones = []  # Non-blocking — Exception darf Goal-Messung nicht stoppen

# Zwei-Stufen-Invariante:
# Pre-SG + Post-SG: Frisson-Floor -1.0 LU
# SG verteilt sonst Dämpfung in Klimax-Passagen bis -8 LU zurück
```
