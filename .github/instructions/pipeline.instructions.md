---
applyTo: "backend/core/unified_restorer_v3.py"
---

# UV3 — Pipeline-Regeln (normativ, Aurik 9.12.x)

## §2.44 Holistic Perceptual Index (HPI) — letztes Export-Gate

```python
# KANONISCH — Restoration:
HPI = MERT_similarity * timbral_fidelity * artifact_freedom * emotional_arc_preservation
# KANONISCH — Studio 2026:
HPI = studio_quality_gain * PQS_improvement * artifact_freedom * emotional_arc_preservation

# MERT-Floor PFLICHT (Bug-Fix v9.12.0):
MERT_similarity = max(raw_mert, 0.5)  # verhindert Gesamt-Kollaps auf 0

# artifact_freedom ist EINZIGER Veto-Faktor:
if artifact_freedom < 0.95:
    return _recovery_cascade()  # KEIN Export, immer Rollback
# HPI > 0 → Export | HPI ≤ 0 → Rollback (§2.60)
```

**VERSA-Primärpflicht**: `use_versa_in_loop=True`. MERT nur Fallback → `metadata["mert_proxy_used"] = True`.
**Referenz-Paradoxon (§0d)**: Bei `carrier_chain_recovery_ratio > 0.15` → Referenz auf `best_carrier_checkpoint` verschieben, NICHT auf degradierten Input.

## §2.45b Hochrestorabilität-Gate

```python
if restorability_score > 80 and snr_db > 40:
    metadata["high_restorability_gate"] = True
    # Phasen mit defect_severity < 0.05 → überspringen
    # Strength für _NEVER_SKIP-Phasen auf Restorability-adaptiven Minimalwert senken
```

## §2.47a PreAnalysis-Handover

- `run_pre_analysis()` **genau 1×** nach Import
- `MediumDetector.detect()` **genau 1×** — nie nochmals auf restauriertem Audio
- Neuer File-Import → Cache **HARD** löschen

## §2.48 Kumulative-Phasen-Interaktions-Guard

```python
# VERBOTEN: feste Konstante
tolerance = 0.15  # FALSCH

# RICHTIG:
tolerance = compute_adaptive_drift_tolerance(
    restorability, material, severity, n_phases
)
# Carrier-Repair-Phasen (_CARRIER_REPAIR_PHASE_PREFIXES) inkrementieren
# consecutive_rollbacks NICHT
```

## §2.49 Artefakt-Freiheits-Gate

```python
# artifact_freedom = min(per-phase-scores) — KEIN Pipeline-Delta
# Musical-Noise: nur Bins wo restored > orig * 1.05
# Phase-Cancellation: Delta-basiert + original_stereo als Referenz
# Frames die im Input bereits anti-phasig waren → NICHT flaggen
```

## §2.51 Stereo — Hard-Fail-Invariante

```python
# §2.51a — drei Hard-Fails, alle sofort Rollback:
assert interchannel_delay_ms <= 1.0    # >1 ms → Hard-Fail
assert lr_imbalance_db <= 6.0          # >6 dB → Hard-Fail
assert true_peak_dbtp <= -1.0          # >-1 dBTP → Hard-Fail

# VERBOTEN: unabhängiges L/R-Processing
# RICHTIG: M/S-Domain oder Linked-Stereo überall
```

## §2.52 PhaseConductor — _NEVER_SKIP

```python
_NEVER_SKIP = {"phase_01", "phase_09", "phase_12", "phase_14", "phase_15"}
# Diese Phasen laufen immer — auch bei hoher Restorability
```

## §2.53b Determinismus

```python
# precomputed_phase_plan ist Source of Truth
# UV3 überspringt _select_phases() + _optimize_phase_plan_intelligence()
# wenn precomputed_phase_plan vorhanden
```

## §2.60 Rollback-Hierarchie (vollständig, KEINE Stufe überspringen)

```
1. Phase-Rollback → Phase-Score negativ markieren
2. Strength-Reduktion 50 % → erneutes PMGG-Check
3. best_carrier_checkpoint (nach Stufe 1–4, vor Enhancement)
4. Pre-Pipeline-Checkpoint (nach TDP, vor allen Phasen)
5. Input-Export mit status="degraded"  ← IMMER besser als Artefakt
VERBOTEN: leerer Export / Abbruch ohne Ausgabe / Export mit bekanntem Artefakt
```

## §2.61 Output-Length-Guard

```python
# KANONISCH — nach JEDER Phase in UV3:
if abs(len(output) - len(input_audio)) > 64:
    logger.error("length_mismatch phase=%s delta=%d", phase_id, len(output) - len(input_audio))
    output = output[:len(input_audio)]  # harter Crop
    metadata["length_corrections"].append(phase_id)
# VERBOTEN: Zero-Padding als primäre Längenkorrektur
```

## §2.64 Per-Phase-Score-Delta (MAS-Konvergenz)

```python
# KANONISCH — jede Phase MUSS diesen Rahmen nutzen:
pre = _fast_goal_snapshot(audio, sr, material)
audio = phase.process(audio, sr)
post = _fast_goal_snapshot(audio, sr, material)
metadata["phase_deltas"][phase_id] = {g: post[g] - pre[g] for g in pre}

# Rollback wenn:
if any(post[g] - pre[g] < -0.03 for g in P1P2_GOALS):
    audio = pre_audio  # Rollback

# MAS-Erreicht-Stop:
if all(mas_gap[g] <= 0.02 for g in P1P2_GOALS):
    metadata["mas_achieved_at_phase"] = phase_id
    break  # Pipeline stoppen — §2.65
```

### §2.64 `_fast_goal_snapshot` — Multi-Segment-Pflicht

```python
# VERBOTEN: Single-Segment-Bias auf Audio-Mitte
spec = fft(mono[N//2: N//2 + frame_size])  # FALSCH

# RICHTIG: 3 Segmente mitteln (25%/50%/75%)
specs = [fft(seg25), fft(seg50), fft(seg75)]
spec = np.mean(specs, axis=0)

# authentizitaet-Proxy: Zentral-Drittel statt Intro
acf_segment = mono[N//3: N//3 + 8192]

# transparenz-Proxy: Vollsignal + SFM-Blend
val = 0.70 * np.log10(p95_full / p05_full + 1e-9) / 4.0 + 0.30 * (1.0 - sfm_avg)
```

## §2.65 MAS-Early-Stop

```python
# VERBOTEN: Pipeline läuft nach _mas_fully_achieved=True weiter
# RICHTIG: UV3-Loop prüft _check_mas_convergence() nach jeder Phase
if self._mas_fully_achieved:
    logger.info("MAS erreicht bei Phase %s — Pipeline-Stop", phase_id)
    break
```

## §6.2a Material-Pflicht-Phasen

| Material | Pflicht-Phasen (unabhängig von DefectScanner-Score) |
|---|---|
| vinyl | phase_09, phase_12, phase_05 |
| tape / cassette | phase_29, phase_24, phase_06, phase_03 |
| reel_tape | phase_29, phase_24, phase_03, phase_55 |
| shellac | phase_03, phase_06, phase_01 |
| mp3_low | phase_23, phase_03, phase_50 |

> `cassette` → intern immer als `tape` in `_MATERIAL_PRIORITY_PHASES`

## §2.29c Restorative-Baseline-Capping

```python
_RESTORATIVE_PHASES = {
    "phase_02", "phase_03", "phase_09", "phase_18",
    "phase_20", "phase_23", "phase_24", "phase_29", "phase_49"
}
# Für diese Phasen:
effective_before[g] = min(measured_before[g], canonical_threshold[g] + 0.05)
# Enhancement-Phasen: echte scores_before (kein Capping)
```

## §2.29e PMGG Team-Koordination

```python
# UV3 schreibt prior_phase_context nach jeder Phase fort
# Phase_50 nach HF-Restauration (phase_06/07/23):
#   Goal-Exclusion: brillanz, transparenz, timbre
#   Emergency-Retries unterdrückt
# CONFLICT_REGISTRY: get_conflict_phases() aus phase_ontology.py
```

## §2.55 PMGG-CIG-Sync-Invariante

```python
# Bei neuer Phase: BEIDE Tabellen synchron aktualisieren
# CIG._PHASE_SPECIFIC_DRIFT_EXCLUSIONS[p] ∩ P1P2
# == PMGG.PHASE_GOAL_EXCLUSIONS[p] ∩ P1P2
# CI-Test: test_pmgg_cig_sync.py
```

## VQI-Gate (Gesangsmaterial)

```python
# PFLICHT wenn panns_singing_confidence >= 0.35:
if panns_singing_confidence >= 0.35:
    vqi = compute_vqi(restored_audio, sr)
    metadata["vqi"] = vqi
    if vqi < 0.72:
        _recovery_cascade()  # kein Veto, aber Recovery
```
