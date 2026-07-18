# Aurik 10.0.0 — Bauplan (1:1 reproduzierbar)

## Stand: 2026-07-18 | Aurik 10.0.0 — Weltspitze | 12 Innovationen integriert

---

## 1. Architektur-Übersicht

```
IMPORT → PRE-ANALYSE → SONG-CONTEXT → OPTIMIERER → KOHÄRENZ-GUARD → EXPORT
───────   ──────────   ────────────   ─────────   ──────────────   ──────
          9 Analyse-    Look-Ahead/    5 Strategien  SongCoherence   WAV/FLAC
          Module        Behind pro     parallel      + Coherence-    /AIFF
                        Segment         Per-Segment-  Fixes (4 St.)  Atomic
                                       Auswahl                      Write
```

---

## 2. Modul-Verzeichnis (Weltspitze-Niveau)

### 2.1 Import (`backend/file_import.py`)

- **Formate**: `.wav .mp3 .flac .ogg .aac .aiff .wma .opus .m4a .alac .caf`
- **Decoder**: soundfile → pedalboard/FFmpeg → pydub-Subprocess (3-stufig)
- **STCG-Import-Guard**: GCC-PHAT + Multi-Point-Verifikation, korrigiert L/R-Drift >64 samples
- **Downmix**: Energie-gewichteter Downmix für >2 Kanäle
- **NaN/Inf**: `nan_to_num` + `clip(-1,1)` nach jedem Decode-Pfad
- **Carrier-Detection**: Heuristik + Forensik + ML-Klassifikation
- **Spezifikationen**: §G-SF-READ, §G13, §2.47

### 2.2 Pre-Analyse (`backend/core/pre_analysis.py`)

- **Medium-Detector**: Forensische Tonträgererkennung (Vinyl, Shellac, Tape, CD, etc.)
- **Era-Classification**: via Bridge API, Jahrzehnt-Schätzung
- **Genre-Classification**: Schlager-aware, via Bridge API
- **Defect-Scanner**: Material-abhängige Thresholds (§2.47a)
- **Restorability**: Bewertet Wiederherstellbarkeit (0–100)
- **Cross-Validation**: Genre-Chain-Konsistenzprüfung (geloggt)
- **Alle silent-except Blöcke**: → `logger.debug()` mit Kontext

### 2.3 Phase-Pipeline (66 Phasen) + Closed-Loop Optimizer

Jede Phase folgt dem Interface:

```python
def process(self, audio: np.ndarray, sample_rate: int,
            material_type: MaterialType, **kwargs) -> PhaseResult
```

| # | Phase | Spezifikation |
|---|---|---|
| 01 | Click Removal | Adaptive Threshold, Material-Profile |
| 02 | Hum Removal | Notch-Filter, Harmonische-Erkennung |
| 03 | Denoise | Spektrale Subtraktion, OMLSA-Fallback |
| 04 | EQ Correction | Material-adaptiv, §ISO-226 |
| 05 | Rumble Filter | Hochpass, Subsonic |
| 06 | Frequency Restoration | Bandwidth-Extension |
| 07 | Harmonic Restoration | Exciter (nur Studio-2026) |
| 08 | Transient Preservation | Attack-Erkennung |
| 09 | Crackle Removal | Impuls-Detektion |
| 12 | Wow & Flutter | pYIN, STCG-geführt, §G-PYIN-CACHE |
| 16 | Final EQ | Material-Profile |
| 17 | Mastering Polish | Finaler Schliff |
| 18 | Noise Gate | Adaptiv, §2.45a |
| 19 | De-Esser | Gender-aware, Aurik-8.0-Stack |
| 23 | Spectral Repair | FFT-basiert |
| 28 | Surface Noise | PMGG-verifiziert |
| 35 | Multiband Compression | Material-adaptiv, §5/5 Peak-Messung |
| 36 | Transient Shaper | Fragile-Guard, §5/5 Peak-Messung |
| 37 | Bass Enhancement | Warmth-adaptiv, §5/5 Peak-Messung |
| 38 | Presence Boost | Era-adaptiv, §5/5 Peak-Messung |
| 39 | Air Band | §0a-Guard (Restoration → verboten), §5/5 Peak |
| 40 | Loudness Normalization | ITU-R BS.1770-4, §5/5 LUFS-Messung |

### 2.4 Qualitäts-Gates

- **PMGG** (`per_phase_musical_goals_gate.py`): Pro-Phase Rollback bei Goal-Regression, Retry-Loop mit `retry_strengths` [0.75, 0.50, 0.30, 0.15]
- **STCG** (`stereo_temporal_coherence_guard.py`): Pre+Post-Pipeline L/R-Korrektur, Multi-Point-Verifikation, Cumulative-Correction-Limit (5ms)
- **DoNoHarmGuardian** (`do_no_harm_guardian.py`): §G-5/5 Finaler Input-vs-Output-Vergleich

### 2.5 Export (`backend/core/audio_exporter.py`)

\n### 2.6 Closed-Loop Perceptual Optimizer (`backend/core/perceptual_optimizer.py`)

- **PerceptualOptimizer**: Parallele Strategien → Per-Segment-Auswahl → Iteration
- **5 Strategien**: passthrough, light, balanced, deep, full
- **Aktivierung**: `restore(..., optimize=True)`
- **Konvergenz**: Abbruch bei ΔMOS < 0.01, max 3 Iterationen
- **Spezifikation**: §CROWN
- **Formate**: WAV/FLAC/AIFF, Atomic-Write via `.tmp` → `os.replace`
- **Dithering**: POW-r Type 3 (primary), TPDF-Fallback
- **Post-Gate**: PerceptualExportOptimizer, VocalClarityMax
- **Listening-Mode EQ**: Adaptiv (Kopfhörer/Lautsprecher)

---

## 3. Modus-Garantien

### 3.1 Restoration-Modus

- **Ziel**: Defekte entfernen, Charakter 100% bewahren
- **§0a**: Air-Band/Harmonic-Exciter VERBOTEN auf Analogmaterial
- **LUFS**: Material-abhängig (Vinyl: −18, Tape: −16, CD: −14)
- **PMGG**: Rollback bei JEDER Goal-Regression
- **DoNoHarmGuardian**: STRENG — max 8 dB Pegeländerung, 20% Brightness-Drop, 15% Naturalness-Drop

### 3.2 Studio-2026-Modus

- **Ziel**: Stream-tauglich, wettbewerbsfähiger Klang
- **§0a**: Air-Band FREI — bewusste Höhenanhebung erlaubt
- **LUFS**: HART −14 LUFS für alle Materialien (EBU R128)
- **PMGG**: Rollback nur bei KRITISCHER Goal-Regression
- **DoNoHarmGuardian**: LOCKER — 20 dB Pegeländerung ok, 40% Brightness

### 3.3 Gemeinsame Garantien

- Kein Song wird verschlechtert (DoNoHarmGuardian)
- Jeder Phase-Skip wird geloggt
- Echte Metriken (keine Dummy-Werte)
- NaN/Inf-Schutz auf allen Pfaden
- 3-stufige Vocal-Detection (spectral → MFCC → energy)
- PsychoAcousticMetrics ist vollwertiger Calculator

---

## 4. Spezifikationen-Referenz

| Spec | Modul | Inhalt |
|---|---|---|
| §G-5/5 | do_no_harm_guardian.py, phase_40 | Weltspitze-Qualitätsgarantie |
| §0a | phase_39 | Air-Band-Verbot im Restoration-Mode |
| §2.46e | phase_39, unified_restorer_v3 | Novelty-Rollback, Harmonic-Exciter-Verbot |
| §G-STEREO-GUARD | unified_restorer_v3 | Mono→Stereo-Notfall-Rekonstruktion |
| §G-PYIN-CACHE | phase_12 | pYIN-Cache (8 Einträge LRU) |
| §G-SF-READ | file_import.py | Soundfile-Wrapper |
| §G13 | file_import.py, STCG | Dual-Confirmation GCC-PHAT |
| §2.47a | pre_analysis.py | Material-Defect-Consistency |
| §ISO-226 | phase_40 | Fletcher-Munson-Lautstärke-Kompensation |

---

## 5. Abweichungsprotokoll (bidirektional behoben)

| Datum | Abweichung | Richtung | Fix |
|---|---|---|---|
| 2026-07-18 | §G-5/5 fehlte in Phase 40 | Spec→Code | Tag ergänzt |
| 2026-07-18 | `.scores` vs `.degraded_metrics` | Code→Spec | Referenz korrigiert (GuardianVerdict ist höherwertig) |
| 2026-07-18 | `retry_strengths` nie definiert | Code→Spec | Definition ergänzt [0.75,0.50,0.30,0.15] |
| 2026-07-18 | `seed`→`dither_seed` in exporter | Code→Spec | Falscher Variablenname korrigiert |
| 2026-07-18 | `_HEAVY_MODEL_*` ohne Defaults | Code→Spec | Default-Werte ergänzt (1.0, 70.0, etc.) |
| 2026-07-18 | `_PYIN_CACHE` undefiniert | Code→Spec | Modul-Level-Deklaration ergänzt |
| 2026-07-18 | `PsychoAcousticMetrics` @dataclass | Spec→Code | @dataclass entfernt, **init**+Calculator-Methoden |
| 2026-07-18 | Phase 35-40 Dummy-Metriken | Spec→Code | Echte Messungen (LUFS, Peak) |

---

## 6. Qualitäts-Metriken (Mai-30-Referenzlauf)

```
Input Quality:   41.2/100 (fair)
Output Quality:  52.4/100 (fair)  Δ +11.2
Restoration:     74.1%
MUSHRA:          91.7 (Excellent)
VQI:             0.802 (acceptable)
RT-Faktor:       22.53×
```

---

## 7. Reproduzierbarkeit

### 7.1 Environment

- Python 3.10+
- `requirements.txt` im Projekt-Root
- `.venv_aurik` Virtual Environment

### 7.2 Pre-Commit-Hooks (alle grün)

```
ruff ✅ | ruff format ✅ | flake8 ✅ | mypy (core) ✅
Anti-Regression-Gate ✅ (9/9 Muster)
check python ast ✅ | debug statements ✅
```

### 7.3 Build

```bash
source .venv_aurik/bin/activate
pip install -r requirements.txt
pre-commit install
python -m pytest tests/ -x -q
```

### 2.11 Spenden-Erinnerung
