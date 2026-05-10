---
applyTo: "backend/core/phases/phase_*.py"
---

# Phasen-Regeln (normativ, Aurik 9.12.x)

## Pflicht-Checkliste bei jeder neuen Phase

```
1. process()-Signatur: (audio, sr, material_type, strength, **kwargs) → np.ndarray
2. assert sr == 48000 am Eingang
3. audio = np.clip(audio, -1.0, 1.0) am Ausgang
4. result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0) vor Return
5. logger.info("phase=%s score=%.2f", phase_id, score) — kein print()
6. CAUSE_TO_PHASES + CAUSES bidirektional ergänzen (V12)
7. Neue HPF/Notch-Phase: 4-stufige Checkliste (s. unten)
```

## §2.46 Carrier-Chain-Inversion — Stufenreihenfolge (HARD)

```
Stufe 1: ADC-Artefakte     → phase_30 (DC), phase_31 (Quantisierung)
Stufe 2: Playback          → phase_04/06 (RIAA), phase_25 (Azimuth), phase_12 (Wow/Flutter)
Stufe 3: Alterungsschäden  → phase_09 (Knistern), phase_24 (Dropout)
Stufe 4: Carrier subtraktiv→ phase_29 (Bandrauschen), phase_03 (Surface Noise)
Stufe 5: Carrier additiv   → phase_06/23 (BW-Erweiterung), phase_07 (Harmonik)
                              ↑ IMMER nach Stufe 4 — sonst werden rekonstruierte
                                Obertöne sofort entrauscht
Stufe 6: Mixer/Preamp      → BEWAHREN (Recording-Chain-Signatur = Original)
```

## §2.46e Hallucination-Guard — ADDITIVE Phasen (phase_37/38/48/32)

```python
# PFLICHT nach jeder additiven Operation:
from backend.core.dsp.hallucination_guard import check_hallucination

guard = check_hallucination(pre_audio, post_audio, sr, mode)
if guard.requires_rollback:
    return pre_audio  # spectral_novelty > 0.15 in Restoration
if guard.score_penalty > 0:
    phase_score -= 0.3  # spectral_novelty > 0.08

# Drei verbotene Halluzinations-Kategorien in Restoration:
# 1. Harmonik über BW-Ceiling des Materials
# 2. Raumklang/Reverb der nicht im Signal nachweisbar ist
# 3. ML-generierte Spektral-Texturen ohne physikalisches Gegenstück
```

## §2.46f Natural-Performance-Artifacts-Guard

```python
# Diese drei Kategorien sind KEINE Defekte — niemals entfernen:
# 1. Atemgeräusche: -55 bis -40 dBFS, 50-500ms, spectral_flatness > 0.4
# 2. Vibrato/Portamento: F0-Modulation 4-7 Hz, Amplitude ≤ ±50 Cent
# 3. Early Reflections: 0-50ms nach Onset → Dereverb wet_mix cap = 0.35

from backend.core.dsp.natural_performance_detector import detect_performance_artifacts
protected_segments = detect_performance_artifacts(audio, sr)
# Phasen müssen protected_segments respektieren
```

## §0a — Crossfire-Modus-Invariante (absolut)

```python
# VERBOTEN in Restoration — diese Phasen NIE aktivieren:
_RESTORATION_FORBIDDEN = {
    "phase_21_exciter",           # §0a: kein künstlicher Harmonik-Zusatz
    "phase_35_multiband_compression",  # §0a: nur Studio 2026
    "phase_42_vocal_enhancement", # §0a: nur Studio 2026
}
# Diese Phasen dürfen auch nicht in CAUSE_TO_PHASES für Restoration-Causes stehen
```

## Material-Ceiling-Pflicht bei ADDITIVEN Phasen

```python
from backend.core.dsp.physical_ceiling import _MATERIAL_BW_CEILING_HZ, _MATERIAL_DR_CEILING_DB

# BW-Erweiterung (phase_06/07/23):
max_freq = _MATERIAL_BW_CEILING_HZ[material]  # Shellac ≤ 8kHz, Vinyl ≤ 16kHz
# Keine Harmonik/Energie über max_freq hinzufügen

# DR-Expansion (phase_26):
max_dr = _MATERIAL_DR_CEILING_DB[material]  # Vinyl ≤ 70dB, Shellac ≤ 45dB
# Expansion über Ceiling = Artefakt → sofortiger Rollback
```

## §2.63 Boundary-Mechanismus (STFT/ML-Phasen)

```python
# KANONISCH — Reflect-Padding VOR STFT:
_pad_len = hop_length * 4
audio_padded = np.pad(audio, _pad_len, mode="reflect")
# ... STFT-Verarbeitung ...
audio_out = audio_out[_pad_len: _pad_len + n_original]  # deterministischer Strip

# VERBOTEN: np.pad(..., mode="constant") NACH STFT als primäre Längenkorrektur
# Stereo-Lag-Invariante: L + R MÜSSEN identischen _pad_len und Strip-Offset haben
```

## HPF/Notch-Phase — 4-stufige Checkliste

```
1. KEIN Loudness-Guard in der Phase-Datei selbst
2. enable_loudness=False in _phase_overrides setzen
3. Phase-ID in _HPF_NOTCH_CUM_RESET_PHASES eintragen
4. _update_positive_makeup_authority aufrufen
```

## §2.63 Stereo-Lag-Invariante

```python
# Wenn L/R separat verarbeitet:
# VERBOTEN: Per-Channel-Resampling als Längenkorrektur
# RICHTIG: identische Kontextlänge, identischer Strip-Offset, identische Zielsamplezahl
assert len(audio_L_out) == len(audio_R_out) == n_original
```

## §LyricsGuided — NR auf Vokal-Stem

```python
# VERBOTEN: NR-Algorithmen auf Vokal-Stems ohne phonem-bewusste Maske
from backend.core.lyrics_guided_enhancement import LyricsGuidedEnhancement

lge = LyricsGuidedEnhancement()
phoneme_mask = lge.get_phoneme_mask(audio, sr)
# phoneme_mask[frame] = True → NR-Bypass (Konsonanten-Burst geschützt)
# Konsonanten /p/, /t/, /k/, /s/ haben breitbandige Energie-Spikes →
# breitband-NR zerstört Artikulation ohne diese Maske
```

## Frisson-Schutz in Phasen mit Dynamik-Eingriff

```python
from backend.core.frisson_candidate_detector import get_frisson_detector

frisson_zones = get_frisson_detector().detect(original_audio, sr)
# Klimax-Segmente: NR-Strength × 0.85
# Strophen: NR-Strength × 1.15
# MDEM: Zwei-Stufen pre-SG + post-SG, Floor -1.0 LU
```

## VQI-Gate für Vokal-Phasen

```python
# PFLICHT nach vokal-beeinflussenden Phasen:
from backend.core.musical_goals.vocal_quality_index import compute_vqi

if panns_singing_confidence >= 0.35:
    vqi_after = compute_vqi(audio_out, sr)
    if vqi_after < vqi_before - 0.05:  # signifikante VQI-Regression
        return audio_in  # Rollback
```

## Passaggio-Glättung

```python
# PFLICHT in Pitch/Register-Phasen:
from backend.core.dsp.vocal_register_detector import detect_vocal_register_temporal

register_sequence = detect_vocal_register_temporal(audio, sr)
# glättet Brust→Kopf-Übergänge ±5 Frames linear
# verhindert Timbre-Knick bei Passaggio-Sprüngen
```
