---
applyTo: "{backend/core/dsp/*.py,plugins/*.py}"
---

# DSP / Plugin-Regeln (normativ, Aurik 9.12.x)

## ML-Device — IMMER über ml_device_manager

```python
# VERBOTEN:
model.to("cuda")                        # direkt, ignoriert AMD/ROCm
providers = ["CUDAExecutionProvider"]   # ignoriert DirectML/ROCm

# RICHTIG (Heavy-Plugin):
from backend.core.ml_device_manager import get_torch_device, get_ort_providers
model.to(get_torch_device("PluginName"))   # fp16 + Tier automatisch
session = ort.InferenceSession(path, providers=get_ort_providers("PluginName"))

# Light-Plugin / DSP:
model.to("cpu")
torch.set_num_threads(os.cpu_count())
providers = ["CPUExecutionProvider"]
```

## §2.62 Psychoakustischer Masking-Guard (NR-Algorithmen)

```python
# PFLICHT vor jedem NR-Aufruf (DeepFilterNet, OMLSA, SGMSE+):
from backend.core.dsp.psychoacoustics import compute_masking_threshold_iso11172

masking_threshold = compute_masking_threshold_iso11172(audio, sr)

# NR-Gain-Floor pro Band — verhindert klinische Stille-Artefakte:
for band in range(n_bands):
    G_floor[band] = max(0.10, masking_threshold[band] / noise_estimate[band])
    # VERBOTEN: G_floor < 0.10 in Bändern mit Musik-Energie > -60 dBFS

# Ergebnis: kein "totes Stille"-Artefakt zwischen Phrasen
```

## DeepFilterNet — Energy-Bias-Pflicht (§0j)

```python
# VERBOTEN: DeepFilterNet ohne energy_bias auf Vokal/Instrumental
# Harmonik-Regionen werden ohne Bias als Rauschen klassifiziert

# RICHTIG:
if panns_vocals >= 0.4:
    energy_bias = -6.0  # dB — Vokal
elif is_instrumental:
    energy_bias = -9.0  # dB — Instrumental

# Vokalregister-adaptiv (§2.35c):
# Kopfstimme: -3 dB | Brust: -6 dB | Fry/Flüstern: -9 dB
from backend.core.dsp.vocal_register_detector import detect_vocal_register_temporal
register = detect_vocal_register_temporal(audio, sr)
energy_bias = _REGISTER_BIAS[register.dominant]
```

## HNR-Guard nach NR auf Gesangsmaterial

```python
# PFLICHT wenn panns_singing >= 0.25 + nach DFN/SGMSE+/OMLSA:
from backend.core.dsp.hnr_guard import apply_hnr_blend

audio_out = apply_hnr_blend(audio_pre_nr, audio_post_nr, sr)
# ΔHNR > 3 dB → automatischer Dry-Blend
# verhindert "klinischen" Klang nach aggressivem NR
```

## LPC-Ordnung — Material-abhängig

```python
# VERBOTEN: LPC-Ordnung < 16 bei 16 kHz oder < 30 bei 48 kHz
# RICHTIG:
if analysis_sr == 48000:
    lpc_order = 30  # bis 40 für breite Formant-Tracks
elif analysis_sr == 16000:
    lpc_order = 16  # bevorzugt für Shellac BW ≤ 8 kHz (Downsampling → 16k)
# lpc_formant_tracker.py verwendet _LPC_ORDER=16 + _LPC_ANALYSIS_SR=16000
```

## Singleton-Pattern — ALLE Kernmodule

```python
import threading

_instance = None
_lock = threading.Lock()

def get_my_plugin():
    global _instance  # oder list-Container: _holder = [None]
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MyPlugin()
    return _instance
```

## ONNX-Chunking (Heavy-Plugins)

```python
# KANONISCH — Chunk-Verarbeitung mit Overlap-Add:
chunk_size = 65536  # ~1.4s bei 48kHz
overlap = 4096
for i in range(0, n_samples, chunk_size - overlap):
    chunk = audio[..., i: i + chunk_size]
    out_chunk = session.run(None, {"input": chunk})[0]
    # Overlap-Add mit Hann-Fenster
    output[..., i: i + len(out_chunk)] += out_chunk * window

# OOM-Fallback → DSP-Kette, nie Crash:
try:
    result = heavy_model.process(audio)
except (RuntimeError, MemoryError):
    metadata["ml_fallbacks_used"]["model_name"] = True
    result = _dsp_fallback(audio, sr)
```

## MIIPHER-Fallback (SNR < 10 dB + Gesang)

```python
from plugins.miipher_plugin import get_miipher_plugin

if noise_snr_db < 10.0 and panns_singing >= 0.35:
    miipher = get_miipher_plugin()
    if miipher.should_activate(noise_snr_db, panns_singing):
        audio = miipher.enhance(audio, sr)
        # Intern: Stub → DeepFilterNet(-6dB) → Wiener-Fallback
```

## Timbral Coherence Guard

```python
from backend.core.dsp.timbral_coherence_guard import (
    extract_song_noise_profile,
    compute_timbral_coherence_score,
)

noise_profile = extract_song_noise_profile(audio_pre_pipeline, sr)
# ... nach NR-Phasen ...
coherence = compute_timbral_coherence_score(audio_post_nr, noise_profile, sr)
assert coherence >= 0.80, "Timbral Coherence unter Pflichtgrenze (§CSTC)"
# Vinyl → rosa Rauschtextur; Tape → Brown+HF-Hiss; CD → Flat/Weiß
```

## Cross-Segment Timbral Coherence — Rauschtextur

```python
# Spektrale Form des Restrauschens MUSS zum Trägerprofil passen:
# Vinyl:   rosa   (1/f)
# Tape:    Brown + HF-Hiss
# CD:      Weiß / Flat
# Shellac: Spezifisches Oberflächenprofil
# Kohärenz-Score ≥ 0.80 Pflicht (§0a Rauschtextur-Invariante)
```

## Multi-Singer Detection

```python
from backend.core.dsp.vocal_register_detector import detect_multi_singer

if detect_multi_singer(audio, sr, panns_singing):
    metadata["multi_singer"] = True
    # Resemblyzer-Gate ÜBERSPRINGEN (Embedding für Einzelidentität, nicht Duett)
    # singer_identity_cosine-Gate deaktiviert
```

## Signal-relatives Gate (V04)

```python
# VERBOTEN:
gate_dbfs = -36.0  # feste Konstante

# RICHTIG:
from backend.core.dsp.gain_utils import compute_signal_relative_gate_dbfs
gate_dbfs = compute_signal_relative_gate_dbfs(
    pre_phase_audio, material_key=material_type
)
# reference_for_gate=pre_phase_audio — IMMER
```

## Bandfilter — Zero-Phase

```python
# VERBOTEN: sosfilt(sos, audio) addiert zu Original
# RICHTIG:
from scipy.signal import sosfiltfilt
filtered = sosfiltfilt(sos, audio)  # zero-phase überall wo Band auf Signal addiert
```

## Peak-Guard

```python
# VERBOTEN:
peak = np.max(np.abs(audio))

# RICHTIG:
peak = np.percentile(np.abs(audio), 99.9)
```
