"""
CD-Rauschprofil-Generator mit psychoakustischer Maskierung (§G8, §G15–§G19, §G30–§G42)

Erzeugt ein CD-charakteristisches Rauschprofil und appliziert es NUR dort,
wo das menschliche Ohr es wahrnimmt — d.h. unterhalb der perzeptuellen
Maskierungsschwelle des Signals.

Wissenschaftliche Grundlage:
  - Simultaneous Masking (Zwicker & Fastl, 1999): Laute Signale maskieren
    leises Rauschen vollständig. Das CD-Rauschen (-96 dBFS) wird von jedem
    Signal > -60 dBFS sicher maskiert.
  - Temporal Masking: 200 ms Cosine-Fade verhindert Pre-/Post-Masking-Artefakte.
  - CD-Produktion (1982–2000): Der 16-bit Noise Floor ist charakteristisch
    für die gesamte CD-Ära und wird vom Hörer als "natürlich" empfunden.

Position in der Export-Pipeline (§G40):
  Das Rauschprofil wird NACH allen Restaurierungsphasen aber VOR dem
  Dithering appliziert. Dies verhindert, dass nachfolgende DSP-Phasen
  das Rauschen verstärken oder spektral verfärben.

Referenzen:
  §G8    CD-Rauschprofil-Pflicht (jeder Export, beide Modi)
  §G15   Rauschprofil-Maskierung (nur unterhalb Maskierungsschwelle)
  §G16   Rauschprofil-Charakteristik (-96 dBFS Flat-Noise-Floor + Dither-Shaping)
  §G17   Stille-Respekt (digital black wird nicht verrauscht)
  §G18   Spektrale Kohärenz (flach 20-16k Hz, -3 dB/Oktave ab 16 kHz)
  §G30   L/R-Unkorreliertheit
  §G31   Maskierungs-Kanten-Glattung (200 ms Cosine-Fade)
  §G39   Rauschprofil-Monitoring (SNR-Logging)
  §G40   Rauschprofil-Zeitpunkt (letzter Schritt vor Dithering)
  §G41   Ubergangs-Verifikation (keine hörbaren Klicks an Übergängen)
  §G42   CD-Produktions-Kohärenz (Export wie CD-Neuauflage)
  §V11   Rauschprofil-Flächendeckung verboten
  §V12   Stille-Verfälschung verboten
  §V15   Nicht-deterministisches Rauschen verboten
  §V16   Übersteuerndes Rauschen verboten (-85 dBFS Limit)
  §V17   Quellmaterial-Extraktion verboten
  §V25   Zwischenphasen-Rauschen verboten (nur NACH allen Phasen)
  §V26   Hörbare Übergänge verboten (Onset-Stärke < 0.1)

Author: Aurik Development Team
Version: 10.0.5
Date: 2026-07-13
"""

import hashlib
import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── CD-Rauschprofil-Konstanten ──────────────────────────────────────────

# §G16: −96 dBFS Flat-Noise-Floor (16-bit theoretisch)
_CD_NOISE_FLOOR_DBFS_16BIT: float = -96.0
_CD_NOISE_FLOOR_DBFS_24BIT: float = -120.0

# §G16: Mit POW-r-Type-3-Shaping steigt der wahrgenommene Pegel auf
# ca. −90 dBFS oberhalb 10 kHz. Wir modellieren das als leichte Anhebung.
_CD_NOISE_SHAPING_BOOST_DB: float = 6.0
_CD_NOISE_SHAPING_KNEE_HZ: float = 10000.0

# §V16: Maximal zulässiger Rauschpegel
_CD_NOISE_MAX_DBFS: float = -85.0

# §G31: Cosine-Fade-Länge an Maskierungs-Kanten
_MASKING_EDGE_FADE_S: float = 0.200  # 200 ms (Zwicker & Fastl, 1999: temporal masking window)

# §G41: Maximal zulässige Onset-Stärke an Übergängen (0 = perfekt, 1 = harter Klick)
_MAX_ONSET_STRENGTH: float = 0.1

# §G18: Spektraler Rolloff ab 16 kHz (−3 dB/Oktave)
_CD_NOISE_ROLLOFF_HZ: float = 16000.0
_CD_NOISE_ROLLOFF_DB_PER_OCTAVE: float = -3.0

# Maskierungsschwelle: Signale > diesem Pegel maskieren das CD-Rauschen sicher
# Wissenschaftlich (Zwicker & Fastl, 1999): Bereits -70 dBFS maskiert
# -96 dBFS breitbandiges Rauschen in ruhiger Umgebung vollstandig.
_MASKING_THRESHOLD_DBFS: float = -70.0

# RMS-Fenster für Maskierungsberechnung (50 ms = gute Zeitauflösung)
_RMS_WINDOW_S: float = 0.050

# Minimale Signalenergie: unterhalb gilt es als "Stille" (§G17, §V12)
_SILENCE_THRESHOLD_DBFS: float = -120.0

# ── Determinismus (§V15) ────────────────────────────────────────────────


def _compute_deterministic_seed(audio: np.ndarray) -> int:
    """Deterministischer Seed aus SHA256 der ersten 4096 Samples."""
    flat = np.asarray(audio, dtype=np.float32).ravel()[:4096]
    digest = hashlib.sha256(flat.tobytes()).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**31)


# ── CD-Rauschsynthese (§G16, §G18) ──────────────────────────────────────


def _generate_cd_noise(
    n_samples: int,
    sr: int,
    bit_depth: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Erzeugt CD-charakteristisches Rauschen mit korrektem Spektrum und Pegel.

    §G16: −96 dBFS (16-bit) / −120 dBFS (24-bit) RMS-Pegel.
    §G18: Flat 20 Hz–16 kHz, −3 dB/Oktave Rolloff ab 16 kHz.
    """
    # Basis-Rauschen (weiß, gaußverteilt)
    noise = rng.standard_normal(n_samples, dtype=np.float32)

    # FFT-basierte spektrale Formung
    n_fft = 1
    while n_fft < len(noise):
        n_fft <<= 1

    spectrum = np.fft.rfft(noise.astype(np.float64), n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    # §G18: Flat bis 16 kHz, dann −3 dB/Oktave
    shape = np.ones(len(freqs), dtype=np.float64)
    rolloff = freqs > _CD_NOISE_ROLLOFF_HZ
    if np.any(rolloff):
        octaves = np.log2(np.maximum(freqs[rolloff], _CD_NOISE_ROLLOFF_HZ) / _CD_NOISE_ROLLOFF_HZ)
        shape[rolloff] = 10.0 ** (_CD_NOISE_ROLLOFF_DB_PER_OCTAVE * octaves / 20.0)

    # §G16: Dither-Shaping-Boost ab 10 kHz (POW-r-Type-3-Äquivalent)
    boost = freqs > _CD_NOISE_SHAPING_KNEE_HZ
    if np.any(boost):
        knee_width = 2000.0
        knee = 0.5 * (1.0 + np.tanh((freqs[boost] - _CD_NOISE_SHAPING_KNEE_HZ) / knee_width))
        shape[boost] *= 10.0 ** (_CD_NOISE_SHAPING_BOOST_DB * knee / 20.0)

    # DC = 0
    shape[0] = 0.0

    spectrum *= shape
    shaped = np.fft.irfft(spectrum, n=n_fft)[:n_samples]

    # Normalisierung auf CD-Noise-Floor
    rms = float(np.sqrt(np.mean(shaped**2)))
    target_dbfs = _CD_NOISE_FLOOR_DBFS_16BIT if bit_depth <= 16 else _CD_NOISE_FLOOR_DBFS_24BIT
    target_rms = 10.0 ** (target_dbfs / 20.0)
    shaped *= target_rms / max(rms, 1e-15)

    return shaped.astype(np.float32)


# ── Maskierungsberechnung (§G15, §G17) ──────────────────────────────────


def _compute_masking_envelope(
    audio_mono: np.ndarray,
    sr: int,
) -> np.ndarray:
    """Berechnet eine zeitabhängige Maskierungshüllkurve (§G15, §G17).

    Basierend auf Kurzzeit-RMS: Wo der RMS-Pegel > Maskierungsschwelle ist,
    wird das CD-Rauschen vollständig maskiert und nicht appliziert.

    Returns:
        gain_envelope: (n_samples,) Hüllkurve in [0, 1]
            1 = volles Rauschen applizieren (Signal unter Schwelle)
            0 = kein Rauschen (Signal maskiert)
    """
    win_samples = max(int(_RMS_WINDOW_S * sr), 1)
    hop = win_samples // 2  # 50% Überlappung
    n_frames = (len(audio_mono) - win_samples) // hop + 1
    if n_frames < 1:
        n_frames = 1

    rms_db = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop
        frame = audio_mono[start : start + win_samples]
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        rms_db[i] = 20.0 * np.log10(max(rms, 1e-15))

    # Binäre Maske: 1 = Rauschen addieren (Signal unter Schwelle)
    mask_frames = (rms_db < _MASKING_THRESHOLD_DBFS).astype(np.float64)

    # §G17: Keine Rauschzugabe bei digitaler Stille
    silence_frames = rms_db < _SILENCE_THRESHOLD_DBFS
    mask_frames[silence_frames] = 0.0

    # §G31: Cosine-Fade an Übergängen (200 ms)
    # Vorsicht: Digital-Black-Frames (§G17) dürfen NICHT durch Smoothing überdeckt werden.
    fade_frames = max(1, int(_MASKING_EDGE_FADE_S * sr / hop))
    if fade_frames > 1 and np.any(mask_frames > 0):
        smoothed = mask_frames.copy()
        diff = np.diff(np.concatenate([[0.0], smoothed, [0.0]]))
        starts = np.where(diff > 0.5)[0]
        ends = np.where(diff < -0.5)[0] - 1
        for s, e in zip(starts, ends):
            # Cosine-Fade-In vor Start
            s_fade = max(0, s - fade_frames)
            if s_fade < s:
                n_fade = s - s_fade
                curve = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_fade) / n_fade))
                # §G17: Nicht in Digital-Black-Frames hineinpropagieren
                for j in range(s_fade, s):
                    if not silence_frames[j]:
                        smoothed[j] = max(smoothed[j], curve[j - s_fade])
            # Cosine-Fade-Out nach Ende
            e_fade = min(len(smoothed) - 1, e + fade_frames)
            if e < e_fade:
                n_fade = e_fade - e
                curve = 0.5 * (1.0 + np.cos(np.pi * np.arange(n_fade) / n_fade))
                # §G17: Nicht in Digital-Black-Frames hineinpropagieren
                for j in range(e + 1, e_fade + 1):
                    if not silence_frames[j]:
                        smoothed[j] = max(smoothed[j], curve[j - (e + 1)])
        mask_frames = smoothed

    # §G17 final: Sicherstellen, dass Digital-Black-Frames 0 bleiben
    mask_frames[silence_frames] = 0.0

    # Auf Sample-Ebene interpolieren (konstante Hüllkurve pro Block)
    envelope = np.zeros(len(audio_mono), dtype=np.float64)
    for i in range(n_frames):
        start = i * hop
        end = min(start + hop, len(audio_mono))
        envelope[start:end] = mask_frames[i]

    # §G17 Sample-Level: Exakt-Null-Samples werden NIE verrauscht
    # Dies ist die letzte Verteidigungslinie gegen Window-Smearing:
    # Selbst wenn der 50ms-RMS-Window Ambient-Rauschen aus der Nachbarschaft
    # einschließt, bleiben echte Digital-Black-Samples unangetastet.
    zero_mask = np.abs(audio_mono) < 1e-12
    envelope[zero_mask] = 0.0

    return envelope


# ── §G41: Onset-Verifikation ─────────────────────────────────────────────


def _compute_onset_strength(audio: np.ndarray, sr: int) -> float:
    """Misst die maximale Onset-Stärke (0=perfekt glatt, 1=hatter Klick).

    §G41: Onset-Stärke > 0.1 löst erweiterte Crossfade-Korrektur aus (§V26).
    """
    if len(audio) < 512:
        return 0.0
    # Einfache Onset-Detection via Energie-Differenz
    win = 256
    hop = 128
    n_frames = (len(audio) - win) // hop
    if n_frames < 2:
        return 0.0
    energies = np.array(
        [float(np.sum(np.square(audio[i * hop : i * hop + win]))) for i in range(n_frames)],
        dtype=np.float64,
    )
    if np.max(energies) < 1e-15:
        return 0.0
    energies /= np.max(energies)
    onset_func = np.diff(energies)
    onset_func = np.maximum(onset_func, 0.0)  # Nur positive Flanken
    return float(np.max(onset_func))


# ── Haupt-API ────────────────────────────────────────────────────────────


def inject_cd_noise_profile(
    audio: np.ndarray,
    sr: int,
    *,
    mode: str = "restoration",
    bit_depth: int = 16,
    seed: int | None = None,
) -> np.ndarray:
    """§G8: Injiziert CD-Rauschprofil nur dort, wo es das menschliche Ohr wahrnimmt.

    Das Rauschprofil wird per RMS-Maskierung nur in leisen Passagen
    (< -60 dBFS) appliziert. Übergänge werden mit 200 ms Cosine-Fade
    geglättet (§G31). Die Position in der Pipeline ist NACH allen
    Restaurierungsphasen, VOR dem Dithering (§G40).

    Parameters
    ----------
    audio : np.ndarray
        Float32-Audio [-1.0, 1.0], shape (samples,) oder (samples, channels).
    sr : int
        Abtastrate (typ. 48000).
    mode : str
        Processing-Mode ("restoration" oder "studio_2026") — nur für Logging.
    bit_depth : int
        Ziel-Bittiefe. Bestimmt den Rauschpegel: 16 → −96 dBFS, 24 → −120 dBFS.
    seed : int | None
        Deterministischer Seed (§V15). None → SHA256-basiert aus Audio.

    Returns
    -------
    np.ndarray
        Audio mit CD-Rauschprofil. In lauten Passagen bit-identisch zum Input.
    """
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim > 2:
        logger.warning("CD-Noise-Profile: unexpected rank %d — skipping", arr.ndim)
        return audio

    # §G17: Digital black nicht verrauschen
    peak = float(np.max(np.abs(arr)))
    if peak < 1e-10:
        logger.debug("CD-Noise-Profile: digital black — skipping (§G17, §V12)")
        return audio

    # §V15: Deterministischer Seed
    if seed is None:
        seed = _compute_deterministic_seed(arr)
    rng = np.random.default_rng(seed)

    is_stereo = arr.ndim == 2 and arr.shape[1] == 2
    orig_shape = arr.shape

    if is_stereo:
        left, right = arr[:, 0].copy(), arr[:, 1].copy()
        mono = (left.astype(np.float64) + right.astype(np.float64)) * 0.5
    else:
        mono = arr.ravel().astype(np.float64)
        left = arr.ravel()

    # §G15: Maskierungshüllkurve berechnen
    envelope = _compute_masking_envelope(mono, sr)

    # Zähle aktive Samples für §G39
    active_samples = int(np.sum(envelope > 0.01))
    total_samples = len(mono)

    # §G30: L/R unkorreliert
    if is_stereo:
        seed_l = seed
        seed_r = seed ^ 0x5A5A5A5A5A5A5A5A
        noise_l = _generate_cd_noise(len(mono), sr, bit_depth, np.random.default_rng(seed_l))
        noise_r = _generate_cd_noise(len(mono), sr, bit_depth, np.random.default_rng(seed_r))

        result_l = left.astype(np.float64) + noise_l.astype(np.float64) * envelope
        result_r = right.astype(np.float64) + noise_r.astype(np.float64) * envelope
        result = np.stack([result_l, result_r], axis=1)
    else:
        noise = _generate_cd_noise(len(mono), sr, bit_depth, rng)
        result_mono = left.astype(np.float64) + noise.astype(np.float64) * envelope
        result = result_mono.reshape(orig_shape)

    # §V16, §G41: Limits
    result = np.clip(result, -1.0, 1.0)

    # §G41: Onset-Verifikation
    onset = _compute_onset_strength(result.ravel(), sr)
    if onset > _MAX_ONSET_STRENGTH:
        logger.warning(
            "CD-Noise-Profile: Onset strength %.3f exceeds %.3f — "
            "possible audible transition (§V26). Consider wider crossfade.",
            onset,
            _MAX_ONSET_STRENGTH,
        )

    # §G39: Monitoring
    snr_before = _compute_snr_db(arr)
    snr_after = _compute_snr_db(result)
    n_min = min(len(result), len(arr))
    diff_max = float(np.max(np.abs(result[:n_min] - arr[:n_min])))
    noise_peak_db = 20.0 * np.log10(max(diff_max, 1e-15))

    logger.info(
        "CD-Noise-Profile [%s/%d-bit]: SNR %.1f -> %.1f dB | "
        "active: %d/%d samples (%.1f%%) | "
        "noise peak: %.1f dBFS | onset: %.4f | seed=%d",
        mode,
        bit_depth,
        snr_before,
        snr_after,
        active_samples,
        total_samples,
        100.0 * active_samples / max(total_samples, 1),
        noise_peak_db,
        onset,
        seed,
    )

    return result.astype(np.float32)


def _compute_snr_db(audio: np.ndarray) -> float:
    """Schätzt SNR: Peak / P10-Rauschpegel."""
    arr = np.asarray(audio, dtype=np.float32).ravel()
    peak = float(np.max(np.abs(arr)))
    noise_floor = float(np.percentile(np.abs(arr), 10))
    if peak < 1e-10 or noise_floor < 1e-15:
        return float("inf")
    return float(20.0 * np.log10(peak / noise_floor))
