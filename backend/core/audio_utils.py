import numpy as np


def safe_to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert audio to mono, handling both (N, 2) and (2, N) layouts safely.

    Respects §2.51 Stereo-Kohärenz-Invariante: Convert to mono without loss
    of phase information or spectral coherence.

    Args:
        audio: Input audio, 1D (mono) or 2D (stereo in any orientation)

    Returns:
        Mono audio as 1D numpy array (or scalar for degenerate inputs)
    """
    if audio.ndim == 1:
        return audio

    # Ensure float64 for precision
    audio = audio.astype(np.float64)

    # Determine orientation and convert safely
    if audio.shape[0] == 2 and audio.shape[1] > 2:
        # (2, N) channels-first → mean over channels (axis=0)
        return np.mean(audio, axis=0)
    elif audio.shape[0] == 2 and audio.shape[1] == 2:
        # Edge case: exactly (2, 2) — ambiguous, but treat as (2, N) channels-first
        # This gives a (2,) output
        return np.mean(audio, axis=0)
    elif audio.shape[1] == 2:
        # (N, 2) channels-last → mean over channels (axis=1)
        return np.mean(audio, axis=1)
    else:
        # Ambiguous: use heuristic based on which dimension is smaller
        # (channels are typically 2, samples >> 2)
        axis = 0 if audio.shape[0] < audio.shape[1] else 1
        return np.mean(audio, axis=axis)


def stereo_channel_view(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return stereo channels as 1D arrays for either (2, N) or (N, 2) layout."""
    if audio.ndim != 2:
        raise ValueError(f"Stereo audio must be 2D, got shape {audio.shape}")
    if audio.shape[0] == 2 and audio.shape[1] > 2:
        return audio[0], audio[1]
    if audio.shape[1] == 2:
        return audio[:, 0], audio[:, 1]
    if audio.shape[0] == 2 and audio.shape[1] == 2:
        return audio[0], audio[1]
    raise ValueError(f"Unsupported stereo layout: {audio.shape}")


def stereo_like(left: np.ndarray, right: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Rebuild stereo audio while preserving the template orientation."""
    if template.ndim != 2:
        raise ValueError(f"Stereo template must be 2D, got shape {template.shape}")
    if template.shape[0] == 2 and template.shape[1] > 2:
        return np.vstack([left, right])
    if template.shape[1] == 2:
        return np.column_stack([left, right])
    if template.shape[0] == 2 and template.shape[1] == 2:
        return np.vstack([left, right])
    raise ValueError(f"Unsupported stereo template layout: {template.shape}")


def to_channels_last(audio: np.ndarray) -> tuple["np.ndarray", bool]:
    """Normalize stereo audio to (N, 2) channels-last layout.

    Returns (normalized_audio, was_transposed) so the caller can restore the
    original orientation with ``restore_layout``.
    """
    if audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 2:
        return audio.T, True
    return audio, False


def restore_layout(audio: np.ndarray, was_transposed: bool) -> np.ndarray:
    """Undo a ``to_channels_last`` transposition if it was applied."""
    if was_transposed and audio.ndim == 2:
        return audio.T
    return audio


def audio_sample_count(audio: np.ndarray) -> int:
    """Return the time-axis sample count for mono or stereo audio."""
    if audio.ndim == 1:
        return int(audio.shape[0])
    if audio.ndim == 2:
        if audio.shape[0] == 2 and audio.shape[1] > 2:
            return int(audio.shape[1])
        return int(audio.shape[0])
    raise ValueError(f"Unsupported audio rank for sample count: {audio.shape}")


def compute_gated_rms_linear(sig: np.ndarray, gate_dbfs: float = -50.0) -> float:
    """Compute frame-gated RMS in linear scale (stereo-safe via mono energy)."""
    x = np.asarray(sig, dtype=np.float64)
    if x.size == 0:
        return 0.0
    if x.ndim == 2:
        if x.shape[0] <= 2 and x.shape[1] > x.shape[0]:
            x = np.mean(x, axis=0)
        else:
            x = np.mean(x, axis=1)
    frame = 480
    n = int(x.shape[0])
    if n < frame:
        return float(np.sqrt(np.mean(x * x)) + 1e-12)

    gate_lin2 = 10.0 ** (gate_dbfs / 10.0)
    vals: list[float] = []
    for i in range(0, n - frame + 1, frame):
        f = x[i : i + frame]
        p = float(np.mean(f * f))
        if p > gate_lin2:
            vals.append(p)
    if not vals:
        return float(np.sqrt(np.mean(x * x)) + 1e-12)
    return float(np.sqrt(float(np.mean(vals))) + 1e-12)


def compute_gated_rms_dbfs(sig: np.ndarray, gate_dbfs: float = -50.0) -> float:
    """Compute frame-gated RMS in dBFS."""
    rms = compute_gated_rms_linear(sig, gate_dbfs=gate_dbfs)
    return float(20.0 * np.log10(rms + 1e-12))


def apply_musical_gain_envelope(
    audio: np.ndarray,
    gain: float,
    gate_dbfs: float = -50.0,
    crossfade_ms: float = 10.0,
    sr: int = 48000,
) -> np.ndarray:
    """§2.45a-II: Apply makeup gain ONLY to musical frames, leaving silence at gain=1.0.

    Silence frames (frame RMS <= gate_dbfs) remain at unity gain.
    A short crossfade (box-blur on the gate envelope) prevents hard clicks at
    music/silence boundaries.

    Args:
        audio:         Input audio (1D or 2D samples-first or channels-first float32).
        gain:          Linear gain factor (>= 1.0; values <= 1.0005 are skipped).
        gate_dbfs:     Frame energy threshold below which a frame is classified as silence.
        crossfade_ms:  Width of the smoothing window at music/silence transitions.
        sr:            Sample rate used to convert crossfade_ms to samples.

    Returns:
        Audio with gain applied only on musical frames, same shape and dtype.
    """
    if gain <= 1.0005:
        return audio
    arr = np.asarray(audio, dtype=np.float32)
    was_2d = arr.ndim == 2
    # Build mono energy signal for gate detection
    if was_2d:
        ch_first = arr.shape[0] <= 2 and arr.shape[1] > arr.shape[0]
        mono = np.mean(arr, axis=0) if ch_first else np.mean(arr, axis=1)
    else:
        mono = arr
    n = len(mono)
    frame_len = 480  # 10 ms @ 48 kHz
    n_full = max(1, n // frame_len)
    gate_env = np.zeros(n, dtype=np.float32)
    for fi in range(n_full):
        s = fi * frame_len
        e = min(s + frame_len, n)
        chunk = mono[s:e].astype(np.float64)
        rms_db = float(20.0 * np.log10(float(np.sqrt(np.mean(chunk * chunk) + 1e-12)) + 1e-12))
        if rms_db > gate_dbfs:
            gate_env[s:e] = 1.0
    tail_s = n_full * frame_len
    if tail_s < n:
        tail = mono[tail_s:].astype(np.float64)
        rms_db = float(20.0 * np.log10(float(np.sqrt(np.mean(tail * tail) + 1e-12)) + 1e-12))
        if rms_db > gate_dbfs:
            gate_env[tail_s:] = 1.0
    # Smooth transitions
    cf_samples = max(1, int(crossfade_ms * sr / 1000.0))
    if cf_samples > 1:
        kernel = np.ones(cf_samples, dtype=np.float32) / cf_samples
        gate_env = np.convolve(gate_env, kernel, mode="same")
        gate_env = np.clip(gate_env, 0.0, 1.0)
    per_sample_gain = (1.0 + (gain - 1.0) * gate_env).astype(np.float32)
    if was_2d:
        ch_first = arr.shape[0] <= 2 and arr.shape[1] > arr.shape[0]
        if ch_first:
            return (arr * per_sample_gain[np.newaxis, :]).astype(np.float32)
        return (arr * per_sample_gain[:, np.newaxis]).astype(np.float32)
    return (arr * per_sample_gain).astype(np.float32)
