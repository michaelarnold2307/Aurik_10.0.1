import numpy as np


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
