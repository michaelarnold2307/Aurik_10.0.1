"""AudioSR Plugin Tests — v10.0.3 CPU-only architecture.

Covers: SBR-DSP fallback, code quality (no obsolete patches),
        warning suppression, budget guard, spectral extension.
ML inference (build_model + generate_batch) requires 13 GB RAM
and is tested separately via integration test.
"""

from __future__ import annotations

import numpy as np
import pytest


# ── SBR-DSP fallback ──────────────────────────────────────────────────

def test_audiosr_hf_extend_uses_sbr_fallback(monkeypatch):
    """SBR-DSP fallback works when spectral exciter fails."""
    from plugins.audiosr_plugin import AudioSRPlugin

    plugin = AudioSRPlugin()

    def _fail(_x, _sr):
        raise RuntimeError("exciter failed")

    monkeypatch.setattr(plugin, "_spectral_exciter", _fail)

    audio = np.random.randn(48_000).astype(np.float32) * 0.05
    out = plugin._hf_extend(audio, 48_000)

    assert out.shape == audio.shape
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) <= 1.0


# ── SBR spectral extension ────────────────────────────────────────────

@pytest.mark.parametrize("label,audio", [
    ("white noise", lambda: np.random.randn(48000).astype(np.float32) * 0.05),
    ("sine 1kHz", lambda: (np.sin(2 * np.pi * 1000 * np.arange(48000) / 48000)).astype(np.float32) * 0.3),
    ("silence", lambda: np.zeros(48000, dtype=np.float32)),
    ("soft clip", lambda: np.clip(np.random.randn(96000).astype(np.float32) * 0.5, -0.3, 0.3)),
])
def test_sbr_output_validity(label, audio):
    """SBR-DSP produces valid output for various signal types."""
    from plugins.audiosr_plugin import AudioSRPlugin

    plugin = AudioSRPlugin()
    a = audio()
    out = plugin._hf_extend(a, 48000)

    assert out.shape == a.shape, f"{label}: shape mismatch"
    assert np.all(np.isfinite(out)), f"{label}: non-finite values"
    assert np.max(np.abs(out)) <= 1.0, f"{label}: output exceeds [-1,1]"


def test_sbr_adds_spectral_energy():
    """SBR-DSP actually adds high-frequency energy."""
    from plugins.audiosr_plugin import AudioSRPlugin

    plugin = AudioSRPlugin()
    noise = np.random.randn(48000).astype(np.float32) * 0.05
    out = plugin._hf_extend(noise, 48000)

    f_in = np.mean(np.abs(np.fft.rfft(noise * np.hanning(len(noise)))))
    f_out = np.mean(np.abs(np.fft.rfft(out * np.hanning(len(out)))))

    assert f_out > f_in, f"HF not extended: {f_in:.4f} -> {f_out:.4f}"


# ── Code quality: no obsolete patches ─────────────────────────────────

def test_no_obsolete_patches_in_source():
    """Verify old GPU/mixed-device patches are fully removed."""
    with open("plugins/audiosr_plugin.py", encoding="utf-8") as f:
        code = f.read()

    obsolete = [
        ("_fsm.cpu()", "old ROCm-Fix v2: first_stage_model CPU move"),
        ('GPU-DDIM fehlgeschlagen', "old GPU-DDIM recovery message"),
        ('CPU-Retry fehlgeschlagen', "old CPU-Retry recovery message"),
        ("_patched_mel2wav", "old mel2wav monkey-patch"),
        ("_patched_decode", "old decode monkey-patch"),
        ('build_model(model_name="basic", device=str(_dev))', "old dynamic device selection"),
    ]

    for pattern, description in obsolete:
        assert pattern not in code, f"Obsolete code found: {description}"


def test_cpu_only_architecture():
    """Verify model is built with device='cpu'."""
    with open("plugins/audiosr_plugin.py", encoding="utf-8") as f:
        code = f.read()

    assert 'build_model(model_name="basic", device="cpu")' in code, \
        "Model must be built with device='cpu'"


def test_warning_suppression_active():
    """Verify RuntimeWarning filter for weight_norm is active."""
    with open("plugins/audiosr_plugin.py", encoding="utf-8") as f:
        code = f.read()

    assert 'filterwarnings' in code, "warnings import missing"
    assert 'invalid value encountered in multiply' in code, \
        "RuntimeWarning filter for weight_norm missing"


def test_nan_clean_after_load():
    """Verify nan_to_num clean loop exists after build_model."""
    with open("plugins/audiosr_plugin.py", encoding="utf-8") as f:
        code = f.read()

    assert "nan_to_num" in code, "nan_to_num clean missing"
    assert "for _p in model.parameters()" in code, "parameter clean loop missing"
    assert "isfinite" in code, "parameter finiteness check missing"


# ── Plugin structure ──────────────────────────────────────────────────

def test_plugin_has_required_methods():
    """Plugin exposes the expected public and private API."""
    from plugins.audiosr_plugin import AudioSRPlugin

    p = AudioSRPlugin()

    assert hasattr(p, "process"), "Missing public method: process"
    assert hasattr(p, "_hf_extend"), "Missing private method: _hf_extend"
    assert hasattr(p, "_spectral_band_replication"), "Missing: _spectral_band_replication"
    assert hasattr(p, "_spectral_exciter"), "Missing: _spectral_exciter"


def test_plugin_budget_constant():
    """AudioSR budget constant is reasonable."""
    from plugins.audiosr_plugin import _AUDIOSR_BUDGET_GB

    assert 4.0 < _AUDIOSR_BUDGET_GB < 7.0, \
        f"Budget {_AUDIOSR_BUDGET_GB} GB outside expected range [4,7]"


# ── SBR band-limited extension ────────────────────────────────────────

def test_sbr_extends_band_limited_signal():
    """SBR extends a band-limited (0-4kHz) signal above the cutoff."""
    from plugins.audiosr_plugin import AudioSRPlugin

    sr = 48000
    n = sr  # 1 second
    noise = np.random.randn(n).astype(np.float32) * 0.1
    spec = np.fft.rfft(noise)
    cutoff_bin = int(4000 / (sr / 2) * len(spec))
    spec[cutoff_bin:] = 0
    bl = np.fft.irfft(spec, n).astype(np.float32)
    bl = bl / max(1e-8, np.max(np.abs(bl))) * 0.1

    plugin = AudioSRPlugin()
    out = plugin._hf_extend(bl, sr)

    f_in_high = np.mean(np.abs(np.fft.rfft(bl * np.hanning(n))[cutoff_bin:]))
    f_out_high = np.mean(np.abs(np.fft.rfft(out * np.hanning(n))[cutoff_bin:]))

    assert f_out_high > f_in_high * 2, \
        f"HF not extended: {f_in_high:.4f} -> {f_out_high:.4f}"
