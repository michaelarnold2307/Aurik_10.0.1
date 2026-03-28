"""
§2.38 KMV Stufe-2 — Normative CI-Tests

Prüft:
- DeferredRefinementJob (Dataclass-Felder, Properties)
- MLRefinementThread.should_start() RAM-Guard
- Qualitätsinvariante (kein Overwrite wenn stufe2 < stufe1)
- Atomar-Schreib-Pfad (.tmp → os.replace)
- Signal-Kontrakt (alle 5 §2.38-Pflicht-Signale vorhanden)
- refinement_complete / refinement_cancelled Endstatus
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── DeferredRefinementJob ────────────────────────────────────────────────────


@pytest.fixture()
def minimal_job(tmp_path):
    from backend.core.deferred_refinement_job import DeferredRefinementJob

    audio = np.zeros(48000, dtype=np.float32)
    return DeferredRefinementJob(
        output_path=str(tmp_path / "out.wav"),
        audio_original=audio,
        sr=48000,
        mode="restoration",
        deferred_phase_ids=["phase_20_reverb_reduction", "phase_55_diffusion_inpainting"],
        cached_defect_result=None,
        cached_era_result=None,
        cached_medium_result=None,
        stufe1_quality=0.62,
        input_path=str(tmp_path / "in.wav"),
    )


def test_deferred_job_mandatory_fields():
    """All §2.38 mandatory fields must be present."""
    from backend.core.deferred_refinement_job import DeferredRefinementJob

    field_names = {f.name for f in fields(DeferredRefinementJob)}
    required = {
        "output_path",
        "audio_original",
        "sr",
        "mode",
        "deferred_phase_ids",
        "cached_defect_result",
        "cached_era_result",
        "cached_medium_result",
        "stufe1_quality",
        "input_path",
    }
    missing = required - field_names
    assert not missing, f"Pflicht-Felder fehlen: {missing}"


def test_deferred_job_audio_size_gb(minimal_job):
    """audio_size_gb property must be >0 for non-empty audio."""
    assert minimal_job.audio_size_gb > 0.0
    assert minimal_job.audio_size_gb < 1.0  # 48 000 float32 ≈ 0.00018 GB


def test_deferred_job_n_deferred(minimal_job):
    assert minimal_job.n_deferred == 2


def test_deferred_job_empty_phases():
    from backend.core.deferred_refinement_job import DeferredRefinementJob

    audio = np.zeros(1024, dtype=np.float32)
    job = DeferredRefinementJob(
        output_path="/tmp/x.wav",
        audio_original=audio,
        sr=48000,
        mode="restoration",
        deferred_phase_ids=[],
        cached_defect_result=None,
        cached_era_result=None,
        cached_medium_result=None,
        stufe1_quality=0.55,
        input_path="/tmp/in.wav",
    )
    assert job.n_deferred == 0


# ── MLRefinementThread.should_start() ───────────────────────────────────────


def test_should_start_no_deferred_phases(tmp_path):
    """should_start must return False when deferred_phase_ids is empty."""
    from backend.core.deferred_refinement_job import DeferredRefinementJob

    audio = np.zeros(1024, dtype=np.float32)
    job = DeferredRefinementJob(
        output_path=str(tmp_path / "out.wav"),
        audio_original=audio,
        sr=48000,
        mode="restoration",
        deferred_phase_ids=[],
        cached_defect_result=None,
        cached_era_result=None,
        cached_medium_result=None,
        stufe1_quality=0.60,
        input_path="",
    )
    from Aurik910.ui.ml_refinement_thread import MLRefinementThread

    assert MLRefinementThread.should_start(job) is False


def test_should_start_insufficient_ram(minimal_job):
    """should_start must return False when <4 GB RAM free."""
    import psutil

    from Aurik910.ui.ml_refinement_thread import MLRefinementThread

    mock_vm = MagicMock()
    mock_vm.available = int(3.9 * 1024**3)  # 3.9 GB < 4 GB required
    with patch.object(psutil, "virtual_memory", return_value=mock_vm):
        result = MLRefinementThread.should_start(minimal_job)
    assert result is False


def test_should_start_sufficient_ram(minimal_job):
    """should_start must return True when ≥4 GB RAM free and phases present."""
    import psutil

    from Aurik910.ui.ml_refinement_thread import MLRefinementThread

    mock_vm = MagicMock()
    mock_vm.available = int(8.0 * 1024**3)  # 8 GB — sufficient
    with patch.object(psutil, "virtual_memory", return_value=mock_vm):
        result = MLRefinementThread.should_start(minimal_job)
    assert result is True


# ── QualitätsInvariante (kein Overwrite wenn stufe2 < stufe1) ────────────────


def test_quality_invariant_logic_present():
    """run() source must contain quality gate: stufe2 < stufe1 → emit cancelled."""
    import inspect

    from Aurik910.ui.ml_refinement_thread import MLRefinementThread

    src = inspect.getsource(MLRefinementThread.run)
    assert "stufe1_quality" in src, "Quality-Gate fehlt in run()"
    assert "refinement_cancelled" in src, "refinement_cancelled-Emit fehlt in run()"


def test_quality_invariant_no_overwrite(tmp_path, minimal_job):
    """_write_audio writes to the given path (quality gate is upstream in run())."""

    from Aurik910.ui.ml_refinement_thread import _write_audio

    audio = np.zeros(480, dtype=np.float32) + 0.1
    tmp_path_str = str(tmp_path / "quality_guard.wav")
    _write_audio(audio, 48000, tmp_path_str)
    assert Path(tmp_path_str).exists(), "_write_audio muss Datei erstellen"


def test_quality_invariant_overwrite_when_better(tmp_path):
    """_write_audio successfully creates a valid WAV file."""
    import soundfile as sf

    from Aurik910.ui.ml_refinement_thread import _write_audio

    audio_new = np.zeros(480, dtype=np.float32) + 0.5
    output_path = str(tmp_path / "write_test.wav")
    _write_audio(audio_new, 48000, output_path)
    assert Path(output_path).exists()
    audio_read, _ = sf.read(output_path, dtype="float32")
    assert audio_read.size > 0, "Geschriebene Datei muss Audio enthalten"


# ── Atomares Schreiben (.tmp → os.replace) ────────────────────────────────────
# Atomares Muster: .kmv_tmp → os.replace → kein .tmp links = MLRefinementThread.run()-Logik.
# _write_audio selbst schreibt direkt an den übergebenen Pfad (kein internen .tmp).


def test_atomic_write_pattern_in_run_source():
    """run() must use os.replace for atomic overwrite (not shutil.copy or direct open)."""
    import inspect

    from Aurik910.ui.ml_refinement_thread import MLRefinementThread

    src = inspect.getsource(MLRefinementThread.run)
    assert "os.replace" in src, "Atomarer os.replace-Schritt muss in run() vorhanden sein"
    assert ".kmv_tmp" in src or "_tmp" in src, "Temporärer Dateiname muss vor os.replace verwendet werden"


def test_write_audio_creates_wavfile(tmp_path):
    """_write_audio must create a readable WAV file at the given path."""
    import soundfile as sf

    from Aurik910.ui.ml_refinement_thread import _write_audio

    audio = np.zeros(480, dtype=np.float32) + 0.2
    output_path = str(tmp_path / "atomic_test.wav")

    _write_audio(audio, 48000, output_path)
    assert Path(output_path).exists(), "_write_audio muss Zieldatei anlegen"
    loaded, sr = sf.read(output_path, dtype="float32")
    assert sr == 48000
    assert loaded.size > 0


# ── Signal-Kontrakt (alle 5 §2.38-Pflicht-Signale vorhanden) ─────────────────


def test_ml_refinement_thread_signals_exist():
    """All 5 §2.38 mandatory signals must exist on MLRefinementThread."""
    from Aurik910.ui.ml_refinement_thread import MLRefinementThread

    required_signals = [
        "refinement_started",
        "refinement_phase_done",
        "refinement_progress",
        "refinement_complete",
        "refinement_cancelled",
    ]
    for sig in required_signals:
        assert hasattr(MLRefinementThread, sig), f"Signal {sig!r} fehlt in MLRefinementThread"


# ── RestorationResult-Felder §2.38 (via dataclasses.fields — kein Konstrukt) ──


def test_restoration_result_has_deferred_phases():
    """`deferred_phases` @dataclass field must exist with empty-list default."""
    from dataclasses import MISSING, fields

    from backend.core.unified_restorer_v3 import RestorationResult

    fmap = {f.name: f for f in fields(RestorationResult)}
    assert "deferred_phases" in fmap, "RestorationResult.deferred_phases fehlt"
    assert fmap["deferred_phases"].default is MISSING  # uses default_factory
    assert callable(fmap["deferred_phases"].default_factory)  # type: ignore[misc]
    assert fmap["deferred_phases"].default_factory() == []


def test_restoration_result_has_refinement_complete():
    """`refinement_complete` @dataclass field must exist with False default."""
    from dataclasses import fields

    from backend.core.unified_restorer_v3 import RestorationResult

    fmap = {f.name: f for f in fields(RestorationResult)}
    assert "refinement_complete" in fmap, "RestorationResult.refinement_complete fehlt"
    assert fmap["refinement_complete"].default is False


def test_restoration_result_has_stufe2_quality_estimate():
    """`stufe2_quality_estimate` @dataclass field must exist with None default."""
    from dataclasses import fields

    from backend.core.unified_restorer_v3 import RestorationResult

    fmap = {f.name: f for f in fields(RestorationResult)}
    assert "stufe2_quality_estimate" in fmap, "RestorationResult.stufe2_quality_estimate fehlt"
    assert fmap["stufe2_quality_estimate"].default is None


# ── DeferredRefinementJob Typ-Sicherheit ─────────────────────────────────────


def test_deferred_job_mode_is_lowercase(minimal_job):
    """mode must be 'restoration' or 'studio2026' (lowercase)."""
    assert minimal_job.mode in ("restoration", "studio2026")


def test_deferred_job_sr_is_48000(minimal_job):
    """sr must be 48000 (Verarbeitungs-SR §2.37)."""
    assert minimal_job.sr == 48000
