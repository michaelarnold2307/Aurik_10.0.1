"""Normative gates for newly introduced RELEASE_MUST invariants.

Covers:
- §0l [RELEASE_MUST] Per-Phase-Strength-Orakel und 15-Ziele-Teamarbeit
- [RELEASE_MUST] Frontend-Version-Anzeige-Invariante
- [RELEASE_MUST] ROCm-TorchAudio-ABI-Invariante
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
UV3_FILE = ROOT / "backend" / "core" / "unified_restorer_v3.py"
MAIN_FILE = ROOT / "Aurik910" / "main.py"
WINDOW_FILE = ROOT / "Aurik910" / "ui" / "modern_window.py"
SPLASH_FILE = ROOT / "Aurik910" / "ui" / "splash_screen.py"
RUN_SCRIPT = ROOT / "run_aurik.sh"


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_release_must_0l_phase_strength_oracle_and_goal_teamwork_wired() -> None:
    """§0l contract must be represented in runtime wiring tokens."""
    src = UV3_FILE.read_text(encoding="utf-8")

    required_tokens = {
        "phase_strength_oracle_rollout": "UV3 must expose phase-strength-oracle rollout control.",
        "_resolve_phase_strength_oracle_rollout_mode": "UV3 must resolve oracle rollout mode.",
        "_is_phase_strength_oracle_enabled_for_phase": "UV3 must gate oracle activation per phase.",
        "goal_weights": "UV3 must carry song goal weights for team-objective optimization.",
        "effective_goal_targets": "UV3 must carry effective goal targets into runtime context.",
    }

    for token, message in required_tokens.items():
        assert token in src, message


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_release_must_frontend_version_display_invariant_is_wired_to_single_source() -> None:
    """All required frontend version display paths must be present and bound to __version__."""
    main_src = MAIN_FILE.read_text(encoding="utf-8")
    window_src = WINDOW_FILE.read_text(encoding="utf-8")
    splash_src = SPLASH_FILE.read_text(encoding="utf-8")

    assert "from Aurik910 import __version__" in main_src, "Aurik910/main.py must import __version__ from Aurik910."
    assert "setApplicationVersion(__version__)" in main_src, "Aurik910/main.py must set app version from __version__."

    assert "from Aurik910 import __version__ as _AURIK_VERSION" in window_src, (
        "Aurik910/ui/modern_window.py must derive title version from Aurik910.__version__."
    )
    assert 'setWindowTitle(f"AURIK Professional v{_AURIK_VERSION}")' in window_src, (
        "Aurik910/ui/modern_window.py must expose version in window title."
    )

    assert "from Aurik910 import __version__ as _VERSION" in splash_src, (
        "Aurik910/ui/splash_screen.py must import __version__ for splash badge."
    )
    assert 'vt = f"v{_VERSION}"' in splash_src, "Aurik910/ui/splash_screen.py must render visible version badge."


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_release_must_rocm_torchaudio_abi_invariant_is_enforced_in_launcher() -> None:
    """run_aurik.sh must validate and repair torch/torchaudio ROCm ABI before launch."""
    src = RUN_SCRIPT.read_text(encoding="utf-8")

    required_tokens = {
        "check_rocm_torchaudio_abi()": "Launcher must define ROCm ABI preflight check.",
        "import torch": "Preflight must import torch.",
        "import torchaudio": "Preflight must import torchaudio.",
        "ROCM_STACK_ERR build mismatch": "Preflight must detect build-tag mismatch.",
        "repair_rocm_torchaudio()": "Launcher must provide torchaudio repair path.",
        "torchaudio==$torch_version": "Repair must pin torchaudio to exact torch version.",
        "check_rocm_torchaudio_abi": "Launcher must execute ABI preflight before app start.",
        "AURIK_TORCHAUDIO_DEGRADED=1": "torchaudio-only failure must trigger selective degraded mode.",
        "GPU bleibt AKTIV": "torchaudio-only failure must keep GPU active.",
        "Fallback auf CPU-venv": "torch base-stack failure must fallback to CPU launcher.",
    }

    for token, message in required_tokens.items():
        assert token in src, message
