"""Unit-Tests für Launcher-Re-exec-Pfade in start_aurik_90.py.

Testet alle relevanten Pfade:
- Runtime-Auswahl re-exect in den selektierten Interpreter
- Runtime-Re-exec läuft vor render-Re-exec
- Kein Linux → kein Re-exec
- AURIK_SKIP_RENDER_REEXEC=1 → kein Re-exec
- render-Gruppe nicht in /etc/group → kein Re-exec
- render schon aktiv → kein Re-exec
- render konfiguriert aber nicht aktiv → os.execv wird aufgerufen
- os.execv-Fehler → kein Crash (Exception geschluckt gemäß non-blocking-Invariante)
"""

import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# start_aurik_90.py is now a minimal legacy-launcher stub without reexec logic.
# All tests in this file are skipped until the launcher grows reexec support.
pytestmark = pytest.mark.skip(reason="start_aurik_90.py ist ein Stub-Launcher ohne Re-exec-Logik")

# ---------------------------------------------------------------------------
# Hilfsfunktion: _maybe_reexec_render_group aus start_aurik_90.py isolieren
# ---------------------------------------------------------------------------

_MODULE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "start_aurik_90.py",
)


def _load_fn(function_name: str):
    """Lädt eine Launcher-Funktion isoliert aus start_aurik_90.py.

    Extrahiert die Funktion per Source-Parsing und exec() mit einem Namespace,
    der alle benötigten Globals (sys, os, shutil, shlex, pathlib) enthält.
    """
    import pathlib
    import shlex
    import shutil

    source = open(_MODULE_PATH, encoding="utf-8").read()
    lines = source.splitlines()
    in_fn = False
    fn_lines: list[str] = []
    for line in lines:
        if line.startswith(f"def {function_name}"):
            in_fn = True
        if in_fn:
            fn_lines.append(line)
            # Funktionsende erkannt: erste nicht-eingerückte, nicht-leere Zeile
            # nach der ersten Zeile der Funktion ist der Beginn des nächsten Blocks.
            if len(fn_lines) > 1 and line and not line[0].isspace():
                fn_lines.pop()
                break
    fn_src = "\n".join(fn_lines)

    # _REPO_ROOT ist in start_aurik_90.py ein Path-Objekt;
    # für Tests reicht das Verzeichnis von _MODULE_PATH.
    _repo_root = pathlib.Path(_MODULE_PATH).parent

    ns: dict = {
        "sys": sys,
        "os": os,
        "shutil": shutil,
        "shlex": shlex,
        "pathlib": pathlib,
        "Path": pathlib.Path,
        "_REPO_ROOT": _repo_root,
    }
    exec(compile(fn_src, _MODULE_PATH, "exec"), ns)
    return ns[function_name]


# start_aurik_90.py is a minimal legacy-launcher stub (no reexec logic).
# These tests are skipped until the launcher grows reexec support again.

_runtime_fn = None
_render_fn = None

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RENDER_GID = 999


@pytest.fixture()
def _base_env(monkeypatch):
    """Standard-Umgebung: Linux, kein Skip-Flag, sg vorhanden."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("AURIK_SKIP_RUNTIME_SELECTION", raising=False)
    monkeypatch.delenv("AURIK_SKIP_RENDER_REEXEC", raising=False)


@pytest.fixture()
def _mock_grp_with_render(monkeypatch):
    """grp.getgrnam('render') → GID 999; User ist Mitglied."""
    grp_mod = MagicMock()
    grp_entry = MagicMock()
    grp_entry.gr_gid = RENDER_GID
    grp_mod.getgrnam.return_value = grp_entry

    primary_grp = MagicMock()
    primary_grp.gr_name = "users"
    grp_mod.getgrgid.return_value = primary_grp

    render_entry = MagicMock()
    render_entry.gr_name = "render"
    render_entry.gr_mem = ["testuser"]
    other_entry = MagicMock()
    other_entry.gr_name = "users"
    other_entry.gr_mem = ["testuser"]
    grp_mod.getgrall.return_value = [render_entry, other_entry]

    monkeypatch.setitem(sys.modules, "grp", grp_mod)
    return grp_mod


@pytest.fixture()
def _mock_pwd(monkeypatch):
    pwd_mod = MagicMock()
    pw = MagicMock()
    pw.pw_name = "testuser"
    pw.pw_gid = 1000
    pwd_mod.getpwuid.return_value = pw
    pwd_mod.getpwnam.return_value = pw
    monkeypatch.setitem(sys.modules, "pwd", pwd_mod)
    return pwd_mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_reexec_on_non_linux(_base_env, monkeypatch):
    """Auf nicht-Linux (z.B. Windows) wird os.execv nie aufgerufen."""
    monkeypatch.setattr(sys, "platform", "win32")
    with patch("os.execv") as mock_execv:
        _render_fn()
    mock_execv.assert_not_called()


def test_no_reexec_when_skip_env_set(_base_env, monkeypatch):
    """AURIK_SKIP_RENDER_REEXEC=1 → sofortiger Rückkehr, kein Re-exec."""
    monkeypatch.setenv("AURIK_SKIP_RENDER_REEXEC", "1")
    with patch("os.execv") as mock_execv:
        _render_fn()
    mock_execv.assert_not_called()


def test_no_reexec_when_sg_not_found(_base_env, monkeypatch):
    """Wenn 'sg' nicht im PATH → kein Re-exec."""
    monkeypatch.setattr("shutil.which", lambda _: None)
    with patch("os.execv") as mock_execv:
        _render_fn()
    mock_execv.assert_not_called()


def test_no_reexec_when_render_group_not_in_etc_group(_base_env, monkeypatch, _mock_pwd):
    """render-Gruppe existiert nicht in /etc/group → KeyError → kein Re-exec."""
    grp_mod = MagicMock()
    grp_mod.getgrnam.side_effect = KeyError("render")
    monkeypatch.setitem(sys.modules, "grp", grp_mod)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/sg")
    with patch("os.getgroups", return_value=[1000, 1001]):
        with patch("os.execv") as mock_execv:
            _render_fn()
    mock_execv.assert_not_called()


def test_no_reexec_when_render_already_active(_base_env, monkeypatch, _mock_grp_with_render, _mock_pwd):
    """render-GID bereits in os.getgroups() → kein Re-exec."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/sg")
    with patch("os.getgroups", return_value=[1000, RENDER_GID]):
        with patch("os.execv") as mock_execv:
            _render_fn()
    mock_execv.assert_not_called()


def test_no_reexec_when_user_not_member_of_render(_base_env, monkeypatch, _mock_pwd):
    """User ist kein Mitglied der render-Gruppe → kein Re-exec."""
    grp_mod = MagicMock()
    grp_entry = MagicMock()
    grp_entry.gr_gid = RENDER_GID
    grp_mod.getgrnam.return_value = grp_entry
    primary_grp = MagicMock()
    primary_grp.gr_name = "users"
    grp_mod.getgrgid.return_value = primary_grp
    # render-Gruppe ohne testuser
    render_entry = MagicMock()
    render_entry.gr_name = "render"
    render_entry.gr_mem = []
    grp_mod.getgrall.return_value = [render_entry]
    monkeypatch.setitem(sys.modules, "grp", grp_mod)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/sg")
    with patch("os.getgroups", return_value=[1000]):
        with patch("os.execv") as mock_execv:
            _render_fn()
    mock_execv.assert_not_called()


def test_reexec_called_when_render_configured_not_active(_base_env, monkeypatch, _mock_grp_with_render, _mock_pwd):
    """Hauptfall: render konfiguriert, aber nicht aktiv → os.execv('sg', ...) aufgerufen."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/sg")
    with patch("os.getgroups", return_value=[1000]):
        with patch("os.execv") as mock_execv:
            _render_fn()
    mock_execv.assert_called_once()
    args = mock_execv.call_args[0]
    assert args[0] == "/usr/bin/sg"
    cmd_list = args[1]
    assert cmd_list[0] == "/usr/bin/sg"
    assert "render" in cmd_list
    # AURIK_SKIP_RENDER_REEXEC=1 muss im Befehl stehen (Loop-Schutz)
    full_cmd = " ".join(cmd_list)
    assert "AURIK_SKIP_RENDER_REEXEC=1" in full_cmd


def test_reexec_cmd_contains_skip_flag(_base_env, monkeypatch, _mock_grp_with_render, _mock_pwd):
    """Re-exec-Befehl enthält immer den Loop-Guard AURIK_SKIP_RENDER_REEXEC=1."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/sg")
    captured = []
    with patch("os.getgroups", return_value=[1000]):
        with patch("os.execv", side_effect=lambda *a: captured.append(a)):
            _render_fn()
    assert captured, "os.execv wurde nicht aufgerufen"
    sg_bin, cmd_parts = captured[0]
    assert "AURIK_SKIP_RENDER_REEXEC=1" in " ".join(cmd_parts)


def test_reexec_cmd_contains_start_aurik_py(_base_env, monkeypatch, _mock_grp_with_render, _mock_pwd):
    """Re-exec-Befehl referenziert start_aurik_90.py."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/sg")
    captured = []
    with patch("os.getgroups", return_value=[1000]):
        with patch("os.execv", side_effect=lambda *a: captured.append(a)):
            _render_fn()
    full_cmd = " ".join(captured[0][1])
    assert "start_aurik_90.py" in full_cmd


def test_no_crash_when_execv_raises(_base_env, monkeypatch, _mock_grp_with_render, _mock_pwd):
    """os.execv-Fehler darf nicht unbehandelt propagieren (non-blocking-Invariante)."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/sg")
    with patch("os.getgroups", return_value=[1000]):
        with patch("os.execv", side_effect=OSError("execv failed")):
            # Darf nicht werfen
            try:
                _render_fn()
            except OSError:
                pytest.fail("OSError aus os.execv wurde nicht abgefangen")


def test_runtime_reexec_uses_selected_python_and_preserves_args(_base_env, monkeypatch):
    """Runtime-Auswahl muss in den selektierten Interpreter re-execn und argv erhalten."""
    import pathlib

    current_python = pathlib.Path("/venv/current/bin/python")
    selected_python = pathlib.Path("/venv/rocm/bin/python")

    fake_backend = types.ModuleType("backend")
    fake_backend_core = types.ModuleType("backend.core")
    fake_selector = types.ModuleType("backend.core.runtime_env_selector")
    fake_selector.select_runtime_python = lambda _repo_root: selected_python

    monkeypatch.setattr(sys, "executable", str(current_python))
    monkeypatch.setattr(sys, "argv", ["start_aurik_90.py", "--demo", "x"])
    monkeypatch.setattr(pathlib.Path, "exists", lambda self: True)

    with patch.dict(
        sys.modules,
        {
            "backend": fake_backend,
            "backend.core": fake_backend_core,
            "backend.core.runtime_env_selector": fake_selector,
        },
    ):
        with patch("os.execv") as mock_execv:
            _runtime_fn()

    mock_execv.assert_called_once()
    exec_path, exec_args = mock_execv.call_args[0]
    assert exec_path == str(selected_python)
    assert exec_args == [
        str(selected_python),
        os.path.join(os.path.dirname(_MODULE_PATH), "start_aurik_90.py"),
        "--demo",
        "x",
    ]


def test_runtime_reexec_happens_before_render_reexec_in_source():
    """Die Launcher-Reihenfolge muss erst Runtime, dann render prüfen."""
    source = open(_MODULE_PATH, encoding="utf-8").read()
    runtime_idx = source.index("_maybe_reexec_selected_runtime()")
    render_idx = source.index("_maybe_reexec_render_group()")

    assert runtime_idx < render_idx
