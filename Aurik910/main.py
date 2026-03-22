#!/usr/bin/env python3
"""
AURIK Professional - Main Application Entry Point
Launch the desktop application for audio restoration
"""

import logging
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Logging-Setup: File + Konsole ─────────────────────────────────────────────
_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_log_file = _LOG_DIR / "aurik_backend.log"

_root_logger = logging.getLogger()
_root_logger.setLevel(logging.DEBUG)

# Datei-Handler (5 MB, Rotation)
from logging.handlers import RotatingFileHandler as _RFH

_fh = _RFH(str(_log_file), maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
_fh.setLevel(logging.INFO)
_fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
_root_logger.addHandler(_fh)

# Konsole-Handler (nur WARNING+)
_ch = logging.StreamHandler(sys.stderr)
_ch.setLevel(logging.WARNING)
_ch.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
_root_logger.addHandler(_ch)
# ──────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox

from Aurik910.ui.modern_window import ModernMainWindow


def _run_startup_model_check(app: QApplication) -> None:
    """Prüft ML-Modelle vor dem Fensteraufbau — zeigt deutschen Dialog bei Problemen.

    Nicht-blockierend bei fehlenden optionalen Modellen.
    Warnung bei fehlenden Primär-Modellen (DSP-Fallback aktiv).
    """
    try:
        from backend.api.bridge import get_startup_check_result  # type: ignore[import]
        result = get_startup_check_result()
        if result is None:
            return
        if not result.all_ok and result.user_message_de:
            icon = QMessageBox.Icon.Critical if result.is_critical else QMessageBox.Icon.Warning
            box = QMessageBox()
            box.setWindowTitle("AURIK — " + result.user_title_de)
            box.setText(result.user_message_de)
            box.setIcon(icon)
            box.setStandardButtons(QMessageBox.StandardButton.Ok)
            box.setDefaultButton(QMessageBox.StandardButton.Ok)
            box.exec_()
    except Exception as exc:
        # Startup-Check darf niemals den App-Start blockieren
        logger.warning("Startup-Modell-Check fehlgeschlagen (non-fatal): %s", exc)


def _warmup_models_background() -> None:
    """Start background warmup of ML models — non-blocking daemon thread."""
    import threading

    def _run() -> None:
        try:
            from backend.api.bridge import warmup_models_background as _wb  # type: ignore[import]
            _wb()
        except Exception as _e:
            logger.debug("Warmup fehlgeschlagen (non-fatal): %s", _e)

    t = threading.Thread(target=_run, daemon=True, name="AurikWarmup")
    t.start()


def main():
    """Launch AURIK Professional"""
    # Enable high DPI scaling (PyQt5-Stubs kennen diese Attribute nicht -> ignore)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)  # type: ignore[attr-defined]
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)  # type: ignore[attr-defined]

    app = QApplication(sys.argv)
    app.setApplicationName("AURIK Professional")
    app.setOrganizationName("AURIK")
    app.setApplicationVersion("9.10.57")

    # Set dark theme style
    app.setStyle("Fusion")
    # Global QToolTip styling — prevents the default black/system-colored tooltip box.
    # Border uses the app's purple accent; background matches the dark UI palette.
    app.setStyleSheet(
        "QToolTip {"
        "  background-color: #1a1a2e;"
        "  color: #d8d8f0;"
        "  border: 1px solid #4a3878;"
        "  border-radius: 6px;"
        "  padding: 6px 10px;"
        "  font-size: 9pt;"
        "}"
    )

    # Create and show main window
    _run_startup_model_check(app)
    window = ModernMainWindow()
    window.show()
    # §9.7.4 Modell-Warmup: ModernMainWindow.__init__ startet via QTimer.singleShot(2000)
    # warmup_models_background() aus backend.api.bridge — kein zweiter Thread hier.
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
