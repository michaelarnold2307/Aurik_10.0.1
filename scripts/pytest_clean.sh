#!/usr/bin/env bash
set -euo pipefail

# Startet pytest mit stabilem Warning-Filter fuer die bekannte Trio-Excepthook-Meldung.
# Alle Argumente werden unveraendert an pytest weitergereicht.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv_aurik/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Fehler: Python-Interpreter nicht gefunden: ${PYTHON_BIN}" >&2
  exit 1
fi

exec "${PYTHON_BIN}" -W "ignore::RuntimeWarning:trio._core._multierror" -m pytest "$@"
