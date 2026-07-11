# Getting Started mit Aurik

> §15.7: 15-Minuten-Setup für neue Nutzer.

## Voraussetzungen

- **Python 3.10+** (3.12 empfohlen)
- **pip** (aktuell)
- **FFmpeg** (für Audio-Format-Konvertierung)
- Optional: AMD-GPU mit ROCm oder CUDA-GPU mit onnxruntime-gpu

## Installation (5 Minuten)

```bash
# 1. Repository klonen
git clone https://github.com/user/aurik.git
cd aurik

# 2. Virtuelle Umgebung erstellen
python3 -m venv .venv_aurik
source .venv_aurik/bin/activate  # Linux/macOS
# ODER: .venv_aurik\Scripts\activate  # Windows

# 3. Aurik installieren
pip install -e ".[dev,test]"

# 4. System-Check
python -m aurik --check
```

## Erste Restaurierung (5 Minuten)

```bash
# Einfache Restaurierung (Quick-Mode)
python -m aurik restore mein_song.wav --mode quick

# Volle Restaurierung mit Material-Erkennung
python -m aurik restore mein_song.wav --mode full

# Batch-Verarbeitung
python -m aurik batch ./input_dir/ --output ./output_dir/
```

## GPU-Beschleunigung prüfen

```bash
# GPU-Fähigkeiten erkennen
python scripts/detect_gpu_capabilities.py

# Falls GPU erkannt wurde:
python -m aurik restore mein_song.wav --gpu
```

## Nächste Schritte

- [Architektur-Übersicht](docs/architecture.md)
- [Tutorial: Vinyl restaurieren](docs/tutorials/tutorial_restore_vinyl.md)
- [Tutorial: Tonband restaurieren](docs/tutorials/tutorial_restore_tape.md)
- [Tutorial: Batch-Verarbeitung](docs/tutorials/tutorial_batch_processing.md)
- [Plugin-Entwicklung](plugins/SDK.md)

## Hilfe

```bash
python -m aurik --help
python -m aurik restore --help
python -m pytest tests/ -x --tb=short  # Tests ausführen
```
