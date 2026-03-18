
# 📚 Aurik 9.x.x — Projektdokumentation

Offizielle Dokumentation von **Aurik 9.10.57** — dem weltweit ersten intelligenten,
kontextbewussten Musik- und Gesangs-Restaurierungs-, Reparatur- und
Rekonstruktions-Denkersystem. Alle Inhalte sind an die KI-Programmierrichtlinien
(`.github/copilot-instructions.md`) ausgerichtet.

**Version:** 9.10.57 | **Phasen:** 56 | **Tests:** 7.747+ | **Materialien:** 17 | **Musical Goals:** 14 | **DefectTypes:** 29

---

## 📖 Quick Navigation

### Für Anwender
- **[Installation Guide](guides/INSTALLATION.md)** – Systemvoraussetzungen & Installation (Linux / Windows)
- **[User Guide](guides/USER_GUIDE.md)** – Vollständiges Benutzerhandbuch
- **[Configuration Guide](guides/CONFIGURATION.md)** – Modi (Restoration / Studio 2026) & Parameter
- **[Troubleshooting Guide](guides/TROUBLESHOOTING.md)** – Problemlösung & FAQ
- **[Quickstart Guide](guides/QUICKSTART_SUPPORT.md)** – Schnelleinstieg

### Für Entwickler
- **[KI-Agent Integration Guide](KI-AGENT-INTEGRATION-GUIDE.md)** – Regeln für KI-Agenten **(Pflicht!)**
- **[KI-Programmierrichtlinien](../.github/copilot-instructions.md)** – Bindende Systemregeln **(Pflicht!)**
- **[Python API Reference](api/PYTHON_API.md)** – API-Dokumentation
- **[Architecture Overview](architecture/ARCHITECTURE.md)** – Systemarchitektur (4 Schichten)
- **[Phases Overview](architecture/PHASES_OVERVIEW.md)** – 56-Phasen-Pipeline (Defect-First)
- **[Pipeline Flow](architecture/PIPELINE_FLOW_ANALYSIS.md)** – Ablauf & Datenfluss
- **[Contributing Guide](development/CONTRIBUTING.md)** – Beitrag leisten
- **[Testing Guide](development/TESTING.md)** – Teststrategie & Best Practices (7.747+ Tests)
- **[Performance Guard Spec](PERFORMANCE_GUARD_SPEC.md)** – 3×-Echtzeit-Budgetregeln

### Status & Fortschritt
- **[Project Status Report](PROJECT_STATUS.md)** – Aktueller Entwicklungsstand v9.10.57
- **[Musical Excellence Analysis](musical_excellence_next_steps.md)** – Qualitätsanalyse & Roadmap
- **[Roadmap](aurik9_roadmap.md)** – Zukunftspläne (Studio 2026+)

---

## 📁 Dokumentationsstruktur

```
docs/
├── guides/                     # Anwender-Guides
│   ├── INSTALLATION.md        # Installation (Linux AppImage & Windows 10/11)
│   ├── CONFIGURATION.md       # Modi (Restoration / Studio 2026), Parameter
│   ├── TROUBLESHOOTING.md     # Problemlösung & FAQ
│   ├── USER_GUIDE.md          # Vollständiges Benutzerhandbuch
│   ├── QUICKSTART_SUPPORT.md  # Schnelleinstieg
│   ├── LOCAL_APP_DEPLOYMENT.md # Desktop-Deployment
│   └── PHONEME_PROCESSING_GUIDE.md  # §2.36 LyricsGuidedEnhancement
│
├── architecture/               # Architektur-Dokumentation
│   ├── ARCHITECTURE.md        # Systemarchitektur (4 Schichten)
│   ├── PHASES_OVERVIEW.md     # 56-Phasen-Pipeline (Defect-First)
│   └── PIPELINE_FLOW_ANALYSIS.md
│
├── api/                        # API-Dokumentation
│   └── PYTHON_API.md          # Python-API-Referenz
│
├── reports/                    # Statusberichte
│   ├── current/               # Aktuelle Berichte (2026)
│   └── phase_completion/      # Phasenabschlussberichte
│
├── development/                # Entwickler-Dokumentation
│   ├── CONTRIBUTING.md        # Beitrag leisten
│   ├── TESTING.md             # Teststrategie (7.747+ Tests)
│   ├── TESTING_BEST_PRACTICES.md
│   ├── DECISION_NO_VST3_PLUGIN.md  # Architektur-Entscheidung
│   └── Packaging_Documentation.md  # AppImage / Windows-Build
│
├── archive/                    # Historische Dokumente (26 Dateien, nicht mehr aktiv)
│   └── README.md              # Archiv-Index
│
├── INDEX.md                    # ⭐ Diese Datei
├── README.md                   # Kurzübersicht → verweist auf INDEX.md
├── KI-AGENT-INTEGRATION-GUIDE.md  # ⚠️ Pflichtlektüre für KI-Agenten
├── PROJECT_STATUS.md          # Aktueller Projektstand v9.10.57
├── AURIK_9.x.x_ARCHITEKTUR.md # Kognitive Pipeline-Übersicht (Detail)
├── DEFECT_SCANNER_SPEC.md     # 29 DefectTypes, 17 MaterialTypes
├── VOCAL_AI_ENHANCEMENT.md    # Vocal-Pipeline §2.8 (Formanten, Breathiness)
├── PERFORMANCE_GUARD_SPEC.md  # 3×-Echtzeit-Budget [RELEASE_MUST]
├── RESOURCE_AWARE_FALLBACK.md # PLM / RAM-Budget
├── UNIFIED_RESTORER_V3_SPEC.md # UV3-Spezifikation
├── MODULAR_PHASES_API.md      # Phasen-API
├── COMPREHENSIVE_METRICS.md   # PQS-Metriken & OQS
├── CI_CD.md                   # CI/CD-Pipeline
├── aurik9_roadmap.md          # Roadmap
├── musical_excellence_next_steps.md  # Qualitätsziele 2026
├── natural_sound_improvement_analysis.md
└── tier2_ml_hybrid_analysis.md
```

---

## 🧠 Normkonformität & KI-Richtlinien

**Bindende Regeln:** `.github/copilot-instructions.md`

Schlüsselbindungen (Auszug):
- Interne SR: immer **48 000 Hz** — vor und nach jedem DSP-Schritt
- **CPU-only**: keine GPU/CUDA — `providers=["CPUExecutionProvider"]`
- **14 Musical Goals**: nach jeder Restaurierung zu prüfen, Regression = Feature ungültig
- **56 Phasen** (Phase 01–56, Defect-First) in `backend/core/phases/`
- **17 MaterialTypes** — auto-erkannt durch `MediumClassifier`
- **29 DefectTypes** — vollständiger Defektkatalog in `core/defect_scanner.py`
- **Desktop-only (Linux AppImage + Windows 10/11)**: keine Cloud, kein Docker, kein pip für Endnutzer
- **§2.36 LyricsGuidedEnhancement**: Whisper-Tiny ONNX + wav2vec2 Forced Alignment
- **try_allocate() Pflicht** vor jedem ML-Modell-Laden (PluginLifecycleManager)
- **AMRB-Benchmark**: Aurik ≥ iZotope RX 11 in ≥ 7/10 Szenarien [RELEASE_MUST]

---

### Erste Schritte
1. [Installation Guide](guides/INSTALLATION.md)
2. [User Guide](guides/USER_GUIDE.md)
3. [Configuration Guide](guides/CONFIGURATION.md)
4. [Troubleshooting Guide](guides/TROUBLESHOOTING.md)

### KI-Agenten (Pflicht)
- [KI-Agent Integration Guide](KI-AGENT-INTEGRATION-GUIDE.md)
- [KI-Programmierrichtlinien](../.github/copilot-instructions.md)

### Architektur & Entwicklung
- [Python API Reference](api/PYTHON_API.md)
- [Contributing Guide](development/CONTRIBUTING.md)
- [Testing Guide](development/TESTING.md)
- [Phases Overview](architecture/PHASES_OVERVIEW.md)
- [Pipeline Flow Analysis](architecture/PIPELINE_FLOW_ANALYSIS.md)
- [Architecture Overview](architecture/ARCHITECTURE.md)

### Projekt-Status
- [Project Status Report](PROJECT_STATUS.md)
- [Musical Excellence Analysis](musical_excellence_next_steps.md)
- [Roadmap](aurik9_roadmap.md)

---

## 📅 Aktuelle Updates

**März 2026 (v9.8.0):**
- Über-SOTA DSP-Algorithmen implementiert: OMLSA/IMCRA, pYIN, NMF-β, PGHI
- Dokumentation vollständig auf v9.8.0 ausgerichtet, 222 Tests grün

**19. Februar 2026 (v9.7.0):**
- Dokumentation vollständig auf v9.7.0 ausgerichtet
- README.md, INDEX.md, PROJECT_STATUS.md, KI-AGENT-INTEGRATION-GUIDE.md aktualisiert
- 55 Phasen (nicht 42), 12 Materialien (nicht 5), 206 Tests (nicht 9)
- 7 Musical Goals mit Pflicht-Schwellwerten dokumentiert
- Magic-Button-Implementierung (restoration.png / studio.png) dokumentiert

**17. Februar 2026:**
- v9.7.0: Kognitive Architektur (PerceptualEmbedder, CausalDefectReasoner,
  GPParameterOptimizer, PerceptualQualityScorer, MusicalGoalsChecker)
- 40 neue Tests (Gesamt: 206)

---

<div align="center">

**Aurik 9.8.0 Projektdokumentation** | Letzte Aktualisierung: März 2026

[🏠 Zurück zur README](../README.md)

</div>
