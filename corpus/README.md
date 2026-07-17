# Echt-Audio-Test-Corpus — Public-Domain-Aufnahmen für Pipeline-Validierung

> §15.2: Auriks Pipeline muss an echtem Audiomaterial validiert werden.
> Alle Aufnahmen in diesem Corpus sind Public Domain oder CC0-lizenziert.

## Verzeichnisstruktur

```
corpus/
├── README.md                ← Diese Datei
├── MANIFEST_SCHEMA.yaml     ← Schema für manifest.yaml
├── shellac/
│   ├── clean/               ← Referenz-Aufnahmen ohne Defekte
│   ├── damaged/             ← Aufnahmen mit typischen Schellack-Defekten
│   └── restored/            ← Von Aurik restaurierte Versionen
├── vinyl/
│   ├── clean/
│   ├── damaged/
│   └── restored/
├── tape/
│   ├── clean/
│   ├── damaged/
│   └── restored/
└── digital/
    ├── clean/
    ├── damaged/
    └── restored/
```

## manifest.yaml (pro Verzeichnis)

Jedes Unterverzeichnis enthält eine `manifest.yaml` mit Metadaten:

```yaml
# Beispiel: corpus/shellac/damaged/manifest.yaml
- file: jazz_78_shellac_1938.wav
  duration_s: 24.5
  sample_rate: 48000
  channels: 1
  material: shellac
  era: 1938
  genre: Jazz
  defects:
    - clicks
    - surface_noise
    - rumble
  source: "Internet Archive — 78rpm Collection"
  source_url: "https://archive.org/details/..."
  license: "Public Domain"
  license_url: "https://creativecommons.org/publicdomain/mark/1.0/"
  checksum_sha256: "abc123..."
  notes: "Transfer von Victor 78rpm, 1938"
```

## Quellen für Public-Domain-Aufnahmen

| Quelle | URL | Typ | Lizenz |
|--------|-----|-----|--------|
| Internet Archive 78rpm | <https://archive.org/details/78rpm> | Schellack, Vinyl | Public Domain |
| Musopen | <https://musopen.org> | Klassik, alle Materialien | Public Domain / CC0 |
| Freesound (CC0) | <https://freesound.org> | Effekte, kurze Clips | CC0 |
| IASA Training Collection | <https://www.iasa-web.org> | Alle historischen Medien | Educational Use |
| Europeana Sounds | <https://www.europeana.eu> | Historische Aufnahmen | Varies (PD/CC) |

## Aufnahmen hinzufügen

```bash
# 1. Audio-Datei ins passende Verzeichnis kopieren
cp meine_aufnahme.wav corpus/shellac/damaged/

# 2. manifest.yaml aktualisieren
# (Feld für Feld ausfüllen, siehe MANIFEST_SCHEMA.yaml)

# 3. Integrität prüfen
python -m pytest tests/corpus/test_corpus_integrity.py -v

# 4. Pipeline-Smoke-Test
python -m pytest tests/corpus/test_corpus_pipeline_smoke.py -v
```

## Rechtlicher Hinweis

⚠️  **Alle Aufnahmen in diesem Corpus MÜSSEN Public Domain oder CC0 sein.**
Kein urheberrechtlich geschütztes Material. Keine Fair-Use-Argumentation.
Bei Unsicherheit: Nicht hinzufügen.

## Minimalanforderungen (§15.2)

- [ ] ≥ 20 Aufnahmen insgesamt
- [ ] ≥ 4 Material-Kategorien (shellac, vinyl, tape, digital)
- [ ] Alle manifest.yaml-Einträge vollständig
- [ ] Alle Checksummen verifiziert
- [ ] `test_corpus_integrity` grün
- [ ] `test_corpus_pipeline_smoke` grün
