"""PhaseProgressNarrator — Klare, für jeden verständliche Fortschrittsmeldungen.

Jede Meldung erklärt in einfacher Sprache, was Aurik gerade tut und warum.
Kein Fachchinesisch — so verständlich, dass es jeder Mensch sofort begreift.

Prinzipien:
  - Alltagssprache, keine Fachbegriffe aus Programmierung oder Tontechnik
  - Konkrete Vergleiche aus der Alltagswelt (wie ein Restaurator, wie ein Archäologe)
  - Jede Aktion wird mit ihrem Zweck erklärt (was + warum)
  - 10 Sekunden Mindesteinblenddauer — Zeit zum Lesen
  - Persönliche Ansprache, als würde ein Freund erklären
"""
from __future__ import annotations

import hashlib
import time as _time
from typing import Any

# ═══════════════════════════════════════════════════════════════════════════════
# Tonträger-Namen — für jeden verständlich
# ═══════════════════════════════════════════════════════════════════════════════
_TRÄGER_NAMEN: dict[str, str] = {
    "vinyl": "einer Vinyl-Schallplatte",
    "shellac": "einer alten Schellackplatte",
    "reel_tape": "einem Tonband (grosse Spulen)",
    "cassette_tape": "einer Musikkassette",
    "cd": "einer CD",
    "mp3_low": "einer MP3-Datei (stark verkleinert)",
    "mp3_high": "einer MP3-Datei",
    "aac": "einer AAC-Datei",
    "streaming": "einem Streaming-Dienst",
    "digital": "einer digitalen Datei",
    "unknown": "einem unbekannten Träger",
}
_TRÄGER_KURZ: dict[str, str] = {
    "vinyl": "Vinyl-Platte", "shellac": "Schellackplatte",
    "reel_tape": "Tonband", "cassette_tape": "Kassette",
    "cd": "CD", "mp3_low": "MP3", "mp3_high": "MP3",
    "aac": "AAC", "streaming": "Stream", "digital": "Digital",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Tonträgerketten-Erzählung — wie Aurik herausfindet, woher die Musik stammt
# ═══════════════════════════════════════════════════════════════════════════════
_KETTEN_ERZAEHLUNG: list[str] = [
    "Aurik hat genau hingehört und erkannt, dass Deine Musik von "
    "{chain} stammt. Jede Art von Musikträger — ob Platte, Band oder "
    "Datei — hinterlässt nämlich ganz bestimmte Spuren im Klang. "
    "So wie ein Detektiv anhand von Fingerabdrücken weiss, wer am Tatort war, "
    "erkennt Aurik an diesen Spuren, woher Deine Musik kommt.",

    "Deine Musik hat eine kleine Reise hinter sich: {chain}. "
    "Das sind {num_stages} verschiedene Stationen. Aurik hat das herausgefunden, "
    "indem es das Klangbild ganz genau untersucht hat. Jede dieser Stationen "
    "hat ihre eigenen, typischen Merkmale — ähnlich wie verschiedene "
    "Fotos desselben Motivs je nach Kamera anders aussehen.",

    "Spannend: Deine Aufnahme war ursprünglich {first_stage} "
    "und hat im Lauf der Zeit {num_stages} weitere Stationen durchlaufen. "
    "Das ist wie bei einem alten Foto, das mehrfach kopiert wurde — "
    "jede Kopie verliert ein bisschen an Schärfe. Aurik weiss genau, "
    "wie es diese Verluste wieder ausgleichen kann.",

    "Aurik vergleicht Deine Musik mit 76 verschiedenen Mustern, "
    "die typisch für verschiedene Musikträger sind. Dabei kam heraus: "
    "{chain}. So wie ein erfahrener Uhrmacher am Klang erkennt, "
    "welches Uhrwerk tickt, so erkennt Aurik, woher Deine Musik stammt.",

    "Die Untersuchung zeigt: {chain}. Aurik hat dafür über 60 verschiedene "
    "Merkmale Deiner Aufnahme geprüft — vom leisesten Rauschen bis zur "
    "höchsten Höhe. Das Ergebnis: ein klares Bild davon, welchen Weg "
    "Deine Musik genommen hat, bevor sie zu Dir kam.",

    "Deine Musik begann ihr Leben als {first_stage}. Seitdem hat sie "
    "eine spannende Reise hinter sich. Aurik weiss jetzt genau, "
    "wie es sie am besten behandelt — so wie ein Restaurator weiss, "
    "ob ein Gemälde auf Leinwand oder Holz gemalt wurde.",

    "Wie erkennt Aurik eigentlich, woher Deine Musik kommt? "
    "Ganz einfach: Es hört sich die leisesten Geräusche an — "
    "das feine Knistern einer Platte, das sanfte Rauschen eines Bandes, "
    "die typischen Verluste einer komprimierten Datei. "
    "Jeder dieser Klänge verrät Aurik etwas über die Herkunft.",

    "Die Spurensuche ergab: {chain}. Stell Dir das vor wie eine "
    "Geschichte, die Deine Musik erzählt — von ihrer Geburt als "
    "{first_stage} bis zu dem Moment, als sie bei Dir ankam. "
    "Aurik hat diese Geschichte gelesen und verstanden.",
]

_WARUM_KETTE_WICHTIG: list[str] = [
    "Warum ist das wichtig? Weil Aurik jede Art von Musik anders "
    "behandeln muss. Eine Schallplatte hat andere Probleme als eine "
    "Kassette — und eine MP3-Datei wieder ganz andere. Wenn Aurik "
    "weiss, woher Deine Musik kommt, kann es die richtigen Werkzeuge "
    "auswählen. So wie ein Arzt eine andere Behandlung braucht als ein "
    "Mechaniker — Aurik braucht für Vinyl andere Methoden als für MP3.",

    "Weil Aurik jetzt genau weiss, dass Deine Musik von {chain} stammt, "
    "kann es jeden einzelnen Arbeitsschritt perfekt darauf abstimmen. "
    "So wie ein Koch weiss, ob ein Steak medium oder well-done sein soll — "
    "Aurik weiss jetzt, ob es sanft oder kräftig vorgehen muss.",
]

# ═══════════════════════════════════════════════════════════════════════════════
# Phase-Erklärungen: WARUM dieser Schritt für DIESEN Song
# ═══════════════════════════════════════════════════════════════════════════════
_WARUM_DIESE_PHASE: dict[str, dict[str, list[str]]] = {
    "phase_01": {
        "_default": [
            "Deine Musik hat kleine Knackser und Klicks — das sind winzige "
            "Störungen, die wie ein kurzes Knacksen klingen. Aurik entfernt "
            "sie jetzt ganz gezielt, ohne den Rest der Musik anzutasten.",
        ],
        "vinyl": [
            "Deine Vinyl-Platte hat die typischen Knackser und Klicks, "
            "die beim Abspielen mit der Nadel entstehen. Aurik sucht jetzt "
            "jeden einzelnen dieser Störer und entfernt ihn — "
            "so vorsichtig, dass die Musik unberührt bleibt.",
            "Die Nadel einer Schallplatte hinterlässt Spuren: winzige Knackser. "
            "Aurik ist darin geübt, sie zu finden und zu beseitigen — "
            "wie ein Restaurator, der Staub von einem Gemälde pustet.",
        ],
    },
    "phase_03": {
        "_default": [
            "Jetzt geht es ans Eingemachte: Aurik entfernt das Grundrauschen. "
            "Das ist wie das leise Hintergrundgeräusch, das man bei alten "
            "Aufnahmen oft hört. Aurik trennt es sauber von der Musik.",
        ],
        "vinyl": [
            "Deine Vinyl-Platte hat ein ganz eigenes, feines Rauschen — "
            "jede Platte klingt da etwas anders. Aurik erkennt dieses "
            "Rauschen und zieht es behutsam aus der Musik heraus.",
        ],
        "reel_tape": [
            "Tonbänder haben ein charakteristisches Rauschen — ein sanftes "
            "Zischen, das im Hintergrund mitschwingt. Aurik entfernt es, "
            "ohne die Wärme des Bandklangs zu verlieren.",
        ],
        "cassette_tape": [
            "Kassetten rauschen von Natur aus stärker als andere Tonträger. "
            "Aurik geht hier besonders geschickt vor — es holt das Rauschen "
            "raus, aber lässt die Musik schön klar klingen.",
        ],
    },
    "phase_05": {
        "_default": [
            "Ganz tiefe Töne, die man kaum hört, können trotzdem stören — "
            "zum Beispiel das Rumpeln eines Plattentellers. Aurik filtert "
            "diese tiefen Störungen jetzt heraus.",
        ],
    },
    "phase_06": {
        "_default": [
            "Mit der Zeit verliert Musik oft ihre hohen Töne — die Brillanz "
            "geht verloren. Aurik stellt diese Höhen wieder her, und zwar "
            "so, als wären sie nie verschwunden gewesen.",
        ],
        "mp3_low": [
            "MP3-Dateien werfen beim Verkleinern viele hohe Töne einfach weg. "
            "Das spart Platz, kostet aber Klang. Aurik holt diese verlorenen "
            "Höhen jetzt Stück für Stück zurück.",
        ],
    },
    "phase_09": {
        "_default": [
            "Das feine Knistern auf der Oberfläche — bei {material} "
            "ganz normal — wird jetzt entfernt. Aurik geht dabei so "
            "behutsam vor, dass die Musik nicht stumpf oder glatt klingt.",
        ],
    },
    "phase_12": {
        "_default": [
            "Bei Bandaufnahmen kommt es manchmal vor, dass die Tonhöhe "
            "leicht schwankt — wie bei einem Plattenspieler, der nicht "
            "ganz rund läuft. Aurik gleicht das jetzt aus, sodass alles "
            "wieder stabil und sauber klingt.",
        ],
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# Phase-Aktivitäten — was Aurik konkret tut (Alltagssprache)
# ═══════════════════════════════════════════════════════════════════════════════
_AKTIVITAETEN: dict[str, list[str]] = {
    "phase_01": [
        "Entfernt Knackser — so präzise wie ein Chirurg …",
        "Jeder Klick wird geortet und sanft entfernt …",
        "Kleine Störer verschwinden, die Musik bleibt …",
        "Was geknackst hat, klingt gleich sauber …",
        "Die Oberfläche wird von Knacksern befreit …",
        "Knackser für Knackser — es wird immer sauberer …",
        "Kurzzeit-Störungen werden entfernt …",
    ],
    "phase_02": [
        "Entfernt das tiefe Brummen aus dem Hintergrund …",
        "Das störende Netzbrummen verschwindet …",
        "Ein sauberer Klang entsteht — ohne Brummen …",
        "Die 50-Hz-Störung wird beseitigt …",
    ],
    "phase_03": [
        "Das Grundrauschen weicht — die Musik atmet auf …",
        "Rauschen und Musik werden voneinander getrennt …",
        "Schicht für Schicht geht das Rauschen weg …",
        "Wie ein Archäologe, der eine Vase ausgräbt — "
        "behutsam wird die Musik freigelegt …",
        "Die Details der Aufnahme kommen langsam zum Vorschein …",
        "Was vorher im Rauschen unterging, wird hörbar …",
        "Aurik arbeitet sich durch das Rauschen — "
        "wie ein Taucher, der zum Meeresgrund hinabsteigt …",
        "Langsam wird es stiller im Hintergrund …",
    ],
    "phase_04": [
        "Bringt Bässe, Mitten und Höhen ins Gleichgewicht …",
        "Die Klangfarbe wird ausbalanciert — nichts dröhnt, "
        "nichts fehlt …",
        "Passt den Klang an wie ein Optiker eine Brille — "
        "so lange, bis alles perfekt scharf ist …",
        "Was zu dumpf klang, wird klarer. Was zu scharf war, "
        "wird weicher …",
        "Die richtige Balance für Deine Musik …",
    ],
    "phase_05": [
        "Entfernt tiefes Rumpeln, das man kaum hört …",
        "Das Wummern des Plattentellers verschwindet …",
        "Alles, was nur die Boxen vibrieren lässt, "
        "wird entfernt …",
    ],
    "phase_06": [
        "Stellt verlorene Höhen wieder her …",
        "Die Brillanz kehrt zurück …",
        "Was dumpf war, bekommt wieder Glanz …",
        "Fehlende Klangfarben werden ergänzt …",
        "Die Musik wird wieder luftig und offen …",
        "Wie beim Öffnen eines Fensters — frische Luft "
        "für Deine Musik …",
    ],
    "phase_07": [
        "Bringt Wärme und Fülle zurück …",
        "Die feinen Klangfarben jedes Tons leben wieder auf …",
        "Die Musik klingt wieder natürlich und voll …",
        "Obertöne, die verloren gingen, werden wieder hörbar …",
    ],
    "phase_08": [
        "Schützt die knackigen Anfänge jedes Tons …",
        "Trommeln und Gitarrenanschläge bleiben lebendig …",
        "Nichts klingt verwaschen — alles bleibt präzise …",
    ],
    "phase_09": [
        "Das Knistern der Oberfläche wird entfernt …",
        "Die Plattenoberfläche klingt gleich viel sauberer …",
        "Feinste Störgeräusche verschwinden …",
    ],
    "phase_12": [
        "Gleicht leichte Tonhöhen-Schwankungen aus …",
        "Der Ton bekommt wieder sicheren Stand …",
        "Die Musik schwebt nicht mehr — alles sitzt fest …",
        "Stabile Tonhöhe von Anfang bis Ende …",
    ],
    "phase_13": [
        "Verbessert das räumliche Klangbild …",
        "Links und rechts werden perfekt ausbalanciert …",
        "Die Musik bekommt Tiefe und Raum …",
    ],
    "phase_17": [
        "Der letzte Feinschliff — wie beim Polieren eines Edelsteins …",
        "Alles wird noch einmal verfeinert und abgerundet …",
        "Die finale Politur für den perfekten Klang …",
    ],
    "phase_18": [
        "Befreit die stillen Momente vom Rauschen …",
        "Zwischen den Tönen herrscht jetzt Ruhe …",
        "Die Pausen in der Musik werden wirklich still …",
    ],
    "phase_19": [
        "Macht scharfe Zischlaute angenehmer …",
        "Die S-Laute werden weicher, ohne dumpf zu werden …",
        "Angenehmer zu hören — weniger scharf …",
    ],
    "phase_20": [
        "Verringert unerwünschten Hall …",
        "Die Musik klingt direkter und näher …",
        "Weniger Nachhall — mehr Klarheit …",
    ],
    "phase_23": [
        "Repariert beschädigte Stellen im Klang …",
        "Lücken werden geschlossen — nichts fehlt mehr …",
        "Was beschädigt war, wird wieder ganz …",
    ],
    "phase_24": [
        "Füllt kurze Aussetzer im Ton …",
        "Wenn der Ton mal weg war, holt Aurik ihn zurück …",
        "Fehlende Momente werden rekonstruiert …",
    ],
    "phase_29": [
        "Das Bandrauschen wird leiser …",
        "Das Zischen des Bandes tritt in den Hintergrund …",
        "Mehr Musik, weniger Rauschen …",
    ],
    "phase_31": [
        "Korrigiert die Geschwindigkeit …",
        "Alles wieder im richtigen Tempo …",
        "Zu schnell oder zu langsam? Jetzt stimmt's …",
    ],
    "phase_40": [
        "Stellt die ideale Lautstärke ein …",
        "Weder zu leise noch zu laut — genau richtig …",
        "Deine Musik klingt auf jedem Gerät optimal …",
    ],
    "phase_42": [
        "Verbessert die Klarheit der Stimme …",
        "Jedes Wort wird deutlicher und präsenter …",
        "Die Stimme steht jetzt im besten Licht …",
    ],
}

_ALLGEMEINE_AKTIVITAETEN: list[str] = [
    "Arbeitet mit höchster Sorgfalt an Deiner Musik …",
    "Jeder Rechenschritt bringt besseren Klang …",
    "Gute Restaurierung braucht ein wenig Zeit — "
    "und die geben wir ihr …",
    "Deine Musik verdient diese Aufmerksamkeit …",
    "Im Hintergrund laufen komplexe Berechnungen — "
    "alles für Deine Musik …",
    "Qualität vor Geschwindigkeit — immer …",
    "So sorgfältig wie ein Uhrmacher …",
    "Deine Musik in guten Händen …",
]


class PhaseProgressNarrator:
    """Erklärt jeden Schritt — für jeden verständlich."""

    def __init__(self) -> None:
        self._session_key = hashlib.md5(str(id(self)).encode()).hexdigest()[:6]
        self._used: dict[str, list[int]] = {}
        self._last_ts: dict[str, float] = {}
        self._rotate_every_s: float = 10.0
        self._context: dict[str, Any] = {}
        self._chain_story_told: bool = False
        self._chain_story_index: int = 0

    # ── Kontext setzen ──────────────────────────────────────────────────────

    def set_context(
        self, *, material: str = "", era_decade: int | None = None,
        transfer_chain: list[str] | None = None,
        defects: list[str] | None = None,
        restorability: float | None = None,
    ) -> None:
        self._context = {
            "material": str(material or "").lower(),
            "era_decade": era_decade,
            "transfer_chain": list(transfer_chain or []),
            "defects": list(defects or []),
            "restorability": restorability,
        }
        self._chain_story_told = False
        self._chain_story_index = 0

    # ── Tonträgerketten-Erzählung ──────────────────────────────────────────

    def chain_narrative(self) -> str:
        """Eine verständliche Erzählung, wie Aurik die Herkunft der Musik erkannt hat."""
        ctx = self._context
        chain = ctx.get("transfer_chain") or []
        if not chain:
            return ""

        kurz = [_TRÄGER_KURZ.get(c, c) for c in chain]
        chain_str = " → ".join(kurz)
        num_stages = len(chain)
        first_stage = kurz[0] if kurz else "eine unbekannte Quelle"

        idx = self._chain_story_index % len(_KETTEN_ERZAEHLUNG)
        self._chain_story_index += 1
        return _KETTEN_ERZAEHLUNG[idx].format(
            chain=chain_str, num_stages=num_stages, first_stage=first_stage)

    def chain_summary(self) -> str:
        """Abschliessende Zusammenfassung der Tonträgerkette."""
        ctx = self._context
        chain = ctx.get("transfer_chain") or []
        material = ctx.get("material", "")

        if not chain and not material:
            return ""

        lang = [_TRÄGER_NAMEN.get(c, c) for c in chain]
        chain_lang = " → ".join(lang) if lang else _TRÄGER_NAMEN.get(material, material)

        teile = [
            "📀 So kam Deine Musik zu Dir: " + chain_lang,
            "",
            "Wie Aurik das herausgefunden hat:",
        ]

        if len(chain) >= 2:
            teile.append(
                f"Deine Musik hat {len(chain)} Stationen durchlaufen. "
                f"So wie ein altes Foto, das mehrfach kopiert wurde, "
                f"hat jede Kopie ihre eigenen Spuren hinterlassen. "
                f"Aurik erkennt diese Spuren und weiss genau, "
                f"wie es Deine Musik am besten behandeln kann."
            )
        elif chain:
            teile.append(
                f"Deine Musik stammt von {lang[0]}. Aurik hat das "
                f"an den typischen Merkmalen dieses Trägers erkannt — "
                f"so wie man eine Geige von einem Klavier unterscheiden kann."
            )
        elif material:
            teile.append(
                f"Deine Musik zeigt alle typischen Eigenschaften "
                f"von {_TRÄGER_NAMEN.get(material, material)}."
            )

        if ctx.get("era_decade"):
            teile.append("")
            teile.append(
                f"Die Aufnahme stammt aus den {ctx['era_decade']}er Jahren — "
                f"eine Zeit mit einem ganz eigenen Klangcharakter."
            )

        return "\n".join(teile)

    # ── Fortschrittsmeldung erzeugen ────────────────────────────────────────

    def message_for(self, phase_id: str, phase_name: str = "",
                    progress_pct: int = 0) -> str:
        now = _time.monotonic()
        phase_key = phase_id or "_unbekannt"
        ctx = self._context

        # ── Tonträgerketten-Interlude bei sehr langen Phasen (>20s) ────────
        chain = ctx.get("transfer_chain") or []
        zeit_seit_letztem = now - self._last_ts.get(phase_key, 0.0)
        if chain and zeit_seit_letztem > 20.0 and not self._chain_story_told:
            self._chain_story_told = True
            self._last_ts[phase_key] = now
            return f"📀 {self.chain_narrative()}"

        # ── "Warum diese Phase?" — nur beim ersten Mal ────────────────────
        erster_aufruf = phase_key not in self._used or not self._used[phase_key]
        if erster_aufruf and progress_pct < 15:
            warum = self._warum(phase_id)
            if warum:
                self._used.setdefault(phase_key, []).append(-1)
                self._last_ts[phase_key] = now
                return warum

        # ── Aktivitäts-Templates durchwechseln ─────────────────────────────
        templates = self._aktivitaeten(phase_id)
        used = self._used.setdefault(phase_key, [])

        if len(used) >= len(templates) + 1:
            used[:] = [i for i in used if i >= 0]
            if len(used) >= len(templates):
                used.clear()

        verfuegbar = [i for i in range(len(templates)) if i not in used]
        if not verfuegbar:
            used.clear()
            verfuegbar = list(range(len(templates)))

        if zeit_seit_letztem >= self._rotate_every_s:
            idx = verfuegbar[hash(f"{phase_key}_{int(now)}") % len(verfuegbar)]
            used.append(idx)
            self._last_ts[phase_key] = now
        else:
            idx = used[-1] if used and used[-1] >= 0 else verfuegbar[0]

        vorlage = templates[idx % len(templates)]

        # ── Präfix ─────────────────────────────────────────────────────────
        symbol = self._symbol(phase_id)
        if phase_name:
            praefix = f"{symbol} {phase_name}: "
        elif progress_pct < 12:
            _anfaenge = ["Starte: ", "Beginne: ", "Bereite vor: "]
            praefix = _anfaenge[idx % len(_anfaenge)]
        elif progress_pct >= 92:
            _enden = ["Fast fertig: ", "Letzter Schliff: ", "Abschluss: "]
            praefix = _enden[idx % len(_enden)]
        else:
            praefix = ""

        return f"{praefix}{vorlage}"

    # ── Hilfsfunktionen ─────────────────────────────────────────────────────

    def _warum(self, phase_id: str) -> str:
        pid = str(phase_id or "").lower()
        ctx = self._context
        material = ctx.get("material", "")
        for schluessel, varianten in _WARUM_DIESE_PHASE.items():
            if schluessel in pid or pid.startswith(schluessel):
                if material and material in varianten:
                    texte = varianten[material]
                else:
                    texte = varianten.get("_default", [])
                if texte:
                    anzahl = len(ctx.get("defects", []) or [])
                    return texte[hash(f"{pid}_{self._session_key}") % len(texte)].format(
                        material=_TRÄGER_NAMEN.get(material, material or "diesem Träger"),
                        defect_count=anzahl)
        return ""

    def _aktivitaeten(self, phase_id: str) -> list[str]:
        pid = str(phase_id or "").lower()
        for schluessel in _AKTIVITAETEN:
            if schluessel in pid or pid.startswith(schluessel):
                return _AKTIVITAETEN[schluessel]
        return _ALLGEMEINE_AKTIVITAETEN

    @staticmethod
    def _symbol(phase_id: str) -> str:
        pid = str(phase_id or "").lower()
        if "rausch" in pid or "noise" in pid or "denoise" in pid: return "🔇"
        if "knack" in pid or "click" in pid or "crackle" in pid: return "🔍"
        if "klang" in pid or "eq" in pid or "frequenz" in pid: return "🎚️"
        if "harmonisch" in pid or "warm" in pid: return "🔥"
        if "stimme" in pid or "vocal" in pid or "gesang" in pid: return "🎤"
        if "stereo" in pid or "raum" in pid: return "🎧"
        if "master" in pid or "polish" in pid or "schliff" in pid: return "✨"
        if "export" in pid or "speichern" in pid: return "💾"
        if "rumble" in pid or "brumm" in pid or "hum" in pid: return "📉"
        if "reparatur" in pid or "repair" in pid: return "🔧"
        if "wow" in pid or "flutter" in pid or "gleichlauf" in pid: return "〰️"
        if "tempo" in pid or "speed" in pid or "pitch" in pid: return "⏱️"
        if "laut" in pid or "loudness" in pid: return "📊"
        if "hall" in pid or "reverb" in pid: return "🏠"
        if "scan" in pid or "analys" in pid or "untersuch" in pid: return "🔬"
        return "⚙️"


# Singleton
_erzaehler: PhaseProgressNarrator | None = None

def get_narrator() -> PhaseProgressNarrator:
    global _erzaehler
    if _erzaehler is None:
        _erzaehler = PhaseProgressNarrator()
    return _erzaehler
