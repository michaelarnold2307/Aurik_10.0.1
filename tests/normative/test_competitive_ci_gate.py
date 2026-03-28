"""Competitive CI-Gate — Aurik muss iZotope RX 11 in der Mehrheit der Szenarien schlagen.

Spec §8.2 Punkt 11 (copilot-instructions.md):
    Aurik ≥ iZotope RX 11 in ≥ 7/10 AMRB-Szenarien (elektrisch messbar).
    Messung via MUSHRA-Score aus run_benchmark() — KEINE Speech-Metriken (PESQ, STOI etc.).

Hinweis: Ein direkter iZotope-Aufruf ist im CI nicht möglich. Als Proxy dient der
AMRB-Baseline-MUSHRA von iZotope RX 11 (71.0) aus AMRB_BASELINES. Aurik muss diesen
Wert in ≥ 7 von 10 Szenarien übertreffen.

VERBOTENE METRIKEN (spec §3.1, §4.4):
    PESQ, STOI, SI-SDR, VISQOL (Speech Mode), DNSMOS, NISQA
    → Stattdessen: MUSHRA (OQS), PQS-MOS, Musical Goals

Ausführung: pytest tests/normative/test_competitive_ci_gate.py -m competitive --timeout=600 -v
Ausschluss: pytest -m "not competitive"

Nightly-Modus (Spec: n_items ≥ 5 für statistische Robustheit):
    AURIK_NIGHTLY_ITEMS=5 pytest ... -m competitive
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pytest

from benchmarks.musical_restoration_benchmark import (
    AMRB_BASELINES,
    BenchmarkConfig,
    BenchmarkReport,
    run_benchmark,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Referenz-Baselines (AMRB_BASELINES §8.1)
# ---------------------------------------------------------------------------
_IZOTOPE_MUSHRA: float = AMRB_BASELINES["iZotope RX 11 (commercial)"]["mushra_overall"]  # 71.0
_IZOTOPE_PQS_MOS: float = AMRB_BASELINES["iZotope RX 11 (commercial)"]["pqs_mos"]  # 3.9
_AURIK_STUDIO_MUSHRA: float = AMRB_BASELINES["Aurik 9.9 (Studio 2026 Mode)"]["mushra_overall"]  # 88.0
_AURIK_RESTORE_MUSHRA: float = AMRB_BASELINES["Aurik 9.9 (Restoration Mode)"]["mushra_overall"]  # 84.0
_MIN_SCENARIOS_TO_WIN: int = 7  # §8.2 Punkt 11: ≥ 7/10 Szenarien


# Spec: "Nightly runs: n_items ≥ 5 für statistische Robustheit"
# CI-Standard: 1 (schnell). Nightly: AURIK_NIGHTLY_ITEMS=5 setzen.
def _resolve_competitive_n_items() -> int:
    """Resolve n_items for competitive benchmark runs.

    Rules:
      - Default CI (env not set): 1
      - Nightly (env set): enforce >= 5 per Spec robustness requirement
    """
    raw = os.environ.get("AURIK_NIGHTLY_ITEMS")
    if raw is None:
        return 1
    try:
        requested = max(1, int(raw))
    except ValueError:
        requested = 5
    return max(5, requested)


_N_ITEMS_DEFAULT: int = _resolve_competitive_n_items()

# Per-Szenario-Schwelle: gut unterhalb des iZotope-Gesamtscores,
# um Messrauschen zu tolerieren. Aurik muss spürbar besser sein.
_PER_SCENARIO_WIN_THRESHOLD: float = _IZOTOPE_MUSHRA  # strikt: muss iZotope schlagen


# ---------------------------------------------------------------------------
# Aurik-restoration_fn
# ---------------------------------------------------------------------------


def _aurik_restoration_fn(audio: np.ndarray, sr: int) -> np.ndarray:
    """Ruft UnifiedRestorerV3 auf; fällt bei Fehler auf Pass-Through zurück."""
    try:
        from backend.core.unified_restorer_v3 import get_restorer  # type: ignore[import]

        result = get_restorer().restore(audio, sr)
        return result.audio
    except Exception as exc:  # pragma: no cover
        logger.warning("Aurik-Engine nicht verfügbar (%s) — Pass-Through (schlechte Scores erwartet).", exc)
        return audio


def _run_competitive(n_items: int = 1, verbose: bool = False) -> BenchmarkReport:
    config = BenchmarkConfig(
        restoration_fn=_aurik_restoration_fn,
        system_name="Aurik 9 Competitive",
        n_items_per_scenario=n_items,
        verbose=verbose,
    )
    return run_benchmark(config)


# ===========================================================================
# Competitive Tests
# ===========================================================================


@pytest.mark.competitive
@pytest.mark.timeout(600)
def test_aurik_beats_izotope_in_majority_of_scenarios() -> None:
    """Aurik MUSHRA muss iZotope RX 11 Baseline (71.0) in ≥ 7/10 Szenarien übertreffen.

    §8.2 Punkt 11: Pflicht-Benchmark für Weltmarktführer-Anspruch.
    VERBOTEN: PESQ, STOI, SI-SDR, VISQOL — ausschließlich MUSHRA (OQS) als Maßstab.
    """
    report = _run_competitive(n_items=_N_ITEMS_DEFAULT, verbose=True)

    scenarios_won = sum(1 for res in report.scenario_results.values() if res.mushra_mean > _PER_SCENARIO_WIN_THRESHOLD)
    losing_scenarios = [
        f"  {sid}: MUSHRA {res.mushra_mean:.1f} ≤ {_PER_SCENARIO_WIN_THRESHOLD:.1f}"
        for sid, res in report.scenario_results.items()
        if res.mushra_mean <= _PER_SCENARIO_WIN_THRESHOLD
    ]

    assert scenarios_won >= _MIN_SCENARIOS_TO_WIN, (
        f"\nCompetitive-Gate NICHT BESTANDEN:\n"
        f"  Szenarien > iZotope RX 11 : {scenarios_won}/10  (Ziel: ≥ {_MIN_SCENARIOS_TO_WIN})\n"
        f"  iZotope RX 11 Baseline    : MUSHRA {_IZOTOPE_MUSHRA:.1f}\n"
        f"  Aurik Gesamt-Score        : {report.overall_score:.1f}/100\n"
        f"  Schwächstes Szenario      : {report.worst_scenario}\n"
        f"\n"
        f"Verlorene Szenarien:\n" + "\n".join(losing_scenarios)
    )


@pytest.mark.competitive
@pytest.mark.timeout(600)
def test_aurik_overall_score_above_izotope_overall() -> None:
    """Aurik Gesamt-MUSHRA muss den iZotope-Gesamt-MUSHRA (71.0) deutlich übertreffen."""
    report = _run_competitive(n_items=_N_ITEMS_DEFAULT)

    margin = report.overall_score - _IZOTOPE_MUSHRA
    assert report.overall_score > _IZOTOPE_MUSHRA, (
        f"Aurik Gesamt-Score ({report.overall_score:.1f}) liegt NICHT über "
        f"iZotope RX 11 Baseline ({_IZOTOPE_MUSHRA:.1f}). "
        f"Differenz: {margin:+.1f} Punkte."
    )
    logger.info(
        "Competitive: Aurik %.1f vs iZotope %.1f (+%.1f Punkte Vorsprung)",
        report.overall_score,
        _IZOTOPE_MUSHRA,
        margin,
    )


@pytest.mark.competitive
@pytest.mark.timeout(600)
def test_aurik_pqs_mos_above_izotope_baseline() -> None:
    """Aurik-PQS-MOS muss iZotope PQS-MOS Baseline (3.9) übertreffen."""
    report = _run_competitive(n_items=_N_ITEMS_DEFAULT)

    # Alle Szenario-PQS-MOS-Werte sammeln
    all_pqs = [
        res.pqs_mos_mean
        for res in report.scenario_results.values()
        if hasattr(res, "pqs_mos_mean") and res.pqs_mos_mean is not None
    ]

    if not all_pqs:
        pytest.fail(
            "Keine PQS-MOS-Daten im BenchmarkReport — PQS-MOS ist eine Pflicht-Metrik"
            " (spec §8.1). BenchmarkReport.scenario_results muss pqs_mos_mean befüllen."
        )

    mean_pqs = float(np.mean(all_pqs))
    assert mean_pqs > _IZOTOPE_PQS_MOS, (
        f"Aurik PQS-MOS ({mean_pqs:.2f}) liegt nicht über iZotope RX 11 Baseline "
        f"({_IZOTOPE_PQS_MOS:.1f}). Metriken: MUSHRA/PQS-MOS zulässig. "
        f"PESQ/STOI sind verboten (§4.4)."
    )


@pytest.mark.timeout(30)
def test_competitive_no_forbidden_metrics_used() -> None:
    """Stellt sicher, dass verbotene Speech-Metriken nicht in benchmark_suite importiert werden.

    §3.1/§4.4: PESQ, STOI, SI-SDR, VISQOL (Speech-Mode), DNSMOS, NISQA sind für
    Musik-Qualitätsbewertung absolut verboten.

    Läuft IMMER (kein @pytest.mark.competitive), da rein strukturell — kein ML-Lauf nötig.
    """
    import importlib

    # Modul laden (oder bereits geladen nutzen)
    module_name = "benchmarks.competitive.benchmark_suite"
    try:
        suite = importlib.import_module(module_name)
    except ImportError:
        pytest.fail(
            f"{module_name} nicht importierbar —"
            " benchmarks/competitive/benchmark_suite.py muss vorhanden und"
            " importierbar sein (spec §4.4: FORBIDDEN_METRICS-Pflicht)."
        )

    # Prüfe, ob FORBIDDEN_METRICS-Konstante existiert
    assert hasattr(suite, "FORBIDDEN_METRICS"), (
        f"{module_name} enthält keine FORBIDDEN_METRICS-Konstante. "
        f"Bitte benchmarks/competitive/benchmark_suite.py gemäß §4.4 aktualisieren."
    )

    forbidden = set(suite.FORBIDDEN_METRICS)
    expected_forbidden = {"pesq", "stoi", "si_sdr", "visqol", "dnsmos", "nisqa"}
    missing = expected_forbidden - {m.lower() for m in forbidden}

    assert not missing, (
        f"FORBIDDEN_METRICS in {module_name} fehlen folgende verbotene Metriken: {missing}\n"
        f"Aktuell deklariert: {forbidden}"
    )


@pytest.mark.timeout(10)
def test_competitive_nightly_items_threshold_guard() -> None:
    """Spec-Guard: gesetztes Nightly-Flag muss n_items >= 5 erzwingen.

    - Ohne Env-Flag bleibt schneller CI-Default bei 1.
    - Mit Env-Flag (Nightly) wird per Resolver auf mindestens 5 geklemmt.
    """
    if os.environ.get("AURIK_NIGHTLY_ITEMS") is None:
        assert _N_ITEMS_DEFAULT == 1
    else:
        assert _N_ITEMS_DEFAULT >= 5


@pytest.mark.timeout(10)
def test_competitive_gate_baseline_is_rx11_not_rx10() -> None:
    """§8.2 RELEASE_MUST: Wettbewerber-Baseline muss iZotope RX 11 sein (nicht RX 10).

    Stellt sicher, dass das Competitive Gate nicht still gegen die alte, niedrigere
    RX-10-Baseline (OQS 71.0) läuft und somit ein Release fälschlich freigibt.

    Läuft IMMER ohne @pytest.mark.competitive — rein strukturelle Invariante.
    """
    rx11_key = "iZotope RX 11 (commercial)"
    assert rx11_key in AMRB_BASELINES, (
        f"AMRB_BASELINES enthält keinen Eintrag für '{rx11_key}'. "
        "Spec §8.2: Competitive Gate muss gegen RX 11 messen — bitte "
        "benchmarks/musical_restoration_benchmark.py aktualisieren."
    )
    baseline = AMRB_BASELINES[rx11_key]
    assert "mushra_overall" in baseline, f"AMRB_BASELINES['{rx11_key}'] fehlt 'mushra_overall'."
    rx11_mushra = baseline["mushra_overall"]
    # RX 11 Baseline ist 71.0 — Guard gegen versehentliches Herabsetzen auf RX 10 (< 68)
    assert rx11_mushra >= 68.0, (
        f"AMRB_BASELINES['{rx11_key}']['mushra_overall'] = {rx11_mushra} liegt unter 68 — "
        "sieht aus wie eine RX-10-Baseline. §8.2: RX-11-MUSHRA-Baseline ≥ 68 erwartet."
    )
    # Gesamtschwelle: Aurik muss diese Baseline schlagen
    assert rx11_mushra == _PER_SCENARIO_WIN_THRESHOLD, (
        f"_PER_SCENARIO_WIN_THRESHOLD ({_PER_SCENARIO_WIN_THRESHOLD}) != "
        f"AMRB_BASELINES[RX11].mushra_overall ({rx11_mushra}). "
        "Gate-Schwelle muss dynamisch aus AMRB_BASELINES bezogen werden."
    )


@pytest.mark.timeout(10)
def test_competitive_gate_min_scenarios_is_seven() -> None:
    """§8.2 RELEASE_MUST: Aurik muss ≥ 7/10 Szenarien gegen RX 11 gewinnen.

    Prüft die strukturelle Invariante — kein ML-Lauf nötig.
    """
    assert _MIN_SCENARIOS_TO_WIN == 7, (
        f"_MIN_SCENARIOS_TO_WIN = {_MIN_SCENARIOS_TO_WIN} ≠ 7. "
        "Spec §8.2 Punkt 11: ≥ 7 von 10 Szenarien müssen iZotope RX 11 schlagen."
    )
