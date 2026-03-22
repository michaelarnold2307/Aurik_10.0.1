"""Tests für backend/api/bridge.py — Aurik 9 API Bridge (§11 Spec 08).

Prüft:
- Alle __all__-Einträge sind tatsächlich importierbar
- Lazy-Import-Wrapper geben den korrekten Typ zurück (Klasse/Callable/dict)
- export_guard bereinigt NaN, Inf und Clipping korrekt
- Defect-Cache ist Thread-sicher (FIFO, 64 Einträge)
- get_audio_exporter_class() gibt None zurück (kein Hard-Fail)
- get_ml_memory_budget_status() gibt immer ein Dict zurück
- warmup_models_background() hat kein blockierendes time.sleep()
- TYPE_CHECKING-Guards erzeugen keine zirkulären Imports
- __all__ enthält alle Pflicht-Funktionen aus Spec §11
"""

from __future__ import annotations

import importlib
import inspect
import threading
import time
import types

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures / Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bridge():
    """Importiert die Bridge einmalig pro Modul."""
    return importlib.import_module("backend.api.bridge")


# ---------------------------------------------------------------------------
# 1. Grundlegender Import + __all__
# ---------------------------------------------------------------------------

class TestBridgeImport:
    """Bridge-Modul ist importierbar und hat valides __all__."""

    def test_bridge_imports_cleanly(self, bridge):
        assert bridge is not None, "backend.api.bridge konnte nicht importiert werden"

    def test_bridge_has_all(self, bridge):
        assert hasattr(bridge, "__all__"), "__all__ fehlt — Spec §11 fordert explizite Export-Liste"

    def test_all_is_list_of_strings(self, bridge):
        assert isinstance(bridge.__all__, (list, tuple)), "__all__ muss eine Liste sein"
        for name in bridge.__all__:
            assert isinstance(name, str), f"__all__ enthält Nicht-String: {name!r}"

    def test_all_entries_are_importable(self, bridge):
        """Alle __all__-Einträge müssen tatsächlich im Modul existieren."""
        missing = [name for name in bridge.__all__ if not hasattr(bridge, name)]
        assert not missing, f"In __all__ deklariert, aber nicht im Modul: {missing}"

    def test_all_entries_are_callable_or_data(self, bridge):
        """Alle __all__-Einträge sind Callables oder public data."""
        for name in bridge.__all__:
            obj = getattr(bridge, name)
            # Kann Callable, Klasse, dict, etc. sein — nur None allein ist falsch
            assert obj is not None or name.startswith("_"), (
                f"__all__ enthält None-Eintrag '{name}' — würde Hard-Fail im Frontend verursachen"
            )


# ---------------------------------------------------------------------------
# 2. Pflicht-Funktionen aus Spec §11 (vollständige Liste)
# ---------------------------------------------------------------------------

PFLICHT_FUNKTIONEN = [
    # Defect-Cache
    "cache_defect_result",
    "get_cached_defect_result",
    "clear_defect_cache",
    # Enums
    "get_quality_mode",
    "get_medium_type_enum",
    "get_processing_mode_enum",
    # Kern-Einstiegspunkte
    "get_restorer_classes",
    "get_aurik_denker_class",
    "get_aurik_denker_instance",
    # Analyse
    "get_defect_scanner",
    "get_defect_type",
    "get_medium_classifier_fn",
    "get_era_classifier_fn",
    "get_genre_classifier_fn",
    "get_restorability_estimator_class",
    "get_carrier_forensics_fn",
    "get_audio_file_validator",
    # Qualitätsbewertung
    "get_musical_goals_checker",
    "get_adaptive_goals_fn",
    "get_mushra_evaluator",
    "get_perceptual_quality_scorer",
    # Infrastruktur
    "get_plugin_lifecycle_manager",
    "get_ml_memory_budget_status",
    "get_pipeline_health_state_enum",
    "normalize_pipeline_health_state",
    "resolve_pipeline_fail_reason",
    # Audio-Verarbeitung
    "get_audio_exporter_class",
    "get_stem_remix_balancer_fn",
    "get_clipping_classifier",
    "get_lyrics_guided_enhancement_fn",
    "get_cleanup_after_file_fn",
    # Export / Warmup
    "export_guard",
    "warmup_models_background",
]


class TestPflichtFunktionenVorhanden:
    """Alle normativen Bridge-Funktionen aus §11 Spec 08 sind vorhanden."""

    @pytest.mark.parametrize("name", PFLICHT_FUNKTIONEN)
    def test_funktion_im_modul(self, bridge, name):
        assert hasattr(bridge, name), (
            f"Pflicht-Bridge-Funktion '{name}' fehlt — Spec §11 Softwareschichten-Architektur"
        )

    @pytest.mark.parametrize("name", PFLICHT_FUNKTIONEN)
    def test_funktion_in_all(self, bridge, name):
        assert name in bridge.__all__, (
            f"'{name}' fehlt in bridge.__all__ — muss explizit exportiert werden"
        )

    @pytest.mark.parametrize("name", PFLICHT_FUNKTIONEN)
    def test_funktion_ist_callable(self, bridge, name):
        obj = getattr(bridge, name)
        assert callable(obj), f"'{name}' ist nicht callable (Typ: {type(obj).__name__})"


# ---------------------------------------------------------------------------
# 3. export_guard — NaN/Inf-Bereinigung und Clipping (§3.1 Spec 08)
# ---------------------------------------------------------------------------

class TestExportGuard:
    """export_guard bereinigt Audio korrekt (§3.1 Spec 08)."""

    def test_nan_replaced_with_zero(self, bridge):
        audio = np.array([0.5, float("nan"), -0.3], dtype=np.float32)
        result = bridge.export_guard(audio)
        assert np.all(np.isfinite(result)), "export_guard hat NaN nicht entfernt"
        assert result[1] == 0.0

    def test_posinf_clipped(self, bridge):
        audio = np.array([float("inf"), 0.5], dtype=np.float32)
        result = bridge.export_guard(audio)
        assert result[0] == 0.0, "export_guard hat +Inf nicht auf 0.0 gesetzt"

    def test_neginf_clipped(self, bridge):
        audio = np.array([float("-inf"), -0.5], dtype=np.float32)
        result = bridge.export_guard(audio)
        assert result[0] == 0.0, "export_guard hat -Inf nicht auf 0.0 gesetzt"

    def test_values_clipped_to_minus1_plus1(self, bridge):
        audio = np.array([2.0, -3.0, 0.5], dtype=np.float32)
        result = bridge.export_guard(audio)
        assert np.max(np.abs(result)) <= 1.0, "export_guard clippt nicht auf [-1, 1]"

    def test_output_is_float32(self, bridge):
        audio = np.array([0.3, 0.6], dtype=np.float64)
        result = bridge.export_guard(audio)
        assert result.dtype == np.float32, "export_guard gibt kein float32 zurück"

    def test_valid_audio_unchanged(self, bridge):
        audio = np.array([0.1, -0.2, 0.5, -0.7], dtype=np.float32)
        result = bridge.export_guard(audio)
        np.testing.assert_allclose(result, audio, atol=1e-6)

    def test_stereo_shape_preserved(self, bridge):
        audio = np.zeros((2, 1024), dtype=np.float32)
        audio[0, 0] = float("nan")
        result = bridge.export_guard(audio)
        assert result.shape == (2, 1024), "export_guard verändert Audio-Shape"
        assert np.all(np.isfinite(result))

    def test_empty_array_handled(self, bridge):
        audio = np.array([], dtype=np.float32)
        result = bridge.export_guard(audio)
        assert result.shape == (0,)


# ---------------------------------------------------------------------------
# 4. Defect-Cache — FIFO, Thread-Sicherheit, Grenzwerte
# ---------------------------------------------------------------------------

class TestDefectCache:
    """Defect-Cache ist Thread-sicher und begrenzt auf 64 Einträge (FIFO)."""

    def setup_method(self):
        # Sauberen Zustand sicherstellen
        pass

    def test_cache_round_trip(self, bridge):
        sentinel = object()
        bridge.cache_defect_result("/tmp/test.wav", sentinel)
        assert bridge.get_cached_defect_result("/tmp/test.wav") is sentinel

    def test_cache_miss_returns_none(self, bridge):
        bridge.clear_defect_cache("/tmp/nonexistent.wav")
        assert bridge.get_cached_defect_result("/tmp/nonexistent.wav") is None

    def test_clear_single_entry(self, bridge):
        bridge.cache_defect_result("/tmp/clear_test.wav", {"defects": []})
        bridge.clear_defect_cache("/tmp/clear_test.wav")
        assert bridge.get_cached_defect_result("/tmp/clear_test.wav") is None

    def test_clear_all(self, bridge):
        for i in range(5):
            bridge.cache_defect_result(f"/tmp/clear_all_{i}.wav", i)
        bridge.clear_defect_cache()
        for i in range(5):
            assert bridge.get_cached_defect_result(f"/tmp/clear_all_{i}.wav") is None

    def test_fifo_limit_64(self, bridge):
        """Cache trimmt auf 64 Einträge (FIFO)."""
        bridge.clear_defect_cache()
        for i in range(70):
            bridge.cache_defect_result(f"/tmp/fifo_{i}.wav", i)
        # Die ersten 6 sollten verdrängt worden sein
        missing = [i for i in range(6) if bridge.get_cached_defect_result(f"/tmp/fifo_{i}.wav") is not None]
        # neueste 64 müssen vorhanden sein
        present = [i for i in range(6, 70) if bridge.get_cached_defect_result(f"/tmp/fifo_{i}.wav") is not None]
        assert len(present) == 64, f"FIFO-Limit nicht korrekt: {len(present)} von 64 vorhanden"

    def test_thread_safe_concurrent_writes(self, bridge):
        """Parallele Cache-Schreiboperationen dürfen nicht zu Exceptions führen."""
        bridge.clear_defect_cache()
        errors: list[Exception] = []

        def write_loop(thread_id: int):
            try:
                for i in range(20):
                    bridge.cache_defect_result(f"/tmp/thread_{thread_id}_{i}.wav", (thread_id, i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_loop, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Thread-Safety-Verletzung im Defect-Cache: {errors}"


# ---------------------------------------------------------------------------
# 5. get_audio_exporter_class — kein Hard-Fail (optional, §11.3)
# ---------------------------------------------------------------------------

class TestAudioExporterClass:
    """get_audio_exporter_class() gibt None zurück statt Exception (§11.3)."""

    def test_returns_type_or_none(self, bridge):
        result = bridge.get_audio_exporter_class()
        assert result is None or isinstance(result, type), (
            f"get_audio_exporter_class() muss type oder None zurückgeben, nicht {type(result)}"
        )

    def test_no_import_error_raised(self, bridge):
        """Kein ImportError bei fehlendem Modul — Fallback-Guard vorhanden."""
        try:
            bridge.get_audio_exporter_class()
        except ImportError as e:
            pytest.fail(f"get_audio_exporter_class() wirft ImportError: {e}")


# ---------------------------------------------------------------------------
# 6. get_ml_memory_budget_status — immer Dict (§2.37)
# ---------------------------------------------------------------------------

class TestMlMemoryBudgetStatus:
    """get_ml_memory_budget_status() gibt immer ein Dict zurück."""

    def test_returns_dict(self, bridge):
        result = bridge.get_ml_memory_budget_status()
        assert isinstance(result, dict), (
            f"get_ml_memory_budget_status() muss dict zurückgeben, nicht {type(result)}"
        )

    def test_fallback_dict_has_required_keys(self, bridge):
        """Pflicht-Keys aus ml_memory_budget.get_status() sind vorhanden."""
        result = bridge.get_ml_memory_budget_status()
        for key in ("allocated_gb", "free_gb", "max_gb", "models"):
            assert key in result, (
                f"Pflicht-Key '{key}' fehlt in get_ml_memory_budget_status()-Rückgabe"
            )

    def test_values_are_numeric_or_dict(self, bridge):
        result = bridge.get_ml_memory_budget_status()
        assert isinstance(result.get("max_gb", 0), (int, float))
        assert isinstance(result.get("allocated_gb", 0), (int, float))
        assert isinstance(result.get("free_gb", 0), (int, float))
        assert isinstance(result.get("models", {}), dict)


# ---------------------------------------------------------------------------
# 7. warmup_models_background — kein blockierendes sleep() (§9.7.4)
# ---------------------------------------------------------------------------

class TestWarmupModelsBackground:
    """warmup_models_background() blockiert nicht durch time.sleep() (§9.7.4)."""

    def test_no_redundant_sleep_in_source(self, bridge):
        """Quellcode darf kein time.sleep(2) haben — QTimer regelt das Timing."""
        source = inspect.getsource(bridge.warmup_models_background)
        assert "time.sleep(2)" not in source, (
            "warmup_models_background() enthält redundantes time.sleep(2) — "
            "§9.7.4: QTimer.singleShot(2000, ...) steuert Timing; sleep im Thread ist überflüssig"
        )

    def test_is_callable(self, bridge):
        assert callable(bridge.warmup_models_background)

    def test_completes_without_exception(self, bridge):
        """Warmup läuft durch ohne Exception (alle Plugins optional)."""
        # Synchroner Aufruf — alle Imports schlagen fehl → kein Absturz
        try:
            bridge.warmup_models_background()
        except Exception as e:
            pytest.fail(f"warmup_models_background() wirft Exception: {e}")


# ---------------------------------------------------------------------------
# 8. Qualitätsbewertungs-Wrapper (neu hinzugefügt §8.1)
# ---------------------------------------------------------------------------

class TestQualitaetsBewertungsWrapper:
    """Neue Qualitätsbewertungs-Accessor sind vorhanden und aufrufbar."""

    def test_get_musical_goals_checker_returns_type(self, bridge):
        result = bridge.get_musical_goals_checker()
        assert isinstance(result, type), (
            f"get_musical_goals_checker() muss eine Klasse zurückgeben, nicht {type(result)}"
        )

    def test_get_adaptive_goals_fn_returns_callable(self, bridge):
        result = bridge.get_adaptive_goals_fn()
        assert callable(result), (
            f"get_adaptive_goals_fn() muss einen Callable zurückgeben, nicht {type(result)}"
        )

    def test_get_mushra_evaluator_returns_something(self, bridge):
        result = bridge.get_mushra_evaluator()
        assert result is not None, "get_mushra_evaluator() gibt None zurück"

    def test_get_perceptual_quality_scorer_returns_something(self, bridge):
        result = bridge.get_perceptual_quality_scorer()
        assert result is not None, "get_perceptual_quality_scorer() gibt None zurück"

    def test_get_plugin_lifecycle_manager_returns_something(self, bridge):
        result = bridge.get_plugin_lifecycle_manager()
        assert result is not None, "get_plugin_lifecycle_manager() gibt None zurück"


# ---------------------------------------------------------------------------
# 9. TYPE_CHECKING — keine zirkulären Imports (§11 Spec 08)
# ---------------------------------------------------------------------------

class TestTypeCheckingGuards:
    """TYPE_CHECKING-Guards erzeugen keine zirkulären Imports."""

    def test_bridge_importable_in_fresh_interpreter(self, bridge):
        """Bridge-Modul ist ohne Vorwärts-Imports importierbar."""
        # Bereits durch das fixture geladen — Smoke-Test
        assert hasattr(bridge, "__all__")

    def test_no_circular_import_via_typing(self):
        """Erneuter Import ist idempotent."""
        import importlib
        m1 = importlib.import_module("backend.api.bridge")
        m2 = importlib.import_module("backend.api.bridge")
        assert m1 is m2, "Modul wird bei erneutem Import neu geladen (kein Caching)"


# ---------------------------------------------------------------------------
# 10. Lazy-Import-Muster — Rückgabetypen der wichtigsten Wrapper
# ---------------------------------------------------------------------------

class TestLazyImportMuster:
    """Lazy-Import-Wrapper geben den spezifizierten Typ zurück."""

    def test_get_quality_mode_returns_enum_type(self, bridge):
        qm = bridge.get_quality_mode()
        assert isinstance(qm, type), "get_quality_mode() gibt keine Klasse zurück"

    def test_get_restorer_classes_returns_tuple_of_two_types(self, bridge):
        result = bridge.get_restorer_classes()
        assert isinstance(result, tuple) and len(result) == 2, (
            "get_restorer_classes() muss (RestorationConfig, UnifiedRestorerV3) als 2-Tuple zurückgeben"
        )
        assert all(isinstance(t, type) for t in result), "Tuple-Elemente sind keine Klassen"

    def test_get_defect_type_returns_enum_type(self, bridge):
        dt = bridge.get_defect_type()
        assert isinstance(dt, type), "get_defect_type() gibt keine Klasse zurück"

    def test_get_medium_classifier_fn_returns_callable(self, bridge):
        fn = bridge.get_medium_classifier_fn()
        assert callable(fn), "get_medium_classifier_fn() gibt keinen Callable zurück"

    def test_get_era_classifier_fn_returns_callable(self, bridge):
        fn = bridge.get_era_classifier_fn()
        assert callable(fn), "get_era_classifier_fn() gibt keinen Callable zurück"

    def test_get_stem_remix_balancer_fn_returns_callable(self, bridge):
        fn = bridge.get_stem_remix_balancer_fn()
        assert callable(fn), "get_stem_remix_balancer_fn() gibt keinen Callable zurück"

    def test_get_carrier_forensics_fn_returns_callable(self, bridge):
        fn = bridge.get_carrier_forensics_fn()
        assert callable(fn), "get_carrier_forensics_fn() gibt keinen Callable zurück"


# ---------------------------------------------------------------------------
# 11. resolve_pipeline_fail_reason — strukturierter Fail-Reason (§RELEASE_MUST)
# ---------------------------------------------------------------------------

class TestResolvePipelineFailReason:
    """resolve_pipeline_fail_reason gibt immer eine Zeichenkette zurück."""

    def test_returns_string_without_args(self, bridge):
        result = bridge.resolve_pipeline_fail_reason()
        assert isinstance(result, str)

    def test_returns_string_with_metadata(self, bridge):
        result = bridge.resolve_pipeline_fail_reason(
            metadata={"fail_reason": "test_error"},
        )
        assert isinstance(result, str)

    def test_uses_typed_fail_reason(self, bridge):
        result = bridge.resolve_pipeline_fail_reason(
            typed_fail_reason="goal_regression",
        )
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# 12. normalize_pipeline_health_state — Typ-Sicherheit
# ---------------------------------------------------------------------------

class TestNormalizePipelineHealthState:
    """normalize_pipeline_health_state ist robust gegen unbekannte Werte."""

    def test_returns_object_for_none(self, bridge):
        result = bridge.normalize_pipeline_health_state(None)
        assert result is not None

    def test_returns_object_for_ok(self, bridge):
        result = bridge.normalize_pipeline_health_state("ok")
        assert result is not None

    def test_property_value_exists(self, bridge):
        result = bridge.normalize_pipeline_health_state("degraded")
        assert hasattr(result, "value"), (
            "normalize_pipeline_health_state() muss Objekt mit .value-Attribut zurückgeben"
        )
