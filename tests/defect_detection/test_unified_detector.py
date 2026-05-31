"""
Unified Defect Detection System - Tests
========================================

Comprehensive tests for defect detection functionality.
"""

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from backend.defect_detection import DefectInstance, DefectType, SeverityLevel, UnifiedDefectDetector
from backend.defect_detection.treatment_recommender import TreatmentRecommender


@pytest.fixture
def detector():
    """Create unified defect detector."""
    return UnifiedDefectDetector()


@pytest.fixture
def clean_audio():
    """Generate clean test audio (1 second, 48 kHz)."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    # Pure sine wave at 440 Hz
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    return audio, sr


@pytest.fixture
def clipped_audio():
    """Generate clipped audio."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    # Sine wave that clips
    audio = 1.5 * np.sin(2 * np.pi * 440 * t)
    audio = np.clip(audio, -1.0, 1.0)
    return audio, sr


@pytest.fixture
def noisy_audio():
    """Generate noisy audio."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    # Sine wave + broadband noise
    signal = 0.3 * np.sin(2 * np.pi * 440 * t)
    noise = 0.2 * np.random.randn(len(t))
    audio = signal + noise
    return audio, sr


@pytest.fixture
def audio_with_hum():
    """Generate audio with 60 Hz hum."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    # Signal + 60 Hz hum + harmonics
    signal = 0.3 * np.sin(2 * np.pi * 440 * t)
    hum = 0.15 * np.sin(2 * np.pi * 60 * t)
    hum += 0.08 * np.sin(2 * np.pi * 120 * t)
    hum += 0.04 * np.sin(2 * np.pi * 180 * t)
    audio = signal + hum
    return audio, sr


@pytest.fixture
def audio_with_dc_offset():
    """Generate audio with DC offset."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.05  # DC offset
    return audio, sr


@pytest.fixture
def stereo_imbalanced_audio():
    """Generate stereo audio with channel imbalance."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    left = 0.3 * np.sin(2 * np.pi * 440 * t)
    right = 0.1 * np.sin(2 * np.pi * 440 * t)  # 10 dB quieter
    audio = np.column_stack([left, right])
    return audio, sr


# ============================================================================
# Clean Audio Tests
# ============================================================================


def test_clean_audio_has_no_defects(detector, clean_audio):
    """Clean audio should have no significant defects."""
    audio, sr = clean_audio
    # SOTA-Weltspitze-Toleranzen für Clean Audio (maximal robust)
    custom_tolerances = {
        "clipping": 0.01,
        "broadband_noise": 0.5,
        "hum": 0.5,
        "stereo_imbalance": 2.0,
        "dc_offset": 0.05,
        "clicks": 0.3,
        "rumble": 0.3,
        "distortion": 0.3,
        "hf_rolloff": 0.3,
    }
    detector = UnifiedDefectDetector(custom_tolerances=custom_tolerances)
    report = detector.analyze(audio, sr)
    assert report.total_defects == 0 or all(d.severity < 0.1 for d in report.defects)
    assert report.overall_quality > 0.9
    assert not report.needs_restoration


def test_clean_audio_quality_score(detector, clean_audio):
    """Clean audio should have high quality score."""
    audio, sr = clean_audio
    report = detector.analyze(audio, sr)

    assert report.overall_quality >= 0.9
    assert report.critical_count == 0
    assert report.severe_count == 0


# ============================================================================
# Clipping Detection Tests
# ============================================================================


def test_detect_clipping(detector, clipped_audio):
    """Should detect clipping in overdriven audio."""
    audio, sr = clipped_audio
    report = detector.analyze(audio, sr)

    # Should detect clipping
    clipping_defects = report.get_defects_by_type(DefectType.CLIPPING)
    assert len(clipping_defects) > 0

    # Check severity
    defect = clipping_defects[0]
    assert defect.severity > 0.3
    assert defect.confidence > 0.7


def test_clipping_lowers_quality(detector, clipped_audio):
    """Clipping should lower overall quality score."""
    audio, sr = clipped_audio
    report = detector.analyze(audio, sr)

    assert report.overall_quality < 0.8
    assert report.needs_restoration


def test_clipping_treatment_recommendation(detector, clipped_audio):
    """Should recommend declipping treatment."""
    audio, sr = clipped_audio
    report = detector.analyze(audio, sr)

    clipping_defects = report.get_defects_by_type(DefectType.CLIPPING)
    if clipping_defects:
        treatment = clipping_defects[0].treatment
        assert treatment is not None
        assert treatment.method == "declip"
        assert "declipper" in treatment.module_path


# ============================================================================
# Noise Detection Tests
# ============================================================================


def test_detect_broadband_noise(detector, noisy_audio):
    """Should detect broadband noise."""
    audio, sr = noisy_audio
    report = detector.analyze(audio, sr)

    noise_defects = report.get_defects_by_type(DefectType.BROADBAND_NOISE)
    assert len(noise_defects) > 0

    defect = noise_defects[0]
    assert defect.severity > 0.2
    assert "snr_db" in defect.metrics


def test_noise_treatment_recommendation(detector, noisy_audio):
    """Should recommend denoising treatment."""
    audio, sr = noisy_audio
    report = detector.analyze(audio, sr)

    noise_defects = report.get_defects_by_type(DefectType.BROADBAND_NOISE)
    if noise_defects:
        treatment = noise_defects[0].treatment
        assert treatment is not None
        assert treatment.method == "denoise"
        assert "denoiser" in treatment.module_path


# ============================================================================
# Hum Detection Tests
# ============================================================================


def test_detect_hum(detector, audio_with_hum):
    """Should detect electrical hum."""
    audio, sr = audio_with_hum
    report = detector.analyze(audio, sr)

    hum_defects = report.get_defects_by_type(DefectType.HUM)
    assert len(hum_defects) > 0

    defect = hum_defects[0]
    assert defect.severity > 0.1
    assert "hum_frequency" in defect.metrics
    assert defect.metrics["hum_frequency"] == 60.0


def test_hum_treatment_recommendation(detector, audio_with_hum):
    """Should recommend hum removal."""
    audio, sr = audio_with_hum
    report = detector.analyze(audio, sr)

    hum_defects = report.get_defects_by_type(DefectType.HUM)
    if hum_defects:
        treatment = hum_defects[0].treatment
        assert treatment is not None
        assert treatment.method == "dehum"
        assert "frequencies" in treatment.params


# ============================================================================
# DC Offset Detection Tests
# ============================================================================


def test_detect_dc_offset(detector, audio_with_dc_offset):
    """Should detect DC offset."""
    audio, sr = audio_with_dc_offset
    report = detector.analyze(audio, sr)

    dc_defects = report.get_defects_by_type(DefectType.DC_OFFSET)
    assert len(dc_defects) > 0

    defect = dc_defects[0]
    assert defect.severity > 0.1
    assert abs(defect.metrics["dc_offset"] - 0.05) < 0.01


def test_dc_offset_treatment_recommendation(detector, audio_with_dc_offset):
    """Should recommend DC offset removal."""
    audio, sr = audio_with_dc_offset
    report = detector.analyze(audio, sr)

    dc_defects = report.get_defects_by_type(DefectType.DC_OFFSET)
    if dc_defects:
        treatment = dc_defects[0].treatment
        assert treatment is not None
        assert treatment.priority == 1  # DC offset is high priority


# ============================================================================
# Stereo Imbalance Tests
# ============================================================================


def test_detect_stereo_imbalance(detector, stereo_imbalanced_audio):
    """Should detect stereo channel imbalance."""
    audio, sr = stereo_imbalanced_audio
    report = detector.analyze(audio, sr)

    imbalance_defects = report.get_defects_by_type(DefectType.STEREO_IMBALANCE)
    assert len(imbalance_defects) > 0

    defect = imbalance_defects[0]
    assert defect.severity > 0.2
    assert "imbalance_db" in defect.metrics
    assert abs(defect.metrics["imbalance_db"]) > 5.0  # Significant imbalance


# ============================================================================
# Report Functionality Tests
# ============================================================================


def test_report_structure(detector, noisy_audio):
    """Test that report has all expected fields."""
    audio, sr = noisy_audio
    report = detector.analyze(audio, sr)

    assert hasattr(report, "defects")
    assert hasattr(report, "overall_quality")
    assert hasattr(report, "needs_restoration")
    assert hasattr(report, "recommended_treatments")
    assert hasattr(report, "audio_duration")
    assert hasattr(report, "sample_rate")
    assert hasattr(report, "analysis_time")


def test_report_to_dict(detector, noisy_audio):
    """Test report serialization to dict."""
    audio, sr = noisy_audio
    report = detector.analyze(audio, sr)

    report_dict = report.to_dict()

    assert "defects" in report_dict
    assert "summary" in report_dict
    assert "recommended_treatments" in report_dict
    assert "metadata" in report_dict


def test_get_critical_defects(detector, clipped_audio):
    """Test filtering critical defects."""
    audio, sr = clipped_audio
    report = detector.analyze(audio, sr)

    critical = report.get_critical_defects()

    # All returned defects should be critical or severe
    for defect in critical:
        assert defect.severity_level in (SeverityLevel.CRITICAL, SeverityLevel.SEVERE)


# ============================================================================
# Treatment Priority Tests
# ============================================================================


def test_treatment_priority_ordering(detector, clipped_audio):
    """Treatments should be ordered by priority."""
    audio, sr = clipped_audio
    report = detector.analyze(audio, sr)

    if len(report.recommended_treatments) > 1:
        priorities = [t.priority for t in report.recommended_treatments]
        assert priorities == sorted(priorities)  # Should be in ascending order


# ============================================================================
# Quick Scan Tests
# ============================================================================


def test_quick_scan_clean_audio(detector, clean_audio):
    """Quick scan on clean audio."""
    audio, sr = clean_audio
    result = detector.quick_scan(audio, sr)

    assert "has_defects" in result
    assert "quality_score" in result
    assert result["quality_score"] > 0.9


def test_quick_scan_defective_audio(detector, clipped_audio):
    """Quick scan on defective audio."""
    audio, sr = clipped_audio
    result = detector.quick_scan(audio, sr)

    assert result["has_defects"] == True
    assert result["quality_score"] < 0.9


# ============================================================================
# Performance Tests
# ============================================================================


def test_analysis_completes_reasonably_fast(detector, clean_audio):
    """Analysis should complete in reasonable time."""
    audio, sr = clean_audio
    report = detector.analyze(audio, sr)

    # 1 second of audio should analyze in < 5 seconds
    assert report.analysis_time < 5.0


def test_detector_listing(detector):
    """Test detector listing functionality."""
    detectors = detector.list_detectors()

    assert len(detectors) > 0
    assert "clipping_detector" in detectors
    assert "noise_detector" in detectors
    assert "hum_detector" in detectors


def test_treatment_recommender_covers_all_legacy_defect_types():
    """Der Legacy-Treatment-Recommender muss alle legacy DefectType-Werte abdecken."""
    from backend.defect_detection.base import LEGACY_TO_CORE_DEFECT_TYPE

    recommender = TreatmentRecommender()
    missing = [defect_type.name for defect_type in DefectType if defect_type not in recommender.treatment_map]

    assert not missing, f"Fehlende Treatment-Mappings: {', '.join(missing)}"
    assert set(recommender.treatment_map.keys()) == set(LEGACY_TO_CORE_DEFECT_TYPE.keys())


def test_treatment_recommender_covers_all_core_defect_types():
    """Der Recommender muss den vollständigen Core-Defektkatalog abdecken."""
    from backend.core.defect_scanner import DefectType as CoreDefectType

    recommender = TreatmentRecommender()
    assert set(recommender.core_treatment_map.keys()) == set(CoreDefectType)


def test_treatment_recommender_productive_for_all_core_defects_both_modes():
    """Für jeden bekannten Core-Defekt muss in beiden Modi eine produktive Empfehlung existieren."""
    from backend.core.defect_scanner import DefectType as CoreDefectType

    recommender = TreatmentRecommender()

    for core_defect_type in CoreDefectType:
        defect = DefectInstance(
            type=DefectType.SPECTRAL_ARTIFACTS,
            severity=0.7,
            confidence=0.9,
            severity_level=SeverityLevel.SEVERE,
            metrics={"core_defect_type": core_defect_type.value},
            description=f"core defect {core_defect_type.value}",
        )

        restoration = recommender.recommend(defect, mode="restoration")
        studio = recommender.recommend(defect, mode="studio2026")

        assert restoration.method != "manual_inspection"
        assert studio.method != "manual_inspection"
        assert restoration.module_path
        assert studio.module_path
        assert restoration.params.get("canonical_core_defect_type") == core_defect_type.value
        assert studio.params.get("canonical_core_defect_type") == core_defect_type.value
        assert restoration.params.get("canonical_mode") == "restoration"
        assert studio.params.get("canonical_mode") == "studio2026"


def test_treatment_recommender_includes_canonical_phase_hints():
    """Empfehlungen sollen die kanonischen Core-Phasen als Hints mitgeben."""
    from backend.core.defect_phase_mapper import DefectPhaseMapper
    from backend.core.defect_scanner import DefectType as CoreDefectType
    from backend.defect_detection.base import to_core_defect_type

    recommender = TreatmentRecommender()
    mapper = DefectPhaseMapper()

    defect = DefectInstance(
        type=DefectType.CLIPPING,
        severity=0.8,
        confidence=0.9,
        severity_level=SeverityLevel.SEVERE,
        metrics={},
        description="test clipping",
    )
    treatment = recommender.recommend(defect)
    expected_phases = mapper.get_primary_phases(CoreDefectType.CLIPPING, mode="studio2026")

    assert treatment.params["canonical_mode"] == "studio2026"
    assert treatment.params["canonical_primary_phase"] == expected_phases[0]
    assert treatment.params["canonical_primary_phases"] == expected_phases
    assert to_core_defect_type(DefectType.CLIPPING) == CoreDefectType.CLIPPING


def test_treatment_recommender_includes_restoration_mode_hints():
    """Empfehlungen müssen im Restoration-Modus den richtigen Kanon-Hint tragen."""
    recommender = TreatmentRecommender()

    defect = DefectInstance(
        type=DefectType.CLIPPING,
        severity=0.8,
        confidence=0.9,
        severity_level=SeverityLevel.SEVERE,
        metrics={},
        description="test clipping",
    )
    treatment = recommender.recommend(defect, mode="Restoration")

    assert treatment.params["canonical_mode"] == "restoration"
    assert "canonical_primary_phase" in treatment.params
    assert "canonical_primary_phases" in treatment.params


def test_unified_detector_forwards_mode_to_recommender(clipped_audio, monkeypatch):
    """Der UnifiedDefectDetector muss den angeforderten Modus an den Recommender durchreichen."""
    from backend.defect_detection.base import TreatmentRecommendation

    detector = UnifiedDefectDetector()
    modes: list[str | None] = []

    def _mock_recommend(defect, mode=None):
        modes.append(mode)
        return TreatmentRecommendation(
            method="mock",
            module_path="mock.module",
            params={"mode": mode},
            priority=1,
            expected_improvement=0.0,
            side_effects=[],
            requires_manual_check=False,
        )

    def _mock_recommend_batch(defects, mode=None):
        return [_mock_recommend(defects[0], mode=mode)] if defects else []

    assert detector.treatment_recommender is not None
    monkeypatch.setattr(detector.treatment_recommender, "recommend", _mock_recommend)
    monkeypatch.setattr(detector.treatment_recommender, "recommend_batch", _mock_recommend_batch)

    audio, sr = clipped_audio
    detector.analyze(audio, sr, mode="Studio 2026")

    assert modes
    assert all(mode == "Studio 2026" for mode in modes)


def test_mode_dosing_restoration_is_more_conservative_for_noise():
    """Restoration muss bei gleicher Defektlage konservativer dosieren als Studio 2026."""
    recommender = TreatmentRecommender()

    defect = DefectInstance(
        type=DefectType.BROADBAND_NOISE,
        severity=0.8,
        confidence=0.9,
        severity_level=SeverityLevel.SEVERE,
        metrics={},
        description="test noise",
    )

    studio = recommender.recommend(defect, mode="studio2026")
    restoration = recommender.recommend(defect, mode="restoration")

    assert restoration.params["reduction_db"] < studio.params["reduction_db"]
    assert restoration.params["sensitivity"] < studio.params["sensitivity"]


def test_mode_dosing_restoration_uses_fewer_clipping_iterations():
    """Restoration soll bei Clipping-Reparatur weniger aggressiv iterieren als Studio 2026."""
    recommender = TreatmentRecommender()

    defect = DefectInstance(
        type=DefectType.CLIPPING,
        severity=0.9,
        confidence=0.9,
        severity_level=SeverityLevel.CRITICAL,
        metrics={},
        description="test clipping",
    )

    studio = recommender.recommend(defect, mode="Studio 2026")
    restoration = recommender.recommend(defect, mode="Restoration")

    assert restoration.params["iterations"] <= studio.params["iterations"]
    assert restoration.params["strength"] <= studio.params["strength"]


def test_unified_detector_core_bridge_adds_mapped_defect(clipped_audio, monkeypatch):
    """Core-Bridge soll zusätzliche Core-Defekte in Legacy-Typen rückprojizieren."""
    from backend.core.defect_scanner import DefectAnalysisResult, DefectScore, MaterialType
    from backend.core.defect_scanner import DefectType as CoreDefectType

    detector = UnifiedDefectDetector(use_core_scanner_bridge=True)

    def _mock_scan(
        audio, sample_rate=None, material_type=None, progress_callback=None, file_ext="", forensic_medium_result=None
    ):
        del audio, sample_rate, material_type, progress_callback, file_ext, forensic_medium_result
        return DefectAnalysisResult(
            material_type=MaterialType.UNKNOWN,
            scores={
                CoreDefectType.WOW: DefectScore(
                    defect_type=CoreDefectType.WOW,
                    severity=0.72,
                    confidence=0.9,
                    locations=[(0.1, 0.5)],
                    metadata={},
                )
            },
            analysis_time_seconds=0.01,
            sample_rate=48000,
            duration_seconds=1.0,
        )

    assert detector.core_defect_scanner is not None
    monkeypatch.setattr(detector.core_defect_scanner, "scan", _mock_scan)

    audio, sr = clipped_audio
    report = detector.analyze(audio, sr)

    phase_defects = report.get_defects_by_type(DefectType.PHASE_ISSUES)
    assert phase_defects, "Core-WOW muss auf Legacy-PHASE_ISSUES abgebildet werden"
    assert any(str(d.detected_by).startswith("core_defect_scanner_bridge:") for d in phase_defects)


def test_unified_detector_core_bridge_respects_thresholds(clipped_audio, monkeypatch):
    """Core-Bridge darf Defekte unterhalb der Schwellwerte nicht in den Report aufnehmen."""
    from backend.core.defect_scanner import DefectAnalysisResult, DefectScore, MaterialType
    from backend.core.defect_scanner import DefectType as CoreDefectType

    detector = UnifiedDefectDetector(use_core_scanner_bridge=True)

    def _mock_scan(
        audio, sample_rate=None, material_type=None, progress_callback=None, file_ext="", forensic_medium_result=None
    ):
        del audio, sample_rate, material_type, progress_callback, file_ext, forensic_medium_result
        return DefectAnalysisResult(
            material_type=MaterialType.UNKNOWN,
            scores={
                CoreDefectType.WOW: DefectScore(
                    defect_type=CoreDefectType.WOW,
                    severity=0.2,
                    confidence=0.4,
                    locations=[(0.1, 0.5)],
                    metadata={},
                )
            },
            analysis_time_seconds=0.01,
            sample_rate=48000,
            duration_seconds=1.0,
        )

    assert detector.core_defect_scanner is not None
    monkeypatch.setattr(detector.core_defect_scanner, "scan", _mock_scan)

    audio, sr = clipped_audio
    report = detector.analyze(audio, sr)

    phase_defects = [
        d
        for d in report.get_defects_by_type(DefectType.PHASE_ISSUES)
        if str(d.detected_by).startswith("core_defect_scanner_bridge")
    ]
    assert not phase_defects


def test_unified_detector_core_bridge_keeps_core_defect_differentiation(clipped_audio, monkeypatch):
    """Core-Bridge darf verschiedene Core-Defekte derselben Legacy-Familie nicht zusammenklappen."""
    from backend.core.defect_scanner import DefectAnalysisResult, DefectScore, MaterialType
    from backend.core.defect_scanner import DefectType as CoreDefectType

    detector = UnifiedDefectDetector(use_core_scanner_bridge=True)

    def _mock_scan(
        audio, sample_rate=None, material_type=None, progress_callback=None, file_ext="", forensic_medium_result=None
    ):
        del audio, sample_rate, material_type, progress_callback, file_ext, forensic_medium_result
        return DefectAnalysisResult(
            material_type=MaterialType.UNKNOWN,
            scores={
                CoreDefectType.WOW: DefectScore(
                    defect_type=CoreDefectType.WOW,
                    severity=0.92,
                    confidence=0.95,
                    locations=[(0.1, 0.5)],
                    metadata={},
                ),
                CoreDefectType.FLUTTER: DefectScore(
                    defect_type=CoreDefectType.FLUTTER,
                    severity=0.9,
                    confidence=0.95,
                    locations=[(0.6, 0.9)],
                    metadata={},
                ),
            },
            analysis_time_seconds=0.01,
            sample_rate=48000,
            duration_seconds=1.0,
        )

    assert detector.core_defect_scanner is not None
    monkeypatch.setattr(detector.core_defect_scanner, "scan", _mock_scan)

    audio, sr = clipped_audio
    report = detector.analyze(audio, sr, mode="Studio 2026")

    bridge_phase_defects = [
        defect
        for defect in report.get_defects_by_type(DefectType.PHASE_ISSUES)
        if str(defect.detected_by).startswith("core_defect_scanner_bridge")
    ]
    core_types = {str(defect.metrics.get("core_defect_type", "")) for defect in bridge_phase_defects}

    assert "wow" in core_types
    assert "flutter" in core_types


def test_treatment_recommender_uses_core_defect_hint_when_present():
    """Recommender muss bei vorhandenem Core-Hint den präzisen Core-Defekt verwenden."""
    recommender = TreatmentRecommender()

    defect = DefectInstance(
        type=DefectType.PHASE_ISSUES,
        severity=0.8,
        confidence=0.9,
        severity_level=SeverityLevel.SEVERE,
        metrics={"core_defect_type": "wow"},
        description="test phase issues mapped from wow",
    )

    treatment = recommender.recommend(defect, mode="restoration")

    assert treatment.params["canonical_core_defect_type"] == "wow"
    assert treatment.params["canonical_mode"] == "restoration"
    assert treatment.method == "stabilize_transport"


def test_unified_detector_core_bridge_restoration_is_stricter(clipped_audio, monkeypatch):
    """Im Restoration-Modus müssen Core-Bridge-Gates strenger sein als im Studio-Modus."""
    from backend.core.defect_scanner import DefectAnalysisResult, DefectScore, MaterialType
    from backend.core.defect_scanner import DefectType as CoreDefectType

    detector = UnifiedDefectDetector(use_core_scanner_bridge=True)

    def _mock_scan(
        audio, sample_rate=None, material_type=None, progress_callback=None, file_ext="", forensic_medium_result=None
    ):
        del audio, sample_rate, material_type, progress_callback, file_ext, forensic_medium_result
        return DefectAnalysisResult(
            material_type=MaterialType.UNKNOWN,
            scores={
                CoreDefectType.WOW: DefectScore(
                    defect_type=CoreDefectType.WOW,
                    severity=0.68,
                    confidence=0.82,
                    locations=[(0.1, 0.5)],
                    metadata={},
                )
            },
            analysis_time_seconds=0.01,
            sample_rate=48000,
            duration_seconds=1.0,
        )

    assert detector.core_defect_scanner is not None
    monkeypatch.setattr(detector.core_defect_scanner, "scan", _mock_scan)

    audio, sr = clipped_audio
    studio_report = detector.analyze(audio, sr, mode="studio2026")
    restoration_report = detector.analyze(audio, sr, mode="restoration")

    studio_bridge = [
        defect
        for defect in studio_report.get_defects_by_type(DefectType.PHASE_ISSUES)
        if str(defect.detected_by).startswith("core_defect_scanner_bridge")
    ]
    restoration_bridge = [
        defect
        for defect in restoration_report.get_defects_by_type(DefectType.PHASE_ISSUES)
        if str(defect.detected_by).startswith("core_defect_scanner_bridge")
    ]

    assert studio_bridge
    assert not restoration_bridge


def test_mode_consistency_same_defect_set_between_restoration_and_studio(clipped_audio):
    """Bei identischem Input muss die Defektmenge in beiden Modi konsistent bleiben."""
    detector = UnifiedDefectDetector(use_core_scanner_bridge=False)
    audio, sr = clipped_audio

    report_restoration = detector.analyze(audio, sr, mode="Restoration")
    report_studio = detector.analyze(audio, sr, mode="Studio 2026")

    defects_restoration = {defect.type for defect in report_restoration.defects}
    defects_studio = {defect.type for defect in report_studio.defects}

    assert defects_restoration == defects_studio


def test_mode_consistency_treatment_hints_and_dosing(clipped_audio):
    """Treatment-Hints müssen modus-konsistent sein, Dosierung darf sich unterscheiden."""
    detector = UnifiedDefectDetector(use_core_scanner_bridge=False)
    audio, sr = clipped_audio

    report_restoration = detector.analyze(audio, sr, mode="Restoration")
    report_studio = detector.analyze(audio, sr, mode="Studio 2026")

    by_type_restoration = {defect.type: defect for defect in report_restoration.defects if defect.treatment is not None}
    by_type_studio = {defect.type: defect for defect in report_studio.defects if defect.treatment is not None}

    common_types = set(by_type_restoration.keys()) & set(by_type_studio.keys())
    assert common_types

    for defect_type in common_types:
        tr = by_type_restoration[defect_type].treatment
        ts = by_type_studio[defect_type].treatment
        assert tr is not None
        assert ts is not None
        assert tr.params.get("canonical_mode") == "restoration"
        assert ts.params.get("canonical_mode") == "studio2026"
        assert tr.params.get("canonical_primary_phase") == ts.params.get("canonical_primary_phase")


def test_registry_accepts_core_defect_type_lookup():
    """Die Registry muss Legacy- und Core-Enums für Lookups akzeptieren."""
    from backend.core.defect_scanner import DefectType as CoreDefectType
    from backend.defect_detection.detectors import ClippingDetector
    from backend.defect_detection.registry import DefectDetectorRegistry

    registry = DefectDetectorRegistry()
    detector = ClippingDetector()
    registry.register(detector)

    assert registry.get_by_type(DefectType.CLIPPING) == [detector]
    assert registry.get_by_type(CoreDefectType.CLIPPING) == [detector]
    assert CoreDefectType.CLIPPING in registry.list_types()
    assert all(isinstance(defect_type, CoreDefectType) for defect_type in registry.list_types())


def test_detector_init_fails_without_core_mapping(monkeypatch):
    """Detektoren müssen beim Erzeugen scheitern, wenn das Core-Mapping fehlt."""
    from backend.defect_detection.base import LEGACY_TO_CORE_DEFECT_TYPE, DefectType
    from backend.defect_detection.detectors import ClippingDetector

    monkeypatch.delitem(LEGACY_TO_CORE_DEFECT_TYPE, DefectType.CLIPPING, raising=False)

    with pytest.raises(ValueError, match="(?i)kanonisches Core-Mapping"):
        ClippingDetector()
