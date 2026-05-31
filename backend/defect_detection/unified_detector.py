"""
Unified Defect Detector
=======================

Main orchestrator for audio defect detection.
Runs all registered detectors and generates comprehensive report.
"""

import logging
import time
from typing import Any

import numpy as np

from backend.core.defect_scanner import DefectScanner as CoreDefectScanner
from backend.core.defect_scanner import DefectType as CoreDefectType
from backend.defect_detection.base import (
    DefectDetector,
    DefectInstance,
    DefectReport,
    DefectType,
    SeverityLevel,
)

# Import and register all detectors
from backend.defect_detection.detectors import (
    AliasingDetector,
    BroadbandNoiseDetector,
    ClicksDetector,
    ClippingDetector,
    DCOffsetDetector,
    DistortionDetector,
    HFRolloffDetector,
    HumDetector,
    RumbleDetector,
    StereoImbalanceDetector,
)
from backend.defect_detection.registry import DefectDetectorRegistry, get_global_registry
from backend.defect_detection.treatment_recommender import TreatmentRecommender

logger = logging.getLogger(__name__)


class UnifiedDefectDetector:
    """
    Einheitliches Audio-Defekterkennungs-System.

    Orchestrates multiple defect detectors to provide:
    - Comprehensive defect analysis
    - Severity scoring for all defect types
    - Treatment recommendations with priorities
    - Overall quality assessment

    Similar to iZotope RX10's "Repair Assistant".

    Usage:
        detector = UnifiedDefectDetector()
        report = detector.analyze(audio, sr)

        # Review defects
        for defect in report.get_critical_defects():
            logger.info("%s: %s", defect.type.value, defect.description)

        # Apply treatments
        for treatment in report.recommended_treatments:
            logger.info("Priority %s: %s", treatment.priority, treatment.method)
    """

    def __init__(
        self,
        registry: DefectDetectorRegistry | None = None,
        enable_treatments: bool = True,
        default_mode: str = "studio2026",
        use_core_scanner_bridge: bool = True,
        core_bridge_min_severity: float = 0.65,
        core_bridge_min_confidence: float = 0.8,
        user_policy: dict | None = None,
        reference_profile: dict | None = None,
        tontraeger_chain: list | None = None,
        audit_context: dict | None = None,
        custom_tolerances: dict | None = None,
    ):
        """
        Initialisiert kontextbewusste Defekterkennung.
        Schwellenwerte und Toleranzen werden aus User-Policy, Referenz, Tonträgerkette und Audit-Kontext gesetzt.
        """
        self.registry = registry or get_global_registry()
        self.enable_treatments = enable_treatments
        self.treatment_recommender = TreatmentRecommender(default_mode=default_mode) if enable_treatments else None
        self.use_core_scanner_bridge = use_core_scanner_bridge
        self.core_bridge_min_severity = float(np.clip(core_bridge_min_severity, 0.0, 1.0))
        self.core_bridge_min_confidence = float(np.clip(core_bridge_min_confidence, 0.0, 1.0))
        self.core_defect_scanner = CoreDefectScanner(sample_rate=48000) if use_core_scanner_bridge else None
        self.user_policy = user_policy or {}
        self.reference_profile = reference_profile or {}
        self.tontraeger_chain = tontraeger_chain or []
        self.audit_context = audit_context or {}
        self.custom_tolerances = custom_tolerances or {}
        self.detector_tolerances = self._build_tolerances()
        # Register all default detectors if registry is empty
        if len(self.registry.list_names()) == 0:
            self._register_default_detectors()

    def _build_tolerances(self) -> dict:
        """
        Erzeugt kontextbewusste Toleranzen für alle Detektoren.
        SOTA-Weltspitze-Niveau: Maximal robust gegen Fehlalarme bei professionellem Material.
        """
        tolerances = {
            "clipping": self.custom_tolerances.get("clipping", 0.01),
            "broadband_noise": self.custom_tolerances.get("broadband_noise", 0.5),
            "hum": self.custom_tolerances.get("hum", 0.5),
            "stereo_imbalance": self.custom_tolerances.get("stereo_imbalance", 2.0),
            "dc_offset": self.custom_tolerances.get("dc_offset", 0.05),
            "clicks": self.custom_tolerances.get("clicks", 0.3),
            "rumble": self.custom_tolerances.get("rumble", 0.3),
            "distortion": self.custom_tolerances.get("distortion", 0.3),
            "hf_rolloff": self.custom_tolerances.get("hf_rolloff", 0.3),
            "aliasing": self.custom_tolerances.get("aliasing", 0.3),
        }
        # Policy/Referenz/Audit können Toleranzen überschreiben
        for key in tolerances:
            if key in self.user_policy:
                tolerances[key] = self.user_policy[key]
            if key in self.reference_profile:
                tolerances[key] = self.reference_profile[key]
            if key in self.audit_context:
                tolerances[key] = self.audit_context[key]
        return tolerances

    def analyze(
        self,
        audio: np.ndarray,
        sr: int,
        detector_names: list[str] | None = None,
        context: dict | None = None,
        mode: str | None = None,
    ) -> DefectReport:
        """
        Analysiert Audio auf alle Defekte.

        Args:
            audio: Audio array (n_samples,) or (n_samples, n_channels)
            sr: Sample rate
            detector_names: Optional list of specific detectors to run

        Returns:
            Comprehensive defect report with treatments
        """
        start_time = time.time()

        # Kontext zusammenführen
        ctx = context or {}
        tolerances = self.detector_tolerances.copy()
        for key in tolerances:
            if key in ctx:
                tolerances[key] = ctx[key]
        # Get detectors to run
        if detector_names:
            _raw: list[DefectDetector | None] = [self.registry.get(name) for name in detector_names]
            detectors: list[DefectDetector] = [d for d in _raw if d is not None]
        else:
            detectors = self.registry.get_all()
        # Run all detectors mit kontextbewussten Toleranzen
        all_defects: list[DefectInstance] = []
        for detector in detectors:
            try:
                # Toleranz für Detektor bestimmen
                tol = tolerances.get(detector.defect_type.value, None)
                # Fallback: Default-Toleranz, falls None (SOTA-Weltspitze)
                if tol is None:
                    if detector.defect_type.value == "clipping":
                        tol = 0.01
                    elif detector.defect_type.value == "broadband_noise" or detector.defect_type.value == "hum":
                        tol = 0.5
                    elif detector.defect_type.value == "stereo_imbalance":
                        tol = 2.0
                    elif detector.defect_type.value == "dc_offset":
                        tol = 0.05
                    elif (
                        detector.defect_type.value == "clicks"
                        or detector.defect_type.value == "rumble"
                        or detector.defect_type.value == "distortion"
                        or detector.defect_type.value == "hf_rolloff"
                        or detector.defect_type.value == "aliasing"
                    ):
                        tol = 0.3
                    else:
                        tol = 0.3
                defects = detector.detect(audio, sr, tolerance=tol)
                # NaN/Inf-Guard für Toleranzwerte
                if tol is not None and not np.isfinite(tol):
                    tol = 0.3
                all_defects.extend(defects)
            except Exception as e:
                logger.error("Warning: Detector %s failed: %s", detector.name, e)
                continue

        # Optional: erweitert Legacy-Erkennung um kanonische Core-Defekte.
        if self.use_core_scanner_bridge and self.core_defect_scanner is not None:
            bridge_defects = self._scan_with_core_bridge(audio, sr, context=ctx, mode=mode)
            all_defects.extend(self._merge_bridge_defects(all_defects, bridge_defects))

        # Generate treatment recommendations
        recommended_treatments = []
        if self.enable_treatments and self.treatment_recommender:
            for defect in all_defects:
                treatment = self.treatment_recommender.recommend(defect, mode=mode)
                defect.treatment = treatment

            # Get unique treatments sorted by priority
            recommended_treatments = self.treatment_recommender.recommend_batch(all_defects, mode=mode)

        # Calculate summary statistics
        severity_counts = dict.fromkeys(SeverityLevel, 0)
        for defect in all_defects:
            severity_counts[defect.severity_level] += 1

        # Overall quality assessment
        overall_quality = self._calculate_overall_quality(all_defects)

        # Determine if restoration needed
        needs_restoration = (
            severity_counts[SeverityLevel.CRITICAL] > 0
            or severity_counts[SeverityLevel.SEVERE] > 0
            or overall_quality < 0.7
        )

        # Audio metadata
        duration = len(audio) / sr if audio.ndim == 1 else audio.shape[0] / sr
        num_channels = 1 if audio.ndim == 1 else audio.shape[1]

        analysis_time = time.time() - start_time

        return DefectReport(
            defects=all_defects,
            total_defects=len(all_defects),
            critical_count=severity_counts[SeverityLevel.CRITICAL],
            severe_count=severity_counts[SeverityLevel.SEVERE],
            moderate_count=severity_counts[SeverityLevel.MODERATE],
            minor_count=severity_counts[SeverityLevel.MINOR],
            overall_quality=overall_quality,
            needs_restoration=needs_restoration,
            recommended_treatments=recommended_treatments,
            audio_duration=duration,
            sample_rate=sr,
            num_channels=num_channels,
            analysis_time=analysis_time,
        )

    def _core_to_legacy_defect_type(self, core_type: CoreDefectType) -> DefectType:
        """Projiziert kanonische Core-Defekte auf das Legacy-Report-Enum."""
        direct_map: dict[CoreDefectType, DefectType] = {
            CoreDefectType.CLIPPING: DefectType.CLIPPING,
            CoreDefectType.CLICKS: DefectType.CLICKS,
            CoreDefectType.CRACKLE: DefectType.CRACKLE,
            CoreDefectType.HUM: DefectType.HUM,
            CoreDefectType.LOW_FREQ_RUMBLE: DefectType.RUMBLE,
            CoreDefectType.BANDWIDTH_LOSS: DefectType.HF_ROLLOFF,
            CoreDefectType.STEREO_IMBALANCE: DefectType.STEREO_IMBALANCE,
            CoreDefectType.DROPOUTS: DefectType.DROPOUTS,
            CoreDefectType.PHASE_ISSUES: DefectType.PHASE_ISSUES,
            CoreDefectType.DC_OFFSET: DefectType.DC_OFFSET,
            CoreDefectType.ALIASING: DefectType.ALIASING,
            CoreDefectType.HIGH_FREQ_NOISE: DefectType.BROADBAND_NOISE,
            CoreDefectType.DIGITAL_ARTIFACTS: DefectType.SPECTRAL_ARTIFACTS,
            CoreDefectType.COMPRESSION_ARTIFACTS: DefectType.DISTORTION,
        }
        if core_type in direct_map:
            return direct_map[core_type]

        phase_family = {
            CoreDefectType.WOW,
            CoreDefectType.FLUTTER,
            CoreDefectType.PITCH_DRIFT,
            CoreDefectType.AZIMUTH_ERROR,
            CoreDefectType.MULTIBAND_WOW_FLUTTER,
            CoreDefectType.SCRAPE_FLUTTER,
            CoreDefectType.SPEED_CALIBRATION_ERROR,
            CoreDefectType.TRANSPORT_BUMP,
        }
        if core_type in phase_family:
            return DefectType.PHASE_ISSUES

        noisy_family = {
            CoreDefectType.MODULATION_NOISE,
            CoreDefectType.NR_BREATHING_ARTIFACT,
            CoreDefectType.MOTOR_INTERFERENCE,
            CoreDefectType.STICKY_SHED_RESIDUE,
        }
        if core_type in noisy_family:
            return DefectType.BROADBAND_NOISE

        return DefectType.SPECTRAL_ARTIFACTS

    def _scan_with_core_bridge(
        self,
        audio: np.ndarray,
        sr: int,
        context: dict | None = None,
        mode: str | None = None,
    ) -> list[DefectInstance]:
        """Führt den kanonischen Core-Scanner aus und konvertiert Ergebnisse ins Legacy-Format."""
        if self.core_defect_scanner is None:
            return []

        ctx = context or {}
        min_severity = float(np.clip(ctx.get("core_bridge_min_severity", self.core_bridge_min_severity), 0.0, 1.0))
        min_confidence = float(
            np.clip(ctx.get("core_bridge_min_confidence", self.core_bridge_min_confidence), 0.0, 1.0)
        )

        normalized_mode = str(mode or "studio2026").strip().lower().replace("_", " ")
        if normalized_mode in {"restoration", "restore"}:
            min_severity = float(np.clip(min_severity + 0.05, 0.0, 1.0))
            min_confidence = float(np.clip(min_confidence + 0.05, 0.0, 1.0))

        try:
            core_result = self.core_defect_scanner.scan(audio, sample_rate=sr)
        except Exception as e:
            logger.debug("Core-Bridge-Scan fehlgeschlagen, fallback auf Legacy-only: %s", e)
            return []

        bridged: list[DefectInstance] = []
        for core_score in core_result.scores.values():
            severity = float(np.clip(core_score.severity, 0.0, 1.0))
            confidence = float(np.clip(core_score.confidence, 0.0, 1.0))
            if severity < min_severity or confidence < min_confidence:
                continue

            legacy_type = self._core_to_legacy_defect_type(core_score.defect_type)
            if core_score.locations:
                start_time = float(core_score.locations[0][0])
                end_time = float(core_score.locations[-1][1])
            else:
                start_time = None
                end_time = None

            core_metrics: dict[str, Any] = {
                "core_severity": severity,
                "core_confidence": confidence,
                "core_defect_type": core_score.defect_type.value,
            }

            bridged.append(
                DefectInstance(
                    type=legacy_type,
                    severity=severity,
                    confidence=confidence,
                    severity_level=SeverityLevel.from_score(severity),
                    start_time=start_time,
                    end_time=end_time,
                    affected_channels=None,
                    metrics=core_metrics,
                    description=f"Core-Bridge: {core_score.defect_type.value}",
                    detected_by=f"core_defect_scanner_bridge:{core_score.defect_type.value}",
                )
            )

        return bridged

    def _merge_bridge_defects(
        self,
        existing_defects: list[DefectInstance],
        bridge_defects: list[DefectInstance],
    ) -> list[DefectInstance]:
        """Verhindert doppelte Defektfamilien und dämpft Bridge-FPs auf cleanem Material."""
        if not bridge_defects:
            return []

        strongest_by_type: dict[tuple[DefectType, str], float] = {}
        for defect in existing_defects:
            core_key = str(defect.metrics.get("core_defect_type", "legacy"))
            key = (defect.type, core_key)
            strongest_by_type[key] = max(strongest_by_type.get(key, 0.0), float(defect.severity))

        max_existing_severity = max((float(defect.severity) for defect in existing_defects), default=0.0)
        clean_baseline = max_existing_severity < 0.15

        merged: list[DefectInstance] = []
        for bridge_defect in bridge_defects:
            core_key = str(bridge_defect.metrics.get("core_defect_type", "legacy"))
            key = (bridge_defect.type, core_key)
            existing_severity = strongest_by_type.get(key, 0.0)
            corroborated = existing_severity >= 0.2
            very_high_conf = float(bridge_defect.severity) >= 0.9 and float(bridge_defect.confidence) >= 0.9

            conservative_clean_types = {
                DefectType.HF_ROLLOFF,
                DefectType.SPECTRAL_ARTIFACTS,
                DefectType.BROADBAND_NOISE,
                DefectType.PHASE_ISSUES,
            }

            if clean_baseline and not (corroborated or very_high_conf):
                continue
            if clean_baseline and bridge_defect.type in conservative_clean_types and not corroborated:
                continue

            if float(bridge_defect.severity) > existing_severity + 0.05:
                merged.append(bridge_defect)
                strongest_by_type[key] = float(bridge_defect.severity)

        return merged

    def _calculate_overall_quality(self, defects: list[DefectInstance]) -> float:
        """
        Calculate overall audio quality score (0.0 - 1.0).

        Algorithm:
        - Start with perfect quality (1.0)
        - Subtract weighted severity scores
        - Weights: Critical=0.3, Severe=0.2, Moderate=0.1, Minor=0.05
        """
        if not defects:
            return 1.0

        quality = 1.0

        for defect in defects:
            # Weight by severity level
            if defect.severity_level == SeverityLevel.CRITICAL:
                quality -= defect.severity * 0.3
            elif defect.severity_level == SeverityLevel.SEVERE:
                quality -= defect.severity * 0.2
            elif defect.severity_level == SeverityLevel.MODERATE:
                quality -= defect.severity * 0.1
            elif defect.severity_level == SeverityLevel.MINOR:
                quality -= defect.severity * 0.05

        return max(quality, 0.0)

    def _register_default_detectors(self):
        """Registriert all default detectors."""
        default_detectors = [
            ClippingDetector(),
            ClicksDetector(),
            BroadbandNoiseDetector(),
            HumDetector(),
            DistortionDetector(),
            RumbleDetector(),
            HFRolloffDetector(),
            StereoImbalanceDetector(),
            DCOffsetDetector(),
            AliasingDetector(),
        ]

        for detector in default_detectors:
            self.registry.register(detector)

    def list_detectors(self) -> list[str]:
        """Listet alle registrierten Detektor-Namen auf."""
        return self.registry.list_names()

    def quick_scan(self, audio: np.ndarray, sr: int) -> dict:
        """
        Quick scan returning simple summary (faster than full analyze).

        Returns:
            {
                'has_defects': bool,
                'critical_count': int,
                'needs_restoration': bool,
                'quality_score': float,
            }
        """
        # Run only fast detectors
        fast_detectors = ["clipping_detector", "dc_offset_detector", "stereo_imbalance_detector"]
        report = self.analyze(audio, sr, detector_names=fast_detectors)

        return {
            "has_defects": report.total_defects > 0,
            "critical_count": report.critical_count,
            "needs_restoration": report.needs_restoration,
            "quality_score": report.overall_quality,
        }
