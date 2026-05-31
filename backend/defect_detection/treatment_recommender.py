"""
Treatment Recommender
=====================

Recommends treatments for detected audio defects.
Maps defects to DSP modules with optimal parameters.
"""

from typing import Any

import numpy as np

from backend.core.defect_phase_mapper import DefectPhaseMapper
from backend.core.defect_scanner import DefectType as CoreDefectType
from backend.defect_detection.base import (
    DefectInstance,
    DefectType,
    SeverityLevel,
    TreatmentRecommendation,
    assert_legacy_mapping_keys,
    require_core_defect_type,
)


class TreatmentRecommender:
    """
    Recommends treatments for detected audio defects.

    Maps defect type + severity to appropriate DSP module + parameters.
    Similar to iZotope RX10's "Repair Assistant".
    """

    def __init__(self, default_mode: str = "studio2026"):
        self._phase_mapper = DefectPhaseMapper()
        self.default_mode = self._normalize_mode(default_mode)
        # Treatment mappings: defect_type -> (module_path, method_name)
        self.treatment_map = {
            DefectType.CLIPPING: ("dsp.automatic_declipper", "declip"),
            DefectType.CLICKS: ("dsp.automatic_declicker", "declick"),
            DefectType.CRACKLE: ("dsp.automatic_decrackler", "decrackle"),
            DefectType.BROADBAND_NOISE: ("dsp.automatic_denoiser", "denoise"),
            DefectType.HUM: ("dsp.automatic_dehum", "dehum"),
            DefectType.BUZZ: ("dsp.automatic_debuzzer", "debuzz"),
            DefectType.DISTORTION: ("dsp.harmonic_exciter", "reduce_distortion"),
            DefectType.RUMBLE: ("dsp.rumble_filter", "filter_rumble"),
            DefectType.HF_ROLLOFF: ("dsp.bandwidth_extender", "extend_bandwidth"),
            DefectType.SPECTRAL_ARTIFACTS: ("dsp.spectral_repair", "repair_spectral_artifacts"),
            DefectType.STEREO_IMBALANCE: ("dsp.stereo_image_correction", "correct_stereo"),
            DefectType.DROPOUTS: ("dsp.dropout_repair", "repair_dropouts"),
            DefectType.PHASE_ISSUES: ("dsp.phase_correction", "correct_phase"),
            DefectType.DC_OFFSET: ("dsp.classic_filters", "remove_dc_offset"),
            DefectType.ALIASING: ("dsp.aliasing_repair", "repair_aliasing"),
        }
        self.core_treatment_map = self._build_core_treatment_map()
        self._validate_treatment_map_completeness()

    def _build_core_treatment_map(self) -> dict[CoreDefectType, tuple[str, str]]:
        """Erstellt eine vollständige Core-Defekt -> Treatment-Zuordnung."""
        base: dict[CoreDefectType, tuple[str, str]] = {
            CoreDefectType.CLIPPING: ("dsp.automatic_declipper", "declip"),
            CoreDefectType.CLICKS: ("dsp.automatic_declicker", "declick"),
            CoreDefectType.CRACKLE: ("dsp.automatic_decrackler", "decrackle"),
            CoreDefectType.HUM: ("dsp.automatic_dehum", "dehum"),
            CoreDefectType.LOW_FREQ_RUMBLE: ("dsp.rumble_filter", "filter_rumble"),
            CoreDefectType.BANDWIDTH_LOSS: ("dsp.bandwidth_extender", "extend_bandwidth"),
            CoreDefectType.STEREO_IMBALANCE: ("dsp.stereo_image_correction", "correct_stereo"),
            CoreDefectType.DROPOUTS: ("dsp.dropout_repair", "repair_dropouts"),
            CoreDefectType.PHASE_ISSUES: ("dsp.phase_correction", "correct_phase"),
            CoreDefectType.DC_OFFSET: ("dsp.classic_filters", "remove_dc_offset"),
            CoreDefectType.ALIASING: ("dsp.aliasing_repair", "repair_aliasing"),
            CoreDefectType.HIGH_FREQ_NOISE: ("dsp.automatic_denoiser", "denoise"),
            CoreDefectType.DIGITAL_ARTIFACTS: ("dsp.spectral_repair", "repair_spectral_artifacts"),
            CoreDefectType.COMPRESSION_ARTIFACTS: ("dsp.harmonic_exciter", "reduce_distortion"),
        }

        phase_family = {
            CoreDefectType.WOW,
            CoreDefectType.FLUTTER,
            CoreDefectType.PITCH_DRIFT,
            CoreDefectType.AZIMUTH_ERROR,
            CoreDefectType.MULTIBAND_WOW_FLUTTER,
            CoreDefectType.SCRAPE_FLUTTER,
            CoreDefectType.SPEED_CALIBRATION_ERROR,
            CoreDefectType.TRANSPORT_BUMP,
            CoreDefectType.FLUTTER_SPECTRAL_SIDEBANDS,
        }
        noise_family = {
            CoreDefectType.MODULATION_NOISE,
            CoreDefectType.NR_BREATHING_ARTIFACT,
            CoreDefectType.MOTOR_INTERFERENCE,
            CoreDefectType.STICKY_SHED_RESIDUE,
            CoreDefectType.TAPE_HEAD_CLOG,
            CoreDefectType.TAPE_HEAD_LEVEL_DIP,
        }
        spectral_family = {
            CoreDefectType.HEAD_WEAR,
            CoreDefectType.TRANSIENT_SMEARING,
            CoreDefectType.PRE_ECHO,
            CoreDefectType.QUANTIZATION_NOISE,
            CoreDefectType.JITTER_ARTIFACTS,
            CoreDefectType.DYNAMIC_COMPRESSION_EXCESS,
            CoreDefectType.SIBILANCE,
            CoreDefectType.VOCAL_HARSHNESS,
            CoreDefectType.STYLUS_DAMAGE,
            CoreDefectType.INNER_GROOVE_DISTORTION,
            CoreDefectType.INTERMODULATION_DISTORTION,
            CoreDefectType.TAPE_SPLICE_ARTIFACT,
            CoreDefectType.HF_REMANENCE_LOSS,
            CoreDefectType.GENERATION_LOSS,
            CoreDefectType.PROXIMITY_EFFECT_EXCESS,
            CoreDefectType.ROOM_MODE_RESONANCE,
            CoreDefectType.OVERLOAD_DISTORTION,
            CoreDefectType.LACQUER_DISC_DEGRADATION,
            CoreDefectType.REVERB_EXCESS,
            CoreDefectType.PRINT_THROUGH,
            CoreDefectType.GROOVE_ECHO,
            CoreDefectType.CROSSTALK,
            CoreDefectType.SOFT_SATURATION,
            CoreDefectType.DOLBY_NR_MISMATCH,
            CoreDefectType.BIAS_ERROR,
            CoreDefectType.RIAA_CURVE_ERROR,
            CoreDefectType.AMPLITUDE_DRIFT,
        }

        for core_type in phase_family:
            base.setdefault(core_type, ("dsp.phase_correction", "stabilize_transport"))
        for core_type in noise_family:
            base.setdefault(core_type, ("dsp.automatic_denoiser", "denoise"))
        for core_type in spectral_family:
            base.setdefault(core_type, ("dsp.spectral_repair", "repair_spectral_artifacts"))

        for core_type in CoreDefectType:
            base.setdefault(core_type, ("dsp.spectral_repair", "repair_spectral_artifacts"))

        return base

    def _validate_treatment_map_completeness(self) -> None:
        """Validiert, dass alle Legacy-Defekte einen Treatment-Pfad besitzen."""
        assert_legacy_mapping_keys(set(self.treatment_map.keys()), context="TreatmentRecommender")
        missing_core = set(CoreDefectType) - set(self.core_treatment_map.keys())
        if missing_core:
            names = ", ".join(sorted(defect_type.name for defect_type in missing_core))
            raise ValueError(f"TreatmentRecommender: Fehlende Core-Defekte in core_treatment_map: {names}")

    def _resolve_core_defect_type(self, defect: DefectInstance) -> CoreDefectType:
        """Löst den präzisesten verfügbaren Core-Defekt auf."""
        core_defect_type_name = defect.metrics.get("core_defect_type")
        if isinstance(core_defect_type_name, str):
            with_core_hint = core_defect_type_name.strip().lower()
            for candidate in CoreDefectType:
                if candidate.value == with_core_hint:
                    return candidate

        return require_core_defect_type(defect.type, context="TreatmentRecommender")

    def _normalize_mode(self, mode: str | None) -> str:
        """Normalisiert externe Modusbezeichner auf den internen Kanon."""
        if mode is None:
            return self.default_mode if hasattr(self, "default_mode") else "studio2026"

        normalized = str(mode).strip().lower().replace("_", " ")
        if normalized in {"restoration", "restore"}:
            return "restoration"
        if normalized in {"studio2026", "studio 2026", "studio"}:
            return "studio2026"
        raise ValueError(f"Ungültiger Modus {mode!r}. Erlaubt sind Restoration oder Studio 2026")

    def recommend(self, defect: DefectInstance, mode: str | None = None) -> TreatmentRecommendation:
        """
        Recommend treatment for a defect.

        Args:
            defect: Detected defect instance

        Returns:
            Treatment recommendation with method, params, priority
        """
        core_defect_type = self._resolve_core_defect_type(defect)
        module_path, method = self.core_treatment_map.get(
            core_defect_type,
            self.treatment_map.get(defect.type, ("", "")),
        )
        if not module_path or not method:
            return self._create_no_treatment(defect)
        normalized_mode = self._normalize_mode(mode)

        # Generate parameters based on defect severity
        params = self._generate_params(defect)
        params.update(self._generate_core_params(core_defect_type, defect))
        params = self._apply_mode_dosing(
            params,
            defect_type=defect.type,
            core_defect_type=core_defect_type,
            mode=normalized_mode,
        )
        params.update(self._canonical_phase_params(defect, mode=normalized_mode))

        # Determine priority (1=highest, 5=lowest)
        priority = self._calculate_priority(defect)

        # Estimate improvement
        expected_improvement = self._estimate_improvement(defect)

        # List potential side effects
        side_effects = self._list_side_effects(defect.type, defect.severity)

        # Check if manual verification needed
        requires_manual = self._requires_manual_check(defect)

        return TreatmentRecommendation(
            method=method,
            module_path=module_path,
            params=params,
            priority=priority,
            expected_improvement=expected_improvement,
            side_effects=side_effects,
            requires_manual_check=requires_manual,
        )

    def _generate_core_params(self, core_defect_type: CoreDefectType, defect: DefectInstance) -> dict[str, Any]:
        """Ergänzt Defektparameter mit Core-defektspezifischer Feindosierung."""
        severity = float(np.clip(defect.severity, 0.0, 1.0))

        if core_defect_type in {
            CoreDefectType.WOW,
            CoreDefectType.FLUTTER,
            CoreDefectType.MULTIBAND_WOW_FLUTTER,
            CoreDefectType.SCRAPE_FLUTTER,
            CoreDefectType.SPEED_CALIBRATION_ERROR,
            CoreDefectType.TRANSPORT_BUMP,
            CoreDefectType.PITCH_DRIFT,
        }:
            return {
                "phase_alignment_strength": min(0.35 + severity * 0.55, 1.0),
                "transport_stabilization": True,
            }

        if core_defect_type in {
            CoreDefectType.NR_BREATHING_ARTIFACT,
            CoreDefectType.MODULATION_NOISE,
            CoreDefectType.STICKY_SHED_RESIDUE,
        }:
            return {
                "reduction_db": min(5.0 + severity * 15.0, 18.0),
                "sensitivity": min(0.45 + severity * 0.35, 0.9),
            }

        if core_defect_type in {
            CoreDefectType.PRE_ECHO,
            CoreDefectType.DIGITAL_ARTIFACTS,
            CoreDefectType.JITTER_ARTIFACTS,
            CoreDefectType.QUANTIZATION_NOISE,
        }:
            return {
                "repair_strength": min(0.45 + severity * 0.5, 0.95),
                "spectral_repair": True,
            }

        if core_defect_type in {
            CoreDefectType.VOCAL_HARSHNESS,
            CoreDefectType.SIBILANCE,
            CoreDefectType.PROXIMITY_EFFECT_EXCESS,
        }:
            return {
                "repair_strength": min(0.40 + severity * 0.45, 0.9),
                "preserve_transients": True,
            }

        return {}

    def recommend_batch(self, defects: list[DefectInstance], mode: str | None = None) -> list[TreatmentRecommendation]:
        """
        Recommend treatments for multiple defects.

        Returns treatments sorted by priority.
        """
        treatments = [self.recommend(d, mode=mode) for d in defects]

        # Sort by priority (lower number = higher priority)
        treatments.sort(key=lambda t: t.priority)

        # Remove duplicates, aber Core-Defektidentität beibehalten.
        seen_methods = set()
        unique_treatments = []
        for t in treatments:
            dedupe_key = (t.method, str(t.params.get("canonical_core_defect_type", "")))
            if dedupe_key not in seen_methods:
                unique_treatments.append(t)
                seen_methods.add(dedupe_key)

        return unique_treatments

    def _generate_params(self, defect: DefectInstance) -> dict[str, Any]:
        """Generiert treatment parameters based on defect severity."""
        severity = defect.severity

        if defect.type == DefectType.CLIPPING:
            return {
                "strength": min(0.3 + severity * 0.7, 1.0),  # 0.3 - 1.0
                "iterations": int(1 + severity * 4),  # 1 - 5
                "window_size": 2048 if severity < 0.5 else 4096,
            }

        elif defect.type == DefectType.CLICKS:
            return {
                "threshold": max(0.5 - severity * 0.4, 0.1),  # 0.5 - 0.1 (lower = more aggressive)
                "window_size": int(32 + severity * 96),  # 32 - 128 samples
                "sensitivity": min(0.5 + severity * 0.5, 1.0),
            }

        elif defect.type == DefectType.CRACKLE:
            return {
                "threshold": max(0.6 - severity * 0.4, 0.2),
                "window_size": 64,
                "attack": 1.0,  # ms
                "release": 10.0,  # ms
            }

        elif defect.type == DefectType.BROADBAND_NOISE:
            return {
                "reduction_db": min(6.0 + severity * 24.0, 30.0),  # 6-30 dB
                "noise_floor_db": -60.0 + severity * 20.0,  # -60 to -40 dB
                "sensitivity": min(0.5 + severity * 0.5, 1.0),
            }

        elif defect.type == DefectType.HUM:
            freqs = defect.metrics.get("frequencies", [50.0, 60.0])
            return {
                "frequencies": freqs,
                "q_factor": 30.0,  # Narrow notch
                "num_harmonics": int(2 + severity * 6),  # 2-8 harmonics
            }

        elif defect.type == DefectType.BUZZ:
            return {
                "frequency_range": (80, 300),
                "reduction_db": min(6.0 + severity * 18.0, 24.0),
            }

        elif defect.type == DefectType.RUMBLE:
            return {
                "cutoff_hz": min(40.0 + severity * 60.0, 100.0),  # 40-100 Hz
                "order": int(2 + severity * 4),  # 2-6
                "filter_type": "highpass",
            }

        elif defect.type == DefectType.HF_ROLLOFF:
            rolloff_freq = defect.metrics.get("rolloff_frequency", 12000)
            return {
                "target_frequency": min(rolloff_freq * 1.5, 20000),
                "gain_db": min(3.0 + severity * 9.0, 12.0),
                "slope": "gentle",
            }

        elif defect.type == DefectType.SPECTRAL_ARTIFACTS:
            return {
                "repair_strength": min(0.4 + severity * 0.6, 1.0),
                "smoothing": min(0.2 + severity * 0.4, 0.8),
                "preserve_transients": True,
            }

        elif defect.type == DefectType.STEREO_IMBALANCE:
            imbalance = defect.metrics.get("imbalance_db", 0.0)
            return {
                "correction_db": imbalance,
                "affected_channel": defect.affected_channels[0] if defect.affected_channels else 0,
            }

        elif defect.type == DefectType.DROPOUTS:
            return {
                "repair_strength": min(0.5 + severity * 0.5, 1.0),
                "interpolation_ms": int(8 + severity * 32),
                "crossfade_ms": int(4 + severity * 16),
            }

        elif defect.type == DefectType.PHASE_ISSUES:
            return {
                "phase_alignment_strength": min(0.4 + severity * 0.6, 1.0),
                "stereo_coherence_target": 0.92,
            }

        elif defect.type == DefectType.DC_OFFSET:
            offset = defect.metrics.get("dc_offset", 0.0)
            return {
                "offset": offset,
                "use_highpass": True,
                "highpass_cutoff": 5.0,  # Hz
            }

        elif defect.type == DefectType.ALIASING:
            return {
                "repair_strength": min(0.45 + severity * 0.45, 1.0),
                "lowpass_cutoff_ratio": 0.48,
                "spectral_repair": True,
            }

        else:
            return {}

    def _apply_mode_dosing(
        self,
        params: dict[str, Any],
        defect_type: DefectType,
        core_defect_type: CoreDefectType,
        mode: str,
    ) -> dict[str, Any]:
        """Wendet modusabhängige Dosierung auf numerische Behandlungsparameter an."""
        if not params:
            return params

        adjusted = dict(params)

        restoration_base: dict[str, float] = {
            "strength": 0.85,
            "repair_strength": 0.85,
            "sensitivity": 0.90,
            "reduction_db": 0.85,
            "gain_db": 0.80,
            "phase_alignment_strength": 0.90,
        }
        studio_base: dict[str, float] = {
            "strength": 1.05,
            "repair_strength": 1.05,
            "sensitivity": 1.05,
            "reduction_db": 1.05,
            "gain_db": 1.10,
            "phase_alignment_strength": 1.05,
        }

        restoration_profiles: dict[DefectType, dict[str, float]] = {
            DefectType.BROADBAND_NOISE: {"reduction_db": 0.80, "sensitivity": 0.88},
            DefectType.BUZZ: {"reduction_db": 0.82},
            DefectType.HF_ROLLOFF: {"gain_db": 0.75},
            DefectType.SPECTRAL_ARTIFACTS: {"repair_strength": 0.82},
            DefectType.CLIPPING: {"strength": 0.82},
            DefectType.PHASE_ISSUES: {"phase_alignment_strength": 0.88},
            DefectType.ALIASING: {"repair_strength": 0.84},
        }
        studio_profiles: dict[DefectType, dict[str, float]] = {
            DefectType.BROADBAND_NOISE: {"reduction_db": 1.08, "sensitivity": 1.08},
            DefectType.BUZZ: {"reduction_db": 1.07},
            DefectType.HF_ROLLOFF: {"gain_db": 1.15},
            DefectType.SPECTRAL_ARTIFACTS: {"repair_strength": 1.08},
            DefectType.CLIPPING: {"strength": 1.08},
            DefectType.PHASE_ISSUES: {"phase_alignment_strength": 1.07},
            DefectType.ALIASING: {"repair_strength": 1.06},
        }

        restoration_core_profiles: dict[CoreDefectType, dict[str, float]] = {
            CoreDefectType.WOW: {"phase_alignment_strength": 0.82},
            CoreDefectType.FLUTTER: {"phase_alignment_strength": 0.82},
            CoreDefectType.MULTIBAND_WOW_FLUTTER: {"phase_alignment_strength": 0.80},
            CoreDefectType.NR_BREATHING_ARTIFACT: {"reduction_db": 0.78, "sensitivity": 0.85},
            CoreDefectType.VOCAL_HARSHNESS: {"repair_strength": 0.80},
            CoreDefectType.SIBILANCE: {"repair_strength": 0.80},
            CoreDefectType.PRE_ECHO: {"repair_strength": 0.82},
            CoreDefectType.ROOM_MODE_RESONANCE: {"repair_strength": 0.84},
        }
        studio_core_profiles: dict[CoreDefectType, dict[str, float]] = {
            CoreDefectType.WOW: {"phase_alignment_strength": 1.10},
            CoreDefectType.FLUTTER: {"phase_alignment_strength": 1.10},
            CoreDefectType.MULTIBAND_WOW_FLUTTER: {"phase_alignment_strength": 1.10},
            CoreDefectType.NR_BREATHING_ARTIFACT: {"reduction_db": 1.10, "sensitivity": 1.10},
            CoreDefectType.VOCAL_HARSHNESS: {"repair_strength": 1.10},
            CoreDefectType.SIBILANCE: {"repair_strength": 1.10},
            CoreDefectType.PRE_ECHO: {"repair_strength": 1.08},
            CoreDefectType.ROOM_MODE_RESONANCE: {"repair_strength": 1.08},
        }

        if mode == "restoration":
            # Restoration bleibt konservativ und artefaktarm.
            multipliers = dict(restoration_base)
            multipliers.update(restoration_profiles.get(defect_type, {}))
            multipliers.update(restoration_core_profiles.get(core_defect_type, {}))

            for key, multiplier in multipliers.items():
                value = adjusted.get(key)
                if isinstance(value, (int, float)):
                    adjusted[key] = float(value) * multiplier

            iterations = adjusted.get("iterations")
            if isinstance(iterations, int):
                adjusted["iterations"] = max(1, int(round(iterations * 0.8)))

        else:
            # Studio 2026 darf etwas entschlossener dosieren, bleibt aber gedeckelt.
            multipliers = dict(studio_base)
            multipliers.update(studio_profiles.get(defect_type, {}))
            multipliers.update(studio_core_profiles.get(core_defect_type, {}))

            for key, multiplier in multipliers.items():
                value = adjusted.get(key)
                if isinstance(value, (int, float)):
                    adjusted[key] = float(value) * multiplier

            iterations = adjusted.get("iterations")
            if isinstance(iterations, int):
                adjusted["iterations"] = max(1, int(round(iterations * 1.1)))

        # Zentrale Caps gegen Übersteuerung von Legacy-Parametern.
        capped_keys = {
            "strength": 1.0,
            "repair_strength": 1.0,
            "sensitivity": 1.0,
            "phase_alignment_strength": 1.0,
        }
        for key, max_value in capped_keys.items():
            value = adjusted.get(key)
            if isinstance(value, (int, float)):
                adjusted[key] = float(max(0.0, min(value, max_value)))

        return adjusted

    def _canonical_phase_params(self, defect: DefectInstance, mode: str) -> dict[str, Any]:
        """Hängt kanonische Phase-Hints aus dem Core-Mapper an die Legacy-Empfehlung an."""
        core_defect_type: CoreDefectType | None = None
        core_defect_type_name = defect.metrics.get("core_defect_type")
        if isinstance(core_defect_type_name, str):
            with_core_hint = core_defect_type_name.strip().lower()
            for candidate in CoreDefectType:
                if candidate.value == with_core_hint:
                    core_defect_type = candidate
                    break

        if core_defect_type is None:
            core_defect_type = require_core_defect_type(defect.type, context="TreatmentRecommender")

        primary_phases = self._phase_mapper.get_primary_phases(core_defect_type, mode=mode)
        if not primary_phases:
            primary_phases = self._phase_mapper.get_all_phases(core_defect_type, mode=mode)

        if not primary_phases:
            return {}

        return {
            "canonical_mode": mode,
            "canonical_core_defect_type": core_defect_type.value,
            "canonical_primary_phase": primary_phases[0],
            "canonical_primary_phases": primary_phases,
        }

    def _calculate_priority(self, defect: DefectInstance) -> int:
        """
        Calculate treatment priority (1=highest, 5=lowest).

        Priority rules:
        - Critical defects: priority 1
        - Severe defects: priority 2
        - Moderate defects: priority 3
        - Minor defects: priority 4
        - Clipping/Distortion always high priority regardless of severity
        """
        # Clipping and distortion are always high priority
        if defect.type in (DefectType.CLIPPING, DefectType.DISTORTION):
            return 1

        # DC offset should be fixed early
        if defect.type == DefectType.DC_OFFSET:
            return 1

        # Phase issues and aliasing are early-stage structural defects
        if defect.type in (DefectType.PHASE_ISSUES, DefectType.ALIASING, DefectType.DROPOUTS):
            return 2

        # Otherwise, prioritize by severity
        if defect.severity_level == SeverityLevel.CRITICAL:
            return 1
        elif defect.severity_level == SeverityLevel.SEVERE:
            return 2
        elif defect.severity_level == SeverityLevel.MODERATE:
            return 3
        elif defect.severity_level == SeverityLevel.MINOR:
            return 4
        else:
            return 5

    def _estimate_improvement(self, defect: DefectInstance) -> float:
        """
        Schätzt expected improvement (0.0 - 1.0) from treatment.

        Based on:
        - Defect type (some are easier to fix)
        - Severity (mild defects easier to fix completely)
        - Confidence (higher confidence = better estimate)
        """
        # Base improvement by defect type
        type_improvements = {
            DefectType.CLIPPING: 0.7,
            DefectType.CLICKS: 0.9,
            DefectType.CRACKLE: 0.8,
            DefectType.BROADBAND_NOISE: 0.75,
            DefectType.HUM: 0.95,
            DefectType.BUZZ: 0.85,
            DefectType.DISTORTION: 0.5,
            DefectType.RUMBLE: 0.9,
            DefectType.HF_ROLLOFF: 0.6,
            DefectType.SPECTRAL_ARTIFACTS: 0.8,
            DefectType.STEREO_IMBALANCE: 0.95,
            DefectType.DROPOUTS: 0.9,
            DefectType.PHASE_ISSUES: 0.85,
            DefectType.DC_OFFSET: 1.0,
            DefectType.ALIASING: 0.82,
        }

        base_improvement = type_improvements.get(defect.type, 0.5)

        # Adjust for severity (severe defects harder to fix completely)
        severity_factor = 1.0 - (defect.severity * 0.3)

        # Adjust for confidence
        confidence_factor = 0.7 + (defect.confidence * 0.3)

        improvement = base_improvement * severity_factor * confidence_factor

        return min(improvement, 1.0)

    def _list_side_effects(self, defect_type: DefectType, severity: float) -> list[str]:
        """Listet auf: potential side effects of treatment."""
        side_effects = []

        if defect_type == DefectType.CLIPPING:
            side_effects.append("May reduce transient impact")
            if severity > 0.7:
                side_effects.append("Possible spectral smearing")

        elif defect_type == DefectType.CLICKS:
            side_effects.append("May smooth fast transients")
            if severity > 0.5:
                side_effects.append("Possible high-frequency dulling")

        elif defect_type == DefectType.BROADBAND_NOISE:
            side_effects.append("May introduce musical noise artifacts")
            side_effects.append("Possible loss of ambience")
            if severity > 0.6:
                side_effects.append("Potential spectral holes")

        elif defect_type == DefectType.HUM:
            side_effects.append("Narrow notch filters (minimal impact)")

        elif defect_type == DefectType.RUMBLE:
            side_effects.append("Reduced low-frequency energy")
            if severity > 0.5:
                side_effects.append("Possible bass thinning")

        elif defect_type == DefectType.HF_ROLLOFF:
            side_effects.append("Possible treble harshness")
            side_effects.append("May amplify noise floor")

        elif defect_type == DefectType.SPECTRAL_ARTIFACTS:
            side_effects.append("May slightly smooth fine spectral texture")

        elif defect_type == DefectType.DROPOUTS:
            side_effects.append("May blur very short gaps if interpolation is too strong")

        elif defect_type == DefectType.PHASE_ISSUES:
            side_effects.append("May narrow stereo width when phase is realigned")

        elif defect_type == DefectType.ALIASING:
            side_effects.append("May soften the very top octave")

        return side_effects

    def _requires_manual_check(self, defect: DefectInstance) -> bool:
        """Bestimmt if manual verification recommended."""
        # High severity defects should be checked
        if defect.severity > 0.8:
            return True

        # Low confidence detections should be checked
        if defect.confidence < 0.6:
            return True

        # Distortion is subjective, needs verification
        return bool(defect.type == DefectType.DISTORTION and defect.severity > 0.5)

    def _create_no_treatment(self, _defect: DefectInstance) -> TreatmentRecommendation:
        """Erstellt a placeholder for defects with no treatment."""
        return TreatmentRecommendation(
            method="manual_inspection",
            module_path="",
            params={},
            priority=5,
            expected_improvement=0.0,
            side_effects=["No automatic treatment available"],
            requires_manual_check=True,
        )
