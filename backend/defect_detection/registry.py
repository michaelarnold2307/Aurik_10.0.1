"""
Defect Detector Registry
=========================

Central registry for managing defect detectors.
"""

from backend.core.defect_scanner import DefectType as CoreDefectType
from backend.defect_detection.base import DefectDetector, DefectType, require_core_defect_type


class DefectDetectorRegistry:
    """
    Registry for managing defect detectors.

    Allows registration, lookup, and batch execution of detectors.
    """

    def __init__(self):
        self._detectors: dict[str, DefectDetector] = {}
        self._detectors_by_type: dict[CoreDefectType, list[DefectDetector]] = {}

    def _normalize_type(self, defect_type: DefectType | CoreDefectType) -> CoreDefectType:
        """Normalisiert Legacy-Typen auf den kanonischen Core-Typ, wenn möglich."""
        if isinstance(defect_type, CoreDefectType):
            return defect_type

        return require_core_defect_type(defect_type, context="DefectDetectorRegistry")

    def register(self, detector: DefectDetector) -> None:
        """
        Registriert a defect detector.

        Args:
            detector: Detector instance to register
        """
        self._detectors[detector.name] = detector

        normalized_type = self._normalize_type(detector.defect_type)
        if normalized_type not in self._detectors_by_type:
            self._detectors_by_type[normalized_type] = []

        self._detectors_by_type[normalized_type].append(detector)

    def get(self, name: str) -> DefectDetector | None:
        """Gibt zurück: detector by name."""
        return self._detectors.get(name)

    def get_by_type(self, defect_type: DefectType | CoreDefectType) -> list[DefectDetector]:
        """Gibt zurück: all detectors for a specific defect type.

        Akzeptiert sowohl den Legacy-Enum als auch den kanonischen Core-Enum.
        """
        normalized_type = self._normalize_type(defect_type)
        return self._detectors_by_type.get(normalized_type, [])

    def get_all(self) -> list[DefectDetector]:
        """Gibt zurück: all registered detectors."""
        return list(self._detectors.values())

    def list_names(self) -> list[str]:
        """Listet alle registrierten Detektor-Namen auf."""
        return list(self._detectors.keys())

    def list_types(self) -> list[CoreDefectType]:
        """Listet auf: all defect types with registered detectors.

        Die Registry speichert intern nur kanonische Core-Typen.
        """
        return list(self._detectors_by_type.keys())


# Global singleton registry
_global_registry = DefectDetectorRegistry()


def get_global_registry() -> DefectDetectorRegistry:
    """Gibt zurück: the global defect detector registry."""
    return _global_registry
