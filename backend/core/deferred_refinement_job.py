"""
backend/core/deferred_refinement_job.py — DeferredRefinementJob §2.38 KMV
==========================================================================

Pure-Python dataclass describing a pending Stufe-2 ML-refinement job.
Has no Qt dependency — importable from both backend and UI layers.

Spec §2.38: All mandatory fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DeferredRefinementJob:
    """Captures all information needed to re-run deferred phases with no RT limit.

    Created by BatchProcessingThread when RestorationResult.deferred_phases is
    non-empty, passed to MLRefinementThread for Stufe-2 processing.

    Fields (all mandatory per §2.38):
        output_path:          Filesystem path of the Stufe-1 export to overwrite.
        audio_original:       Original input audio (float32, 48 kHz).  Registered
                              in ml_memory_budget before Stufe-2 starts.
        sr:                   Sample rate (always 48000).
        mode:                 Restoration mode string ("restoration" | "studio2026").
        deferred_phase_ids:   Phase IDs skipped in Stufe 1 due to RT budget.
        cached_defect_result: DefectScan result from Stufe 1 (reused, no re-scan).
        cached_era_result:    EraClassifier result from Stufe 1 (reused).
        cached_medium_result: MediumClassifier result from Stufe 1 (reused).
        stufe1_quality:       quality_estimate from Stufe-1 RestorationResult.
                              Stufe-2 export only overwrites if >= this value.
        input_path:           Original input file path (for progress messages).
    """

    output_path: str
    audio_original: np.ndarray
    sr: int
    mode: str
    deferred_phase_ids: list[str]
    cached_defect_result: Any
    cached_era_result: Any
    cached_medium_result: Any
    stufe1_quality: float
    input_path: str = ""
    # Internal: set to True once ml_memory_budget.try_allocate succeeds
    _budget_registered: bool = field(default=False, init=False, repr=False)

    @property
    def audio_size_gb(self) -> float:
        """Approximate RAM footprint of audio_original in GB."""
        if self.audio_original is None:
            return 0.0
        nbytes = self.audio_original.nbytes
        return nbytes / (1024**3)

    @property
    def n_deferred(self) -> int:
        return len(self.deferred_phase_ids)
