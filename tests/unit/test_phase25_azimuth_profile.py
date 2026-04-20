"""Unit tests for phase_25_azimuth_correction._compute_azimuth_profile (§2.56)."""

from backend.core.phases.phase_25_azimuth_correction import AzimuthCorrectionPhaseV2


class TestAzimuthProfile:
    def _p(self, material="tape", qm="restoration", rest=50.0):
        return AzimuthCorrectionPhaseV2._compute_azimuth_profile(material, qm, rest)

    def test_returns_required_keys(self):
        p = self._p()
        assert "xcorr_window_samples" in p

    def test_value_in_bounds(self):
        for mat in ("tape", "reel_tape", "shellac", "cassette", "cd_digital", "unknown"):
            for qm in ("fast", "restoration", "quality", "maximum"):
                p = self._p(mat, qm)
                assert 2048 <= p["xcorr_window_samples"] <= 8192, (
                    f"xcorr={p['xcorr_window_samples']} out of [2048,8192] mat={mat} qm={qm}"
                )

    def test_value_is_power_of_two(self):
        for mat in ("tape", "shellac", "cd_digital"):
            p = self._p(mat)
            v = p["xcorr_window_samples"]
            assert v & (v - 1) == 0, f"xcorr={v} is not a power of two"

    def test_quality_increases_window(self):
        base = self._p("tape", "restoration")
        qual = self._p("tape", "quality")
        assert qual["xcorr_window_samples"] >= base["xcorr_window_samples"]

    def test_fast_decreases_window(self):
        base = self._p("tape", "restoration")
        fast = self._p("tape", "fast")
        assert fast["xcorr_window_samples"] <= base["xcorr_window_samples"]

    def test_low_rest_decreases_window(self):
        high = self._p("tape", "restoration", 80.0)
        low = self._p("tape", "restoration", 20.0)
        assert low["xcorr_window_samples"] <= high["xcorr_window_samples"]

    def test_none_quality_mode(self):
        p = self._p("tape", None)
        assert 2048 <= p["xcorr_window_samples"] <= 8192

    def test_unknown_material(self):
        p = self._p("totally_unknown_xyz")
        assert 2048 <= p["xcorr_window_samples"] <= 8192
