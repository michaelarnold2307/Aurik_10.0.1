"""Unit tests for phase_01_click_removal._compute_click_removal_profile (§2.56)."""

from backend.core.phases.phase_01_click_removal import ClickRemovalPhase


class TestClickRemovalProfile:
    def _p(self, material="vinyl", qm="balanced", rest=50.0):
        return ClickRemovalPhase._compute_click_removal_profile(material, qm, rest)

    def test_returns_required_keys(self):
        p = self._p()
        for key in ("ml_severity_threshold", "cubic_ctx", "spectral_ctx"):
            assert key in p

    def test_ml_threshold_bounds(self):
        for mat in ("vinyl", "shellac", "wax_cylinder", "cd_digital", "unknown"):
            for qm in ("fast", "balanced", "quality", "maximum"):
                p = self._p(mat, qm)
                assert 0.35 <= p["ml_severity_threshold"] <= 0.80, (
                    f"threshold={p['ml_severity_threshold']} out of bounds for mat={mat} qm={qm}"
                )

    def test_cubic_ctx_bounds(self):
        for mat in ("vinyl", "shellac", "cd_digital"):
            p = self._p(mat)
            assert 6 <= int(p["cubic_ctx"]) <= 20

    def test_spectral_ctx_bounds(self):
        for mat in ("vinyl", "shellac", "wax_cylinder", "cd_digital"):
            p = self._p(mat)
            assert 64 <= int(p["spectral_ctx"]) <= 256

    def test_quality_lowers_threshold(self):
        base = self._p("vinyl", "balanced")
        qual = self._p("vinyl", "quality")
        assert qual["ml_severity_threshold"] <= base["ml_severity_threshold"]

    def test_fast_raises_threshold(self):
        base = self._p("vinyl", "balanced")
        fast = self._p("vinyl", "fast")
        assert fast["ml_severity_threshold"] >= base["ml_severity_threshold"]

    def test_quality_increases_cubic_ctx(self):
        base = self._p("vinyl", "balanced")
        qual = self._p("vinyl", "quality")
        assert qual["cubic_ctx"] >= base["cubic_ctx"]

    def test_quality_increases_spectral_ctx(self):
        base = self._p("vinyl", "balanced")
        qual = self._p("vinyl", "quality")
        assert qual["spectral_ctx"] >= base["spectral_ctx"]

    def test_low_rest_lowers_threshold(self):
        high = self._p("vinyl", "balanced", 80.0)
        low = self._p("vinyl", "balanced", 20.0)
        assert low["ml_severity_threshold"] <= high["ml_severity_threshold"]

    def test_shellac_lower_threshold_than_cd(self):
        shellac = self._p("shellac")
        cd = self._p("cd_digital")
        assert shellac["ml_severity_threshold"] <= cd["ml_severity_threshold"]

    def test_none_quality_mode(self):
        p = self._p("vinyl", None)
        assert 0.35 <= p["ml_severity_threshold"] <= 0.80

    def test_unknown_material(self):
        p = self._p("super_exotic_xyz")
        assert 0.35 <= p["ml_severity_threshold"] <= 0.80
