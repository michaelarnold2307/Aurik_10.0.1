"""Unit tests for phase_19_de_esser._compute_de_esser_profile (§2.56)."""

from backend.core.phases.phase_19_de_esser import DeEsserPhase


class TestDeEsserProfile:
    def _p(self, material="vinyl", qm="balanced", rest=50.0):
        return DeEsserPhase._compute_de_esser_profile(material, qm, rest)

    def test_returns_required_keys(self):
        p = self._p()
        assert "lookahead_ms" in p

    def test_lookahead_in_bounds(self):
        for mat in ("vinyl", "shellac", "wax_cylinder", "cd_digital", "tape", "unknown"):
            for qm in ("fast", "balanced", "quality", "maximum"):
                p = self._p(mat, qm)
                assert 2.0 <= p["lookahead_ms"] <= 10.0, (
                    f"lookahead={p['lookahead_ms']} out of [2,10] for mat={mat} qm={qm}"
                )

    def test_quality_increases_lookahead(self):
        base = self._p("vinyl", "balanced")
        qual = self._p("vinyl", "quality")
        assert qual["lookahead_ms"] >= base["lookahead_ms"]

    def test_fast_decreases_lookahead(self):
        base = self._p("vinyl", "balanced")
        fast = self._p("vinyl", "fast")
        assert fast["lookahead_ms"] <= base["lookahead_ms"]

    def test_shellac_larger_lookahead_than_cd(self):
        shellac = self._p("shellac", "balanced")
        cd = self._p("cd_digital", "balanced")
        assert shellac["lookahead_ms"] >= cd["lookahead_ms"]

    def test_wax_cylinder_highest_base(self):
        wax = self._p("wax_cylinder", "balanced")
        cd = self._p("cd_digital", "balanced")
        assert wax["lookahead_ms"] >= cd["lookahead_ms"]

    def test_low_rest_increases_lookahead(self):
        high = self._p("vinyl", "balanced", 80.0)
        low = self._p("vinyl", "balanced", 20.0)
        assert low["lookahead_ms"] >= high["lookahead_ms"]

    def test_none_quality_mode(self):
        p = self._p("vinyl", None)
        assert 2.0 <= p["lookahead_ms"] <= 10.0

    def test_unknown_material(self):
        p = self._p("totally_unknown_xyz")
        assert 2.0 <= p["lookahead_ms"] <= 10.0
