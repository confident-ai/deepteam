"""Tests for RiskAssessment.load() — the companion to save()."""

import json
import os
import tempfile

import pytest

from deepteam.red_teamer.risk_assessment import (
    RiskAssessment,
    RedTeamingOverview,
    VulnerabilityTypeResult,
    AttackMethodResult,
    _build_vulnerability_type_lookup,
    _resolve_vulnerability_type,
)
from deepteam.test_case import RTTestCase, RTTurn
from deepteam.vulnerabilities.bias import BiasType
from deepteam.vulnerabilities.toxicity import ToxicityType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_case(
    vulnerability="Bias",
    vulnerability_type=BiasType.GENDER,
    input_text="test input",
    actual_output="test output",
    score=1.0,
    reason="safe",
    attack_method="Baseline",
    risk_category="Bias",
    turns=None,
):
    return RTTestCase(
        vulnerability=vulnerability,
        vulnerability_type=vulnerability_type,
        input=input_text,
        actual_output=actual_output,
        score=score,
        reason=reason,
        attack_method=attack_method,
        risk_category=risk_category,
        turns=turns,
    )


def _make_assessment(test_cases=None):
    if test_cases is None:
        test_cases = [_make_test_case()]

    overview = RedTeamingOverview(
        vulnerability_type_results=[
            VulnerabilityTypeResult(
                vulnerability="Bias",
                vulnerability_type=BiasType.GENDER,
                pass_rate=1.0,
                passing=1,
                failing=0,
                errored=0,
            )
        ],
        attack_method_results=[
            AttackMethodResult(
                attack_method="Baseline",
                pass_rate=1.0,
                passing=1,
                failing=0,
                errored=0,
            )
        ],
        errored=0,
        run_duration=1.23,
    )
    return RiskAssessment(overview=overview, test_cases=test_cases)


# ---------------------------------------------------------------------------
# Tests: _build_vulnerability_type_lookup
# ---------------------------------------------------------------------------

class TestVulnerabilityTypeLookup:
    def test_lookup_contains_known_values(self):
        lookup = _build_vulnerability_type_lookup()
        assert "gender" in lookup
        assert any(m.value == "gender" for m in lookup["gender"])

    def test_lookup_returns_enum_members(self):
        lookup = _build_vulnerability_type_lookup()
        for members in lookup.values():
            for m in members:
                assert hasattr(m, "value")
                assert hasattr(m, "name")


# ---------------------------------------------------------------------------
# Tests: _resolve_vulnerability_type
# ---------------------------------------------------------------------------

class TestResolveVulnerabilityType:
    def test_resolves_known_value(self):
        lookup = _build_vulnerability_type_lookup()
        result = _resolve_vulnerability_type("gender", lookup)
        assert result == BiasType.GENDER

    def test_falls_back_to_string_for_unknown(self):
        lookup = _build_vulnerability_type_lookup()
        result = _resolve_vulnerability_type("unknown_custom_type", lookup)
        assert result == "unknown_custom_type"


# ---------------------------------------------------------------------------
# Tests: RiskAssessment.load
# ---------------------------------------------------------------------------

class TestRiskAssessmentLoad:
    def test_roundtrip_save_load(self, tmp_path):
        """save() then load() produces equivalent data."""
        assessment = _make_assessment()
        saved_path = assessment.save(str(tmp_path))

        loaded = RiskAssessment.load(saved_path)

        assert len(loaded.test_cases) == len(assessment.test_cases)
        tc = loaded.test_cases[0]
        assert tc.vulnerability == "Bias"
        assert tc.vulnerability_type == BiasType.GENDER
        assert tc.input == "test input"
        assert tc.actual_output == "test output"
        assert tc.score == 1.0

    def test_roundtrip_preserves_overview(self, tmp_path):
        assessment = _make_assessment()
        saved_path = assessment.save(str(tmp_path))

        loaded = RiskAssessment.load(saved_path)

        vtr = loaded.overview.vulnerability_type_results[0]
        assert vtr.vulnerability_type == BiasType.GENDER
        assert vtr.pass_rate == 1.0

    def test_roundtrip_multi_turn(self, tmp_path):
        """Multi-turn test cases with RTTurn objects survive roundtrip."""
        turns = [
            RTTurn(role="user", content="hello"),
            RTTurn(role="assistant", content="hi there"),
        ]
        tc = _make_test_case(actual_output=None, turns=turns)
        assessment = _make_assessment(test_cases=[tc])
        saved_path = assessment.save(str(tmp_path))

        loaded = RiskAssessment.load(saved_path)

        loaded_tc = loaded.test_cases[0]
        assert loaded_tc.turns is not None
        assert len(loaded_tc.turns) == 2
        assert loaded_tc.turns[0].role == "user"
        assert loaded_tc.turns[0].content == "hello"

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            RiskAssessment.load("/nonexistent/path/file.json")

    def test_load_invalid_json(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json {{{")
        with pytest.raises(json.JSONDecodeError):
            RiskAssessment.load(str(bad_file))

    def test_multiple_vulnerability_types(self, tmp_path):
        """Different vulnerability types are all restored correctly."""
        tc1 = _make_test_case(
            vulnerability="Bias",
            vulnerability_type=BiasType.RELIGION,
        )
        tc2 = _make_test_case(
            vulnerability="Toxicity",
            vulnerability_type=ToxicityType.INSULTS,
        )
        overview = RedTeamingOverview(
            vulnerability_type_results=[
                VulnerabilityTypeResult(
                    vulnerability="Bias",
                    vulnerability_type=BiasType.RELIGION,
                    pass_rate=1.0,
                    passing=1,
                    failing=0,
                    errored=0,
                ),
                VulnerabilityTypeResult(
                    vulnerability="Toxicity",
                    vulnerability_type=ToxicityType.INSULTS,
                    pass_rate=1.0,
                    passing=1,
                    failing=0,
                    errored=0,
                ),
            ],
            attack_method_results=[],
            errored=0,
            run_duration=0.5,
        )
        assessment = RiskAssessment(
            overview=overview, test_cases=[tc1, tc2]
        )
        saved_path = assessment.save(str(tmp_path))

        loaded = RiskAssessment.load(saved_path)

        assert loaded.test_cases[0].vulnerability_type == BiasType.RELIGION
        assert loaded.test_cases[1].vulnerability_type == ToxicityType.INSULTS
