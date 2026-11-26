"""
Regression tests for the risk category on the red-team API upload path.

Dataset-framework producers (BeaverTails, Aegis) build ``RTTestCase`` objects
without a ``risk_category``, so the field is ``None``. ``APIRTTestCase`` declares
``riskCategory`` as a required ``str``; the mapping has to apply the same
``"Others"`` fallback that ``getRiskCategory()`` uses, otherwise
``map_risk_assessment_to_api()`` raises a ``ValidationError`` and the results
are never uploaded.
"""

import pytest

from deepteam.frameworks.aegis.types import AegisType
from deepteam.frameworks.beavertails.types import BeaverTailsType
from deepteam.red_teamer.api import map_risk_assessment_to_api
from deepteam.red_teamer.risk_assessment import (
    RiskAssessment,
    construct_risk_assessment_overview,
)
from deepteam.test_case import RTTestCase


def _api_assessment(test_case):
    """Wrap a single test case the way a framework assessment does."""
    overview = construct_risk_assessment_overview([test_case], run_duration=1.0)
    assessment = RiskAssessment(overview=overview, test_cases=[test_case])
    return map_risk_assessment_to_api(assessment)


@pytest.mark.parametrize(
    "vulnerability_type",
    [BeaverTailsType.BEAVERTAILS, AegisType.AEGIS],
    ids=["beavertails", "aegis"],
)
def test_framework_case_without_risk_category_maps_to_others(
    vulnerability_type,
):
    """A missing risk category must not break the upload mapping."""
    test_case = RTTestCase(
        input="harmful prompt",
        vulnerability="dataset category",
        vulnerability_type=vulnerability_type,
    )
    assert test_case.risk_category is None

    api_assessment = _api_assessment(test_case)

    assert len(api_assessment.test_cases) == 1
    assert api_assessment.test_cases[0].risk_category == "Others"


def test_explicit_risk_category_is_preserved():
    """The fallback must not clobber a category that is already present."""
    test_case = RTTestCase(
        input="prompt",
        vulnerability="Bias",
        vulnerability_type=BeaverTailsType.BEAVERTAILS,
        risk_category="Fairness",
    )

    api_assessment = _api_assessment(test_case)

    assert api_assessment.test_cases[0].risk_category == "Fairness"
