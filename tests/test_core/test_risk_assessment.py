from deepteam.test_case import RTTestCase
from deepteam.vulnerabilities.bias import BiasType
from deepteam.red_teamer.risk_assessment import (
    construct_risk_assessment_overview,
)


def _case(score=None, error=None, attack_method=None):
    return RTTestCase(
        vulnerability="Bias",
        vulnerability_type=BiasType.RACE,
        input="input",
        actual_output=None if error else "output",
        score=score,
        error=error,
        attack_method=attack_method,
    )


def test_overview_errored_count_is_not_clobbered():
    """The top-level ``errored`` total (rendered to the user as the
    '(N errored)' header) must reflect every errored test case, even when
    some cases succeed. Regression for a variable-shadowing bug where the
    per-group loop reused the ``errored`` accumulator name and overwrote it
    with 0."""
    test_cases = [
        _case(score=1.0, attack_method="Base64"),  # passing
        _case(score=0.0, attack_method="Base64"),  # failing
        _case(error="model timeout", attack_method="Base64"),  # errored
        _case(error="rate limited", attack_method="Base64"),  # errored
    ]

    overview = construct_risk_assessment_overview(
        red_teaming_test_cases=test_cases, run_duration=1.0
    )

    assert overview.errored == 2
