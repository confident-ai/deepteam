from typing import List, Optional

from deepeval.models import DeepEvalBaseLLM
from deepteam.vulnerabilities import BaseVulnerability
from deepteam.attacks import BaseAttack
from deepteam.attacks.multi_turn.types import CallbackType
from deepteam.red_teamer import RedTeamer
from deepteam.frameworks.frameworks import AISafetyFramework


def red_team(
    model_callback: CallbackType,
    vulnerabilities: Optional[List[BaseVulnerability]] = None,
    attacks: Optional[List[BaseAttack]] = None,
    framework: Optional[AISafetyFramework] = None,
    simulator_model: DeepEvalBaseLLM = "gpt-4o-mini",
    evaluation_model: DeepEvalBaseLLM = "gpt-4o-mini",
    attacks_per_vulnerability_type: int = 1,
    ignore_errors: bool = True,
    async_mode: bool = True,
    max_concurrent: int = 10,
    target_purpose: Optional[str] = None,
    reuse_simulated_test_cases: bool = False,
):
    red_teamer = RedTeamer(
        async_mode=async_mode,
        max_concurrent=max_concurrent,
        target_purpose=target_purpose,
        simulator_model=simulator_model,
        evaluation_model=evaluation_model,
    )
    risk_assessment = red_teamer.red_team(
        model_callback=model_callback,
        vulnerabilities=vulnerabilities,
        attacks=attacks,
        simulator_model=simulator_model,
        evaluation_model=evaluation_model,
        framework=framework,
        attacks_per_vulnerability_type=attacks_per_vulnerability_type,
        ignore_errors=ignore_errors,
        reuse_simulated_test_cases=reuse_simulated_test_cases,
    )
    return risk_assessment
