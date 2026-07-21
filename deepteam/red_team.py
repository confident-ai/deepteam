from typing import List, Optional
from contextlib import nullcontext

from deepeval.models import DeepEvalBaseLLM
from deepteam.vulnerabilities import BaseVulnerability
from deepteam.attacks import BaseAttack
from deepteam.attacks.multi_turn.types import CallbackType
from deepteam.red_teamer import RedTeamer
from deepteam.attacks.attack_engine import AttackEngine
from deepteam.frameworks.frameworks import AISafetyFramework
from deepteam.utils import progress_callback_context, ProgressCallback


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
    attack_engine: Optional[AttackEngine] = None,
    on_progress: Optional[ProgressCallback] = None,
):
    red_teamer = RedTeamer(
        async_mode=async_mode,
        max_concurrent=max_concurrent,
        target_purpose=target_purpose,
        simulator_model=simulator_model,
        evaluation_model=evaluation_model,
        attack_engine=attack_engine,
    )
    ctx = (
        progress_callback_context(on_progress)
        if on_progress is not None
        else nullcontext()
    )
    with ctx:
        risk_assessment = red_teamer.red_team(
            model_callback=model_callback,
            vulnerabilities=vulnerabilities,
            attacks=attacks,
            simulator_model=simulator_model,
            evaluation_model=evaluation_model,
            framework=framework,
            attacks_per_vulnerability_type=attacks_per_vulnerability_type,
            ignore_errors=ignore_errors,
            attack_engine=attack_engine,
        )
    return risk_assessment
