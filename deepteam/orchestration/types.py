from typing import List, Optional

from pydantic import BaseModel

from deepteam.attacks.multi_turn.progression import ProgressionResult
from deepteam.orchestration.schema import CriteriaVerdict
from deepteam.test_case.test_case import RTTurn


class Escalation(BaseModel):
    turn: int
    guidance: str


class SubAgentReport(BaseModel):
    criteria: str
    scenario: str
    target_analysis: str
    instructions: str
    progression: ProgressionResult
    verdicts: List[CriteriaVerdict] = []
    escalations: List[Escalation] = []


class MetCriteria(BaseModel):
    criteria: str
    how_it_was_met: str
    round: int


class RoundResult(BaseModel):
    number: int
    scenario: str
    reports: List[SubAgentReport] = []
    winner: Optional[SubAgentReport] = None


class OrchestrationResult(BaseModel):
    recon: ProgressionResult
    analysis: str
    rounds: List[RoundResult] = []
    met_criteria: List[MetCriteria] = []
    conversation: List[RTTurn] = []
