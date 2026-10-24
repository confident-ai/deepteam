from pydantic import BaseModel


class NextTurn(BaseModel):
    message: str


class BehaviorAnalysis(BaseModel):
    analysis: str


class ReconVerdict(BaseModel):
    observations: str
    directive: str


class SubAgentInstructions(BaseModel):
    instructions: str


class ScenarioRewrite(BaseModel):
    scenario: str


class EscalationGuidance(BaseModel):
    guidance: str


class CriteriaVerdict(BaseModel):
    reason: str
    score: int
    criteria_met: bool
    improvement_guidance: str
    escalate_to_parent: bool

    @property
    def unit_score(self) -> float:
        return max(0.0, min(1.0, (self.score - 1) / 9))
