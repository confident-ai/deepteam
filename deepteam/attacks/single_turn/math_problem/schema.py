from pydantic import BaseModel


class EnhancedAttack(BaseModel):
    math_strategy: str
    input: str


class ComplianceData(BaseModel):
    non_compliant: bool


class IsMathProblem(BaseModel):
    is_math_problem: bool
