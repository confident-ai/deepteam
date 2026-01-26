from pydantic import BaseModel


class EnhancedInjection(BaseModel):
    strategy_reasoning: str
    input: str


class ComplianceData(BaseModel):
    non_compliant: bool


class IsValidInjection(BaseModel):
    is_valid_injection: bool
