from pydantic import BaseModel


class EnhancedAttack(BaseModel):
    poisoning_strategy: str
    input: str


class ComplianceData(BaseModel):
    non_compliant: bool


class IsContextPoisoned(BaseModel):
    is_context_poisoned: bool
