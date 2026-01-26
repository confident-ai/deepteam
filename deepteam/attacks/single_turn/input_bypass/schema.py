from pydantic import BaseModel


class EnhancedBypass(BaseModel):
    bypass_strategy: str
    input: str


class ComplianceData(BaseModel):
    non_compliant: bool


class IsBypassValid(BaseModel):
    is_valid_bypass: bool
