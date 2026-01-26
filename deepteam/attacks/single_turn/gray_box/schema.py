from pydantic import BaseModel


class EnhancedAttack(BaseModel):
    deception_strategy: str
    input: str


class ComplianceData(BaseModel):
    non_compliant: bool


class IsGrayBox(BaseModel):
    is_gray_box: bool
