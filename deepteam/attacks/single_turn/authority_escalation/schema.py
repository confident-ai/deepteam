from pydantic import BaseModel


class EnhancedAuthorityAttack(BaseModel):
    authority_role: str
    input: str


class ComplianceData(BaseModel):
    non_compliant: bool


class IsAuthorityValid(BaseModel):
    is_valid_authority: bool
