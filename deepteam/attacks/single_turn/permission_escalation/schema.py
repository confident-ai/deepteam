from pydantic import BaseModel


class EnhancedPermission(BaseModel):
    escalation_logic: str
    input: str


class ComplianceData(BaseModel):
    non_compliant: bool


class IsPermissionValid(BaseModel):
    is_valid_permission: bool
