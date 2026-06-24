import os
from typing import List

CODE_CONTEXT_LIMIT = int(os.getenv("CODE_CONTEXT_LIMIT", "40000"))

DEFAULT_CODE_SCAN_VULNERABILITIES: List[str] = [
    "Unexpected Code Execution",
    "Shell Injection",
    "SQL Injection",
    "SSRF",
    "Excessive Agency",
    "Indirect Instruction",
    "Tool Metadata Poisoning",
    "Tool Orchestration Abuse",
    "BFLA",
    "BOLA",
    "RBAC",
    "Debug Access",
    "Prompt Leakage",
    "PII Leakage",
]
