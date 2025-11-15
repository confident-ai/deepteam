from typing import List, Literal
from deepteam.frameworks import AISafetyFramework
from deepteam.vulnerabilities import (
    BaseVulnerability,
)
from deepteam.attacks import BaseAttack
from deepteam.frameworks.owasp.risk_categories import OWASP_CATEGORIES

"""
OWASP Top 10 for LLMs 2025 Framework Mapping
============================================

This framework defines the 10 OWASP LLM risk categories as DeepTeam `RiskCategory`
objects, mapping realistic red-teaming attacks and corresponding vulnerabilities.

Each category includes:
- Attacks: techniques for prompting, probing, or jailbreaking.
- Vulnerabilities: typed or custom definitions capturing observable weaknesses.

Reference: https://genai.owasp.org/llm-top-10/
"""

ALLOWED_TYPES = [
    "LLM_01",
    "LLM_02",
    "LLM_03",
    "LLM_04",
    "LLM_05",
    "LLM_06",
    "LLM_07",
    "LLM_08",
    "LLM_09",
    "LLM_10",
]


class OWASPTop10(AISafetyFramework):

    def __init__(
        self,
        categories: List[
            Literal[
                "LLM_01",
                "LLM_02",
                "LLM_03",
                "LLM_04",
                "LLM_05",
                "LLM_06",
                "LLM_07",
                "LLM_08",
                "LLM_09",
                "LLM_10",
            ]
        ] = [
            "LLM_01",
            "LLM_02",
            "LLM_03",
            "LLM_04",
            "LLM_05",
            "LLM_06",
            "LLM_07",
            "LLM_08",
            "LLM_09",
            "LLM_10",
        ],
    ):
        self.name = "OWASP"
        self.description = "The OWASP Top 10 for LLMs 2025"
        self.risk_categories = OWASP_CATEGORIES
        self.categories = categories
        self.risk_categories = []
        for category in categories:
            for risk_category in OWASP_CATEGORIES:
                if risk_category.name == category:
                    self.risk_categories.append(risk_category)

    def get_name(self):
        return "OWASP"
