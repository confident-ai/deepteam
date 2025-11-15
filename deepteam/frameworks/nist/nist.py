from typing import List, Literal
from deepteam.frameworks import AISafetyFramework
from deepteam.vulnerabilities import BaseVulnerability
from deepteam.attacks import BaseAttack
from deepteam.frameworks.nist.risk_categories import NIST_CATEGORIES

"""
NIST Cybersecurity Framework Mapping for LLMs 2025
==================================================

This framework maps NIST Cybersecurity Framework risk categories to LLM-specific threats and vulnerabilities.
It includes realistic attack techniques and weaknesses tied to each category.

Each category includes:
- Attacks: Techniques for exploiting or bypassing LLM security.
- Vulnerabilities: Weaknesses that can be exploited in LLM systems.

Reference: https://www.nist.gov/itl/ai-risk-management-framework
"""

ALLOWED_TYPES = ["measure_1", "measure_2", "measure_3", "measure_4"]


class NIST(AISafetyFramework):
    def __init__(
        self,
        categories: List[
            Literal["measure_1", "measure_2", "measure_3", "measure_4"]
        ] = ["measure_1", "measure_2", "measure_3", "measure_4"],
    ):
        self.name = "NIST AI RMF"
        self.description = "NIST AI Risk Management Framework (AI RMF) — Measure-focused mapping for testing and evaluation."
        self.categories = categories
        self.risk_categories = []
        for category in categories:
            for risk_category in NIST_CATEGORIES:
                if risk_category.name == category:
                    self.risk_categories.append(risk_category)

    def get_name(self):
        return "NIST AI RMF"
