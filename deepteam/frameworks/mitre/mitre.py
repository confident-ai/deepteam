from typing import List, Literal
from deepteam.frameworks import AISafetyFramework
from deepteam.frameworks.mitre.risk_categories import MITRE_CATEGORIES

"""
MITRE ATT&CK Framework Mapping for LLMs 2025
===========================================

This framework maps MITRE ATT&CK tactics and techniques to LLM-specific threats and vulnerabilities.
It includes red-teaming attack methods and corresponding weaknesses tied to each MITRE category.

Each category includes:
- Attacks: Techniques for exploiting, bypassing, or manipulating LLM systems.
- Vulnerabilities: Weaknesses that can be leveraged by attackers in LLM models.

Reference: https://attack.mitre.org
"""

ALLOWED_TYPES = [
    "reconnaissance",
    "resource_development",
    "initial_access",
    "ml_attack_staging",
    "exfiltration",
    "impact",
]


class MITRE(AISafetyFramework):
    def __init__(
        self,
        categories: List[
            Literal[
                "reconnaissance",
                "resource_development",
                "initial_access",
                "ml_attack_staging",
                "exfiltration",
                "impact",
            ]
        ] = [
            "reconnaissance",
            "resource_development",
            "initial_access",
            "ml_attack_staging",
            "exfiltration",
            "impact",
        ],
    ):
        self.name = "MITRE ATLAS"
        self.description = "MITRE ATLAS mapping for LLM/ML adversarial techniques (project-specific) to understand and defend against threats specific to AI and ML systems."
        self.categories = categories
        self.risk_categories = []
        for category in categories:
            for risk_category in MITRE_CATEGORIES:
                if risk_category.name == category:
                    self.risk_categories.append(risk_category)

    def get_name(self):
        return "MITRE ATLAS"
