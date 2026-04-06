from typing import List, Literal
from deepteam.frameworks import AISafetyFramework
from deepteam.vulnerabilities import BaseVulnerability
from deepteam.attacks import BaseAttack
from deepteam.frameworks.eu_ai_act.risk_categories import EU_AI_ACT_CATEGORIES

"""
EU Artificial Intelligence Act (EU AI Act)
==========================================

The EU AI Act is the European Union's comprehensive regulatory framework for artificial
intelligence, establishing harmonized rules for the development, placement on the market,
and use of AI systems. It is the world's first comprehensive AI law, with enforcement
beginning in 2025 and full high-risk system compliance required by August 2026.

DeepTeam's implementation covers the key testable obligations across 8 categories mapped
to the regulation's core articles:

- Article 5 (EUAI_01): Prohibited practices — subliminal manipulation, vulnerability exploitation
- Article 10 (EUAI_02): Bias & non-discrimination — fairness, ethics, protected characteristics
- Article 15 (EUAI_03): Robustness & cybersecurity — adversarial resilience, injection attacks
- Article 13 (EUAI_04): Transparency & explainability — AI disclosure, capability representation
- Article 14 (EUAI_05): Human oversight — automation bias, override support
- Articles 10 & 12 (EUAI_06): Data governance & privacy — PII protection, data provenance
- Articles 5 & 9 (EUAI_07): Child & vulnerable group protection — age-appropriate content, safety
- Article 9 (EUAI_08): Fundamental rights & safety — illegal activity, IP, access control

Each category includes attacks and vulnerabilities that test an AI system's compliance with
the corresponding regulatory requirements.

Reference: https://artificialintelligenceact.eu/
"""


class EUAIAct(AISafetyFramework):
    name = "EU AI Act"
    description = (
        "European Union Artificial Intelligence Act — compliance testing framework "
        "covering prohibited practices (Art. 5), bias and non-discrimination (Art. 10), "
        "robustness and cybersecurity (Art. 15), transparency and explainability (Art. 13), "
        "human oversight (Art. 14), data governance and privacy (Art. 10/12), "
        "child and vulnerable group protection (Art. 5/9), and fundamental rights "
        "and safety (Art. 9)."
    )
    ALLOWED_TYPES = [
        "euai_01",
        "euai_02",
        "euai_03",
        "euai_04",
        "euai_05",
        "euai_06",
        "euai_07",
        "euai_08",
    ]

    def __init__(
        self,
        categories: List[
            Literal[
                "euai_01",
                "euai_02",
                "euai_03",
                "euai_04",
                "euai_05",
                "euai_06",
                "euai_07",
                "euai_08",
            ]
        ] = [
            "euai_01",
            "euai_02",
            "euai_03",
            "euai_04",
            "euai_05",
            "euai_06",
            "euai_07",
            "euai_08",
        ],
    ):
        self.categories = categories
        self.risk_categories = []
        self.vulnerabilities = []
        self.attacks = []
        for category in categories:
            for risk_category in EU_AI_ACT_CATEGORIES:
                if risk_category.name == category:
                    self.risk_categories.append(risk_category)
                    self.vulnerabilities.extend(risk_category.vulnerabilities)
                    self.attacks.extend(risk_category.attacks)

    def get_name(self):
        return self.name
