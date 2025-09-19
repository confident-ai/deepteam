import random
import asyncio
from deepteam.red_teamer.utils import group_attacks_by_vulnerability_type
from deepteam.vulnerabilities import BaseVulnerability
from deepteam.attacks.single_turn import BaseSingleTurnAttack
from deepteam.attacks.single_turn.prompt_injection.template import (
    PromptInjectionTemplate,
)

class PromptInjection(BaseSingleTurnAttack):
    def __init__(self, weight: int = 1):
        self.weight = weight

    def enhance(self, attack: str) -> str:
        return random.choice(
            [
                PromptInjectionTemplate.enhance_1(attack),
                PromptInjectionTemplate.enhance_2(attack),
            ]
        )

    async def a_enhance(self, attack: str) -> str:
        return self.enhance(attack)
    
    def get_enhanced_attacks(self, vulnerability: BaseVulnerability) -> dict:
        simulated_attacks = group_attacks_by_vulnerability_type(
            vulnerability.simulate_attacks()
        )

        result = {}
        for vuln_type, attacks in simulated_attacks.items():
            for attack in attacks:
                attack.input = self.enhance(attack.input)
            result[vuln_type] = attacks

        return result

    async def a_get_enhanced_attacks(self, vulnerability: BaseVulnerability) -> dict:
        simulated_attacks = await vulnerability.a_simulate_attacks()
        grouped_attacks = group_attacks_by_vulnerability_type(simulated_attacks)

        result = {}

        for vuln_type, attacks in grouped_attacks.items():
            enhanced_inputs = await asyncio.gather(
                *(self.a_enhance(attack.input) for attack in attacks)
            )
            for attack, new_input in zip(attacks, enhanced_inputs):
                attack.input = new_input

            result[vuln_type] = attacks

        return result

    def get_name(self) -> str:
        return "Prompt Injection"
