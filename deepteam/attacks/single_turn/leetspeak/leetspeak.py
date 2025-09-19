from deepteam.attacks.single_turn import BaseSingleTurnAttack
import asyncio
from deepteam.red_teamer.utils import group_attacks_by_vulnerability_type
from deepteam.vulnerabilities import BaseVulnerability


class Leetspeak(BaseSingleTurnAttack):
    def __init__(self, weight: int = 1):
        self.weight = weight

    def enhance(self, attack: str) -> str:
        """Enhance the attack using Leetspeak transformation."""
        leet_map = {
            "a": "4",
            "e": "3",
            "i": "1",
            "o": "0",
            "s": "5",
            "t": "7",
            "l": "1",
            "A": "4",
            "E": "3",
            "I": "1",
            "O": "0",
            "S": "5",
            "T": "7",
            "L": "1",
        }
        return "".join(leet_map.get(char, char) for char in attack)

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

    async def a_get_enhanced_attacks(
        self, vulnerability: BaseVulnerability
    ) -> dict:
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
        return "Leetspeak"
