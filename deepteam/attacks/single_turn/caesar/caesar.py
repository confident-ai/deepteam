from deepteam.attacks.single_turn import BaseSingleTurnAttack, AttackParameter
from deepteam.attacks.base_attack import Exploitability


class Caesar(BaseSingleTurnAttack):
    name = "Caesar"
    exploitability = Exploitability.HIGH
    description = "A character-rotation encoding attack that applies a configurable Caesar (ASCII letter-shift) cipher to obfuscate harmful content while keeping the transformation reversible for a capable model."
    parameters = {
        "shift": AttackParameter(
            type="integer",
            default=3,
            description="Number of alphabet positions to shift each letter.",
        ),
    }

    def __init__(self, shift: int = 3, weight: int = 1):
        self.shift = shift
        self.weight = weight

    def enhance(self, attack: str) -> str:
        """Enhance the attack using a Caesar (ASCII letter-shift) cipher."""
        shift = self.shift % 26
        result = []
        for char in attack:
            if "a" <= char <= "z":
                result.append(
                    chr((ord(char) - ord("a") + shift) % 26 + ord("a"))
                )
            elif "A" <= char <= "Z":
                result.append(
                    chr((ord(char) - ord("A") + shift) % 26 + ord("A"))
                )
            else:
                result.append(char)
        return "".join(result)

    async def a_enhance(self, attack: str) -> str:
        return self.enhance(attack)

    def get_name(self) -> str:
        return self.name
