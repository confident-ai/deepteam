from typing import Optional, Union

from deepeval.metrics.utils import initialize_model
from deepeval.models import DeepEvalBaseLLM

from deepteam.attacks.base_attack import Exploitability
from deepteam.attacks.single_turn import AttackParameter, BaseSingleTurnAttack
from deepteam.attacks.single_turn.bijection_learning.template import (
    BijectionLearningTemplate,
    build_bijection,
)


class BijectionLearning(BaseSingleTurnAttack):
    name = "Bijection Learning"
    exploitability = Exploitability.HIGH
    description = "An in-context cipher attack that teaches the target a random letter-level bijection, sends the harmful request encoded in it, and reads the encoded reply back. Cipher complexity is a tunable dial, so it is a family of encodings rather than one fixed scheme."
    parameters = {
        "dispersion": AttackParameter(
            type="integer",
            default=14,
            description="How many of the 26 letters are enciphered (0=plaintext, 26=all). The complexity dial.",
        ),
        "encoding_length": AttackParameter(
            type="integer",
            default=2,
            description="Digits per enciphered letter. 0 keeps a letter->letter permutation.",
        ),
    }

    def __init__(
        self,
        weight: int = 1,
        dispersion: int = 14,
        encoding_length: int = 2,
    ):
        self.weight = weight
        self.dispersion = dispersion
        self.encoding_length = encoding_length

    def enhance(
        self,
        attack: str,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = None,
    ) -> str:
        return self._build_prompt(attack)

    async def a_enhance(self, attack, simulator_model=None):
        return self.enhance(attack, simulator_model)


    def _build_prompt(self, attack: str) -> str:
        mapping = build_bijection(self.dispersion, self.encoding_length)
        self.mapping = mapping

        prompt = BijectionLearningTemplate.system_prompt(mapping, attack)
        return prompt

    def decode(
        self,
        encoded_response: str,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = None,
    ) -> str:
        """Decode a target reply written in the last-built bijection. Optional
        helper; needs `enhance` to have been called first."""
        from deepteam.attacks.attack_simulator.utils import generate
        from deepteam.attacks.single_turn.bijection_learning.schema import (
            DecodedResponse,
        )

        model, _ = initialize_model(simulator_model)
        res: DecodedResponse = generate(
            BijectionLearningTemplate.decode_prompt(
                encoded_response, self.mapping
            ),
            DecodedResponse,
            model,
        )
        return res.decoded

    def get_name(self) -> str:
        return self.name
