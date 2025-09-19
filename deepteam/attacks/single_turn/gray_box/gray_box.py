from pydantic import BaseModel
from tqdm import tqdm  # Sync version
from tqdm.asyncio import tqdm as async_tqdm_bar  # Async version
from typing import Optional, Union

import asyncio
from deepteam.red_teamer.utils import group_attacks_by_vulnerability_type
from deepteam.vulnerabilities import BaseVulnerability

from deepeval.metrics.utils import initialize_model
from deepeval.models import DeepEvalBaseLLM

from deepteam.attacks.single_turn import BaseSingleTurnAttack
from deepteam.attacks.single_turn.gray_box.template import GrayBoxTemplate
from deepteam.attacks.single_turn.gray_box.schema import (
    EnhancedAttack,
    ComplianceData,
    IsGrayBox,
)
from deepteam.attacks.attack_simulator.utils import (
    generate,
    a_generate,
)


class GrayBox(BaseSingleTurnAttack):

    def __init__(self, weight: int = 1, max_retries: int = 5):
        self.weight = weight
        self.max_retries = max_retries

    def enhance(
        self,
        attack: str,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = None,
    ) -> str:
        self.simulator_model, _ = initialize_model(simulator_model)

        prompt = GrayBoxTemplate.enhance(attack)

        # Progress bar for retries (total count is double the retries: 1 for generation, 1 for compliance check)
        with tqdm(
            total=self.max_retries * 3,
            desc="...... 🔓 Gray Box",
            unit="step",
            leave=False,
        ) as pbar:

            for _ in range(self.max_retries):
                # Generate the enhanced attack
                res: EnhancedAttack = generate(
                    prompt, EnhancedAttack, self.simulator_model
                )
                enhanced_attack = res.input
                pbar.update(1)  # Update the progress bar for generation

                # Check for compliance using a compliance template
                compliance_prompt = GrayBoxTemplate.non_compliant(
                    res.model_dump()
                )
                compliance_res: ComplianceData = generate(
                    compliance_prompt, ComplianceData, self.simulator_model
                )
                pbar.update(1)  # Update the progress bar for compliance

                # Check if rewritten prompt is a gray box attack
                is_gray_box_prompt = GrayBoxTemplate.is_gray_box(
                    res.model_dump()
                )
                is_gray_box_res: IsGrayBox = generate(
                    is_gray_box_prompt, IsGrayBox, self.simulator_model
                )
                pbar.update(1)  # Update the progress bar for is gray box attack

                if (
                    not compliance_res.non_compliant
                    and is_gray_box_res.is_gray_box
                ):
                    # If it's compliant and is a gray box attack, return the enhanced prompt
                    return enhanced_attack

        # If all retries fail, return the original attack
        return attack

    async def a_enhance(
        self,
        attack: str,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = None,
    ) -> str:
        self.simulator_model, _ = initialize_model(simulator_model)
        prompt = GrayBoxTemplate.enhance(attack)

        # Async progress bar for retries (double the count to cover both generation and compliance check)
        pbar = async_tqdm_bar(
            total=self.max_retries * 3,
            desc="...... 🔓 Gray Box",
            unit="step",
            leave=False,
        )

        try:
            for _ in range(self.max_retries):
                # Generate the enhanced attack asynchronously
                res: EnhancedAttack = await a_generate(
                    prompt, EnhancedAttack, self.simulator_model
                )
                enhanced_attack = res.input
                pbar.update(1)  # Update the progress bar for generation

                # Check for compliance using a compliance template
                compliance_prompt = GrayBoxTemplate.non_compliant(
                    res.model_dump()
                )
                compliance_res: ComplianceData = await a_generate(
                    compliance_prompt, ComplianceData, self.simulator_model
                )
                pbar.update(1)  # Update the progress bar for compliance

                # Check if rewritten prompt is a gray box attack
                is_gray_box_prompt = GrayBoxTemplate.is_gray_box(
                    res.model_dump()
                )
                is_gray_box_res: IsGrayBox = await a_generate(
                    is_gray_box_prompt, IsGrayBox, self.simulator_model
                )
                pbar.update(1)  # Update the progress bar for is gray box attack

                if (
                    not compliance_res.non_compliant
                    and is_gray_box_res.is_gray_box
                ):
                    # If it's compliant and is a gray box attack, return the enhanced prompt
                    return enhanced_attack

        finally:
            # Close the progress bar after the loop
            pbar.close()

        # If all retries fail, return the original attack
        return attack

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
        return "Gray Box"
