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
from deepteam.attacks.single_turn.roleplay.template import (
    RoleplayTemplate,
)
from deepteam.attacks.single_turn.roleplay.schema import (
    EnhancedAttack,
    ComplianceData,
    IsRoleplay,
)
from deepteam.attacks.attack_simulator.utils import (
    generate,
    a_generate,
)


class Roleplay(BaseSingleTurnAttack):

    def __init__(
        self,
        persona: Optional[str] = "A helpful assistant",
        role: Optional[str] = "A helpful assistant",
        weight: int = 1,
        max_retries: int = 3,
    ):
        self.weight = weight
        self.max_retries = max_retries
        self.persona = persona
        self.role = role

    def enhance(
        self,
        attack: str,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = None,
    ) -> str:
        self.simulator_model, _ = initialize_model(simulator_model)
        prompt = RoleplayTemplate.enhance(attack, self.persona, self.role)

        # Progress bar for retries (total count is triple the retries: 1 for generation, 1 for compliance check, 1 for roleplay check)
        with tqdm(
            total=self.max_retries * 3,
            desc="...... 🎭 Roleplay",
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
                compliance_prompt = RoleplayTemplate.non_compliant(
                    res.model_dump()
                )
                compliance_res: ComplianceData = generate(
                    compliance_prompt, ComplianceData, self.simulator_model
                )
                pbar.update(1)  # Update the progress bar for compliance

                # Check if rewritten prompt is a roleplay attack
                is_roleplay_prompt = RoleplayTemplate.is_roleplay(
                    res.model_dump()
                )
                is_roleplay_res: IsRoleplay = generate(
                    is_roleplay_prompt, IsRoleplay, self.simulator_model
                )
                pbar.update(1)  # Update the progress bar

                if (
                    not compliance_res.non_compliant
                    and is_roleplay_res.is_roleplay
                ):
                    # If it's compliant and is a roleplay attack, return the enhanced prompt
                    return enhanced_attack

        # If all retries fail, return the original attack
        return attack

    async def a_enhance(
        self,
        attack: str,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = None,
    ) -> str:
        self.simulator_model, _ = initialize_model(simulator_model)
        prompt = RoleplayTemplate.enhance(attack, self.persona, self.role)

        # Async progress bar for retries (triple the count to cover generation, compliance check, and roleplay check)
        pbar = async_tqdm_bar(
            total=self.max_retries * 3,
            desc="...... 🎭 Roleplay",
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
                compliance_prompt = RoleplayTemplate.non_compliant(
                    res.model_dump()
                )
                compliance_res: ComplianceData = await a_generate(
                    compliance_prompt, ComplianceData, self.simulator_model
                )
                pbar.update(1)  # Update the progress bar for compliance

                # Check if rewritten prompt is a roleplay attack
                is_roleplay_prompt = RoleplayTemplate.is_roleplay(
                    res.model_dump()
                )
                is_roleplay_res: IsRoleplay = await a_generate(
                    is_roleplay_prompt, IsRoleplay, self.simulator_model
                )
                pbar.update(1)  # Update the progress bar

                if (
                    not compliance_res.non_compliant
                    and is_roleplay_res.is_roleplay
                ):
                    # If it's compliant and is a roleplay attack, return the enhanced prompt
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
        return "Roleplay"
