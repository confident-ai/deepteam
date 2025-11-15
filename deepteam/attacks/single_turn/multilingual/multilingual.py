from pydantic import BaseModel
from typing import Optional, Union

from deepeval.metrics.utils import initialize_model
from deepeval.models import DeepEvalBaseLLM

from deepteam.utils import create_progress, update_pbar, add_pbar
from deepteam.attacks.single_turn import BaseSingleTurnAttack
from deepteam.attacks.single_turn.multilingual.template import (
    MultilingualTemplate,
)
from deepteam.attacks.single_turn.multilingual.schema import (
    EnhancedAttack,
    ComplianceData,
    IsTranslation,
)
from deepteam.attacks.attack_simulator.utils import (
    generate,
    a_generate,
)


class Multilingual(BaseSingleTurnAttack):
    def __init__(self, weight: int = 1, max_retries: int = 5):
        self.weight = weight
        self.max_retries = max_retries

    def enhance(
        self,
        attack: str,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = None,
    ) -> str:
        self.simulator_model, _ = initialize_model(simulator_model)
        prompt = MultilingualTemplate.enhance(attack)

        # Progress bar for retries (total count is double the retries: 1 for generation, 1 for compliance check)
        progress = create_progress()
        with progress:
            task_id = add_pbar(
                progress,
                description="...... 🌍 Multilingual Enhancement",
                total=self.max_retries * 3,
            )

            for _ in range(self.max_retries):
                # Generate the enhanced prompt
                res: EnhancedAttack = generate(
                    prompt, EnhancedAttack, self.simulator_model
                )
                enhanced_attack = res.input
                update_pbar(
                    progress, task_id
                )  # Update the progress bar for generation

                # Check for compliance using a compliance template
                compliance_prompt = MultilingualTemplate.non_compliant(
                    res.model_dump()
                )
                compliance_res: ComplianceData = generate(
                    compliance_prompt, ComplianceData, self.simulator_model
                )
                update_pbar(
                    progress, task_id
                )  # Update the progress bar for compliance

                # Check if rewritten prompt is a translation
                is_translation_prompt = MultilingualTemplate.is_translation(
                    res.model_dump()
                )
                is_translation_res: IsTranslation = generate(
                    is_translation_prompt, IsTranslation, self.simulator_model
                )
                update_pbar(
                    progress, task_id
                )  # Update the progress bar for is a translation

                if (
                    not compliance_res.non_compliant
                    and is_translation_res.is_translation
                ):
                    # If it's compliant and is a translation, return the enhanced prompt
                    update_pbar(progress, task_id, advance_to_end=True)
                    return enhanced_attack

        # If all retries fail, return the original prompt
        return attack

    async def a_enhance(
        self,
        attack: str,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = None,
    ) -> str:
        self.simulator_model, _ = initialize_model(simulator_model)
        prompt = MultilingualTemplate.enhance(attack)

        # Async progress bar for retries (double the count to cover both generation and compliance check)
        progress = create_progress()
        with progress:
            task_id = add_pbar(
                progress,
                description="...... 🌍 Multilingual Enhancement",
                total=self.max_retries * 3,
            )

            try:
                for _ in range(self.max_retries):
                    # Generate the enhanced prompt asynchronously
                    res: EnhancedAttack = await a_generate(
                        prompt, EnhancedAttack, self.simulator_model
                    )
                    enhanced_attack = res.input
                    update_pbar(
                        progress, task_id
                    )  # Update the progress bar for generation

                    # Check for compliance using a compliance template
                    compliance_prompt = MultilingualTemplate.non_compliant(
                        res.model_dump()
                    )
                    compliance_res: ComplianceData = await a_generate(
                        compliance_prompt, ComplianceData, self.simulator_model
                    )
                    update_pbar(
                        progress, task_id
                    )  # Update the progress bar for compliance

                    # Check if rewritten prompt is a translation
                    is_translation_prompt = MultilingualTemplate.is_translation(
                        res.model_dump()
                    )
                    is_translation_res: IsTranslation = await a_generate(
                        is_translation_prompt,
                        IsTranslation,
                        self.simulator_model,
                    )
                    update_pbar(
                        progress, task_id
                    )  # Update the progress bar for is a translation

                    if (
                        not compliance_res.non_compliant
                        and is_translation_res.is_translation
                    ):
                        # If it's compliant and is a translation, return the enhanced prompt
                        update_pbar(progress, task_id, advance_to_end=True)
                        return enhanced_attack

            finally:
                # Close the progress bar after the loop
                update_pbar(progress, task_id, advance_to_end=True)

        # If all retries fail, return the original prompt
        return attack

    def get_name(self) -> str:
        return "Multilingual"
