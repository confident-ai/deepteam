from tqdm import tqdm
from typing import Optional, Union, List, Dict
import random

from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import initialize_model

from deepteam.attacks import BaseAttack
from deepteam.attacks.multi_turn.linear_jailbreaking.schema import (
    Improvement,
    Feedback,
)
from deepteam.attacks.multi_turn.linear_jailbreaking.template import (
    JailBreakingTemplate,
)
from deepteam.attacks.attack_simulator.utils import (
    generate,
    a_generate,
)
from deepteam.attacks.multi_turn.types import CallbackType
from deepteam.attacks import BaseAttack
from deepteam.attacks.multi_turn.utils import enhance_attack, a_enhance_attack
from deepteam.attacks.multi_turn.base_schema import NonRefusal
from deepteam.attacks.multi_turn.base_template import BaseMultiTurnTemplate
from deepteam.errors import ModelRefusalError
from deepteam.test_case.test_case import RTTurn
from deepteam.vulnerabilities.types import VulnerabilityType
from deepteam.vulnerabilities import BaseVulnerability


class LinearJailbreaking(BaseAttack):
    def __init__(
        self,
        weight: int = 1,
        num_turns: int = 5,
        attacks: Optional[List[BaseAttack]] = None,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = "gpt-4o-mini",
    ):
        self.weight = weight
        self.multi_turn = True
        self.num_turns = num_turns
        self.simulator_model = simulator_model
        self.attacks = attacks

        if self.attacks is not None:
            if not isinstance(self.attacks, list) or not all(
                attack.multi_turn == False for attack in self.attacks
            ):
                raise ValueError(
                    "The 'attacks' passed must be a list of single-turn attacks"
                )

    def _get_turns(
        self,
        model_callback: CallbackType,
        turns: Optional[List[RTTurn]] = None,
        vulnerability: str = None,
        vulnerability_type: str = None,
    ) -> List[RTTurn]:
        if turns is None:
            turns = []

        self.simulator_model, _ = initialize_model(self.simulator_model)

        vulnerability_data = (
            f"Vulnerability: {vulnerability} | Type: {vulnerability_type}"
        )

        pbar = tqdm(
            total=self.num_turns,
            desc="...... ⛓️  Linear Jailbreaking",
            leave=False,
        )

        # Get current attack from last user turn (simple reverse search)
        current_attack = None
        original_attack = None
        for turn in reversed(turns):
            if turn.role == "user":
                current_attack = turn.content
                original_attack = turn.content
                break

        if current_attack is None:
            raise ValueError("No user turns found in conversation history")

        # If the last turn is from user, we need a model response before simulation
        if len(turns) == 0 or turns[-1].role == "user":
            target_response = model_callback(current_attack, turns)
            turns.append(RTTurn(role="assistant", content=target_response))
        else:
            target_response = turns[-1].content

        # Main simulation loop
        for _ in range(self.num_turns):
            judge_prompt = JailBreakingTemplate.linear_judge(
                original_attack,
                current_attack,
                target_response,
                vulnerability_data,
            )
            feedback: Feedback = generate(
                judge_prompt, Feedback, self.simulator_model
            )

            if feedback.jailbroken:
                pbar.n = pbar.total
                pbar.close()
                break

            improvement_prompt = JailBreakingTemplate.improvement_prompt(
                turns, feedback.suggestion, vulnerability_data
            )
            improvement: Improvement = generate(
                improvement_prompt, Improvement, self.simulator_model
            )
            next_attack = improvement.new_prompt

            non_refusal_prompt = BaseMultiTurnTemplate.non_refusal(
                original_attack, next_attack
            )
            non_refusal_res: NonRefusal = generate(
                non_refusal_prompt, NonRefusal, self.simulator_model
            )

            if non_refusal_res.classification == "Refusal":
                pbar.n = pbar.total
                pbar.close()
                raise ModelRefusalError(entity=self.get_name())

            current_attack = next_attack

            # Randomly enhancing a turn attack
            if self.attacks and random.random() < 0.5:
                attack = random.choice(self.attacks)
                current_attack = enhance_attack(
                    attack, current_attack, self.simulator_model
                )

            target_response = model_callback(current_attack, turns)
            turns.append(RTTurn(role="user", content=current_attack))
            turns.append(RTTurn(role="assistant", content=target_response))

            pbar.update(1)

        pbar.close()
        return turns

    async def _a_get_turns(
        self,
        model_callback: CallbackType,
        turns: Optional[List[RTTurn]] = None,
        vulnerability: str = None,
        vulnerability_type: str = None,
    ) -> List[RTTurn]:
        if turns is None:
            turns = []

        self.simulator_model, _ = initialize_model(self.simulator_model)

        vulnerability_data = (
            f"Vulnerability: {vulnerability} | Type: {vulnerability_type}"
        )

        pbar = tqdm(
            total=self.num_turns,
            desc="...... ⛓️  Linear Jailbreaking",
            leave=False,
        )

        # Get current attack from last user turn (simple reverse search)
        current_attack = None
        original_attack = None
        for turn in reversed(turns):
            if turn.role == "user":
                current_attack = turn.content
                original_attack = turn.content
                break

        if current_attack is None:
            raise ValueError("No user turns found in conversation history")

        # If last turn is user, generate a model response before the loop
        if len(turns) == 0 or turns[-1].role == "user":
            target_response = await model_callback(current_attack, turns)
            turns.append(RTTurn(role="assistant", content=target_response))
        else:
            target_response = turns[-1].content

        # Main simulation loop
        for _ in range(self.num_turns):
            judge_prompt = JailBreakingTemplate.linear_judge(
                original_attack,
                current_attack,
                target_response,
                vulnerability_data,
            )
            feedback: Feedback = await a_generate(
                judge_prompt, Feedback, self.simulator_model
            )

            if feedback.jailbroken:
                pbar.n = pbar.total
                pbar.close()
                break

            improvement_prompt = JailBreakingTemplate.improvement_prompt(
                turns, feedback.suggestion, vulnerability_data
            )
            improvement: Improvement = await a_generate(
                improvement_prompt, Improvement, self.simulator_model
            )
            next_attack = improvement.new_prompt

            non_refusal_prompt = BaseMultiTurnTemplate.non_refusal(
                original_attack, next_attack
            )
            non_refusal_res: NonRefusal = await a_generate(
                non_refusal_prompt, NonRefusal, self.simulator_model
            )

            if non_refusal_res.classification == "Refusal":
                pbar.n = pbar.total
                pbar.close()
                raise ModelRefusalError(entity=self.get_name())

            current_attack = next_attack

            # Randomly enhancing a turn attack
            if self.attacks and random.random() < 0.5:
                attack = random.choice(self.attacks)
                current_attack = await a_enhance_attack(
                    attack, current_attack, self.simulator_model
                )

            target_response = await model_callback(current_attack, turns)
            turns.append(RTTurn(role="user", content=current_attack))
            turns.append(RTTurn(role="assistant", content=target_response))

            pbar.update(1)

        pbar.close()
        return turns

    def enhance(
        self,
        vulnerability: BaseVulnerability,
        model_callback: CallbackType,
        turns: Optional[List[RTTurn]] = None,
    ) -> Dict[VulnerabilityType, List[RTTurn]]:
        from deepteam.red_teamer.utils import (
            group_attacks_by_vulnerability_type,
        )

        # Simulate and group attacks
        simulated_attacks = group_attacks_by_vulnerability_type(
            vulnerability.simulate_attacks()
        )

        result = {}

        for vuln_type, attacks in simulated_attacks.items():
            for attack in attacks:
                # Defensive copy to avoid mutating external turns
                inner_turns = list(turns) if turns else []

                # Case 1: No turns, or last is user -> create assistant response
                if len(inner_turns) == 0 or inner_turns[-1].role == "user":
                    inner_turns = [RTTurn(role="user", content=attack.input)]
                    assistant_response = model_callback(
                        attack.input, inner_turns
                    )
                    inner_turns.append(
                        RTTurn(role="assistant", content=assistant_response)
                    )

                # Case 2: Last is assistant -> find preceding user
                elif inner_turns[-1].role == "assistant":
                    user_turn_content = None
                    for turn in reversed(inner_turns[:-1]):
                        if turn.role == "user":
                            user_turn_content = turn.content
                            break

                    if user_turn_content:
                        inner_turns = [
                            RTTurn(role="user", content=user_turn_content),
                            RTTurn(
                                role="assistant",
                                content=inner_turns[-1].content,
                            ),
                        ]
                    else:
                        # Fallback if no user found
                        inner_turns = [
                            RTTurn(role="user", content=attack.input)
                        ]
                        assistant_response = model_callback(
                            attack.input, inner_turns
                        )
                        inner_turns.append(
                            RTTurn(role="assistant", content=assistant_response)
                        )

                else:
                    # Unrecognized state — fallback to default
                    inner_turns = [RTTurn(role="user", content=attack.input)]
                    assistant_response = model_callback(
                        attack.input, inner_turns
                    )
                    inner_turns.append(
                        RTTurn(role="assistant", content=assistant_response)
                    )

                # Run enhancement loop and assign full turn history
                vulnerability_name = vulnerability.get_name()
                enhanced_turns = self._get_turns(
                    model_callback=model_callback,
                    turns=inner_turns,
                    vulnerability=vulnerability_name,
                    vulnerability_type=vuln_type,
                )

            result[vuln_type] = enhanced_turns

        return result

    async def a_enhance(
        self,
        vulnerability: BaseVulnerability,
        model_callback: CallbackType,
        turns: Optional[List[RTTurn]] = None,
    ) -> Dict[VulnerabilityType, List[RTTurn]]:
        from deepteam.red_teamer.utils import (
            group_attacks_by_vulnerability_type,
        )

        # Simulate and group attacks asynchronously
        simulated_attacks = await vulnerability.a_simulate_attacks()
        grouped_attacks = group_attacks_by_vulnerability_type(simulated_attacks)

        result = {}

        for vuln_type, attacks in grouped_attacks.items():
            for attack in attacks:
                # Defensive copy to avoid mutating external turns
                inner_turns = list(turns) if turns else []

                # Case 1: No turns, or last is user -> create assistant response
                if len(inner_turns) == 0 or inner_turns[-1].role == "user":
                    inner_turns = [RTTurn(role="user", content=attack.input)]
                    assistant_response = await model_callback(
                        attack.input, inner_turns
                    )
                    inner_turns.append(
                        RTTurn(role="assistant", content=assistant_response)
                    )

                # Case 2: Last is assistant -> find preceding user
                elif inner_turns[-1].role == "assistant":
                    user_turn_content = None
                    for turn in reversed(inner_turns[:-1]):
                        if turn.role == "user":
                            user_turn_content = turn.content
                            break

                    if user_turn_content:
                        inner_turns = [
                            RTTurn(role="user", content=user_turn_content),
                            RTTurn(
                                role="assistant",
                                content=inner_turns[-1].content,
                            ),
                        ]
                    else:
                        # Fallback if no user found
                        inner_turns = [
                            RTTurn(role="user", content=attack.input)
                        ]
                        assistant_response = await model_callback(
                            attack.input, inner_turns
                        )
                        inner_turns.append(
                            RTTurn(role="assistant", content=assistant_response)
                        )

                else:
                    # Unrecognized state — fallback to default
                    inner_turns = [RTTurn(role="user", content=attack.input)]
                    assistant_response = await model_callback(
                        attack.input, inner_turns
                    )
                    inner_turns.append(
                        RTTurn(role="assistant", content=assistant_response)
                    )

                # Run enhancement loop and assign full turn history
                vulnerability_name = vulnerability.get_name()
                enhanced_turns = await self._a_get_turns(
                    model_callback=model_callback,
                    turns=inner_turns,
                    vulnerability=vulnerability_name,
                    vulnerability_type=vuln_type,
                )

            result[vuln_type] = enhanced_turns

        return result

    def get_name(self) -> str:
        return "Linear Jailbreaking"
