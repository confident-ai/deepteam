from tqdm import tqdm
from typing import Optional, Union, List


from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import initialize_model
from deepeval.test_case import Turn

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
from deepteam.attacks.multi_turn.utils import update_turn_history
from deepteam.attacks.multi_turn.base_schema import NonRefusal
from deepteam.attacks.multi_turn.base_template import BaseMultiTurnTemplate
from deepteam.errors import ModelRefusalError
from deepteam.test_case.test_case import RTTurn


class LinearJailbreaking(BaseAttack):
    def __init__(
        self,
        weight: int = 1,
        num_turns: int = 5,
    ):
        self.weight = weight
        self.num_turns = num_turns

    def enhance(
        self,
        model_callback: CallbackType,
        turns: Optional[List[RTTurn]] = None,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = None,
    ) -> List[RTTurn]:
        if turns is None:
            turns = []

        self.simulator_model, _ = initialize_model(simulator_model)

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
                original_attack, current_attack, target_response
            )
            feedback: Feedback = generate(
                judge_prompt, Feedback, self.simulator_model
            )

            if feedback.jailbroken:
                pbar.n = pbar.total
                pbar.close()
                break

            improvement_prompt = JailBreakingTemplate.improvement_prompt(
                turns, feedback.suggestion
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
            target_response = model_callback(current_attack, turns)
            turns.append(RTTurn(role="user", content=current_attack))
            turns.append(RTTurn(role="assistant", content=target_response))

            pbar.update(1)

        pbar.close()
        return turns

    async def a_enhance(
        self,
        model_callback: CallbackType,
        turns: Optional[List[RTTurn]] = None,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = None,
    ) -> List[RTTurn]:
        if turns is None:
            turns = []

        self.simulator_model, _ = initialize_model(simulator_model)

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
                original_attack, current_attack, target_response
            )
            feedback: Feedback = await a_generate(
                judge_prompt, Feedback, self.simulator_model
            )

            if feedback.jailbroken:
                pbar.n = pbar.total
                pbar.close()
                break

            improvement_prompt = JailBreakingTemplate.improvement_prompt(
                turns, feedback.suggestion
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

            target_response = await model_callback(current_attack, turns)
            turns.append(RTTurn(role="user", content=current_attack))
            turns.append(RTTurn(role="assistant", content=target_response))

            pbar.update(1)

        pbar.close()
        return turns

    def get_name(self) -> str:
        return "Linear Jailbreaking"
