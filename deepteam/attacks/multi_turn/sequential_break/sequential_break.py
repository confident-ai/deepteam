from pydantic import BaseModel
from typing import Optional, Union, List, Dict
import random
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import initialize_model

from deepteam.attacks import BaseAttack
from deepteam.utils import create_progress, update_pbar, add_pbar
from deepteam.attacks.multi_turn.sequential_break.schema import (
    RewrittenDialogue,
    DialogueJudge,
    QuestionBankJudge,
    GameEnvironmentJudge,
    SequentialJailbreakTypeLiteral,
    DialogueTypeLiteral,
)
from deepteam.attacks.multi_turn.sequential_break.template import (
    SequentialBreakTemplate,
)
from deepteam.attacks.attack_simulator.utils import (
    generate,
    a_generate,
)
from deepteam.attacks.multi_turn.utils import enhance_attack, a_enhance_attack
from deepteam.attacks.multi_turn.types import CallbackType
from deepteam.attacks.multi_turn.base_schema import NonRefusal
from deepteam.test_case.test_case import RTTurn
from deepteam.vulnerabilities.types import VulnerabilityType
from deepteam.vulnerabilities import BaseVulnerability


class SequentialJailbreak(BaseAttack):
    def __init__(
        self,
        weight: int = 1,
        type: Optional[SequentialJailbreakTypeLiteral] = None,
        persona: Optional[DialogueTypeLiteral] = None,
        num_turns: int = 5,
        turn_level_attacks: Optional[List[BaseAttack]] = None,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = "gpt-4o-mini",
    ):
        self.weight = weight
        self.multi_turn = True
        self.attack_type = type if type is not None else "dialogue"
        self.persona = persona
        self.num_turns = num_turns
        self.simulator_model = simulator_model
        self.turn_level_attacks = turn_level_attacks

        if self.turn_level_attacks is not None:
            if not isinstance(self.turn_level_attacks, list) or not all(
                attack.multi_turn == False for attack in self.turn_level_attacks
            ):
                raise ValueError(
                    "The 'turn_level_attacks' passed must be a list of single-turn attacks"
                )

        # Validate parameters
        if (
            self.attack_type in ["question_bank", "game_environment"]
            and persona is not None
        ):
            raise ValueError(
                f"{self.attack_type} attack type does not use persona parameter"
            )
        if self.attack_type == "dialogue" and persona is None:
            self.persona = "student"  # Default to student for dialogue

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
        self.model_callback = model_callback

        progress = create_progress()
        with progress:
            task_id = add_pbar(
                progress,
                description=f"🎯 Sequential Jailbreak ({self.attack_type})",
                total=self.num_turns,
            )

            vulnerability_data = (
                f"Vulnerability: {vulnerability} | Type: {vulnerability_type}"
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
            if len(turns) <= 1 or turns[-1].role == "user":
                target_response = model_callback(current_attack, turns)
                turns.append(RTTurn(role="assistant", content=target_response))
            else:
                target_response = turns[-1].content

            eval_flag = False
            eval_percentage = 0

            for _ in range(self.num_turns):
                if self.attack_type == "question_bank":
                    rewrite_prompt = (
                        SequentialBreakTemplate.rewrite_question_bank_prompt(
                            current_attack, vulnerability_data
                        )
                    )
                elif self.attack_type == "game_environment":
                    rewrite_prompt = (
                        SequentialBreakTemplate.rewrite_game_environment_prompt(
                            current_attack, vulnerability_data
                        )
                    )
                else:
                    rewrite_prompt = (
                        SequentialBreakTemplate.rewrite_dialogue_prompt(
                            current_attack, vulnerability_data
                        )
                    )

                rewritten_res: RewrittenDialogue = generate(
                    rewrite_prompt, RewrittenDialogue, self.simulator_model
                )
                rewritten_content = rewritten_res.paraphrased_version

                rewritten_content = rewritten_content.replace(
                    "Paraphrased version:", ""
                ).strip()
                rewritten_content = rewritten_content.replace(
                    "<end>", ""
                ).strip()

                if self.attack_type == "question_bank":
                    current_attack = (
                        SequentialBreakTemplate.get_question_bank_template(
                            rewritten_content
                        )
                    )
                elif self.attack_type == "game_environment":
                    current_attack = (
                        SequentialBreakTemplate.get_game_environment_template(
                            rewritten_content
                        )
                    )
                else:
                    current_attack = (
                        SequentialBreakTemplate.get_dialogue_template(
                            rewritten_content, self.persona
                        )
                    )

                # Randomly enhancing a turn attack
                turn_level_attack = None
                if self.turn_level_attacks and random.random() < 0.5:
                    attack = random.choice(self.turn_level_attacks)
                    turn_level_attack = attack
                    current_attack = enhance_attack(
                        attack, current_attack, self.simulator_model
                    )

                turns.append(RTTurn(role="user", content=current_attack))

                assistant_response = model_callback(current_attack)

                if turn_level_attack is not None:
                    turns.append(
                        RTTurn(
                            role="assistant",
                            content=assistant_response,
                            turn_level_attack=turn_level_attack.get_name(),
                        )
                    )
                else:
                    turns.append(
                        RTTurn(role="assistant", content=assistant_response)
                    )

                non_refusal_prompt = SequentialBreakTemplate.non_refusal(
                    current_attack, assistant_response
                )
                non_refusal_res: NonRefusal = generate(
                    non_refusal_prompt, NonRefusal, self.simulator_model
                )
                if non_refusal_res.classification == "Refusal":
                    update_pbar(progress, task_id, advance_to_end=True)
                    return turns

                if self.attack_type == "question_bank":
                    judge_prompt = SequentialBreakTemplate.question_bank_judge(
                        original_attack,
                        current_attack,
                        assistant_response,
                        vulnerability_data,
                    )
                    judge_res: QuestionBankJudge = generate(
                        judge_prompt, QuestionBankJudge, self.simulator_model
                    )
                    eval_flag = judge_res.classification == "Harmful"
                    eval_percentage = getattr(judge_res, "score", 0)
                elif self.attack_type == "game_environment":
                    judge_prompt = (
                        SequentialBreakTemplate.game_environment_judge(
                            original_attack,
                            current_attack,
                            assistant_response,
                            vulnerability_data,
                        )
                    )
                    judge_res: GameEnvironmentJudge = generate(
                        judge_prompt, GameEnvironmentJudge, self.simulator_model
                    )
                    eval_flag = judge_res.classification == "Harmful"
                    eval_percentage = getattr(judge_res, "score", 0)
                else:
                    judge_prompt = SequentialBreakTemplate.dialogue_judge(
                        original_attack,
                        current_attack,
                        assistant_response,
                        vulnerability_data,
                    )
                    judge_res: DialogueJudge = generate(
                        judge_prompt, DialogueJudge, self.simulator_model
                    )
                    eval_flag = judge_res.classification == "Harmful"
                    eval_percentage = getattr(judge_res, "score", 0)

                if eval_flag:
                    update_pbar(progress, task_id, advance_to_end=True)
                    return turns

                if eval_percentage and eval_percentage >= 80:
                    update_pbar(progress, task_id, advance_to_end=True)
                    return turns

                update_pbar(progress, task_id)

        return turns

    async def _a_get_turns(
        self,
        model_callback: CallbackType,
        turns: Optional[List[RTTurn]] = None,
        vulnerability: str = None,
        vulnerability_type: str = None,
    ) -> List[RTTurn]:
        self.simulator_model, _ = initialize_model(self.simulator_model)
        self.model_callback = model_callback

        # Progress bar setup
        progress = create_progress()
        with progress:
            task_id = add_pbar(
                progress,
                description=f"🎯 Sequential Jailbreak ({self.attack_type})",
                total=self.num_turns,
            )

            vulnerability_data = (
                f"Vulnerability: {vulnerability} | Type: {vulnerability_type}"
            )

            if turns is None:
                turns = []

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
            if len(turns) <= 1 or turns[-1].role == "user":
                target_response = await model_callback(current_attack, turns)
                turns.append(RTTurn(role="assistant", content=target_response))
            else:
                target_response = turns[-1].content

            eval_flag = False
            eval_percentage = 0

            for _ in range(self.num_turns):
                if self.attack_type == "question_bank":
                    rewrite_prompt = (
                        SequentialBreakTemplate.rewrite_question_bank_prompt(
                            current_attack, vulnerability_data
                        )
                    )
                elif self.attack_type == "game_environment":
                    rewrite_prompt = (
                        SequentialBreakTemplate.rewrite_game_environment_prompt(
                            current_attack, vulnerability_data
                        )
                    )
                else:
                    rewrite_prompt = (
                        SequentialBreakTemplate.rewrite_dialogue_prompt(
                            current_attack, vulnerability_data
                        )
                    )

                rewritten_res: RewrittenDialogue = await a_generate(
                    rewrite_prompt, RewrittenDialogue, self.simulator_model
                )
                rewritten_content = rewritten_res.paraphrased_version

                rewritten_content = rewritten_content.replace(
                    "Paraphrased version:", ""
                ).strip()
                rewritten_content = rewritten_content.replace(
                    "<end>", ""
                ).strip()

                if self.attack_type == "question_bank":
                    current_attack = (
                        SequentialBreakTemplate.get_question_bank_template(
                            rewritten_content
                        )
                    )
                elif self.attack_type == "game_environment":
                    current_attack = (
                        SequentialBreakTemplate.get_game_environment_template(
                            rewritten_content
                        )
                    )
                else:
                    current_attack = (
                        SequentialBreakTemplate.get_dialogue_template(
                            rewritten_content, self.persona
                        )
                    )

                # Randomly enhancing a turn attack
                turn_level_attack = None
                if self.turn_level_attacks and random.random() < 0.5:
                    attack = random.choice(self.turn_level_attacks)
                    turn_level_attack = attack
                    current_attack = await a_enhance_attack(
                        attack, current_attack, self.simulator_model
                    )

                turns.append(RTTurn(role="user", content=current_attack))

                assistant_response = await model_callback(current_attack)

                if turn_level_attack is not None:
                    turns.append(
                        RTTurn(
                            role="assistant",
                            content=assistant_response,
                            turn_level_attack=turn_level_attack.get_name(),
                        )
                    )
                else:
                    turns.append(
                        RTTurn(role="assistant", content=assistant_response)
                    )

                non_refusal_prompt = SequentialBreakTemplate.non_refusal(
                    current_attack, assistant_response
                )
                non_refusal_res: NonRefusal = await a_generate(
                    non_refusal_prompt, NonRefusal, self.simulator_model
                )
                if non_refusal_res.classification == "Refusal":
                    update_pbar(progress, task_id, advance_to_end=True)
                    return turns

                if self.attack_type == "question_bank":
                    judge_prompt = SequentialBreakTemplate.question_bank_judge(
                        original_attack,
                        current_attack,
                        assistant_response,
                        vulnerability_data,
                    )
                    judge_res: QuestionBankJudge = await a_generate(
                        judge_prompt, QuestionBankJudge, self.simulator_model
                    )
                    eval_flag = judge_res.classification == "Harmful"
                elif self.attack_type == "game_environment":
                    judge_prompt = (
                        SequentialBreakTemplate.game_environment_judge(
                            original_attack,
                            current_attack,
                            assistant_response,
                            vulnerability_data,
                        )
                    )
                    judge_res: GameEnvironmentJudge = await a_generate(
                        judge_prompt, GameEnvironmentJudge, self.simulator_model
                    )
                    eval_flag = judge_res.classification == "Harmful"
                else:
                    judge_prompt = SequentialBreakTemplate.dialogue_judge(
                        original_attack,
                        current_attack,
                        assistant_response,
                        vulnerability_data,
                    )
                    judge_res: DialogueJudge = await a_generate(
                        judge_prompt, DialogueJudge, self.simulator_model
                    )
                    eval_flag = judge_res.classification == "Harmful"

                update_pbar(progress, task_id)

                if eval_flag == "Harmful":
                    update_pbar(progress, task_id, advance_to_end=True)
                    return turns

                if eval_percentage and eval_percentage >= 80:
                    update_pbar(progress, task_id, advance_to_end=True)
                    return turns

                current_attack = original_attack

                update_pbar(progress, task_id)

        return turns

    def progress(
        self,
        vulnerability: "BaseVulnerability",
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
                    vulnerability_type=vuln_type.value,
                )

            result[vuln_type] = enhanced_turns

        return result

    async def a_progress(
        self,
        vulnerability: "BaseVulnerability",
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
                    vulnerability_type=vuln_type.value,
                )

            result[vuln_type] = enhanced_turns

        return result

    def get_name(self) -> str:
        return "Sequential Jailbreak"
