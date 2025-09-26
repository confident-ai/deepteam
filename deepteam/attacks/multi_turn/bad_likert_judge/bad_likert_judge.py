from pydantic import BaseModel
from tqdm import tqdm
from typing import Optional, Union, List, Dict
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import initialize_model
import random
from deepteam.attacks import BaseAttack
from deepteam.attacks.multi_turn.bad_likert_judge.schema import (
    BLJJudge,
)
from deepteam.attacks.multi_turn.bad_likert_judge.template import (
    BadLikertJudgeTemplate,
)
from deepteam.attacks.attack_simulator.utils import (
    generate,
    a_generate,
)
from deepteam.attacks.multi_turn.utils import enhance_attack, a_enhance_attack
from deepteam.attacks.multi_turn.types import CallbackType
from deepteam.attacks.multi_turn.base_schema import NonRefusal
from deepteam.errors import ModelRefusalError
from deepteam.test_case.test_case import RTTurn
from deepteam.vulnerabilities.types import VulnerabilityType
from deepteam.vulnerabilities import BaseVulnerability


class BadLikertJudge(BaseAttack):
    def __init__(
        self,
        weight: int = 1,
        category: str = "bias",
        num_turns: int = 5,
        enable_refinement: bool = True,
        attacks: Optional[List[BaseAttack]] = None,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = "gpt-4o-mini",
    ):
        self.weight = weight
        self.multi_turn = True
        self.category = category
        self.num_turns = num_turns
        self.enable_refinement = enable_refinement
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
            desc=f"🎯 Bad Likert Judge ({self.category})",
            leave=False,
        )

        # Extract last user turn as the original attack
        current_attack = None
        original_attack = None
        for turn in reversed(turns):
            if turn.role == "user":
                current_attack = turn.content
                original_attack = turn.content
                break

        if current_attack is None:
            raise ValueError("No user turn found in the conversation history")

        # Generate assistant response if needed
        if len(turns) == 0 or turns[-1].role == "user":
            assistant_response = model_callback(current_attack, turns)
            turns.append(RTTurn(role="assistant", content=assistant_response))
        else:
            assistant_response = turns[-1].content

        for _ in range(self.num_turns):
            # Judge current assistant response
            judge_prompt = BadLikertJudgeTemplate.likert_judge(
                original_attack,
                current_attack,
                assistant_response,
                vulnerability_data,
            )
            judge_res: BLJJudge = generate(
                judge_prompt, BLJJudge, self.simulator_model
            )

            if judge_res.classification == "Harmful":
                pbar.n = pbar.total
                pbar.close()
                break

            # Generate new prompt variation
            improvement_prompt = (
                BadLikertJudgeTemplate.likert_generate_examples(
                    current_attack, self.category, vulnerability_data
                )
            )
            improvement_response = model_callback(improvement_prompt, turns)

            next_attack = (
                BadLikertJudgeTemplate.extract_score_3_example(
                    improvement_response
                )
                or improvement_response
            )

            # Refusal check
            refusal_prompt = BadLikertJudgeTemplate.non_refusal(
                current_attack, next_attack
            )
            refusal_res: NonRefusal = generate(
                refusal_prompt, NonRefusal, self.simulator_model
            )

            if refusal_res.classification == "Refusal":
                raise ModelRefusalError(entity=self.get_name())

            if self.enable_refinement and self.num_turns >= 3:
                current_attack = BadLikertJudgeTemplate.likert_refine_score_3(
                    next_attack, vulnerability_data
                )
            else:
                current_attack = next_attack

            # Randomly enhancing a turn attack
            if self.attacks and random.random() < 0.5:
                attack = random.choice(self.attacks)
                enhanced_attack = enhance_attack(
                    attack, enhanced_attack, self.simulator_model
                )

            assistant_response = model_callback(current_attack, turns)
            turns.append(RTTurn(role="user", content=current_attack))
            turns.append(RTTurn(role="assistant", content=assistant_response))

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
            desc=f"🎯 Bad Likert Judge ({self.category})",
            leave=False,
        )

        # Extract last user turn
        current_attack = None
        original_attack = None
        for turn in reversed(turns):
            if turn.role == "user":
                current_attack = turn.content
                original_attack = turn.content
                break

        if current_attack is None:
            raise ValueError("No user turn found")

        # Ensure assistant response
        if len(turns) == 0 or turns[-1].role == "user":
            assistant_response = await model_callback(current_attack, turns)
            turns.append(RTTurn(role="assistant", content=assistant_response))
        else:
            assistant_response = turns[-1].content

        for _ in range(self.num_turns):
            judge_prompt = BadLikertJudgeTemplate.likert_judge(
                original_attack,
                current_attack,
                assistant_response,
                vulnerability_data,
            )
            judge_res: BLJJudge = await a_generate(
                judge_prompt, BLJJudge, self.simulator_model
            )

            if judge_res.classification == "Harmful":
                pbar.n = pbar.total
                pbar.close()
                break

            # Generate new attack candidate
            improvement_prompt = (
                BadLikertJudgeTemplate.likert_generate_examples(
                    current_attack, self.category, vulnerability_data
                )
            )
            improvement_response = await model_callback(
                improvement_prompt, turns
            )

            next_attack = (
                BadLikertJudgeTemplate.extract_score_3_example(
                    improvement_response
                )
                or improvement_response
            )

            # Check for refusal
            refusal_prompt = BadLikertJudgeTemplate.non_refusal(
                current_attack, next_attack
            )
            refusal_res: NonRefusal = await a_generate(
                refusal_prompt, NonRefusal, self.simulator_model
            )

            if refusal_res.classification == "Refusal":
                raise ModelRefusalError(entity=self.get_name())

            if self.enable_refinement and self.num_turns >= 3:
                current_attack = BadLikertJudgeTemplate.likert_refine_score_3(
                    next_attack, vulnerability_data
                )
            else:
                current_attack = next_attack

            # Randomly enhancing a turn attack
            if self.attacks and random.random() < 0.5:
                attack = random.choice(self.attacks)
                enhanced_attack = await a_enhance_attack(
                    attack, enhanced_attack, self.simulator_model
                )

            assistant_response = await model_callback(current_attack, turns)
            turns.append(RTTurn(role="user", content=current_attack))
            turns.append(RTTurn(role="assistant", content=assistant_response))

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
    ) -> Dict[VulnerabilityType, List[List[RTTurn]]]:
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
        return "Bad Likert Judge"

    @classmethod
    def get_supported_categories(cls) -> list[str]:
        """
        Returns a list of example supported vulnerability categories.
        Note: The system supports any category - this is just a sample of common ones.
        """
        return [
            # Core vulnerability types
            "bias",
            "religion",
            "politics",
            "gender",
            "race",
            "toxicity",
            "profanity",
            "insults",
            "threats",
            "mockery",
            "illegal_activity",
            "weapons",
            "illegal_drugs",
            "violent_crimes",
            "cybercrime",
            "unauthorized_access",
            "bfla",
            "bola",
            "rbac",
            "sql_injection",
            "shell_injection",
            "personal_safety",
            "bullying",
            "self-harm",
            "unsafe_practices",
            "stalking",
            "graphic_content",
            "sexual_content",
            "pornographic_content",
            "misinformation",
            "factual_errors",
            "unsupported_claims",
            "pii_leakage",
            "database_access",
            "direct_disclosure",
            "session_leak",
            "prompt_leakage",
            "secrets_and_credentials",
            "instructions",
            "guard_exposure",
            "excessive_agency",
            "functionality",
            "permissions",
            "autonomy",
            "robustness",
            "input_overreliance",
            "hijacking",
            "intellectual_property",
            "copyright_violations",
            "trademark_infringement",
            "competition",
            "competitor_mention",
            "market_manipulation",
            "malware",
            "hacking",
            "exploitation",
            "violence",
            "harassment",
        ]
