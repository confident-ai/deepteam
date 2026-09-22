import random
from typing import Optional, Union

from deepeval.metrics.utils import initialize_model
from deepeval.models import DeepEvalBaseLLM

from deepteam.attacks.attack_simulator.utils import a_generate, generate
from deepteam.attacks.base_attack import Exploitability
from deepteam.attacks.single_turn import AttackParameter, BaseSingleTurnAttack
from deepteam.attacks.single_turn.chain_of_thought_hijacking.schema import (
    GeneratedPuzzle,
    IsBenignPuzzle,
    PuzzleType,
    ReasoningLength,
)
from deepteam.attacks.single_turn.chain_of_thought_hijacking.template import (
    PUZZLE_TYPES,
    REASONING_DIRECTIVES,
    ChainOfThoughtHijackingTemplate,
)
from deepteam.utils import add_pbar, create_progress, update_pbar


class ChainOfThoughtHijacking(BaseSingleTurnAttack):
    name = "Chain-of-Thought Hijacking"
    exploitability = Exploitability.HIGH
    description = "A reasoning-dilution attack that buries the harmful request behind a long benign puzzle, so the target spends its reasoning budget on the puzzle and its refusal signal weakens before the payload is reached."
    parameters = {
        "puzzle_type": AttackParameter(
            type="enum",
            options=list(PUZZLE_TYPES),
            description="Family of benign reasoning puzzle used as the preface. Random each call when unset.",
        ),
        "reasoning_length": AttackParameter(
            type="enum",
            default="extended",
            options=list(REASONING_DIRECTIVES),
            description="How hard the prompt pushes for a long reasoning trace.",
        ),
        "target_tokens": AttackParameter(
            type="integer",
            default=3000,
            description="Roughly how many tokens of reasoning the puzzle should demand.",
        ),
        "role_preface": AttackParameter(
            type="string",
            description="Optional role or context line placed before the puzzle.",
        ),
    }

    def __init__(
        self,
        weight: int = 1,
        max_retries: int = 3,
        puzzle_type: Optional[PuzzleType] = None,
        reasoning_length: ReasoningLength = "extended",
        target_tokens: int = 3000,
        role_preface: Optional[str] = None,
    ):
        self.weight = weight
        self.max_retries = max_retries
        self.puzzle_type = puzzle_type
        self.reasoning_length = reasoning_length
        self.target_tokens = target_tokens
        self.role_preface = role_preface

        if puzzle_type is not None and puzzle_type not in PUZZLE_TYPES:
            raise ValueError(
                f"'puzzle_type' must be one of {list(PUZZLE_TYPES)}"
            )
        if reasoning_length not in REASONING_DIRECTIVES:
            raise ValueError(
                f"'reasoning_length' must be one of {list(REASONING_DIRECTIVES)}"
            )

    def enhance(
        self,
        attack: str,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = None,
    ) -> str:
        self.simulator_model, _ = initialize_model(simulator_model)
        puzzle_type = self._pick_puzzle_type()

        progress = create_progress()
        with progress:
            pbar = add_pbar(
                progress,
                description="...... 🧩 Chain-of-Thought Hijacking",
                total=self.max_retries * 2,
            )

            for _ in range(self.max_retries):
                puzzle: GeneratedPuzzle = generate(
                    ChainOfThoughtHijackingTemplate.generate_puzzle(
                        puzzle_type, self.target_tokens, seed_topic=None
                    ),
                    GeneratedPuzzle,
                    self.simulator_model,
                )
                update_pbar(progress, pbar)

                verdict: IsBenignPuzzle = generate(
                    ChainOfThoughtHijackingTemplate.is_benign_puzzle(
                        puzzle.puzzle
                    ),
                    IsBenignPuzzle,
                    self.simulator_model,
                )
                update_pbar(progress, pbar)

                if verdict.is_benign_puzzle:
                    update_pbar(progress, pbar, advance_to_end=True)
                    return self._assemble_prompt(puzzle.puzzle, attack)

            update_pbar(progress, pbar, advance_to_end=True)
        return attack

    async def a_enhance(
        self,
        attack: str,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = None,
    ) -> str:
        self.simulator_model, _ = initialize_model(simulator_model)
        puzzle_type = self._pick_puzzle_type()

        progress = create_progress()
        with progress:
            pbar = add_pbar(
                progress,
                description="...... 🧩 Chain-of-Thought Hijacking",
                total=self.max_retries * 2,
            )

            try:
                for _ in range(self.max_retries):
                    puzzle: GeneratedPuzzle = await a_generate(
                        ChainOfThoughtHijackingTemplate.generate_puzzle(
                            puzzle_type, self.target_tokens, seed_topic=None
                        ),
                        GeneratedPuzzle,
                        self.simulator_model,
                    )
                    update_pbar(progress, pbar)

                    verdict: IsBenignPuzzle = await a_generate(
                        ChainOfThoughtHijackingTemplate.is_benign_puzzle(
                            puzzle.puzzle
                        ),
                        IsBenignPuzzle,
                        self.simulator_model,
                    )
                    update_pbar(progress, pbar)

                    if verdict.is_benign_puzzle:
                        return self._assemble_prompt(puzzle.puzzle, attack)
            finally:
                update_pbar(progress, pbar, advance_to_end=True)

        return attack

    def _pick_puzzle_type(self) -> str:
        return self.puzzle_type or random.choice(PUZZLE_TYPES)

    def _assemble_prompt(self, puzzle: str, attack: str) -> str:
        return ChainOfThoughtHijackingTemplate.hijack(
            puzzle=puzzle,
            attack=attack,
            reasoning_length=self.reasoning_length,
            role_preface=self.role_preface,
        )

    def get_name(self) -> str:
        return self.name
