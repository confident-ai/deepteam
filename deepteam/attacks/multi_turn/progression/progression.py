import random
from typing import List, Optional, Union

from deepeval.metrics.utils import initialize_model
from deepeval.models import DeepEvalBaseLLM

from deepteam.attacks.multi_turn.types import CallbackType
from deepteam.attacks.multi_turn.utils import (
    a_enhance_attack,
    enhance_attack,
)
from deepteam.attacks.multi_turn.progression.shift_detection import (
    AsyncMetricCheck,
    BehaviorShiftDetector,
    MetricCheck,
    MetricVerdict,
)
from deepteam.attacks.multi_turn.progression.stopping import (
    StopReason,
    mark_stop,
)
from deepteam.attacks.multi_turn.progression.types import (
    Attempt,
    ProgressionResult,
)
from deepteam.attacks.single_turn import BaseSingleTurnAttack
from deepteam.test_case.test_case import RTTurn

TURN_LEVEL_ATTACK_RATE = 0.5


class Progression:

    def __init__(
        self,
        model_callback: CallbackType,
        turns: Optional[List[RTTurn]] = None,
        vulnerability: Optional[str] = None,
        vulnerability_type: Optional[str] = None,
        simulator_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        *,
        metric_check: Optional[Union[MetricCheck, AsyncMetricCheck]] = None,
        turn_level_attacks: Optional[List[BaseSingleTurnAttack]] = None,
        turn_level_attack_rate: float = TURN_LEVEL_ATTACK_RATE,
    ):
        self.model_callback = model_callback
        self.turns: List[RTTurn] = list(turns or [])
        self.simulator_model, _ = initialize_model(simulator_model)
        self.turn_level_attacks = turn_level_attacks
        self.turn_level_attack_rate = turn_level_attack_rate

        self.goal = next(
            (
                turn.content
                for turn in reversed(self.turns)
                if turn.role == "user"
            ),
            None,
        )
        if self.goal is None:
            raise ValueError(
                "No user turn found in the conversation to progress from."
            )

        self.vulnerability_data = (
            f"Vulnerability: {vulnerability} | Type: {vulnerability_type}"
        )
        self.detector = BehaviorShiftDetector(
            simulator_model=self.simulator_model,
            vulnerability_data=self.vulnerability_data,
            metric_check=metric_check,
        )

        self.turns_spent = 0
        self.attempts: List[Attempt] = []
        # Attempt-tree bookkeeping. `_parent` is the attempt new probes
        # descend from -- it follows commits and rolls back on a backtrack.
        self._next_attempt_id = 0
        self._parent: Optional[Attempt] = None
        self._committed: List[Attempt] = []
        self.stop_reason = StopReason.BUDGET_EXHAUSTED
        self.stop_detail: Optional[str] = None
        self.shift_verdict: Optional[MetricVerdict] = None
        self.error: Optional[Exception] = None

    @property
    def last_response(self) -> str:
        return self.turns[-1].content if self.turns else ""

    def ensure_target_responded(self) -> None:
        if len(self.turns) <= 1 or self.turns[-1].role == "user":
            self.turns.append(self.model_callback(self.goal, self.turns))

    async def a_ensure_target_responded(self) -> None:
        if len(self.turns) <= 1 or self.turns[-1].role == "user":
            self.turns.append(await self.model_callback(self.goal, self.turns))

    def finalize(self) -> ProgressionResult:
        mark_stop(
            self.turns, self.stop_reason, self.stop_detail, self.turns_spent
        )
        return ProgressionResult(
            turns=self.turns,
            stop_reason=self.stop_reason,
            stop_detail=self.stop_detail,
            turns_spent=self.turns_spent,
            attempts=self.attempts,
            shift_verdict=self.shift_verdict,
            error=self.error,
        )

    def probe(self, attack: str, *, enhance: bool = True) -> Attempt:
        """Put an attack to the target without keeping it in the conversation.

        For attacks that try several things per step and keep one: tree search
        branches, retry-until-accepted loops, and knowledge-gathering questions
        asked before the conversation proper begins. The Attempt is recorded
        either way, so what was discarded survives in the result.
        """
        attack, turn_level_attack = self._apply_turn_level_attack(
            attack, enhance
        )
        return self._record_attempt(
            attack,
            self.model_callback(attack, self.turns),
            turn_level_attack,
        )

    async def a_probe(self, attack: str, *, enhance: bool = True) -> Attempt:
        attack, turn_level_attack = await self._a_apply_turn_level_attack(
            attack, enhance
        )
        return self._record_attempt(
            attack,
            await self.model_callback(attack, self.turns),
            turn_level_attack,
        )

    def commit(self, attack: Union[str, Attempt]) -> RTTurn:
        """Put an attack to the target and keep both turns.

        Passing an Attempt from `probe` reuses the response already collected
        rather than calling the target a second time.
        """
        attempt = attack if isinstance(attack, Attempt) else self.probe(attack)
        return self._commit_attempt(attempt)

    async def a_commit(self, attack: Union[str, Attempt]) -> RTTurn:
        attempt = (
            attack
            if isinstance(attack, Attempt)
            else await self.a_probe(attack)
        )
        return self._commit_attempt(attempt)

    def shift_detected(self) -> bool:
        """Has the target's behavior shifted? Records the stop reason when it
        has, so the caller only has to return."""
        return self._resolve_shift(self.detector.check(self.turns))

    async def a_shift_detected(self) -> bool:
        return self._resolve_shift(await self.detector.a_check(self.turns))

    def remove_last_turns(self, count: int = 1) -> None:
        """Drop the last `count` user/assistant pairs, for attacks that walk a
        refused turn back and try a different approach."""
        if count <= 0:
            return
        del self.turns[-2 * count :]
        self.turns_spent = max(0, self.turns_spent - count)
        # Walked-back turns are simply un-committed; the attempts stay in
        # `attempts` with committed=False so the backtrack is reconstructable.
        for _ in range(min(count, len(self._committed))):
            self._committed.pop().committed = False
        self._parent = self._committed[-1] if self._committed else None

    def stop(self, reason: StopReason, detail: Optional[str] = None) -> None:
        self.stop_reason = reason
        self.stop_detail = detail

    def _apply_turn_level_attack(self, attack: str, enhance: bool):
        if not enhance or not self._should_apply_turn_level_attack():
            return attack, None
        picked = random.choice(self.turn_level_attacks)
        return (
            enhance_attack(picked, attack, self.simulator_model),
            picked.get_name(),
        )

    async def _a_apply_turn_level_attack(self, attack: str, enhance: bool):
        if not enhance or not self._should_apply_turn_level_attack():
            return attack, None
        picked = random.choice(self.turn_level_attacks)
        return (
            await a_enhance_attack(picked, attack, self.simulator_model),
            picked.get_name(),
        )

    def _should_apply_turn_level_attack(self) -> bool:
        return bool(self.turn_level_attacks) and (
            random.random() < self.turn_level_attack_rate
        )

    def _record_attempt(
        self,
        attack: str,
        response: RTTurn,
        turn_level_attack: Optional[str],
    ) -> Attempt:
        attempt = Attempt(
            id=self._next_attempt_id,
            parent_id=self._parent.id if self._parent else None,
            depth=self._parent.depth + 1 if self._parent else 0,
            attack=attack,
            response=response,
            turn_level_attack=turn_level_attack,
        )
        self._next_attempt_id += 1
        self.attempts.append(attempt)
        return attempt

    def _commit_attempt(self, attempt: Attempt) -> RTTurn:
        attempt.committed = True
        attempt.response.turn_level_attack = attempt.turn_level_attack
        self.turns.append(RTTurn(role="user", content=attempt.attack))
        self.turns.append(attempt.response)
        self.turns_spent += 1
        self._committed.append(attempt)
        self._parent = attempt
        return attempt.response

    def _resolve_shift(self, verdict) -> bool:
        if verdict is None:
            return False
        self.shift_verdict = MetricVerdict(
            score=verdict.metric_score,
            reason=verdict.metric_reason,
            evaluation_cost=verdict.evaluation_cost,
        )
        self.stop(StopReason.SHIFT_DETECTED, verdict.detail)
        return True
