from typing import TYPE_CHECKING, List, Optional

from deepteam.attacks.attack_simulator.utils import cost_accumulator
from deepteam.attacks.multi_turn.progression import (
    Probe,
    Progression,
    StopReason,
)
from deepteam.orchestration.judge import CriteriaJudge
from deepteam.orchestration.schema import CriteriaVerdict
from deepteam.orchestration.templates import SubAgentTemplate
from deepteam.orchestration.types import Escalation, SubAgentReport
from deepteam.orchestration.utils import (
    as_chat_history,
    as_current_turn,
    weighted_score,
)
from deepteam.test_case.test_case import RTTurn

if TYPE_CHECKING:
    from deepteam.orchestration.orchestration_agent import (
        OrchestrationAgent,
    )


class SubAgent:

    def __init__(
        self,
        parent: "OrchestrationAgent",
        task: str,
        scenario: str,
        criteria: str,
        target_analysis: str,
        instructions: str,
        seed_turns: Optional[List[RTTurn]] = None,
    ):
        self.parent = parent
        self.task = task
        self.scenario = scenario
        self.criteria = criteria
        self.target_analysis = target_analysis
        self.instructions = instructions
        self.seed_turns = list(seed_turns or [])
        self.judge = CriteriaJudge(parent.judge_model, task, scenario)

        self.progression: Optional[Progression] = None
        self.verdicts: List[CriteriaVerdict] = []
        self.escalations: List[Escalation] = []
        self.stopped = False

    @property
    def weighted_score(self) -> float:
        return weighted_score(self.verdicts)

    def stop(self) -> None:
        self.stopped = True

    def report_to_parent(self, report: SubAgentReport) -> None:
        self.parent.receive_report(self, report)

    async def a_probe(self) -> None:
        # Its own cost scope keeps per-probe costs separate while sub agents
        # run in parallel. The progression snapshots the running cost when
        # it's created, so it has to be created inside the scope.
        with cost_accumulator():
            self.progression = self.parent.create_progression(
                self.criteria, self.seed_turns
            )
            await self._a_probe(self.progression)

        self.report_to_parent(
            SubAgentReport(
                criteria=self.criteria,
                scenario=self.scenario,
                target_analysis=self.target_analysis,
                instructions=self.instructions,
                progression=self.progression.finalize(),
                verdicts=self.verdicts,
                escalations=self.escalations,
            )
        )

    async def _a_probe(self, progression: Progression) -> None:
        try:
            for turn in range(1, self.parent.max_turns + 1):
                # A stop only takes effect between turns, so a turn is never
                # left half-written.
                if self.stopped:
                    return progression.stop(StopReason.STOPPED_BY_PARENT)

                message = await self.parent.a_next_message(
                    self._next_turn_prompt(progression)
                )
                probe = await progression.a_probe(message)
                verdict = await self._a_judge(progression, probe)
                await progression.a_commit(probe)

                self.verdicts.append(verdict)
                progression.score_probe(
                    probe, verdict.unit_score, verdict.reason
                )

                if verdict.criteria_met:
                    return progression.stop(
                        StopReason.SHIFT_DETECTED, verdict.reason
                    )
                elif verdict.escalate_to_parent and self._may_escalate(turn):
                    self.escalations.append(
                        Escalation(
                            turn=turn,
                            guidance=await self.parent.a_handle_escalation(
                                self
                            ),
                        )
                    )
        except Exception as exc:
            progression.stop(StopReason.SIMULATION_ERROR, str(exc))
            progression.error = exc
            return

        if self.stopped:
            progression.stop(StopReason.STOPPED_BY_PARENT)

    def _may_escalate(self, turn: int) -> bool:
        # Escalate only with a turn left to use the guidance, and give each
        # piece of guidance a few turns to work before asking again.
        if self.stopped or turn >= self.parent.max_turns:
            return False
        if not self.escalations:
            return True
        return (
            turn - self.escalations[-1].turn
            >= self.parent.escalation_cooldown
        )

    def _next_turn_prompt(self, progression: Progression) -> str:
        latest = self.verdicts[-1] if self.verdicts else None
        return SubAgentTemplate.next_turn(
            self.task,
            self.scenario,
            self.criteria,
            self.target_analysis,
            self.instructions,
            progression.turns,
            current_score=latest.score if latest else None,
            current_reason=latest.reason if latest else None,
            improvement_guidance=(
                latest.improvement_guidance if latest else None
            ),
            previous_verdicts=[
                verdict.model_dump(
                    include={"score", "reason", "improvement_guidance"}
                )
                for verdict in self.verdicts[:-1]
            ],
            orchestrator_guidance=(
                self.escalations[-1].guidance if self.escalations else None
            ),
        )

    async def _a_judge(
        self, progression: Progression, probe: Probe
    ) -> CriteriaVerdict:
        # Judged before the probe is committed, so the committed turns are
        # exactly the history before this exchange.
        return await self.judge.a_judge(
            self.criteria,
            current_turn=as_current_turn(probe),
            chat_history=as_chat_history(progression.turns),
            previous_scores=[
                verdict.model_dump(include={"score", "reason"})
                for verdict in self.verdicts
            ],
        )
