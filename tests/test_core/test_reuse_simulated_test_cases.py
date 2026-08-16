from typing import List

from deepteam.red_teamer.red_teamer import RedTeamer
from deepteam.test_case import RTTestCase, RTTurn
from deepteam.vulnerabilities.toxicity.types import ToxicityType


class FakeMetric:
    def __init__(self):
        self.score = 0.5
        self.reason = "reason"
        self.evaluation_cost = 0.0
        self.measured_turns = None

    def measure(self, test_case):
        self.measured_turns = test_case.turns


class FakeVulnerability:
    types = [ToxicityType.PROFANITY]

    def __init__(self):
        self.evaluation_model = None
        self.metric = FakeMetric()

    def _get_metric(self, vulnerability_type):
        return self.metric


def make_test_case() -> RTTestCase:
    tc = RTTestCase(vulnerability="Toxicity")
    tc.vulnerability_type = ToxicityType.PROFANITY
    tc.input = "attack"
    tc.turns = [
        RTTurn(role="user", content="attack1"),
        RTTurn(role="assistant", content="I cannot help with that."),
        RTTurn(role="user", content="attack2"),
        RTTurn(role="assistant", content="I cannot help with that."),
    ]
    tc.error = None
    tc.score = None
    return tc


def test_multi_turn_reuse_replays_user_turns_through_callback():
    calls = []

    def model_callback(prompt: str, turns: List[RTTurn]) -> RTTurn:
        calls.append((prompt, len(turns)))
        return RTTurn(role="assistant", content=f"fresh-{prompt}")

    tc = make_test_case()
    fake_self = type("FakeSelf", (), {"evaluation_model": None})()
    vuln = FakeVulnerability()

    result = RedTeamer._attack(
        fake_self,
        model_callback,
        tc,
        "Toxicity",
        ToxicityType.PROFANITY,
        [vuln],
        False,
        reuse_simulated_test_cases=True,
    )

    # Both user turns were replayed against the model, each seeing the
    # conversation replayed so far (0 and 2 turns respectively).
    assert calls == [("attack1", 0), ("attack2", 2)]
    # Placeholder assistant responses were replaced by fresh ones.
    assert [turn.content for turn in result.turns] == [
        "attack1",
        "fresh-attack1",
        "attack2",
        "fresh-attack2",
    ]
    # The metric measured the replayed conversation, not the saved one.
    assert vuln.metric.measured_turns is result.turns


def test_multi_turn_reuse_disabled_keeps_saved_turns():
    tc = make_test_case()
    fake_self = type("FakeSelf", (), {"evaluation_model": None})()
    vuln = FakeVulnerability()

    RedTeamer._attack(
        fake_self,
        lambda prompt: RTTurn(role="assistant", content="unused"),
        tc,
        "Toxicity",
        ToxicityType.PROFANITY,
        [vuln],
        False,
        reuse_simulated_test_cases=False,
    )

    # Without the reuse flag the saved turns are evaluated unchanged.
    assert [turn.content for turn in tc.turns] == [
        "attack1",
        "I cannot help with that.",
        "attack2",
        "I cannot help with that.",
    ]
