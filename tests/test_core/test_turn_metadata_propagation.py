import asyncio
from unittest.mock import patch

from deepteam.red_teamer.red_teamer import RedTeamer
from deepteam.test_case import RTTestCase, RTTurn
from deepteam.vulnerabilities import Bias


class _StubMetric:
    score = 1.0
    reason = "stub"
    evaluation_cost = 0.0

    def measure(self, test_case, *args, **kwargs):
        pass

    async def a_measure(self, test_case, *args, **kwargs):
        pass


def _make_simulated_test_case(vulnerability_type):
    return RTTestCase(
        vulnerability="Bias",
        vulnerability_type=vulnerability_type,
        input="some adversarial prompt",
    )


class TestTurnMetadataPropagation:
    """`model_callback` can return metadata on its `RTTurn` (e.g. request
    tracing info for a custom metric). It must reach the `RTTestCase` the
    same way `retrieval_context`/`tools_called` already do, instead of
    being silently dropped before the metric runs."""

    def test_sync_attack_copies_metadata_onto_test_case(self):
        red_teamer = RedTeamer()
        vulnerability = Bias(types=["gender"])
        vulnerability_type = vulnerability.types[0]

        def model_callback(input):
            return RTTurn(
                role="assistant",
                content="some response",
                metadata={"trace_id": "abc123"},
            )

        with patch.object(
            Bias, "_get_metric", return_value=_StubMetric()
        ):
            result = red_teamer._attack(
                model_callback=model_callback,
                simulated_test_case=_make_simulated_test_case(
                    vulnerability_type
                ),
                vulnerability="Bias",
                vulnerability_type=vulnerability_type,
                vulnerabilities=[vulnerability],
                ignore_errors=False,
            )

        assert result.metadata == {"trace_id": "abc123"}

    def test_async_attack_copies_metadata_onto_test_case(self):
        red_teamer = RedTeamer()
        vulnerability = Bias(types=["gender"])
        vulnerability_type = vulnerability.types[0]

        async def model_callback(input):
            return RTTurn(
                role="assistant",
                content="some response",
                metadata={"trace_id": "xyz789"},
            )

        with patch.object(
            Bias, "_get_metric", return_value=_StubMetric()
        ):
            result = asyncio.run(
                red_teamer._a_attack(
                    model_callback=model_callback,
                    simulated_test_case=_make_simulated_test_case(
                        vulnerability_type
                    ),
                    vulnerability="Bias",
                    vulnerability_type=vulnerability_type,
                    vulnerabilities=[vulnerability],
                    ignore_errors=False,
                )
            )

        assert result.metadata == {"trace_id": "xyz789"}
