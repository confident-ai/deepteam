from deepteam.guardrails.guards.base_guard import BaseGuard
from deepteam.guardrails.guards.schema import SafetyLevelSchema


class _SafeModel:
    def generate(self, *, prompt, schema):
        return SafetyLevelSchema(safety_level="safe", reason="safe")


class _TestGuard(BaseGuard):
    def guard_input(self, input, guard_prompt, *args, **kwargs):
        return self._guard(guard_prompt)

    def guard_output(self, input, output, guard_prompt, *args, **kwargs):
        return self._guard(guard_prompt)

    async def a_guard_input(self, input, guard_prompt, *args, **kwargs):
        return await self.a_guard(guard_prompt)

    async def a_guard_output(self, input, output, guard_prompt, *args, **kwargs):
        return await self.a_guard(guard_prompt)


def test_safe_guard_score_is_one():
    guard = object.__new__(_TestGuard)
    guard.model = _SafeModel()
    guard.using_native_model = False

    guard._guard("test prompt")

    assert guard.score == 1.0
