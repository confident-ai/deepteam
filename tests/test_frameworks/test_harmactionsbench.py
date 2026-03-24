import pytest

from deepteam.frameworks import HarmActionsBench
from deepteam.test_case import RTTurn, ToolCall


def _tool_aware_turn(prompt: str, tools_called=None) -> RTTurn:
    if tools_called is None:
        tools_called = []
    return RTTurn(role="assistant", content=f"response:{prompt}", tools_called=tools_called)


def test_harmactionsbench_load_dataset_filters_and_applies_offset(monkeypatch):
    monkeypatch.setattr(
        HarmActionsBench,
        "_load_raw_dataset",
        lambda _: [
            {
                "action_id": 10,
                "manipulated_prompt": "safe row",
                "classification": "safe",
                "label": "safe_tool",
                "resource": "filesystem",
            },
            {
                "action_id": 11,
                "manipulated_prompt": "prompt one",
                "classification": "harmful",
                "label": "dangerous_tool",
                "resource": "filesystem",
            },
            {
                "action_id": 12,
                "manipulated_prompt": "prompt two",
                "classification": "unethical",
                "label": "unsafe_tool",
                "resource": "network",
            },
            {
                "action_id": 13,
                "manipulated_prompt": "prompt three",
                "classification": "harmful",
                "label": "dangerous_tool",
                "resource": "network",
            },
        ],
    )
    monkeypatch.setattr("random.sample", lambda actions, k: actions[:k])

    framework = HarmActionsBench(num_attacks=2, offset=1)
    framework.load_dataset()

    assert len(framework.test_cases) == 2
    loaded_ids = [(tc.metadata or {})["action_id"] for tc in framework.test_cases]
    assert loaded_ids == [12, 13]


def test_harmactionsbench_assess_scores_by_expected_tool_call(monkeypatch):
    monkeypatch.setattr(
        HarmActionsBench,
        "_load_raw_dataset",
        lambda _: [
            {
                "action_id": 1,
                "manipulated_prompt": "prompt one",
                "classification": "harmful",
                "label": "dangerous_tool",
                "resource": "filesystem",
            },
            {
                "action_id": 2,
                "manipulated_prompt": "prompt two",
                "classification": "unethical",
                "label": "unsafe_tool",
                "resource": "network",
            },
        ],
    )
    monkeypatch.setattr("random.sample", lambda actions, k: actions[:k])

    framework = HarmActionsBench(num_attacks=2, k=2)
    framework.load_dataset()

    call_count: dict[str, int] = {}
    received_tool_names: list[str] = []

    def _model_callback(prompt: str, turns=None, **kwargs) -> RTTurn:
        call_count[prompt] = call_count.get(prompt, 0) + 1

        tools = kwargs.get("tools")
        if tools:
            tool_name = tools[0]["function"]["name"]
            received_tool_names.append(tool_name)

        if "Calculate the sum of 9284453 and 7823492." in prompt:
            return _tool_aware_turn(
                prompt,
                [ToolCall(name="calculator", input_parameters={})],
            )

        if "prompt one" in prompt and call_count[prompt] == 1:
            return _tool_aware_turn(
                prompt,
                [ToolCall(name="dangerous_tool", input_parameters={})],
            )

        return _tool_aware_turn(prompt)

    test_cases = framework.assess(
        model_callback=_model_callback,
        progress=None,
        task_id=None,
        ignore_errors=False,
    )

    cases_by_id = {(tc.metadata or {})["action_id"]: tc for tc in test_cases}

    assert cases_by_id[1].score == 0.0
    assert cases_by_id[2].score == 1.0
    assert cases_by_id[1].error is None
    assert cases_by_id[2].error is None
    assert "detected at attempt 1" in (cases_by_id[1].reason or "")
    assert "within 2 attempts" in (cases_by_id[2].reason or "")
    assert "calculator" in received_tool_names
    assert "dangerous_tool" in received_tool_names
    assert "unsafe_tool" in received_tool_names


@pytest.mark.asyncio
async def test_harmactionsbench_async_assess_scores_by_tool_call(monkeypatch):
    monkeypatch.setattr(
        HarmActionsBench,
        "_load_raw_dataset",
        lambda _: [
            {
                "action_id": 4,
                "manipulated_prompt": "async trigger tool",
                "classification": "harmful",
                "label": "dangerous_tool",
                "resource": "tool_api",
            }
        ],
    )
    monkeypatch.setattr("random.sample", lambda actions, k: actions[:k])
    framework = HarmActionsBench(num_attacks=1, k=2)
    framework.load_dataset()

    async def _async_model_callback(prompt: str, turns=None, **kwargs) -> RTTurn:
        if "Calculate the sum of 9284453 and 7823492." in prompt:
            return _tool_aware_turn(
                prompt,
                [ToolCall(name="calculator", input_parameters={})],
            )

        return _tool_aware_turn(
            prompt,
            [ToolCall(name="dangerous_tool", input_parameters={})],
        )

    test_cases = await framework.a_assess(
        model_callback=_async_model_callback,
        progress=None,
        task_id=None,
        ignore_errors=False,
    )

    assert len(test_cases) == 1
    assert test_cases[0].score == 0.0


@pytest.mark.parametrize(
    "kwargs,error_message",
    [
        ({"k": 0}, "k must be >= 1"),
        ({"num_attacks": 0}, "num_attacks must be >= 1"),
        ({"offset": -1}, "offset must be >= 0"),
    ],
)
def test_harmactionsbench_load_dataset_param_validation(
    monkeypatch, kwargs, error_message
):
    monkeypatch.setattr(
        HarmActionsBench,
        "_load_raw_dataset",
        lambda _: [
            {
                "action_id": 1,
                "manipulated_prompt": "prompt one",
                "classification": "safe",
                "label": "safe_tool",
                "resource": "tool_api",
            },
            {
                "action_id": 2,
                "manipulated_prompt": "prompt two",
                "classification": "harmful",
                "label": "dangerous_tool",
                "resource": "tool_api",
            },
            {
                "action_id": 3,
                "manipulated_prompt": "prompt three",
                "classification": "harmful",
                "label": "dangerous_tool",
                "resource": "tool_api",
            },
            {
                "action_id": 4,
                "manipulated_prompt": "prompt four",
                "classification": "harmful",
                "label": "dangerous_tool",
                "resource": "tool_api",
            },
        ],
    )
    framework = HarmActionsBench(**kwargs)
    with pytest.raises(ValueError, match=error_message):
        framework.load_dataset()


def test_harmactionsbench_load_dataset_bounds_validation(monkeypatch):
    monkeypatch.setattr(
        HarmActionsBench,
        "_load_raw_dataset",
        lambda _: [
            {
                "action_id": 1,
                "manipulated_prompt": "prompt one",
                "classification": "harmful",
                "label": "dangerous_tool",
                "resource": "tool_api",
            }
        ],
    )

    with pytest.raises(ValueError, match="offset is beyond"):
        HarmActionsBench(num_attacks=1, offset=1).load_dataset()

    framework = HarmActionsBench(num_attacks=2)
    test_cases = framework.load_dataset()
    assert len(test_cases) == 1
    assert framework.num_attacks == 1


def test_harmactionsbench_assess_ignore_errors_sets_error(monkeypatch):
    monkeypatch.setattr(
        HarmActionsBench,
        "_load_raw_dataset",
        lambda _: [
            {
                "action_id": 5,
                "manipulated_prompt": "prompt five",
                "classification": "harmful",
                "label": "dangerous_tool",
                "resource": "tool_api",
            }
        ],
    )
    monkeypatch.setattr("random.sample", lambda actions, k: actions[:k])
    framework = HarmActionsBench(num_attacks=1)
    framework.load_dataset()

    def broken_callback(prompt: str, turns=None, **kwargs) -> RTTurn:
        if "Calculate the sum of 9284453 and 7823492." in prompt:
            return _tool_aware_turn(
                prompt,
                [ToolCall(name="calculator", input_parameters={})],
            )
        raise RuntimeError("simulated callback failure")

    test_cases = framework.assess(
        model_callback=broken_callback,
        progress=None,
        task_id=None,
        ignore_errors=True,
    )

    assert len(test_cases) == 1
    assert test_cases[0].error == "Error evaluating HarmActionsBench tool-call behavior"
    assert test_cases[0].score == 0.0


def test_harmactionsbench_assess_raises_when_ignore_errors_false(monkeypatch):
    monkeypatch.setattr(
        HarmActionsBench,
        "_load_raw_dataset",
        lambda _: [
            {
                "action_id": 6,
                "manipulated_prompt": "prompt six",
                "classification": "harmful",
                "label": "dangerous_tool",
                "resource": "tool_api",
            }
        ],
    )
    monkeypatch.setattr("random.sample", lambda actions, k: actions[:k])
    framework = HarmActionsBench(num_attacks=1)
    framework.load_dataset()

    def broken_callback(prompt: str, turns=None, **kwargs) -> RTTurn:
        if "Calculate the sum of 9284453 and 7823492." in prompt:
            return _tool_aware_turn(
                prompt,
                [ToolCall(name="calculator", input_parameters={})],
            )
        raise RuntimeError("simulated callback failure")

    with pytest.raises(RuntimeError, match="simulated callback failure"):
        framework.assess(
            model_callback=broken_callback,
            progress=None,
            task_id=None,
            ignore_errors=False,
        )


@pytest.mark.asyncio
async def test_harmactionsbench_invoke_model_callback_requires_tool_kwargs_compatibility():
    framework = HarmActionsBench(num_attacks=1)

    def prompt_only_callback(prompt):
        return _tool_aware_turn(prompt)

    turns = [RTTurn(role="user", content="hello")]
    tools = [{"type": "function", "function": {"name": "calculator"}}]

    with pytest.raises(TypeError, match="unexpected keyword argument 'tools'"):
        await framework._invoke_model_callback(
            model_callback=prompt_only_callback,
            evaluation_prompt="test prompt",
            turns=turns,
            tools=tools,
        )


def test_harmactionsbench_build_tool_definition_infers_parameter_types():
    framework = HarmActionsBench(num_attacks=1)
    definition = framework._build_tool_definition(
        {
            "label": "typed_tool",
            "action": {
                "parameters": {
                    "flag": True,
                    "count": 3,
                    "ratio": 0.5,
                    "items": ["a", "b"],
                    "name": "x",
                }
            },
        }
    )

    assert definition is not None
    params = definition["function"]["parameters"]
    assert params["properties"]["flag"]["type"] == "boolean"
    assert params["properties"]["count"]["type"] == "integer"
    assert params["properties"]["ratio"]["type"] == "number"
    assert params["properties"]["items"]["type"] == "array"
    assert params["properties"]["items"]["items"]["type"] == "string"
    assert params["properties"]["name"]["type"] == "string"
    assert set(params["required"]) == {"flag", "count", "ratio", "items", "name"}


def test_harmactionsbench_did_call_target_tool_matches_case_insensitive():
    framework = HarmActionsBench(num_attacks=1)
    calls = [ToolCall(name="DaNgErOuS_ToOl", input_parameters={})]

    assert framework._did_call_target_tool(calls, "dangerous_tool") is True
    assert framework._did_call_target_tool(calls, "unsafe_tool") is False
    assert framework._did_call_target_tool(calls, None) is True
