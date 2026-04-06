import pytest

from deepteam.frameworks import HarmActionsEval
from deepteam.test_case import RTTurn, ToolCall


def _tool_aware_turn(prompt: str, tools_called=None) -> RTTurn:
    if tools_called is None:
        tools_called = []
    return RTTurn(role="assistant", content=f"response:{prompt}", tools_called=tools_called)


def _row(
    action_id: int,
    prompt: str,
    classification: str,
    tool_name: str,
    arguments: dict | None = None,
    include_manipulated_prompt: bool = True,
):
    if arguments is None:
        arguments = {"payload": prompt}

    row = {
        "action_id": action_id,
        "prompt": prompt,
        "classification": classification,
        "action": {
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": arguments,
            },
        },
    }
    if include_manipulated_prompt:
        row["manipulated_prompt"] = prompt
    return row


def test_harmactionseval_load_dataset_filters_and_applies_offset(monkeypatch):
    monkeypatch.setattr(
        HarmActionsEval,
        "_load_raw_dataset",
        lambda _: [
            _row(10, "safe row", "safe", "safe_tool"),
            _row(11, "prompt one", "harmful", "dangerous_tool"),
            _row(12, "prompt two", "unethical", "unsafe_tool"),
            _row(13, "prompt three", "harmful", "dangerous_tool"),
        ],
    )
    monkeypatch.setattr("random.sample", lambda actions, k: actions[:k])

    framework = HarmActionsEval(num_attacks=2, offset=1)
    framework.load_dataset()

    assert len(framework.test_cases) == 2
    loaded_ids = [(tc.metadata or {})["action_id"] for tc in framework.test_cases]
    assert loaded_ids == [12, 13]


def test_harmactionseval_assess_scores_by_expected_tool_call(monkeypatch):
    monkeypatch.setattr(
        HarmActionsEval,
        "_load_raw_dataset",
        lambda _: [
            _row(1, "prompt one", "harmful", "dangerous_tool", {"target": "a"}),
            _row(2, "prompt two", "unethical", "unsafe_tool", {"target": "b"}),
        ],
    )
    monkeypatch.setattr("random.sample", lambda actions, k: actions[:k])

    framework = HarmActionsEval(num_attacks=2, k=2)
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
async def test_harmactionseval_async_assess_scores_by_tool_call(monkeypatch):
    monkeypatch.setattr(
        HarmActionsEval,
        "_load_raw_dataset",
        lambda _: [
            _row(4, "async trigger tool", "harmful", "dangerous_tool", {"kind": "x"})
        ],
    )
    monkeypatch.setattr("random.sample", lambda actions, k: actions[:k])
    framework = HarmActionsEval(num_attacks=1, k=2)
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
def test_harmactionseval_load_dataset_param_validation(
    monkeypatch, kwargs, error_message
):
    monkeypatch.setattr(
        HarmActionsEval,
        "_load_raw_dataset",
        lambda _: [
            _row(1, "prompt one", "safe", "safe_tool"),
            _row(2, "prompt two", "harmful", "dangerous_tool"),
            _row(3, "prompt three", "harmful", "dangerous_tool"),
            _row(4, "prompt four", "harmful", "dangerous_tool"),
        ],
    )
    framework = HarmActionsEval(**kwargs)
    with pytest.raises(ValueError, match=error_message):
        framework.load_dataset()


def test_harmactionseval_load_dataset_bounds_validation(monkeypatch):
    monkeypatch.setattr(
        HarmActionsEval,
        "_load_raw_dataset",
        lambda _: [_row(1, "prompt one", "harmful", "dangerous_tool")],
    )

    with pytest.raises(ValueError, match="offset is beyond"):
        HarmActionsEval(num_attacks=1, offset=1).load_dataset()

    framework = HarmActionsEval(num_attacks=2)
    test_cases = framework.load_dataset()
    assert len(test_cases) == 1
    assert framework.num_attacks == 1


def test_harmactionseval_load_dataset_rejects_legacy_row_schema(monkeypatch):
    monkeypatch.setattr(
        HarmActionsEval,
        "_load_raw_dataset",
        lambda _: [
            {
                "action_id": 99,
                "prompt": "legacy prompt",
                "manipulated_prompt": "legacy prompt",
                "classification": "harmful",
                "label": "legacy_tool",
                "action": {
                    "type": "function",
                    "server_label": "legacy_tool",
                    "parameters": {"x": 1},
                },
            }
        ],
    )

    framework = HarmActionsEval(num_attacks=1)
    with pytest.raises(ValueError, match="missing action.function object"):
        framework.load_dataset()


def test_harmactionseval_assess_ignore_errors_sets_error(monkeypatch):
    monkeypatch.setattr(
        HarmActionsEval,
        "_load_raw_dataset",
        lambda _: [_row(5, "prompt five", "harmful", "dangerous_tool")],
    )
    monkeypatch.setattr("random.sample", lambda actions, k: actions[:k])
    framework = HarmActionsEval(num_attacks=1)
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
    assert test_cases[0].error == "Error evaluating HarmActionsEval tool-call behavior"
    assert test_cases[0].score == 0.0


def test_harmactionseval_assess_raises_when_ignore_errors_false(monkeypatch):
    monkeypatch.setattr(
        HarmActionsEval,
        "_load_raw_dataset",
        lambda _: [_row(6, "prompt six", "harmful", "dangerous_tool")],
    )
    monkeypatch.setattr("random.sample", lambda actions, k: actions[:k])
    framework = HarmActionsEval(num_attacks=1)
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
async def test_harmactionseval_invoke_model_callback_requires_tool_kwargs_compatibility():
    framework = HarmActionsEval(num_attacks=1)

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


def test_harmactionseval_build_tool_definition_infers_parameter_types():
    framework = HarmActionsEval(num_attacks=1)
    definition = framework._build_tool_definition(
        _row(
            action_id=1,
            prompt="typed prompt",
            classification="harmful",
            tool_name="typed_tool",
            arguments={
                "flag": True,
                "count": 3,
                "ratio": 0.5,
                "items": ["a", "b"],
                "name": "x",
            },
        )
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


def test_harmactionseval_to_test_case_falls_back_to_prompt_when_manipulated_missing(monkeypatch):
    framework = HarmActionsEval(num_attacks=1)

    action = _row(
        action_id=77,
        prompt="fallback prompt",
        classification="harmful",
        tool_name="fallback_tool",
        include_manipulated_prompt=False,
    )

    test_case = framework._to_test_case(action)
    assert test_case.input == "fallback prompt"


def test_harmactionseval_did_call_target_tool_matches_case_insensitive():
    framework = HarmActionsEval(num_attacks=1)
    calls = [ToolCall(name="DaNgErOuS_ToOl", input_parameters={})]

    assert framework._did_call_target_tool(calls, "dangerous_tool") is True
    assert framework._did_call_target_tool(calls, "unsafe_tool") is False
    assert framework._did_call_target_tool(calls, None) is True
