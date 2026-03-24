from typing import List, Dict, Union, Optional, Any
import inspect
import json
from functools import wraps
from deepteam.test_case import RTTestCase, RTTurn, ToolCall
from deepteam.vulnerabilities.types import VulnerabilityType
from deepeval.models import (
    DeepEvalBaseLLM,
    GPTModel,
    GeminiModel,
    GrokModel,
    KimiModel,
    AnthropicModel,
    DeepSeekModel,
    OllamaModel,
)
from deepeval.metrics.utils import initialize_model

MODEL_PROVIDER_MAPPING = {
    "openai": GPTModel,
    "anthropic": AnthropicModel,
    "google": GeminiModel,
    "xai": GrokModel,
    "moonshotai": KimiModel,
    "deepseek": DeepSeekModel,
    "ollama": OllamaModel,
}


def group_attacks_by_vulnerability_type(
    simulated_attacks: List[RTTestCase],
) -> Dict[VulnerabilityType, List[RTTestCase]]:
    vulnerability_type_to_attacks_map: Dict[
        VulnerabilityType, List[RTTestCase]
    ] = {}

    for simulated_attack in simulated_attacks:
        if (
            simulated_attack.vulnerability_type
            not in vulnerability_type_to_attacks_map
        ):
            vulnerability_type_to_attacks_map[
                simulated_attack.vulnerability_type
            ] = [simulated_attack]
        else:
            vulnerability_type_to_attacks_map[
                simulated_attack.vulnerability_type
            ].append(simulated_attack)

    return vulnerability_type_to_attacks_map


def wrap_model_callback(model_callback, async_mode):

    sig = inspect.signature(model_callback)
    params = list(sig.parameters.values())

    accepts_turns = len(params) > 1 or any(
        p.kind
        in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for p in params
    )

    def _validate_response(response):
        """Helper to strictly validate the callback's return value."""
        if isinstance(response, RTTurn):
            if response.role == "assistant":
                return response
            else:
                raise ValueError(
                    "The target model must return an 'RTTurn' with role='assistant' only."
                )
        elif isinstance(response, str):
            return RTTurn(role="assistant", content=response)
        else:
            raise TypeError(
                f"The target model callback has returned an invalid response of type {type(response)}. "
                "Please return either a string or an 'RTTurn' with role='assistant'."
            )

    @wraps(model_callback)
    def get_sync_model_callback(
        input: str, turns: Optional[List[RTTurn]] = None, **kwargs
    ):
        if accepts_turns:
            response = model_callback(input, turns, **kwargs)
        else:
            response = model_callback(input, **kwargs)

        return _validate_response(response)

    @wraps(model_callback)
    async def get_async_model_callback(
        input: str, turns: Optional[List[RTTurn]] = None, **kwargs
    ):
        if accepts_turns:
            response = await model_callback(input, turns, **kwargs)
        else:
            response = await model_callback(input, **kwargs)

        return _validate_response(response)

    return get_async_model_callback if async_mode else get_sync_model_callback


def resolve_model_callback(
    model_callback: Union[str, DeepEvalBaseLLM], async_mode: bool
):
    if isinstance(model_callback, str):
        if "/" in model_callback:
            provider, model = model_callback.split("/", 1)
            if provider in MODEL_PROVIDER_MAPPING.keys():
                model_class = MODEL_PROVIDER_MAPPING.get(provider)
                model_callback = model_class(model)
            else:
                raise ValueError(
                    f"Invalid provider for model_callback {model_callback}, please pass a string with provider prefix separated by a '/'. Ex: 'openai/gpt-4.1'. Avaliable provider prefixes: {[key for key in MODEL_PROVIDER_MAPPING.keys()]}"
                )
        else:
            raise ValueError(
                f"Invalid string for model_callback {model_callback}, please pass a string with provider prefix separated by a '/'. Ex: 'openai/gpt-4.1'. Avaliable provider prefixes: {[key for key in MODEL_PROVIDER_MAPPING.keys()]}"
            )

    model_callback, _ = initialize_model(model_callback)

    def _parse_tool_calls(message: Any):
        parsed_tool_calls = []
        for tool_call in getattr(message, "tool_calls", []) or []:
            function = getattr(tool_call, "function", None)
            name = getattr(function, "name", None)
            arguments = getattr(function, "arguments", None)

            input_parameters = {}
            if isinstance(arguments, str) and arguments.strip():
                try:
                    loaded = json.loads(arguments)
                    if isinstance(loaded, dict):
                        input_parameters = loaded
                except json.JSONDecodeError:
                    input_parameters = {"raw": arguments}

            if name:
                parsed_tool_calls.append(
                    ToolCall(name=name, input_parameters=input_parameters)
                )

        return parsed_tool_calls

    if not async_mode:

        def new_model_callback(
            input: str,
            turns: Optional[List[RTTurn]] = None,
            **kwargs: Any,
        ):
            if "tools" in kwargs and hasattr(model_callback, "generate_raw_response"):
                original_generation_kwargs = dict(
                    getattr(model_callback, "generation_kwargs", {}) or {}
                )
                try:
                    model_callback.generation_kwargs = {
                        **original_generation_kwargs,
                        **kwargs,
                    }
                    completion, _ = model_callback.generate_raw_response(
                        input
                    )
                    message = completion.choices[0].message
                    return RTTurn(
                        role="assistant",
                        content=message.content or "",
                        tools_called=_parse_tool_calls(message),
                    )
                finally:
                    model_callback.generation_kwargs = (
                        original_generation_kwargs
                    )

            res = model_callback.generate(input)
            if isinstance(res, tuple):
                res, _ = res
            return RTTurn(role="assistant", content=res)

    else:

        async def new_model_callback(
            input: str,
            turns: Optional[List[RTTurn]] = None,
            **kwargs: Any,
        ):
            if "tools" in kwargs and hasattr(model_callback, "a_generate_raw_response"):
                original_generation_kwargs = dict(
                    getattr(model_callback, "generation_kwargs", {}) or {}
                )
                try:
                    model_callback.generation_kwargs = {
                        **original_generation_kwargs,
                        **kwargs,
                    }
                    completion, _ = await model_callback.a_generate_raw_response(
                        input
                    )
                    message = completion.choices[0].message
                    return RTTurn(
                        role="assistant",
                        content=message.content or "",
                        tools_called=_parse_tool_calls(message),
                    )
                finally:
                    model_callback.generation_kwargs = (
                        original_generation_kwargs
                    )

            res = await model_callback.a_generate(input)
            if isinstance(res, tuple):
                res, _ = res
            return RTTurn(role="assistant", content=res)

    return new_model_callback
