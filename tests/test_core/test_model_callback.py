import pytest
import inspect
import asyncio
from deepteam.test_case import RTTurn
from deepteam.red_teamer.utils import (
    resolve_model_callback,
    wrap_model_callback,
)
from deepeval.models import GPTModel
from deepteam.vulnerabilities import Bias
from deepteam import red_team


def custom_model_callback(input: str, turns=None):
    return f"Cannot respond to {input}"


async def custom_async_model_callback(input: str, turns=None):
    return f"Cannot respond to {input}"


def custom_invalid_model_callback(input: str, turns=None):
    return {"res": f"Cannot respond to {input}"}


async def custom_invalid_async_model_callback(input: str, turns=None):
    return {"res": f"Cannot respond to {input}"}


class TestModelCallbackVariations:

    def test_openai_valid_sync_model(self):
        OPENAI_MODEL = "openai/gpt-4.1"
        model_callback = resolve_model_callback(OPENAI_MODEL, False)
        assert not inspect.iscoroutinefunction(model_callback)
        assert model_callback("Hello") is not None

    def test_openai_valid_async_model(self):
        OPENAI_MODEL = "openai/gpt-4.1"
        model_callback = resolve_model_callback(OPENAI_MODEL, True)
        assert inspect.iscoroutinefunction(model_callback)
        assert asyncio.run(model_callback("Hello")) is not None

    def test_deepeval_base_llm_sync_model(self):
        model = GPTModel(model="gpt-3.5-turbo")
        model_callback = resolve_model_callback(model, False)
        assert not inspect.iscoroutinefunction(model_callback)
        assert model_callback("Hello") is not None

    def test_deepeval_base_llm_async_model(self):
        model = GPTModel(model="gpt-3.5-turbo")
        model_callback = resolve_model_callback(model, True)
        assert inspect.iscoroutinefunction(model_callback)
        assert asyncio.run(model_callback("Hello")) is not None

    def test_sync_wrapper_returns_valid_model_callback(self):
        model_callback = wrap_model_callback(custom_model_callback, False)
        response = model_callback("Hello this is a test")

        assert inspect.iscoroutinefunction(model_callback) is False
        assert isinstance(response, RTTurn)
        assert response.role == "assistant"

    def test_async_wrapper_returns_valid_model_callback(self):
        model_callback = wrap_model_callback(custom_async_model_callback, True)
        response = asyncio.run(model_callback("Hello this is a test"))

        assert inspect.iscoroutinefunction(model_callback) is True
        assert isinstance(response, RTTurn)
        assert response.role == "assistant"

    def test_sync_wrapper_raises_error_for_invalid_response(self):
        model_callback = wrap_model_callback(
            custom_invalid_model_callback, False
        )
        with pytest.raises(TypeError):
            model_callback("Hello this is a test")

    def test_async_wrapper_raises_error_for_invalid_response(self):
        model_callback = wrap_model_callback(
            custom_invalid_async_model_callback, True
        )
        with pytest.raises(TypeError):
            asyncio.run(model_callback("Hello this is a test"))

    def test_simple_red_teaming_async(self):
        OPENAI_MODEL = "openai/gpt-4.1"
        risk_assessment = red_team(
            model_callback=OPENAI_MODEL,
            vulnerabilities=[Bias(types=["gender"])],
            async_mode=True,
            ignore_errors=False,
        )
        assert risk_assessment is not None

    def test_simple_red_teaming_sync(self):
        OPENAI_MODEL = "openai/gpt-4.1"
        risk_assessment = red_team(
            model_callback=OPENAI_MODEL,
            vulnerabilities=[Bias(types=["gender"])],
            async_mode=False,
            ignore_errors=False,
        )
        assert risk_assessment is not None


def _build_turns_callback(style, async_mode, captured):
    """Build a callback that records the turns it receives under `style`."""

    def record(received_turns):
        captured["turns"] = received_turns
        return "response"

    if style == "var-keyword":

        if async_mode:

            async def cb(input, **kwargs):
                return record(kwargs.get("turns"))

        else:

            def cb(input, **kwargs):
                return record(kwargs.get("turns"))

    elif style == "keyword-only":

        if async_mode:

            async def cb(input, *, turns=None):
                return record(turns)

        else:

            def cb(input, *, turns=None):
                return record(turns)

    else:  # positional-or-keyword

        if async_mode:

            async def cb(input, turns=None):
                return record(turns)

        else:

            def cb(input, turns=None):
                return record(turns)

    return cb


TURNS_CALLBACK_STYLES = [
    pytest.param("var-keyword", id="var-keyword"),
    pytest.param("keyword-only", id="keyword-only"),
    pytest.param("positional-or-keyword", id="positional-or-keyword"),
]


@pytest.mark.parametrize("style", TURNS_CALLBACK_STYLES)
def test_sync_wrapper_forwards_turns_by_supported_signature(style):
    captured = {}
    callback = _build_turns_callback(style, False, captured)
    wrapped = wrap_model_callback(callback, False)
    turns = [RTTurn(role="user", content="hello")]

    response = wrapped("prompt", turns)

    assert isinstance(response, RTTurn)
    assert response.role == "assistant"
    assert captured["turns"] is turns


@pytest.mark.parametrize("style", TURNS_CALLBACK_STYLES)
def test_async_wrapper_forwards_turns_by_supported_signature(style):
    captured = {}
    callback = _build_turns_callback(style, True, captured)
    wrapped = wrap_model_callback(callback, True)
    turns = [RTTurn(role="user", content="hello")]

    response = asyncio.run(wrapped("prompt", turns))

    assert isinstance(response, RTTurn)
    assert response.role == "assistant"
    assert captured["turns"] is turns
