import pytest
import inspect
import asyncio
from deepteam.red_teamer.utils import resolve_model_callback
from deepeval.models import GPTModel
from deepteam.vulnerabilities import Bias
from deepteam.attacks.single_turn import Roleplay
from deepteam import red_team


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

    def test_simple_red_teaming_async(self):
        OPENAI_MODEL = "openai/gpt-4.1"
        risk_assessment = red_team(
            model_callback=OPENAI_MODEL,
            attacks=[Roleplay()],
            vulnerabilities=[Bias(types=["gender"])],
            async_mode=True,
        )
        assert risk_assessment is not None

    def test_simple_red_teaming_sync(self):
        OPENAI_MODEL = "openai/gpt-4.1"
        risk_assessment = red_team(
            model_callback=OPENAI_MODEL,
            attacks=[Roleplay()],
            vulnerabilities=[Bias(types=["gender"])],
            async_mode=False,
        )
        assert risk_assessment is not None
