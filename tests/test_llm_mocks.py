import pytest
from unittest.mock import MagicMock, patch
import asyncio
import os

from deepeval.models import DeepEvalBaseLLM
from deepteam.metrics.base_red_teaming_metric import BaseRedTeamingMetric
from deepteam.attacks.attack_simulator import AttackSimulator


class MockLLM(DeepEvalBaseLLM):
    """Mock LLM for testing purposes."""
    
    def __init__(self, responses=None):
        """
        Initialize the mock LLM.
        
        Args:
            responses: A dictionary mapping prompts to responses.
                      If None, a default response will be used.
        """
        self.responses = responses or {}
        self.call_history = []
    
    def generate(self, prompt, **kwargs):
        """Mock synchronous generation."""
        self.call_history.append({"prompt": prompt, "kwargs": kwargs})
        return self.responses.get(prompt, f"Mock response to: {prompt}")
    
    async def a_generate(self, prompt, **kwargs):
        """Mock asynchronous generation."""
        self.call_history.append({"prompt": prompt, "kwargs": kwargs})
        return self.responses.get(prompt, f"Mock async response to: {prompt}")
        
    def get_model_name(self):
        """Return the model name."""
        return "mock-model"
        
    def load_model(self):
        """Load the model."""
        return self


@pytest.fixture
def mock_llm():
    """Fixture for a mock LLM with default responses."""
    return MockLLM()


@pytest.fixture
def mock_llm_with_responses():
    """Fixture for a mock LLM with custom responses."""
    responses = {
        "Test prompt 1": "Test response 1",
        "Test prompt 2": "Test response 2",
    }
    return MockLLM(responses=responses)


class TestLLMMocks:
    def test_mock_llm_sync(self, mock_llm):
        """Test synchronous generation with mock LLM."""
        response = mock_llm.generate("Test prompt")
        assert response == "Mock response to: Test prompt"
        assert len(mock_llm.call_history) == 1
        assert mock_llm.call_history[0]["prompt"] == "Test prompt"
    
    @pytest.mark.asyncio
    async def test_mock_llm_async(self, mock_llm):
        """Test asynchronous generation with mock LLM."""
        response = await mock_llm.a_generate("Test prompt")
        assert response == "Mock async response to: Test prompt"
        assert len(mock_llm.call_history) == 1
        assert mock_llm.call_history[0]["prompt"] == "Test prompt"
    
    def test_mock_llm_with_responses(self, mock_llm_with_responses):
        """Test generation with custom responses."""
        response1 = mock_llm_with_responses.generate("Test prompt 1")
        response2 = mock_llm_with_responses.generate("Test prompt 2")
        response3 = mock_llm_with_responses.generate("Unknown prompt")
        
        assert response1 == "Test response 1"
        assert response2 == "Test response 2"
        assert response3 == "Mock response to: Unknown prompt"
        assert len(mock_llm_with_responses.call_history) == 3


@pytest.fixture(autouse=True)
def mock_openai_api_key():
    """Mock OpenAI API key for testing."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
        yield

@pytest.fixture(autouse=True)
def mock_gpt_model_validation():
    """Mock GPT model validation to allow mock-model."""
    with patch("deepeval.models.llms.openai_model.get_actual_model_name", return_value="gpt-4"):
        with patch("deepeval.models.llms.openai_model.valid_gpt_models", ["gpt-4", "mock-model"]):
            yield


class TestMetricWithMockLLM:
    @patch("deepeval.metrics.utils.initialize_model")
    def test_metric_with_mock_llm(self, mock_initialize):
        """Test a metric with a mock LLM."""
        mock_llm = MockLLM()
        mock_initialize.return_value = (mock_llm, None)
        
        class TestMetric(BaseRedTeamingMetric):
            def __init__(self, model=None, async_mode=False):
                super().__init__()
                self.model = mock_llm
                self.async_mode = async_mode
                
            def measure(self, test_case):
                response = self.model.generate(test_case.input)
                self.score = 1 if "good" in response.lower() else 0
                self.reason = f"Score is {self.score}"
                
            async def a_measure(self, test_case):
                response = await self.model.a_generate(test_case.input)
                self.score = 1 if "good" in response.lower() else 0
                self.reason = f"Score is {self.score}"
        
        metric = TestMetric(model="mock-model")
        
        test_case = MagicMock()
        test_case.input = "Is this a good response?"
        
        mock_llm.responses[test_case.input] = "Yes, this is a good response."
        
        metric.measure(test_case)
        
        assert metric.score == 1
        assert metric.reason == "Score is 1"
        assert len(mock_llm.call_history) == 1
        assert mock_llm.call_history[0]["prompt"] == "Is this a good response?"


class TestAttackSimulatorWithMockLLM:
    @patch("deepeval.metrics.utils.initialize_model")
    def test_attack_simulator_with_mock_llm(self, mock_initialize):
        """Test the AttackSimulator with a mock LLM."""
        mock_llm = MockLLM()
        mock_initialize.return_value = (mock_llm, None)
        
        simulator = AttackSimulator(
            simulator_model="mock-model",
            purpose="test purpose",
            max_concurrent=5
        )
        
        mock_llm.responses["Test prompt"] = '{"attack": "Test attack"}'
        
        simulator.simulator_model = mock_llm
        
        assert simulator.simulator_model == mock_llm
