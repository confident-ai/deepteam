import pytest
from unittest.mock import MagicMock, patch
import asyncio

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


@pytest.fixture
def patch_initialize_model():
    """Fixture to patch the initialize_model function."""
    with patch("deepteam.metrics.utils.initialize_model") as mock_initialize:
        mock_llm = MockLLM()
        mock_initialize.return_value = (mock_llm, None)
        yield mock_initialize, mock_llm


class TestMetricWithMockLLM:
    @patch("deepteam.metrics.base_red_teaming_metric.initialize_model")
    def test_metric_with_mock_llm(self, mock_initialize):
        """Test a metric with a mock LLM."""
        mock_llm = MockLLM()
        mock_initialize.return_value = (mock_llm, None)
        
        class TestMetric(BaseRedTeamingMetric):
            def measure(self, test_case):
                response = self.model.generate(test_case.input)
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
    @patch("deepteam.attacks.attack_simulator.attack_simulator.initialize_model")
    def test_attack_simulator_with_mock_llm(self, mock_initialize):
        """Test the AttackSimulator with a mock LLM."""
        mock_llm = MockLLM()
        mock_initialize.return_value = (mock_llm, None)
        
        simulator = AttackSimulator(simulator_model="mock-model")
        
        mock_llm.responses["Test prompt"] = '{"attack": "Test attack"}'
        
        simulator.simulator_model = mock_llm
        
        assert simulator.simulator_model == mock_llm
