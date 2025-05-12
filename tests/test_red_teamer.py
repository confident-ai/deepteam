import pytest
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from deepteam.red_teamer.red_teamer import RedTeamer
from deepteam.vulnerabilities import BaseVulnerability
from deepteam.attacks import BaseAttack
from deepteam.attacks.attack_simulator import SimulatedAttack
from deepteam.red_teamer.risk_assessment import RiskAssessment, RedTeamingTestCase
from deepteam.vulnerabilities.types import VulnerabilityType
from deepeval.test_case import LLMTestCase


class MockVulnerability(BaseVulnerability):
    def __init__(self, types):
        super().__init__(types=types)

    def get_name(self) -> str:
        return "MockVulnerability"


class MockAttack(BaseAttack):
    def enhance(self, attack: str, *args, **kwargs) -> str:
        return f"Enhanced: {attack}"

    def get_name(self) -> str:
        return "MockAttack"


@pytest.fixture(autouse=True)
def mock_openai_api_key():
    """Mock OpenAI API key for testing."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
        yield

@pytest.fixture(autouse=True)
def mock_gpt_model_validation():
    """Mock GPT model validation to allow mock-model."""
    with patch("deepeval.models.llms.openai_model.GPTModel._validate_model_name") as mock_validate:
        mock_validate.return_value = None  # Do nothing, validation passes
        yield

@pytest.fixture
def mock_vulnerability_type():
    from deepteam.vulnerabilities.types import BiasType
    return BiasType.GENDER


@pytest.fixture
def mock_vulnerability(mock_vulnerability_type):
    return MockVulnerability(types=[mock_vulnerability_type])


@pytest.fixture
def mock_attack():
    return MockAttack()


@pytest.fixture
def mock_sync_callback():
    def callback(input_text):
        return f"Response to: {input_text}"
    return callback


@pytest.fixture
def mock_async_callback():
    async def async_callback(input_text):
        return f"Async response to: {input_text}"
    return async_callback


@pytest.fixture
def mock_simulated_attack(mock_vulnerability_type):
    return SimulatedAttack(
        vulnerability="MockVulnerability",
        vulnerability_type=mock_vulnerability_type,
        input="Test attack input",
        attack_method="MockAttack"
    )


@pytest.fixture
def mock_simulated_attacks(mock_vulnerability_type):
    return [
        SimulatedAttack(
            vulnerability="MockVulnerability",
            vulnerability_type=mock_vulnerability_type,
            input="Test attack input 1",
            attack_method="MockAttack"
        ),
        SimulatedAttack(
            vulnerability="MockVulnerability",
            vulnerability_type=mock_vulnerability_type,
            input="Test attack input 2",
            attack_method="MockAttack"
        )
    ]


class TestRedTeamer:
    @patch("deepteam.red_teamer.red_teamer.AttackSimulator")
    @patch("deepteam.red_teamer.red_teamer.initialize_model")
    def test_init(self, mock_initialize_model, mock_attack_simulator_class):
        mock_initialize_model.return_value = ("mock_model", None)
        
        red_teamer = RedTeamer(
            simulator_model="test-model",
            evaluation_model="test-eval-model",
            target_purpose="test purpose",
            async_mode=True,
            max_concurrent=5
        )
        
        assert red_teamer.target_purpose == "test purpose"
        assert red_teamer.async_mode is True
        assert red_teamer.max_concurrent == 5
        
        mock_initialize_model.assert_any_call("test-model")
        mock_initialize_model.assert_any_call("test-eval-model")
        
        mock_attack_simulator_class.assert_called_once()
        
    @patch("deepteam.red_teamer.red_teamer.group_attacks_by_vulnerability_type")
    @patch("deepteam.red_teamer.red_teamer.RedTeamer.get_red_teaming_metrics_map")
    def test_red_team_sync(self, mock_get_metrics_map, mock_group_attacks, 
                          mock_sync_callback, mock_vulnerability, mock_attack, 
                          mock_simulated_attacks, mock_vulnerability_type):
        mock_metrics_map = {mock_vulnerability_type: MagicMock}
        mock_get_metrics_map.return_value = mock_metrics_map
        
        mock_metric_instance = MagicMock()
        mock_metric_instance.score = 1
        mock_metric_instance.reason = "Test reason"
        mock_metrics_map[mock_vulnerability_type].return_value = mock_metric_instance
        
        mock_group_attacks.return_value = {mock_vulnerability_type: mock_simulated_attacks}
        
        red_teamer = MagicMock(spec=RedTeamer)
        red_teamer.async_mode = False
        red_teamer.max_concurrent = 10
        red_teamer.get_red_teaming_metrics_map.return_value = mock_metrics_map
        red_teamer.attack_simulator = MagicMock()
        red_teamer.attack_simulator.simulate.return_value = mock_simulated_attacks
        
        red_teamer.__gt__ = lambda self, other: False
        red_teamer.__lt__ = lambda self, other: False
        red_teamer.__ge__ = lambda self, other: False
        red_teamer.__le__ = lambda self, other: False
        red_teamer.__eq__ = lambda self, other: False
        
        RedTeamer.red_team(red_teamer, 
                          model_callback=mock_sync_callback,
                          vulnerabilities=[mock_vulnerability],
                          attacks=[mock_attack])
        
        red_teamer.attack_simulator.simulate.assert_called_once_with(
            attacks_per_vulnerability_type=1,
            vulnerabilities=[mock_vulnerability],
            attacks=[mock_attack],
            ignore_errors=False
        )
        
        red_teamer.get_red_teaming_metrics_map.assert_called_once()
        
        assert red_teamer._print_risk_assessment.called

    @pytest.mark.asyncio
    @patch("deepteam.red_teamer.red_teamer.asyncio.gather")
    @patch("deepteam.red_teamer.red_teamer.RedTeamer.get_red_teaming_metrics_map")
    async def test_a_red_team(self, mock_get_metrics_map, mock_gather, 
                             mock_async_callback, mock_vulnerability, mock_attack, 
                             mock_simulated_attacks, mock_vulnerability_type):
        mock_metrics_map = {mock_vulnerability_type: MagicMock}
        mock_get_metrics_map.return_value = mock_metrics_map
        
        mock_gather.return_value = [
            [RedTeamingTestCase(
                vulnerability="MockVulnerability",
                vulnerability_type=mock_vulnerability_type,
                riskCategory="Test Risk",
                attackMethod="MockAttack",
                input="Test input",
                score=1,
                reason="Test reason"
            )]
        ]
        
        red_teamer = MagicMock(spec=RedTeamer)
        red_teamer.async_mode = True
        red_teamer.max_concurrent = 10
        red_teamer.get_red_teaming_metrics_map.return_value = mock_metrics_map
        red_teamer.attack_simulator = MagicMock()
        red_teamer.attack_simulator.a_simulate = AsyncMock(return_value=mock_simulated_attacks)
        
        red_teamer.__gt__ = lambda self, other: False
        red_teamer.__lt__ = lambda self, other: False
        red_teamer.__ge__ = lambda self, other: False
        red_teamer.__le__ = lambda self, other: False
        red_teamer.__eq__ = lambda self, other: False
        
        await RedTeamer.a_red_team(red_teamer, 
                                  model_callback=mock_async_callback,
                                  vulnerabilities=[mock_vulnerability],
                                  attacks=[mock_attack])
        
        red_teamer.attack_simulator.a_simulate.assert_called_once_with(
            attacks_per_vulnerability_type=1,
            vulnerabilities=[mock_vulnerability],
            attacks=[mock_attack],
            ignore_errors=False
        )
        
        red_teamer.get_red_teaming_metrics_map.assert_called_once()
        
        assert red_teamer._print_risk_assessment.called

    @pytest.mark.asyncio
    @patch("deepteam.red_teamer.red_teamer.LLMTestCase")
    @patch("deepeval.metrics.utils.initialize_model")
    async def test_a_attack(self, mock_initialize_model, mock_llm_test_case, 
                           mock_async_callback, mock_vulnerability_type, 
                           mock_simulated_attack):
        mock_model = MagicMock()
        mock_initialize_model.return_value = (mock_model, None)
        
        mock_metrics_map = {mock_vulnerability_type: MagicMock}
        mock_metric_instance = MagicMock()
        mock_metric_instance.a_measure = AsyncMock()
        mock_metric_instance.score = 1
        mock_metric_instance.reason = "Test reason"
        mock_metrics_map[mock_vulnerability_type].return_value = mock_metric_instance
        
        with patch("deepteam.red_teamer.red_teamer.AttackSimulator"):
            with patch.object(RedTeamer, "__init__", return_value=None):
                red_teamer = RedTeamer()
                red_teamer.evaluation_model = mock_model
                red_teamer.async_mode = True
                red_teamer.max_concurrent = 5
        
        result = await red_teamer._a_attack(
            model_callback=mock_async_callback,
            simulated_attack=mock_simulated_attack,
            vulnerability="MockVulnerability",
            vulnerability_type=mock_vulnerability_type,
            metrics_map=mock_metrics_map,
            ignore_errors=False
        )
        
        assert result.vulnerability == "MockVulnerability"
        assert result.vulnerability_type == mock_vulnerability_type
        assert result.score == 1
        assert result.reason == "Test reason"
        
        mock_async_callback.assert_called_once_with(mock_simulated_attack.input)
        mock_metric_instance.a_measure.assert_called_once()

    @patch("deepeval.metrics.utils.initialize_model")
    def test_get_red_teaming_metrics_map(self, mock_initialize_model):
        mock_model = MagicMock()
        mock_initialize_model.return_value = (mock_model, None)
        
        with patch("deepteam.red_teamer.red_teamer.AttackSimulator"):
            with patch.object(RedTeamer, "__init__", return_value=None):
                red_teamer = RedTeamer()
                red_teamer.evaluation_model = mock_model
                red_teamer.target_purpose = "test purpose"
        
        with patch("deepteam.red_teamer.red_teamer.BiasMetric", return_value=MagicMock()):
            with patch("deepteam.red_teamer.red_teamer.ToxicityMetric", return_value=MagicMock()):
                metrics_map = red_teamer.get_red_teaming_metrics_map()
        
        assert len(metrics_map) > 0
        
        from deepteam.vulnerabilities.types import BiasType
        assert BiasType.GENDER in metrics_map
