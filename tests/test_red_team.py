import pytest
import os
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

from deepteam import red_team
from deepteam.red_teamer import RedTeamer
from deepteam.vulnerabilities import BaseVulnerability
from deepteam.attacks import BaseAttack
from deepteam.red_teamer.risk_assessment import RiskAssessment
from deepteam.vulnerabilities.types import BiasType


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

@pytest.fixture
def mock_vulnerability():
    return MockVulnerability(types=[BiasType.GENDER])


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
def mock_risk_assessment():
    return MagicMock(spec=RiskAssessment)


class TestRedTeam:
    @patch("deepteam.red_team.RedTeamer", autospec=True)
    def test_red_team_function(self, mock_red_teamer_class, mock_sync_callback, mock_vulnerability, mock_attack):
        mock_red_teamer = mock_red_teamer_class.return_value
        mock_red_teamer.red_team.return_value = "mock_risk_assessment"
        
        result = red_team(
            model_callback=mock_sync_callback,
            vulnerabilities=[mock_vulnerability],
            attacks=[mock_attack],
            run_async=False
        )
        
        mock_red_teamer_class.assert_called_once_with(
            async_mode=False,
            max_concurrent=10
        )
        
        mock_red_teamer.red_team.assert_called_once_with(
            model_callback=mock_sync_callback,
            vulnerabilities=[mock_vulnerability],
            attacks=[mock_attack],
            attacks_per_vulnerability_type=1,
            ignore_errors=False
        )
        
        assert result == "mock_risk_assessment"

    @patch("deepteam.red_team.RedTeamer", autospec=True)
    def test_red_team_function_async(self, mock_red_teamer_class, mock_async_callback, mock_vulnerability, mock_attack):
        mock_red_teamer = mock_red_teamer_class.return_value
        mock_red_teamer.red_team.return_value = "mock_risk_assessment"
        
        result = red_team(
            model_callback=mock_async_callback,
            vulnerabilities=[mock_vulnerability],
            attacks=[mock_attack],
            run_async=True,
            max_concurrent=5
        )
        
        mock_red_teamer_class.assert_called_once_with(
            async_mode=True,
            max_concurrent=5
        )
        
        mock_red_teamer.red_team.assert_called_once_with(
            model_callback=mock_async_callback,
            vulnerabilities=[mock_vulnerability],
            attacks=[mock_attack],
            attacks_per_vulnerability_type=1,
            ignore_errors=False
        )
        
        assert result == "mock_risk_assessment"
