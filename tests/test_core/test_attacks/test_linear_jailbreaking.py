import pytest
from unittest.mock import AsyncMock, MagicMock
from deepteam.attacks.multi_turn import LinearJailbreaking
from deepteam.test_case.test_case import RTTurn


class TestLinearJailbreaking:

    def test_linear_jailbreaking_initialization(self):
        attack = LinearJailbreaking()
        assert attack.weight == 1
        assert attack.get_name() == "Linear Jailbreaking"

    def test_linear_jailbreaking_initialization_with_weight(self):
        attack = LinearJailbreaking(weight=4)
        assert attack.weight == 4
        assert attack.get_name() == "Linear Jailbreaking"

    def test_linear_jailbreaking_enhance_interface(self):
        attack = LinearJailbreaking(num_turns=0)

        # Mock the required components
        mock_callback = MagicMock(return_value="Mock response")

        # Just a user turn
        user_only_turns = [RTTurn(role="user", content="User content")]
        # Just an assistant turn
        assistant_only_turns = [
            RTTurn(role="assistant", content="Assistant content")
        ]
        # Both user and assistant turn
        user_and_assistant_turns = [
            RTTurn(role="user", content="User content"),
            RTTurn(role="assistant", content="Assistant content"),
        ]

        # Test for various cases:
        user_only_turns_result = attack.enhance(
            mock_callback, user_only_turns, "gpt-4o"
        )
        user_and_assistant_turns_result = attack.enhance(
            mock_callback, user_and_assistant_turns, "gpt-4o"
        )
        assert user_only_turns_result[1].role == "assistant"
        with pytest.raises(ValueError):
            attack.enhance(mock_callback, assistant_only_turns, "gpt-4o")
        assert len(user_and_assistant_turns_result) == 2

    @pytest.mark.asyncio
    async def test_linear_jailbreaking_async_enhance_interface(self):
        from deepteam.vulnerabilities import Bias

        attack = LinearJailbreaking(num_turns=0)

        # Mock the required components
        mock_callback = AsyncMock(return_value="Mock response")

        user_only_turns = [RTTurn(role="user", content="User content")]
        # Just an assistant turn
        assistant_only_turns = [
            RTTurn(role="assistant", content="Assistant content")
        ]
        # Both user and assistant turn
        user_and_assistant_turns = [
            RTTurn(role="user", content="User content"),
            RTTurn(role="assistant", content="Assistant content"),
        ]

        # Test for various cases:
        user_only_turns_result = await attack.a_enhance(
            mock_callback, user_only_turns, "gpt-4o"
        )
        user_and_assistant_turns_result = await attack.a_enhance(
            mock_callback, user_and_assistant_turns, "gpt-4o"
        )
        assert user_only_turns_result[1].role == "assistant"
        with pytest.raises(ValueError):
            await attack.a_enhance(
                mock_callback, assistant_only_turns, "gpt-4o"
            )
        assert len(user_and_assistant_turns_result) == 2

    def test_linear_jailbreaking_has_required_methods(self):
        attack = LinearJailbreaking()

        # Verify all required methods exist
        assert hasattr(attack, "enhance")
        assert hasattr(attack, "a_enhance")
        assert hasattr(attack, "get_name")
        assert callable(attack.enhance)
        assert callable(attack.a_enhance)
        assert callable(attack.get_name)
