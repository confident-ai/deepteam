"""
Regression tests for deepteam.attacks.multi_turn.utils.append_target_turn.

These tests cover the type-safety fix that ensures the turn history
always contains RTTurn objects, regardless of what model_callback returns.
"""
import pytest

from deepteam.test_case import RTTurn
from deepteam.attacks.multi_turn.utils import append_target_turn


class TestAppendTargetTurn:
    """Regression tests for the turn-history type-safety fix."""

    def test_string_response_is_wrapped_in_rtturn(self):
        """model_callback returns a str -- should be wrapped, not appended raw."""
        turns = [RTTurn(role="user", content="hello")]
        append_target_turn(turns, "hi there")

        assert len(turns) == 2
        assert isinstance(turns[-1], RTTurn)
        assert turns[-1].role == "assistant"
        assert turns[-1].content == "hi there"

    def test_rtturn_response_is_appended_unchanged(self):
        """Backwards compatibility: passing an RTTurn must still work."""
        turns = [RTTurn(role="user", content="hello")]
        response = RTTurn(role="assistant", content="hi there")
        append_target_turn(turns, response)

        assert len(turns) == 2
        assert turns[-1] is response  # same object, not a copy
        assert turns[-1].role == "assistant"
        assert turns[-1].content == "hi there"

    def test_non_string_non_rtturn_response_is_coerced(self):
        """Fallback path: arbitrary objects get str()-coerced into content."""
        turns = [RTTurn(role="user", content="hello")]
        append_target_turn(turns, 42)

        assert len(turns) == 2
        assert isinstance(turns[-1], RTTurn)
        assert turns[-1].role == "assistant"
        assert turns[-1].content == "42"

    def test_turn_level_attack_attribute_set_on_string_response(self):
        """
        Regression for the second latent AttributeError:
        when target_response is a str AND turn_level_attack is provided,
        the attribute assignment must run on the wrapped RTTurn, not the str.
        """
        turns = [RTTurn(role="user", content="hello")]
        append_target_turn(
            turns, "hi there", turn_level_attack="PromptInjection"
        )

        assert isinstance(turns[-1], RTTurn)
        assert turns[-1].turn_level_attack == "PromptInjection"
        assert turns[-1].content == "hi there"

    def test_turn_level_attack_attribute_set_on_rtturn_response(self):
        """Existing path: RTTurn input with attack name set."""
        turns = [RTTurn(role="user", content="hello")]
        response = RTTurn(role="assistant", content="hi there")
        append_target_turn(
            turns, response, turn_level_attack="LinearJailbreaking"
        )

        assert turns[-1].turn_level_attack == "LinearJailbreaking"

    def test_no_turn_level_attack_does_not_set_attribute(self):
        """When turn_level_attack is None, the attribute should not be set
        (or should remain falsy / None)."""
        turns = [RTTurn(role="user", content="hello")]
        append_target_turn(turns, "hi there")

        # Attribute should not be set when not provided
        assert (
            not hasattr(turns[-1], "turn_level_attack")
            or turns[-1].turn_level_attack is None
        )

    def test_history_is_iterable_with_role_attribute_after_append(self):
        """
        Integration-style check: simulate the downstream pattern that
        originally crashed. Iterating turns and reading .role must
        succeed for every element, including those appended from raw strings.
        """
        turns = [RTTurn(role="user", content="user message")]
        append_target_turn(turns, "assistant string response")
        append_target_turn(
            turns, RTTurn(role="user", content="follow-up")
        )
        append_target_turn(turns, "another assistant string")

        # This is the loop that used to fail in JailBreakingTemplate
        roles = [turn.role for turn in turns]
        assert roles == ["user", "assistant", "user", "assistant"]
