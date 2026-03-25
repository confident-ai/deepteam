import pytest

from deepteam.utils import sanitize_prompt_for_encoding, set_ascii_strict_mode


def test_sanitize_smart_quotes_and_ellipsis():
    input_str = "“Hello,” she said — it’s a test…"
    expected = '"Hello," she said - it\'s a test...'
    output = sanitize_prompt_for_encoding(input_str)
    assert output == expected


@pytest.mark.parametrize(
    "inp,exp",
    [
        ("", ""),
        (" ", " "),
    ],
)
def test_sanitize_empty_or_spaces(inp, exp):
    assert sanitize_prompt_for_encoding(inp) == exp


def test_sanitize_non_latin_preserved():
    input_str = "Café naïve résumé"
    # Non-ASCII content should be preserved in non-strict mode
    output = sanitize_prompt_for_encoding(input_str)
    assert output == input_str


def test_sanitize_emoji_replacement():
    input_str = "Good job! 👍🏻"
    expected = "Good job! [EMOJI]"
    output = sanitize_prompt_for_encoding(input_str)
    assert output == expected


def test_sanitize_strict_ascii_only():
    # Enable strict ASCII mode for testing
    set_ascii_strict_mode(True)
    try:
        input_str = "Résumé café — naïve 😊"
        output = sanitize_prompt_for_encoding(input_str)
        # In strict mode, output must be ASCII-only
        assert all(ord(ch) < 128 for ch in output)
        # And non-ASCII characters should be dropped or replaced
        assert "Résumé" not in output
        assert "😊" not in output
    finally:
        # Reset mode after test
        set_ascii_strict_mode(False)
