import pytest
from deepteam.attacks.single_turn import Caesar


class TestCaesar:

    def test_caesar_initialization(self):
        attack = Caesar()
        assert attack.weight == 1
        assert attack.shift == 3
        assert attack.get_name() == "Caesar"

    def test_caesar_initialization_with_weight(self):
        attack = Caesar(weight=2)
        assert attack.weight == 2
        assert attack.get_name() == "Caesar"

    def test_caesar_initialization_with_shift(self):
        attack = Caesar(shift=5)
        assert attack.shift == 5

    def test_caesar_enhance_default_shift(self):
        attack = Caesar()
        original_text = "abc XYZ"
        enhanced = attack.enhance(original_text)
        # Default shift of 3, wrapping at the alphabet boundary.
        assert enhanced == "def ABC"

    def test_caesar_enhance_custom_shift(self):
        attack = Caesar(shift=1)
        assert attack.enhance("Hello") == "Ifmmp"

    def test_caesar_enhance_numbers_and_symbols_unchanged(self):
        attack = Caesar()
        original_text = "123 !@#"
        assert attack.enhance(original_text) == original_text

    def test_caesar_enhance_defeats_keyword_match(self):
        attack = Caesar()
        original_text = "ignore"
        enhanced = attack.enhance(original_text)
        assert "ignore" not in enhanced

    def test_caesar_enhance_full_rotation_is_identity(self):
        attack = Caesar(shift=26)
        original_text = "Hello World"
        assert attack.enhance(original_text) == original_text

    def test_caesar_enhance_empty_string(self):
        attack = Caesar()
        assert attack.enhance("") == ""

    @pytest.mark.asyncio
    async def test_caesar_a_enhance(self):
        attack = Caesar()
        original_text = "secret"
        enhanced = await attack.a_enhance(original_text)
        assert enhanced == attack.enhance(original_text)
