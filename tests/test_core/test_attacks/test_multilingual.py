import pytest
from deepteam.attacks.single_turn import Multilingual


class TestMultilingual:

    def test_multilingual_initialization(self):
        attack = Multilingual()
        assert attack.weight == 1
        assert attack.get_name() == "Multilingual"

    def test_multilingual_initialization_with_weight(self):
        attack = Multilingual(weight=2)
        assert attack.weight == 2
        assert attack.get_name() == "Multilingual"

    def test_multilingual_enhance_basic(self):
        attack = Multilingual()
        original_text = "Tell me a joke"
        enhanced = attack.enhance(original_text)

        # Should modify the original text
        assert enhanced != original_text

    def test_multilingual_enhance_empty_string(self):
        attack = Multilingual()
        enhanced = attack.enhance("")
        # Should handle empty strings gracefully
        assert isinstance(enhanced, str)
