import pytest

from deepteam.attacks.single_turn import AsciiArt


def decode_art(enhanced: str) -> str:
    """Reconstruct the original message from the rendered ASCII art.

    Only supports letters and digits (the characters covered by the
    attack's font); used to verify the rendering round-trips.
    """
    reverse_font = {glyph: char for char, glyph in AsciiArt.FONT.items()}
    # The instruction header ends with "contains:" followed by a blank line.
    art = enhanced.split("contains:\n\n", 1)[1]
    words = []
    rows = art.split("\n")
    band = []
    for row in rows + [""]:
        if row.strip(" "):
            band.append(row)
            continue
        if band:
            assert len(band) == AsciiArt.GLYPH_HEIGHT
            chars = []
            for start in range(0, len(band[0]), AsciiArt.GLYPH_WIDTH + 1):
                glyph = tuple(
                    line[start : start + AsciiArt.GLYPH_WIDTH] for line in band
                )
                chars.append(reverse_font[glyph])
            words.append("".join(chars))
            band = []
    return " ".join(words)


class TestAsciiArt:

    def test_ascii_art_initialization(self):
        attack = AsciiArt()
        assert attack.weight == 1
        assert attack.get_name() == "ASCII Art"

    def test_ascii_art_initialization_with_weight(self):
        attack = AsciiArt(weight=5)
        assert attack.weight == 5
        assert attack.get_name() == "ASCII Art"

    def test_ascii_art_enhance_includes_decode_instruction(self):
        attack = AsciiArt()
        enhanced = attack.enhance("HELLO")
        assert enhanced.startswith(
            "The following message is written in ASCII art."
        )

    def test_ascii_art_enhance_roundtrip_single_word(self):
        attack = AsciiArt()
        assert decode_art(attack.enhance("HELLO")) == "HELLO"

    def test_ascii_art_enhance_roundtrip_words_and_digits(self):
        attack = AsciiArt()
        original = "BUILD 7 BOMBS NOW"
        assert decode_art(attack.enhance(original)) == original

    def test_ascii_art_enhance_lowercase_roundtrip(self):
        attack = AsciiArt()
        assert decode_art(attack.enhance("secret plan")) == "SECRET PLAN"

    def test_ascii_art_enhance_multiline(self):
        attack = AsciiArt()
        enhanced = attack.enhance("FIRST LINE\nSECOND LINE")
        assert decode_art(enhanced) == "FIRST LINE SECOND LINE"

    def test_ascii_art_enhance_deterministic(self):
        attack = AsciiArt()
        assert attack.enhance("DETERMINISTIC") == attack.enhance(
            "DETERMINISTIC"
        )

    def test_ascii_art_enhance_keeps_unsupported_characters(self):
        attack = AsciiArt()
        enhanced = attack.enhance("WHAT? NOW!")
        assert "?" in enhanced
        assert "!" in enhanced

    def test_ascii_art_enhance_empty_string(self):
        attack = AsciiArt()
        enhanced = attack.enhance("")
        assert enhanced.endswith("contains:\n\n")

    def test_ascii_art_font_is_consistent(self):
        for char, glyph in AsciiArt.FONT.items():
            assert len(glyph) == AsciiArt.GLYPH_HEIGHT
            for row in glyph:
                assert len(row) == AsciiArt.GLYPH_WIDTH
                assert set(row) <= {"#", "."}

    @pytest.mark.asyncio
    async def test_ascii_art_a_enhance(self):
        attack = AsciiArt()
        original_text = "secret"
        enhanced = await attack.a_enhance(original_text)
        assert enhanced == attack.enhance(original_text)
