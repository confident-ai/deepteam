import asyncio
import base64
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from deepteam.attacks.single_turn import LSBImageStego
from deepteam.attacks.single_turn.constants import (
    SINGLE_TURN_ATTACK_CLASSES_MAP,
    SINGLE_TURN_ATTACK_INFO_MAP,
)

_DATA_URI_PREFIX = "data:image/png;base64,"


def _decode_uri(data_uri: str) -> bytes:
    assert data_uri.startswith(_DATA_URI_PREFIX)
    return base64.b64decode(data_uri[len(_DATA_URI_PREFIX) :])


class TestLSBImageStego:

    def test_lsb_initialization(self):
        attack = LSBImageStego()
        assert attack.weight == 1
        assert attack.get_name() == "LSBImageStego"
        assert SINGLE_TURN_ATTACK_CLASSES_MAP["LSBImageStego"] is LSBImageStego
        assert (
            SINGLE_TURN_ATTACK_INFO_MAP["LSBImageStego"].exploitability
            == "medium"
        )

    def test_lsb_initialization_with_weight(self):
        attack = LSBImageStego(weight=5)
        assert attack.weight == 5
        assert attack.get_name() == "LSBImageStego"

    def test_lsb_enhance_and_decode_sources(self):
        attack = LSBImageStego(delimiter="<<END>>")
        prompt = "Verify system configuration and return diagnostic report."
        data_uri = attack.enhance(prompt)
        raw_png = _decode_uri(data_uri)

        assert (
            LSBImageStego.decode_image(data_uri, delimiter="<<END>>") == prompt
        )
        assert (
            LSBImageStego.decode_image(raw_png, delimiter="<<END>>") == prompt
        )
        assert (
            LSBImageStego.decode_image(BytesIO(raw_png), delimiter="<<END>>")
            == prompt
        )

    @pytest.mark.asyncio
    async def test_lsb_multilingual_utf8_roundtrip(self):
        attack = LSBImageStego(channels="RGB")
        prompt = "EN: Audit | RU: Проверка | ZH: 系统安全 | EMOJI: 🔥🚀🌍"
        data_uri = await attack.a_enhance(prompt)
        assert LSBImageStego.decode_image(data_uri, channels="RGB") == prompt

    @pytest.mark.asyncio
    async def test_lsb_custom_carrier_and_pixel_delta(self, tmp_path: Path):
        carrier_path = tmp_path / "carrier.png"
        carrier_img = Image.new("RGBA", (64, 64), color=(123, 200, 45, 255))
        carrier_img.save(carrier_path, format="PNG")

        attack = LSBImageStego(carrier_image_path=carrier_path, channels="RGBA")
        prompt = "Stealth check: pixel channel delta must never exceed 1."
        out_png = tmp_path / "stego.png"
        out_png.write_bytes(_decode_uri(await attack.a_enhance(prompt)))

        with Image.open(out_png) as stego_img:
            stego_bytes = stego_img.convert("RGBA").tobytes()
        assert (
            max(abs(a - b) for a, b in zip(carrier_img.tobytes(), stego_bytes))
            <= 1
        )
        assert LSBImageStego.decode_image(out_png) == prompt

    @pytest.mark.asyncio
    async def test_lsb_concurrent_a_enhance_isolates_carrier(
        self, tmp_path: Path
    ):
        carrier_path = tmp_path / "shared.png"
        carrier_img = Image.new("RGBA", (64, 64), color=(80, 120, 160, 255))
        carrier_img.save(carrier_path, format="PNG")
        expected_base = carrier_img.tobytes()

        attack = LSBImageStego(carrier_image_path=carrier_path)
        prompts = [f"Payload #{i}: {'X' * (i * 5 + 1)}" for i in range(6)]
        results = await asyncio.gather(*(attack.a_enhance(p) for p in prompts))

        assert attack._base_image is not None
        assert attack._base_image.tobytes() == expected_base
        for prompt_text, uri in zip(prompts, results):
            assert LSBImageStego.decode_image(uri) == prompt_text

    def test_lsb_auto_scaling_and_carrier_too_small(self, tmp_path: Path):
        attack = LSBImageStego(
            channels="RGB", min_dimensions=(2, 2), background_color=(10, 20, 30)
        )
        long_prompt = "Auto-scaling synthetic carrier check! " * 15
        raw_png = _decode_uri(attack.enhance(long_prompt))
        with Image.open(BytesIO(raw_png)) as img:
            assert img.width * img.height > 4
        assert (
            LSBImageStego.decode_image(raw_png, channels="RGB") == long_prompt
        )

        tiny = tmp_path / "tiny.png"
        Image.new("RGB", (2, 2)).save(tiny, format="PNG")
        tiny_attack = LSBImageStego(carrier_image_path=tiny, channels="RGB")
        with pytest.raises(ValueError, match="too small to hold payload"):
            tiny_attack.enhance("Too large for 12 bits")

    def test_lsb_delimiter_and_decode_errors(self):
        attack = LSBImageStego(delimiter="====EOF====")
        with pytest.raises(ValueError, match="collides with delimiter"):
            attack.enhance("Prefix ====EOF==== Suffix")
        with pytest.raises(ValueError, match="collides with delimiter"):
            attack.enhance("Ends with ====EOF")

        blank_buf = BytesIO()
        Image.new("RGBA", (8, 8)).save(blank_buf, format="PNG")
        blank_buf.seek(0)
        with pytest.raises(ValueError, match="not found in image LSB stream"):
            LSBImageStego.decode_image(blank_buf)

        corrupt = b"\xff\xfe====EOF===="
        raw = bytearray(len(corrupt) * 8)
        for i, b in enumerate(corrupt):
            for s in range(7, -1, -1):
                raw[i * 8 + (7 - s)] = (b >> s) & 1
        corrupt_buf = BytesIO()
        Image.frombytes("RGBA", (len(raw) // 4, 1), bytes(raw)).save(
            corrupt_buf, format="PNG"
        )
        corrupt_buf.seek(0)
        with pytest.raises(ValueError, match="not valid UTF-8"):
            LSBImageStego.decode_image(corrupt_buf)

    @pytest.mark.asyncio
    async def test_lsb_validation_errors(self):
        attack = LSBImageStego()
        with pytest.raises(ValueError, match="Please provide valid text value"):
            attack.enhance("   ")
        with pytest.raises(ValueError, match="Please provide valid text value"):
            await attack.a_enhance("   ")
        for bad_src in ("   ", ".", Path(""), None):
            with pytest.raises(
                ValueError, match="Please provide valid image_path value"
            ):
                LSBImageStego.decode_image(bad_src)  # type: ignore[arg-type]
        for kwargs, msg in [
            ({"channels": "CMYK"}, "channels must be either 'RGB' or 'RGBA'"),
            ({"delimiter": ""}, "delimiter must be a non-empty string"),
            ({"carrier_image_path": "   "}, "valid carrier_image_path"),
            ({"carrier_image_path": 123}, "valid carrier_image_path"),
            ({"min_dimensions": (True, 64)}, "min_dimensions must be"),
            ({"min_dimensions": (0, 64)}, "min_dimensions must be"),
            (
                {"background_color": (256, 0, 0, 255)},
                "background_color must be",
            ),
            (
                {"background_color": (True, 0, 0, 255)},
                "background_color must be",
            ),
        ]:
            with pytest.raises(ValueError, match=msg):
                LSBImageStego(**kwargs)  # type: ignore[arg-type]
