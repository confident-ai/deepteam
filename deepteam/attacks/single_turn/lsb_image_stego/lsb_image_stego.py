import asyncio
import base64
import math
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

from PIL import Image

from deepteam.attacks.base_attack import Exploitability
from deepteam.attacks.single_turn import AttackParameter, BaseSingleTurnAttack

StegoChannelMode = Literal["RGB", "RGBA"]
_DATA_URI_PREFIX = "data:image/png;base64,"


def _validate_stego_params(channels: str, delimiter: str) -> None:
    if channels not in ("RGB", "RGBA"):
        raise ValueError("channels must be either 'RGB' or 'RGBA'")
    if not isinstance(delimiter, str) or not delimiter:
        raise ValueError("delimiter must be a non-empty string")


class LSBImageStego(BaseSingleTurnAttack):
    name = "LSBImageStego"
    exploitability = Exploitability.MEDIUM
    description = (
        "An encoding-based attack that embeds prompts into the least "
        "significant bits (LSB) of a lossless PNG image."
    )
    parameters = {
        "carrier_image_path": AttackParameter(
            type="string",
            description="Optional path to a carrier image used for LSB embedding.",
        ),
        "channels": AttackParameter(
            type="enum",
            default="RGBA",
            options=["RGB", "RGBA"],
            description="Pixel channel mode used for LSB embedding.",
        ),
        "delimiter": AttackParameter(
            type="string",
            default="====EOF====",
            description="Sentinel appended after the UTF-8 payload.",
        ),
    }

    def __init__(
        self,
        weight: int = 1,
        carrier_image_path: Optional[Union[str, Path]] = None,
        channels: StegoChannelMode = "RGBA",
        delimiter: str = "====EOF====",
        min_dimensions: Tuple[int, int] = (64, 64),
        background_color: Union[
            Tuple[int, int, int], Tuple[int, int, int, int]
        ] = (240, 244, 248, 255),
    ):
        _validate_stego_params(channels, delimiter)
        if not isinstance(min_dimensions, tuple) or (
            len(min_dimensions) != 2
            or any(type(d) is not int or d < 1 for d in min_dimensions)
        ):
            raise ValueError(
                "min_dimensions must be a (width, height) tuple with positive integers"
            )
        if not isinstance(background_color, tuple) or (
            len(background_color) not in (3, 4)
            or any(
                type(c) is not int or not (0 <= c <= 255)
                for c in background_color
            )
        ):
            raise ValueError(
                "background_color must be an RGB or RGBA tuple with channel values in 0..255"
            )

        self.weight = weight
        self.channels: StegoChannelMode = channels
        self.delimiter = delimiter
        self.min_dimensions = min_dimensions
        self.background_color: Tuple[int, int, int, int] = (
            (*background_color, 255)
            if len(background_color) == 3
            else background_color
        )
        self._base_image: Optional[Image.Image] = None
        if carrier_image_path is not None:
            if not isinstance(carrier_image_path, (str, Path)):
                raise ValueError("Please provide a valid carrier_image_path")
            carrier_str = str(carrier_image_path)
            if not carrier_str.strip() or carrier_str == ".":
                raise ValueError("Please provide a valid carrier_image_path")
            with Image.open(carrier_str) as img:
                img.load()
                self._base_image = img.convert(channels).copy()

    def enhance(self, attack: str) -> str:
        if not isinstance(attack, str) or not attack.strip():
            raise ValueError("Please provide valid text value")
        payload_bytes = attack.encode("utf-8")
        delimiter_bytes = self.delimiter.encode("utf-8")
        framed_bytes = payload_bytes + delimiter_bytes
        if framed_bytes.find(delimiter_bytes) != len(payload_bytes):
            raise ValueError(
                f"Prompt collides with delimiter {self.delimiter!r}; choose a different delimiter"
            )

        required_bits = len(framed_bytes) * 8
        channels_per_pixel = 4 if self.channels == "RGBA" else 3

        if self._base_image is not None:
            size = self._base_image.size
            raw_bytes = bytearray(self._base_image.tobytes())
            if required_bits > len(raw_bytes):
                raise ValueError(
                    f"Carrier image ({size[0]}x{size[1]}, "
                    f"{len(raw_bytes)} bits capacity) is too small to hold payload "
                    f"of {required_bits} bits."
                )
        else:
            min_pixels = math.ceil(required_bits / channels_per_pixel)
            min_w, min_h = self.min_dimensions
            if min_w * min_h >= min_pixels:
                size = (min_w, min_h)
            else:
                w = max(min_w, math.ceil(math.sqrt(min_pixels)))
                size = (w, max(min_h, math.ceil(min_pixels / w)))
            raw_bytes = bytearray(
                self.background_color[:channels_per_pixel]
            ) * (size[0] * size[1])

        bit_index = 0
        for byte_val in framed_bytes:
            for shift in range(7, -1, -1):
                raw_bytes[bit_index] = (raw_bytes[bit_index] & 0xFE) | (
                    (byte_val >> shift) & 1
                )
                bit_index += 1

        with Image.frombytes(self.channels, size, bytes(raw_bytes)) as img:
            buf = BytesIO()
            img.save(fp=buf, format="PNG")
            encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"{_DATA_URI_PREFIX}{encoded}"

    async def a_enhance(self, attack: str) -> str:
        return await asyncio.to_thread(self.enhance, attack)

    @classmethod
    def decode_image(
        cls,
        image_path: Union[str, Path, BytesIO, bytes],
        channels: StegoChannelMode = "RGBA",
        delimiter: str = "====EOF====",
    ) -> str:
        _validate_stego_params(channels, delimiter)
        if isinstance(image_path, (str, Path)):
            path_str = str(image_path)
            if not path_str.strip() or path_str == ".":
                raise ValueError("Please provide valid image_path value")
            source: Union[str, Path, BytesIO] = (
                BytesIO(base64.b64decode(path_str[len(_DATA_URI_PREFIX) :]))
                if isinstance(image_path, str)
                and path_str.startswith(_DATA_URI_PREFIX)
                else image_path
            )
        elif isinstance(image_path, bytes):
            source = BytesIO(image_path)
        elif isinstance(image_path, BytesIO):
            source = image_path
        else:
            raise ValueError("Please provide valid image_path value")

        with Image.open(source) as img:
            raw_bytes = img.convert(channels).tobytes()

        delimiter_bytes = delimiter.encode("utf-8")
        extracted, current_byte, bit_count = bytearray(), 0, 0
        for channel_val in raw_bytes:
            current_byte = (current_byte << 1) | (channel_val & 1)
            bit_count += 1
            if bit_count == 8:
                extracted.append(current_byte)
                if extracted.endswith(delimiter_bytes):
                    try:
                        return extracted[: -len(delimiter_bytes)].decode(
                            "utf-8"
                        )
                    except UnicodeDecodeError as exc:
                        raise ValueError(
                            "Decoded LSB payload before delimiter is not valid UTF-8"
                        ) from exc
                current_byte, bit_count = 0, 0

        raise ValueError(
            f"Delimiter {delimiter!r} not found in image LSB stream"
        )

    def get_name(self) -> str:
        return self.name
