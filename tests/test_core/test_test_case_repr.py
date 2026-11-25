from enum import Enum

from deepteam.test_case import RTTestCase


class _VulnerabilityType(Enum):
    BIAS = "bias"


def test_repr_with_default_vulnerability_type_none():
    case = RTTestCase(vulnerability="Bias")

    rendered = repr(case)

    assert "vulnerability_type=None" in rendered


def test_repr_with_string_vulnerability_type():
    # pydantic validates the declared Enum type on normal construction, so use
    # model_construct to exercise repr with a plain string value.
    case = RTTestCase.model_construct(
        vulnerability="Bias", vulnerability_type="custom-type"
    )

    assert "vulnerability_type=custom-type" in repr(case)


def test_repr_with_enum_vulnerability_type():
    case = RTTestCase(
        vulnerability="Bias", vulnerability_type=_VulnerabilityType.BIAS
    )

    assert "vulnerability_type=bias" in repr(case)
