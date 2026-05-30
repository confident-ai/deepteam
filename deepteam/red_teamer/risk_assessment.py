from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Any, Union
import datetime
import os
import json
from enum import Enum

from deepeval.test_case import Turn
from deepteam.vulnerabilities.types import VulnerabilityType
from deepteam.test_case import RTTestCase, RTTurn


class TestCasesList(list):

    def to_df(self) -> "pd.DataFrame":
        import pandas as pd

        data = []
        for case in self:
            case_data = {
                "Vulnerability": case.vulnerability,
                "Vulnerability Type": str(case.vulnerability_type.value),
                "Risk Category": case.risk_category,
                "Attack Enhancement": case.attack_method,
                "Input": case.input,
                "Actual Output": case.actual_output,
                "Score": case.score,
                "Reason": case.reason,
                "Error": case.error,
                "Status": (
                    "Passed"
                    if case.score and case.score > 0
                    else "Errored" if case.error else "Failed"
                ),
            }
            if case.metadata:
                case_data.update(case.metadata)
            data.append(case_data)
        return pd.DataFrame(data)


class VulnerabilityTypeResultsList(list):

    def to_df(self) -> "pd.DataFrame":
        import pandas as pd

        data = []
        for case in self:
            case_data = {
                "Vulnerability": case.vulnerability,
                "Vulnerability Type": str(case.vulnerability_type.value),
                "Pass Rate": case.pass_rate,
                "Passing": case.passing,
                "Failing": case.failing,
                "Errored": case.errored,
            }
            data.append(case_data)
        return pd.DataFrame(data)


class AttackMethodResultList(list):

    def to_df(self) -> "pd.DataFrame":
        import pandas as pd

        data = []
        for case in self:
            case_data = {
                "Attack Method": case.attack_method,
                "Pass Rate": case.pass_rate,
                "Passing": case.passing,
                "Failing": case.failing,
                "Errored": case.errored,
            }
            data.append(case_data)
        return pd.DataFrame(data)


class VulnerabilityTypeResult(BaseModel):
    vulnerability: str
    vulnerability_type: Union[VulnerabilityType, Enum]
    pass_rate: float
    passing: int
    failing: int
    errored: int


class AttackMethodResult(BaseModel):
    pass_rate: float
    passing: int
    failing: int
    errored: int
    attack_method: Optional[str] = None


class RedTeamingOverview(BaseModel):
    vulnerability_type_results: List[VulnerabilityTypeResult]
    attack_method_results: List[AttackMethodResult]
    errored: int
    run_duration: float

    def to_df(self):
        import pandas as pd

        data = []
        for result in self.vulnerability_type_results:
            data.append(
                {
                    "Vulnerability": result.vulnerability,
                    "Vulnerability Type": str(result.vulnerability_type.value),
                    "Total": result.passing + result.failing + result.errored,
                    "Pass Rate": result.pass_rate,
                    "Passing": result.passing,
                    "Failing": result.failing,
                    "Errored": result.errored,
                }
            )
        return pd.DataFrame(data)


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


class RiskAssessment(BaseModel):
    overview: RedTeamingOverview
    test_cases: List[RTTestCase]

    def __init__(self, **data):
        super().__init__(**data)
        self.test_cases: TestCasesList = TestCasesList[RTTestCase](
            self.test_cases
        )
        self.overview.vulnerability_type_results: (
            VulnerabilityTypeResultsList
        ) = VulnerabilityTypeResultsList[VulnerabilityTypeResult](
            self.overview.vulnerability_type_results
        )
        self.overview.attack_method_results: AttackMethodResultList = (
            AttackMethodResultList[AttackMethodResult](
                self.overview.attack_method_results
            )
        )

    @classmethod
    def load(cls, path: str) -> "RiskAssessment":
        """Load a previously saved RiskAssessment from a JSON file.

        Args:
            path: Path to the JSON file produced by `save()`.

        Returns:
            A fully reconstructed RiskAssessment instance with
            properly typed vulnerability enums and RTTurn objects.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        with open(path, "r") as f:
            data = json.load(f)

        enum_lookup = _build_vulnerability_type_lookup()

        for tc in data.get("test_cases", []):
            vt_value = tc.get("vulnerability_type")
            if vt_value is not None:
                tc["vulnerability_type"] = _resolve_vulnerability_type(
                    vt_value, enum_lookup
                )

            # Reconstruct RTTurn objects for multi-turn test cases
            raw_turns = tc.get("turns")
            if raw_turns is not None:
                tc["turns"] = [
                    RTTurn(**turn) if isinstance(turn, dict) else turn
                    for turn in raw_turns
                ]

        # Reconstruct vulnerability_type enums in overview results
        for vtr in data.get("overview", {}).get(
            "vulnerability_type_results", []
        ):
            vt_value = vtr.get("vulnerability_type")
            if vt_value is not None:
                vtr["vulnerability_type"] = _resolve_vulnerability_type(
                    vt_value, enum_lookup
                )

        return cls(**data)

    def save(self, to: str) -> str:
        try:
            new_filename = (
                datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
            )

            if not os.path.exists(to):
                try:
                    os.makedirs(to)
                except OSError as e:
                    raise OSError(f"Cannot create directory '{to}': {e}")

            full_file_path = os.path.join(to, new_filename)

            # Convert model to a dictionary
            data = self.model_dump(by_alias=True)

            # Write to JSON file
            with open(full_file_path, "w") as f:
                json.dump(data, f, indent=2, cls=EnumEncoder)

            print(
                f"🎉 Success! 🎉 Your risk assessment file has been saved to:\n📁 {full_file_path} ✅"
            )

            return full_file_path

        except OSError as e:
            raise OSError(f"Failed to save file to '{to}': {e}") from e


def _build_vulnerability_type_lookup() -> Dict[str, list]:
    """Build a reverse lookup from enum *value* → list of enum members
    across all known VulnerabilityType enum classes."""
    import importlib
    import pkgutil
    import deepteam.vulnerabilities as _vuln_pkg

    enum_classes: list = []
    for _importer, modname, _ispkg in pkgutil.walk_packages(
        _vuln_pkg.__path__, prefix=_vuln_pkg.__name__ + "."
    ):
        if not modname.endswith(".types"):
            continue
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, Enum)
                and attr is not Enum
            ):
                enum_classes.append(attr)

    lookup: Dict[str, list] = {}
    for cls in enum_classes:
        for member in cls:
            lookup.setdefault(member.value, []).append(member)
    return lookup


def _resolve_vulnerability_type(value: str, lookup: Dict[str, list]):
    """Resolve a string enum value back to the correct enum member.

    If the value is ambiguous (present in multiple enum classes),
    returns the first match — which is sufficient for all built-in
    vulnerability types where only `privilege_escalation` overlaps.

    Falls back to returning the raw string if no match is found
    (e.g. custom vulnerability types).
    """
    candidates = lookup.get(value)
    if candidates:
        return candidates[0]
    return value


def construct_risk_assessment_overview(
    red_teaming_test_cases: List[RTTestCase], run_duration: float
) -> RedTeamingOverview:
    # Group test cases by vulnerability type
    vulnerability_type_to_cases: Dict[
        VulnerabilityType,
        List[RTTestCase],
    ] = {}
    attack_method_to_cases: Dict[str, List[RTTestCase]] = {}

    errored = 0
    for test_case in red_teaming_test_cases:
        if test_case.error:
            errored += 1
            continue

        # Group by vulnerability type
        if test_case.vulnerability_type not in vulnerability_type_to_cases:
            vulnerability_type_to_cases[test_case.vulnerability_type] = []
        vulnerability_type_to_cases[test_case.vulnerability_type].append(
            test_case
        )

        # Group by attack method
        if test_case.attack_method is not None:
            if test_case.attack_method not in attack_method_to_cases:
                attack_method_to_cases[test_case.attack_method] = []
            attack_method_to_cases[test_case.attack_method].append(test_case)

    vulnerability_type_results = []
    attack_method_results = []

    # Stats per vulnerability type
    for vuln_type, test_cases in vulnerability_type_to_cases.items():
        passing = sum(
            1 for tc in test_cases if tc.score is not None and tc.score > 0
        )
        errored = sum(1 for tc in test_cases if tc.error is not None)
        failing = len(test_cases) - passing - errored
        valid_cases = len(test_cases) - errored
        pass_rate = (passing / valid_cases) if valid_cases > 0 else 0.0

        vulnerability_type_results.append(
            VulnerabilityTypeResult(
                vulnerability=test_cases[0].vulnerability if test_cases else "",
                vulnerability_type=vuln_type,
                pass_rate=pass_rate,
                passing=passing,
                failing=failing,
                errored=errored,
            )
        )

    # Stats per attack method
    if attack_method_to_cases is not None:
        for attack_method, test_cases in attack_method_to_cases.items():
            passing = sum(
                1 for tc in test_cases if tc.score is not None and tc.score > 0
            )
            errored = sum(1 for tc in test_cases if tc.error is not None)
            failing = len(test_cases) - passing - errored
            valid_cases = len(test_cases) - errored
            pass_rate = (passing / valid_cases) if valid_cases > 0 else 0.0

            attack_method_results.append(
                AttackMethodResult(
                    attack_method=attack_method,
                    pass_rate=pass_rate,
                    passing=passing,
                    failing=failing,
                    errored=errored,
                )
            )

    return RedTeamingOverview(
        vulnerability_type_results=vulnerability_type_results,
        attack_method_results=attack_method_results,
        errored=errored,
        run_duration=run_duration,
    )
