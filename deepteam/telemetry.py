from contextlib import contextmanager
import os
import uuid
from enum import Enum
from typing import List, Optional


class Feature(Enum):
    REDTEAMING = "redteaming"
    GUARDRAIL = "guardrail"
    UNKNOWN = "unknown"


TELEMETRY_DATA_FILE = ".deepteam/telemetry.txt"


#########################################################
### Telemetry Config ####################################
#########################################################


def telemetry_opt_out():
    return os.getenv("DEEPTEAM_TELEMETRY_OPT_OUT") == "YES"


posthog = None

if not telemetry_opt_out():
    try:
        from posthog import Posthog

        posthog = Posthog(
            project_api_key="phc_IXvGRcscJJoIb049PtjIZ65JnXQguOUZ5B5MncunFdB",
            host="https://us.i.posthog.com",
        )
    except ImportError:
        pass


def is_running_in_jupyter_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" in get_ipython().config:
            return True
    except Exception:
        pass
    return False


IS_RUNNING_IN_JUPYTER = (
    "jupyter" if is_running_in_jupyter_notebook() else "other"
)

#########################################################
### Context Managers ####################################
#########################################################


@contextmanager
def capture_red_teamer_run(
    vulnerabilities: List[str],
    attacks: List[str],
    framework: Optional[str] = None,
):
    if posthog is not None:
        properties = {
            "environment": IS_RUNNING_IN_JUPYTER,
            "user.status": get_status(),
            "feature_status.redteaming": get_feature_status(
                Feature.REDTEAMING
            ),
            "vulnerabilities": sorted(set(vulnerabilities)),
            "attacks": sorted(set(attacks)),
        }
        if framework is not None:
            properties["framework"] = framework
        _capture("Invoked redteamer", properties)
        set_last_feature(Feature.REDTEAMING)
    yield


@contextmanager
def capture_guardrail_run(type: str, guards: List[str]):
    if posthog is not None:
        _capture(
            "Invoked guardrail",
            {
                "environment": IS_RUNNING_IN_JUPYTER,
                "user.status": get_status(),
                "feature_status.guardrail": get_feature_status(
                    Feature.GUARDRAIL
                ),
                "guardrail.type": type,
                "guards": sorted(set(guards)),
            },
        )
        set_last_feature(Feature.GUARDRAIL)
    yield


def _capture(event: str, properties: dict) -> None:
    try:
        posthog.capture(
            distinct_id=get_unique_id(), event=event, properties=properties
        )
    except Exception:
        pass


#########################################################
### Helper Functions ####################################
#########################################################


def read_telemetry_file() -> dict:
    """Reads the telemetry data file and returns the key-value pairs as a dictionary."""
    if not os.path.exists(TELEMETRY_DATA_FILE):
        return {}
    with open(TELEMETRY_DATA_FILE, "r") as file:
        lines = file.readlines()
    data = {}
    for line in lines:
        key, _, value = line.strip().partition("=")
        data[key] = value
    return data


def write_telemetry_file(data: dict):
    """Writes the given key-value pairs to the telemetry data file."""
    os.makedirs(os.path.dirname(TELEMETRY_DATA_FILE), exist_ok=True)
    with open(TELEMETRY_DATA_FILE, "w") as file:
        for key, value in data.items():
            file.write(f"{key}={value}\n")


def get_status() -> str:
    """Gets the status from the telemetry file."""
    data = read_telemetry_file()
    return data.get("DEEPTEAM_STATUS", "new")


def get_unique_id() -> str:
    """Gets or generates a unique ID and updates the telemetry file."""
    data = read_telemetry_file()
    unique_id = data.get("DEEPTEAM_ID")
    if not unique_id:
        unique_id = str(uuid.uuid4())
        data["DEEPTEAM_ID"] = unique_id
        data["DEEPTEAM_STATUS"] = "new"
    else:
        data["DEEPTEAM_STATUS"] = "old"
    write_telemetry_file(data)
    return unique_id


def get_last_feature() -> Feature:
    """Gets the last feature from the telemetry file."""
    data = read_telemetry_file()
    last_feature = data.get("DEEPTEAM_LAST_FEATURE")
    if last_feature and last_feature in Feature._value2member_map_:
        return Feature(last_feature)
    return Feature.UNKNOWN


def set_last_feature(feature: Feature):
    """Sets the last feature in the telemetry file."""
    if feature not in Feature:
        raise ValueError(f"Invalid feature: {feature}")
    data = read_telemetry_file()
    data["DEEPTEAM_LAST_FEATURE"] = feature.value
    feature_status_key = f"DEEPTEAM_{feature.value.upper()}_STATUS"
    data[feature_status_key] = "old"
    write_telemetry_file(data)


def get_feature_status(feature: Feature) -> str:
    """Gets the status of a feature ('new' or 'old') from the telemetry file."""
    data = read_telemetry_file()
    feature_status_key = f"DEEPTEAM_{feature.value.upper()}_STATUS"
    return data.get(feature_status_key, "new")
