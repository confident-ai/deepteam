"""Guard against the default simulator model pointing at a retired model.

OpenAI retired ``gpt-3.5-turbo-0125`` on October 23, 2026, so requests
naming it fail. These tests inspect the declared defaults (no model is
instantiated and no API key is needed) to make sure the shipped defaults
never name a retired model again. See deepteam#306.
"""

import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "deepteam"
RETIRED_SIMULATOR_MODEL = "gpt-3.5-turbo-0125"
CURRENT_SIMULATOR_MODEL = "gpt-5.6-terra"


def _simulator_model_defaults():
    """Map of source file -> declared ``simulator_model`` default."""
    defaults = {}
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name != "__init__":
                continue
            args = list(reversed(node.args.args))
            values = list(reversed(node.args.defaults))
            for arg, default in zip(args, values):
                if arg.arg != "simulator_model":
                    continue
                if isinstance(default, ast.Constant):
                    defaults[str(path.relative_to(REPO_ROOT))] = default.value
    return defaults


def test_no_simulator_model_default_uses_retired_model():
    defaults = _simulator_model_defaults()
    assert defaults, "expected simulator_model defaults to be found"
    retired = {
        path: value
        for path, value in defaults.items()
        if value == RETIRED_SIMULATOR_MODEL
    }
    assert not retired, (
        f"simulator_model defaults still use retired model "
        f"{RETIRED_SIMULATOR_MODEL}: {retired}"
    )


def test_core_simulator_model_defaults_are_current():
    defaults = _simulator_model_defaults()
    for path in (
        "deepteam/red_teamer/red_teamer.py",
        "deepteam/attacks/attack_engine/attack_engine.py",
    ):
        assert defaults.get(path) == CURRENT_SIMULATOR_MODEL, (
            f"{path} defaults simulator_model to "
            f"{defaults.get(path)!r}, expected "
            f"{CURRENT_SIMULATOR_MODEL!r}"
        )


def test_cli_yaml_simulator_default_is_current():
    cli_source = (PACKAGE_DIR / "cli" / "main.py").read_text()
    assert RETIRED_SIMULATOR_MODEL not in cli_source
    assert CURRENT_SIMULATOR_MODEL in cli_source


def test_docs_do_not_document_retired_default():
    stale = [
        str(path.relative_to(REPO_ROOT))
        for path in sorted((REPO_ROOT / "docs").rglob("*.mdx"))
        if path.is_file() and RETIRED_SIMULATOR_MODEL in path.read_text()
    ]
    assert not stale, (
        f"docs still document retired default "
        f"{RETIRED_SIMULATOR_MODEL}: {stale}"
    )
