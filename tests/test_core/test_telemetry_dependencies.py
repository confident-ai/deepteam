import ast
import importlib.metadata
import sys
from pathlib import Path

TELEMETRY_MODULE = Path(__file__).parents[2] / "deepteam" / "telemetry.py"

# Import names that do not match their distribution name on PyPI.
DISTRIBUTION_NAME_OVERRIDES = {"sentry_sdk": "sentry-sdk"}

# `sys.stdlib_module_names` only exists on Python 3.10+, while the package
# supports Python 3.9.
STDLIB_FALLBACK = frozenset(
    {"contextlib", "enum", "logging", "os", "socket", "sys", "typing", "uuid"}
)


def get_declared_distribution_names():
    requirements = importlib.metadata.requires("deepteam") or []

    names = set()
    for requirement in requirements:
        # Strip markers (e.g. `x ; python_version >= "3.10"`) and specifiers.
        name = requirement.split(";")[0].strip()
        for separator in "<>=!~[(":
            name = name.split(separator)[0]
        names.add(name.strip().lower())

    return names


def get_top_level_third_party_modules():
    tree = ast.parse(TELEMETRY_MODULE.read_text(encoding="utf-8"))
    stdlib = frozenset(
        getattr(sys, "stdlib_module_names", ())
    ) | STDLIB_FALLBACK

    modules = set()
    for node in tree.body:
        # Only module level imports count: imports inside a `try/except
        # ImportError` block are optional by design.
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module
        ):
            modules.add(node.module.split(".")[0])

    return {module for module in modules if module not in stdlib}


def test_telemetry_imports_are_declared_dependencies():
    declared = get_declared_distribution_names()

    missing = sorted(
        distribution
        for distribution in (
            DISTRIBUTION_NAME_OVERRIDES.get(module, module.replace("_", "-"))
            for module in get_top_level_third_party_modules()
        )
        if distribution.lower() not in declared
    )

    assert not missing, (
        "deepteam/telemetry.py imports these at module level, but they are "
        f"not declared as dependencies: {missing}"
    )
