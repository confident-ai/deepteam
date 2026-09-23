import importlib.util
import os
import sys
import types

TELEMETRY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "deepteam",
    "telemetry.py",
)


class _BlockSentryFinder:
    """MetaPathFinder that refuses to import 'sentry_sdk'.

    Reproduces issue #263: a clean install of deepteam can end up without
    sentry_sdk on the path, and the old module-level `import sentry_sdk` used
    to raise ModuleNotFoundError and break every import of deepteam.telemetry.
    """

    def find_spec(self, name, path, target=None):
        if name == "sentry_sdk" or name.startswith("sentry_sdk."):
            raise ImportError("blocked for test")
        return None


def _load_telemetry(block_sentry):
    # Load telemetry.py as a standalone module, bypassing the package
    # __init__ (which pulls in deepeval etc.) so we can unit-test the
    # optional-sentry behaviour in isolation.
    for mod in list(sys.modules):
        if mod == "telemetry" or mod.startswith("telemetry."):
            del sys.modules[mod]

    saved_path = list(sys.meta_path)
    finder = _BlockSentryFinder()
    if block_sentry:
        # Drop any cached sentry_sdk so the blocker finder actually triggers
        # (a previous test may have imported it already).
        sys.modules.pop("sentry_sdk", None)
        sys.meta_path.insert(0, finder)
    try:
        spec = importlib.util.spec_from_file_location(
            "telemetry", TELEMETRY_PATH, submodule_search_locations=[]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["telemetry"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.meta_path[:] = saved_path


def test_telemetry_imports_with_sentry_sdk():
    telemetry = _load_telemetry(block_sentry=False)
    assert telemetry is not None
    assert telemetry.sentry_sdk is not None
    # With sentry_sdk present and telemetry not opted out, init is attempted.
    assert telemetry._telemetry_available is True


def test_telemetry_imports_without_sentry_sdk():
    # The key assertion for issue #263: importing deepteam.telemetry must NOT
    # raise ModuleNotFoundError when sentry_sdk is missing from the environment.
    telemetry = _load_telemetry(block_sentry=True)
    assert telemetry is not None
    # The optional module alias is None so call sites guard on it.
    assert telemetry.sentry_sdk is None
