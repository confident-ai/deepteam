import os
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "config", Path(__file__).resolve().parents[1] / "deepteam/cli/config.py"
)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)


def test_config_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "cfg")
    config.set_key("TEST_KEY", "1")
    assert config.get_key("TEST_KEY") == "1"
    config.apply_env()
    assert os.environ.get("TEST_KEY") == "1"
    config.remove_key("TEST_KEY")
    assert config.get_key("TEST_KEY") is None