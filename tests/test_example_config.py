import os
import sys
import asyncio
import yaml
import pytest

# Add the parent directory to the path to import deepteam
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepteam.cli.main import _load_config, _load_callback_from_file


def test_example_config_loads():
    """Test that the example config loads correctly"""
    config_path = os.path.join(os.path.dirname(__file__), "..", "examples", "example_config.yaml")
    config = _load_config(config_path)
    
    # Check basic structure
    assert "models" in config
    assert "target" in config
    assert "system_config" in config
    assert "default_vulnerabilities" in config
    assert "attacks" in config
    
    # Check models
    assert config["models"]["simulator"] == "gpt-3.5-turbo-0125"
    assert config["models"]["evaluation"] == "gpt-4o"
    
    # Check target
    assert "callback" in config["target"]
    assert config["target"]["callback"]["file"] == "target_callback.py"
    assert config["target"]["callback"]["function"] == "model_callback"
    
    # Check system config
    assert config["system_config"]["max_concurrent"] == 10
    assert config["system_config"]["attacks_per_vulnerability_type"] == 3


def test_target_callback_loads():
    """Test that the target callback can be loaded"""
    callback_path = os.path.join(os.path.dirname(__file__), "..", "examples", "target_callback.py")
    
    # Test loading the callback function
    callback = _load_callback_from_file(callback_path, "model_callback")
    
    # Verify it's callable and async
    assert callable(callback)
    assert asyncio.iscoroutinefunction(callback)


@pytest.mark.asyncio
async def test_target_callback_execution():
    """Test that the target callback can execute (mock test)"""
    callback_path = os.path.join(os.path.dirname(__file__), "..", "examples", "target_callback.py")
    callback = _load_callback_from_file(callback_path, "model_callback")
    
    # Note: This would make an actual API call - in real tests you'd mock this
    # For now, just verify it's an async function that takes a string
    import inspect
    sig = inspect.signature(callback)
    assert len(sig.parameters) == 1
    assert "input" in sig.parameters
    assert sig.parameters["input"].annotation == str


if __name__ == "__main__":
    test_example_config_loads()
    test_target_callback_loads()
    print("✅ All tests passed!") 