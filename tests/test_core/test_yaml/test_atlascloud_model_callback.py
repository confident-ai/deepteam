from deepteam.cli import model_callback


class DummyLocalModel:
    def __init__(
        self,
        model_name,
        base_url=None,
        api_key=None,
        temperature=0,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature


def _patch_local_model(monkeypatch):
    monkeypatch.setattr(model_callback, "LocalModel", DummyLocalModel)


def _clear_atlascloud_env(monkeypatch):
    for name in (
        "ATLASCLOUD_API_KEY",
        "ATLAS_CLOUD_API_KEY",
        "ATLASCLOUD_API_BASE",
        "ATLASCLOUD_BASE_URL",
        "ATLAS_CLOUD_API_BASE",
        "ATLAS_CLOUD_BASE_URL",
        "ATLASCLOUD_MODEL",
        "ATLAS_CLOUD_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_atlascloud_provider_uses_defaults(monkeypatch):
    _patch_local_model(monkeypatch)
    _clear_atlascloud_env(monkeypatch)

    model = model_callback.load_model({"provider": "atlascloud"})

    assert model.model_name == model_callback.ATLASCLOUD_DEFAULT_MODEL
    assert model.base_url == model_callback.ATLASCLOUD_BASE_URL
    assert model.api_key is None
    assert model.temperature == 0


def test_atlascloud_provider_reads_environment_fallbacks(monkeypatch):
    _patch_local_model(monkeypatch)
    _clear_atlascloud_env(monkeypatch)
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "env-key")
    monkeypatch.setenv("ATLASCLOUD_API_BASE", "https://atlas.example/v1")
    monkeypatch.setenv("ATLASCLOUD_MODEL", "deepseek-ai/deepseek-v4-pro")

    model = model_callback.load_model(
        {"provider": "atlas-cloud", "temperature": 0.2}
    )

    assert model.model_name == "deepseek-ai/deepseek-v4-pro"
    assert model.base_url == "https://atlas.example/v1"
    assert model.api_key == "env-key"
    assert model.temperature == 0.2


def test_atlascloud_explicit_config_takes_precedence(monkeypatch):
    _patch_local_model(monkeypatch)
    _clear_atlascloud_env(monkeypatch)
    monkeypatch.setenv("ATLAS_CLOUD_API_KEY", "env-key")
    monkeypatch.setenv("ATLAS_CLOUD_API_BASE", "https://env.example/v1")
    monkeypatch.setenv("ATLAS_CLOUD_MODEL", "env/model")

    model = model_callback.load_model(
        {
            "provider": "atlas",
            "model": "explicit/model",
            "base_url": "https://explicit.example/v1",
            "api_key": "explicit-key",
            "temperature": 0.7,
        }
    )

    assert model.model_name == "explicit/model"
    assert model.base_url == "https://explicit.example/v1"
    assert model.api_key == "explicit-key"
    assert model.temperature == 0.7


def test_atlascloud_provider_aliases(monkeypatch):
    _patch_local_model(monkeypatch)
    _clear_atlascloud_env(monkeypatch)

    for provider in ("atlas", "atlas-cloud", "atlascloud"):
        model = model_callback.load_model(
            {"provider": provider, "model": "qwen/qwen3.5-flash"}
        )
        assert model.model_name == "qwen/qwen3.5-flash"
        assert model.base_url == model_callback.ATLASCLOUD_BASE_URL
