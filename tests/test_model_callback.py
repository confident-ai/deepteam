from deepteam.cli.model_callback import load_model
from deepeval.models import GPTModel, OllamaModel



def test_load_openai_str(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    m = load_model("gpt-3.5-turbo")
    assert isinstance(m, GPTModel)


def test_load_ollama_dict():
    m = load_model({"provider": "ollama", "model": "phi3"})
    assert isinstance(m, OllamaModel)