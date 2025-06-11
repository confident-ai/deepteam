import yaml
import typer
import importlib.util
import sys
import os
from typing import Callable, Awaitable

from . import config
from .model_callback import load_model

from deepteam.red_teamer import RedTeamer
from deepteam.vulnerabilities import (
    Bias,
    Toxicity,
    Misinformation,
    IllegalActivity,
    PromptLeakage,
    PIILeakage,
    UnauthorizedAccess,
    ExcessiveAgency,
    Robustness,
    IntellectualProperty,
    Competition,
    GraphicContent,
    PersonalSafety,
    CustomVulnerability,
)
from deepteam.attacks.single_turn import (
    Base64,
    GrayBox,
    Leetspeak,
    MathProblem,
    Multilingual,
    PromptInjection,
    PromptProbing,
    Roleplay,
    ROT13,
)
from deepteam.attacks.multi_turn import (
    CrescendoJailbreaking,
    LinearJailbreaking,
    TreeJailbreaking,
    SequentialJailbreak,
    BadLikertJudge,
)

app = typer.Typer(name="deepteam")

VULN_CLASSES = [
    Bias,
    Toxicity,
    Misinformation,
    IllegalActivity,
    PromptLeakage,
    PIILeakage,
    UnauthorizedAccess,
    ExcessiveAgency,
    Robustness,
    IntellectualProperty,
    Competition,
    GraphicContent,
    PersonalSafety,
]
VULN_MAP = {cls().get_name(): cls for cls in VULN_CLASSES}

ATTACK_CLASSES = [
    Base64,
    GrayBox,
    Leetspeak,
    MathProblem,
    Multilingual,
    PromptInjection,
    PromptProbing,
    Roleplay,
    ROT13,
    CrescendoJailbreaking,
    LinearJailbreaking,
    TreeJailbreaking,
    SequentialJailbreak,
    BadLikertJudge,
]
ATTACK_MAP = {cls().get_name(): cls for cls in ATTACK_CLASSES}


def _build_vulnerability(cfg: dict):
    name = cfg.get("name")
    if not name:
        raise ValueError("Vulnerability entry missing 'name'")
    if name == "CustomVulnerability":
        return CustomVulnerability(
            name=cfg.get("custom_name", "Custom"),
            types=cfg.get("types"),
            custom_prompt=cfg.get("prompt"),
        )
    cls = VULN_MAP.get(name)
    if not cls:
        raise ValueError(f"Unknown vulnerability: {name}")
    return cls(types=cfg.get("types"))


def _build_attack(cfg: dict):
    name = cfg.get("name")
    if not name:
        raise ValueError("Attack entry missing 'name'")
    cls = ATTACK_MAP.get(name)
    if not cls:
        raise ValueError(f"Unknown attack: {name}")
    kwargs = {}
    if "weight" in cfg:
        kwargs["weight"] = cfg["weight"]
    if "type" in cfg:
        kwargs["type"] = cfg["type"]
    if "persona" in cfg:
        kwargs["persona"] = cfg["persona"]
    if "category" in cfg:
        kwargs["category"] = cfg["category"]
    if "turns" in cfg:
        kwargs["turns"] = cfg["turns"]
    if "enable_refinement" in cfg:
        kwargs["enable_refinement"] = cfg["enable_refinement"]
    return cls(**kwargs)


def _load_callback_from_file(file_path: str, function_name: str) -> Callable[[str], Awaitable[str]]:
    """Load a callback function from a Python file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Target callback file not found: {file_path}")
    
    spec = importlib.util.spec_from_file_location("target_module", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules["target_module"] = module
    spec.loader.exec_module(module)
    
    if not hasattr(module, function_name):
        raise AttributeError(f"Function '{function_name}' not found in {file_path}")
    
    callback = getattr(module, function_name)
    if not callable(callback):
        raise TypeError(f"'{function_name}' in {file_path} is not callable")
    
    return callback


def _load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


@app.command()
def run(
    config_file: str,
    max_concurrent: int = typer.Option(None, "-c", "--max-concurrent", help="Maximum concurrent operations (overrides config)"),
    attacks_per_vuln: int = typer.Option(None, "-a", "--attacks-per-vuln", help="Number of attacks per vulnerability type (overrides config)")
):
    """Run a red teaming execution based on a YAML configuration"""
    cfg = _load_config(config_file)
    config.apply_env()

    # Parse red teaming models (moved out of target section)
    models_cfg = cfg.get("models", {})
    simulator_model_spec = models_cfg.get("simulator", "gpt-3.5-turbo-0125")
    evaluation_model_spec = models_cfg.get("evaluation", "gpt-4o")
    
    simulator_model = load_model(simulator_model_spec)
    evaluation_model = load_model(evaluation_model_spec)

    # Parse system configuration (renamed from options)
    system_config = cfg.get("system_config", {})
    final_max_concurrent = max_concurrent if max_concurrent is not None else system_config.get("max_concurrent", 10)
    final_attacks_per_vuln = attacks_per_vuln if attacks_per_vuln is not None else system_config.get("attacks_per_vulnerability_type", 1)

    # Parse target configuration
    target_cfg = cfg.get("target", {})
    target_purpose = target_cfg.get("purpose", "")

    red_teamer = RedTeamer(
        simulator_model=simulator_model,
        evaluation_model=evaluation_model,
        target_purpose=target_purpose,
        async_mode=system_config.get("run_async", True),
        max_concurrent=final_max_concurrent,
    )

    vulnerabilities_cfg = cfg.get("default_vulnerabilities", [])
    vulnerabilities_cfg += cfg.get("custom_vulnerabilities", [])
    vulnerabilities = [_build_vulnerability(v) for v in vulnerabilities_cfg]

    attacks = [_build_attack(a) for a in cfg.get("attacks", [])]

    # Load target model callback - support both model specs and custom callbacks
    model_callback = None
    
    if "callback" in target_cfg:
        # Load custom callback from file
        callback_cfg = target_cfg["callback"]
        file_path = callback_cfg.get("file")
        function_name = callback_cfg.get("function", "model_callback")
        
        if not file_path:
            raise ValueError("Target callback configuration missing 'file' field")
        
        model_callback = _load_callback_from_file(file_path, function_name)
        
    elif "model" in target_cfg:
        # Use model specification for simple cases
        target_model_spec = target_cfg["model"]
        target_model = load_model(target_model_spec)
        
        async def model_callback(input: str) -> str:
            response = await target_model.a_generate(input)
            # Ensure we return a string, handle different response types
            if isinstance(response, tuple):
                return str(response[0]) if response else "Empty response"
            return str(response)
    else:
        raise ValueError("Target configuration must specify either 'model' or 'callback'")

    risk = red_teamer.red_team(
        model_callback=model_callback,
        vulnerabilities=vulnerabilities,
        attacks=attacks,
        attacks_per_vulnerability_type=final_attacks_per_vuln,
        ignore_errors=system_config.get("ignore_errors", False),
    )

    red_teamer._print_risk_assessment()
    return risk


@app.command()
def login(api_key: str = typer.Argument(..., help="API Key (auto-detects provider based on prefix)")):
    """Store API key for later runs. Auto-detects provider from key prefix."""
    # Auto-detect provider based on key prefix
    if api_key.startswith("sk-proj"):
        config.set_key("OPENAI_API_KEY", api_key)
        typer.echo("OpenAI API key saved.")
    elif api_key.startswith("sk-ant-"):
        config.set_key("ANTHROPIC_API_KEY", api_key)
        typer.echo("Anthropic API key saved.")
    elif api_key.startswith("AIza"):
        config.set_key("GOOGLE_API_KEY", api_key)
        typer.echo("Google/Gemini API key saved.")
    else:
        # Default to OpenAI for backwards compatibility
        config.set_key("OPENAI_API_KEY", api_key)
        typer.echo("API key saved as OpenAI (unrecognized prefix).")


@app.command()
def logout():
    """Remove stored API key."""
    config.remove_key("OPENAI_API_KEY")
    typer.echo("Logged out.")


@app.command("set-azure-openai")
def set_azure_openai(
    openai_api_key: str = typer.Option(..., "--openai-api-key"),
    openai_endpoint: str = typer.Option(..., "--openai-endpoint"),
    openai_api_version: str = typer.Option(..., "--openai-api-version"),
    openai_model_name: str = typer.Option(..., "--openai-model-name"),
    deployment_name: str = typer.Option(..., "--deployment-name"),
    model_version: str = typer.Option(None, "--model-version"),
):
    """Configure Azure OpenAI credentials."""
    config.set_key("AZURE_OPENAI_API_KEY", openai_api_key)
    config.set_key("AZURE_OPENAI_ENDPOINT", openai_endpoint)
    config.set_key("OPENAI_API_VERSION", openai_api_version)
    config.set_key("AZURE_MODEL_NAME", openai_model_name)
    config.set_key("AZURE_DEPLOYMENT_NAME", deployment_name)
    if model_version:
        config.set_key("AZURE_MODEL_VERSION", model_version)
    config.set_key("USE_AZURE_OPENAI", "YES")
    config.set_key("USE_LOCAL_MODEL", "NO")
    typer.echo("Azure OpenAI configured.")


@app.command("unset-azure-openai")
def unset_azure_openai():
    """Remove Azure OpenAI configuration."""
    for key in [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "OPENAI_API_VERSION",
        "AZURE_MODEL_NAME",
        "AZURE_DEPLOYMENT_NAME",
        "AZURE_MODEL_VERSION",
        "USE_AZURE_OPENAI",
    ]:
        config.remove_key(key)
    typer.echo("Azure OpenAI unset.")


@app.command("set-local-model")
def set_local_model(
    model_name: str = typer.Argument(...),
    base_url: str = typer.Option(..., "--base-url"),
    api_key: str = typer.Option(None, "--api-key"),
):
    """Configure a local model endpoint."""
    config.set_key("LOCAL_MODEL_NAME", model_name)
    config.set_key("LOCAL_MODEL_BASE_URL", base_url)
    if api_key:
        config.set_key("LOCAL_MODEL_API_KEY", api_key)
    typer.echo("Local model configured.")


@app.command("unset-local-model")
def unset_local_model():
    """Remove local model configuration."""
    config.remove_key("LOCAL_MODEL_NAME")
    config.remove_key("LOCAL_MODEL_BASE_URL")
    config.remove_key("LOCAL_MODEL_API_KEY")
    typer.echo("Local model unset.")

@app.command("set-ollama")
def set_ollama(
    model_name: str = typer.Argument(...),
    base_url: str = typer.Option("http://localhost:11434", "--base-url"),
):
    """Use a local Ollama model."""
    config.set_key("LOCAL_MODEL_NAME", model_name)
    config.set_key("LOCAL_MODEL_BASE_URL", base_url)
    config.set_key("LOCAL_MODEL_API_KEY", "ollama")
    config.set_key("USE_LOCAL_MODEL", "YES")
    config.set_key("USE_AZURE_OPENAI", "NO")
    typer.echo("Ollama model configured.")


@app.command("unset-ollama")
def unset_ollama():
    """Stop using local Ollama model."""
    for key in [
        "LOCAL_MODEL_NAME",
        "LOCAL_MODEL_BASE_URL",
        "LOCAL_MODEL_API_KEY",
        "USE_LOCAL_MODEL",
    ]:
        config.remove_key(key)
    typer.echo("Ollama model unset.")


@app.command("set-gemini")
def set_gemini(
    model_name: str = typer.Option(None, "--model-name"),
    google_api_key: str = typer.Option(None, "--google-api-key"),
    project_id: str = typer.Option(None, "--project-id"),
    location: str = typer.Option(None, "--location"),
):
    """Configure Gemini models via API key or Vertex AI."""
    if not google_api_key and not (project_id and location):
        typer.echo(
            "Provide --google-api-key or both --project-id and --location.", err=True
        )
        raise typer.Exit(code=1)
    config.set_key("USE_GEMINI_MODEL", "YES")
    if model_name:
        config.set_key("GEMINI_MODEL_NAME", model_name)
    if google_api_key:
        config.set_key("GOOGLE_API_KEY", google_api_key)
    else:
        config.set_key("GOOGLE_GENAI_USE_VERTEXAI", "YES")
        config.set_key("GOOGLE_CLOUD_PROJECT", project_id)
        config.set_key("GOOGLE_CLOUD_LOCATION", location)
    typer.echo("Gemini configured.")


@app.command("unset-gemini")
def unset_gemini():
    """Remove Gemini configuration."""
    for key in [
        "USE_GEMINI_MODEL",
        "GEMINI_MODEL_NAME",
        "GOOGLE_API_KEY",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
    ]:
        config.remove_key(key)
    typer.echo("Gemini unset.")

if __name__ == "__main__":
    app()