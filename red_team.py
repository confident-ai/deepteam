from deepteam import red_team
from deepteam.vulnerabilities import (
    Bias,
    Toxicity,
    Competition,
    ExcessiveAgency,
    Misinformation,
    PersonalSafety,
    IntellectualProperty,
)
from deepteam.attacks.single_turn import PromptInjection
from deepeval.models import GeminiModel

async def model_callback(input: str) -> str:
    # Replace this with your LLM application
    return "Sorry, I can't do that."


bias = Bias(types=["race", "gender", "politics", "religion"])
toxicity = Toxicity(types=["insults"])
competition = Competition(types=["competitor mention"])
excessive_agency = ExcessiveAgency(types=["autonomy"])
misinformation = Misinformation(types=["expertize misrepresentation"])
personal_safety = PersonalSafety(types=["bullying"])
intellectual_property = IntellectualProperty(
    types=["copyright violations", "imitation"]
)


prompt_injection = PromptInjection()

red_team(
    model_callback=model_callback,
    vulnerabilities=[
        bias,
        toxicity,
        competition,
        excessive_agency,
        misinformation,
        personal_safety,
        intellectual_property,
    ],
    attacks=[prompt_injection],
)
