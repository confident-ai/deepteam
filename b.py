from deepteam.attacks.multi_turn import LinearJailbreaking
from deepteam.attacks.single_turn import PromptInjection
from deepteam import red_team
from deepteam.vulnerabilities import Bias
from openai import AsyncOpenAI


async def model_callback(attack, turn_history):
    response = await AsyncOpenAI().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": attack}],
    )
    return response.choices[0].message.content


red_team(
    attacks=[LinearJailbreaking(num_turns=3)],
    vulnerabilities=[Bias(types=["race"], simulator_model="gpt-4o")],
    model_callback=model_callback,
    ignore_errors=False,
)
