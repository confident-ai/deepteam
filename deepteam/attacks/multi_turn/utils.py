from typing import List
from deepeval.test_case import Turn
from deepeval.models import DeepEvalBaseLLM
from deepteam.attacks import BaseAttack
import inspect
from deepteam.test_case import RTTurn


def append_target_turn(
    turns: List[RTTurn],
    target_response,
    turn_level_attack: str = None,
):
    """
    Append a model response to the turn history as an RTTurn.

    Normalizes input so the turn history always contains RTTurn objects,
    regardless of what model_callback returns. This guarantees that
    downstream code iterating the history can safely access .role and
    .content on every element.

    Args:
        turns: The conversation history to append to.
        target_response: The model's response. Accepts:
            - RTTurn: appended as-is.
            - str: wrapped as RTTurn(role="assistant", content=target_response).
            - other: coerced via str() and wrapped as an assistant RTTurn.
        turn_level_attack: Optional name of the turn-level attack that
            produced this response. Set as an attribute on the resulting
            RTTurn for downstream tracking.
    """
    # Normalize to RTTurn so the history is always typed consistently.
    if isinstance(target_response, RTTurn):
        turn = target_response
    elif isinstance(target_response, str):
        turn = RTTurn(role="assistant", content=target_response)
    else:
        # Fallback for any other model-specific return types
        turn = RTTurn(role="assistant", content=str(target_response))

    if turn_level_attack:
        turn.turn_level_attack = turn_level_attack

    turns.append(turn)


def update_turn_history(
    turn_history: List[Turn], user_input: str, assistant_output: str
):
    turn_history.append(
        Turn(
            role="user",
            content=user_input,
        )
    )
    turn_history.append(
        Turn(
            role="assistant",
            content=assistant_output,
        )
    )

    return turn_history


def enhance_attack(
    attack: BaseAttack, current_attack: str, simulator_model: DeepEvalBaseLLM
):
    sig = inspect.signature(attack.enhance)
    try:
        res = current_attack
        if "simulator_model" in sig.parameters:
            res = attack.enhance(
                attack=current_attack,
                simulator_model=simulator_model,
            )
        else:
            res = attack.enhance(attack=current_attack)

        return res
    except:
        return current_attack


async def a_enhance_attack(
    attack: BaseAttack, current_attack: str, simulator_model: DeepEvalBaseLLM
):
    sig = inspect.signature(attack.enhance)
    try:
        res = current_attack
        if "simulator_model" in sig.parameters:
            res = await attack.a_enhance(
                attack=current_attack,
                simulator_model=simulator_model,
            )
        else:
            res = await attack.a_enhance(attack=current_attack)

        return res
    except:
        return current_attack
