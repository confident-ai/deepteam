#!/usr/bin/env python3
"""
This script evaluates a language model using the HarmActionsEval framework.
"""


from deepteam import red_team
from deepteam.frameworks import HarmActionsEval
from deepeval.models import GPTModel


model = GPTModel(temperature=1)

framework = HarmActionsEval(k=1)


if __name__ == "__main__":
    risk_assessment = red_team(
        model_callback=model,  # type: ignore
        framework=framework,
        async_mode=False,
        ignore_errors=False,
    )

    total = len(risk_assessment.test_cases)
    safe = sum(1 for tc in risk_assessment.test_cases if tc.score == 1.0)

    print(
        f"HarmActionsEval complete: {safe}/{total} actions remained safe within SafeActions@{framework.k}."
    )
