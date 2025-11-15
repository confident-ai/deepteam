import asyncio
import random
from typing import Literal, Optional, Union
from deepeval.models import DeepEvalBaseLLM
from rich.progress import Progress

from deepteam.utils import update_pbar
from deepteam.frameworks import AISafetyFramework
from deepteam.attacks.multi_turn.types import CallbackType
from deepteam.frameworks.beavertails.types import BeaverTailsType
from deepteam.test_case import RTTestCase
from deepteam.metrics import HarmMetric

"""
BeaverTails Safety Framework Mapping
====================================

This framework evaluates model safety using the BeaverTails dataset, a human-labeled
collection of unsafe QA pairs across 14 harm categories. It samples harmful prompts
and maps them to corresponding vulnerability categories.

Each test case includes:
- Input: Unsafe user prompts from BeaverTails.
- Vulnerability: One or more harm categories assigned in the dataset.
- Metric: Harm scoring using the selected evaluation model.

Reference: https://huggingface.co/datasets/PKU-Alignment/BeaverTails
"""


class BeaverTails(AISafetyFramework):
    _DATASET = "PKU-Alignment/BeaverTails"

    def __init__(
        self,
        num_attacks: int = 15,
        data_split: Literal["330k_train", "30k_train"] = "30k_train",
        async_mode: bool = True,
        verbose_mode: bool = False,
        evaluation_model: Optional[Union[str, DeepEvalBaseLLM]] = "gpt-4o",
    ):
        self.name = "BeaverTails"
        self.description = "BeaverTails is an AI safety-focused collection comprising a series of datasets, it contains human-labeled data consisting of question-answer (QA) pairs from 14 different harm categories."
        self.num_attacks = num_attacks
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self.evaluation_model = evaluation_model
        self.data_split = data_split
        self._has_dataset = True

    def load_dataset(self):
        from datasets import load_dataset as hf_load_dataset

        dataset = hf_load_dataset(self._DATASET, split=self.data_split)
        attacks = [row for row in dataset if row["is_safe"] == False]

        if len(attacks) < self.num_attacks:
            raise ValueError(
                f"There are only {len(attacks)} harmful attacks that can be used in the {self.get_name()} dataset. Please set your 'num_attacks' to less than or equal to {len(attacks)}."
            )

        sampled = random.sample(attacks, k=self.num_attacks)
        test_cases = [
            RTTestCase(
                input=attack["prompt"],
                vulnerability=" ".join(
                    [
                        category
                        for category, boolean in attack["category"].items()
                        if boolean
                    ]
                ),
                vulnerability_type=BeaverTailsType.BEAVERTAILS,
            )
            for attack in sampled
        ]
        self.test_cases = test_cases
        return self.test_cases

    def assess(
        self,
        model_callback: CallbackType,
        progress: Optional[Progress],
        task_id: Optional[int],
        ignore_errors: bool = True,
    ):
        for test_case in self.test_cases:
            try:
                test_case.actual_output = model_callback(test_case.input)
                metric = self._get_metric(test_case.vulnerability)
                metric.measure(test_case)
                test_case.score = metric.score
                test_case.reason = metric.reason
            except Exception as e:
                if ignore_errors:
                    test_case.error = e.message
                else:
                    raise
            update_pbar(progress=progress, pbar_id=task_id)

        return self.test_cases

    async def a_assess(
        self,
        model_callback: CallbackType,
        progress: Optional[Progress],
        task_id: Optional[int],
        ignore_errors: bool = True,
    ):

        async def evaluate_test_case(test_case):
            try:
                test_case.actual_output = await model_callback(test_case.input)
                metric = self._get_metric(test_case.vulnerability)
                await metric.a_measure(test_case)
                test_case.score = metric.score
                test_case.reason = metric.reason
            except Exception as e:
                if ignore_errors:
                    test_case.error = e.message
                else:
                    raise
            update_pbar(progress=progress, pbar_id=task_id)

        tasks = [evaluate_test_case(tc) for tc in self.test_cases]

        await asyncio.gather(*tasks)

        return self.test_cases

    def _get_metric(self, harm_category: str):
        return HarmMetric(
            harm_category=harm_category,
            model=self.evaluation_model,
            async_mode=self.async_mode,
            verbose_mode=self.verbose_mode,
        )

    def get_name(self):
        return "BeaverTails"
