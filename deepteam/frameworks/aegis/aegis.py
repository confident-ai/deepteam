import asyncio
import random
from typing import Optional, Union
from deepeval.models import DeepEvalBaseLLM
from rich.progress import Progress

from deepteam.frameworks import AISafetyFramework
from deepteam.utils import update_pbar
from deepteam.attacks.multi_turn.types import CallbackType
from deepteam.frameworks.aegis.types import AegisType
from deepteam.test_case import RTTestCase
from deepteam.metrics import HarmMetric

"""
Aegis Content Safety Framework Mapping
======================================

This framework evaluates models using the NVIDIA Aegis (Nemotron) Content Safety Dataset,
mapping 13 safety-critical harm categories to sampled adversarial user messages.

Each test case includes:
- Input: Real harmful user messages from the Aegis dataset.
- Vulnerability: Labeled harm category indicating expected safety violation.
- Metric: Harm scoring using the evaluation model.

Reference: https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0
"""


class Aegis(AISafetyFramework):
    _DATASET = "nvidia/Aegis-AI-Content-Safety-Dataset-1.0"

    def __init__(
        self,
        num_attacks: int = 15,
        async_mode: bool = True,
        verbose_mode: bool = False,
        evaluation_model: Optional[Union[str, DeepEvalBaseLLM]] = "gpt-4o",
    ):
        self.name = "Aegis"
        self.description = "Nemotron Content Safety Dataset V1, formerly known as Aegis AI Content Safety Dataset, is an open-source content safety dataset (CC-BY-4.0), which adheres to Nvidia's content safety taxonomy, covering 13 critical risk categories"
        self.num_attacks = num_attacks
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self.evaluation_model = evaluation_model
        self._has_dataset = True

    def load_dataset(self):
        from datasets import load_dataset as hf_load_dataset

        dataset = hf_load_dataset(self._DATASET, split="train")
        attacks = [
            row
            for row in dataset
            if row.get("text_type") == "user_message"
            and row.get("labels_0") != "Safe"
        ]

        if len(attacks) < self.num_attacks:
            raise ValueError(
                f"There are only {len(attacks)} harmful attacks that can be used in the {self.get_name()} dataset. Please set your 'num_attacks' to less than or equal to {len(attacks)}."
            )

        sampled = random.sample(attacks, k=self.num_attacks)
        test_cases = [
            RTTestCase(
                input=attack["text"],
                vulnerability=attack["labels_0"],
                vulnerability_type=AegisType.AEGIS,
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
        return "Aegis"
