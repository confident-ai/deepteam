from typing import Optional, Tuple, Union

from deepteam.test_case import RTTestCase
from deepeval.utils import get_or_create_event_loop
from deepeval.metrics.utils import (
    construct_verbose_logs,
    trimAndLoadJson,
    initialize_model,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.indicator import metric_progress_indicator

from deepteam.metrics import BaseRedTeamingMetric
from deepteam.metrics.bias.template import (
    BiasTemplate,
)
from deepteam.metrics.bias.schema import *
from deepteam.utils import format_turns


class BiasMetric(BaseRedTeamingMetric):
    def __init__(
        self,
        purpose: str,
        bias_category: Optional[str] = None,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        async_mode: bool = True,
        verbose_mode: bool = False,
    ):
        self.purpose = purpose
        self.bias_category = bias_category
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        try:
            self.system_prompt = model.get_system_prompt()
        except:
            self.system_prompt = ""

    def measure(self, test_case: RTTestCase) -> float:

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(self, _show_indicator=False):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(test_case, _show_indicator=False)
                )
            else:
                score, reason = self.evaluate(test_case)
                self.reason = reason
                self.score = score
                self.success = self.score == 1
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[f"Score: {self.score}\nReason: {self.reason}"],
                )
                return self.score

    async def a_measure(
        self,
        test_case: RTTestCase,
        _show_indicator: bool = False,
    ) -> float:

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self, async_mode=True, _show_indicator=_show_indicator
        ):
            score, reason = await self._a_evaluate(test_case)
            self.reason = reason
            self.score = score
            self.success = self.score == 1
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[f"Score: {self.score}\nReason: {self.reason}"],
            )
            return self.score

    async def a_generate_purpose(self):
        if self.purpose:
            return self.purpose
        prompt = BiasTemplate.extract_purpose(self.system_prompt)
        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt=prompt, schema=Purpose
            )
            self.evaluation_cost += cost
            return res.purpose
        else:
            try:
                res: Purpose = await self.model.a_generate(
                    prompt=prompt, schema=Purpose
                )
                return res.purpose
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["purpose"]

    def generate_purpose(self):
        if self.purpose:
            return self.purpose
        prompt = BiasTemplate.extract_purpose(self.system_prompt)
        if self.using_native_model:
            res, cost = self.model.generate(prompt=prompt, schema=Purpose)
            self.evaluation_cost += cost
            return res.purpose
        else:
            try:
                res: Purpose = self.model.generate(
                    prompt=prompt, schema=Purpose
                )
                return res.purpose
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["purpose"]

    async def _a_evaluate(
        self, test_case: RTTestCase
    ) -> Tuple[Union[int, float], str]:

        if test_case.turns is not None:
            actual_output = format_turns(test_case.turns)
        else:
            actual_output = (
                "AI'S ACTUAL OUTPUT TO EVALUATE: \n"
                + test_case.actual_output
                + "\n"
            )

        prompt = BiasTemplate.generate_evaluation_results(
            input=test_case.input,
            actual_output=actual_output,
            purpose=self.purpose,
            bias_category=self.bias_category,
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt=prompt, schema=ReasonScore
            )
            self.evaluation_cost += cost
            return res.score, res.reason
        else:
            try:
                res: ReasonScore = await self.model.a_generate(
                    prompt=prompt, schema=ReasonScore
                )
                return res.score, res.reason
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["score"], data["reason"]

    def evaluate(self, test_case: RTTestCase) -> Tuple[Union[int, float], str]:

        if test_case.turns is not None:
            actual_output = format_turns(test_case.turns)
        else:
            actual_output = (
                "AI'S ACTUAL OUTPUT TO EVALUATE: \n"
                + test_case.actual_output
                + "\n"
            )

        prompt = BiasTemplate.generate_evaluation_results(
            input=test_case.input,
            actual_output=actual_output,
            purpose=self.purpose,
            bias_category=self.bias_category,
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt=prompt, schema=ReasonScore)
            self.evaluation_cost += cost
            return res.score, res.reason
        else:
            try:
                res: ReasonScore = self.model.generate(
                    prompt=prompt, schema=ReasonScore
                )
                return res.score, res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["score"], data["reason"]

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.score == 1
            except:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return f"Bias (Red Teaming)"
