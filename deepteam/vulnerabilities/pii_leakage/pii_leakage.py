from typing import List, Literal, Optional, Union, Dict
import asyncio

from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import initialize_model, trimAndLoadJson
from deepeval.utils import get_or_create_event_loop

from deepteam.utils import validate_model_callback_signature

from deepteam.vulnerabilities import BaseVulnerability
from deepteam.vulnerabilities.pii_leakage import PIILeakageType
from deepteam.vulnerabilities.utils import validate_vulnerability_types
from deepteam.metrics import PIIMetric, BaseRedTeamingMetric
from deepteam.attacks.multi_turn.types import CallbackType
from deepteam.test_case import RTTestCase
from deepteam.attacks.attack_simulator.schema import SyntheticDataList
from deepteam.risks import getRiskCategory
from .template import PIILeakageTemplate

PIILeakageLiteral = Literal[
    "api_and_database access",
    "direct_disclosure",
    "session_leak",
    "social_manipulation",
]


class PIILeakage(BaseVulnerability):
    def __init__(
        self,
        async_mode: bool = True,
        verbose_mode: bool = False,
        simulator_model: Optional[
            Union[str, DeepEvalBaseLLM]
        ] = "gpt-3.5-turbo-0125",
        evaluation_model: Optional[Union[str, DeepEvalBaseLLM]] = "gpt-4o",
        types: Optional[List[PIILeakageLiteral]] = [
            type.value for type in PIILeakageType
        ],
        purpose: Optional[str] = None,
    ):
        enum_types = validate_vulnerability_types(
            self.get_name(), types=types, allowed_type=PIILeakageType
        )
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self.simulator_model = simulator_model
        self.evaluation_model = evaluation_model
        self.purpose = purpose
        super().__init__(types=enum_types)

    def assess(
        self,
        model_callback: CallbackType,
        purpose: Optional[str] = None,
    ) -> Dict[PIILeakageType, List[RTTestCase]]:

        validate_model_callback_signature(
            model_callback=model_callback,
            async_mode=self.async_mode,
        )

        if self.async_mode:
            loop = get_or_create_event_loop()
            return loop.run_until_complete(
                self.a_assess(
                    model_callback=model_callback,
                    purpose=purpose,
                )
            )

        simulated_test_cases = self.simulate_attacks(purpose)

        results: Dict[PIILeakageType, List[RTTestCase]] = {}
        res: Dict[PIILeakageType, PIIMetric] = {}
        simulated_attacks: Dict[str, str] = {}

        for simulated_test_case in simulated_test_cases:
            vulnerability_type = simulated_test_case.vulnerability_type

            target_output = model_callback(simulated_test_case.input)
            red_teaming_test_case = RTTestCase(
                vulnerability=simulated_test_case.vulnerability,
                vulnerability_type=vulnerability_type,
                attackMethod=simulated_test_case.attack_method,
                riskCategory=getRiskCategory(vulnerability_type),
                input=simulated_test_case.input,
                actual_output=target_output,
            )

            metric = self._get_metric(vulnerability_type)
            metric.measure(red_teaming_test_case)

            red_teaming_test_case.score = metric.score
            red_teaming_test_case.reason = metric.reason

            res[vulnerability_type] = metric
            simulated_attacks[vulnerability_type.value] = (
                simulated_test_case.input
            )
            results.setdefault(vulnerability_type, []).append(
                red_teaming_test_case
            )

        self.res = res
        self.simulated_attacks = simulated_attacks

        return results

    async def a_assess(
        self,
        model_callback: CallbackType,
        purpose: Optional[str] = None,
    ) -> Dict[PIILeakageType, List[RTTestCase]]:

        validate_model_callback_signature(
            model_callback=model_callback,
            async_mode=self.async_mode,
        )

        simulated_test_cases = await self.a_simulate_attacks(purpose)

        results: Dict[PIILeakageType, List[RTTestCase]] = {}
        res: Dict[PIILeakageType, PIIMetric] = {}
        simulated_attacks: Dict[str, str] = {}

        async def process_attack(simulated_test_case: RTTestCase):
            vulnerability_type = simulated_test_case.vulnerability_type

            target_output = await model_callback(simulated_test_case.input)

            red_teaming_test_case = RTTestCase(
                vulnerability=simulated_test_case.vulnerability,
                vulnerability_type=vulnerability_type,
                attackMethod=simulated_test_case.attack_method,
                riskCategory=getRiskCategory(vulnerability_type),
                input=simulated_test_case.input,
                actual_output=target_output,
            )

            metric = self._get_metric(vulnerability_type)
            await metric.a_measure(red_teaming_test_case)

            red_teaming_test_case.score = metric.score
            red_teaming_test_case.reason = metric.reason

            return (
                vulnerability_type,
                red_teaming_test_case,
                metric,
                simulated_test_case.input,
            )

        all_tasks = [
            process_attack(simulated_test_case)
            for simulated_test_case in simulated_test_cases
            if simulated_test_case.vulnerability_type in self.types
        ]

        for task in asyncio.as_completed(all_tasks):
            vuln_type, test_case, metric, input_text = await task
            results.setdefault(vuln_type, []).append(test_case)
            res[vuln_type] = metric
            simulated_attacks[vuln_type.value] = input_text

        self.res = res
        self.simulated_attacks = simulated_attacks

        return results

    def simulate_attacks(
        self,
        purpose: Optional[str] = None,
        attacks_per_vulnerability_type: int = 1,
    ) -> List[RTTestCase]:

        self.simulator_model, self.using_native_model = initialize_model(
            self.simulator_model
        )

        self.purpose = purpose

        templates = dict()
        simulated_test_cases: List[RTTestCase] = []

        for type in self.types:
            templates[type] = templates.get(type, [])
            templates[type].append(
                PIILeakageTemplate.generate_baseline_attacks(
                    type, attacks_per_vulnerability_type, self.purpose
                )
            )

        for type in self.types:
            for prompt in templates[type]:
                if self.using_native_model:
                    res, _ = self.simulator_model.generate(
                        prompt, schema=SyntheticDataList
                    )
                    local_attacks = [item.input for item in res.data]
                else:
                    try:
                        res: SyntheticDataList = self.simulator_model.generate(
                            prompt, schema=SyntheticDataList
                        )
                        local_attacks = [item.input for item in res.data]
                    except TypeError:
                        res = self.simulator_model.generate(prompt)
                        data = trimAndLoadJson(res)
                        local_attacks = [item["input"] for item in data["data"]]

            simulated_test_cases.extend(
                [
                    RTTestCase(
                        vulnerability=self.get_name(),
                        vulnerability_type=type,
                        input=local_attack,
                    )
                    for local_attack in local_attacks
                ]
            )

        return simulated_test_cases

    async def a_simulate_attacks(
        self,
        purpose: Optional[str] = None,
        attacks_per_vulnerability_type: int = 1,
    ) -> List[RTTestCase]:

        self.simulator_model, self.using_native_model = initialize_model(
            self.simulator_model
        )

        self.purpose = purpose

        templates = dict()
        simulated_test_cases: List[RTTestCase] = []

        for type in self.types:
            templates[type] = templates.get(type, [])
            templates[type].append(
                PIILeakageTemplate.generate_baseline_attacks(
                    type, attacks_per_vulnerability_type, self.purpose
                )
            )

        for type in self.types:
            for prompt in templates[type]:
                if self.using_native_model:
                    res, _ = await self.simulator_model.a_generate(
                        prompt, schema=SyntheticDataList
                    )
                    local_attacks = [item.input for item in res.data]
                else:
                    try:
                        res: SyntheticDataList = (
                            await self.simulator_model.a_generate(
                                prompt, schema=SyntheticDataList
                            )
                        )
                        local_attacks = [item.input for item in res.data]
                    except TypeError:
                        res = await self.simulator_model.a_generate(prompt)
                        data = trimAndLoadJson(res)
                        local_attacks = [item["input"] for item in data["data"]]

            simulated_test_cases.extend(
                [
                    RTTestCase(
                        vulnerability=self.get_name(),
                        vulnerability_type=type,
                        input=local_attack,
                    )
                    for local_attack in local_attacks
                ]
            )

        return simulated_test_cases

    def _get_metric(
        self,
        type: PIILeakageType,
    ) -> BaseRedTeamingMetric:
        return PIIMetric(
            purpose=self.purpose,
            model=self.evaluation_model,
            async_mode=self.async_mode,
            verbose_mode=self.verbose_mode,
        )

    def is_vulnerable(self) -> bool:
        self.vulnerable = False
        try:
            for _, metric_data in self.res.items():
                if metric_data.score < 1:
                    self.vulnerable = True
        except:
            self.vulnerable = False
        return self.vulnerable

    def get_name(self) -> str:
        return "PII Leakage"
