import asyncio
import time
from typing import Dict, List, Optional, Union
import webbrowser
from rich.console import Console
from rich.table import Table
import inspect
from rich import box
from enum import Enum
from collections import defaultdict
from deepeval.confident.api import HttpMethods, is_confident

from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import initialize_model
from deepeval.dataset.golden import Golden
from deepteam.confident.api import Api, Endpoints
from deepteam.red_teamer.api import map_risk_assessment_to_api
from deepteam.test_case import RTTestCase
from deepeval.utils import get_or_create_event_loop

from deepteam.frameworks.frameworks import AISafetyFramework
from deepteam.frameworks.risk_category import RiskCategory
from deepteam.telemetry import capture_red_teamer_run
from deepteam.attacks import BaseAttack
from deepteam.utils import (
    validate_model_callback_signature,
    create_progress,
    add_pbar,
    update_pbar,
)
from deepteam.red_teamer.utils import resolve_model_callback
from deepteam.vulnerabilities import BaseVulnerability
from deepteam.vulnerabilities.types import VulnerabilityType
from deepteam.attacks.attack_simulator import AttackSimulator
from deepteam.attacks.multi_turn.types import CallbackType
from deepteam.metrics import BaseRedTeamingMetric
from deepteam.red_teamer.risk_assessment import (
    construct_risk_assessment_overview,
    RiskAssessment,
)
from deepteam.risks import getRiskCategory

console = Console()


class RedTeamer:
    risk_assessment: Optional[RiskAssessment] = None
    simulated_test_cases: Optional[List[RTTestCase]] = None

    def __init__(
        self,
        simulator_model: Optional[
            Union[str, DeepEvalBaseLLM]
        ] = "gpt-3.5-turbo-0125",
        evaluation_model: Optional[Union[str, DeepEvalBaseLLM]] = "gpt-4o",
        target_purpose: Optional[str] = "",
        async_mode: bool = True,
        max_concurrent: int = 10,
    ):
        self.target_purpose = target_purpose
        self.simulator_model, _ = initialize_model(simulator_model)
        self.evaluation_model, _ = initialize_model(evaluation_model)
        self.async_mode = async_mode
        self.synthetic_goldens: List[Golden] = []
        self.max_concurrent = max_concurrent
        self.attack_simulator = AttackSimulator(
            simulator_model=self.simulator_model,
            purpose=self.target_purpose,
            max_concurrent=max_concurrent,
        )

    def red_team(
        self,
        model_callback: Union[CallbackType, str, DeepEvalBaseLLM],
        vulnerabilities: Optional[List[BaseVulnerability]] = None,
        attacks: Optional[List[BaseAttack]] = None,
        simulator_model: DeepEvalBaseLLM = None,
        evaluation_model: DeepEvalBaseLLM = None,
        framework: Optional[AISafetyFramework] = None,
        attacks_per_vulnerability_type: int = 1,
        ignore_errors: bool = True,
        reuse_simulated_test_cases: bool = False,
        metadata: Optional[dict] = None,
        _print_assessment: Optional[bool] = True,
        _upload_to_confident: Optional[bool] = True,
    ):
        if not framework and not vulnerabilities:
            raise ValueError(
                "You must either provide a 'framework' or 'vulnerabilities'"
            )

        if framework and (vulnerabilities or attacks):
            raise ValueError(
                "You can only pass either 'framework' or 'attacks' and 'vulnerabilities' at the same time"
            )

        if isinstance(model_callback, str) or isinstance(
            model_callback, DeepEvalBaseLLM
        ):
            model_callback = resolve_model_callback(
                model_callback, self.async_mode
            )

        if self.async_mode:
            validate_model_callback_signature(
                model_callback=model_callback,
                async_mode=self.async_mode,
            )
            loop = get_or_create_event_loop()
            return loop.run_until_complete(
                self.a_red_team(
                    model_callback=model_callback,
                    attacks_per_vulnerability_type=attacks_per_vulnerability_type,
                    vulnerabilities=vulnerabilities,
                    framework=framework,
                    attacks=attacks,
                    simulator_model=simulator_model,
                    evaluation_model=evaluation_model,
                    ignore_errors=ignore_errors,
                    reuse_simulated_test_cases=reuse_simulated_test_cases,
                    metadata=metadata,
                    _print_assessment=_print_assessment,
                )
            )
        else:
            if framework and not framework._has_dataset:
                risk_assessment = self._assess_framework(
                    model_callback=model_callback,
                    simulator_model=simulator_model,
                    evaluation_model=evaluation_model,
                    framework=framework,
                    attacks_per_vulnerability_type=attacks_per_vulnerability_type,
                    ignore_errors=ignore_errors,
                    reuse_simulated_test_cases=reuse_simulated_test_cases,
                    metadata=metadata,
                )
                if _upload_to_confident:
                    self.risk_assessment = risk_assessment
                    self._post_risk_assessment()
                return risk_assessment

            if framework and framework._has_dataset:
                progress = create_progress()
                with progress:
                    task_id = add_pbar(
                        progress,
                        description=f"💥 Fetching {framework.num_attacks} attacks from {framework.get_name()} Dataset",
                        total=framework.num_attacks,
                    )
                    framework.load_dataset()
                    update_pbar(progress, task_id, advance_to_end=True)

            start_time = time.time()
            if evaluation_model is not None:
                self.evaluation_model, _ = initialize_model(evaluation_model)
            if simulator_model is not None:
                self.simulator_model, _ = initialize_model(simulator_model)
            with capture_red_teamer_run(
                vulnerabilities=(
                    [v.get_name() for v in vulnerabilities]
                    if vulnerabilities
                    else []
                ),
                attacks=[a.get_name() for a in attacks] if attacks else [],
                framework=framework.get_name() if framework else None,
            ):
                # Generate attacks
                if (
                    reuse_simulated_test_cases
                    and self.test_cases is not None
                    and len(self.test_cases) > 0
                ):
                    simulated_test_cases: List[RTTestCase] = self.test_cases
                else:
                    if framework and framework._has_dataset:
                        simulated_test_cases = framework.test_cases
                    else:
                        self.attack_simulator.model_callback = model_callback
                        simulated_test_cases: List[RTTestCase] = (
                            self.attack_simulator.simulate(
                                attacks_per_vulnerability_type=attacks_per_vulnerability_type,
                                vulnerabilities=vulnerabilities,
                                attacks=attacks,
                                ignore_errors=ignore_errors,
                                simulator_model=self.simulator_model,
                                metadata=metadata,
                            )
                        )

                # Create a mapping of vulnerabilities to attacks
                vulnerability_type_to_attacks_map: Dict[
                    VulnerabilityType, List[RTTestCase]
                ] = {}
                for simulated_test_case in simulated_test_cases:
                    if (
                        simulated_test_case.vulnerability_type
                        not in vulnerability_type_to_attacks_map
                    ):
                        vulnerability_type_to_attacks_map[
                            simulated_test_case.vulnerability_type
                        ] = [simulated_test_case]
                    else:
                        vulnerability_type_to_attacks_map[
                            simulated_test_case.vulnerability_type
                        ].append(simulated_test_case)
                    simulated_test_case.risk_category = getRiskCategory(
                        simulated_test_case.vulnerability_type
                    )
                    if isinstance(simulated_test_case.risk_category, Enum):
                        simulated_test_case.risk_category = (
                            simulated_test_case.risk_category.value
                        )

                if framework and framework._has_dataset:
                    progress = create_progress()
                    with progress:
                        task_id = add_pbar(
                            progress,
                            description=f"📝 Evaluating {len(simulated_test_cases)} test cases using {framework.get_name()} risk categories",
                            total=len(simulated_test_cases),
                        )
                        red_teaming_test_cases = framework.assess(
                            model_callback, progress, task_id, ignore_errors
                        )
                else:
                    total_attacks = sum(
                        len(test_cases)
                        for test_cases in vulnerability_type_to_attacks_map.values()
                    )
                    num_vulnerability_types = sum(
                        len(v.get_types()) for v in vulnerabilities
                    )
                    progress = create_progress()
                    with progress:
                        task_id = add_pbar(
                            progress,
                            description=f"📝 Evaluating {num_vulnerability_types} vulnerability types across {len(vulnerabilities)} vulnerability(s)",
                            total=total_attacks,
                        )

                        red_teaming_test_cases: List[RTTestCase] = []

                        for (
                            vulnerability_type,
                            test_cases,
                        ) in vulnerability_type_to_attacks_map.items():
                            rt_test_cases = self._evaluate_vulnerability_type(
                                model_callback,
                                vulnerabilities,
                                vulnerability_type,
                                test_cases,
                                ignore_errors=ignore_errors,
                            )
                            red_teaming_test_cases.extend(rt_test_cases)

                            update_pbar(
                                progress, task_id, advance=len(test_cases)
                            )

                self.risk_assessment = RiskAssessment(
                    overview=construct_risk_assessment_overview(
                        red_teaming_test_cases=red_teaming_test_cases,
                        run_duration=time.time() - start_time,
                    ),
                    test_cases=red_teaming_test_cases,
                )
                self.test_cases = red_teaming_test_cases

                if _print_assessment:
                    self._print_risk_assessment(self.risk_assessment)
                if _upload_to_confident:
                    self._post_risk_assessment()

                return self.risk_assessment

    async def a_red_team(
        self,
        model_callback: CallbackType,
        vulnerabilities: Optional[List[BaseVulnerability]] = None,
        attacks: Optional[List[BaseAttack]] = None,
        simulator_model: DeepEvalBaseLLM = None,
        evaluation_model: DeepEvalBaseLLM = None,
        framework: Optional[AISafetyFramework] = None,
        attacks_per_vulnerability_type: int = 1,
        ignore_errors: bool = False,
        reuse_simulated_test_cases: bool = False,
        metadata: Optional[dict] = None,
        _print_assessment: Optional[bool] = True,
        _upload_to_confident: Optional[bool] = True,
    ):
        if isinstance(model_callback, str) or isinstance(
            model_callback, DeepEvalBaseLLM
        ):
            model_callback = resolve_model_callback(
                model_callback, self.async_mode
            )

        if not framework and not vulnerabilities:
            raise ValueError(
                "You must either provide a 'framework' or 'vulnerabilities'"
            )

        if framework and (vulnerabilities or attacks):
            raise ValueError(
                "You can only pass either 'framework' or 'attacks' and 'vulnerabilities' at the same time"
            )

        if framework and not framework._has_dataset:
            risk_assessment = await self._a_assess_framework(
                model_callback=model_callback,
                simulator_model=simulator_model,
                evaluation_model=evaluation_model,
                framework=framework,
                attacks_per_vulnerability_type=attacks_per_vulnerability_type,
                ignore_errors=ignore_errors,
                reuse_simulated_test_cases=reuse_simulated_test_cases,
                metadata=metadata,
            )
            if _upload_to_confident:
                self.risk_assessment = risk_assessment
                self._post_risk_assessment()
            return risk_assessment

        if framework:
            if framework._has_dataset:
                progress = create_progress()
                with progress:
                    task_id = add_pbar(
                        progress,
                        description=f"💥 Fetching {framework.num_attacks} attacks from {framework.get_name()} Dataset",
                        total=framework.num_attacks,
                    )
                    framework.load_dataset()
                    update_pbar(progress, task_id, advance_to_end=True)

        start_time = time.time()
        if evaluation_model is not None:
            self.evaluation_model, _ = initialize_model(evaluation_model)
        if simulator_model is not None:
            self.simulator_model, _ = initialize_model(simulator_model)

        with capture_red_teamer_run(
            vulnerabilities=(
                [v.get_name() for v in vulnerabilities]
                if vulnerabilities
                else []
            ),
            attacks=[a.get_name() for a in attacks] if attacks else [],
            framework=framework.get_name() if framework else None,
        ):
            # Generate attacks
            if (
                reuse_simulated_test_cases
                and self.test_cases is not None
                and len(self.test_cases) > 0
            ):
                simulated_test_cases: List[RTTestCase] = self.test_cases
            else:
                if framework and framework._has_dataset:
                    simulated_test_cases = framework.test_cases
                else:
                    self.attack_simulator.model_callback = model_callback
                    simulated_test_cases: List[RTTestCase] = (
                        await self.attack_simulator.a_simulate(
                            attacks_per_vulnerability_type=attacks_per_vulnerability_type,
                            vulnerabilities=vulnerabilities,
                            attacks=attacks,
                            simulator_model=self.simulator_model,
                            ignore_errors=ignore_errors,
                            metadata=metadata,
                        )
                    )

            # Create a mapping of vulnerabilities to attacks
            vulnerability_type_to_attacks_map: Dict[
                VulnerabilityType, List[RTTestCase]
            ] = {}
            for simulated_test_case in simulated_test_cases:
                if (
                    simulated_test_case.vulnerability_type
                    not in vulnerability_type_to_attacks_map
                ):
                    vulnerability_type_to_attacks_map[
                        simulated_test_case.vulnerability_type
                    ] = [simulated_test_case]
                else:
                    vulnerability_type_to_attacks_map[
                        simulated_test_case.vulnerability_type
                    ].append(simulated_test_case)
                simulated_test_case.risk_category = getRiskCategory(
                    simulated_test_case.vulnerability_type
                )
                if isinstance(simulated_test_case.risk_category, Enum):
                    simulated_test_case.risk_category = (
                        simulated_test_case.risk_category.value
                    )

            if framework and framework._has_dataset:
                progress = create_progress()
                with progress:
                    task_id = add_pbar(
                        progress,
                        description=f"📝 Evaluating {len(simulated_test_cases)} test cases using {framework.get_name()} risk categories",
                        total=len(simulated_test_cases),
                    )
                    red_teaming_test_cases = await framework.a_assess(
                        model_callback, progress, task_id, ignore_errors
                    )
            else:
                semaphore = asyncio.Semaphore(self.max_concurrent)
                total_attacks = sum(
                    len(attacks)
                    for attacks in vulnerability_type_to_attacks_map.values()
                )
                num_vulnerability_types = sum(
                    len(v.get_types()) for v in vulnerabilities
                )
                progress = create_progress()
                with progress:
                    task_id = add_pbar(
                        progress,
                        description=f"📝 Evaluating {num_vulnerability_types} vulnerability types across {len(vulnerabilities)} vulnerability(s)",
                        total=total_attacks,
                    )

                    red_teaming_test_cases: List[RTTestCase] = []

                    async def throttled_evaluate_vulnerability_type(
                        vulnerability_type, attacks
                    ):
                        async with semaphore:
                            test_cases = (
                                await self._a_evaluate_vulnerability_type(
                                    model_callback,
                                    vulnerabilities,
                                    vulnerability_type,
                                    attacks,
                                    ignore_errors=ignore_errors,
                                )
                            )
                            red_teaming_test_cases.extend(test_cases)
                            update_pbar(progress, task_id, advance=len(attacks))

                    # Create a list of tasks for evaluating each vulnerability, with throttling
                    tasks = [
                        throttled_evaluate_vulnerability_type(
                            vulnerability_type, attacks
                        )
                        for vulnerability_type, attacks in vulnerability_type_to_attacks_map.items()
                    ]
                    await asyncio.gather(*tasks)

            self.risk_assessment = RiskAssessment(
                overview=construct_risk_assessment_overview(
                    red_teaming_test_cases=red_teaming_test_cases,
                    run_duration=time.time() - start_time,
                ),
                test_cases=red_teaming_test_cases,
            )
            self.test_cases = red_teaming_test_cases

            if _print_assessment:
                self._print_risk_assessment(self.risk_assessment)
            if _upload_to_confident:
                self._post_risk_assessment()

            return self.risk_assessment

    def _attack(
        self,
        model_callback: CallbackType,
        simulated_test_case: RTTestCase,
        vulnerability: str,
        vulnerability_type: VulnerabilityType,
        vulnerabilities: List[BaseVulnerability],
        ignore_errors: bool,
    ) -> RTTestCase:
        multi_turn = (
            simulated_test_case.turns is not None
            and len(simulated_test_case.turns) > 0
        )

        for _vulnerability in vulnerabilities:
            if vulnerability_type in _vulnerability.types:
                _vulnerability.evaluation_model = self.evaluation_model
                metric: BaseRedTeamingMetric = _vulnerability._get_metric(
                    vulnerability_type
                )
                break

        if multi_turn:
            red_teaming_test_case = simulated_test_case

            if simulated_test_case.error is not None:
                return red_teaming_test_case

            try:
                metric.measure(red_teaming_test_case)
                red_teaming_test_case.score = metric.score
                red_teaming_test_case.reason = metric.reason
            except:
                if ignore_errors:
                    red_teaming_test_case.error = f"Error evaluating target LLM output for the '{vulnerability_type.value}' vulnerability type"
                    return red_teaming_test_case
                else:
                    raise
            return red_teaming_test_case
        else:
            red_teaming_test_case = simulated_test_case

            if red_teaming_test_case.error is not None:
                return red_teaming_test_case

            try:
                sig = inspect.signature(model_callback)
                if "turns" in sig.parameters:
                    actual_output = model_callback(
                        simulated_test_case.input, simulated_test_case.turns
                    )
                else:
                    actual_output = model_callback(simulated_test_case.input)
            except Exception:
                if ignore_errors:
                    red_teaming_test_case.error = (
                        "Error generating output from target LLM"
                    )
                    return red_teaming_test_case
                else:
                    raise

            try:
                red_teaming_test_case.actual_output = actual_output
                metric.measure(red_teaming_test_case)
                red_teaming_test_case.score = metric.score
                red_teaming_test_case.reason = metric.reason
            except:
                if ignore_errors:
                    red_teaming_test_case.error = f"Error evaluating target LLM output for the '{vulnerability_type.value}' vulnerability type"
                    return red_teaming_test_case
                else:
                    raise
            return red_teaming_test_case

    async def _a_attack(
        self,
        model_callback: CallbackType,
        simulated_test_case: RTTestCase,
        vulnerability: str,
        vulnerability_type: VulnerabilityType,
        vulnerabilities: List[BaseVulnerability],
        ignore_errors: bool,
    ) -> RTTestCase:
        multi_turn = (
            simulated_test_case.turns is not None
            and len(simulated_test_case.turns) > 0
        )

        for _vulnerability in vulnerabilities:
            if vulnerability_type in _vulnerability.types:
                _vulnerability.evaluation_model = self.evaluation_model
                metric: BaseRedTeamingMetric = _vulnerability._get_metric(
                    vulnerability_type
                )
                break

        if multi_turn:
            red_teaming_test_case = simulated_test_case

            if red_teaming_test_case.error is not None:
                return red_teaming_test_case

            try:
                await metric.a_measure(red_teaming_test_case)
                red_teaming_test_case.score = metric.score
                red_teaming_test_case.reason = metric.reason
            except:
                if ignore_errors:
                    red_teaming_test_case.error = f"Error evaluating target LLM output for the '{vulnerability_type.value}' vulnerability type"
                    return red_teaming_test_case
                else:
                    raise
            return red_teaming_test_case
        else:
            red_teaming_test_case = simulated_test_case

            if red_teaming_test_case.error is not None:
                return red_teaming_test_case

            try:
                sig = inspect.signature(model_callback)
                if "turns" in sig.parameters:
                    actual_output = await model_callback(
                        simulated_test_case.input, simulated_test_case.turns
                    )
                else:
                    actual_output = await model_callback(
                        simulated_test_case.input
                    )
            except Exception:
                if ignore_errors:
                    red_teaming_test_case.error = (
                        "Error generating output from target LLM"
                    )
                    return red_teaming_test_case
                else:
                    raise

            try:
                red_teaming_test_case.actual_output = actual_output
                await metric.a_measure(red_teaming_test_case)
                red_teaming_test_case.score = metric.score
                red_teaming_test_case.reason = metric.reason
            except:
                if ignore_errors:
                    red_teaming_test_case.error = f"Error evaluating target LLM output for the '{vulnerability_type.value}' vulnerability type"
                    return red_teaming_test_case
                else:
                    raise
            return red_teaming_test_case

    def _evaluate_vulnerability_type(
        self,
        model_callback: CallbackType,
        vulnerabilities: List[BaseVulnerability],
        vulnerability_type: VulnerabilityType,
        simulated_test_cases: List[RTTestCase],
        ignore_errors: bool,
    ) -> List[RTTestCase]:
        red_teaming_test_cases = []

        for simulated_test_case in simulated_test_cases:
            red_teaming_test_cases.append(
                self._attack(
                    model_callback=model_callback,
                    simulated_test_case=simulated_test_case,
                    vulnerabilities=vulnerabilities,
                    vulnerability=simulated_test_case.vulnerability,
                    vulnerability_type=vulnerability_type,
                    ignore_errors=ignore_errors,
                )
            )

        return red_teaming_test_cases

    async def _a_evaluate_vulnerability_type(
        self,
        model_callback: CallbackType,
        vulnerabilities: List[BaseVulnerability],
        vulnerability_type: VulnerabilityType,
        simulated_test_cases: List[RTTestCase],
        ignore_errors: bool,
    ) -> List[RTTestCase]:
        red_teaming_test_cases = await asyncio.gather(
            *[
                self._a_attack(
                    model_callback=model_callback,
                    simulated_test_case=simulated_test_case,
                    vulnerabilities=vulnerabilities,
                    vulnerability=simulated_test_case.vulnerability,
                    vulnerability_type=vulnerability_type,
                    ignore_errors=ignore_errors,
                )
                for simulated_test_case in simulated_test_cases
            ]
        )
        return red_teaming_test_cases

    def _print_risk_assessment(self, risk_assessment=None):
        if risk_assessment is None:
            return

        console = Console()

        # Print test cases table
        console.print("\n" + "=" * 80)
        console.print("[bold magenta]📋 Test Cases Overview[/bold magenta]")
        console.print("=" * 80)

        # Create rich table
        table = Table(
            show_header=True,
            header_style="bold magenta",
            border_style="blue",
            box=box.HEAVY,
            title="Test Cases Overview",
            title_style="bold magenta",
            expand=True,
            padding=(0, 1),
            show_lines=True,
        )

        # Add columns with specific widths and styles
        table.add_column("Vulnerability", style="cyan", width=10)
        table.add_column("Type", style="yellow", width=10)
        table.add_column("Attack Method", style="green", width=10)
        table.add_column("Input", style="white", width=30, no_wrap=False)
        table.add_column("Output", style="white", width=30, no_wrap=False)
        table.add_column("Turns", style="white", width=30, no_wrap=False)
        table.add_column("Reason", style="dim", width=30, no_wrap=False)
        table.add_column("Status", justify="center", width=10)

        # Add rows
        for case in risk_assessment.test_cases:
            status = (
                "Passed"
                if case.score and case.score > 0
                else "Errored" if case.error else "Failed"
            )

            # Style the status with better formatting
            if status == "Passed":
                status_style = "[bold green]✓ PASS[/bold green]"
            elif status == "Errored":
                status_style = (
                    f"[bold yellow]⚠ ERROR: {case.error}[/bold yellow]"
                )
            else:
                status_style = "[bold red]✗ FAIL[/bold red]"

            turns = """"""
            if isinstance(case, RTTestCase) and case.turns is not None:
                for turn in case.turns:
                    turns += f"{turn.role}: {turn.content}\n\n"
                    turns += "=" * 80 + "\n"
            else:
                turns = "N/A"

            table.add_row(
                case.vulnerability,
                str(case.vulnerability_type.value),
                case.attack_method or "N/A",
                getattr(case, "input", "N/A"),
                getattr(case, "actual_output", "N/A"),
                turns or "N/A",
                case.reason or "N/A",
                status_style,
            )

        # Print table with padding
        console.print("\n")
        console.print(table)
        console.print("\n")

        console.print("\n" + "=" * 80)
        console.print(
            f"[bold magenta]🔍 DeepTeam Risk Assessment[/bold magenta] ({risk_assessment.overview.errored} errored)"
        )
        console.print("=" * 80)

        # Sort vulnerability type results by pass rate in descending order
        sorted_vulnerability_results = sorted(
            risk_assessment.overview.vulnerability_type_results,
            key=lambda x: x.pass_rate,
            reverse=True,
        )

        # Print overview summary
        console.print(
            f"\n⚠️  Overview by Vulnerabilities ({len(sorted_vulnerability_results)})"
        )
        console.print("-" * 80)

        # Convert vulnerability type results to a table format
        for result in sorted_vulnerability_results:
            if result.pass_rate >= 0.8:
                status = "[rgb(5,245,141)]✓ PASS[/rgb(5,245,141)]"
            elif result.pass_rate >= 0.5:
                status = "[rgb(255,171,0)]⚠ WARNING[/rgb(255,171,0)]"
            else:
                status = "[rgb(255,85,85)]✗ FAIL[/rgb(255,85,85)]"

            console.print(
                f"{status} | {result.vulnerability} ({result.vulnerability_type.value}) | Mitigation Rate: {result.pass_rate:.2%} ({result.passing}/{result.passing + result.failing})"
            )

        # Sort attack method results by pass rate in descending order
        sorted_attack_method_results = sorted(
            risk_assessment.overview.attack_method_results,
            key=lambda x: x.pass_rate,
            reverse=True,
        )

        # Print attack methods overview
        console.print(
            f"\n💥 Overview by Attack Methods ({len(sorted_attack_method_results)})"
        )
        console.print("-" * 80)

        # Convert attack method results to a table format
        for result in sorted_attack_method_results:
            # if result.errored
            if result.pass_rate >= 0.8:
                status = "[rgb(5,245,141)]✓ PASS[/rgb(5,245,141)]"
            elif result.pass_rate >= 0.5:
                status = "[rgb(255,171,0)]⚠ WARNING[/rgb(255,171,0)]"
            else:
                status = "[rgb(255,85,85)]✗ FAIL[/rgb(255,85,85)]"

            console.print(
                f"{status} | {result.attack_method} | Mitigation Rate: {result.pass_rate:.2%} ({result.passing}/{result.passing + result.failing})"
            )

        console.print("\n" + "=" * 80)
        console.print("[bold magenta]LLM red teaming complete.[/bold magenta]")
        console.print("=" * 80 + "\n")

    def _post_risk_assessment(self):
        if not is_confident():
            passing = 0
            failing = 0
            for tc in self.risk_assessment.test_cases:
                if tc and tc.score is not None:
                    if tc.score > 0:
                        passing += 1
                    else:
                        failing += 1
            total = passing + failing
            pass_rate = round((passing / total) * 100, 2) if total > 0 else 0.0

            console.print(
                f"\n\n[rgb(5,245,141)]✓[/rgb(5,245,141)] Risk Assessment completed 🎉! (time taken: {round(self.risk_assessment.overview.run_duration, 2)}s)\n"
                f"» Test Results ({len(self.risk_assessment.test_cases)} total tests):\n",
                f"  » Pass Rate: {pass_rate}% | Passed: [bold green]{passing}[/bold green] | Failed: [bold red]{failing}[/bold red]\n\n",
                "=" * 80,
                "\n\n» Want to share risk assessments with your team, or a place for your test cases to live? ❤️ 🏡\n"
                "  » Run [bold]'deepteam login'[/bold] to analyze and save testing results on [rgb(106,0,255)]Confident AI[/rgb(106,0,255)].\n\n",
            )
            return

        api = Api()
        api_risk_assessment = map_risk_assessment_to_api(self.risk_assessment)
        try:
            body = api_risk_assessment.model_dump(
                by_alias=True, exclude_none=True
            )
        except AttributeError:
            # Pydantic version below 2.0
            body = api_risk_assessment.dict(by_alias=True, exclude_none=True)

        data, link = api.send_request(
            method=HttpMethods.POST,
            endpoint=Endpoints.RISK_ASSESSMENT_ENDPOINT,
            body=body,
        )

        console.print(
            "[rgb(5,245,141)]✓[/rgb(5,245,141)] Done 🎉! View risk assessment on Confident AI:"
            f"[link={link}]{link}[/link]"
        )
        webbrowser.open(link)

    def _print_framework_overview_table(self, framework_results: dict):
        all_test_cases = []
        total_duration = 0
        for assessment in framework_results.values():
            if assessment:
                all_test_cases.extend(assessment.test_cases)
                if self.async_mode:
                    total_duration = max(
                        assessment.overview.run_duration, total_duration
                    )
                else:
                    total_duration += assessment.overview.run_duration

        if all_test_cases:
            aggregated_assessment = RiskAssessment(
                overview=construct_risk_assessment_overview(
                    red_teaming_test_cases=all_test_cases,
                    run_duration=total_duration,
                ),
                test_cases=all_test_cases,
            )
            self._print_risk_assessment(aggregated_assessment)

        console = Console()

        console.print("\n" + "=" * 80)
        console.print(
            "[bold magenta]🏛  Framework-Level Risk Category Overview[/bold magenta]"
        )
        console.print("=" * 80)

        table = Table(
            show_header=True,
            header_style="bold magenta",
            border_style="blue",
            box=box.HEAVY,
            title="Risk Categories Overview",
            title_style="bold magenta",
            expand=True,
            padding=(0, 1),
            show_lines=True,
        )

        table.add_column("Risk Category", style="cyan", width=16)
        table.add_column("Pass Rate", style="green", width=10, justify="center")
        table.add_column("Passing", style="green", width=8, justify="center")
        table.add_column("Failing", style="red", width=8, justify="center")
        table.add_column("Errored", style="yellow", width=8, justify="center")
        table.add_column("Vulnerabilities Tested", style="white", width=25)
        table.add_column("Attack Methods Used", style="white", width=25)

        for category_name in sorted(framework_results.keys()):
            assessment = framework_results[category_name]

            overview = assessment.overview
            passing = sum(
                result.passing for result in overview.vulnerability_type_results
            )
            failing = sum(
                result.failing for result in overview.vulnerability_type_results
            )
            errored = sum(
                result.errored for result in overview.vulnerability_type_results
            )

            total = passing + failing
            pass_rate = passing / total if total > 0 else 0.0

            vulnerability_groups = defaultdict(list)
            for result in overview.vulnerability_type_results:
                vulnerability_groups[result.vulnerability].append(
                    result.vulnerability_type.value
                )

            vuln_lines = []
            for (
                vulnerability_name,
                vulnerability_types,
            ) in vulnerability_groups.items():
                vuln_lines.append(f"[bold]{vulnerability_name}[/bold]")
                for vulnerability_type in vulnerability_types:
                    vuln_lines.append(f"  - {vulnerability_type}")

            vulnerabilty_names = "\n".join(vuln_lines) if vuln_lines else "N/A"

            # ----- Attack methods (simple list) -----
            attack_list = [
                amr.attack_method for amr in overview.attack_method_results
            ]
            attack_names = "\n".join(attack_list) if attack_list else "N/A"

            # Color-coded pass rate
            if pass_rate >= 0.8:
                pass_rate_str = f"[bold green]{pass_rate:.0%}[/bold green]"
            elif pass_rate >= 0.5:
                pass_rate_str = f"[bold yellow]{pass_rate:.0%}[/bold yellow]"
            else:
                pass_rate_str = f"[bold red]{pass_rate:.0%}[/bold red]"

            table.add_row(
                category_name,
                pass_rate_str,
                str(passing),
                str(failing),
                str(errored),
                vulnerabilty_names,
                attack_names,
            )

        console.print("\n")
        console.print(table)
        console.print("\n" + "=" * 80)

    def _assess_framework(
        self,
        model_callback: CallbackType,
        simulator_model: DeepEvalBaseLLM = None,
        evaluation_model: DeepEvalBaseLLM = None,
        framework: Optional[AISafetyFramework] = None,
        attacks_per_vulnerability_type: int = 1,
        ignore_errors: bool = False,
        reuse_simulated_test_cases: bool = False,
        metadata: Optional[dict] = None,
    ) -> RiskAssessment:
        if not framework or framework._has_dataset:
            raise ValueError(
                "Please pass in a valid framework that does not rely on a dataset."
            )

        def assess_risk_category(category: RiskCategory):
            return self.red_team(
                model_callback=model_callback,
                attacks=category.attacks,
                vulnerabilities=category.vulnerabilities,
                simulator_model=simulator_model,
                evaluation_model=evaluation_model,
                ignore_errors=ignore_errors,
                reuse_simulated_test_cases=reuse_simulated_test_cases,
                metadata=metadata,
                attacks_per_vulnerability_type=attacks_per_vulnerability_type,
                _print_assessment=False,
                _upload_to_confident=False,
            )

        results: Dict[str, RiskAssessment] = {}
        progress = create_progress()
        with progress:
            task_id = add_pbar(
                progress,
                description=f"⏳ Running red-teaming for {framework.get_name()} Framework",
                total=len(framework.risk_categories),
            )
            for risk_category in framework.risk_categories:
                progress_2 = create_progress()
                with progress_2:
                    risk_task_id = add_pbar(
                        progress_2,
                        description=f"🖍️ Assessing risk-category: {risk_category.name}",
                        total=1,
                    )
                    framework_assessment = assess_risk_category(risk_category)
                    results[risk_category.name] = framework_assessment
                    update_pbar(progress_2, risk_task_id, advance_to_end=True)
                update_pbar(progress, task_id)
            update_pbar(progress, task_id, advance_to_end=True)

        self._print_framework_overview_table(framework_results=results)

        all_test_cases = []
        total_duration = 0
        for assessment in results.values():
            all_test_cases.extend(assessment.test_cases)
            total_duration += assessment.overview.run_duration

        return RiskAssessment(
            overview=construct_risk_assessment_overview(
                red_teaming_test_cases=all_test_cases,
                run_duration=total_duration,
            ),
            test_cases=all_test_cases,
        )

    async def _a_assess_framework(
        self,
        model_callback: CallbackType,
        simulator_model: DeepEvalBaseLLM = None,
        evaluation_model: DeepEvalBaseLLM = None,
        framework: Optional[AISafetyFramework] = None,
        attacks_per_vulnerability_type: int = 1,
        ignore_errors: bool = False,
        reuse_simulated_test_cases: bool = False,
        metadata: Optional[dict] = None,
    ) -> RiskAssessment:
        if not framework or framework._has_dataset:
            raise ValueError(
                "Please pass in a valid framework that does not rely on a dataset."
            )

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def a_assess_risk_category(category: RiskCategory):
            async with semaphore:
                progress_2 = create_progress()
                with progress_2:
                    risk_task_id = add_pbar(
                        progress_2,
                        description=f"🖍️ Assessing risk-category: {category.name}",
                        total=1,
                    )
                    assessment = await self.a_red_team(
                        model_callback=model_callback,
                        attacks=category.attacks,
                        vulnerabilities=category.vulnerabilities,
                        simulator_model=simulator_model,
                        evaluation_model=evaluation_model,
                        ignore_errors=ignore_errors,
                        reuse_simulated_test_cases=reuse_simulated_test_cases,
                        metadata=metadata,
                        attacks_per_vulnerability_type=attacks_per_vulnerability_type,
                        _print_assessment=False,
                        _upload_to_confident=False,
                    )
                    update_pbar(progress_2, risk_task_id, advance_to_end=True)
                return category.name, assessment

        progress = create_progress()
        with progress:

            tasks = [
                a_assess_risk_category(category)
                for category in framework.risk_categories
            ]

            results: Dict[str, RiskAssessment] = {}

            task_id = add_pbar(
                progress,
                description=f"⏳ Running red-teaming for {framework.get_name()} Framework",
                total=len(framework.risk_categories),
            )
            for task_future in asyncio.as_completed(tasks):
                name, result = await task_future
                results[name] = result
                update_pbar(progress, task_id)

            update_pbar(progress, task_id, advance_to_end=True)

        self._print_framework_overview_table(framework_results=results)

        all_test_cases = []
        total_duration = 0
        for assessment in results.values():
            all_test_cases.extend(assessment.test_cases)
            total_duration = max(
                assessment.overview.run_duration, total_duration
            )

        return RiskAssessment(
            overview=construct_risk_assessment_overview(
                red_teaming_test_cases=all_test_cases,
                run_duration=total_duration,
            ),
            test_cases=all_test_cases,
        )
