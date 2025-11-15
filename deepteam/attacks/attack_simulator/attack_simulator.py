import random
import asyncio
from pydantic import BaseModel
from typing import List, Optional, Union
import inspect

from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import initialize_model
from deepeval.metrics.utils import initialize_model

from deepteam.attacks import BaseAttack
from deepteam.utils import create_progress, add_pbar, update_pbar
from deepteam.vulnerabilities import BaseVulnerability
from deepteam.attacks.multi_turn.types import CallbackType
from deepteam.errors import ModelRefusalError
from deepteam.test_case import RTTestCase


class BaselineAttack:
    def get_name(self):
        return "Baseline Attack"

    async def a_enhance(self, attack, *args, **kwargs):
        return attack


class AttackSimulator:
    model_callback: Union[CallbackType, None] = None

    def __init__(
        self,
        purpose: str,
        max_concurrent: int,
        simulator_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
    ):
        # Initialize models and async mode
        self.purpose = purpose
        self.simulator_model, self.using_native_model = initialize_model(
            simulator_model
        )
        # Define list of attacks and unaligned vulnerabilities
        self.test_cases: List[RTTestCase] = []
        self.max_concurrent = max_concurrent

    ##################################################
    ### Generating Attacks ###########################
    ##################################################

    def simulate(
        self,
        attacks_per_vulnerability_type: int,
        vulnerabilities: List[BaseVulnerability],
        ignore_errors: bool,
        attacks: Optional[List[BaseAttack]] = None,
        simulator_model: DeepEvalBaseLLM = None,
        metadata: Optional[dict] = None,
    ) -> List[RTTestCase]:
        # Simulate unenhanced attacks for each vulnerability
        test_cases: List[RTTestCase] = []
        num_vulnerability_types = sum(
            len(v.get_types()) for v in vulnerabilities
        )

        progress = create_progress()
        with progress:
            task_id = add_pbar(
                progress,
                description=f"💥 Generating {num_vulnerability_types * attacks_per_vulnerability_type} attacks (for {num_vulnerability_types} vulnerability types across {len(vulnerabilities)} vulnerability(s))",
                total=len(vulnerabilities),
            )
            for vulnerability in vulnerabilities:
                if simulator_model is not None:
                    vulnerability.simulator_model = simulator_model
                test_cases.extend(
                    self.simulate_baseline_attacks(
                        attacks_per_vulnerability_type=attacks_per_vulnerability_type,
                        vulnerability=vulnerability,
                        ignore_errors=ignore_errors,
                        metadata=metadata,
                    )
                )
                update_pbar(progress, task_id)

            if attacks is not None:
                # Enhance attacks by sampling from the provided distribution
                attack_weights = [attack.weight for attack in attacks]

                task_id_simulation = add_pbar(
                    progress,
                    description=f"✨ Simulating {num_vulnerability_types * attacks_per_vulnerability_type} attacks (using {len(attacks)} method(s))",
                    total=len(test_cases),
                )

                for test_case in test_cases:
                    # Randomly sample an enhancement based on the distribution
                    sampled_attack = random.choices(
                        attacks, weights=attack_weights, k=1
                    )[0]
                    self.enhance_attack(
                        attack=sampled_attack,
                        test_case=test_case,
                        ignore_errors=ignore_errors,
                    )
                    update_pbar(progress, task_id_simulation)

        self.test_cases.extend(test_cases)
        return test_cases

    async def a_simulate(
        self,
        attacks_per_vulnerability_type: int,
        vulnerabilities: List[BaseVulnerability],
        ignore_errors: bool,
        metadata: Optional[dict] = None,
        simulator_model: DeepEvalBaseLLM = None,
        attacks: Optional[List[BaseAttack]] = None,
    ) -> List[RTTestCase]:
        # Create a semaphore to control the number of concurrent tasks
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # Simulate unenhanced attacks for each vulnerability
        test_cases: List[RTTestCase] = []
        num_vulnerability_types = sum(
            len(v.get_types()) for v in vulnerabilities
        )
        progress = create_progress()
        with progress:

            task_id_vuln = add_pbar(
                progress,
                description=f"💥 Generating {num_vulnerability_types * attacks_per_vulnerability_type} attacks (for {num_vulnerability_types} vulnerability types across {len(vulnerabilities)} vulnerability(s))",
                total=len(vulnerabilities),
            )

            async def throttled_simulate_baseline_attack(vulnerability):
                if simulator_model is not None:
                    vulnerability.simulator_model = simulator_model
                async with semaphore:  # Throttling applied here
                    result = await self.a_simulate_baseline_attacks(
                        attacks_per_vulnerability_type=attacks_per_vulnerability_type,
                        vulnerability=vulnerability,
                        ignore_errors=ignore_errors,
                        metadata=metadata,
                    )
                    update_pbar(progress, task_id_vuln)
                    return result

            simulate_tasks = [
                asyncio.create_task(
                    throttled_simulate_baseline_attack(vulnerability)
                )
                for vulnerability in vulnerabilities
            ]

            attack_results = await asyncio.gather(*simulate_tasks)
            for result in attack_results:
                test_cases.extend(result)

            if attacks is not None:
                # Enhance attacks by sampling from the provided distribution
                task_id_attack = add_pbar(
                    progress,
                    description=f"✨ Simulating {num_vulnerability_types * attacks_per_vulnerability_type} attacks (using {len(attacks)} method(s))",
                    total=len(test_cases),
                )

                async def throttled_attack_method(
                    test_case: RTTestCase,
                ):
                    async with semaphore:  # Throttling applied here
                        # Randomly sample an enhancement based on the distribution
                        if not attacks:
                            attack = BaselineAttack()
                        else:
                            attack_weights = [
                                attack.weight for attack in attacks
                            ]
                            attack = random.choices(
                                attacks, weights=attack_weights, k=1
                            )[0]

                        await self.a_enhance_attack(
                            attack=attack,
                            test_case=test_case,
                            ignore_errors=ignore_errors,
                        )
                        update_pbar(progress, task_id_attack)

                await asyncio.gather(
                    *[
                        asyncio.create_task(throttled_attack_method(test_case))
                        for test_case in test_cases
                    ]
                )

        self.test_cases.extend(test_cases)
        return test_cases

    ##################################################
    ### Simulating Base (Unenhanced) Attacks #########
    ##################################################

    def simulate_baseline_attacks(
        self,
        attacks_per_vulnerability_type: int,
        vulnerability: BaseVulnerability,
        ignore_errors: bool,
        metadata: Optional[dict] = None,
    ) -> List[RTTestCase]:
        try:
            return vulnerability.simulate_attacks(
                purpose=self.purpose,
                attacks_per_vulnerability_type=attacks_per_vulnerability_type,
            )
        except Exception as e:
            if ignore_errors:
                return [
                    RTTestCase(
                        vulnerability=vulnerability.get_name(),
                        vulnerability_type=vulnerability_type,
                        error=f"Error simulating adversarial attacks: {str(e)}",
                        metadata=metadata,
                    )
                    for vulnerability_type in vulnerability.get_types()
                    for _ in range(attacks_per_vulnerability_type)
                ]
            else:
                raise

    async def a_simulate_baseline_attacks(
        self,
        attacks_per_vulnerability_type: int,
        vulnerability: BaseVulnerability,
        ignore_errors: bool,
        metadata: Optional[dict] = None,
    ) -> List[RTTestCase]:
        try:
            return await vulnerability.a_simulate_attacks(
                purpose=self.purpose,
                attacks_per_vulnerability_type=attacks_per_vulnerability_type,
            )
        except Exception as e:
            if ignore_errors:
                return [
                    RTTestCase(
                        vulnerability=vulnerability.get_name(),
                        vulnerability_type=vulnerability_type,
                        error=f"Error simulating adversarial attacks: {str(e)}",
                        metadata=metadata,
                    )
                    for vulnerability_type in vulnerability.get_types()
                    for _ in range(attacks_per_vulnerability_type)
                ]
            else:
                raise

    ##################################################
    ### Enhance attacks ##############################
    ##################################################

    def enhance_attack(
        self,
        attack: BaseAttack,
        test_case: RTTestCase,
        ignore_errors: bool,
    ):
        from deepteam.attacks.multi_turn import (
            BadLikertJudge,
            CrescendoJailbreaking,
            LinearJailbreaking,
            SequentialJailbreak,
            TreeJailbreaking,
        )
        from deepteam.test_case.test_case import RTTurn

        MULTI_TURN_ATTACKS = [
            BadLikertJudge,
            CrescendoJailbreaking,
            LinearJailbreaking,
            TreeJailbreaking,
            SequentialJailbreak,
        ]

        # Check if the attack is a multi-turn attack
        if type(attack) in MULTI_TURN_ATTACKS:
            # This is a multi-turn attack
            attack_input = test_case.input
            if attack_input is None:
                return test_case

            test_case.attack_method = attack.get_name()
            turns = [RTTurn(role="user", content=attack_input)]

            try:
                res = attack._get_turns(
                    model_callback=self.model_callback,
                    turns=turns,
                    vulnerability=test_case.vulnerability,
                    vulnerability_type=test_case.vulnerability_type.value,
                )

                test_case.turns = res

            except ModelRefusalError as e:
                if ignore_errors:
                    test_case.error = e.message
                    return test_case
                else:
                    raise
            except Exception as e:
                if ignore_errors:
                    test_case.error = (
                        f"Error enhancing multi-turn attack: {str(e)}"
                    )
                    return test_case
                else:
                    raise

            return test_case

        # Regular attack handling (non-multi-turn)
        attack_input = test_case.input
        if attack_input is None:
            return test_case

        test_case.attack_method = attack.get_name()
        sig = inspect.signature(attack.enhance)

        try:
            res = None
            if (
                "simulator_model" in sig.parameters
                and "model_callback" in sig.parameters
            ):
                res = attack.enhance(
                    attack=attack_input,
                    simulator_model=self.simulator_model,
                    model_callback=self.model_callback,
                )
            elif "simulator_model" in sig.parameters:
                res = attack.enhance(
                    attack=attack_input,
                    simulator_model=self.simulator_model,
                )
            elif "model_callback" in sig.parameters:
                res = attack.enhance(
                    attack=attack_input,
                    model_callback=self.model_callback,
                )
            else:
                res = attack.enhance(attack=attack_input)

            test_case.input = res

        except ModelRefusalError as e:
            if ignore_errors:
                test_case.error = e.message
                return test_case
            else:
                raise
        except Exception as e:
            if ignore_errors:
                test_case.error = f"Error enhancing regular attack: {str(e)}"
                return test_case
            else:
                raise

        return test_case

    async def a_enhance_attack(
        self,
        attack: BaseAttack,
        test_case: RTTestCase,
        ignore_errors: bool,
    ):
        from deepteam.attacks.multi_turn import (
            BadLikertJudge,
            CrescendoJailbreaking,
            LinearJailbreaking,
            SequentialJailbreak,
            TreeJailbreaking,
        )
        from deepteam.test_case.test_case import RTTurn

        MULTI_TURN_ATTACKS = [
            BadLikertJudge,
            CrescendoJailbreaking,
            LinearJailbreaking,
            TreeJailbreaking,
            SequentialJailbreak,
        ]

        if type(attack) in MULTI_TURN_ATTACKS:
            # This is multi-turn attack
            attack_input = test_case.input
            if attack_input is None:
                return test_case

            test_case.attack_method = attack.get_name()
            sig = inspect.signature(attack.a_enhance)
            turns = [RTTurn(role="user", content=attack_input)]

            try:
                res = await attack._a_get_turns(
                    model_callback=self.model_callback,
                    turns=turns,
                    vulnerability=test_case.vulnerability,
                    vulnerability_type=test_case.vulnerability_type.value,
                )

                test_case.turns = res

            except ModelRefusalError as e:
                if ignore_errors:
                    test_case.error = e.message
                    return test_case
                else:
                    raise
            except:
                if ignore_errors:
                    test_case.error = "Error enhancing attack"
                    return test_case
                else:
                    raise

            return test_case

        attack_input = test_case.input
        if attack_input is None:
            return test_case

        test_case.attack_method = attack.get_name()
        sig = inspect.signature(attack.a_enhance)

        try:
            res = None
            if (
                "simulator_model" in sig.parameters
                and "model_callback" in sig.parameters
            ):
                res = await attack.a_enhance(
                    attack=attack_input,
                    simulator_model=self.simulator_model,
                    model_callback=self.model_callback,
                )
            elif "simulator_model" in sig.parameters:
                res = await attack.a_enhance(
                    attack=attack_input,
                    simulator_model=self.simulator_model,
                )
            elif "model_callback" in sig.parameters:
                res = await attack.a_enhance(
                    attack=attack_input,
                    model_callback=self.model_callback,
                )
            else:
                res = await attack.a_enhance(attack=attack_input)

            test_case.input = res

        except ModelRefusalError as e:
            if ignore_errors:
                test_case.error = e.message
                return test_case
            else:
                raise
        except:
            if ignore_errors:
                test_case.error = "Error enhancing attack"
                return test_case
            else:
                raise

        return test_case
