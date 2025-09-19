import random
import asyncio
from tqdm import tqdm
from pydantic import BaseModel
from typing import List, Optional, Union
import inspect
from enum import Enum


from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import initialize_model, trimAndLoadJson
from deepeval.test_case import Turn
from deepeval.metrics.utils import initialize_model

from deepteam.attacks import BaseAttack
from deepteam.vulnerabilities import BaseVulnerability
from deepteam.vulnerabilities.types import VulnerabilityType
from deepteam.attacks.multi_turn.types import CallbackType
from deepteam.errors import ModelRefusalError


class SimulatedAttack(BaseModel):
    vulnerability: str
    vulnerability_type: Union[Enum, VulnerabilityType]
    input: Optional[str] = None
    attack_method: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[dict] = None
    turn_history: Optional[List[Turn]] = None


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
        self.simulated_attacks: List[SimulatedAttack] = []
        self.max_concurrent = max_concurrent

    ##################################################
    ### Generating Attacks ###########################
    ##################################################

    def simulate(
        self,
        attacks_per_vulnerability_type: int,
        vulnerabilities: List[BaseVulnerability],
        attacks: List[BaseAttack],
        ignore_errors: bool,
        metadata: Optional[dict] = None,
    ) -> List[SimulatedAttack]:
        # Simulate unenhanced attacks for each vulnerability
        baseline_attacks: List[SimulatedAttack] = []
        num_vulnerability_types = sum(
            len(v.get_types()) for v in vulnerabilities
        )
        pbar = tqdm(
            vulnerabilities,
            desc=f"💥 Generating {num_vulnerability_types * attacks_per_vulnerability_type} attacks (for {num_vulnerability_types} vulnerability types across {len(vulnerabilities)} vulnerability(s))",
        )
        for vulnerability in pbar:
            baseline_attacks.extend(
                self.simulate_baseline_attacks(
                    attacks_per_vulnerability_type=attacks_per_vulnerability_type,
                    vulnerability=vulnerability,
                    ignore_errors=ignore_errors,
                    metadata=metadata,
                )
            )
        # Enhance attacks by sampling from the provided distribution
        simulated_attacks: List[SimulatedAttack] = []
        pbar = tqdm(
            baseline_attacks,
            desc=f"✨ Simulating {num_vulnerability_types * attacks_per_vulnerability_type} attacks (using {len(attacks)} method(s))",
        )
        attack_weights = [attack.weight for attack in attacks]

        for baseline_attack in pbar:
            # Randomly sample an enhancement based on the distribution
            sampled_attack = random.choices(
                attacks, weights=attack_weights, k=1
            )[0]
            enhanced_attack = self.enhance_attack(
                attack=sampled_attack,
                simulated_attack=baseline_attack,
                ignore_errors=ignore_errors,
            )
            simulated_attacks.append(enhanced_attack)

        self.simulated_attacks.extend(simulated_attacks)
        return simulated_attacks

    async def a_simulate(
        self,
        attacks_per_vulnerability_type: int,
        vulnerabilities: List[BaseVulnerability],
        ignore_errors: bool,
        metadata: Optional[dict] = None,
        attacks: Optional[List[BaseAttack]] = None,
    ) -> List[SimulatedAttack]:
        # Create a semaphore to control the number of concurrent tasks
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # Simulate unenhanced attacks for each vulnerability
        baseline_attacks: List[SimulatedAttack] = []
        num_vulnerability_types = sum(
            len(v.get_types()) for v in vulnerabilities
        )
        pbar = tqdm(
            vulnerabilities,
            desc=f"💥 Generating {num_vulnerability_types * attacks_per_vulnerability_type} attacks (for {num_vulnerability_types} vulnerability types across {len(vulnerabilities)} vulnerability(s))",
        )

        async def throttled_simulate_baseline_attack(vulnerability):
            async with semaphore:  # Throttling applied here
                result = await self.a_simulate_baseline_attacks(
                    attacks_per_vulnerability_type=attacks_per_vulnerability_type,
                    vulnerability=vulnerability,
                    ignore_errors=ignore_errors,
                    metadata=metadata,
                )
                pbar.update(1)
                return result

        simulate_tasks = [
            asyncio.create_task(
                throttled_simulate_baseline_attack(vulnerability)
            )
            for vulnerability in vulnerabilities
        ]

        attack_results = await asyncio.gather(*simulate_tasks)
        for result in attack_results:
            baseline_attacks.extend(result)
        pbar.close()

        # Enhance attacks by sampling from the provided distribution
        enhanced_attacks: List[SimulatedAttack] = []
        pbar = tqdm(
            total=len(baseline_attacks),
            desc=f"✨ Simulating {num_vulnerability_types * attacks_per_vulnerability_type} attacks (using {len(attacks)} method(s))",
        )

        async def throttled_attack_method(
            baseline_attack: SimulatedAttack,
        ):
            async with semaphore:  # Throttling applied here
                # Randomly sample an enhancement based on the distribution
                if not attacks:
                    attack = BaselineAttack()
                else:
                    attack_weights = [attack.weight for attack in attacks]
                    attack = random.choices(
                        attacks, weights=attack_weights, k=1
                    )[0]

                result = await self.a_enhance_attack(
                    attack=attack,
                    simulated_attack=baseline_attack,
                    ignore_errors=ignore_errors,
                )
                pbar.update(1)
                return result

        enhanced_attacks.extend(
            await asyncio.gather(
                *[
                    asyncio.create_task(
                        throttled_attack_method(baseline_attack)
                    )
                    for baseline_attack in baseline_attacks
                ]
            )
        )
        pbar.close()

        self.simulated_attacks.extend(enhanced_attacks)
        return enhanced_attacks

    ##################################################
    ### Simulating Base (Unenhanced) Attacks #########
    ##################################################

    def simulate_baseline_attacks(
        self,
        attacks_per_vulnerability_type: int,
        vulnerability: BaseVulnerability,
        ignore_errors: bool,
        metadata: Optional[dict] = None,
    ) -> List[SimulatedAttack]:
        try:
            return vulnerability.simulate_attacks(
                purpose=self.purpose,
                attacks_per_vulnerability_type=attacks_per_vulnerability_type,
            )
        except Exception as e:
            if ignore_errors:
                return [
                    SimulatedAttack(
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
    ) -> List[SimulatedAttack]:
        try:
            return await vulnerability.a_simulate_attacks(
                purpose=self.purpose,
                attacks_per_vulnerability_type=attacks_per_vulnerability_type,
            )
        except Exception as e:
            if ignore_errors:
                return [
                    SimulatedAttack(
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
        simulated_attack: SimulatedAttack,
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
            attack_input = simulated_attack.input
            if attack_input is None:
                return simulated_attack

            simulated_attack.attack_method = attack.get_name()
            sig = inspect.signature(attack.a_enhance)
            turns = [RTTurn(role="user", content=attack_input)]

            try:
                res = None
                if (
                    "simulator_model" in sig.parameters
                    and "model_callback" in sig.parameters
                    and "turns" in sig.parameters
                ):
                    res = attack.enhance(
                        self.model_callback, turns, self.simulator_model
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

                simulated_attack.turn_history = res

            except ModelRefusalError as e:
                if ignore_errors:
                    simulated_attack.error = e.message
                    return simulated_attack
                else:
                    raise
            except:
                if ignore_errors:
                    simulated_attack.error = "Error enhancing attack"
                    return simulated_attack
                else:
                    raise

            return simulated_attack

        attack_input = simulated_attack.input
        if attack_input is None:
            return simulated_attack

        simulated_attack.attack_method = attack.get_name()
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

            simulated_attack.input = res

        except ModelRefusalError as e:
            if ignore_errors:
                simulated_attack.error = e.message
                return simulated_attack
            else:
                raise
        except:
            if ignore_errors:
                simulated_attack.error = "Error enhancing attack"
                return simulated_attack
            else:
                raise

        return simulated_attack

    async def a_enhance_attack(
        self,
        attack: BaseAttack,
        simulated_attack: SimulatedAttack,
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
            attack_input = simulated_attack.input
            if attack_input is None:
                return simulated_attack

            simulated_attack.attack_method = attack.get_name()
            sig = inspect.signature(attack.a_enhance)
            turns = [RTTurn(role="user", content=attack_input)]

            try:
                res = None
                if (
                    "simulator_model" in sig.parameters
                    and "model_callback" in sig.parameters
                    and "turns" in sig.parameters
                ):
                    res = await attack.a_enhance(
                        self.model_callback, turns, self.simulator_model
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

                simulated_attack.turn_history = res

            except ModelRefusalError as e:
                if ignore_errors:
                    simulated_attack.error = e.message
                    return simulated_attack
                else:
                    raise
            except:
                if ignore_errors:
                    simulated_attack.error = "Error enhancing attack"
                    return simulated_attack
                else:
                    raise

            return simulated_attack

        attack_input = simulated_attack.input
        if attack_input is None:
            return simulated_attack

        simulated_attack.attack_method = attack.get_name()
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

            simulated_attack.input = res

        except ModelRefusalError as e:
            if ignore_errors:
                simulated_attack.error = e.message
                return simulated_attack
            else:
                raise
        except:
            if ignore_errors:
                simulated_attack.error = "Error enhancing attack"
                return simulated_attack
            else:
                raise

        return simulated_attack
