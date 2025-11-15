from rich.progress import Progress
import asyncio
import time
import json
from typing import Optional, Union, List, Dict
import random

from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import initialize_model

from deepteam.utils import create_progress, update_pbar, add_pbar
from deepteam.attacks import BaseAttack
from deepteam.attacks.multi_turn.tree_jailbreaking.schema import (
    ImprovementPrompt,
    Rating,
)
from deepteam.attacks.multi_turn.tree_jailbreaking.template import (
    JailBreakingTemplate,
)
from deepteam.attacks.attack_simulator.utils import (
    generate,
    a_generate,
)
from deepteam.attacks.multi_turn.utils import enhance_attack, a_enhance_attack
from deepteam.attacks.multi_turn.types import CallbackType
from deepteam.attacks.multi_turn.base_schema import NonRefusal
from deepteam.test_case.test_case import RTTurn
from deepteam.vulnerabilities.types import VulnerabilityType
from deepteam.vulnerabilities import BaseVulnerability


class TreeNode:
    def __init__(
        self,
        prompt: str,
        score: int,
        depth: int,
        conversation_history=None,
        parent: Optional["TreeNode"] = None,
        turn_level_attack: Optional[BaseAttack] = None,
    ):
        self.prompt = prompt
        self.score = score
        self.depth = depth
        self.children = []
        self.parent = parent
        self.turn_level_attack = turn_level_attack
        self.conversation_history = conversation_history or []


class TreeJailbreaking(BaseAttack):

    def __init__(
        self,
        weight: int = 1,
        max_depth: int = 5,
        turn_level_attacks: Optional[List[BaseAttack]] = None,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = "gpt-4o-mini",
    ):
        self.weight = weight
        self.multi_turn = True
        self.simulator_model = simulator_model
        self.max_depth = max_depth
        self.turn_level_attacks = turn_level_attacks

        if self.turn_level_attacks is not None:
            if not isinstance(self.turn_level_attacks, list) or not all(
                attack.multi_turn == False for attack in self.turn_level_attacks
            ):
                raise ValueError(
                    "The 'attacks' passed must be a list of single-turn attacks"
                )

    def _get_turns(
        self,
        model_callback: CallbackType,
        turns: Optional[List[RTTurn]] = None,
        vulnerability: str = None,
        vulnerability_type: str = None,
    ) -> List[RTTurn]:
        self.simulator_model, _ = initialize_model(self.simulator_model)
        self.model_callback = model_callback

        if turns is None:
            turns = []

        # Extract the last user input to use as the root attack
        attack = None
        for turn in reversed(turns):
            if turn.role == "user":
                attack = turn.content
                break

        if not attack:
            raise ValueError(
                "No user input found in the conversation to start from."
            )

        # Ensure the conversation is initialized correctly
        if len(turns) <= 1 or turns[-1].role == "user":
            assistant_response = model_callback(attack, turns)
            turns.append(RTTurn(role="assistant", content=assistant_response))

        MAX_RUNTIME = 50.0
        start_time = time.time()
        root = TreeNode(prompt=attack, score=0, depth=0)

        vulnerability_data = (
            f"Vulnerability: {vulnerability} | Type: {vulnerability_type}"
        )

        # Progress bar
        progress = create_progress()
        with progress:
            task_id = add_pbar(
                progress,
                description="...... ⛓️  Tree Jailbreaking",
                total=MAX_RUNTIME,
            )

            try:
                best_node = self.tree_search(
                    root,
                    attack,
                    start_time,
                    MAX_RUNTIME,
                    progress,
                    task_id,
                    vulnerability_data,
                )
            finally:
                update_pbar(progress, task_id, advance_to_end=True)

            # Reconstruct path from best_node to root
            path_nodes = []
            current = best_node
            while current is not None:
                path_nodes.append(current)
                current = current.parent
            path_nodes.reverse()

            if path_nodes:
                first_prompt = path_nodes[0].prompt
                if any(
                    t.content == first_prompt and t.role == "user"
                    for t in turns
                ):
                    path_nodes = path_nodes[1:]

            # Append turns along the best path
            for node in path_nodes:
                turns.append(RTTurn(role="user", content=node.prompt))
                assistant_response = model_callback(node.prompt, turns)
                turns.append(
                    RTTurn(role="assistant", content=assistant_response)
                )
                if node.turn_level_attack is not None:
                    turns.append(
                        RTTurn(
                            role="assistant",
                            content=assistant_response,
                            turn_level_attack=node.turn_level_attack.get_name(),
                        )
                    )
                else:
                    turns.append(
                        RTTurn(role="assistant", content=assistant_response)
                    )

        return turns

    async def _a_get_turns(
        self,
        model_callback: CallbackType,
        turns: Optional[List[RTTurn]] = None,
        vulnerability: str = None,
        vulnerability_type: str = None,
    ) -> List[RTTurn]:
        self.simulator_model, _ = initialize_model(self.simulator_model)
        self.model_callback = model_callback

        if turns is None:
            turns = []

        # Extract the last user input to use as the root attack
        attack = None
        for turn in reversed(turns):
            if turn.role == "user":
                attack = turn.content
                break

        if not attack:
            raise ValueError(
                "No user input found in the conversation to start from."
            )

        # Ensure the conversation is initialized correctly
        if len(turns) <= 1 or turns[-1].role == "user":
            assistant_response = await model_callback(attack, turns)
            turns.append(RTTurn(role="assistant", content=assistant_response))

        MAX_RUNTIME = 50.0
        start_time = time.time()
        root = TreeNode(prompt=attack, score=0, depth=0)

        vulnerability_data = (
            f"Vulnerability: {vulnerability} | Type: {vulnerability_type}"
        )

        # Async progress bar
        progress = create_progress()
        with progress:
            task_id = add_pbar(
                progress,
                description="...... ⛓️  Tree Jailbreaking",
                total=MAX_RUNTIME,
            )

            try:
                best_node = await self.a_tree_search(
                    root,
                    attack,
                    start_time,
                    MAX_RUNTIME,
                    vulnerability_data,
                    progress,
                    task_id,
                )
            finally:
                update_pbar(progress, task_id, advance_to_end=True)

            # Reconstruct best path
            path_nodes = []
            current = best_node
            while current is not None:
                path_nodes.append(current)
                current = current.parent
            path_nodes.reverse()

            if path_nodes:
                first_prompt = path_nodes[0].prompt
                if any(
                    t.content == first_prompt and t.role == "user"
                    for t in turns
                ):
                    path_nodes = path_nodes[1:]

            # Append turns from best path
            for node in path_nodes:
                turns.append(RTTurn(role="user", content=node.prompt))
                assistant_response = await model_callback(node.prompt, turns)
                turns.append(
                    RTTurn(role="assistant", content=assistant_response)
                )

        return turns

    def progress(
        self,
        vulnerability: BaseVulnerability,
        model_callback: CallbackType,
        turns: Optional[List[RTTurn]] = None,
    ) -> Dict[VulnerabilityType, List[RTTurn]]:
        from deepteam.red_teamer.utils import (
            group_attacks_by_vulnerability_type,
        )

        self.simulator_model, _ = initialize_model(self.simulator_model)
        self.model_callback = model_callback

        simulated_attacks = vulnerability.simulate_attacks()
        grouped_attacks = group_attacks_by_vulnerability_type(simulated_attacks)

        result: Dict[VulnerabilityType, List[List[RTTurn]]] = {}

        for vuln_type, attacks in grouped_attacks.items():
            for attack in attacks:
                # Prepare vulnerability data string for prompt injection

                # Defensive copy of turns if any
                inner_turns = list(turns) if turns else []

                # Initialize conversation if needed
                if len(inner_turns) == 0 or inner_turns[-1].role == "user":
                    inner_turns = [RTTurn(role="user", content=attack.input)]
                    assistant_response = model_callback(
                        attack.input, inner_turns
                    )
                    inner_turns.append(
                        RTTurn(role="assistant", content=assistant_response)
                    )

                elif inner_turns[-1].role == "assistant":
                    user_turn_content = None
                    for turn in reversed(inner_turns[:-1]):
                        if turn.role == "user":
                            user_turn_content = turn.content
                            break
                    if user_turn_content:
                        inner_turns = [
                            RTTurn(role="user", content=user_turn_content),
                            RTTurn(
                                role="assistant",
                                content=inner_turns[-1].content,
                            ),
                        ]
                    else:
                        inner_turns = [
                            RTTurn(role="user", content=attack.input)
                        ]
                        assistant_response = model_callback(
                            attack.input, inner_turns
                        )
                        inner_turns.append(
                            RTTurn(role="assistant", content=assistant_response)
                        )
                else:
                    inner_turns = [RTTurn(role="user", content=attack.input)]
                    assistant_response = model_callback(
                        attack.input, inner_turns
                    )
                    inner_turns.append(
                        RTTurn(role="assistant", content=assistant_response)
                    )

                # Run tree-based multi-turn search
                vulnerability_name = vulnerability.get_name()
                enhanced_turns = self._get_turns(
                    model_callback=model_callback,
                    turns=inner_turns,
                    vulnerability=vulnerability_name,
                    vulnerability_type=vuln_type.value,
                )

            result[vuln_type] = enhanced_turns

        return result

    def tree_search(
        self,
        root: TreeNode,
        goal: str,
        start_time: float,
        MAX_RUNTIME: float,
        progress: Progress,
        task_id: int,
        vulnerability_data: str = "",
    ) -> TreeNode:
        """Synchronous tree search implementation with runtime limit."""
        queue = [root]
        best_node = root

        while queue and (time.time() - start_time < MAX_RUNTIME):
            node = queue.pop(0)
            self.expand_node(node, goal, vulnerability_data=vulnerability_data)
            queue.extend(node.children)
            queue.sort(key=lambda x: x.score, reverse=True)

            if node.score > best_node.score:
                best_node = node

            # Update progress bar
            elapsed_time = time.time() - start_time
            update_pbar(
                progress,
                task_id,
                elapsed_time - progress.tasks[task_id].completed,
            )

        return best_node

    def expand_node(
        self,
        node: TreeNode,
        initial_attack: str,
        PRUNE_THRESHOLD: int = 0,
        vulnerability_data: str = "",
    ):
        if node.depth >= self.max_depth:
            return

        """Expands a tree node by generating multiple children nodes based on branching logic."""
        if node.depth == 0:
            conversation_json = [
                {
                    "role": "system",
                    "content": JailBreakingTemplate.enhance_attack(
                        initial_attack, vulnerability_data
                    ),
                }
            ]
        else:
            conversation_json = node.conversation_history

        conversation_json = list(conversation_json)
        conversation_json.append(
            {
                "role": "user",
                "content": f"Current prompt: {node.prompt}\n\nOBJECTIVE: {initial_attack}\n\nSCORE: {node.score}",
            }
        )

        # Generate multiple branches
        num_branches = self.calculate_branches(node.score, node.depth)
        for _ in range(num_branches):
            conversation = json.dumps(conversation_json)
            res: ImprovementPrompt = generate(
                conversation, ImprovementPrompt, self.simulator_model
            )
            enhanced_attack = res.prompt

            # Randomly enhancing a turn attack
            turn_level_attack = None
            if self.turn_level_attacks and random.random() < 0.5:
                attack = random.choice(self.turn_level_attacks)
                turn_level_attack = attack
                enhanced_attack = enhance_attack(
                    attack, enhanced_attack, self.simulator_model
                )

            # Check if enhanced attack is a refusal statement
            non_refusal_prompt = JailBreakingTemplate.non_refusal(
                initial_attack, enhanced_attack
            )
            res: NonRefusal = generate(
                non_refusal_prompt, NonRefusal, self.simulator_model
            )
            classification = res.classification
            if classification == "Refusal":
                continue  # Skip this child

            # Generate a response from the target LLM
            target_response = self.model_callback(enhanced_attack)

            # Calculate the score for the enhanced attack
            judge_prompt = JailBreakingTemplate.linear_judge(
                initial_attack,
                enhanced_attack,
                target_response,
                vulnerability_data,
            )
            res: Rating = generate(judge_prompt, Rating, self.simulator_model)
            score = res.rating

            # Prune if the score is too low
            if score <= PRUNE_THRESHOLD:
                continue  # Skip creating this child

            # Create a new child node
            child_node = TreeNode(
                prompt=enhanced_attack,
                score=score,
                depth=node.depth + 1,
                conversation_history=conversation_json,
                parent=node,
                turn_level_attack=turn_level_attack,
            )
            node.children.append(child_node)

    async def a_progress(
        self,
        vulnerability: BaseVulnerability,
        model_callback: CallbackType,
        turns: Optional[List[RTTurn]] = None,
    ) -> Dict[VulnerabilityType, List[RTTurn]]:
        from deepteam.red_teamer.utils import (
            group_attacks_by_vulnerability_type,
        )

        self.simulator_model, _ = initialize_model(self.simulator_model)
        self.model_callback = model_callback

        simulated_attacks = await vulnerability.a_simulate_attacks()
        grouped_attacks = group_attacks_by_vulnerability_type(simulated_attacks)

        result: Dict[VulnerabilityType, List[List[RTTurn]]] = {}

        for vuln_type, attacks in grouped_attacks.items():
            for attack in attacks:
                # Defensive copy to avoid mutating external turns
                inner_turns = list(turns) if turns else []

                # Case 1: No turns, or last is user -> create assistant response
                if len(inner_turns) == 0 or inner_turns[-1].role == "user":
                    inner_turns = [RTTurn(role="user", content=attack.input)]
                    assistant_response = await model_callback(
                        attack.input, inner_turns
                    )
                    inner_turns.append(
                        RTTurn(role="assistant", content=assistant_response)
                    )

                # Case 2: Last is assistant -> find preceding user
                elif inner_turns[-1].role == "assistant":
                    user_turn_content = None
                    for turn in reversed(inner_turns[:-1]):
                        if turn.role == "user":
                            user_turn_content = turn.content
                            break

                    if user_turn_content:
                        inner_turns = [
                            RTTurn(role="user", content=user_turn_content),
                            RTTurn(
                                role="assistant",
                                content=inner_turns[-1].content,
                            ),
                        ]
                    else:
                        # Fallback if no user found
                        inner_turns = [
                            RTTurn(role="user", content=attack.input)
                        ]
                        assistant_response = await model_callback(
                            attack.input, inner_turns
                        )
                        inner_turns.append(
                            RTTurn(role="assistant", content=assistant_response)
                        )

                else:
                    # Unrecognized state — fallback to default
                    inner_turns = [RTTurn(role="user", content=attack.input)]
                    assistant_response = await model_callback(
                        attack.input, inner_turns
                    )
                    inner_turns.append(
                        RTTurn(role="assistant", content=assistant_response)
                    )

                # Run enhancement loop and assign full turn history
                vulnerability_name = vulnerability.get_name()
                enhanced_turns = await self._a_get_turns(
                    model_callback=model_callback,
                    turns=inner_turns,
                    vulnerability=vulnerability_name,
                    vulnerability_type=vuln_type.value,
                )

            result[vuln_type] = enhanced_turns

        return result

    async def a_tree_search(
        self,
        root: TreeNode,
        goal: str,
        start_time: float,
        MAX_RUNTIME: float,
        vulnerability_data: str,
        progress: Progress,
        task_id: int,
    ) -> TreeNode:
        """Asynchronous tree search implementation with runtime limit."""
        queue = [root]
        best_node = root

        while queue and (time.time() - start_time < MAX_RUNTIME):
            node = queue.pop(0)
            await self.a_expand_node(
                node, goal, vulnerability_data=vulnerability_data
            )
            queue.extend(node.children)
            queue.sort(key=lambda x: x.score, reverse=True)

            if node.score > best_node.score:
                best_node = node

            elapsed_time = time.time() - start_time
            update_pbar(progress, task_id, elapsed_time)

        update_pbar(progress, task_id, advance_to_end=True)
        return best_node

    async def a_expand_node(
        self,
        node: TreeNode,
        initial_attack: str,
        PRUNE_THRESHOLD: int = 0,
        vulnerability_data: str = "",
    ):
        if node.depth >= self.max_depth:
            return

        """Expands a tree node asynchronously by generating multiple children nodes."""
        if node.depth == 0:
            conversation_json = [
                {
                    "role": "system",
                    "content": JailBreakingTemplate.enhance_attack(
                        initial_attack, vulnerability_data
                    ),
                }
            ]
        else:
            conversation_json = node.conversation_history

        conversation_json = list(conversation_json)
        conversation_json.append(
            {
                "role": "user",
                "content": f"Current prompt: {node.prompt}\n\nOBJECTIVE: {initial_attack}\n\nSCORE: {node.score}",
            }
        )

        # Generate branches asynchronously
        num_branches = self.calculate_branches(node.score, node.depth)
        tasks = [
            self.a_generate_child(
                node,
                conversation_json,
                initial_attack,
                PRUNE_THRESHOLD,
                vulnerability_data,
            )
            for _ in range(num_branches)
        ]
        children = await asyncio.gather(*tasks)

        # Filter out None values for pruned branches
        node.children.extend([child for child in children if child])

    async def a_generate_child(
        self,
        node: TreeNode,
        conversation_json: dict,
        initial_attack: str,
        PRUNE_THRESHOLD: int,
        vulnerability_data: str = "",
    ):
        """Asynchronously generates a child node."""
        conversation = json.dumps(conversation_json)
        res: ImprovementPrompt = await a_generate(
            conversation, ImprovementPrompt, self.simulator_model
        )
        enhanced_attack = res.prompt

        # Randomly enhancing a turn attack
        turn_level_attack = None
        if self.turn_level_attacks and random.random() < 0.5:
            attack = random.choice(self.turn_level_attacks)
            turn_level_attack = attack
            enhanced_attack = await a_enhance_attack(
                attack, enhanced_attack, self.simulator_model
            )

        # Check if enhanced attack is a refusal statement
        non_refusal_prompt = JailBreakingTemplate.non_refusal(
            initial_attack, enhanced_attack
        )
        res: NonRefusal = await a_generate(
            non_refusal_prompt, NonRefusal, self.simulator_model
        )
        classification = res.classification
        if classification == "Refusal":
            return None  # Skip this child

        # Generate a response from the target LLM asynchronously
        target_response = await self.model_callback(enhanced_attack)

        # Calculate the score for the enhanced attack asynchronously
        judge_prompt = JailBreakingTemplate.linear_judge(
            initial_attack, enhanced_attack, target_response, vulnerability_data
        )
        res: Rating = await a_generate(
            judge_prompt, Rating, self.simulator_model
        )
        score = res.rating

        # Prune if the score is too low
        if score <= PRUNE_THRESHOLD:
            return None

        # Create a new child node
        return TreeNode(
            prompt=enhanced_attack,
            score=score,
            depth=node.depth + 1,
            conversation_history=conversation_json,
            parent=node,
            turn_level_attack=turn_level_attack,
        )

    ##################################################
    ### General Jailbreaking #########################
    ##################################################

    def calculate_branches(self, score: int, depth: int) -> int:
        """Calculate the number of branches (children) based on the score."""
        BASE_BRANCHES = 3
        MAX_BRANCHES = 5
        MIN_BRANCHES = 1

        branches = BASE_BRANCHES

        # Increase branches for high scores
        if score >= 8:
            branches += 2
        elif score >= 6:
            branches += 1

        # Decrease branches for low scores
        if score <= 3:
            branches -= 1

        return max(MIN_BRANCHES, min(MAX_BRANCHES, branches))

    def get_name(self) -> str:
        return "Tree Jailbreaking"
