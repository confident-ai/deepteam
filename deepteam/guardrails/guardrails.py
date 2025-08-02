from typing import List, Dict, Any, Union
from deepeval.models.base_model import DeepEvalBaseLLM
import time
import asyncio

from deepteam.guardrails.base_guard import BaseGuard
from deepteam.guardrails.types import GuardType


class GuardResult:
    def __init__(self, guard_results: Dict[str, Any]):
        self.guard_results = guard_results
        # Calculate breached based on 3-tier system
        # Only 0.0 (unsafe) is considered breached
        # 0.5 (uncertain) and 1.0 (safe) are not breached
        self.breached = any(
            result.get("score", 0) < 1.0 
            for result in guard_results.values()
        )


class Guardrails:
    """
    Open-source guardrails system for production LLM safety.
    Fast binary classification to guard inputs and outputs.
    """
    
    def __init__(
        self, 
        input_guards: List[BaseGuard],
        output_guards: List[BaseGuard],
        evaluation_model: Union[str, Dict, DeepEvalBaseLLM] = "gpt-4.1",
        sample_rate: float = 1.0
    ):
        """
        Initialize Guardrails with separate input and output guards.
        
        Args:
            input_guards: List of guards to check inputs before they reach your LLM
            output_guards: List of guards to check outputs before they reach your users
            evaluation_model: Model specification (string, dict, or DeepEvalBaseLLM instance) for guard evaluation (default: "gpt-4.1")
                            Supports OpenAI, Azure, Anthropic, Gemini, Ollama, Custom models
            sample_rate: Fraction of requests to actually guard (0.0 to 1.0, default: 1.0)
        """
        # Validate sample_rate 
        if not (0.0 <= sample_rate <= 1.0):
            raise ValueError(f"sample_rate must be between 0.0 and 1.0, got {sample_rate}")
        
        self.sample_rate = sample_rate
        self.evaluation_model = evaluation_model
        self._request_count = 0  
        
        # Update all guards to use the specified evaluation model
        self.input_guards = self._update_guards_model(input_guards, evaluation_model)
        self.output_guards = self._update_guards_model(output_guards, evaluation_model)

    def _update_guards_model(self, guards: List[BaseGuard], evaluation_model: Union[str, Dict, DeepEvalBaseLLM]) -> List[BaseGuard]:
        """Update all guards to use the specified evaluation model"""
        updated_guards = []
        for guard in guards:
            # Create new instance with updated model
            guard_class = guard.__class__
            new_guard = guard_class(evaluation_model=evaluation_model)
            updated_guards.append(new_guard)
        return updated_guards

    def _should_process(self) -> bool:
        """Deterministic sampling: exactly sample_rate fraction of requests"""
        self._request_count += 1
        if self.sample_rate == 0.0:
            return False
        if self.sample_rate == 1.0:
            return True
        
        # Simple deterministic approach: process every nth request
        interval = int(1 / self.sample_rate)
        return self._request_count % interval == 0

    def guard_input(self, input: str) -> GuardResult:
        """
        Guard an input string using all configured input guards.
        Returns GuardResult with binary safe/unsafe classification.
        """
        if len(self.input_guards) == 0:
            raise TypeError(
                "Guardrails cannot guard inputs when no input_guards are provided."
            )

        # Deterministic sampling
        if not self._should_process():
            return GuardResult(guard_results={})

        guard_results = {}
        
        for guard in self.input_guards:
            start_time = time.time()
            try:
                is_safe = guard.guard_input(input)
                latency = time.time() - start_time
                
                guard_results[guard.__name__] = {
                    "safe": is_safe,
                    "safety_level": getattr(guard, "safety_level", "unsafe"),
                    "latency": latency,
                    "reason": getattr(guard, "reason", None),
                    "score": getattr(guard, "score", None)
                }
                    
            except Exception as e:
                guard_results[guard.__name__] = {
                    "safe": False,
                    "safety_level": "unsafe",
                    "latency": time.time() - start_time,
                    "error": str(e)
                }

        return GuardResult(guard_results=guard_results)

    def guard_output(self, input: str, output: str) -> GuardResult:
        """
        Guard an output string using all configured output guards.
        Returns GuardResult with binary safe/unsafe classification.
        """
        if len(self.output_guards) == 0:
            raise TypeError(
                "Guardrails cannot guard outputs when no output_guards are provided."
            )

        # Deterministic sampling
        if not self._should_process():
            return GuardResult(guard_results={})

        guard_results = {}
        
        for guard in self.output_guards:
            start_time = time.time()
            try:
                is_safe = guard.guard_output(input, output)
                latency = time.time() - start_time
                
                guard_results[guard.__name__] = {
                    "safe": is_safe,
                    "safety_level": getattr(guard, "safety_level", "unsafe"),
                    "latency": latency,
                    "reason": getattr(guard, "reason", None),
                    "score": getattr(guard, "score", None)
                }
                    
            except Exception as e:
                guard_results[guard.__name__] = {
                    "safe": False,
                    "safety_level": "unsafe",
                    "latency": time.time() - start_time,
                    "error": str(e)
                }

        return GuardResult(guard_results=guard_results)

    async def a_guard_input(self, input: str) -> GuardResult:
        """
        Async version of guard_input for better performance.
        """
        if len(self.input_guards) == 0:
            raise TypeError(
                "Guardrails cannot guard inputs when no input_guards are provided."
            )

        # Deterministic sampling
        if not self._should_process():
            return GuardResult(guard_results={})

        tasks = []
        for guard in self.input_guards:
            task = asyncio.create_task(self._async_guard_input_single(guard, input))
            tasks.append((guard, task))

        guard_results = {}

        for guard, task in tasks:
            try:
                result = await task
                guard_results[guard.__name__] = result
            except Exception as e:
                guard_results[guard.__name__] = {
                    "safe": False,
                    "safety_level": "unsafe",
                    "error": str(e)
                }

        return GuardResult(guard_results=guard_results)

    async def a_guard_output(self, input: str, output: str) -> GuardResult:
        """
        Async version of guard_output for better performance.
        """
        if len(self.output_guards) == 0:
            raise TypeError(
                "Guardrails cannot guard outputs when no output_guards are provided."
            )

        # Deterministic sampling
        if not self._should_process():
            return GuardResult(guard_results={})

        tasks = []
        for guard in self.output_guards:
            task = asyncio.create_task(self._async_guard_output_single(guard, input, output))
            tasks.append((guard, task))

        guard_results = {}

        for guard, task in tasks:
            try:
                result = await task
                guard_results[guard.__name__] = result
            except Exception as e:
                guard_results[guard.__name__] = {
                    "safe": False,
                    "safety_level": "unsafe",
                    "error": str(e)
                }

        return GuardResult(guard_results=guard_results)

    async def _async_guard_input_single(self, guard: BaseGuard, input: str):
        """Helper method for async input guarding."""
        start_time = time.time()
        is_safe = await guard.a_guard_input(input)
        latency = time.time() - start_time
        
        return {
            "safe": is_safe,
            "safety_level": getattr(guard, "safety_level", "unsafe"),
            "latency": latency,
            "reason": getattr(guard, "reason", None),
            "score": getattr(guard, "score", None)
        }

    async def _async_guard_output_single(self, guard: BaseGuard, input: str, output: str):
        """Helper method for async output guarding."""
        start_time = time.time()
        is_safe = await guard.a_guard_output(input, output)
        latency = time.time() - start_time
        
        return {
            "safe": is_safe,
            "safety_level": getattr(guard, "safety_level", "unsafe"),
            "latency": latency,
            "reason": getattr(guard, "reason", None),
            "score": getattr(guard, "score", None)
        }
