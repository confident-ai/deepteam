from abc import ABC, abstractmethod
from typing import Optional, Dict, Union
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models.base_model import DeepEvalBaseLLM
import json
from deepteam.guardrails.types import GuardType
from deepteam.cli.model_callback import load_model

class BaseGuard(ABC):
    score: Optional[float] = None
    reason: Optional[str] = None
    evaluation_model: Optional[str] = None
    error: Optional[str] = None
    latency: Optional[float] = None
    guard_type: GuardType

    def __init__(self, evaluation_model: Union[str, Dict, DeepEvalBaseLLM] = "gpt-4.1"):
        if isinstance(evaluation_model, DeepEvalBaseLLM):
            self.model = evaluation_model
        else:
            self.model = load_model(evaluation_model)
        self.evaluation_model = self.model.get_model_name()

    @property
    def __name__(self):
        return "Base Guard"
    
    def guard(self, input: str = None, output: str = None, **kwargs) -> float:
        """
        Guard method - returns score: 1.0 (safe), 0.5 (uncertain), 0.0 (unsafe)
        """
        try:
            if output is not None:
                prompt = self._get_output_prompt(input or "", output)
            else:
                prompt = self._get_input_prompt(input)
                
            llm_output = self.model.generate(prompt)
            
            # Handle tuple response
            if isinstance(llm_output, tuple):
                llm_output = llm_output[0]
                
            return self._parse_json_response(llm_output)
            
        except Exception as e:
            self.error = str(e)
            self.score = 0.0  
            return 0.0
    
    async def a_guard(self, input: str = None, output: str = None, **kwargs) -> float:
        """
        Async guard method - returns score: 1.0 (safe), 0.5 (uncertain), 0.0 (unsafe)
        """
        try:
            if output is not None:
                prompt = self._get_output_prompt(input or "", output)
            else:  
                prompt = self._get_input_prompt(input)
                
            llm_output = await self.model.a_generate(prompt)
            
            # Handle tuple response
            if isinstance(llm_output, tuple):
                llm_output = llm_output[0]
                
            return self._parse_json_response(llm_output)
            
        except Exception as e:
            self.error = str(e)
            self.score = 0.0  
            return 0.0
    
    def _parse_json_response(self, response: str) -> float:
        """
        Parse JSON response from LLM to determine safety level
        Returns 1.0 (safe), 0.5 (uncertain), or 0.0 (unsafe) for internal use
        """
        try:
            data = trim_and_load_json(response)
            safe_value = data.get("safe", False)
            self.reason = data.get("reason", "No explanation provided")
            
            # Convert to words that developers see
            if safe_value == True or safe_value == "true":
                self.safety_level = "safe"
                self.score = 1.0
            elif safe_value == "uncertain":
                self.safety_level = "uncertain"
                self.score = 0.5
            else:  # False, "false", or any other value
                self.safety_level = "unsafe"
                self.score = 0.0
                
            return self.score
        except Exception as e:
            self.error = str(e)
            self.safety_level = "unsafe"
            self.score = 0.0
            return 0.0
    
    @abstractmethod
    def _get_input_prompt(self, input_text: str) -> str:
        """
        Get prompt for evaluating input text
        """
        pass
    
    @abstractmethod
    def _get_output_prompt(self, input_text: str, output_text: str) -> str:
        """
        Get prompt for evaluating output text
        """
        pass

    @abstractmethod
    async def a_guard_input(self, input: str, *args, **kwargs) -> float:
        raise NotImplementedError(
            f"Async execution for {self.__class__.__name__} not supported yet."
        )

    @abstractmethod
    async def a_guard_output(
        self, input: str, output: str, *args, **kwargs
    ) -> float:
        raise NotImplementedError(
            f"Async execution for {self.__class__.__name__} not supported yet."
        )
