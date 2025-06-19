import requests
import json
import asyncio
from deepeval.models import DeepEvalBaseLLM


class CustomFireworksLLM(DeepEvalBaseLLM):
    def __init__(self, model_name="accounts/fireworks/models/llama4-scout-instruct-basic"):
        self.api_url = "https://api.fireworks.ai/inference/v1/chat/completions"
        self.api_token = "fw_3ZN7ZzUGBGry228zT9RcVdww"
        self.model_name = model_name

    def get_model_name(self):
        return f"Fireworks {self.model_name}"

    def load_model(self):
        # For API-based models, we return self since there's no local model to load
        return self

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 16874,
            "temperature": 1,
            "top_p": 1,
            "top_k": 50,
            "repetition_penalty": 1,
            "n": 1,
            "ignore_eos": False,
            "stop": None,
            "stream": False,
        }
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_token}"
        }

        response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return ""

    async def a_generate(self, prompt: str) -> str:
        # Async version - runs the API call in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt) 