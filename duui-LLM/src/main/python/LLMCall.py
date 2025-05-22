from openai import OpenAI
from typing import Union


class OpenAIProcessing:
    def __init__(self, url: str, port: int, seed: int = None, temperature: float = None, api_key: str = None):
        if api_key is None:
            self.openai = OpenAI(
                base_url=f"http://{url}:{port}/v1/",
                api_key="ollama"
            )
        else:
            self.openai = OpenAI(
                # required but ignored
                api_key=api_key,
            )
        self.seed = seed if seed is not None else 0
        self.temperature = temperature if temperature is not None else 1.0
        self.url = url
        self.port = port

    def process_messages(self, model_name, messages):
        return self.openai.chat.completions.create(
                model=model_name,
                seed=self.seed,
                messages=messages,
                temperature=self.temperature,
            ).to_dict()

    def process(self, text: str, model_name: str, system_prompt: Union[None,str]=None, prefix_prompt: Union[None, str]=None, suffix_prompt: Union[str, None]=None):
        prefix = "" if prefix_prompt is None else prefix_prompt
        suffix = "" if suffix_prompt is None else suffix_prompt
        input_prompt = f"{prefix}{text}{suffix}"
        if system_prompt is not None:
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": input_prompt
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": input_prompt
                }
            ]
        return self.openai.chat.completions.create(
            model=model_name,
            seed=self.seed,
            messages=messages,
            temperature=self.temperature,
        ).to_dict()
