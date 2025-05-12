import os
import json
import logging
import requests
from uuid import uuid4
from typing import List
from .duui_api_models import LLMResult, LLMPrompt
from .utils import handle_errors


class Qwen2_5VL:
    def __init__(self,
                 api_url="http://localhost:6659/v1/chat/completions",
                 control_url="http://localhost:6659",
                 model_name="Qwen/Qwen2.5-VL-7B-Instruct",
                 other_model_control_url="http://localhost:6658",
                 logging_level="INFO"):

        self.api_url = api_url
        self.control_url = control_url
        self.model_name = model_name
        self.other_model_control_url = other_model_control_url

        logging.basicConfig(level=logging_level)
        self.logger = logging.getLogger(__name__)

    def _generate_dummy_ref(self):
        return str(uuid4().int % 1_000_000)

    def _build_chat_request(self, messages):
        return {
            "model": self.model_name,
            "messages": messages,
        }

    def _check_and_switch_if_asleep(self):
        r = requests.get(f"{self.control_url}/is_sleep")
        if r.ok and r.json() is True:
            # Wake self and sleep the other model
            requests.post(f"{self.control_url}/wake_up")
            requests.post(f"{self.other_model_control_url}/sleep")

    @handle_errors
    def process_text(self, prompt: LLMPrompt) -> LLMResult:
        self._check_and_switch_if_asleep()
        messages = [{"role": m.role, "content": m.content} for m in prompt.messages]
        body = self._build_chat_request(messages)
        headers = {"Content-Type": "application/json"}

        response = requests.post(self.api_url, data=json.dumps(body), headers=headers)
        response.raise_for_status()
        result = response.json()

        generated_text = result["choices"][0]["message"]["content"]
        prompt_ref = prompt.ref or self._generate_dummy_ref()
        message_ref = self._generate_dummy_ref()

        return LLMResult(
            meta=json.dumps({"response": generated_text, "model_name": self.model_name}),
            prompt_ref=prompt_ref,
            message_ref=message_ref
        )
