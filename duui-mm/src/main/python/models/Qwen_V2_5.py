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
                 model_name="Qwen/Qwen2.5-VL-7B-Instruct",
                 logging_level="INFO"):

        self.api_url = api_url
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging_level)

    def _generate_dummy_ref(self):
        return str(uuid4().int % 1_000_000)

    def _build_chat_request(self, messages):
        return {
            "model": self.model_name,
            "messages": messages,
        }

    @staticmethod
    def _check_and_switch_if_asleep():
        r = requests.get("http://localhost:6659/is_sleeping")
        if r.ok and r.json().get('is_sleeping', True) is True:
            requests.post("http://localhost:6658/sleep")
            requests.post("http://localhost:6659/wake_up")

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
        return LLMResult(
            meta=json.dumps({"response": generated_text, "model_name": self.model_name}),
            prompt_ref=prompt.ref or self._generate_dummy_ref(),
            message_ref=self._generate_dummy_ref()
        )

    @handle_errors
    def process_image(self, image_base64, prompt: LLMPrompt):
        self._check_and_switch_if_asleep()
        image_url = "data:image/jpeg;base64," + image_base64
        last_msg = next((m.content for m in reversed(prompt.messages) if m.role == "user"), "Describe the image.")
        content = [
            {"type": "text", "text": f"<|user|><|image_1|>{last_msg}<|end|><|assistant|>"},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
        body = self._build_chat_request([{"role": "user", "content": content}])
        headers = {"Content-Type": "application/json"}
        response = requests.post(self.api_url, data=json.dumps(body), headers=headers)
        response.raise_for_status()
        result = response.json()
        response_text = result["choices"][0]["message"]["content"]
        return LLMResult(meta=json.dumps({"response": response_text}),
                         prompt_ref=prompt.ref or self._generate_dummy_ref(),
                         message_ref=self._generate_dummy_ref())

    @handle_errors
    def process_audio(self, base64_audio, prompt: LLMPrompt):
        self._check_and_switch_if_asleep()
        audio_url = "data:audio/wav;base64," + base64_audio
        task_prompt = next((m.content for m in reversed(prompt.messages) if m.role == "user"), "Transcribe the audio.")
        prompt_text = f"<|user|><|audio_1|>{task_prompt}<|end|><|assistant|>"
        content = [
            {"type": "text", "text": prompt_text},
            {"type": "audio_url", "audio_url": {"url": audio_url}}
        ]
        body = self._build_chat_request([{"role": "user", "content": content}])
        headers = {"Content-Type": "application/json"}
        response = requests.post(self.api_url, data=json.dumps(body), headers=headers)
        response.raise_for_status()
        result = response.json()
        response_text = result["choices"][0]["message"]["content"]
        return LLMResult(meta=json.dumps({"response": response_text}),
                         prompt_ref=prompt.ref or self._generate_dummy_ref(),
                         message_ref=self._generate_dummy_ref())

    @handle_errors
    def process_video_frames(self, prompt: LLMPrompt, frames: List[str]):
        self._check_and_switch_if_asleep()
        image_urls = ["data:image/jpeg;base64," + f for f in frames]
        placeholders = ''.join([f"<|image_{i+1}|>" for i in range(len(image_urls))])
        last_msg = next((m.content for m in reversed(prompt.messages) if m.role == "user"), "Summarize the content.")
        full_text = f"<|user|>{placeholders}{last_msg}<|end|><|assistant|>"
        content = [{"type": "text", "text": full_text}] + [
            {"type": "image_url", "image_url": {"url": url}} for url in image_urls
        ]
        body = self._build_chat_request([{"role": "user", "content": content}])
        headers = {"Content-Type": "application/json"}
        response = requests.post(self.api_url, data=json.dumps(body), headers=headers)
        response.raise_for_status()
        result = response.json()
        response_text = result["choices"][0]["message"]["content"]
        return LLMResult(meta=json.dumps({"response": response_text}),
                         prompt_ref=prompt.ref or self._generate_dummy_ref(),
                         message_ref=self._generate_dummy_ref())

    @handle_errors
    def process_video_and_audio(self, audio_base64, frames_base64, prompt: LLMPrompt):
        self._check_and_switch_if_asleep()
        image_urls = ["data:image/jpeg;base64," + f for f in frames_base64]
        audio_url = "data:audio/wav;base64," + audio_base64
        image_placeholders = ''.join([f"<|image_{i+1}|>" for i in range(len(image_urls))])
        prompt_text = next((m.content for m in reversed(prompt.messages) if m.role == "user"), "")
        full_text = f"<|user|>{image_placeholders}<|audio_1|>{prompt_text}<|end|><|assistant|>"
        content = [{"type": "text", "text": full_text}] + \
                  [{"type": "image_url", "image_url": {"url": url}} for url in image_urls] + \
                  [{"type": "audio_url", "audio_url": {"url": audio_url}}]
        body = self._build_chat_request([{"role": "user", "content": content}])
        headers = {"Content-Type": "application/json"}
        response = requests.post(self.api_url, data=json.dumps(body), headers=headers)
        response.raise_for_status()
        result = response.json()
        response_text = result["choices"][0]["message"]["content"]
        return LLMResult(meta=json.dumps({"response": response_text}),
                         prompt_ref=prompt.ref or self._generate_dummy_ref(),
                         message_ref=self._generate_dummy_ref())


    @handle_errors
    def process_video(self, video_base64: str, prompt: LLMPrompt):
        self._check_and_switch_if_asleep()

        video_url = "data:video/mp4;base64," + video_base64
        prompt_text = next((m.content for m in reversed(prompt.messages) if m.role == "user"), "Analyze the video.")

        full_text = f"<|user|><|video_1|>{prompt_text}<|end|><|assistant|>"
        content = [
            {"type": "text", "text": full_text},
            {"type": "video_url", "video_url": {"url": video_url}}
        ]

        body = self._build_chat_request([{"role": "user", "content": content}])
        headers = {"Content-Type": "application/json"}

        response = requests.post(self.api_url, data=json.dumps(body), headers=headers)
        response.raise_for_status()
        result = response.json()

        response_text = result["choices"][0]["message"]["content"]

        return LLMResult(
            meta=json.dumps({"response": response_text}),
            prompt_ref=prompt.ref or self._generate_dummy_ref(),
            message_ref=self._generate_dummy_ref()
        )
