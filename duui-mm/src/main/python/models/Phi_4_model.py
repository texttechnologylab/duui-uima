
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

import json
import logging
from uuid import uuid4
from typing import List
import requests
from .duui_api_models import LLMResult, LLMPrompt
from .utils import handle_errors, convert_base64_to_image, convert_base64_to_audio, save_base64_to_temp_file, extract_frames_ffmpeg, extract_audio_base64_ffmpeg


class MicrosoftPhi4:
    def __init__(self,
                 api_url="http://localhost:8000/v1/chat/completions",
                 model_name="microsoft/Phi-4-multimodal-instruct",
                 model_version="0af439b3adb8c23fda473c4f86001dbf9a226021",
                 model_lang="multi",
                 model_source="https://huggingface.co/microsoft/Phi-4-multimodal-instruct",
                 logging_level="INFO"):

        self.api_url = api_url
        self.model_name = model_name
        self.model_version = model_version
        self.model_lang = model_lang
        self.model_source = model_source

        logging.basicConfig(level=logging_level)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _generate_dummy_ref():
        return str(uuid4().int % 1_000_000)

    def _build_chat_request(self, messages):
        return {
            "model": self.model_name,
            "messages": messages,
        }

    @handle_errors
    def process_text(self, prompt: LLMPrompt) -> LLMResult:
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

    @handle_errors
    def process_image(self, image_base64, prompt: LLMPrompt):
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
        prompt_ref = prompt.ref or self._generate_dummy_ref()
        message_ref = self._generate_dummy_ref()

        print(("Response from API:", response_text))
        print(response_text)

        return LLMResult(meta=json.dumps({"response": response_text}), prompt_ref=prompt_ref, message_ref=message_ref)

    @handle_errors
    def process_audio(self, base64_audio, prompt: LLMPrompt):
        audio_url = "data:audio/wav;base64," + base64_audio
        task_prompt = next((m.content for m in reversed(prompt.messages) if m.role == "user"), "Transcribe the audio clip into text.")
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
        prompt_ref = prompt.ref or self._generate_dummy_ref()
        message_ref = self._generate_dummy_ref()

        return LLMResult(meta=json.dumps({"response": response_text}), prompt_ref=prompt_ref, message_ref=message_ref)

    @handle_errors
    def process_video_frames(self, prompt: LLMPrompt, frames: List[str]):
        image_urls = ["data:image/jpeg;base64," + f for f in frames]
        placeholders = ''.join([f"<|image_{i+1}|>" for i in range(len(image_urls))])
        last_msg = next((m.content for m in reversed(prompt.messages) if m.role == "user"), "Summarize the content of the images.")
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

        prompt_ref = prompt.ref or self._generate_dummy_ref()
        message_ref = self._generate_dummy_ref()
        return LLMResult(meta=json.dumps({"response": response_text}), prompt_ref=prompt_ref, message_ref=message_ref)

    @handle_errors
    def process_video_and_audio(self, audio_base64, frames_base64, prompt: LLMPrompt):
        image_urls = ["data:image/jpeg;base64," + f for f in frames_base64]
        audio_url = "data:audio/wav;base64," + audio_base64
        image_placeholders = ''.join([f"<|image_{i+1}|>" for i in range(len(image_urls))])
        prompt_text = next((m.content for m in reversed(prompt.messages) if m.role == "user"), "")
        full_text = f"<|user|>{image_placeholders}<|audio_1|>{prompt_text}<|end|><|assistant|>"

        content = [{"type": "text", "text": full_text}] + [
            {"type": "image_url", "image_url": {"url": url}} for url in image_urls
        ] + [{"type": "audio_url", "audio_url": {"url": audio_url}}]

        body = self._build_chat_request([{"role": "user", "content": content}])
        headers = {"Content-Type": "application/json"}

        response = requests.post(self.api_url, data=json.dumps(body), headers=headers)
        response.raise_for_status()
        result = response.json()
        response_text = result["choices"][0]["message"]["content"]

        prompt_ref = prompt.ref or self._generate_dummy_ref()
        message_ref = self._generate_dummy_ref()
        return LLMResult(meta=json.dumps({"response": response_text}), prompt_ref=prompt_ref, message_ref=message_ref)

    def get_info(self):
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_source": self.model_source,
            "model_lang": self.model_lang
        }


class Phi4ModelVLLM:
    def __init__(self, api_url="http://localhost:8000/v1/chat/completions", model="microsoft/Phi-4-multimodal-instruct"):
        self.api_url = api_url
        self.model = model

    def _generate_dummy_ref(self):
        from uuid import uuid4
        return int(uuid4().int % 1_000_000)

    def _extract_prompt_and_image_url(self, prompt: LLMPrompt):
        image_url = None
        for msg in reversed(prompt.messages):
            if msg.role == "user":
                if isinstance(msg.content, list):
                    for c in msg.content:
                        if c.get("type") == "image_url":
                            image_url = c["image_url"]["url"]
                else:
                    text = msg.content
                break
        return text or "Describe the image.", image_url

    @handle_errors
    def process_image(self, image_base64: str, prompt: LLMPrompt) -> LLMResult:
        text_prompt, image_url = self._extract_prompt_and_image_url(prompt)
        if not image_url:
            raise ValueError("No image URL found in prompt.")

        body = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }]
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post(self.api_url, data=json.dumps(body), headers=headers)
        response.raise_for_status()

        result = response.json()
        generated_text = result["choices"][0]["message"]["content"]

        prompt_ref = prompt.ref or self._generate_dummy_ref()
        message_ref = self._generate_dummy_ref()

        return LLMResult(
            meta=json.dumps({"response": generated_text, "model_name": self.model}),
            prompt_ref=prompt_ref,
            message_ref=message_ref
        )
