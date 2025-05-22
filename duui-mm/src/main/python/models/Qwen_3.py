import os
import json
import logging
import requests
from uuid import uuid4
import tempfile
from .duui_api_models import LLMResult, LLMPrompt
from .utils import handle_errors, extract_frames_ffmpeg, video_has_audio
import torch
from typing import List, Optional
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer
import base64
import logging

class BaseQwen3:
    def __init__(self,
                 model_name: str,
                 version: str,
                 logging_level: str = "INFO",
                 torch_dtype: torch.dtype = torch.bfloat16):

        self.model_name = model_name
        self.revision = version
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging_level)

        self._load_transformers_model(torch_dtype)

    def _load_transformers_model(self, torch_dtype):
        """Load the model and tokenizer using the Transformers library."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map="auto"
        )

    def _generate_dummy_ref(self):
        return str(uuid4().int % 1_000_000)

    @handle_errors
    def process_text(self, prompt: LLMPrompt) -> LLMResult:
        return self._process_text_with_transformers(prompt)

    def _process_text_with_transformers(self, prompt: LLMPrompt) -> LLMResult:
        messages = [{"role": m.role, "content": m.content} for m in prompt.messages]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=32768)
        output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()

        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        return LLMResult(
            meta=json.dumps({"response": content, "model_name": self.model_name, "thinking_content": thinking_content}),
            prompt_ref=prompt.ref or self._generate_dummy_ref(),
            message_ref=self._generate_dummy_ref()
        )

    @handle_errors
    def process_image(self, image_base64: str, prompt: LLMPrompt) -> LLMResult:
        return LLMResult(
            meta=json.dumps({"response": "Image processing is not supported in Qwen3"}),
            prompt_ref=prompt.ref or self._generate_dummy_ref(),
            message_ref=self._generate_dummy_ref()
        )

    @handle_errors
    def process_audio(self, base64_audio, prompt: LLMPrompt) -> LLMResult:
        return LLMResult(
            meta=json.dumps({"response": "Audio processing is not supported in Qwen3"}),
            prompt_ref=prompt.ref or self._generate_dummy_ref(),
            message_ref=self._generate_dummy_ref()
        )

    @handle_errors
    def process_video_frames(self, prompt: LLMPrompt, frames: List[str]) -> LLMResult:
        return LLMResult(
            meta=json.dumps({"response": "Video frame processing is not supported in Qwen3"}),
            prompt_ref=prompt.ref or self._generate_dummy_ref(),
            message_ref=self._generate_dummy_ref()
        )

    @handle_errors
    def process_video_and_audio(self, audio_base64, frames_base64, prompt: LLMPrompt) -> LLMResult:
        return LLMResult(
            meta=json.dumps({"response": "Video and audio processing is not supported in Qwen3"}),
            prompt_ref=prompt.ref or self._generate_dummy_ref(),
            message_ref=self._generate_dummy_ref()
        )

    @handle_errors
    def process_video(self, video_base64: str, prompt: LLMPrompt) -> LLMResult:
        return LLMResult(
            meta=json.dumps({"response": "Video processing is not supported in Qwen3"}),
            prompt_ref=prompt.ref or self._generate_dummy_ref(),
            message_ref=self._generate_dummy_ref()
        )

class Qwen3_32B(BaseQwen3):
    def __init__(self, version: str, logging_level: str = "INFO"):
        super().__init__(
            model_name="Qwen/Qwen3-32B",
            version=version,
            logging_level=logging_level
        )

class Qwen3_14B(BaseQwen3):
    def __init__(self, version: str, logging_level: str = "INFO"):
        super().__init__(
            model_name="Qwen/Qwen3-14B",
            version=version,
            logging_level=logging_level
        )

class Qwen3_8B(BaseQwen3):
    def __init__(self, version: str, logging_level: str = "INFO"):
        super().__init__(
            model_name="Qwen/Qwen3-8B",
            version=version,
            logging_level=logging_level
        )

class Qwen3_4B(BaseQwen3):
    def __init__(self, version: str, logging_level: str = "INFO"):
        super().__init__(
            model_name="Qwen/Qwen3-4B",
            version=version,
            logging_level=logging_level
        )

class Qwen3_1_7B(BaseQwen3):
    def __init__(self, version: str, logging_level: str = "INFO"):
        super().__init__(
            model_name="Qwen/Qwen3-1.7B",
            version=version,
            logging_level=logging_level
        )

class Qwen3_0_6B(BaseQwen3):
    def __init__(self, version: str, logging_level: str = "INFO"):
        super().__init__(
            model_name="Qwen/Qwen3-0.6B",
            version=version,
            logging_level=logging_level
        )
