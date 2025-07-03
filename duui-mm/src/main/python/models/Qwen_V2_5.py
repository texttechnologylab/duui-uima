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
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import base64
import json
import logging

import numpy as np
class VllmQwen2_5VL:
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

        return LLMResult(meta=json.dumps({"response": "Audio Alone is not supported in QenV2.5L"}),
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

        print("prompt_text ", prompt_text)

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

        print("result ", result)
        response_text = result["choices"][0]["message"]["content"]

        return LLMResult(
            meta=json.dumps({"response": response_text}),
            prompt_ref=prompt.ref or self._generate_dummy_ref(),
            message_ref=self._generate_dummy_ref()
        )



class BaseQwen2_5VL:
    def __init__(self,
                 model_name: str,
                 version:str,
                 logging_level: str = "INFO",
                 torch_dtype: torch.dtype = torch.bfloat16,
                 attn_implementation: str = "flash_attention_2"):

        self.model_name = model_name
        self.revision = version
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging_level)

        self._load_transformers_model(torch_dtype, attn_implementation)

    def _load_transformers_model(self, torch_dtype, attn_implementation):
        """Load the model and processor using the Transformers library."""
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            revision=self.revision,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def _generate_dummy_ref(self):
        return str(uuid4().int % 1_000_000)

    @handle_errors
    def process_text(self, prompt: LLMPrompt) -> LLMResult:
        return self._process_text_with_transformers(prompt)

    def _process_text_with_transformers(self, prompt: LLMPrompt) -> LLMResult:
        messages = [{"role": m.role, "content": m.content} for m in prompt.messages]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], padding=True, return_tensors="pt").to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return LLMResult(
            meta=json.dumps({"response": output_text, "model_name": self.model_name}),
            prompt_ref=prompt.ref or self._generate_dummy_ref(),
            message_ref=self._generate_dummy_ref()
        )

    @handle_errors
    def process_image(self, image_base64: str, prompt: LLMPrompt) -> LLMResult:
        image_url = "data:image/jpeg;base64," + image_base64
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_url},
                    {"type": "text", "text": next((m.content for m in reversed(prompt.messages) if m.role == "user"), "Describe this image.")},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=512)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return LLMResult(
            meta=json.dumps({"response": output_text}),
            prompt_ref=prompt.ref or self._generate_dummy_ref(),
            message_ref=self._generate_dummy_ref()
        )

    @handle_errors
    def process_audio(self, base64_audio, prompt: LLMPrompt):
        return LLMResult(
            meta=json.dumps({"response": "Audio Alone is not supported in QenV2.5L"}),
            prompt_ref=prompt.ref or self._generate_dummy_ref(),
            message_ref=self._generate_dummy_ref()
        )

    @handle_errors
    def process_video_frames(self, prompt: LLMPrompt, frames: List[str]) -> LLMResult:
        # Create messages with video frames
        frame_messages = [
            {
                "role": "user",
                "content": [
                    *[{ "type": "image", "image": f"data:image/jpeg;base64,{frame}" } for frame in frames],
                    {"type": "text", "text": next((m.content for m in reversed(prompt.messages) if m.role == "user"), "Summarize the content.")}
                ],
            }
        ]

        text = self.processor.apply_chat_template(frame_messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(frame_messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return LLMResult(
            meta=json.dumps({"response": output_text}),
            prompt_ref=prompt.ref or self._generate_dummy_ref(),
            message_ref=self._generate_dummy_ref()
        )

    @handle_errors
    def process_video_and_audio(self, audio_base64, frames_base64, prompt: LLMPrompt) -> LLMResult:
        # Create messages with video frames and audio
        video_audio_messages = [
            {
                "role": "user",
                "content": [
                    *[{ "type": "image", "image": f"data:image/jpeg;base64,{frame}" } for frame in frames_base64],
                    {"type": "audio", "audio": f"data:audio/wav;base64,{audio_base64}"},
                    {"type": "text", "text": next((m.content for m in reversed(prompt.messages) if m.role == "user"), "")}
                ],
            }
        ]

        text = self.processor.apply_chat_template(video_audio_messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(video_audio_messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return LLMResult(
            meta=json.dumps({"response": output_text}),
            prompt_ref=prompt.ref or self._generate_dummy_ref(),
            message_ref=self._generate_dummy_ref()
        )

    @handle_errors
    def process_video(self, video_base64: str, prompt: LLMPrompt) -> LLMResult:
        import tempfile, base64, os, json
        import subprocess

        # Save base64 video to temporary file
        video_path = tempfile.mktemp(suffix=".mp4")
        with open(video_path, "wb") as f:
            f.write(base64.b64decode(video_base64))

        # Get user prompt
        user_prompt = next((m.content for m in reversed(prompt.messages) if m.role == "user"), "Analyze the video.")

        # Start message with video path (local file URL format for Qwen 2.5)
        content = [
            {
                "type": "video",
                "video": f"file://{video_path}",
                "fps": 1.0,  # or a different value depending on desired granularity
                "max_pixels": 360 * 420,  # optional: limits resolution
            },
            {"type": "text", "text": user_prompt},
        ]

        # Add audio if available
        audio_path = None
        if video_has_audio(video_path):
            audio_path = tempfile.mktemp(suffix=".wav")
            audio_cmd = [
                "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1", audio_path,
                "-hide_banner", "-loglevel", "error"
            ]
            subprocess.run(audio_cmd, check=True)

            with open(audio_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
                audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
            content.append({"type": "audio", "audio": f"data:audio/wav;base64,{audio_base64}"})

        # Prepare messages
        messages = [{"role": "user", "content": content}]

        # Tokenize prompt
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision and video inputs
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        # Prepare model input
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,  # includes fps
        ).to("cuda")

        # Generate output
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Cleanup
        os.remove(video_path)
        if audio_path:
            os.remove(audio_path)

        # Return result
        return LLMResult(
            meta=json.dumps({"response": output_text}),
            prompt_ref=prompt.ref or self._generate_dummy_ref(),
            message_ref=str(self._generate_dummy_ref())
        )


class Qwen2_5_VL_7B_Instruct(BaseQwen2_5VL):
    def __init__(self, version: str,  logging_level: str = "INFO"):
        super().__init__(
            model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            logging_level=logging_level,
            version = version
        )

class Qwen2_5_VL_7B_Instruct_AWQ(BaseQwen2_5VL):
    def __init__(self,  version: str, logging_level: str = "INFO"):
        super().__init__(
            model_name="Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
            logging_level=logging_level,
            version = version
        )

class Qwen2_5_VL_3B_Instruct(BaseQwen2_5VL):
    def __init__(self,  version: str, logging_level: str = "INFO"):
        super().__init__(
            model_name="Qwen/Qwen2.5-VL-3B-Instruct",
            logging_level=logging_level,
            version = version
        )

class Qwen2_5_VL_3B_Instruct_AWQ(BaseQwen2_5VL):
    def __init__(self,  version: str, logging_level: str = "INFO"):
        super().__init__(
            model_name="Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
            logging_level=logging_level,
            version = version
        )

class Qwen2_5_VL_32B_Instruct(BaseQwen2_5VL):
    def __init__(self,  version: str, logging_level: str = "INFO"):
        super().__init__(
            model_name="Qwen/Qwen2.5-VL-32B-Instruct",
            logging_level=logging_level,
            version = version
        )

class Qwen2_5_VL_32B_Instruct_AWQ(BaseQwen2_5VL):
    def __init__(self,  version: str, logging_level: str = "INFO"):
        super().__init__(
            model_name="Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
            logging_level=logging_level,
            version = version
        )

class Qwen2_5_VL_72B_Instruct(BaseQwen2_5VL):
    def __init__(self,  version: str, logging_level: str = "INFO"):
        super().__init__(
            model_name="Qwen/Qwen2.5-VL-72B-Instruct",
            logging_level=logging_level,
            version = version
        )

class Qwen2_5_VL_72B_Instruct_AWQ(BaseQwen2_5VL):
    def __init__(self,  version: str, logging_level: str = "INFO"):
        super().__init__(
            model_name="Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
            logging_level=logging_level,
            version = version
        )
