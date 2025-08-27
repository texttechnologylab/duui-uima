
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

import base64
import subprocess
import tempfile
import os
import cv2
import numpy as np
from typing import Tuple, List
import json
import logging
from uuid import uuid4
from typing import List
import requests
from .duui_api_models import LLMResult, LLMPrompt
from .utils import handle_errors, decouple_video
import soundfile

import base64
import torch
import json
import tempfile
import logging
import subprocess
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig

class VllmMicrosoftPhi4:
    def __init__(self,
                 api_url="http://localhost:6658/v1/chat/completions",
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

    @staticmethod
    def _check_and_switch_if_asleep():
        r = requests.get("http://localhost:6658/is_sleeping")
        if r.ok and r.json().get('is_sleeping', True) is True:
            requests.post("http://localhost:6659/sleep")
            requests.post("http://localhost:6658/wake_up")


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
        prompt_ref = prompt.ref or self._generate_dummy_ref()
        message_ref = self._generate_dummy_ref()

        # print(("Response from API:", response_text))
        print(response_text)

        return LLMResult(meta=json.dumps({"response": response_text}), prompt_ref=prompt_ref, message_ref=message_ref)

    @handle_errors
    def process_audio(self, base64_audio, prompt: LLMPrompt):
        self._check_and_switch_if_asleep()

        audio_url = "data:audio/wav;base64," + base64_audio
        task_prompt = next((m.content for m in reversed(prompt.messages) if m.role == "user"), "Transcribe the audio clip into text.")
        prompt_text = f"<|user|><|audio_1|>{task_prompt}<|end|><|assistant|>"
        # print(prompt_text)

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

        # print("response is: ", response_text)
        prompt_ref = prompt.ref or self._generate_dummy_ref()
        message_ref = self._generate_dummy_ref()

        return LLMResult(meta=json.dumps({"response": response_text}), prompt_ref=prompt_ref, message_ref=message_ref)

    @handle_errors
    def process_video_frames(self, prompt: LLMPrompt, frames: List[str]):
        self._check_and_switch_if_asleep()

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

        # print("api response is: ", response_text)
        prompt_ref = prompt.ref or self._generate_dummy_ref()
        message_ref = self._generate_dummy_ref()
        return LLMResult(meta=json.dumps({"response": response_text}), prompt_ref=prompt_ref, message_ref=message_ref)

    @handle_errors
    def process_video_and_audio(self, audio_base64, frames_base64, prompt: LLMPrompt):
        self._check_and_switch_if_asleep()

        image_urls = [
            "data:image/jpeg;base64," + f for f in frames_base64 if f.strip()
        ]
        audio_url = (
            "data:audio/wav;base64," + audio_base64 if audio_base64.strip() else None
        )

        image_placeholders = ''.join([f"<|image_{i+1}|>" for i in range(len(image_urls))])
        prompt_text = next((m.content for m in reversed(prompt.messages) if m.role == "user"), "")
        full_text = f"<|user|>{image_placeholders}"
        if audio_url:
            full_text += "<|audio_1|>"
        full_text += f"{prompt_text}<|end|><|assistant|>"

        content = [{"type": "text", "text": full_text}] + \
                  [{"type": "image_url", "image_url": {"url": url}} for url in image_urls]

        if audio_url:
            content.append({"type": "audio_url", "audio_url": {"url": audio_url}})

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
    def process_video(self, videobase64:str, prompt: LLMPrompt):
        self._check_and_switch_if_asleep()

        audio_b64, frames_b64_list = decouple_video(videobase64)
        print("total of ", len(frames_b64_list), " frames")
        # frames_b64_list = frames_b64_list[:3]
        return self.process_video_and_audio(audio_b64, frames_b64_list, prompt)






    def get_info(self):
            return {
                "model_name": self.model_name,
                "model_version": self.model_version,
                "model_source": self.model_source,
                "model_lang": self.model_lang
            }




class TransformersMicrosoftPhi4:
    def __init__(self, model_name="microsoft/Phi-4-multimodal-instruct", logging_level="INFO", version ='0af439b3adb8c23fda473c4f86001dbf9a226021'):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True,
            torch_dtype='auto',
            _attn_implementation='flash_attention_2',
            revision = version,
            device_map='auto'
        )
        self.generation_config = GenerationConfig.from_pretrained(self.model_name, 'generation_config.json')

        logging.basicConfig(level=logging_level)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _generate_dummy_ref():
        return str(uuid4().int % 1_000_000)

    def _build_text(self, prompt: LLMPrompt, media_tokens: str = ""):
        user_prompt = next((m.content for m in reversed(prompt.messages) if m.role == "user"), "What do you see?")
        return f"<|user|>{media_tokens}{user_prompt}<|end|><|assistant|>"

    @handle_errors
    def process_text(self, prompt: LLMPrompt) -> LLMResult:
        text = self._build_text(prompt)
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated = output_ids[0, inputs.input_ids.shape[-1]:]
        result_text = self.processor.decode(generated, skip_special_tokens=True)

        return LLMResult(
            meta=json.dumps({"response": result_text, "model_name": self.model_name}),
            prompt_ref=prompt.ref or self._generate_dummy_ref(),
            message_ref=self._generate_dummy_ref()
        )

    @handle_errors
    def process_image(self, image_base64: str, prompt: LLMPrompt):
        image_data = base64.b64decode(image_base64)
        image = Image.open(tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)).convert("RGB")
        image.fp.write(image_data)
        image.fp.close()

        text = self._build_text(prompt, media_tokens="<|image_1|>")
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=2000, generation_config=self.generation_config)
        result_text = self.processor.decode(output_ids[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True)

        return LLMResult(
            meta=json.dumps({"response": result_text}),
            prompt_ref=prompt.ref or self._generate_dummy_ref(),
            message_ref=self._generate_dummy_ref()
        )

    @handle_errors
    def process_audio(self, audio_base64: str, prompt: LLMPrompt):
        audio_path = tempfile.mktemp(suffix=".wav")
        with open(audio_path, "wb") as f:
            f.write(base64.b64decode(audio_base64))

        text = self._build_text(prompt, media_tokens="<|audio_1|>")
        inputs = self.processor(text=[text], audios=[audio_path], return_tensors="pt").to(self.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=2000, generation_config=self.generation_config)
        result_text = self.processor.decode(output_ids[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True)

        os.remove(audio_path)
        return LLMResult(
            meta=json.dumps({"response": result_text}),
            prompt_ref=prompt.ref or self._generate_dummy_ref(),
            message_ref=self._generate_dummy_ref()
        )

    @handle_errors
    def process_video_frames(self, prompt: LLMPrompt, frames: List[str]):
        images = [Image.open(tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)).convert("RGB") for _ in frames]
        for img, b64 in zip(images, frames):
            img.fp.write(base64.b64decode(b64))
            img.fp.close()

        placeholders = ''.join([f"<|image_{i+1}|>" for i in range(len(images))])
        text = self._build_text(prompt, media_tokens=placeholders)

        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=2000, generation_config=self.generation_config)
        result_text = self.processor.decode(output_ids[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True)

        return LLMResult(
            meta=json.dumps({"response": result_text}),
            prompt_ref=prompt.ref or self._generate_dummy_ref(),
            message_ref=self._generate_dummy_ref()
        )

    def _load_image_from_base64(self, base64_str):
        image_data = base64.b64decode(base64_str)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_file.write(image_data)
            temp_file.flush()
            temp_path = temp_file.name

        return Image.open(temp_path).convert("RGB")


    @handle_errors
    def process_video_and_audio(self, audio_base64, frames_base64, prompt: LLMPrompt):
        images = [self._load_image_from_base64(b64) for b64 in frames_base64]

        audio_path = None
        if audio_base64.strip():
            audio_path = tempfile.mktemp(suffix=".wav")
            with open(audio_path, "wb") as f:
                f.write(base64.b64decode(audio_base64))

        media_tokens = ''.join([f"<|image_{i+1}|>" for i in range(len(images))])
        if audio_path:
            media_tokens += "<|audio_1|>"

        audio = soundfile.read(audio_path)
        text = self._build_text(prompt, media_tokens=media_tokens)
        inputs = self.processor(
            text=[text], images=images, audios=[audio] if audio_path else None, return_tensors="pt"
        ).to(self.device)

        inputs["num_logits_to_keep"] = torch.tensor([50], device=self.device)
        inputs = {k: v for k, v in inputs.items() if v is not None and v.numel() > 0}

        output_ids = self.model.generate(**inputs, max_new_tokens=2000, generation_config=self.generation_config)
        result_text = self.processor.decode(output_ids[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

        if audio_path:
            os.remove(audio_path)

        return LLMResult(
            meta=json.dumps({"response": result_text}),
            prompt_ref=prompt.ref or self._generate_dummy_ref(),
            message_ref=self._generate_dummy_ref()
        )

    @handle_errors
    def process_video(self, videobase64: str, prompt: LLMPrompt):
        audio_b64, frames_b64_list = decouple_video(videobase64)
        frames_b64_list = frames_b64_list[:3]
        # TODO: remove this
        return self.process_video_and_audio(audio_b64, frames_b64_list, prompt)

    def get_info(self):
        return {
            "model_name": self.model_name,
            "model_source": f"https://huggingface.co/{self.model_name}",
            "model_lang": "multi",
            "implementation": "transformers"
        }
