from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import torch
import json
from .utils import convert_base64_to_image, handle_errors
from .duui_api_models import LLMResult, LLMPrompt
import json
import requests


class MolmoBaseModel:
    def __init__(self, model_name: str, revision: str = None, device: str = "auto", logging_level="INFO"):
        self.model_name = model_name
        self.revision = revision
        self.device = device

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            revision=revision,
            torch_dtype="auto",
            device_map=device
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            revision=revision,
            torch_dtype="auto",
            device_map=device
        )

        self.generation_config = GenerationConfig(
            max_new_tokens=200,
            stop_strings=["<|endoftext|>"]
        )

    def _generate_dummy_ref(self):
        from uuid import uuid4
        return int(uuid4().int % 1_000_000)

    def _extract_prompt_text(self, prompt: LLMPrompt) -> str:
        return next((m.content for m in reversed(prompt.messages) if m.role == "user"), "Describe the image.")

    @handle_errors
    def process_image(self, image_base64: str, prompt: LLMPrompt) -> LLMResult:
        image = convert_base64_to_image(image_base64)
        text_prompt = self._extract_prompt_text(prompt)

        inputs = self.processor.process(images=[image], text=text_prompt)
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        output = self.model.generate_from_batch(inputs, self.generation_config, tokenizer=self.processor.tokenizer)
        generated_tokens = output[0, inputs["input_ids"].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        prompt_ref = prompt.ref or self._generate_dummy_ref()
        last_user_msg = next((m for m in reversed(prompt.messages) if m.role == "user"), None)
        message_ref = last_user_msg.ref if last_user_msg and last_user_msg.ref else self._generate_dummy_ref()

        return LLMResult(
            meta=json.dumps({"response": generated_text, "model_name": self.model_name}),
            prompt_ref=prompt_ref,
            message_ref=message_ref
        )


class MolmoE1BModel(MolmoBaseModel):
    def __init__(self, revision=None, device="auto"):
        super().__init__('allenai/MolmoE-1B-0924', revision=revision, device=device)


class Molmo7BOModel(MolmoBaseModel):
    def __init__(self, revision=None, device="auto"):
        super().__init__('allenai/Molmo-7B-O-0924', revision=revision, device=device)


class Molmo7BDModel(MolmoBaseModel):
    def __init__(self, revision=None, device="auto"):
        super().__init__('allenai/Molmo-7B-D-0924', revision=revision, device=device)


class Molmo72BModel(MolmoBaseModel):
    def __init__(self, revision=None, device="auto"):
        super().__init__('allenai/Molmo-72B-0924', revision=revision, device=device)



class Molmo7BDModelVLLM:
    def __init__(self,
                 api_url="http://localhost:6650/v1/chat/completions",
                 model_name="allenai/Molmo-7B-D-0924",
                 model_version="ac032b93b84a7f10c9578ec59f9f20ee9a8990a2",
                 model_lang="multi",
                 model_source="https://huggingface.co/allenai/Molmo-7B-D-0924"):
        self.api_url = api_url
        self.model_name = model_name
        self.model_version = model_version
        self.model_lang = model_lang
        self.model_source = model_source

        self.sleep_check_url = self.api_url.replace("/v1/chat/completions", "/is_sleep")
        self.wake_url = self.api_url.replace("/v1/chat/completions", "/wake_up")
        self.sleep_url = self.api_url.replace("/v1/chat/completions", "/sleep")

    def _generate_dummy_ref(self):
        from uuid import uuid4
        return str(uuid4().int % 1_000_000)

    def _extract_prompt_text(self, prompt: LLMPrompt) -> str:
        return next((m.content for m in reversed(prompt.messages) if m.role == "user"), "Describe the image.")

    def _ensure_awake(self):
        try:
            sleep_status = requests.get(self.sleep_check_url).json()
            if sleep_status.get("is_sleep", False):
                requests.post(self.wake_url)
        except Exception as e:
            print(f"Warning: Failed to check or wake model: {e}")

    def _put_others_to_sleep(self, others: list):
        for other_url in others:
            try:
                requests.post(other_url)
            except Exception as e:
                print(f"Warning: Failed to sleep other model at {other_url}: {e}")

    def _build_chat_request(self, messages):
        return {
            "model": self.model_name,
            "messages": messages,
        }

    @handle_errors
    def process_image(self, image_base64: str, prompt: LLMPrompt, other_sleep_urls: list = None) -> LLMResult:
        self._ensure_awake()
        if other_sleep_urls:
            self._put_others_to_sleep(other_sleep_urls)

        image_url = "data:image/jpeg;base64," + image_base64
        user_text = self._extract_prompt_text(prompt)

        content = [
            {"type": "text", "text": f"<|user|><|image_1|>{user_text}<|end|><|assistant|>"},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]

        messages = [{"role": "user", "content": content}]
        body = self._build_chat_request(messages)

        headers = {"Content-Type": "application/json"}
        response = requests.post(self.api_url, data=json.dumps(body), headers=headers)
        response.raise_for_status()
        result = response.json()

        response_text = result["choices"][0]["message"]["content"]
        print("response is ", response_text)
        prompt_ref = prompt.ref or self._generate_dummy_ref()
        last_user_msg = next((m for m in reversed(prompt.messages) if m.role == "user"), None)
        message_ref = last_user_msg.ref if last_user_msg and last_user_msg.ref else self._generate_dummy_ref()

        return LLMResult(
            meta=json.dumps({"response": response_text, "model_name": self.model_name}),
            prompt_ref=prompt_ref,
            message_ref=message_ref
        )

    def get_info(self):
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_source": self.model_source,
            "model_lang": self.model_lang
        }
