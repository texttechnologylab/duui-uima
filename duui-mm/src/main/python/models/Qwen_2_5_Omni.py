import torch
import json
import logging
import base64
from uuid import uuid4
import soundfile as sf
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from .duui_api_models import LLMPrompt, LLMResult
from .utils import handle_errors, convert_base64_to_audio, convert_base64_to_image, convert_base64_to_video, decouple_video


class QwenOmni3B:
    def __init__(self,
                 model_name="Qwen/Qwen2.5-Omni-3B",
                 device="auto",
                 logging_level="INFO"):
        self.model_name = model_name
        self.device = device

        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device,
            attn_implementation="flash_attention_2"
        )

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging_level)

    def _generate_dummy_ref(self):
        return int(uuid4().int % 1_000_000)

    def _extract_user_text(self, prompt: LLMPrompt):
        for msg in reversed(prompt.messages):
            if msg.role == "user" and isinstance(msg.content, str):
                return msg.content
        return ""

    def _build_conversation(self, text=None, image=None, audio=None, video=None):
        content = []
        if text:
            content.append({"type": "text", "text": text})
        if image:
            content.append({"type": "image", "image": image})
        if audio:
            content.append({"type": "audio", "audio": audio})
        if video:
            content.append({"type": "video", "video": video})


        return [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are Qwen, a multimodal assistant that can understand and reason over text, image, audio, and video."}
                    ]
                },
                {
                    "role": "user",
                    "content": content
                }
            ]

    def _run_model(self, conversation, use_audio_in_video):
        raw_text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)

        inputs = self.processor(
            text=raw_text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio_in_video
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        # Generate output
        text_ids, audio_out = self.model.generate(**inputs, use_audio_in_video=use_audio_in_video)
        response_text = self.processor.batch_decode(
            text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return response_text, audio_out

    @handle_errors
    def process_text(self, prompt: LLMPrompt) -> LLMResult:
        user_text = self._extract_user_text(prompt)
        conversation = self._build_conversation(text=user_text)
        response_text, _ = self._run_model(conversation, use_audio_in_video=False)

        return self._make_result(prompt, response_text)

    @handle_errors
    def process_image(self, image_b64: str, prompt: LLMPrompt):
        user_text = self._extract_user_text(prompt)
        image = convert_base64_to_image(image_b64)
        conversation = self._build_conversation(text=user_text, image=image)
        response_text, _ = self._run_model(conversation, use_audio_in_video=False)

        return self._make_result(prompt, response_text)

    @handle_errors
    def process_audio(self, audio_b64: str, prompt: LLMPrompt):
        user_text = self._extract_user_text(prompt)
        audio = convert_base64_to_audio(audio_b64)
        conversation = self._build_conversation(text=user_text, audio=audio)
        response_text, _ = self._run_model(conversation, use_audio_in_video=True)

        return self._make_result(prompt, response_text)

    @handle_errors
    def process_video(self, video_b64: str, prompt: LLMPrompt):
        # audio_b64, frames_b64 = decouple_video(video_b64)
        user_text = self._extract_user_text(prompt)

        conversation = self._build_conversation(text=user_text, video=video_b64)
        response_text, _ = self._run_model(conversation, use_audio_in_video=True)

        return self._make_result(prompt, response_text)

    @handle_errors
    def process_video_and_audio(self, video_b64: str, prompt: LLMPrompt):
        audio_b64, frames_b64 = decouple_video(video_b64)
        audio = convert_base64_to_audio(audio_b64)
        frames = [convert_base64_to_image(f) for f in frames_b64[:3]]
        user_text = self._extract_user_text(prompt)

        # Just use first few frames as images (no true video object here)
        # Alternative: treat as video if your processor supports real video input
        conversation = self._build_conversation(text=user_text, image=frames[0], audio=audio)
        response_text, _ = self._run_model(conversation, use_audio_in_video=True)

        return self._make_result(prompt, response_text)

    def _make_result(self, prompt: LLMPrompt, response_text: str) -> LLMResult:
        prompt_ref = prompt.ref or self._generate_dummy_ref()
        message_ref = self._generate_dummy_ref()

        return LLMResult(
            meta=json.dumps({"response": response_text, "model_name": self.model_name}),
            prompt_ref=prompt_ref,
            message_ref=message_ref
        )

    def get_info(self):
        return {
            "model_name": self.model_name,
            "model_version": "unknown",
            "model_source": "https://huggingface.co/" + self.model_name,
            "model_lang": "multi"
        }
