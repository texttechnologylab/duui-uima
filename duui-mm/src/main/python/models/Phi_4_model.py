from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, AutoModelForVision2Seq

import logging

from .utils import convert_base64_to_image, convert_base64_to_audio, save_base64_to_temp_file, handle_errors, extract_audio_base64_ffmpeg, extract_frames_ffmpeg
from .duui_api_models import LLMResult, LLMPrompt
import json
from typing import List, Optional

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''


class MicrosoftPhi4:
    """
    This class is a model for the Microsoft Phi-4 stock price prediction.
    It uses a simple linear regression model to predict the stock price based on the previous day's price.
    """

    def __init__(self,
                 model_name = "microsoft/Phi-4-multimodal-instruct",
                 device='cpu',
                 model_version = "0af439b3adb8c23fda473c4f86001dbf9a226021",
                 model_lang = "multi",
                 model_source= "https://huggingface.co/microsoft/Phi-4-multimodal-instruct",
                 logging_level = "INFO"):

        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     trust_remote_code=True,
                                                     _attn_implementation='flash_attention_2',
                                                     torch_dtype="auto",
                                                    revision=model_version).eval()
        self.model.to(device)

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.generation_config = GenerationConfig.from_pretrained(model_name)
        self.generation_args = {"max_new_tokens": 2048, "temperature": 0.5, "do_sample": True}



        logging.basicConfig(level=logging_level)
        self.logger = logging.getLogger(__name__)

        self.device, self.model_name, self.model_source, self.model_version, self.model_lang = device, model_name, model_source, model_version, model_lang


    @handle_errors
    def process(self, mode, prompt , **kwargs):
        """
        This method processes the input based on the mode and returns the output.
        The mode can be 'text', 'image', 'audio', or 'video'.
        """
        if mode == "text":
            return self.process_text(prompt)
        elif mode == "image":
            return self.process_image(kwargs['image'], prompt)
        elif mode == "audio":
            return self.process_audio(kwargs['audio'], prompt)
        elif mode == "video_frames":
            return self.process_video_frames(prompt, kwargs['frames'])
        elif mode == "video_and_audio":
            return self.process_video_and_audio(kwargs['audio'], kwargs['frames'], prompt)

    @handle_errors
    def process_image(self, image_base64, prompt: LLMPrompt):
        """
        This method takes an image and returns the processed image.
        The image should be a base64 image.
        """
        # convert image from base64 to PIL
        image = convert_base64_to_image(image_base64)
        messages, prompt_ref, message_ref = self._extract_messages_and_refs(prompt)

        # Construct full user message with image tag and last message
        image_tag = "<|image_0|>"
        last_msg = next((m.content for m in reversed(prompt.messages) if m.role == "user"), "")
        full_prompt = f"{image_tag}{last_msg}"

        chat_prompt = self.processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": full_prompt}],
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(text=chat_prompt, images=image, return_tensors="pt").to(self.device)


        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.5,
            do_sample=True,
            generation_config=self.generation_config,
        )

        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return LLMResult(meta=json.dumps({"response": response}), prompt_ref=prompt_ref, message_ref=message_ref)


    @staticmethod
    def _generate_dummy_ref():
        from uuid import uuid4
        return int(uuid4().int % 1_000_000)

    def _extract_messages_and_refs(self, prompt: LLMPrompt):
        messages = [{"role": m.role, "content": m.content} for m in prompt.messages]
        prompt_ref = prompt.ref or self._generate_dummy_ref()
        last_user_msg = next((m for m in reversed(prompt.messages) if m.role == "user"), None)
        message_ref = last_user_msg.ref if last_user_msg and last_user_msg.ref else self._generate_dummy_ref()
        return messages, prompt_ref, message_ref


    @handle_errors
    def process_text(self, prompt: LLMPrompt) -> LLMResult:
        """
        Process a structured LLMPrompt and return an LLMResult with CAS refs and response metadata.
        """
        # Extract messages as needed for the tokenizer
        messages = [{"role": m.role, "content": m.content} for m in prompt.messages]

        # Apply chat template
        chat_prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize and move to device
        inputs = self.processor(chat_prompt, images=None, return_tensors="pt")
        inputs = inputs.to(self.device)

        # Generate response
        generated_ids = self.model.generate(
            **inputs,
            **self.generation_args,
            generation_config=self.generation_config,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Determine prompt and message refs (fallback to dummy refs if not provided)
        prompt_ref = prompt.ref or self._generate_dummy_ref()
        last_user_msg = next((m for m in reversed(prompt.messages) if m.role == "user"), None)
        message_ref = last_user_msg.ref if last_user_msg and last_user_msg.ref else self._generate_dummy_ref()

        # Return structured result
        return LLMResult(
            meta=json.dumps({
                "response": generated_text,
                "model_name": self.model_name
            }),
            prompt_ref=prompt_ref,
            message_ref=message_ref
        )

    @handle_errors
    def process_audio(self, base64_audio, prompt: LLMPrompt) -> LLMResult:
        """
        Processes a base64-encoded audio input and returns generated text based on the user's prompt.
        """
        waveform, sample_rate = convert_base64_to_audio(base64_audio)
        messages, prompt_ref, message_ref = self._extract_messages_and_refs(prompt)

        # Simple audio prompt formatting using the last user message
        last_msg = next((m.content for m in reversed(prompt.messages) if m.role == "user"), "")
        user_tag = "<|user|>"
        assistant_tag = "<|assistant|>"
        end_tag = "<|end|>"
        audio_tag = "<|audio_1|>"

        full_prompt = f"{user_tag}{audio_tag}{last_msg}{end_tag}{assistant_tag}"

        inputs = self.processor(
            text=full_prompt,
            audios=[(waveform, sample_rate)],
            return_tensors='pt'
        ).to(self.device)


        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=1200,
            temperature=0.5,
            do_sample=True,
            generation_config=self.generation_config,
        )
        generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]

        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return LLMResult(meta=json.dumps({"response": response}), prompt_ref=prompt_ref, message_ref=message_ref)


    @handle_errors
    def process_video_frames(self, prompt: LLMPrompt, frames: List[str]):
        """
        This method takes a list of base64-encoded images (frames of a video)
        and a user prompt, and returns the model's generated response.

        Args:
            prompt (str): Instruction for the model based on the video frames.
            frames (List[str]): List of base64-encoded images.

        Returns:
            str: Model-generated response based on the visual content and prompt.
        """
        # Convert base64 frames to PIL images
        images = [convert_base64_to_image(f) for f in frames]
        placeholder = "".join(f"<|image_{i}|>" for i in range(len(images)))

        messages, prompt_ref, message_ref = self._extract_messages_and_refs(prompt)
        last_msg = next((m.content for m in reversed(prompt.messages) if m.role == "user"), "")

        full_msg = f"{placeholder},Please summarize the content of these images as they are a series of frames representing a video. Based on the summary, {last_msg}"

        chat_prompt = self.processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": full_msg}],
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(text=chat_prompt, images=images, return_tensors='pt').to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=4096,
            temperature=0.5,
            top_p=1.0,
            do_sample=True,
            generation_config=self.generation_config,
        )

        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return LLMResult(meta=json.dumps({"response": response}), prompt_ref=prompt_ref, message_ref=message_ref)


    @handle_errors
    def process_video_and_audio(self, audio_base64, frames_base64, prompt: LLMPrompt):
        """
        Process a video composed of audio and base64-encoded image frames.

        Args:
            audio_base64 (str): Base64-encoded audio.
            frames_base64 (List[str]): List of base64-encoded image frames.
            prompt (str): Instruction to the model based on the audio and video frames.

        Returns:
            str: Model-generated text based on the combined video and audio input.
        """
        waveform, sample_rate = convert_base64_to_audio(audio_base64)
        images = [convert_base64_to_image(f) for f in frames_base64]

        image_tags = "".join(f"<|image_{i}|>" for i in range(len(images)))
        audio_tag = "<|audio_1|>"

        messages, prompt_ref, message_ref = self._extract_messages_and_refs(prompt)
        last_msg = next((m.content for m in reversed(prompt.messages) if m.role == "user"), "")

        full_msg = f"{audio_tag}{image_tags} These frames represent a video, and the attached audio is its soundtrack. Based on both, {last_msg}"

        chat_prompt = self.processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": full_msg}],
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=chat_prompt,
            images=images,
            audios=[(waveform, sample_rate)],
            return_tensors="pt"
        ).to(self.device)

        generation_config = GenerationConfig.from_pretrained(self.model_name)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=4096,
            temperature=0.5,
            top_p=1.0,
            do_sample=True,
            generation_config=generation_config,
        )

        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return LLMResult(meta=json.dumps({"response": response}), prompt_ref=prompt_ref, message_ref=message_ref)


    @handle_errors

    def process_video(self, video_base64, prompt):
        """
        Extracts frames + audio from video and processes them with the prompt.
        """
        video_path = save_base64_to_temp_file(video_base64, suffix=".mp4")

        frames = extract_frames_ffmpeg(video_path, every_n_seconds=5)
        audio = extract_audio_base64_ffmpeg(video_path)

        # Placeholder to match multi-modal model input
        frame_placeholders = [f"<|image_{i}|>" for i in range(len(frames))]
        audio_placeholder = "<|audio|>"
        combined_prompt = "".join(frame_placeholders) + audio_placeholder + prompt

        messages = [{"role": "user", "content": combined_prompt}]
        prompt_encoded = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(text=prompt_encoded, images=frames, audios=[audio], return_tensors="pt")
        inputs.to(self.device)


        generated_ids = self.model.generate(
            **inputs, **self.generation_args, generation_config=self.generation_config
        )
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]



    def get_info(self):
        """
        This method returns the model information.
        """
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_source": self.model_source,
            "model_lang": self.model_lang
        }
