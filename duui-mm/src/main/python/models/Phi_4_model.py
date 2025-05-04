from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, AutoModelForVision2Seq

import logging

from .utils import convert_base64_to_image, convert_image_to_base64, convert_base64_to_audio, convert_audio_to_base64, handle_errors

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
                                                     torch_dtype="auto").eval()
        self.model.to(device)

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

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
    def process_image(self, image_base64, prompt):
        """
        This method takes an image and returns the processed image.
        The image should be a base64 image.
        """
        # convert image from base64 to PIL
        image = convert_base64_to_image(image_base64)

        placeholder  = f"<|image_0|>"
        messages = [
            {
                "role": "user",
                "content": (
                        placeholder
                        + prompt
                )
            }
        ]
        prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # get the inputs for the model
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs.to(self.device)

        generation_config = GenerationConfig.from_pretrained(self.model_name)

        generation_args = {
            "max_new_tokens": 512,
            "temperature": 0.5,
            "do_sample": True,
        }

        generated_ids = self.model.generate(
            **inputs,
            **generation_args,
            generation_config=generation_config,
        )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return generated_text

    @handle_errors
    def process_text(self, prompt):
        """
        This method takes a text and returns the processed text.
        The text should be a string that describes the data to be processed.
        """
        # Process the text and return a response
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # get the inputs for the model
        inputs = self.processor.tokenizer(prompt, return_tensors="pt")
        inputs.to(self.device)
        generation_config = GenerationConfig.from_pretrained(self.model_name)
        generation_args = {
            "max_new_tokens": 512,
            "temperature": 0.5,
            "do_sample": True,
        }
        generated_ids = self.model.generate(
            **inputs,
            **generation_args,
            generation_config=generation_config,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text

    @handle_errors
    def process_audio(self, base64_audio, prompt):
        """
        Processes a base64-encoded audio input and returns generated text based on the user's prompt.
        """
        # Convert base64 to waveform
        waveform, sample_rate = convert_base64_to_audio(base64_audio)

        # Prompt formatting
        user_tag = "<|user|>"
        assistant_tag = "<|assistant|>"
        end_tag = "<|end|>"
        audio_tag = "<|audio_1|>"

        full_prompt = f"{user_tag}{audio_tag}{prompt}{end_tag}{assistant_tag}"

        # Prepare inputs for the model
        inputs = self.processor(
            text=full_prompt,
            audios=[(waveform, sample_rate)],
            return_tensors='pt'
        )
        inputs = inputs.to(self.device)

        # Load generation configuration
        generation_config = GenerationConfig.from_pretrained(self.model_name)

        generation_args = {
            "max_new_tokens": 1200,
            "temperature": 0.5,
            "do_sample": True,
        }

        # Generate the output
        generate_ids = self.model.generate(
            **inputs,
            **generation_args,
            generation_config=generation_config,
        )

        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]

        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return response

    @handle_errors
    def process_video_frames(self, prompt, frames):
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
        images = [convert_base64_to_image(frame) for frame in frames]

        # Create image placeholders for prompt
        placeholder = "".join(f"<|image_{i}|>" for i in range(len(images)))

        # Create structured message with user content
        messages = [
            {
                "role": "user",
                "content": f"{placeholder},Please summarize the content of these images as they are a series of frames represents video. AND based on the the summery, {prompt}"
            }
        ]

        # Format the prompt
        chat_prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Prepare model inputs
        inputs = self.processor(text=chat_prompt, images=images, return_tensors='pt')
        inputs = inputs.to(self.device)

        # Load generation config
        generation_config = GenerationConfig.from_pretrained(self.model_name)

        generation_args = {
            "max_new_tokens": 4096,
            "temperature": 0.5,
            "top_p": 1.0,
            "do_sample": True
        }

        # Generate output
        generate_ids = self.model.generate(
            **inputs,
            **generation_args,
            generation_config=generation_config,
        )

        # Remove the prompt portion
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]

        # Decode the generated output
        response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return response

    @handle_errors
    def process_video_and_audio(self, audio_base64, frames_base64, prompt="Describe the audio-visual content and generate a summary or interpretation."):
        """
        Process a video composed of audio and base64-encoded image frames.

        Args:
            audio_base64 (str): Base64-encoded audio.
            frames_base64 (List[str]): List of base64-encoded image frames.
            prompt (str): Instruction to the model based on the audio and video frames.

        Returns:
            str: Model-generated text based on the combined video and audio input.
        """
        # Decode audio from base64
        waveform, sample_rate = convert_base64_to_audio(audio_base64)

        # Decode frames from base64 to PIL
        images = [convert_base64_to_image(frame) for frame in frames_base64]

        # Build placeholders
        image_tags = "".join(f"<|image_{i}|>" for i in range(len(images)))
        audio_tag = "<|audio_1|>"

        # Build the chat message
        messages = [
            {
                "role": "user",
                "content": f"{audio_tag}{image_tags} These frames represent a video, and the attached audio is its soundtrack. Based on both, {prompt}"
            }
        ]

        # Format the prompt
        chat_prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Prepare model input
        inputs = self.processor(
            text=chat_prompt,
            images=images,
            audios=[(waveform, sample_rate)],
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)

        # Load generation configuration
        generation_config = GenerationConfig.from_pretrained(self.model_name)

        generation_args = {
            "max_new_tokens": 4096,
            "temperature": 0.5,
            "top_p": 1.0,
            "do_sample": True
        }

        # Generate output
        generate_ids = self.model.generate(
            **inputs,
            **generation_args,
            generation_config=generation_config,
        )

        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]

        # Decode the output
        response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return response



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
