import base64
import logging
import requests
from typing import Optional, List
from models.ollama_models import OllamaConfig, OllamaRequest, OllamaResponse
from services.utils import convert_base64_to_image

import base64
import tempfile
import os
from io import BytesIO
from PIL import Image




logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, config: OllamaConfig):
        self.base_url = f"{config.host}:{config.port}/api"
        self.auth_token = config.auth_token
        self.headers = (
            {"Authorization": f"Bearer {self.auth_token}"}
            if self.auth_token
            else {}
        )
    # def generate(self, request: OllamaRequest) -> OllamaResponse:
    #     try:
    #         # Initialize the content array with the text prompt
    #         content = [{"type": "text", "text": request.prompt}]
    #
    #         # 1. Handle Images
    #         if request.images:
    #             for base64_image in request.images:
    #                 # Strip potential header if the input already contains "data:image/..."
    #                 img_data = base64_image.split(",")[-1]
    #                 content.append({
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": f"data:image/jpeg;base64,{img_data}"
    #                     }
    #                 })
    #
    #         # 2. Handle Audio
    #         if request.audio:
    #             audio_data = request.audio.split(",")[-1]
    #             content.append({
    #                 "type": "input_audio",
    #                 "input_audio": {
    #                     "data": audio_data,
    #                     "format": "wav"
    #                 }
    #             })
    #
    #         # 3. Handle Video
    #         if request.video:
    #             video_data = request.video.split(",")[-1]
    #             content.append({
    #                 "type": "video_url", # Note: Check your specific provider's key for video
    #                 "video_url": {
    #                     "url": f"data:video/mp4;base64,{video_data}"
    #                 }
    #             })
    #
    #         # Construct the messages array
    #         messages = [{"role": "user", "content": content}]
    #
    #         # Add system prompt as the first message if provided
    #         if request.system_prompt:
    #             messages.insert(0, {"role": "system", "content": request.system_prompt})
    #
    #         # Construct the payload
    #         payload = {
    #             "model": request.model,
    #             "messages": messages,
    #             "stream": False
    #         }
    #
    #         response = requests.post(
    #             f"{self.base_url}/chat/completions",
    #             json=payload,
    #             headers=self.headers,
    #         )
    #         response.raise_for_status()
    #
    #         # Extract the response text
    #         response_json = response.json()
    #         response_text = ""
    #         if "choices" in response_json and len(response_json["choices"]) > 0:
    #             response_text = response_json["choices"][0]["message"]["content"]
    #
    #         return OllamaResponse(
    #             response=response_text,
    #             model=request.model,
    #             status="success",
    #         )
    #
    #     except Exception as e:
    #         logger.error(f"Ollama API Error: {e}")
    #         return OllamaResponse(
    #             response="",
    #             model=request.model,
    #             status="error",
    #             error=str(e),
    #         )

    def generate(self, request: OllamaRequest) -> OllamaResponse:
        try:
            # 1. Prepare the content list with the text prompt
            content = [{"type": "text", "text": request.prompt}]

            # 2. Add Images (The primary reason for 400 is often the URI format)
            if request.images:
                for base64_image in request.images:
                    # Ensure we have a clean base64 string without any existing headers
                    clean_base64 = base64_image.split(",")[-1]
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/JPG;base64,{clean_base64}"
                            # "url": f"# data:image/jpeg;base64,{clean_base64}"
                        }
                    })

            # 3. Handle Audio/Video with Caution
            # NOTE: Most Ollama OpenAI-compatible endpoints will throw a 400 if they
            # see 'input_audio' or 'video_url'. We only add them if present.
            if request.audio:
                # If your server strictly follows OpenAI's newer spec:
                content.append({
                    "type": "input_audio",
                    "input_audio": {"data": request.audio.split(",")[-1], "format": "wav"}
                })

            # Construct the messages array
            messages = [{"role": "user", "content": content}]

            # Insert system prompt at the beginning
            if request.system_prompt:
                messages.insert(0, {"role": "system", "content": request.system_prompt})

            payload = {
                "model": request.model,
                "messages": messages,
                # "stream": False
            }

            # Make the request
            response = requests.post(
                f"{self.base_url}/chat/completions", # Confirm this is the correct endpoint
                json=payload,
                headers=self.headers,
            )

            if response.status_code != 200:
                logger.error(f"Ollama Error Response: {response.text}")
                response.raise_for_status()

            response_json = response.json()
            response_text = response_json["choices"][0]["message"]["content"]

            return OllamaResponse(
                response=response_text,
                model=request.model,
                status="success",
            )

        except Exception as e:
            logger.error(f"Ollama API Error: {e}")
            return OllamaResponse(
                response="",
                model=request.model,
                status="error",
                error=str(e),
            )
    def ping(self, request: OllamaRequest):
        try:
            chat_payload = {
                "model": request.model,
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "ping"}]}
                ]
            }
            chat_resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=chat_payload,
                headers=self.headers
            )

            chat_resp.raise_for_status()
            chat_data = chat_resp.json()
            if not ("choices" in chat_data and len(chat_data["choices"]) > 0):
                print("Chat completion endpoint responded but returned no choices.")
                return False
            print("Chat completion endpoint reachable.")

        except Exception as e:
            logger.error(f"Ollama API Error: {e}")
            return OllamaResponse(
                response="",
                model=request.model,
                status="error",
                error=str(e),
            )
