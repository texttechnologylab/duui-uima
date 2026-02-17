import base64
from typing import List, Union
from fastapi import UploadFile
from models.duui_models import ImageType, AudioType, VideoTypes, LLMPrompt
from models.ollama_models import OllamaConfig, OllamaRequest, OllamaResponse

from PIL import Image

from io import BytesIO
import base64
import soundfile as sf
import io
import functools
import traceback
import tempfile
import subprocess
import os
from typing import Tuple, List
import json
import cv2
from uuid import uuid4

async def encode_file_to_base64(file: UploadFile) -> str:
    content = await file.read()
    return base64.b64encode(content).decode("utf-8")

def map_duui_to_ollama(
        model_name,
        system_pormpt: LLMPrompt,
        prompt: LLMPrompt,
        encoded_images: List[str] = None,
        encoded_audios: List[str] = None,
        encoded_videos: List[str] = None,
) -> OllamaRequest:
    system_prompt = system_pormpt if system_pormpt else None
    prompt = prompt.messages[-1].content if prompt else ""

    return OllamaRequest(
        model=model_name,
        prompt=prompt,
        system_prompt=system_prompt,
        images=encoded_images,
        audio=encoded_audios[0] if encoded_audios else None,
        video=encoded_videos[0] if encoded_videos else None,
    )


def convert_base64_to_image(base64_string):
    return Image.open(BytesIO(base64.b64decode(base64_string)))

def convert_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def convert_base64_to_audio(base64_audio):
    """
    Converts a base64-encoded audio string to a NumPy waveform and sample rate.
    """
    audio_bytes = base64.b64decode(base64_audio)
    audio_buffer = io.BytesIO(audio_bytes)
    waveform, sample_rate = sf.read(audio_buffer)
    return waveform, sample_rate

def convert_audio_to_base64(waveform, sample_rate, format="WAV"):
    """
    Converts a NumPy waveform and sample rate to a base64-encoded audio string.
    """
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, waveform, sample_rate, format=format)
    audio_bytes = audio_buffer.getvalue()
    base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
    return base64_audio


def find_label_positions(text, label):
    start = text.find(label)
    end = start + len(label) if start != -1 else -1
    return start, end



def save_base64_to_temp_file(base64_str, suffix=""):
    data = base64.b64decode(base64_str)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        return tmp.name


def decouple_video(videobase64: str):
    # Decode the video base64 and save to temp file
    video_bytes = base64.b64decode(videobase64)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f_video:
        f_video.write(video_bytes)
        video_path = f_video.name

    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

    # Extract audio
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
            audio_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        with open(audio_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")
    except subprocess.CalledProcessError:
        print("⚠️ No audio stream found or ffmpeg failed. Continuing without audio.")
        audio_base64 = None

    # Extract frames
    frame_dir = tempfile.mkdtemp()
    frame_pattern = os.path.join(frame_dir, "frame_%03d.jpg")
    try:
        subprocess.run([
            "ffmpeg", "-i", video_path, "-q:v", "2", frame_pattern
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg frame extraction failed: {e}")

    # Collect every 5th frame + first and last
    frame_files = sorted(os.listdir(frame_dir))
    selected_indices = set([0, len(frame_files) - 1] + list(range(5, len(frame_files), 5)))
    selected_frames = [
        os.path.join(frame_dir, f) for i, f in enumerate(frame_files) if i in selected_indices
    ]

    frames_b64 = []
    for frame_file in selected_frames:
        with open(frame_file, "rb") as f:
            frames_b64.append(base64.b64encode(f.read()).decode("utf-8"))

    return audio_base64, frames_b64


def convert_base64_to_video(b64):
    return BytesIO(base64.b64decode(b64))



def video_has_audio(video_path: str) -> bool:
    """Returns True if the video file has an audio stream."""
    cmd = [
        "ffprobe", "-i", video_path,
        "-show_streams", "-select_streams", "a", "-loglevel", "error"
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return bool(result.stdout.strip())