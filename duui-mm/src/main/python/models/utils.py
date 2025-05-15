
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
from .duui_api_models import LLMResult
from uuid import uuid4

def handle_errors(func):
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            self.logger.error(f"{func.__name__} failed: {e}", exc_info=True)
            dummy_ref = int(uuid4().int % 1_000_000)
            return LLMResult(
                meta=json.dumps({"error": f"{func.__name__} failed", "detail": str(e)}),
                prompt_ref=dummy_ref,
                message_ref=dummy_ref,
            )
    return wrapper

def extract_frames_ffmpeg(video_path, every_n_seconds=5):
    output_dir = tempfile.mkdtemp()
    output_pattern = os.path.join(output_dir, "frame_%04d.png")

    # Extract frames every N seconds
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps=1/{every_n_seconds}",
        output_pattern,
        "-hide_banner",
        "-loglevel", "error"
    ]
    subprocess.run(cmd, check=True)

    # Load images as PIL
    frames = []
    for filename in sorted(os.listdir(output_dir)):
        if filename.endswith(".png"):
            frame_path = os.path.join(output_dir, filename)
            with Image.open(frame_path) as img:
                frames.append(img.copy())
            os.remove(frame_path)
    os.rmdir(output_dir)
    return frames


def extract_audio_base64_ffmpeg(video_path):
    audio_output = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

    # Extract full audio
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",  # no video
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "2",
        audio_output,
        "-hide_banner",
        "-loglevel", "error"
    ]
    subprocess.run(cmd, check=True)

    # Read and encode to base64
    with open(audio_output, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode("utf-8")
    os.remove(audio_output)
    return audio_base64

def fix_unicode_problems(text):
    # fix imgji in python string and prevent json error on response
    # File "/usr/local/lib/python3.8/site-packages/starlette/responses.py", line 190, in render
    # UnicodeEncodeError: 'utf-8' codec can't encode characters in position xx-yy: surrogates not allowed
    clean_text = text.encode('utf-16', 'surrogatepass').decode('utf-16', 'surrogateescape')
    return clean_text

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


def convert_base64_to_image(b64):
    return Image.open(BytesIO(base64.b64decode(b64)))

def convert_base64_to_audio(b64):
    return BytesIO(base64.b64decode(b64))

def convert_base64_to_video(b64):
    return BytesIO(base64.b64decode(b64))
