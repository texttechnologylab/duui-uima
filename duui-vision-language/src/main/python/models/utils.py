
from PIL import Image
from io import BytesIO
import base64
import functools
import traceback
import tempfile
import subprocess
import os


def handle_errors(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error in `{method.__name__}`: {e}")
            self.logger.debug(traceback.format_exc())
            return {"error": f"{method.__name__} failed", "detail": str(e)}
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





def save_base64_to_temp_file(base64_str, suffix=""):
    data = base64.b64decode(base64_str)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        return tmp.name
