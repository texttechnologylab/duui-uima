
import cv2
from PIL import Image
import numpy as np
from io import BytesIO
import base64
import soundfile as sf
import io
import functools
import traceback

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


def is_overlapping(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)


def plot_bbox(image, entities):
    """Draw high-quality bounding boxes with smart label placement using OpenCV."""
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))[:, :, ::-1]  # RGB to BGR

    image_h, image_w = image.shape[:2]
    image_copy = image.copy()
    used_label_boxes = []

    for entity in entities:
        if not entity.bounding_box:
            continue

        x1, y1, x2, y2 = map(int, entity.bounding_box[0])
        label = entity.name
        color = tuple(np.random.randint(0, 255, size=3).tolist())

        # Draw the bounding box
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, thickness=2)

        # Label settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        # Get label size
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_pad = 4
        box_w = text_w + 2 * label_pad
        box_h = text_h + 2 * label_pad

        # Default label position: above the bbox
        label_x1 = x1
        label_y1 = y1 - box_h if y1 - box_h > 0 else y1 + 2
        label_x2 = label_x1 + box_w
        label_y2 = label_y1 + box_h

        # Clamp horizontally
        if label_x2 > image_w:
            label_x1 = image_w - box_w
            label_x2 = image_w

        # Avoid overlapping with other labels
        max_attempts = 10
        attempts = 0
        while any(is_overlapping((label_x1, label_y1, label_x2, label_y2), box) for box in used_label_boxes) and attempts < max_attempts:
            label_y1 += box_h + 2
            label_y2 += box_h + 2
            if label_y2 > image_h:
                label_y1 = max(0, y1 - box_h)  # fallback to above box
                label_y2 = label_y1 + box_h
                break
            attempts += 1

        used_label_boxes.append((label_x1, label_y1, label_x2, label_y2))

        # Draw label background
        cv2.rectangle(image_copy, (label_x1, label_y1), (label_x2, label_y2), color, -1)

        # Draw label text
        text_x = label_x1 + label_pad
        text_y = label_y2 - label_pad
        cv2.putText(image_copy, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return Image.fromarray(image_copy[:, :, ::-1])  # Convert BGR to RGB




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