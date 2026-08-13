from __future__ import annotations

import base64
import binascii
import json
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Callable, Dict, List, Union

from PIL import Image


FrameProcessor = Callable[[Image.Image], Image.Image]


def _run(command: List[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required video tool is not installed: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{command[0]} failed: {stderr}") from exc


def _parse_frame_rate(value: str) -> float:
    try:
        rate = float(Fraction(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"Invalid video frame rate reported by ffprobe: {value}") from exc
    if rate <= 0:
        raise ValueError(f"Video frame rate must be positive, received {value}")
    return rate


def probe_video(video_path: Union[str, Path]) -> Dict[str, Union[float, int, bool]]:
    # Read metadata from the media.
    result = _run([
        "ffprobe",
        "-v", "error",
        "-show_entries", "stream=codec_type,width,height,avg_frame_rate:format=duration",
        "-of", "json",
        str(video_path),
    ])
    metadata = json.loads(result.stdout)
    streams = metadata.get("streams", [])
    video_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        raise ValueError("Input does not contain a video stream")

    duration_value = metadata.get("format", {}).get("duration")
    duration = float(duration_value) if duration_value is not None else -1.0
    return {
        "fps": _parse_frame_rate(video_stream["avg_frame_rate"]),
        "length": duration,
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "has_audio": any(stream.get("codec_type") == "audio" for stream in streams),
    }


def process_video(video_base64: str, frame_interval: int, process_frame: FrameProcessor = lambda frame: frame,) -> Dict[str, Union[str, float]]:
    if frame_interval < 1:
        raise ValueError("frame_interval must be at least 1")

    try:
        video_bytes = base64.b64decode(video_base64, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Video src is not valid base64") from exc

    with tempfile.TemporaryDirectory(prefix="duui-face-anon-") as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "input.mp4"
        frames_path = temp_path / "frames"
        output_path = temp_path / "output.mp4"
        input_path.write_bytes(video_bytes)
        frames_path.mkdir()

        input_metadata = probe_video(input_path)
        # Lowering the output FPS by the same factor preserves playback duration.
        output_fps = float(input_metadata["fps"]) / frame_interval
        # FFmpeg numbers decoded frames from zero, so this retains 0, X, 2X, ...
        select_filter = f"select=not(mod(n\\,{frame_interval}))"
        _run([
            "ffmpeg",
            "-v", "error",
            "-i", str(input_path),
            "-vf", select_filter,
            "-vsync", "vfr",
            str(frames_path / "frame_%09d.png"),
        ])

        frame_files = sorted(frames_path.glob("frame_*.png"))
        if not frame_files:
            raise ValueError("No frames could be decoded from the input video")

        for frame_path in frame_files:
            with Image.open(frame_path) as frame:
                # The callback is the boundary between video transport and redaction.
                processed_frame = process_frame(frame.convert("RGB"))
                if not isinstance(processed_frame, Image.Image):
                    raise TypeError("process_frame must return a PIL Image")
                if processed_frame.size != frame.size:
                    raise ValueError("process_frame must preserve the video frame dimensions")
                processed_frame.convert("RGB").save(frame_path, format="PNG")

        encode_command = [
            "ffmpeg",
            "-v", "error",
            "-framerate", f"{output_fps:.12g}",
            "-i", str(frames_path / "frame_%09d.png"),
            "-i", str(input_path),
            "-map", "0:v:0",
            "-map", "1:a?",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ]
        if input_metadata["has_audio"]:
            encode_command.extend(["-c:a", "aac"])
        encode_command.extend(["-y", str(output_path)])
        _run(encode_command)

        # Report properties of the encoded file, which may differ slightly from input.
        output_metadata = probe_video(output_path)
        return {
            "src": base64.b64encode(output_path.read_bytes()).decode("ascii"),
            "length": float(output_metadata["length"]),
            "fps": float(output_metadata["fps"]),
        }
