"""Media transport between the face and speaker DUUI components."""

from __future__ import annotations

import base64
import binascii
import json
import math
import os
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path


MAX_MEDIA_BYTES = int(os.getenv("MAX_MEDIA_BYTES", str(500 * 1024 * 1024)))


def decode_media(value: str, label: str) -> bytes:
    if not value or len(value) > (MAX_MEDIA_BYTES * 4 // 3) + 4:
        raise ValueError(f"{label} is empty or exceeds the {MAX_MEDIA_BYTES}-byte limit")
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{label} must be valid Base64") from exc
    if not data or len(data) > MAX_MEDIA_BYTES:
        raise ValueError(f"{label} is empty or exceeds the {MAX_MEDIA_BYTES}-byte limit")
    return data


def run(command: list[str]) -> bytes:
    try:
        result = subprocess.run(command, capture_output=True, check=True, timeout=3600)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required media tool is missing: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{command[0]} timed out") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"{command[0]} failed: {message}") from exc
    return result.stdout


def probe(path: Path) -> dict:
    data = json.loads(run([
        "ffprobe", "-v", "error", "-show_entries",
        "stream=codec_type,avg_frame_rate,duration:format=duration", "-of", "json", str(path)
    ]))
    streams = data.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError("Input has no video stream")
    try:
        fps = float(Fraction(video["avg_frame_rate"]))
        duration = float(video.get("duration") or data["format"]["duration"])
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("Could not determine video timing") from exc
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Video has invalid frame rate or duration")
    return {
        "fps": fps,
        "length": duration,
        "has_audio": any(stream.get("codec_type") == "audio" for stream in streams),
    }


def extract_audio(video_b64: str) -> str:
    """Return the full audio track as a mono 16 kHz WAV for speaker anonymization."""
    data = decode_media(video_b64, "video")
    with tempfile.TemporaryDirectory() as directory:
        video_path = Path(directory) / "video.mp4"
        audio_path = Path(directory) / "audio.wav"
        video_path.write_bytes(data)
        info = probe(video_path)
        if not info["has_audio"]:
            return ""
        run([
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(video_path),
            "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", "-y", str(audio_path)
        ])
        return base64.b64encode(audio_path.read_bytes()).decode("ascii")


def mux_audio(video_b64: str, audio_b64: str) -> tuple[str, float, float]:
    """Replace the video's audio; never carry its original audio into the output."""
    video_data = decode_media(video_b64, "video")
    with tempfile.TemporaryDirectory() as directory:
        video_path = Path(directory) / "video.mp4"
        video_path.write_bytes(video_data)
        info = probe(video_path)
        if not audio_b64:
            if info["has_audio"]:
                raise ValueError("Anonymized audio is empty for a video with an audio track")
            return video_b64, info["length"], info["fps"]

        audio_path = Path(directory) / "voice.wav"
        output_path = Path(directory) / "output.mp4"
        audio_path.write_bytes(decode_media(audio_b64, "anonymized audio"))
        run([
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(video_path),
            "-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-af", "apad",
            "-t", f"{info['length']:.9f}", "-map_metadata", "-1", "-map_chapters", "-1",
            "-movflags", "+faststart", "-y", str(output_path)
        ])
        output_info = probe(output_path)
        if not output_info["has_audio"]:
            raise ValueError("Muxed video has no anonymized audio")
        return (base64.b64encode(output_path.read_bytes()).decode("ascii"),
                output_info["length"], output_info["fps"])
