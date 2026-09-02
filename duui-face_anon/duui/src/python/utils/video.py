from __future__ import annotations

import base64
import binascii
import json
import math
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import cv2
import numpy as np
from PIL import Image


FrameProcessor = Callable[[Image.Image], Image.Image]

_CENTER_WEIGHT = 0.7
_SHARPNESS_WEIGHT = 0.3
_DIVERSITY_WEIGHT = 0.5
_DEDUP_THRESHOLD = 0.96


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


def _frame_feature(frame: Image.Image) -> Tuple[np.ndarray, float]:
    rgb = np.asarray(frame.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    patch = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255.0
    histograms = [
        cv2.calcHist([rgb], [channel], None, [16], [0, 256]).ravel()
        for channel in range(3)
    ]
    histograms = [histogram / max(float(histogram.sum()), 1.0) for histogram in histograms]
    feature = np.concatenate([patch, *histograms]).astype(np.float32)
    norm = np.linalg.norm(feature)
    if norm > 0:
        feature /= norm
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return feature, sharpness


def _rank_candidates(candidates: List[Dict], start: float, end: float, k: int) -> List[Dict]:
    if not candidates:
        return []

    sharpness = np.asarray([candidate["sharpness"] for candidate in candidates], dtype=np.float32)
    sharpness_range = float(sharpness.max() - sharpness.min())
    normalized_sharpness = (
        (sharpness - sharpness.min()) / sharpness_range
        if sharpness_range > 0
        else np.zeros_like(sharpness)
    )
    midpoint = (start + end) / 2.0
    half_duration = max((end - start) / 2.0, 1e-9)
    for index, candidate in enumerate(candidates):
        center_score = 1.0 - min(1.0, abs(candidate["timestamp"] - midpoint) / half_duration)
        candidate["relevance"] = (
            _CENTER_WEIGHT * center_score
            + _SHARPNESS_WEIGHT * float(normalized_sharpness[index])
        )

    selected = []
    remaining = list(candidates)
    while remaining and len(selected) < min(k, len(candidates)):
        eligible = []
        for candidate in remaining:
            max_similarity = max(
                (float(np.dot(candidate["feature"], chosen["feature"])) for chosen in selected),
                default=0.0,
            )
            if selected and max_similarity > _DEDUP_THRESHOLD:
                continue
            score = (
                (1.0 - _DIVERSITY_WEIGHT) * candidate["relevance"]
                - _DIVERSITY_WEIGHT * max_similarity
            )
            eligible.append((score, candidate))

        # Static segments may not contain k non-duplicates, so fill from the best remaining frames.
        chosen = (
            max(eligible, key=lambda item: item[0])[1]
            if eligible
            else max(remaining, key=lambda candidate: candidate["relevance"])
        )
        selected.append(chosen)
        remaining = [candidate for candidate in remaining if candidate is not chosen]

    return selected


def _extract_adaptive_frames(
    input_path: Path,
    duration: float,
    segment_duration: float,
    representative_frames: int,
    frames_path: Path,
) -> List[Tuple[float, Path]]:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError("Could not open input video")

    selected_frames = []
    candidate_count = max(representative_frames, 4 * representative_frames, 8)
    try:
        segment_count = max(1, int(math.ceil(duration / segment_duration)))
        for segment_index in range(segment_count):
            start = segment_index * segment_duration
            end = min(duration, start + segment_duration)
            if end <= start:
                continue

            candidates = []
            seen_frame_indices = set()
            for candidate_index in range(1, candidate_count + 1):
                timestamp = start + candidate_index / (candidate_count + 1) * (end - start)
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                ok, bgr_frame = capture.read()
                if not ok:
                    continue
                frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                if frame_index in seen_frame_indices:
                    continue
                seen_frame_indices.add(frame_index)
                frame = Image.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
                feature, sharpness = _frame_feature(frame)
                candidates.append({
                    "timestamp": timestamp,
                    "frame": frame,
                    "feature": feature,
                    "sharpness": sharpness,
                })

            ranked = sorted(
                _rank_candidates(candidates, start, end, representative_frames),
                key=lambda candidate: candidate["timestamp"],
            )
            for candidate in ranked:
                frame_path = frames_path / f"frame_{len(selected_frames) + 1:09d}.png"
                candidate["frame"].save(frame_path, format="PNG")
                selected_frames.append((candidate["timestamp"], frame_path))
    finally:
        capture.release()

    return sorted(selected_frames, key=lambda item: item[0])


def process_video(
    video_base64: str,
    frame_interval: int,
    process_frame: FrameProcessor = lambda frame: frame,
    sampling_mode: str = "uniform",
    segment_duration: float = 10.0,
    representative_frames: int = 5,
) -> Dict[str, Union[str, float]]:
    if frame_interval < 1:
        raise ValueError("frame_interval must be at least 1")
    if sampling_mode not in ("uniform", "adaptive"):
        raise ValueError(f"Unknown sampling_mode: {sampling_mode}")
    if segment_duration <= 0:
        raise ValueError("segment_duration must be positive")
    if representative_frames < 1:
        raise ValueError("representative_frames must be at least 1")

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
        selected_timestamps = None
        if sampling_mode == "uniform":
            # Lowering the output FPS by the same factor preserves playback duration.
            output_fps = float(input_metadata["fps"]) / frame_interval
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
        else:
            duration = float(input_metadata["length"])
            if duration <= 0:
                raise ValueError("Adaptive sampling requires a video with a known duration")
            selected = _extract_adaptive_frames(
                input_path,
                duration,
                segment_duration,
                representative_frames,
                frames_path,
            )
            selected_timestamps = [timestamp for timestamp, _ in selected]
            frame_files = [frame_path for _, frame_path in selected]

        if not frame_files:
            raise ValueError("No frames could be selected from the input video")

        output_frame_size = None
        for frame_path in frame_files:
            with Image.open(frame_path) as frame:
                # The callback is the boundary between video transport and frame processing.
                processed_frame = process_frame(frame.convert("RGB"))
                if not isinstance(processed_frame, Image.Image):
                    raise TypeError("process_frame must return a PIL Image")
                if output_frame_size is None:
                    output_frame_size = processed_frame.size
                elif processed_frame.size != output_frame_size:
                    raise ValueError("process_frame must return the same dimensions for every frame")
                processed_frame.convert("RGB").save(frame_path, format="PNG")

        encode_command = ["ffmpeg", "-v", "error"]
        if sampling_mode == "uniform":
            encode_command.extend([
                "-framerate", f"{output_fps:.12g}",
                "-i", str(frames_path / "frame_%09d.png"),
            ])
        else:
            boundaries = [0.0]
            boundaries.extend(
                (selected_timestamps[index] + selected_timestamps[index + 1]) / 2.0
                for index in range(len(selected_timestamps) - 1)
            )
            boundaries.append(float(input_metadata["length"]))
            concat_lines = []
            for index, frame_path in enumerate(frame_files):
                concat_lines.append(f"file '{frame_path}'")
                duration = max(0.001, boundaries[index + 1] - boundaries[index])
                concat_lines.append(f"duration {duration:.9f}")
            # The concat demuxer needs the last file repeated for its duration to take effect.
            concat_lines.append(f"file '{frame_files[-1]}'")
            concat_path = temp_path / "frames.txt"
            concat_path.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")
            encode_command.extend(["-f", "concat", "-safe", "0", "-i", str(concat_path)])

        encode_command.extend([
            "-i", str(input_path),
            "-map", "0:v:0",
            "-map", "1:a?",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ])
        if sampling_mode == "adaptive":
            encode_command.extend(["-vsync", "vfr"])
        if input_metadata["has_audio"]:
            encode_command.extend(["-c:a", "aac"])
        encode_command.extend(["-shortest", "-y", str(output_path)])
        _run(encode_command)

        # Report properties of the encoded file, which may differ slightly from input.
        output_metadata = probe_video(output_path)
        return {
            "src": base64.b64encode(output_path.read_bytes()).decode("ascii"),
            "length": float(output_metadata["length"]),
            "fps": float(output_metadata["fps"]),
        }
