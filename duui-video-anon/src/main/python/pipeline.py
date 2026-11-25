"""Coordinate remote DUUI face and speaker stages with local media processing."""

from __future__ import annotations

import base64
import os
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from duui_logging import log_info

from media import decode_media, extract_audio, mux_audio, probe


RUNNER_CLASS = "org.texttechnologylab.duui.videoanon.VideoAnonPipeline"
JAVA_CLASSPATH = os.getenv("DUUI_JAVA_CLASSPATH", "/app/java/classes:/app/java/lib/*")


def required_service_urls() -> dict[str, str]:
    urls = {}
    for name in ("DUUI_FACE_ANON_URL", "DUUI_SPEAKER_ANON_URL"):
        value = os.getenv(name, "").strip()
        try:
            parsed = urlsplit(value)
            port = parsed.port
            valid = (parsed.scheme in ("http", "https") and parsed.hostname is not None
                     and parsed.username is None and parsed.password is None
                     and (port is None or port > 0) and not parsed.query
                     and not parsed.fragment and not any(c.isspace() for c in value))
        except ValueError:
            valid = False
        if not valid:
            raise RuntimeError(f"{name} must be an explicit HTTP(S) service URL")
        urls[name] = value
    return urls


def run_java_stage(stage: str, input_path: Path, output_path: Path,
                   environment: dict[str, str], root: Path) -> None:
    log_info(f"Starting DUUI {stage} stage")
    command = [
        "java", "--add-opens", "java.base/java.util=ALL-UNNAMED",
        "-cp", JAVA_CLASSPATH, RUNNER_CLASS,
        stage, str(input_path), str(output_path),
    ]
    try:
        with (root / f"{stage}.log").open("w+") as log:
            try:
                subprocess.run(command, env=environment, check=True, timeout=7200,
                               stdout=log, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as exc:
                log.seek(0)
                detail = log.read()[-4000:].strip()
                raise RuntimeError(
                    f"DUUI {stage} stage failed with exit code {exc.returncode}: {detail}"
                ) from exc
    except FileNotFoundError as exc:
        raise RuntimeError("Java 21 is required by duui-video-anon") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"DUUI {stage} stage timed out") from exc
    log_info(f"Completed DUUI {stage} stage")


def run_pipeline(video_b64: str, mimetype: str | None,
                 options: dict[str, str]) -> tuple[str, float, float]:
    service_urls = required_service_urls()
    raw = decode_media(video_b64, "video")
    if mimetype not in (None, "video/mp4", "video/webm"):
        raise ValueError("Input must be an MP4 or WebM video")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        input_path = root / ("input.webm" if mimetype == "video/webm" else "input.mp4")
        input_path.write_bytes(raw)
        probe(input_path)

        environment = os.environ.copy()
        environment.update(service_urls)
        for key, setting in (
            ("anon_type", "DUUI_FACE_MODE"),
            ("redact_type", "DUUI_REDACT_TYPE"),
            ("language", "DUUI_LANGUAGE"),
            ("hf_token", "HF_TOKEN"),
        ):
            if options.get(key):
                environment[setting] = options[key]

        face_path = root / "face.mp4"
        run_java_stage("face", input_path, face_path, environment, root)
        face_video = base64.b64encode(face_path.read_bytes()).decode("ascii")
        audio = extract_audio(face_video)

        anonymized_audio = ""
        if audio:
            audio_path = root / "extracted.wav"
            audio_path.write_bytes(base64.b64decode(audio, validate=True))
            speaker_path = root / "anonymized.wav"
            run_java_stage("speaker", audio_path, speaker_path, environment, root)
            anonymized_audio = base64.b64encode(speaker_path.read_bytes()).decode("ascii")
        else:
            log_info("Skipping speaker stage for silent video")

        return mux_audio(face_video, anonymized_audio)
