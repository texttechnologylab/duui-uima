import base64
import asyncio
import json
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import anyio.to_thread
import httpx
from duui_logging import log_info

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "main" / "python"))
import app  # noqa: E402
import media  # noqa: E402
import pipeline  # noqa: E402


class PipelineOrchestrationTest(unittest.TestCase):
    def test_audio_passes_through_local_extract_and_mux(self):
        video = base64.b64encode(b"input video").decode("ascii")
        face_video = base64.b64encode(b"face video").decode("ascii")
        extracted_audio = base64.b64encode(b"extracted audio").decode("ascii")
        calls = []

        def java_stage(stage, input_path, output_path, environment, root):
            calls.append((stage, input_path.read_bytes()))
            output_path.write_bytes(b"face video" if stage == "face" else b"anonymized audio")

        def extract(value):
            calls.append(("extract", value))
            return extracted_audio

        def mux(video_b64, audio_b64):
            calls.append(("mux", video_b64, audio_b64))
            return "output", 2.0, 25.0

        with patch.dict(pipeline.os.environ, {
                "DUUI_FACE_ANON_URL": "http://face",
                "DUUI_SPEAKER_ANON_URL": "http://speaker"}), \
                patch.object(pipeline, "probe"), \
                patch.object(pipeline, "run_java_stage", side_effect=java_stage), \
                patch.object(pipeline, "extract_audio", side_effect=extract), \
                patch.object(pipeline, "mux_audio", side_effect=mux):
            result = pipeline.run_pipeline(video, "video/mp4", {})

        self.assertEqual(result, ("output", 2.0, 25.0))
        self.assertEqual(calls, [
            ("face", b"input video"),
            ("extract", face_video),
            ("speaker", b"extracted audio"),
            ("mux", face_video, base64.b64encode(b"anonymized audio").decode("ascii")),
        ])

    def test_silent_video_skips_speaker_stage(self):
        video = base64.b64encode(b"input video").decode("ascii")
        calls = []

        def java_stage(stage, input_path, output_path, environment, root):
            calls.append(stage)
            output_path.write_bytes(b"face video")

        with patch.dict(pipeline.os.environ, {
                "DUUI_FACE_ANON_URL": "http://face",
                "DUUI_SPEAKER_ANON_URL": "http://speaker"}), \
                patch.object(pipeline, "probe"), \
                patch.object(pipeline, "run_java_stage", side_effect=java_stage), \
                patch.object(pipeline, "extract_audio", return_value=""), \
                patch.object(pipeline, "mux_audio", return_value=("output", 2.0, 25.0)) as mux:
            pipeline.run_pipeline(video, "video/mp4", {})

        self.assertEqual(calls, ["face"])
        mux.assert_called_once_with(base64.b64encode(b"face video").decode("ascii"), "")


class ConcurrentRequestsTest(unittest.IsolatedAsyncioTestCase):
    async def test_two_pipeline_requests_complete_together(self):
        both_face_stages_started = threading.Barrier(2)
        limiter = anyio.to_thread.current_default_thread_limiter()
        original_tokens = limiter.total_tokens
        limiter.total_tokens = 2

        def java_stage(stage, input_path, output_path, environment, root):
            log_info(f"Worker {stage}: {input_path.read_bytes().decode('ascii')}")
            if stage == "face":
                both_face_stages_started.wait(timeout=10)
                output_path.write_bytes(b"face:" + input_path.read_bytes())
            else:
                output_path.write_bytes(b"anon:" + input_path.read_bytes())

        def extract(face_video):
            return base64.b64encode(
                b"audio:" + base64.b64decode(face_video)).decode("ascii")

        def mux(face_video, audio):
            result = b"mux:" + base64.b64decode(face_video) + b":" + base64.b64decode(audio)
            return base64.b64encode(result).decode("ascii"), 2.0, 25.0

        try:
            transport = httpx.ASGITransport(app=app.app)
            with patch.dict(pipeline.os.environ, {
                    "DUUI_FACE_ANON_URL": "http://face",
                    "DUUI_SPEAKER_ANON_URL": "http://speaker"}), \
                    patch.object(pipeline, "probe"), \
                    patch.object(pipeline, "run_java_stage", side_effect=java_stage), \
                    patch.object(pipeline, "extract_audio", side_effect=extract), \
                    patch.object(pipeline, "mux_audio", side_effect=mux):
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    requests = [
                        client.post("/v1/process", headers={"DUUI-Log-Collect": "true"}, json={
                            "operation": "pipeline",
                            "video": {"src": base64.b64encode(value).decode("ascii"),
                                      "mimetype": "video/mp4"},
                        })
                        for value in (b"request-1", b"request-2")
                    ]
                    responses = await asyncio.wait_for(asyncio.gather(*requests), timeout=20)
        finally:
            limiter.total_tokens = original_tokens

        for value, response in zip((b"request-1", b"request-2"), responses):
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["operation"], "pipeline")
            self.assertEqual(base64.b64decode(response.json()["video"]["src"]),
                             b"mux:face:" + value + b":anon:audio:face:" + value)
            records = json.loads(response.headers["DUUI-Logs"])
            messages = [record["message"] for record in records]
            self.assertIn(f"Worker face: {value.decode('ascii')}", messages)
            self.assertIn(f"Worker speaker: audio:face:{value.decode('ascii')}", messages)
            other = b"request-2" if value == b"request-1" else b"request-1"
            self.assertFalse(any(other.decode("ascii") in message for message in messages))
            self.assertIn("Completed pipeline request", messages)
            self.assertTrue(all(record["logger"] == "duui-video-anon" for record in records))


class RequestLoggingTest(unittest.IsolatedAsyncioTestCase):
    async def post(self, payload, collect=True):
        headers = {"DUUI-Log-Collect": "true"} if collect else {}
        transport = httpx.ASGITransport(app=app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/v1/process", json=payload, headers=headers)

    async def test_success_collects_worker_logs_without_media_or_options(self):
        def extract(video):
            log_info("Audio extraction worker completed")
            return "audio-result"

        with patch.object(app, "extract_audio", side_effect=extract):
            response = await self.post({
                "operation": "extract", "video": {"src": "private-video-payload"},
                "options": {"hf_token": "private-token"},
            })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"operation": "extract", "audio": "audio-result"})
        payload = response.headers["DUUI-Logs"]
        self.assertNotIn("private-video-payload", payload)
        self.assertNotIn("private-token", payload)
        self.assertEqual([record["message"] for record in json.loads(payload)], [
            "Starting extract request", "Audio extraction worker completed",
            "Completed extract request",
        ])

    async def test_collection_is_optional(self):
        with patch.object(app, "extract_audio", return_value="audio"):
            response = await self.post({"operation": "extract", "video": {"src": "video"}},
                                       collect=False)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("DUUI-Logs", response.headers)

    async def test_invalid_request_returns_warning_without_validation_payload(self):
        response = await self.post({"operation": "unknown", "video": {"src": "private-video"}})
        self.assertEqual(response.status_code, 422)
        records = json.loads(response.headers["DUUI-Logs"])
        self.assertEqual([(record["level"], record["message"]) for record in records],
                         [("WARN", "Invalid DUUI media request")])
        self.assertNotIn("private-video", response.headers["DUUI-Logs"])

    async def test_processing_failure_returns_error_with_traceback(self):
        with patch.object(app, "extract_audio", side_effect=RuntimeError("Media tool failed")):
            response = await self.post({"operation": "extract", "video": {"src": "video"}})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "Media tool failed")
        records = json.loads(response.headers["DUUI-Logs"])
        error = next(record for record in records if record["level"] == "ERROR")
        self.assertEqual(error["message"], "extract request failed")
        self.assertIn("RuntimeError: Media tool failed", error["stacktrace"])


class ProcessingErrorsTest(unittest.IsolatedAsyncioTestCase):
    async def post(self, operation, collect=True):
        headers = {"DUUI-Log-Collect": "true"} if collect else {}
        # Keep ASGITransport's default exception propagation so an unhandled
        # exception fails the test instead of silently becoming a plain-text 500.
        transport = httpx.ASGITransport(app=app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/v1/process", headers=headers, json={
                "operation": operation, "video": {"src": "dmlkZW8="}, "audio": "YXVkaW8=",
            })

    async def test_unexpected_worker_failures_return_json_and_collected_tracebacks(self):
        failures = (
            (OSError, "private filesystem detail"),
            (FileNotFoundError, "private missing output path"),
            (TypeError, "private unexpected FFprobe field"),
        )
        for operation, worker in (("extract", "extract_audio"),
                                  ("pipeline", "run_pipeline"), ("mux", "mux_audio")):
            for error_type, message in failures:
                failure = error_type(message)
                with self.subTest(operation=operation, failure=type(failure).__name__):
                    with patch.object(app, worker, side_effect=failure):
                        response = await self.post(operation)
                    self.assertEqual(response.status_code, 500)
                    self.assertEqual(response.headers["content-type"], "application/json")
                    self.assertEqual(response.json(), {"detail": "Internal media processing error"})
                    self.assertNotIn("private", response.text)
                    records = json.loads(response.headers["DUUI-Logs"])
                    error = next(record for record in records if record["level"] == "ERROR")
                    self.assertEqual(error["message"], f"Unexpected failure in {operation} request")
                    self.assertIn(f"{type(failure).__name__}: {failure}", error["stacktrace"])

    async def test_unexpected_failure_without_log_collection_returns_json(self):
        with patch.object(app, "extract_audio", side_effect=OSError("disk failure")):
            response = await self.post("extract", collect=False)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "Internal media processing error"})
        self.assertNotIn("DUUI-Logs", response.headers)

    async def test_malformed_ffprobe_output_returns_structured_error(self):
        for output, status in ((b"not JSON", 422), (b"[]", 500),
                               (b'{"streams": [null]}', 500), (b"\xff", 422)):
            with self.subTest(output=output), patch.object(media, "run", return_value=output):
                response = await self.post("extract")
            self.assertEqual(response.status_code, status)
            self.assertIsInstance(response.json()["detail"], str)
            records = json.loads(response.headers["DUUI-Logs"])
            self.assertTrue(any(record["level"] == "ERROR" and record["stacktrace"]
                                for record in records))

    async def test_missing_extracted_output_returns_structured_error(self):
        with patch.object(media, "probe", return_value={"has_audio": True}), \
                patch.object(media, "run", return_value=b""):
            response = await self.post("extract")
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "Internal media processing error"})
        records = json.loads(response.headers["DUUI-Logs"])
        self.assertTrue(any("FileNotFoundError" in (record.get("stacktrace") or "")
                            for record in records))


class StartupConfigurationTest(unittest.IsolatedAsyncioTestCase):
    async def test_missing_service_url_fails_at_startup(self):
        with patch.dict(pipeline.os.environ,
                        {"DUUI_FACE_ANON_URL": "http://face"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "DUUI_SPEAKER_ANON_URL"):
                async with app.app.router.lifespan_context(app.app):
                    pass

    async def test_malformed_service_url_fails_at_startup(self):
        with patch.dict(pipeline.os.environ, {
                "DUUI_FACE_ANON_URL": "http://face",
                "DUUI_SPEAKER_ANON_URL": "localhost:9716"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "DUUI_SPEAKER_ANON_URL"):
                async with app.app.router.lifespan_context(app.app):
                    pass

    async def test_explicit_service_urls_pass_startup(self):
        with patch.dict(pipeline.os.environ, {
                "DUUI_FACE_ANON_URL": "http://face:9714",
                "DUUI_SPEAKER_ANON_URL": "https://speaker:9716"}, clear=True):
            async with app.app.router.lifespan_context(app.app):
                pass


if __name__ == "__main__":
    unittest.main()
