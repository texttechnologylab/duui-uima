import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, Field, ValidationError

from media import extract_audio, mux_audio
from pipeline import required_service_urls, run_pipeline


ROOT = Path(__file__).parent


@asynccontextmanager
async def lifespan(_: FastAPI):
    required_service_urls()
    yield


app = FastAPI(
    lifespan=lifespan,
    docs_url="/api",
    redoc_url=None,
    title="DUUI Video Anonymization",
    description="Video anonymization for TTLab DUUI",
    version="1.0",
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "Tim Wolf",
        "url": "https://www.texttechnologylab.org",
        "email": "T.Wolf@em.uni-frankfurt.de",
    },
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)


class Video(BaseModel):
    src: str
    mimetype: str | None = None
    length: float = -1
    fps: float = -1
    begin: int = 0
    end: int = 0


class ProcessRequest(BaseModel):
    operation: Literal["pipeline", "extract", "mux"] = "pipeline"
    video: Video
    audio: str | None = None
    options: dict[str, str] = Field(default_factory=dict)


@app.get("/v1/typesystem")
def typesystem() -> Response:
    return Response((ROOT / "typesystem.xml").read_bytes(), media_type="application/xml")


@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def communication_layer() -> str:
    return (ROOT / "communication.lua").read_text(encoding="utf-8")


@app.get("/v1/documentation")
def documentation() -> dict:
    return {"annotator_name": "duui-video-anon", "version": "1.0.0",
            "implementation_lang": "Python"}


@app.get("/v1/details/input_output")
def input_output() -> dict:
    return {"inputs": ["org.texttechnologylab.annotation.type.Video"],
            "outputs": ["org.texttechnologylab.annotation.type.Video"]}


@app.get("/v1/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/process")
async def process(raw_request: Request) -> dict:
    try:
        data = await raw_request.json()
        # DUUI's Lua output stream can arrive as a JSON-encoded string.
        if isinstance(data, str):
            data = json.loads(data)
        if isinstance(data, dict) and data.get("options") == []:
            data["options"] = {}
        request = ProcessRequest.model_validate(data)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Invalid DUUI media request") from exc
    try:
        if request.operation == "extract":
            audio = await run_in_threadpool(extract_audio, request.video.src)
            return {"operation": "extract", "audio": audio}
        if request.operation == "pipeline":
            src, length, fps = await run_in_threadpool(
                run_pipeline, request.video.src, request.video.mimetype, request.options)
            return {"operation": "pipeline", "video": {
                "src": src, "length": length, "fps": fps,
                "begin": request.video.begin, "end": request.video.end
            }}
        if request.audio is None:
            raise ValueError("Mux requires anonymized audio")
        src, length, fps = await run_in_threadpool(
            mux_audio, request.video.src, request.audio)
        return {"operation": "mux", "video": {
            "src": src, "length": length, "fps": fps,
            "begin": request.video.begin, "end": request.video.end
        }}
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
