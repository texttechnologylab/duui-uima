from __future__ import annotations

import logging
import json
from functools import lru_cache
from typing import Any, List, Optional

import torch
import uvicorn
from cassis import load_typesystem
from fastapi import FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from transformers import pipeline as hf_pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_MODEL = "openai/privacy-filter"

MODE_REMOVE = "remove"
MODE_PLACEHOLDER = "placeholder"  # default: replace with [category]
MODE_PSEUDO = "pseudo"            # TODO: not yet supported


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class DetectedSpan(BaseModel):
    label: str
    start: int
    end: int
    text: str
    placeholder: str  # replacement text used; empty string for remove mode


class DUUIRequest(BaseModel):
    text: str
    options: dict[str, Any] = Field(default_factory=dict)
    selection: Optional[dict] = None

    @field_validator("options", mode="before")
    @classmethod
    def coerce_options(cls, v: Any) -> dict:
        if v is None or isinstance(v, list):
            return {}
        if not isinstance(v, dict):
            return {}
        return v

    @field_validator("text", mode="before")
    @classmethod
    def coerce_text(cls, v: Any) -> str:
        return "" if v is None else str(v)


class DUUIResponse(BaseModel):
    text: str
    detected_spans: List[DetectedSpan]
    redacted_text: str
    warning: Optional[str] = None


class DUUIDocumentation(BaseModel):
    annotator_name: str
    version: str
    implementation_lang: str


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    duui_tool_name: str = "DUUI Anonymize"
    duui_tool_version: str = "1.0"
    default_model: str = DEFAULT_MODEL


settings = Settings()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    docs_url="/api",
    redoc_url=None,
    title="DUUI Anonymize",
    description="PII detection and redaction for TTLab DUUI using openai/privacy-filter",
    version="1.0",
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "Ali Abusaleh",
        "url": "https://www.texttechnologylab.org",
        "email": "abusaleh@em.uni-frankfurt.de",
    },
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    body = await request.body()
    logger.error("422 validation errors: %s", exc.errors())
    logger.error("Raw body: %s", body.decode("utf-8", errors="replace"))
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder({"detail": exc.errors(), "body": body.decode("utf-8", errors="replace")}),
    )


# ---------------------------------------------------------------------------
# Static assets
# ---------------------------------------------------------------------------

with open("communication.lua", "rb") as _f:
    _communication_lua: str = _f.read().decode("utf-8")

with open("typesystem.xml", "rb") as _f:
    _typesystem = load_typesystem(_f)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/details/input_output")
def get_input_output() -> JSONResponse:
    return JSONResponse(content=jsonable_encoder({
        "inputs": [],
        "outputs": ["de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.Anomaly"],
    }))


@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    return Response(content=_typesystem.to_xml().encode("utf-8"), media_type="application/xml")


@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return _communication_lua


@app.get("/v1/documentation")
def get_documentation() -> DUUIDocumentation:
    return DUUIDocumentation(
        annotator_name=settings.duui_tool_name,
        version=settings.duui_tool_version,
        implementation_lang="Python",
    )


@app.post("/v1/process")
async def post_process(raw_request: Request) -> DUUIResponse:
    body = await raw_request.body()
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RequestValidationError([{"type": "json_invalid", "loc": ("body",), "msg": str(exc), "input": body}])
    request = DUUIRequest.model_validate(data)
    return _process(request)


# ---------------------------------------------------------------------------
# Business logic
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _load_pipeline(model: str, device: str):
    dev = 0 if device == "cuda" else -1
    logger.info("Loading pipeline: model=%s device=%s", model, device)
    return hf_pipeline(
        task="token-classification",
        model=model,
        aggregation_strategy="simple",
        device=dev,
    )


def _resolve_selection(options: dict[str, Any], text_length: int) -> Optional[tuple[int, int]]:
    sel = options.get("selection")
    if isinstance(sel, dict):
        begin = sel.get("begin", sel.get("start"))
        end = sel.get("end", sel.get("stop"))
    else:
        begin = options.get("selection_begin", options.get("selection_start"))
        end = options.get("selection_end", options.get("selection_stop"))

    if begin is None or end is None:
        return None
    begin, end = int(begin), int(end)
    if begin < 0 or end < begin or end > text_length:
        raise ValueError(f"selection must satisfy 0 <= begin <= end <= {text_length}")
    return begin, end


def _build_redacted(text: str, spans: list[DetectedSpan], mode: str) -> str:
    """Apply mode transformation to text using already-computed spans."""
    if not spans:
        return text
    parts: list[str] = []
    cursor = 0
    for span in sorted(spans, key=lambda s: s.start):
        if span.start < cursor:
            continue
        parts.append(text[cursor:span.start])
        if mode == MODE_PLACEHOLDER:
            parts.append(span.placeholder)   # e.g. [private_person]
        # MODE_REMOVE: append nothing - the PII is deleted
        cursor = span.end
    parts.append(text[cursor:])
    return "".join(parts)


def _process(request: DUUIRequest) -> DUUIResponse:
    options = request.options
    model = str(options.get("model", settings.default_model))
    device = str(options.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    mode = str(options.get("mode", MODE_PLACEHOLDER))

    print(f"Processing request: model={model} device={device} mode={mode} text_length={len(request.text)}")
    
    # pseudo mode - not yet supported
    if mode == MODE_PSEUDO:
        return DUUIResponse(
            text=request.text,
            detected_spans=[],
            redacted_text=request.text,
            warning="pseudo mode is not yet supported - input returned unchanged",
        )

    if not request.text:
        return DUUIResponse(text="", detected_spans=[], redacted_text="")

    sel = _resolve_selection(options, text_length=len(request.text))
    selected_text = request.text[sel[0]:sel[1]] if sel else request.text
    offset = sel[0] if sel else 0

    pipe = _load_pipeline(model, device)
    raw = pipe(selected_text)

    spans = [
        DetectedSpan(
            label=item["entity_group"],
            start=int(item["start"]) + offset,
            end=int(item["end"]) + offset,
            text=str(item["word"]).strip(),
            placeholder=f"[{item['entity_group']}]" if mode == MODE_PLACEHOLDER else "",
        )
        for item in raw
    ]

    redacted_text = _build_redacted(request.text, spans, mode)
    if sel is not None:
        # only the selected window was processed; rebuild full text around it
        local_spans = [
            DetectedSpan(label=s.label, start=s.start - offset, end=s.end - offset,
                         text=s.text, placeholder=s.placeholder)
            for s in spans
        ]
        redacted_window = _build_redacted(selected_text, local_spans, mode)
        redacted_text = request.text[:sel[0]] + redacted_window + request.text[sel[1]:]

    return DUUIResponse(
        text=request.text,
        detected_spans=spans,
        redacted_text=redacted_text,
    )


if __name__ == "__main__":
    uvicorn.run("duui_anonymize:app", host="0.0.0.0", port=9714, workers=1)
