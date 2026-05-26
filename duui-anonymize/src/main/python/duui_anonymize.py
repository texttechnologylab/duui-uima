from __future__ import annotations

import logging
from functools import lru_cache
import json
from enum import Enum
from typing import Any, List, Optional, Union

import torch
import uvicorn
from cassis import load_typesystem
from fastapi import FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

from opf import DecodeOptions, OPF

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class DetectedSpan(BaseModel):
    label: str
    start: int
    end: int
    text: str
    placeholder: str


class SelectionRange(BaseModel):
    begin: int
    end: int


class DUUIRequest(BaseModel):
    text: str
    options: dict[str, Any] = Field(default_factory=dict)
    selection: Optional[SelectionRange] = None

    @field_validator("options", mode="before")
    @classmethod
    def coerce_options(cls, v: Any) -> dict:
        """
        Lua JSON libraries encode empty tables as [] instead of {}.
        Accept None, empty list, or any list by falling back to an empty dict.
        """
        if v is None:
            return {}
        if isinstance(v, list):
            return {}
        if not isinstance(v, dict):
            return {}
        return v

    @field_validator("text", mode="before")
    @classmethod
    def coerce_text(cls, v: Any) -> str:
        """Tolerate Java String objects forwarded via LuaJ."""
        if v is None:
            return ""
        return str(v)


class DUUIResponse(BaseModel):
    schema_version: int
    summary: dict[str, Any]
    text: str
    detected_spans: List[DetectedSpan]
    redacted_text: str
    warning: Optional[str] = None
    selection: Optional[SelectionRange] = None


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
    default_model: Optional[str] = None


class RedactionMode(str, Enum):
    REPLACEMENT = "replacement"
    PSEUDO = "pseudo"


settings = Settings()
DEFAULT_PLACEHOLDER = "<REDACTED>"
DEFAULT_MODE = RedactionMode.REPLACEMENT.value
PSEUDO_MODE = RedactionMode.PSEUDO.value

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    docs_url="/api",
    redoc_url=None,
    title="DUUI Anonymize",
    description="Text anonymization / PII redaction for TTLab DUUI using the OpenAI Privacy Filter",
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
    logger.error("422 Unprocessable Entity — validation errors: %s", exc.errors())
    logger.error("Raw request body: %s", body.decode("utf-8", errors="replace"))
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder({"detail": exc.errors(), "body": body.decode("utf-8", errors="replace")}),
    )


# ---------------------------------------------------------------------------
# Static assets loaded at startup
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
    xml_content = _typesystem.to_xml().encode("utf-8")
    return Response(content=xml_content, media_type="application/xml")


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
    # DUUI does not set Content-Type: application/json, so FastAPI will not
    # deserialize the body automatically. We parse it manually here.
    body = await raw_request.body()
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RequestValidationError([{"type": "json_invalid", "loc": ("body",), "msg": str(exc), "input": body}])
    request = DUUIRequest.model_validate(data)
    options = dict(request.options)
    selection = _resolve_selection(request.selection, options, text_length=len(request.text))
    return _redact_text(request.text, selection, options)


# ---------------------------------------------------------------------------
# Business logic
# ---------------------------------------------------------------------------

def _resolve_selection(
    request_selection: Optional[SelectionRange],
    options: dict[str, Any],
    *,
    text_length: int,
) -> Optional[SelectionRange]:
    if request_selection is not None:
        begin = int(request_selection.begin)
        end_val = int(request_selection.end)
    else:
        raw = options.pop("selection", None)
        if isinstance(raw, dict):
            begin = raw.get("begin")
            end_val = raw.get("end")
        else:
            begin = options.pop("selection_begin", options.pop("selection_start", None))
            end_val = options.pop("selection_end", options.pop("selection_stop", None))

    if begin is None or end_val is None:
        return None

    begin = int(begin)
    end_val = int(end_val)
    if begin < 0 or end_val < begin or end_val > text_length:
        raise ValueError("selection must satisfy 0 <= begin <= end <= text length")
    return SelectionRange(begin=begin, end=end_val)


def _json_cache_key(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _split_options(options: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    _skip = {"decode", "selection", "selection_begin", "selection_end", "selection_start", "selection_stop"}
    _redactor_keys = {"model", "context_window_length", "trim_whitespace", "device",
                      "output_mode", "discard_overlapping_predicted_spans", "mode", "placeholder"}
    _decode_keys = {"decode_mode", "viterbi_calibration_path", "calibration_path"}

    redactor_opts: dict[str, Any] = {}
    decode_opts: dict[str, Any] = {}
    for key, value in options.items():
        if key in _skip:
            continue
        if key in _redactor_keys:
            redactor_opts[key] = value
        elif key in _decode_keys:
            k = "viterbi_calibration_path" if key == "calibration_path" else key
            decode_opts[k] = value
    return redactor_opts, decode_opts


@lru_cache(maxsize=8)
def _build_redactor(options_json: str) -> OPF:
    opts = json.loads(options_json)
    device = opts.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    return OPF(
        model=opts.get("model", settings.default_model),
        context_window_length=opts.get("context_window_length"),
        trim_whitespace=bool(opts.get("trim_whitespace", True)),
        device=device,
        output_mode=opts.get("output_mode", "typed"),
        discard_overlapping_predicted_spans=bool(opts.get("discard_overlapping_predicted_spans", False)),
        output_text_only=False,
    )


def _compose_redacted(text: str, spans: list[DetectedSpan], *, placeholder: str) -> str:
    if not spans:
        return text
    parts: list[str] = []
    cursor = 0
    for span in sorted(spans, key=lambda s: (s.start, s.end)):
        if span.start < cursor:
            continue
        parts.append(text[cursor:span.start])
        parts.append(placeholder)
        cursor = max(cursor, span.end)
    parts.append(text[cursor:])
    return "".join(parts)


def _parse_spans(payload: Any, *, offset: int = 0) -> list[DetectedSpan]:
    spans: list[DetectedSpan] = []
    for item in payload:
        if isinstance(item, dict):
            label = item.get("label")
            start = item.get("start")
            end_val = item.get("end")
            text = item.get("text")
            placeholder = item.get("placeholder")
        else:
            label = getattr(item, "label", None)
            start = getattr(item, "start", None)
            end_val = getattr(item, "end", None)
            text = getattr(item, "text", None)
            placeholder = getattr(item, "placeholder", None)
        spans.append(DetectedSpan(
            label=str(label),
            start=int(start) + offset,
            end=int(end_val) + offset,
            text=str(text),
            placeholder=str(placeholder),
        ))
    return spans


def _redact_text(
    text: str,
    selection: Optional[SelectionRange],
    options: dict[str, Any],
) -> DUUIResponse:
    redactor_opts, decode_opts = _split_options(options)
    mode = str(redactor_opts.get("mode", DEFAULT_MODE))
    placeholder = str(redactor_opts.get("placeholder", DEFAULT_PLACEHOLDER))

    if mode == PSEUDO_MODE:
        return DUUIResponse(
            schema_version=1,
            summary={"mode": PSEUDO_MODE, "span_count": 0, "by_label": {}, "decoded_mismatch": False},
            text=text,
            detected_spans=[],
            redacted_text=text,
            warning="pseudo mode returns the input unchanged",
            selection=selection,
        )

    redactor = _build_redactor(_json_cache_key(redactor_opts))
    decode = DecodeOptions(**decode_opts) if decode_opts else None

    selected_text = text
    offset = 0
    if selection is not None:
        offset = selection.begin
        selected_text = text[selection.begin:selection.end]

    result = redactor.redact(selected_text, decode=decode)

    if isinstance(result, str):
        redacted_text = result if selection is None else (
            text[:selection.begin] + result + text[selection.end:]
        )
        return DUUIResponse(
            schema_version=1,
            summary={"mode": mode, "span_count": 0, "by_label": {}, "decoded_mismatch": False},
            text=text,
            detected_spans=[],
            redacted_text=redacted_text,
            selection=selection,
        )

    detected_spans = _parse_spans(result.detected_spans, offset=offset)
    local_spans = [
        DetectedSpan(label=s.label, start=s.start - offset, end=s.end - offset,
                     text=s.text, placeholder=placeholder)
        for s in detected_spans
    ]
    redacted_text = _compose_redacted(selected_text, local_spans, placeholder=placeholder)
    if selection is not None:
        redacted_text = text[:selection.begin] + redacted_text + text[selection.end:]

    return DUUIResponse(
        schema_version=int(result.schema_version),
        summary={**dict(result.summary), "mode": mode},
        text=text,
        detected_spans=detected_spans,
        redacted_text=redacted_text,
        warning=result.warning,
        selection=selection,
    )


if __name__ == "__main__":
    uvicorn.run("duui_anonymize:app", host="0.0.0.0", port=9714, workers=1)
