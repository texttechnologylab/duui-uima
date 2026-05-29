from functools import lru_cache
import json
from enum import Enum
from typing import Any, List

import torch
import uvicorn
from cassis import load_typesystem
from fastapi import FastAPI, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from starlette.responses import JSONResponse

from opf import DecodeOptions, OPF


class DetectedSpan(BaseModel):
    """One detected privacy span returned by OPF."""

    label: str
    start: int
    end: int
    text: str
    placeholder: str


class SelectionRange(BaseModel):
    """Optional text selection inside the source document."""

    begin: int
    end: int


class DUUIRequest(BaseModel):
    """Request sent by DUUI and transformed by the Lua communication layer."""

    text: str
    options: dict[str, Any] = Field(default_factory=dict)
    selection: SelectionRange | None = None


class DUUIResponse(BaseModel):
    """Response of this annotator."""

    schema_version: int
    summary: dict[str, Any]
    text: str
    detected_spans: List[DetectedSpan]
    redacted_text: str
    warning: str | None = None
    selection: SelectionRange | None = None


class DUUIDocumentation(BaseModel):
    """Documentation response."""

    annotator_name: str
    version: str
    implementation_lang: str


class Settings(BaseSettings):
    """Runtime settings for the DUUI service."""

    duui_tool_name: str = "OpenAI Privacy Filter"
    duui_tool_version: str = "1.0"
    default_model: str | None = None


class RedactionMode(str, Enum):
    REPLACEMENT = "replacement"
    PSEUDO = "pseudo"


class PrivacyFilterService:
    """Class-based service wrapper for OPF redaction."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def split_options(self, options: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        return _split_options(options)

    def selection_from_options(
        self,
        request_selection: SelectionRange | None,
        options: dict[str, Any],
        *,
        text_length: int,
    ) -> SelectionRange | None:
        return _selection_from_options(
            request_selection,
            options,
            text_length=text_length,
        )

    def redact_text(
        self,
        text: str,
        request_selection: SelectionRange | None,
        options: dict[str, Any],
    ) -> DUUIResponse:
        return _redact_text(text, request_selection, options)


settings = Settings()
service = PrivacyFilterService(settings)
DEFAULT_PLACEHOLDER = "<REDACTED>"
DEFAULT_MODE = RedactionMode.REPLACEMENT.value
PSEUDO_MODE = RedactionMode.PSEUDO.value


app = FastAPI(
    docs_url="/api",
    redoc_url=None,
    title="OpenAI Privacy Filter",
    description="Text privacy redaction for TTLab DUUI",
    version="1.0",
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "Daniel Bundan",
        "url": "bundan.me",
        "email": "s1486849@stud.uni-frankfurt.de",
    },
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)

communication = "communication.lua"
with open(communication, 'rb') as f:
    communication = f.read().decode("utf-8")


# Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = 'typesystem.xml'
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)


# Get input / output of the annotator
@app.get("/v1/details/input_output")
def get_input_output() -> JSONResponse:
    json_item = {
        "inputs": [],
        "outputs": ["de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.Anomaly"]
    }

    json_compatible_item_data = jsonable_encoder(json_item)
    return JSONResponse(content=json_compatible_item_data)


# Get typesystem of this annotator
@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    # TODO remove cassis dependency, as only needed for typesystem at the moment?
    xml = typesystem.to_xml()
    xml_content = xml.encode("utf-8")

    return Response(
        content=xml_content,
        media_type="application/xml"
    )


# Return Lua communication script
@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return communication


# Return documentation info
@app.get("/v1/documentation")
def get_documentation() -> DUUIDocumentation:

    documentation = DUUIDocumentation(
        annotator_name=settings.duui_tool_name,
        version=settings.duui_tool_version,
        implementation_lang="Python",
    )
    return documentation


def _selection_from_options(
    request_selection: SelectionRange | None,
    options: dict[str, Any],
    *,
    text_length: int,
) -> SelectionRange | None:
    if request_selection is not None:
        begin = int(request_selection.begin)
        end = int(request_selection.end)
    else:
        selection = options.pop("selection", None)
        if isinstance(selection, dict):
            begin = selection.get("begin")
            end = selection.get("end")
        else:
            begin = options.pop("selection_begin", options.pop("selection_start", None))
            end = options.pop("selection_end", options.pop("selection_stop", None))

    if begin is None or end is None:
        return None

    begin = int(begin)
    end = int(end)
    if begin < 0 or end < begin or end > text_length:
        raise ValueError("selection must satisfy 0 <= begin <= end <= text length")
    return SelectionRange(begin=begin, end=end)


def _json_key(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _split_options(options: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    redactor_options: dict[str, Any] = {}
    decode_options: dict[str, Any] = {}

    for key, value in options.items():
        if key in {"decode", "selection", "selection_begin", "selection_end", "selection_start", "selection_stop"}:
            continue
        if key == "model":
            redactor_options["model"] = value
        elif key == "context_window_length":
            redactor_options["context_window_length"] = value
        elif key == "trim_whitespace":
            redactor_options["trim_whitespace"] = value
        elif key == "device":
            redactor_options["device"] = value
        elif key == "output_mode":
            redactor_options["output_mode"] = value
        elif key == "discard_overlapping_predicted_spans":
            redactor_options["discard_overlapping_predicted_spans"] = value
        elif key == "mode":
            redactor_options["mode"] = value
        elif key == "placeholder":
            redactor_options["placeholder"] = value
        elif key == "decode_mode":
            decode_options["decode_mode"] = value
        elif key in {"viterbi_calibration_path", "calibration_path"}:
            decode_options["viterbi_calibration_path"] = value
        elif key == "output_text_only":
            continue

    return redactor_options, decode_options


@lru_cache(maxsize=8)
def _build_redactor(options_json: str) -> OPF:
    options = json.loads(options_json)
    device = options.get("device")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    redactor = OPF(
        model=options.get("model", settings.default_model),
        context_window_length=options.get("context_window_length"),
        trim_whitespace=bool(options.get("trim_whitespace", True)),
        device=device,
        output_mode=options.get("output_mode", "typed"),
        discard_overlapping_predicted_spans=bool(
            options.get("discard_overlapping_predicted_spans", False)
        ),
        output_text_only=False,
    )

    return redactor


def _compose_replacement_text(
    text: str,
    spans: list[DetectedSpan],
    *,
    placeholder: str = DEFAULT_PLACEHOLDER,
) -> str:
    if not spans:
        return text

    redacted_parts: list[str] = []
    cursor = 0
    for span in sorted(spans, key=lambda item: (item.start, item.end)):
        if span.start < cursor:
            continue
        redacted_parts.append(text[cursor:span.start])
        redacted_parts.append(placeholder)
        cursor = max(cursor, span.end)
    redacted_parts.append(text[cursor:])
    return "".join(redacted_parts)


def _detect_spans(payload: Any, *, offset: int = 0) -> list[DetectedSpan]:
    detected_spans: list[DetectedSpan] = []
    for span in payload:
        if isinstance(span, dict):
            label = span.get("label")
            start = span.get("start")
            end = span.get("end")
            text = span.get("text")
            placeholder = span.get("placeholder")
        else:
            label = getattr(span, "label", None)
            start = getattr(span, "start", None)
            end = getattr(span, "end", None)
            text = getattr(span, "text", None)
            placeholder = getattr(span, "placeholder", None)

        detected_spans.append(
            DetectedSpan(
                label=str(label),
                start=int(start) + offset,
                end=int(end) + offset,
                text=str(text),
                placeholder=str(placeholder),
            )
        )

    return detected_spans


def _render_pseudo_response(
    *,
    text: str,
    request_selection: SelectionRange | None,
    options: dict[str, Any],
) -> DUUIResponse:
    summary = {
        "mode": PSEUDO_MODE,
        "span_count": 0,
        "by_label": {},
        "decoded_mismatch": False,
    }
    return DUUIResponse(
        schema_version=1,
        summary=summary,
        text=text,
        detected_spans=[],
        redacted_text=text,
        warning="pseudo mode is a stub and returns the input unchanged",
        selection=request_selection,
    )


def _redact_text(text: str, request_selection: SelectionRange | None, options: dict[str, Any]) -> DUUIResponse:
    constructor_options, decode_options = _split_options(options)
    mode = str(constructor_options.get("mode", DEFAULT_MODE))
    placeholder = str(constructor_options.get("placeholder", DEFAULT_PLACEHOLDER))

    if mode == PSEUDO_MODE:
        return _render_pseudo_response(
            text=text,
            request_selection=request_selection,
            options=constructor_options,
        )

    redactor = _build_redactor(_json_key(constructor_options))
    decode = DecodeOptions(**decode_options) if decode_options else None

    selected_text = text
    selection_offset = 0
    if request_selection is not None:
        selection_offset = request_selection.begin
        selected_text = text[request_selection.begin:request_selection.end]

    result = redactor.redact(selected_text, decode=decode)

    if isinstance(result, str):
        redacted_text = result if request_selection is None else (
            text[:request_selection.begin] + result + text[request_selection.end:]
        )
        return DUUIResponse(
            schema_version=1,
            summary={
                "mode": mode,
                "span_count": 0,
                "by_label": {},
                "decoded_mismatch": False,
            },
            text=text,
            detected_spans=[],
            redacted_text=redacted_text,
            warning=None,
            selection=request_selection,
        )

    detected_spans = _detect_spans(result.detected_spans, offset=selection_offset)
    redacted_text = _compose_replacement_text(
        selected_text,
        [
            DetectedSpan(
                label=span.label,
                start=span.start - selection_offset,
                end=span.end - selection_offset,
                text=span.text,
                placeholder=placeholder,
            )
            for span in detected_spans
        ],
        placeholder=placeholder,
    )
    if request_selection is not None:
        redacted_text = text[:request_selection.begin] + redacted_text + text[request_selection.end:]

    return DUUIResponse(
        schema_version=int(result.schema_version),
        summary={**dict(result.summary), "mode": mode},
        text=text,
        detected_spans=detected_spans,
        redacted_text=redacted_text,
        warning=result.warning,
        selection=request_selection,
    )


# Process request from DUUI
@app.post("/v1/process")
def post_process(request: DUUIRequest) -> DUUIResponse:
    selection = service.selection_from_options(
        request.selection,
        dict(request.options),
        text_length=len(request.text),
    )
    return service.redact_text(request.text, selection, dict(request.options))


if __name__ == "__main__":
    uvicorn.run("duui_opf:app", host="0.0.0.0", port=9714, workers=1)
