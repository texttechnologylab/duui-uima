from __future__ import annotations

import json
import logging
from functools import lru_cache
from threading import Lock
from time import time
from typing import Any, Dict, Final, Iterable, List, Optional, Tuple

import torch
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
import os

os.environ["CUDA_LAUNCH_BLOCKING"]="1"
os.environ["TORCH_USE_CUDA_DSA"]="1"

try:
    from pydantic_settings import BaseSettings
except ImportError:  # pydantic v1 fallback
    from pydantic import BaseSettings  # type: ignore

from time_recognition_backend import MODEL_REGISTRY, create_time_recognizer, resolve_model_name


model_lock = Lock()


def _string_list(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


class UimaSentence(BaseModel):
    text: str
    begin: int
    end: int


class UimaSentenceSelection(BaseModel):
    selection: str
    sentences: List[UimaSentence]


class Settings(BaseSettings):
    annotator_name: str = "DUUI TimeX3"
    annotator_version: str = "0.1.0"
    log_level: str = "INFO"

    # Exactly one time recognizer per container.
    # Use one registry alias or one exact Hugging Face model id.
    # Comma-separated lists and "all" are intentionally rejected.
    model_name: str = "microsoft"
    model_version: str = "latest"
    model_cache_size: int = 1
    model_source: str = ""

    # One running container instance is bound to exactly one language.
    model_lang: str = "de"

    threshold: float = 0.0
    batch_size: int = 8

    typesystem_filename: str = "TypeSystemTime.xml"
    lua_communication_script_filename: str = "duui_time.lua"

    class Config:
        env_prefix = ""
        case_sensitive = False


settings = Settings()
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger(__name__)

lru_cache_with_size = lru_cache(maxsize=max(1, settings.model_cache_size))
device = "cuda:0" if torch.cuda.is_available() else "cpu"
logger.info("USING %s", device)


class DUUIRequest(BaseModel):
    doc_len: Optional[int] = None
    lang: Optional[str] = None
    selections: List[UimaSentenceSelection]

    # Runtime parameters passed through the Lua layer from .withParameter(...).
    threshold: Optional[float] = None
    batch_size: Optional[int] = None

    # Optional reference date/time for relative temporal expressions.
    # Accepts ISO-like values such as 2026-06-09 or 2026-06-09T00:00:00+02:00.
    document_creation_time: Optional[str] = None
    reference_time: Optional[str] = None

    # Optional service URLs passed via DUUI .withParameter(...).
    # They are relevant only for the duckling and sutime backends.
    duckling_url: Optional[str] = None
    duckling_timezone: Optional[str] = None
    corenlp_url: Optional[str] = None


class DocumentModification(BaseModel):
    user: str
    timestamp: int
    comment: str


class AnnotationMeta(BaseModel):
    name: str
    version: str
    modelName: str
    modelVersion: str


TIME_BASE_TYPE: Final[str] = "org.texttechnologylab.annotation.semaf.isotimeml.TimeX3"
TIME_TYPES: Final[Dict[str, str]] = {
    "DATE": "org.texttechnologylab.annotation.semaf.isotimeml.time.Date",
    "TIME": "org.texttechnologylab.annotation.semaf.isotimeml.time.Time",
    "DURATION": "org.texttechnologylab.annotation.semaf.isotimeml.time.Duration",
    "SET": "org.texttechnologylab.annotation.semaf.isotimeml.time.Set",
    "UNKNOWN": TIME_BASE_TYPE,
}


class Timex3Annotation(BaseModel):
    begin: int
    end: int
    value: Optional[str] = None
    timex_type: str
    time_type: str = TIME_BASE_TYPE
    covered_text: Optional[str] = None
    score: Optional[float] = None
    model_name: Optional[str] = None

    # Optional IsoTimeML/TimeX3 features.
    function_in_document: Optional[str] = None
    temporal_function: Optional[bool] = None
    quant: Optional[str] = None
    freq: Optional[str] = None


class DUUIResponse(BaseModel):
    meta: AnnotationMeta
    modification_meta: DocumentModification
    begin: List[int]
    end: List[int]
    results: List[str]
    factors: List[float]
    len_results: List[int]
    timex_value: List[Optional[str]]
    time_type: List[str]
    covered_text: List[str]
    model: List[str]
    tags: List[Timex3Annotation]
    model_name: str
    model_version: str
    model_source: str
    model_lang: str


class TextImagerCapability(BaseModel):
    supported_languages: List[str]
    reproducible: bool


class TextImagerDocumentation(BaseModel):
    annotator_name: str
    version: str
    implementation_lang: str
    meta: Dict[str, Any]
    docker_container_id: Optional[str]
    parameters: Dict[str, Any]
    capability: TextImagerCapability
    implementation_specific: Optional[str]


def read_required_text_file(filename: str) -> str:
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
    if not content.strip():
        raise RuntimeError(f"Required Lua communication script is empty: {filename}")
    return content


def read_required_binary_file(filename: str) -> bytes:
    with open(filename, "rb") as f:
        content = f.read()
    if not content.strip():
        raise RuntimeError(f"Required UIMA type system XML is empty: {filename}")
    return content


lua_communication_script = read_required_text_file(settings.lua_communication_script_filename)
type_system = read_required_binary_file(settings.typesystem_filename)


app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title=settings.annotator_name,
    description="DUUI TimeX3 annotator built from the DUUI NER template",
    version=settings.annotator_version,
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "TTLab Team",
        "url": "https://texttechnologylab.org",
        "email": "bagci@em.uni-frankfurt.de",
    },
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)


@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    return Response(content=type_system, media_type="application/xml")


@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return lua_communication_script


@app.get("/v1/documentation")
def get_documentation() -> TextImagerDocumentation:
    selected_model = get_selected_model_name(settings.model_name)
    _, selected_cfg = resolve_model_name(selected_model)
    return TextImagerDocumentation(
        annotator_name=settings.annotator_name,
        version=settings.annotator_version,
        implementation_lang="Python",
        meta={
            "device": device,
            "available_models": MODEL_REGISTRY,
            "selected_model": selected_model,
            "selected_model_id": selected_cfg.get("model_id", selected_model),
            "backend_module": "time_recognition_backend.py",
            "time_types": TIME_TYPES,
        },
        docker_container_id=None,
        parameters={
            "model_name": "exactly one registry alias or one exact Hugging Face model id",
            "model_lang": "exactly one language per running container instance",
            "threshold": settings.threshold,
            "batch_size": settings.batch_size,
            "document_creation_time": "optional ISO reference date/time for relative expressions",
        },
        capability=TextImagerCapability(supported_languages=[settings.model_lang], reproducible=True),
        implementation_specific=None,
    )


def get_selected_model_name(model_name: str) -> str:
    """Return exactly one configured recognizer for this container."""
    selected = (model_name or "").strip()
    if not selected:
        selected = "microsoft"

    if selected.lower() == "all" or "," in selected:
        supported = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            "This DUUI container supports exactly one MODEL_NAME. "
            "Start one container per recognizer/language instead of using 'all' or comma-separated lists. "
            f"Supported aliases: {supported}"
        )

    alias, _ = resolve_model_name(selected)
    return alias


def validate_language(request_language: Optional[str]) -> str:
    service_language = (settings.model_lang or "").strip().lower()
    if not service_language or service_language == "multi":
        raise ValueError(
            "MODEL_LANG must be one concrete language for this DUUI component, "
            "for example MODEL_LANG=de or MODEL_LANG=en."
        )

    if request_language is None or not str(request_language).strip():
        return service_language

    request_language_normalized = str(request_language).strip().lower()
    if request_language_normalized != service_language:
        raise ValueError(
            f"This service was started for language '{service_language}', "
            f"but the request uses language '{request_language_normalized}'. "
            "Start a separate service instance for another language."
        )
    return service_language


@lru_cache_with_size
def load_model(model_name: str, language: str):
    return create_time_recognizer(model_name, language=language, device=device)


def fix_unicode_problems(text: str) -> str:
    return text.encode("utf-16", "surrogatepass").decode("utf-16", "surrogateescape")


def iter_batches(items: List[UimaSentence], batch_size: int) -> Iterable[List[UimaSentence]]:
    size = max(1, int(batch_size))
    for start in range(0, len(items), size):
        yield items[start:start + size]


def process_selection(
        model_name: str,
        selection: UimaSentenceSelection,
        language: str,
        threshold: float,
        batch_size: int,
        document_creation_time: Optional[str],
        duckling_url: Optional[str] = None,
        duckling_timezone: Optional[str] = None,
        corenlp_url: Optional[str] = None,
) -> Dict[str, Any]:
    begin: List[int] = []
    end: List[int] = []
    results_out: List[str] = []
    factors: List[float] = []
    len_results: List[int] = []
    timex_value_out: List[Optional[str]] = []
    time_type_out: List[str] = []
    covered_text_out: List[str] = []
    model_out: List[str] = []
    tags: List[Timex3Annotation] = []

    for s in selection.sentences:
        s.text = fix_unicode_problems(s.text)

    for batch in iter_batches(selection.sentences, batch_size):
        texts = [s.text for s in batch]

        with model_lock:
            recognizer = load_model(model_name, language)
            predictions = recognizer.predict(
                texts,
                language=language,
                document_creation_time=document_creation_time,
                threshold=threshold,
                batch_size=batch_size,
                duckling_url=duckling_url,
                duckling_timezone=duckling_timezone,
                corenlp_url=corenlp_url,
            )

        for sentence, sentence_times in zip(batch, predictions):
            for ent in sentence_times:
                rel_start = int(ent["start"])
                rel_end = int(ent["end"])
                if rel_end <= rel_start:
                    continue

                abs_begin = sentence.begin + rel_start
                abs_end = sentence.begin + rel_end
                timex_type = str(ent.get("timex_type") or ent.get("label") or "UNKNOWN")
                timex_value = ent.get("value")
                score = float(ent.get("score", 1.0))
                covered = str(ent.get("text", sentence.text[rel_start:rel_end]))
                time_type = str(ent.get("time_type") or TIME_TYPES.get(timex_type, TIME_BASE_TYPE))
                entity_model_name = str(ent.get("model_name", model_name))

                tag = Timex3Annotation(
                    begin=abs_begin,
                    end=abs_end,
                    value=timex_value,
                    timex_type=timex_type,
                    time_type=time_type,
                    covered_text=covered,
                    score=score,
                    model_name=entity_model_name,
                    function_in_document=ent.get("function_in_document"),
                    temporal_function=ent.get("temporal_function"),
                    quant=ent.get("quant"),
                    freq=ent.get("freq"),
                )

                begin.append(abs_begin)
                end.append(abs_end)
                results_out.append(timex_type)
                factors.append(score)
                len_results.append(1)
                timex_value_out.append(timex_value)
                time_type_out.append(time_type)
                covered_text_out.append(covered)
                model_out.append(entity_model_name)
                tags.append(tag)

    return {
        "begin": begin,
        "end": end,
        "results": results_out,
        "factors": factors,
        "len_results": len_results,
        "timex_value": timex_value_out,
        "time_type": time_type_out,
        "covered_text": covered_text_out,
        "model": model_out,
        "tags": tags,
    }


def model_meta_values(model_name: str) -> Tuple[str, str]:
    _, cfg = resolve_model_name(model_name)
    model_source = settings.model_source or cfg.get("model_source", "")
    model_lang = settings.model_lang
    return model_source, model_lang


def _validate_duui_request_payload(payload: Any) -> DUUIRequest:
    """Validate DUUI payloads that may arrive as object or JSON-encoded string.

    Some DUUI/Lua combinations send the serialized Lua output as a JSON string,
    e.g. "{\"selections\":[...]}" instead of a JSON object.
    FastAPI/Pydantic then returns 422 before our code runs. Reading the raw
    request body and decoding once or twice makes the endpoint compatible with
    both shapes.
    """
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")

    if isinstance(payload, str):
        payload = payload.strip()
        if not payload:
            raise ValueError("Request body is empty.")
        payload = json.loads(payload)

    if isinstance(payload, str):
        payload = json.loads(payload)

    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    if hasattr(DUUIRequest, "model_validate"):
        return DUUIRequest.model_validate(payload)
    return DUUIRequest.parse_obj(payload)


@app.post("/v1/process", response_model=DUUIResponse)
async def post_process(raw_request: Request):
    try:
        request = _validate_duui_request_payload(await raw_request.body())
    except Exception as ex:
        return JSONResponse(status_code=400, content={"message": f"Invalid request body: {ex}"})
    if not request.selections:
        return JSONResponse(status_code=400, content={"message": "The request must contain sentence selections."})

    try:
        language = validate_language(request.lang)
        model_name = get_selected_model_name(settings.model_name)
    except Exception as ex:
        return JSONResponse(status_code=400, content={"message": str(ex)})

    effective_threshold = request.threshold if request.threshold is not None else settings.threshold
    effective_batch_size = request.batch_size if request.batch_size is not None else settings.batch_size
    reference_time = request.document_creation_time or request.reference_time
    duckling_url = request.duckling_url or None
    duckling_timezone = request.duckling_timezone or None
    corenlp_url = request.corenlp_url or None

    if effective_threshold < 0.0 or effective_threshold > 1.0:
        return JSONResponse(status_code=400, content={"message": "threshold must be between 0.0 and 1.0"})
    if effective_batch_size < 1:
        return JSONResponse(status_code=400, content={"message": "batch_size must be >= 1"})

    modification_timestamp_seconds = int(time())
    meta_model_name = model_name
    meta = AnnotationMeta(
        name=settings.annotator_name,
        version=settings.annotator_version,
        modelName=meta_model_name,
        modelVersion=settings.model_version,
    )
    modification_meta = DocumentModification(
        user=settings.annotator_name,
        timestamp=modification_timestamp_seconds,
        comment=f"{settings.annotator_name} ({settings.annotator_version})",
    )

    begin: List[int] = []
    end: List[int] = []
    len_results: List[int] = []
    results: List[str] = []
    factors: List[float] = []
    timex_value: List[Optional[str]] = []
    time_type: List[str] = []
    covered_text: List[str] = []
    model: List[str] = []
    tags: List[Timex3Annotation] = []

    try:
        for selection in request.selections:
            processed = process_selection(
                model_name=model_name,
                selection=selection,
                language=language,
                threshold=effective_threshold,
                batch_size=effective_batch_size,
                document_creation_time=reference_time,
                duckling_url=duckling_url,
                duckling_timezone=duckling_timezone,
                corenlp_url=corenlp_url,
            )
            begin += processed["begin"]
            end += processed["end"]
            len_results += processed["len_results"]
            results += processed["results"]
            factors += processed["factors"]
            timex_value += processed["timex_value"]
            time_type += processed["time_type"]
            covered_text += processed["covered_text"]
            model += processed["model"]
            tags += processed["tags"]

        model_source, model_lang = model_meta_values(model_name)
        return DUUIResponse(
            meta=meta,
            modification_meta=modification_meta,
            begin=begin,
            end=end,
            results=results,
            factors=factors,
            len_results=len_results,
            timex_value=timex_value,
            time_type=time_type,
            covered_text=covered_text,
            model=model,
            tags=tags,
            model_name=meta_model_name,
            model_version=settings.model_version,
            model_source=model_source,
            model_lang=model_lang,
        )
    except Exception as ex:
        logger.exception("TimeX3 processing failed")
        return JSONResponse(status_code=500, content={"message": str(ex)})