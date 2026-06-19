from __future__ import annotations

import os
import logging
from functools import lru_cache
from threading import Lock
from time import time
from typing import Any, Dict, Final, Iterable, List, Optional, Tuple, Union

import torch
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

try:
    from pydantic_settings import BaseSettings
except ImportError:  # pydantic v1 fallback
    from pydantic import BaseSettings  # type: ignore

from ner_classification_backend import MODEL_REGISTRY, create_ner_classifier, resolve_model_name


model_lock = Lock()
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
os.environ["TORCH_USE_CUDA_DSA"]="1"

class UimaSentence(BaseModel):
    text: str
    begin: int
    end: int


class UimaSentenceSelection(BaseModel):
    selection: str
    sentences: List[UimaSentence]


class Settings(BaseSettings):
    annotator_name: str = "DUUI NER"
    annotator_version: str = "0.1.0"
    log_level: str = "INFO"

    # Exactly one model per container.
    # Use one registry alias or one exact HuggingFace model id.
    # Comma-separated lists and "all" are intentionally rejected.
    model_name: str = "wikineural-multilingual-ner"
    model_version: str = "latest"
    # Optional HuggingFace revision/commit hash. Empty/default values are ignored.
    model_cache_size: int = 1
    model_source: str = ""
    model_lang: str = "multi"

    # Labels are used by GLiNER/GLiNER2. HF token-classification ignores them.
    ner_labels: str = "person,organization,location,date,event,product,taxon,other"

    threshold: float = 0.5
    batch_size: int = 8

    # Runtime tokenizer/thread defaults. These can also be passed per /v1/process request.
    tokenizers_parallelism: str = "false"
    rayon_num_threads: int = 1
    omp_num_threads: int = 1
    mkl_num_threads: int = 1
    use_fast_tokenizer: bool = False

    typesystem_filename: str = "TypeSystemNER.xml"
    lua_communication_script_filename: str = "duui_ner.lua"

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
    # If omitted, the defaults from Settings are used.
    threshold: Optional[float] = None
    batch_size: Optional[int] = None
    labels: Optional[Union[str, List[str]]] = None

    # Runtime tokenizer/thread parameters passed through DUUI .withParameter(...).
    # They are applied at the beginning of every /v1/process call.
    tokenizers_parallelism: Optional[Union[bool, str]] = None
    rayon_num_threads: Optional[Union[int, str]] = None
    omp_num_threads: Optional[Union[int, str]] = None
    mkl_num_threads: Optional[Union[int, str]] = None
    use_fast_tokenizer: Optional[Union[bool, str]] = None


class DocumentModification(BaseModel):
    user: str
    timestamp: int
    comment: str


class AnnotationMeta(BaseModel):
    name: str
    version: str
    modelName: str
    modelVersion: str


ner_types: Final[Dict[str, str]] = {
    "Animal": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Animal",
    "Cardinal": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Cardinal",
    "ContactInfo": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.ContactInfo",
    "Date": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Date",
    "Disease": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Disease",
    "Event": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Event",
    "Fac": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Fac",
    "FacDesc": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.FacDesc",
    "Game": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Game",
    "Gpe": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Gpe",
    "GpeDesc": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.GpeDesc",
    "Language": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Language",
    "Law": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Law",
    "Location": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Location",
    "Money": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Money",
    "NamedEntity": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity",
    "Taxon": "org.texttechnologylab.annotation.type.Taxon",
    "Nationality": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Nationality",
    "Norp": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Norp",
    "Ordinal": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Ordinal",
    "OrgDesc": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.OrgDesc",
    "Organization": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Organization",
    "PerDesc": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.PerDesc",
    "Percent": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Percent",
    "Person": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Person",
    "Plant": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Plant",
    "Product": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Product",
    "ProductDesc": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.ProductDesc",
    "Quantity": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Quantity",
    "Substance": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Substance",
    "Time": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Time",
    "WorkOfArt": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.WorkOfArt",
}
ner_base_type: Final[str] = ner_types["NamedEntity"]

ner_tag_map: Final[Dict[str, str]] = {
    "PER": "Person",
    "PERSON": "Person",
    "person": "Person",
    "ORG": "Organization",
    "ORGANIZATION": "Organization",
    "organization": "Organization",
    "LOC": "Location",
    "LOCATION": "Location",
    "location": "Location",
    "GPE": "Gpe",
    "date": "Date",
    "DATE": "Date",
    "event": "Event",
    "EVENT": "Event",
    "product": "Product",
    "PRODUCT": "Product",
    "taxon": "Taxon",
    "TAXON": "Taxon",
    "Taxon": "Taxon",
    "taxa": "Taxon",
    "TAXA": "Taxon",
    "MISC": "NamedEntity",
    "misc": "NamedEntity",
    "OTHER": "NamedEntity",
    "other": "NamedEntity",
}


class DkproNer(BaseModel):
    begin: int
    end: int
    value: str
    identifier: Optional[str] = None
    ner_type: str = ner_base_type
    covered_text: Optional[str] = None
    score: Optional[float] = None
    model_name: Optional[str] = None


class DUUIResponse(BaseModel):
    meta: AnnotationMeta
    modification_meta: DocumentModification
    begin: List[int]
    end: List[int]
    results: List[str]
    factors: List[float]
    len_results: List[int]
    ner_type: List[str]
    covered_text: List[str]
    model: List[str]
    tags: List[DkproNer]
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
    description="DUUI NER annotator using the second code as backend",
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
            "backend_module": "ner_classification_backend.py",
            "ner_labels": get_ner_labels(),
        },
        docker_container_id=None,
        parameters={
            "model_name": "exactly one registry alias or one exact HuggingFace model id",
            "threshold": settings.threshold,
            "batch_size": settings.batch_size,
            "labels": settings.ner_labels,
        },
        capability=TextImagerCapability(supported_languages=["multi"], reproducible=True),
        implementation_specific=None,
    )


def parse_ner_labels(labels: Optional[Union[str, List[str]]] = None) -> List[str]:
    raw_labels: Union[str, List[str]] = settings.ner_labels if labels is None else labels
    if isinstance(raw_labels, list):
        return [str(label).strip() for label in raw_labels if str(label).strip()]
    return [label.strip() for label in str(raw_labels).split(",") if label.strip()]


def get_ner_labels() -> List[str]:
    return parse_ner_labels(None)


_LAST_RUNTIME_ENV: Dict[str, str] = {}


def parse_bool_like(value: Optional[Union[bool, str]], default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def parse_positive_int_like(value: Optional[Union[int, str]], default: int) -> int:
    if value is None:
        return max(1, int(default))
    try:
        return max(1, int(str(value).strip()))
    except Exception:
        return max(1, int(default))


def apply_runtime_env_from_request(request: DUUIRequest) -> bool:
    """Apply tokenizer/thread runtime settings for every process call.

    The values are intentionally written into os.environ on each request so DUUI
    can pass them via .withParameter(...). If values change after a model has
    been cached, the model cache is cleared so the backend can pick up the new
    USE_FAST_TOKENIZER value on the next load.
    """
    values = {
        "TOKENIZERS_PARALLELISM": "true" if parse_bool_like(
            request.tokenizers_parallelism,
            str(settings.tokenizers_parallelism).strip().lower() in {"1", "true", "yes", "y", "on"},
            ) else "false",
        "RAYON_NUM_THREADS": str(parse_positive_int_like(request.rayon_num_threads, settings.rayon_num_threads)),
        "OMP_NUM_THREADS": str(parse_positive_int_like(request.omp_num_threads, settings.omp_num_threads)),
        "MKL_NUM_THREADS": str(parse_positive_int_like(request.mkl_num_threads, settings.mkl_num_threads)),
        "USE_FAST_TOKENIZER": "true" if parse_bool_like(request.use_fast_tokenizer, settings.use_fast_tokenizer) else "false",
    }

    changed = False
    for key, value in values.items():
        if os.environ.get(key) != value:
            os.environ[key] = value
            changed = True

    global _LAST_RUNTIME_ENV
    cache_relevant_changed = _LAST_RUNTIME_ENV.get("USE_FAST_TOKENIZER") != values["USE_FAST_TOKENIZER"]
    _LAST_RUNTIME_ENV = values
    return cache_relevant_changed


def get_selected_model_name(model_name: str) -> str:
    """Return exactly one configured model for this container.

    DUUI should start one container per model. Therefore values such as
    "all" or "model_a,model_b" are rejected deliberately.
    """
    selected = (model_name or "").strip()
    if not selected:
        selected = "wikineural-multilingual-ner"

    if selected.lower() == "all" or "," in selected:
        supported = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            "This DUUI container supports exactly one MODEL_NAME. "
            "Start one container per model instead of using 'all' or comma-separated lists. "
            f"Supported aliases: {supported}"
        )

    # Validate alias or exact HuggingFace model id.
    alias, _ = resolve_model_name(selected)
    return alias


def get_ner_type(o_tag: str) -> str:
    if not o_tag:
        return ner_base_type

    label = o_tag.strip()
    if label in ner_tag_map:
        return ner_types.get(ner_tag_map[label], ner_base_type)

    upper_label = label.upper()
    if upper_label in ner_tag_map:
        return ner_types.get(ner_tag_map[upper_label], ner_base_type)

    tag = "".join(map(str.title, label.replace("-", "_").split("_")))
    return ner_types.get(tag, ner_base_type)


@lru_cache_with_size
def load_model(model_name: str):
    # This is the key integration point: DUUI loads the second-code backend here.
    return create_ner_classifier(model_name, device=device)


def fix_unicode_problems(text: str) -> str:
    return text.encode("utf-16", "surrogatepass").decode("utf-16", "surrogateescape")


def iter_batches(items: List[UimaSentence], batch_size: int) -> Iterable[List[UimaSentence]]:
    size = max(1, int(batch_size))
    for start in range(0, len(items), size):
        yield items[start:start + size]


def process_selection(
        model_name: str,
        selection: UimaSentenceSelection,
        labels: List[str],
        threshold: float,
        batch_size: int,
        model_version: str = "",
) -> Dict[str, Any]:
    begin: List[int] = []
    end: List[int] = []
    results_out: List[str] = []
    factors: List[float] = []
    len_results: List[int] = []
    ner_type_out: List[str] = []
    covered_text_out: List[str] = []
    model_out: List[str] = []
    tags: List[DkproNer] = []

    for s in selection.sentences:
        s.text = fix_unicode_problems(s.text)

    for batch in iter_batches(selection.sentences, batch_size):
        texts = [s.text for s in batch]

        with model_lock:
            classifier = load_model(model_name)
            predictions = classifier.predict(
                texts,
                labels=labels,
                threshold=threshold,
                batch_size=batch_size,
            )

            for sentence, sentence_entities in zip(batch, predictions):
                for ent in sentence_entities:
                    rel_start = int(ent["start"])
                    rel_end = int(ent["end"])
                    if rel_end <= rel_start:
                        continue

                    abs_begin = sentence.begin + rel_start
                    abs_end = sentence.begin + rel_end
                    value = str(ent["label"])
                    score = float(ent.get("score", 0.0))
                    covered = str(ent.get("text", sentence.text[rel_start:rel_end]))
                    ner_type = get_ner_type(value)
                    entity_model_name = str(ent.get("model_name", model_name))

                    tag = DkproNer(
                        begin=abs_begin,
                        end=abs_end,
                        value=value,
                        identifier=None,
                        ner_type=ner_type,
                        covered_text=covered,
                        score=score,
                        model_name=entity_model_name,
                    )

                    begin.append(abs_begin)
                    end.append(abs_end)
                    results_out.append(value)
                    factors.append(score)
                    len_results.append(1)
                    ner_type_out.append(ner_type)
                    covered_text_out.append(covered)
                    model_out.append(entity_model_name)
                    tags.append(tag)

    return {
        "begin": begin,
        "end": end,
        "results": results_out,
        "factors": factors,
        "len_results": len_results,
        "ner_type": ner_type_out,
        "covered_text": covered_text_out,
        "model": model_out,
        "tags": tags,
    }


def model_meta_values(model_name: str) -> Tuple[str, str]:
    _, cfg = resolve_model_name(model_name)
    model_source = settings.model_source or cfg.get("model_source", "")
    model_lang = settings.model_lang or cfg.get("model_lang", "multi")
    return model_source, model_lang


@app.post("/v1/process", response_model=DUUIResponse)
def post_process(request: DUUIRequest):
    runtime_env_changed = apply_runtime_env_from_request(request)
    if runtime_env_changed:
        with model_lock:
            load_model.cache_clear()

    if not request.selections:
        return JSONResponse(status_code=400, content={"message": "The request must contain sentence selections."})

    try:
        model_name = get_selected_model_name(settings.model_name)
    except Exception as ex:
        return JSONResponse(status_code=400, content={"message": str(ex)})

    effective_threshold = request.threshold if request.threshold is not None else settings.threshold
    effective_batch_size = request.batch_size if request.batch_size is not None else settings.batch_size
    effective_labels = parse_ner_labels(request.labels)

    if effective_threshold < 0.0 or effective_threshold > 1.0:
        return JSONResponse(status_code=400, content={"message": "threshold must be between 0.0 and 1.0"})
    if effective_batch_size < 1:
        return JSONResponse(status_code=400, content={"message": "batch_size must be >= 1"})
    if not effective_labels:
        return JSONResponse(status_code=400, content={"message": "labels must not be empty"})

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
    ner_type_out: List[str] = []
    covered_text_out: List[str] = []
    model_out: List[str] = []
    tags: List[DkproNer] = []

    try:
        for selection in request.selections:
            processed = process_selection(
                model_name=model_name,
                selection=selection,
                labels=effective_labels,
                threshold=effective_threshold,
                batch_size=effective_batch_size,
                model_version=settings.model_version,
            )
            begin += processed["begin"]
            end += processed["end"]
            len_results += processed["len_results"]
            results += processed["results"]
            factors += processed["factors"]
            ner_type_out += processed["ner_type"]
            covered_text_out += processed["covered_text"]
            model_out += processed["model"]
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
            ner_type=ner_type_out,
            covered_text=covered_text_out,
            model=model_out,
            tags=tags,
            model_name=meta_model_name,
            model_version=settings.model_version,
            model_source=model_source,
            model_lang=model_lang,
        )
    except Exception as ex:
        logger.exception("NER processing failed")
        return JSONResponse(status_code=500, content={"message": str(ex)})