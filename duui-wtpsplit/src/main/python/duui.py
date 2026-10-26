import logging
from importlib.metadata import version as package_version
from platform import python_version
from sys import version as sys_version
from threading import Lock
from time import time
from typing import List, Optional

import torch
from cassis import load_typesystem
from cassis.cas import Utf16CodepointOffsetConverter
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import PlainTextResponse
from huggingface_hub import snapshot_download
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from wtpsplit import SaT

WTPSPLIT_VERSION = package_version("wtpsplit")


class Settings(BaseSettings):
    annotator_name: str
    annotator_version: str
    log_level: str
    # SaT model name (loaded by name, so that SaT chooses its default threshold for it) and its Hugging Face revision
    model_name: str
    model_revision: str
    # tokenizer of the model and its Hugging Face revision
    tokenizer_name: str
    tokenizer_revision: str

    class Config:
        env_prefix = 'duui_wtpsplit_'


settings = Settings()

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)
logger.info("TTLab DUUI wtpsplit")
logger.info("Name: %s", settings.annotator_name)
logger.info("Version: %s", settings.annotator_version)

UIMA_TYPE_PARAGRAPH = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph"
UIMA_TYPE_SENTENCE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# the tokenizer needs only its config and vocabulary, not the model weights (same patterns in the Dockerfile)
TOKENIZER_FILES = ["*.json", "*.model"]

# Models are downloaded at the pinned revisions when the image is built, the container runs offline
logger.info("Loading SaT \"%s\" (%s), tokenizer \"%s\" (%s) on %s", settings.model_name, settings.model_revision,
            settings.tokenizer_name, settings.tokenizer_revision, DEVICE)
sat = SaT(
    settings.model_name,
    tokenizer_name_or_path=snapshot_download(settings.tokenizer_name, revision=settings.tokenizer_revision,
                                             allow_patterns=TOKENIZER_FILES, local_files_only=True),
    from_pretrained_kwargs={"revision": settings.model_revision},
)
# full precision (no half()), for the same results on CPU and GPU as far as possible
sat.to(DEVICE)
sat.eval()
sat_lock = Lock()


class Span(BaseModel):
    begin: int
    end: int


class TextImagerRequest(BaseModel):
    text: str
    len: int
    # Split one by one, sentences never cross them; the whole text if there are none
    paragraphs: Optional[List[Span]] = None
    # "true" (SaT default): every line break ends a sentence, "false": line breaks are read as spaces
    split_on_input_newlines: Optional[str] = None


class AnnotationMeta(BaseModel):
    name: str
    version: str
    modelName: str
    modelVersion: str


class DocumentModification(BaseModel):
    user: str
    timestamp: int
    comment: str


class TextImagerResponse(BaseModel):
    sentences: List[Span]
    meta: AnnotationMeta
    modification_meta: DocumentModification


class TextImagerCapability(BaseModel):
    supported_languages: List[str]
    reproducible: bool


class TextImagerDocumentation(BaseModel):
    annotator_name: str
    version: str
    implementation_lang: Optional[str]
    meta: Optional[dict]
    docker_container_id: Optional[str]
    parameters: Optional[dict]
    capability: TextImagerCapability
    implementation_specific: Optional[str]


class TextImagerInputOutput(BaseModel):
    inputs: List[str]
    outputs: List[str]


with open("TypeSystem.xml", "rb") as f:
    typesystem_xml_content = load_typesystem(f).to_xml().encode("utf-8")

with open("communication.lua", "rb") as f:
    lua_communication_script = f.read().decode("utf-8")

app = FastAPI(
    title=settings.annotator_name,
    description="wtpsplit (Segment any Text) sentence segmentation for TTLab DUUI",
    version=settings.annotator_version,
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "TTLab Team - Daniel Baumartz",
        "url": "https://texttechnologylab.org",
        "email": "baumartz@em.uni-frankfurt.de",
    },
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)


def utf16_to_utf8(text):
    # same as in duui-spacy: DUUI <= 1.5.7 sends characters outside the BMP as two lone surrogates, joined here
    clean_text = text.encode('utf-16', 'surrogatepass').decode('utf-16', 'surrogateescape')
    return clean_text


@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return lua_communication_script


@app.get("/v1/documentation")
def get_documentation() -> TextImagerDocumentation:
    return TextImagerDocumentation(
        annotator_name=settings.annotator_name,
        version=settings.annotator_version,
        implementation_lang="Python",
        meta={
            "python_version": python_version(),
            "python_version_full": sys_version,
            "wtpsplit_version": WTPSPLIT_VERSION,
            "torch_version": torch.__version__,
            "model_name": settings.model_name,
            "model_revision": settings.model_revision,
            "tokenizer_name": settings.tokenizer_name,
            "tokenizer_revision": settings.tokenizer_revision,
            "device": DEVICE,
        },
        docker_container_id="[TODO]",
        parameters={
            "split_on_input_newlines": ["true", "false"],
        },
        # SaT is multilingual and does not use the document language
        capability=TextImagerCapability(supported_languages=[], reproducible=True),
        implementation_specific=None,
    )


@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    return Response(content=typesystem_xml_content, media_type="application/xml")


@app.get("/v1/details/input_output")
def get_input_output() -> TextImagerInputOutput:
    return TextImagerInputOutput(inputs=[UIMA_TYPE_PARAGRAPH], outputs=[UIMA_TYPE_SENTENCE])


@app.post("/v1/process")
def post_process(request: TextImagerRequest) -> TextImagerResponse:
    modification_timestamp_seconds = int(time())

    split_on_input_newlines = (request.split_on_input_newlines or "true").strip().lower()
    if split_on_input_newlines not in ("true", "false"):
        raise HTTPException(status_code=400, detail=f"Parameter \"split_on_input_newlines\" must be \"true\" or \"false\", not \"{request.split_on_input_newlines}\"")
    split_on_input_newlines = split_on_input_newlines == "true"

    # CAS offsets are UTF-16 code units, Python strings use code points
    text = utf16_to_utf8(request.text)
    utf16_converter = Utf16CodepointOffsetConverter()
    utf16_converter.create_offset_mapping(text)

    units = request.paragraphs if request.paragraphs else [Span(begin=0, end=request.len)]
    unit_bounds = [(utf16_converter.external_to_python(u.begin), utf16_converter.external_to_python(u.end)) for u in units]
    unit_texts = [text[b:e] for b, e in unit_bounds]

    with sat_lock:
        unit_segments = list(sat.split(unit_texts, split_on_input_newlines=split_on_input_newlines))

    sentences = []
    for (unit_begin, _), unit_text, segments in zip(unit_bounds, unit_texts, unit_segments):
        # SaT returns the sentences as strings in text order, without the line breaks it split on
        cursor = 0
        for segment in segments:
            pos = unit_text.find(segment, cursor)
            if pos < 0:
                raise Exception(f"SaT segment not found in its paragraph: {segment!r}")
            cursor = pos + len(segment)
            # sentence without surrounding whitespace, whitespace-only segments are dropped
            stripped = segment.strip()
            if not stripped:
                continue
            begin = unit_begin + pos + segment.index(stripped)
            sentences.append(Span(
                begin=utf16_converter.python_to_external(begin),
                end=utf16_converter.python_to_external(begin + len(stripped)),
            ))

    logger.info("%d sentences in %d units, %d seconds", len(sentences), len(units), int(time()) - modification_timestamp_seconds)

    return TextImagerResponse(
        sentences=sentences,
        meta=AnnotationMeta(
            name=settings.annotator_name,
            version=settings.annotator_version,
            modelName=settings.model_name,
            modelVersion=settings.model_revision,
        ),
        modification_meta=DocumentModification(
            user=settings.annotator_name,
            timestamp=modification_timestamp_seconds,
            comment=f"{settings.annotator_name} ({settings.annotator_version}), wtpsplit ({WTPSPLIT_VERSION}), "
                    f"{settings.model_name} ({settings.model_revision}), {settings.tokenizer_name} "
                    f"({settings.tokenizer_revision}), split_on_input_newlines={split_on_input_newlines}, {DEVICE}",
        ),
    )
