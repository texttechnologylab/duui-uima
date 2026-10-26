import logging
import re
from importlib.metadata import version as package_version
from platform import python_version
from sys import version as sys_version
from threading import Lock
from time import time
from typing import List, Optional

import torch
from cassis import load_typesystem
from cassis.cas import Utf16CodepointOffsetConverter
from chonky import ParagraphSplitter
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings

CHONKY_VERSION = package_version("chonky")


class Settings(BaseSettings):
    annotator_name: str
    annotator_version: str
    log_level: str
    # loaded by its Hugging Face id (chonky sets the tokenizer length by the id) at this revision
    model_name: str
    model_revision: str

    class Config:
        env_prefix = 'duui_chonky_'


settings = Settings()

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)
logger.info("TTLab DUUI chonky")
logger.info("Name: %s", settings.annotator_name)
logger.info("Version: %s", settings.annotator_version)

UIMA_TYPE_PARAGRAPH = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Existing line breaks that are added as paragraph boundaries to those of chonky
KEEP_LINE_BREAKS = {
    "none": None,
    # every line break
    "line": re.compile(r"\n"),
    # blank lines: two line breaks with only other whitespace between them
    "blank_line": re.compile(r"\n[^\S\n]*\n"),
}

# The model was downloaded at the pinned revision when the image was built, the container runs offline; chonky loads
# the tokenizer from the same snapshot ("main" at build time = the pinned revision, checked in the Dockerfile)
logger.info("Loading chonky \"%s\" (%s) on %s", settings.model_name, settings.model_revision, DEVICE)
splitter = ParagraphSplitter(model_id=settings.model_name, device=DEVICE, revision=settings.model_revision)
splitter_lock = Lock()


class TextImagerRequest(BaseModel):
    text: str
    keep_line_breaks: Optional[str] = None


class Span(BaseModel):
    begin: int
    end: int


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
    paragraphs: List[Span]
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
    description="chonky paragraph segmentation for TTLab DUUI",
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
            "chonky_version": CHONKY_VERSION,
            "torch_version": torch.__version__,
            "model_name": settings.model_name,
            "model_revision": settings.model_revision,
            "device": DEVICE,
        },
        docker_container_id="[TODO]",
        parameters={
            "keep_line_breaks": sorted(KEEP_LINE_BREAKS),
        },
        # multilingual model, the document language is not used
        capability=TextImagerCapability(supported_languages=[], reproducible=True),
        implementation_specific=None,
    )


@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    return Response(content=typesystem_xml_content, media_type="application/xml")


@app.get("/v1/details/input_output")
def get_input_output() -> TextImagerInputOutput:
    return TextImagerInputOutput(inputs=[], outputs=[UIMA_TYPE_PARAGRAPH])


@app.post("/v1/process")
def post_process(request: TextImagerRequest) -> TextImagerResponse:
    modification_timestamp_seconds = int(time())

    keep_line_breaks = (request.keep_line_breaks or "none").strip().lower()
    if keep_line_breaks not in KEEP_LINE_BREAKS:
        raise HTTPException(status_code=400, detail=f"Parameter \"keep_line_breaks\" must be one of {', '.join(sorted(KEEP_LINE_BREAKS))}, not \"{request.keep_line_breaks}\"")

    # CAS offsets are UTF-16 code units, Python strings use code points
    text = utf16_to_utf8(request.text)
    utf16_converter = Utf16CodepointOffsetConverter()
    utf16_converter.create_offset_mapping(text)

    # chonky yields consecutive slices of the text: their ends are the boundaries
    boundaries = {0, len(text)}
    if text.strip():
        with splitter_lock:
            position = 0
            for chunk in splitter(text):
                position += len(chunk)
                boundaries.add(position)
        if position != len(text):
            raise Exception(f"chonky chunks do not cover the text: {position} of {len(text)} characters")

    pattern = KEEP_LINE_BREAKS[keep_line_breaks]
    if pattern is not None:
        for match in pattern.finditer(text):
            boundaries.add(match.start())
            boundaries.add(match.end())

    # paragraphs without surrounding whitespace, whitespace-only segments are dropped
    paragraphs = []
    cuts = sorted(boundaries)
    for begin, end in zip(cuts, cuts[1:]):
        segment = text[begin:end]
        stripped = segment.strip()
        if not stripped:
            continue
        stripped_begin = begin + segment.index(stripped)
        paragraphs.append(Span(
            begin=utf16_converter.python_to_external(stripped_begin),
            end=utf16_converter.python_to_external(stripped_begin + len(stripped)),
        ))

    logger.info("%d paragraphs, %d seconds", len(paragraphs), int(time()) - modification_timestamp_seconds)

    return TextImagerResponse(
        paragraphs=paragraphs,
        meta=AnnotationMeta(
            name=settings.annotator_name,
            version=settings.annotator_version,
            modelName=settings.model_name,
            modelVersion=settings.model_revision,
        ),
        modification_meta=DocumentModification(
            user=settings.annotator_name,
            timestamp=modification_timestamp_seconds,
            comment=f"{settings.annotator_name} ({settings.annotator_version}), chonky ({CHONKY_VERSION}), "
                    f"{settings.model_name} ({settings.model_revision}), keep_line_breaks={keep_line_breaks}, {DEVICE}",
        ),
    )
