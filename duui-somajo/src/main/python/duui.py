import logging
from functools import lru_cache
from importlib.metadata import version as package_version
from platform import python_version
from sys import version as sys_version
from threading import Lock
from time import time
from typing import List, Optional

from cassis import load_typesystem
from cassis.cas import Utf16CodepointOffsetConverter
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from somajo import SoMaJo

SOMAJO_VERSION = package_version("SoMaJo")

# SoMaJo models by document language, other languages are not supported
SOMAJO_LANGUAGES = {
    "de": "de_CMC",
    "en": "en_PTB",
}


class Settings(BaseSettings):
    annotator_name: str
    annotator_version: str
    log_level: str

    class Config:
        env_prefix = 'duui_somajo_'


settings = Settings()

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)
logger.info("TTLab DUUI SoMaJo")
logger.info("Name: %s", settings.annotator_name)
logger.info("Version: %s", settings.annotator_version)

UIMA_TYPE_SENTENCE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
UIMA_TYPE_TOKEN = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"


class Span(BaseModel):
    begin: int
    end: int


class TextImagerRequest(BaseModel):
    text: str
    len: int
    lang: str
    # Tokenized one by one, tokens never cross them; the whole text if there are none
    sentences: Optional[List[Span]] = None


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
    tokens: List[Span]
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
    description="SoMaJo tokenizer for TTLab DUUI",
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

# a tokenizer instance is not shared between concurrent requests
tokenizer_lock = Lock()


@lru_cache(maxsize=None)
def get_tokenizer(somajo_language: str) -> SoMaJo:
    logger.info("Loading SoMaJo \"%s\"", somajo_language)
    # sentences come from the CAS, no camel case split (SoMaJo default)
    return SoMaJo(somajo_language, split_camel_case=False, split_sentences=False, character_offsets=True)


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
            "somajo_version": SOMAJO_VERSION,
            "somajo_languages": SOMAJO_LANGUAGES,
        },
        docker_container_id="[TODO]",
        parameters={},
        capability=TextImagerCapability(supported_languages=sorted(SOMAJO_LANGUAGES), reproducible=True),
        implementation_specific=None,
    )


@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    return Response(content=typesystem_xml_content, media_type="application/xml")


@app.get("/v1/details/input_output")
def get_input_output() -> TextImagerInputOutput:
    return TextImagerInputOutput(inputs=[UIMA_TYPE_SENTENCE], outputs=[UIMA_TYPE_TOKEN])


@app.post("/v1/process")
def post_process(request: TextImagerRequest) -> TextImagerResponse:
    modification_timestamp_seconds = int(time())

    if request.lang not in SOMAJO_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Language \"{request.lang}\" is not supported, only: {', '.join(sorted(SOMAJO_LANGUAGES))}")
    somajo_language = SOMAJO_LANGUAGES[request.lang]

    # CAS offsets are UTF-16 code units, Python strings use code points
    text = utf16_to_utf8(request.text)
    utf16_converter = Utf16CodepointOffsetConverter()
    utf16_converter.create_offset_mapping(text)

    units = request.sentences if request.sentences else [Span(begin=0, end=request.len)]

    tokens = []
    with tokenizer_lock:
        tokenizer = get_tokenizer(somajo_language)
        for unit in units:
            unit_begin = utf16_converter.external_to_python(unit.begin)
            unit_end = utf16_converter.external_to_python(unit.end)
            # one input paragraph gives one chunk (split_sentences=False), or none if it has no tokens
            for chunk in tokenizer.tokenize_text([text[unit_begin:unit_end]]):
                for token in chunk:
                    token_begin, token_end = token.character_offset
                    tokens.append(Span(
                        begin=utf16_converter.python_to_external(unit_begin + token_begin),
                        end=utf16_converter.python_to_external(unit_begin + token_end),
                    ))

    logger.info("%d tokens in %d units, %d seconds", len(tokens), len(units), int(time()) - modification_timestamp_seconds)

    return TextImagerResponse(
        tokens=tokens,
        meta=AnnotationMeta(
            name=settings.annotator_name,
            version=settings.annotator_version,
            modelName=f"SoMaJo {somajo_language}",
            modelVersion=SOMAJO_VERSION,
        ),
        modification_meta=DocumentModification(
            user=settings.annotator_name,
            timestamp=modification_timestamp_seconds,
            comment=f"{settings.annotator_name} ({settings.annotator_version}), SoMaJo ({SOMAJO_VERSION}) {somajo_language}",
        ),
    )
