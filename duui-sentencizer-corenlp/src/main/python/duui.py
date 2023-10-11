import logging
from platform import python_version
from sys import version as sys_version
from time import time
from typing import List, Optional

import corenlp
from cassis import load_typesystem
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings


# NOTE adjust if changed in requirements.txt
# TODO extract to python package
CORENLP_VERSION = "3.9.2"


class Settings(BaseSettings):
    annotator_name: str
    annotator_version: str
    log_level: str

    class Config:
        env_prefix = 'duui_sentencizer_corenlp_'


settings = Settings()

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)
logger.info("TTLab TextImager DUUI corenlp")
logger.info("Name: %s", settings.annotator_name)
logger.info("Version: %s", settings.annotator_version)

UIMA_TYPE_SENTENCE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"

TEXTIMAGER_ANNOTATOR_OUTPUT_TYPES = {
    UIMA_TYPE_SENTENCE
}

TEXTIMAGER_ANNOTATOR_INPUT_TYPES = {
    ""  # Text
}

SUPPORTED_LANGS = {
    "ar",
    "zh",
    "en",
    "fr",
    "de",
    "hu",
    "it",
    "es",
}


class TextImagerRequest(BaseModel):
    text: str
    len: int
    lang: str


class AnnotationMeta(BaseModel):
    name: str
    version: str
    modelName: str
    modelVersion: str


class DocumentModification(BaseModel):
    user: str
    timestamp: int
    comment: str


class Sentence(BaseModel):
    begin: int
    end: int


class TextImagerResponse(BaseModel):
    sentences: List[Sentence]
    meta: Optional[AnnotationMeta]
    modification_meta: Optional[DocumentModification]


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


typesystem_filename = 'TypeSystem.xml'
logger.debug("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    typesystem_xml_content = typesystem.to_xml().encode("utf-8")
    logger.debug("Base typesystem:")
    logger.debug(typesystem_xml_content)

lua_communication_script_filename = "communication.lua"
logger.debug("Loading Lua communication script from \"%s\"", lua_communication_script_filename)
with open(lua_communication_script_filename, 'rb') as f:
    lua_communication_script = f.read().decode("utf-8")
    logger.debug("Lua communication script:")
    logger.debug(lua_communication_script_filename)

pipe = corenlp.CoreNLPClient(annotators=["ssplit"])

app = FastAPI(
    title=settings.annotator_name,
    description="CoreNLP implementation for TTLab TextImager DUUI",
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


@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return lua_communication_script


@app.get("/v1/documentation")
def get_documentation() -> TextImagerDocumentation:
    capabilities = TextImagerCapability(
        supported_languages=sorted(list(SUPPORTED_LANGS)),
        reproducible=True
    )

    documentation = TextImagerDocumentation(
        annotator_name=settings.annotator_name,
        version=settings.annotator_version,
        implementation_lang="Python",
        meta={
            "python_version": python_version(),
            "python_version_full": sys_version,
            "segtoc_version": CORENLP_VERSION,
        },
        docker_container_id="[TODO]",
        parameters={},
        capability=capabilities,
        implementation_specific=None,
    )

    return documentation


@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    return Response(
        content=typesystem_xml_content,
        media_type="application/xml"
    )


@app.get("/v1/details/input_output")
def get_input_output() -> TextImagerInputOutput:
    return TextImagerInputOutput(
        inputs=TEXTIMAGER_ANNOTATOR_INPUT_TYPES,
        outputs=TEXTIMAGER_ANNOTATOR_OUTPUT_TYPES
    )


@app.post("/v1/process")
def post_process(request: TextImagerRequest) -> TextImagerResponse:
    modification_timestamp_seconds = int(time())

    sentences = []
    meta = None
    modification_meta = None

    try:
        doc = pipe.annotate(request.text)
        for sent in doc.sentence:
            sentences.append(Sentence(
                begin=sent.characterOffsetBegin,
                end=sent.characterOffsetEnd,
            ))

        meta = AnnotationMeta(
            name=settings.annotator_name,
            version=settings.annotator_version,
            modelName="corenlp",
            modelVersion=CORENLP_VERSION
        )

        modification_meta = DocumentModification(
            user=settings.annotator_name,
            timestamp=modification_timestamp_seconds,
            comment=f"{settings.annotator_name} ({settings.annotator_version}), corenlp ({CORENLP_VERSION})"
        )

    except Exception as ex:
        logger.exception(ex)

    logger.debug(sentences)
    logger.debug(meta)
    logger.debug(modification_meta)

    duration = int(time()) - modification_timestamp_seconds
    logger.info("Processed in %d seconds", duration)

    return TextImagerResponse(
        sentences=sentences,
        meta=meta,
        modification_meta=modification_meta,
    )
