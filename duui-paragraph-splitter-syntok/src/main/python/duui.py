import logging
from platform import python_version
from sys import version as sys_version
from time import time
from typing import List, Optional

import syntok.segmenter as syntok_segmenter
from cassis import load_typesystem
from cassis.cas import Utf16CodepointOffsetConverter
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings


# NOTE adjust if changed in requirements.txt
# TODO extract to python package
SYNTOC_VERSION = "1.4.4"


class Settings(BaseSettings):
    annotator_name: str
    annotator_version: str
    log_level: str

    class Config:
        env_prefix = 'duui_syntok_'


settings = Settings()

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)
logger.info("TTLab TextImager DUUI syntok")
logger.info("Name: %s", settings.annotator_name)
logger.info("Version: %s", settings.annotator_version)

UIMA_TYPE_SENTENCE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
UIMA_TYPE_PARAGRAPH = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph"

TEXTIMAGER_ANNOTATOR_OUTPUT_TYPES = {
    UIMA_TYPE_SENTENCE,
    UIMA_TYPE_PARAGRAPH
}

TEXTIMAGER_ANNOTATOR_INPUT_TYPES = {
    ""  # Text
}

SUPPORTED_LANGS = {
    "en",
    "es",
    "de",
}


class TextImagerRequest(BaseModel):
    text: str
    len: int
    lang: str
    write_sentences: Optional[bool] = False
    write_paragraphs: Optional[bool] = True


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


class Paragraph(BaseModel):
    begin: int
    end: int


class TextImagerResponse(BaseModel):
    sentences: List[Sentence]
    paragraphs: List[Paragraph]
    meta: Optional[AnnotationMeta]
    modification_meta: Optional[DocumentModification]
    write_sentences: bool
    write_paragraphs: bool


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
# typesystem_filename = 'src/main/resources/TypeSystem.xml'
logger.debug("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    typesystem_xml_content = typesystem.to_xml().encode("utf-8")
    logger.debug("Base typesystem:")
    logger.debug(typesystem_xml_content)

lua_communication_script_filename = "communication.lua"
# lua_communication_script_filename = "src/main/lua/communication.lua"
logger.debug("Loading Lua communication script from \"%s\"", lua_communication_script_filename)
with open(lua_communication_script_filename, 'rb') as f:
    lua_communication_script = f.read().decode("utf-8")
    logger.debug("Lua communication script:")
    logger.debug(lua_communication_script_filename)

app = FastAPI(
    title=settings.annotator_name,
    description="syntok implementation for TTLab TextImager DUUI",
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
            "syntoc_version": SYNTOC_VERSION,
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
        inputs=sorted(TEXTIMAGER_ANNOTATOR_INPUT_TYPES),
        outputs=sorted(TEXTIMAGER_ANNOTATOR_OUTPUT_TYPES)
    )


@app.post("/v1/process")
def post_process(request: TextImagerRequest) -> TextImagerResponse:
    modification_timestamp_seconds = int(time())

    paragraphs = []
    sentences = []
    meta = None
    modification_meta = None

    try:
        utf16_converter = Utf16CodepointOffsetConverter()
        utf16_converter.create_offset_mapping(request.text)

        text_len = len(request.text)

        doc = syntok_segmenter.analyze(request.text)
        for paragraph in doc:
            para = list(paragraph)

            try:
                first = para[0][0]
                para_begin = first.offset + len(first.spacing)

                last = para[-1][-1]
                para_end = last.offset + len(last.spacing) + len(last.value)
                para_end = min(para_end, text_len)

                paragraphs.append(Paragraph(
                    begin=utf16_converter.python_to_external(para_begin),
                    end=utf16_converter.python_to_external(para_end),
                ))
            except Exception as ex:
                logger.exception("Error processing paragraph: %s", ex)

            for i, s in enumerate(para):
                try:
                    first = s[0]
                    last = s[-1]

                    sent_begin = first.offset + len(first.spacing)

                    sent_end = last.offset + len(last.spacing) + len(last.value)
                    sent_end = min(sent_end, text_len)

                    sentences.append(Sentence(
                        begin=utf16_converter.python_to_external(sent_begin),
                        end=utf16_converter.python_to_external(sent_end),
                    ))
                except Exception as ex:
                    logger.exception("Error processing sentence: %s", ex)

        meta = AnnotationMeta(
            name=settings.annotator_name,
            version=settings.annotator_version,
            modelName="syntok",
            modelVersion=SYNTOC_VERSION
        )

        modification_meta = DocumentModification(
            user=settings.annotator_name,
            timestamp=modification_timestamp_seconds,
            comment=f"{settings.annotator_name} ({settings.annotator_version}), syntok ({SYNTOC_VERSION})"
        )

    except Exception as ex:
        logger.exception(ex)

    # add at least one paragraph
    if not paragraphs and request.text:
        paragraphs.append(Paragraph(
            begin=0, end=request.len
        ))

    if not sentences and request.text:
        sentences.append(Sentence(
            begin=0, end=request.len
        ))

    logger.debug(paragraphs)
    logger.debug(sentences)
    logger.debug(meta)
    logger.debug(modification_meta)

    duration = int(time()) - modification_timestamp_seconds
    logger.info("Processed in %d seconds", duration)

    return TextImagerResponse(
        paragraphs=paragraphs,
        sentences=sentences,
        meta=meta,
        modification_meta=modification_meta,
        write_sentences=request.write_sentences,
        write_paragraphs=request.write_paragraphs
    )
