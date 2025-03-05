import json
import logging
from platform import python_version
from sys import version as sys_version
from time import time
from typing import List, Optional

import readability
from cassis import load_typesystem
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    annotator_name: str
    annotator_version: str
    log_level: str

    class Config:
        env_prefix = 'duui_readability_'


settings = Settings()

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)
logger.info("TTLab TextImager DUUI Readability")
logger.info("Name: %s", settings.annotator_name)
logger.info("Version: %s", settings.annotator_version)

TEXTIMAGER_ANNOTATOR_OUTPUT_TYPES = [
    "org.hucompute.textimager.uima.type.category.CategoryCoveredTagged"
]

TEXTIMAGER_ANNOTATOR_INPUT_TYPES = [
    # full text
]

SUPPORTED_LANGS = [
    # TODO
]


class TextImagerRequest(BaseModel):
    lang: str
    text: str
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


class TextImagerCategory(BaseModel):
    value: str
    score: Optional[float]
    tags: str
    begin: int
    end: int


class TextImagerResponse(BaseModel):
    results: List[TextImagerCategory]
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


typesystem_filename = 'src/main/resources/TypeSystem.xml'
logger.debug("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    typesystem_xml_content = typesystem.to_xml().encode("utf-8")
    logger.debug("Base typesystem:")
    logger.debug(typesystem_xml_content)

lua_communication_script_filename = "src/main/lua/communication.lua"
logger.debug("Loading Lua communication script from \"%s\"", lua_communication_script_filename)
with open(lua_communication_script_filename, 'rb') as f:
    lua_communication_script = f.read().decode("utf-8")
    logger.debug("Lua communication script:")
    logger.debug(lua_communication_script_filename)

app = FastAPI(
    title=settings.annotator_name,
    description="TTLab TextImager Readability",
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
        supported_languages=SUPPORTED_LANGS,
        reproducible=True
    )

    documentation = TextImagerDocumentation(
        annotator_name=settings.annotator_name,
        version=settings.annotator_version,
        implementation_lang="Python",
        meta={
            "python_version": python_version(),
            "python_version_full": sys_version,
            f"{readability.__title__}": readability.__version__
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

    results = []
    r = readability.Readability(request.text)
    if r._statistics.num_words < 100:
        logger.error("Text has less than 100 words, readability analysis might not be generated")

    try:
        rr = r.flesch_kincaid()
        results.append(TextImagerCategory(
            value="flesch_kincaid",
            score=rr.score,
            tags=json.dumps({
                "score": rr.score,
                "grade_level": rr.grade_level
            }),
            begin=request.begin,
            end=request.end
        ))
    except Exception as ex:
        logger.exception(ex)
        results.append(TextImagerCategory(
            value="flesch_kincaid",
            score=None,
            tags=json.dumps({
                "error": str(ex)
            }),
            begin=request.begin,
            end=request.end
        ))

    try:
        rr = r.flesch()
        results.append(TextImagerCategory(
            value="flesch",
            score=rr.score,
            tags=json.dumps({
                "score": rr.score,
                "ease": rr.ease,
                "grade_levels": rr.grade_levels
            }),
            begin=request.begin,
            end=request.end
        ))
    except Exception as ex:
        logger.exception(ex)
        results.append(TextImagerCategory(
            value="flesch",
            score=None,
            tags=json.dumps({
                "error": str(ex)
            }),
            begin=request.begin,
            end=request.end
        ))

    try:
        rr = r.gunning_fog()
        results.append(TextImagerCategory(
            value="gunning_fog",
            score=rr.score,
            tags=json.dumps({
                "score": rr.score,
                "grade_level": rr.grade_level
            }),
            begin=request.begin,
            end=request.end
        ))
    except Exception as ex:
        logger.exception(ex)
        results.append(TextImagerCategory(
            value="gunning_fog",
            score=None,
            tags=json.dumps({
                "error": str(ex)
            }),
            begin=request.begin,
            end=request.end
        ))

    try:
        rr = r.coleman_liau()
        results.append(TextImagerCategory(
            value="coleman_liau",
            score=rr.score,
            tags=json.dumps({
                "score": rr.score,
                "grade_level": rr.grade_level
            }),
            begin=request.begin,
            end=request.end
        ))
    except Exception as ex:
        logger.exception(ex)
        results.append(TextImagerCategory(
            value="coleman_liau",
            score=None,
            tags=json.dumps({
                "error": str(ex)
            }),
            begin=request.begin,
            end=request.end
        ))

    try:
        rr = r.dale_chall()
        results.append(TextImagerCategory(
            value="dale_chall",
            score=rr.score,
            tags=json.dumps({
                "score": rr.score,
                "grade_levels": rr.grade_levels
            }),
            begin=request.begin,
            end=request.end
        ))
    except Exception as ex:
        logger.exception(ex)
        results.append(TextImagerCategory(
            value="dale_chall",
            score=None,
            tags=json.dumps({
                "error": str(ex)
            }),
            begin=request.begin,
            end=request.end
        ))

    try:
        rr = r.ari()
        results.append(TextImagerCategory(
            value="ari",
            score=rr.score,
            tags=json.dumps({
                "score": rr.score,
                "grade_levels": rr.grade_levels,
                "ages": rr.ages
            }),
            begin=request.begin,
            end=request.end
        ))
    except Exception as ex:
        logger.exception(ex)
        results.append(TextImagerCategory(
            value="ari",
            score=None,
            tags=json.dumps({
                "error": str(ex)
            }),
            begin=request.begin,
            end=request.end
        ))

    try:
        rr = r.linsear_write()
        results.append(TextImagerCategory(
            value="linsear_write",
            score=rr.score,
            tags=json.dumps({
                "score": rr.score,
                "grade_level": rr.grade_level,
            }),
            begin=request.begin,
            end=request.end
        ))
    except Exception as ex:
        logger.exception(ex)
        results.append(TextImagerCategory(
            value="linsear_write",
            score=None,
            tags=json.dumps({
                "error": str(ex)
            }),
            begin=request.begin,
            end=request.end
        ))

    try:
        rr = r.smog()
        results.append(TextImagerCategory(
            value="smog",
            score=rr.score,
            tags=json.dumps({
                "score": rr.score,
                "grade_level": rr.grade_level,
            }),
            begin=request.begin,
            end=request.end
        ))
    except Exception as ex:
        logger.exception(ex)
        results.append(TextImagerCategory(
            value="smog",
            score=None,
            tags=json.dumps({
                "error": str(ex)
            }),
            begin=request.begin,
            end=request.end
        ))

    try:
        rr = r.spache()
        results.append(TextImagerCategory(
            value="spache",
            score=rr.score,
            tags=json.dumps({
                "score": rr.score,
                "grade_level": rr.grade_level,
            }),
            begin=request.begin,
            end=request.end
        ))
    except Exception as ex:
        logger.exception(ex)
        results.append(TextImagerCategory(
            value="spache",
            score=None,
            tags=json.dumps({
                "error": str(ex)
            }),
            begin=request.begin,
            end=request.end
        ))

    meta = AnnotationMeta(
        name=settings.annotator_name,
        version=settings.annotator_version,
        modelName=readability.__title__,
        modelVersion=readability.__version__
    )
    logger.debug(meta)

    modification_meta = DocumentModification(
        user=settings.annotator_name,
        timestamp=modification_timestamp_seconds,
        comment=f"{settings.annotator_name} ({settings.annotator_version}), {readability.__title__} ({readability.__version__})"
    )
    logger.debug(modification_meta)

    duration = int(time()) - modification_timestamp_seconds
    logger.info("Processed in %d seconds", duration)

    return TextImagerResponse(
        results=results,
        meta=meta,
        modification_meta=modification_meta,
    )
