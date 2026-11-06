import logging
from platform import python_version
from sys import version as sys_version
from time import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    annotator_name: str
    annotator_version: str
    log_level: str

    class Config:
        env_prefix = 'duui_remove_annotations_'


settings = Settings()

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)
logger.info("TTLab DUUI remove annotations")
logger.info("Name: %s", settings.annotator_name)
logger.info("Version: %s", settings.annotator_version)

UIMA_TYPE_DOCUMENT_MODIFICATION = "org.texttechnologylab.annotation.DocumentModification"


class TextImagerRequest(BaseModel):
    # Comma-separated full type names
    types: Optional[str] = None
    # "true": remove only annotations of exactly these types, "false": also of their subtypes
    exact: Optional[str] = None


class AnnotationMeta(BaseModel):
    name: str
    version: str


class TextImagerResponse(BaseModel):
    types: List[str]
    exact: bool
    meta: AnnotationMeta
    timestamp: int


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
    typesystem_xml_content = f.read()

with open("communication.lua", "rb") as f:
    lua_communication_script = f.read().decode("utf-8")

app = FastAPI(
    title=settings.annotator_name,
    description="Removes all annotations of given types from the CAS, for TTLab DUUI",
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
    return TextImagerDocumentation(
        annotator_name=settings.annotator_name,
        version=settings.annotator_version,
        implementation_lang="Python",
        meta={
            "python_version": python_version(),
            "python_version_full": sys_version,
        },
        docker_container_id="[TODO]",
        parameters={
            "types": "comma-separated full type names (required)",
            "exact": ["true", "false"],
        },
        # language independent
        capability=TextImagerCapability(supported_languages=[], reproducible=True),
        implementation_specific=None,
    )


@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    return Response(content=typesystem_xml_content, media_type="application/xml")


@app.get("/v1/details/input_output")
def get_input_output() -> TextImagerInputOutput:
    return TextImagerInputOutput(inputs=[], outputs=[UIMA_TYPE_DOCUMENT_MODIFICATION])


@app.post("/v1/process")
def post_process(request: TextImagerRequest) -> TextImagerResponse:
    # the annotations are removed in the Lua script, here only the parameters are checked
    types = [t.strip() for t in (request.types or "").split(",") if t.strip()]
    if not types:
        raise HTTPException(status_code=400, detail="Parameter \"types\" is missing or empty")

    exact = (request.exact or "true").strip().lower()
    if exact not in ("true", "false"):
        raise HTTPException(status_code=400, detail=f"Parameter \"exact\" must be \"true\" or \"false\", not \"{request.exact}\"")

    logger.info("Removing types (exact: %s): %s", exact, ", ".join(types))
    return TextImagerResponse(
        types=types,
        exact=exact == "true",
        meta=AnnotationMeta(name=settings.annotator_name, version=settings.annotator_version),
        timestamp=int(time()),
    )
