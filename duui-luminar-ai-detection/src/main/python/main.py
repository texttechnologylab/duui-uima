import logging
import torch
import warnings
import nltk

from typing import Optional, Literal
from cassis import load_typesystem
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from transformers import pipeline
from luminar.detector import LuminarSequenceDetector
from luminar.utils import get_best_device

class Settings(BaseSettings):
    """
    Tool settings, this is used to configure the tool using environment variables given to Docker
    """
    annotator_name: str = "Luminar AI Detection"
    annotator_version: str = "0.1.0"
    log_level: Literal["CRITICAL","ERROR","WARNING","INFO","DEBUG","NOTSET"] = "INFO"

    model_config = SettingsConfigDict(
        env_prefix="DUUI_",      # env vars like DUUI_ANNOTATOR_NAME, etc.
        env_file=".env",         # load from a .env file if present
        extra="ignore",
    )


class DUUIRequest(BaseModel):
    """
    This is the request sent by DUUI to this tool, i.e. the input data. This is beeing created by the Lua transformation and is thus specific to the tool.
    """
    text: str
    lang: str
    doc_len: int


class DUUIResponse(BaseModel):
    """
    This is the response of this tool back to DUUI, i.e. the output data. This is beeing transformed back to UIMA/CAS by Lua and is thus specific to the tool.
    """
    detections: list

settings = Settings()
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)
logger.info("========== TTLab DUUI Luminar AI Detection ==========")
logger.info("Name: %s", settings.annotator_name)
logger.info("Version: %s", settings.annotator_version)

# Loading the typesystem
typesystem_filename = 'LuminarAIDetection.xml'
logger.info("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    logger.debug("Loaded typesystem:")
    logger.debug(typesystem.to_xml())

# Load the Lua communication layer
lua_communication = "luminar_ai_detection.lua"
logger.info("Loading Lua communication script from \"%s\"", lua_communication)
with open(lua_communication, 'rb') as f:
    lua_communication_script = f.read().decode("utf-8")
    logger.debug("Lua communication script:")
    logger.debug(lua_communication_script)

print("\n----")
detector = LuminarSequenceDetector(model_path="TheItCrOw/LuminarSeq", feature_agent="gpt2", device=get_best_device())
print("----\n")

app = FastAPI(
    title=settings.annotator_name,
    description="Luminar AI detection for TTLab DUUI",
    version=settings.annotator_version,
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "TTLab Team",
        "url": "https://texttechnologylab.org",
        "email": "boenisch@em.uni-frankfurt.de",
    },
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)

@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    """
    This is a DUUI API endpoint that needs to be present in every tool.
    :return: The Lua communication script
    """
    return lua_communication_script


@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    """
    This is a DUUI API endpoint that needs to be present in every tool.
    :return: The typesystem as XML, this should include all types the tool can produce
    """
    xml = typesystem.to_xml()
    xml_content = xml.encode("utf-8")

    return Response(
        content=xml_content,
        media_type="application/xml"
    )


@app.post("/v1/process")
def post_process(request: DUUIRequest) -> DUUIResponse:
    """
    This is a DUUI API endpoint that needs to be present in every tool.
    :param request: The request object containing the data transformed by Lua.
    :return: The processed data.
    """
    detections = []
    try:
        logger.debug("Received:")
        logger.debug(request)

        result = detector.detect(request.text)
        logger.debug("Result: ")
        logger.debug(result)

        for prob, span in zip(result['probs'], result['char_spans']):
            detections.append({
                "detectionScore": prob / 100,
                "begin": span[0],
                "end": span[0],
                "level": "SEQUENCE",
                "model": "TheItCrOw/LuminarSeq"
            })
        detections.append({
            "detectionScore": result["avg"] / 100,
            "begin": 0,
            "end": len(request.text),
            "level": "DOCUMENT",
            "model": "TheItCrOw/LuminarSeq"
        })

    except ValueError as ex:
        logger.exception("AI Detector threw a value error: ", ex)
    except Exception as ex:
        logger.exception("Unknown error: ", ex)

    # Return the response back to DUUI where it will be transformed using Lua
    return DUUIResponse(
        detections=detections,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",  # or "your_module_name:app"
        host="0.0.0.0",
        port=8000,
        reload=False  # True only for dev (not in Docker)
    )
