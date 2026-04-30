import json
import logging
from functools import lru_cache
from platform import python_version
from sys import version as sys_version
from threading import Lock
from time import time
from typing import List, Optional, Union, Tuple
from urllib.parse import urlparse

from cassis import load_typesystem
from cassis.cas import Utf16CodepointOffsetConverter
from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings

# TODO: adjust paths for deployment as docker image
lua_communication_script_path = "../lua/communication_layer.lua"
typesystem_path = "../resources/typesystem.xml"

# Settings
# These are automatically loaded from env variables
class Settings(BaseSettings):
    # Name of this annotator
    annotator_name: str
    # Version of this annotator
    annotator_version: str
    # Log level
    log_level: str

    class Config:
        env_prefix = "DUUI_NEER_MATCH_"

# Load settings
settings = Settings()

# Init logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=settings.log_level,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.info("TTLab Neer Match Annotator")
logger.info("Name: %s", settings.annotator_name)
logger.info("Version: %s", settings.annotator_version)

class DuuiRequest(BaseModel):
    text: str
    lang: str

class DuuiResponse(BaseModel):
    text: str

# load Lua communication script
lua_communication_script: str
with open(lua_communication_script_path, "r") as f:
    lua_communication_script = f.read()
    logger.info("Loaded Lua communication script")
    logger.debug("Lua communication script:\n%s", lua_communication_script)

# load typesystem
typesystem: str
with open(typesystem_path, "r") as f:
    typesystem = f.read()
    logger.info("Loaded typesystem")
    logger.debug("Typesystem:\n%s", typesystem)


# FastAPI app
app = FastAPI(
    title=settings.annotator_name,
    description="Annotator for matching entities using neer-match",
    version=settings.annotator_version,
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)

# Return Lua communication script
@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return lua_communication_script

# Return typesystem
@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    return Response(content=typesystem, media_type="application/xml")

# process duui request
@app.post("/v1/process")
def post_process(request: DuuiRequest) -> DuuiResponse:
    logger.info("Received request with text of length %d and language '%s'", len(request.text), request.lang)
    response_text = request.text.upper()  # TODO: replace with actual processing logic
    return DuuiResponse(text=response_text)
