import json
import logging
from functools import lru_cache
from platform import python_version
from sys import version as sys_version
from threading import Lock
from time import time
from typing import List, Optional, Union, Tuple, Dict
from urllib.parse import urlparse

from cassis import load_typesystem
from cassis.cas import Utf16CodepointOffsetConverter
from neer_match.matching_model import DLMatchingModel
from neer_match.similarity_map import SimilarityMap
import tensorflow as tf
from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse
import pandas as pd
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

class NeerMatchProperties(BaseModel):
    # the threshold for the matching process, between 0 and 1
    threshold: Optional[float] = None
    # the maximum number of matches to return per entity
    limit: int = 10
    # the batch size for processing (if supported by the model)
    batch_size: int = 16
    # the model to use for matching
    model: str = "example1"

class DuuiRequest(BaseModel):
    # The entities to be matched, as list of strings
    entities: List[str]
    # The target texts to match against, as list of strings
    targets: List[str]
    # Optional properties for the matching process
    properties: NeerMatchProperties

class MatchSuggestion(BaseModel):
    # the matched target (value & index)
    target: str
    target_index: int
    # the similarity score between 0 and 1
    score: float

class MatchResult(BaseModel):
    # the matched entity (value & index)
    entity: str
    entity_index: int
    # the list of suggestions for this entity
    suggestions: List[MatchSuggestion]

class DuuiResponse(BaseModel):
    # the list of match results
    results: List[MatchResult]

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

def get_model(model_name: str) -> DLMatchingModel:
    # TODO: implement model loading based on model_name (and possibly caching)
    return DLMatchingModel()

# process duui request
@app.post("/v1/process")
def post_process(request: DuuiRequest) -> DuuiResponse:
    model = get_model(request.properties.model)
    left_data = pd.DataFrame({"value": request.entities})
    right_data = pd.DataFrame({"value": request.targets})

    # suggest using model
    suggestions = model.suggest(
        left_data, right_data,
        count=request.properties.limit,
        batch_size=request.properties.batch_size
    )
    suggestions.sort_values(by=["left", "prediction"], ascending=[True, False], inplace=True)

    results: List[MatchResult] = [MatchResult(
        entity=request.entities[i],
        entity_index=i,
        suggestions=[]
    ) for i in range(len(request.entities))]
    # extract suggestions and group by entity index
    for _, row in suggestions.iterrows():
        entity_index = int(row["left"])
        target_index = int(row["right"])
        score = float(row["prediction"])
        if request.properties.threshold is not None and score < request.properties.threshold:
            continue
        results[entity_index].suggestions.append(MatchSuggestion(
            target=request.targets[target_index],
            target_index=target_index,
            score=score
        ))
    return DuuiResponse(results=results)
