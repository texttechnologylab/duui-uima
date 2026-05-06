import logging
import os.path
from functools import lru_cache
from typing import List, Optional, Literal

import pandas as pd
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse
from neer_match.matching_model import DLMatchingModel, NSMatchingModel
from neer_match.similarity_map import SimilarityMap
from pydantic import BaseModel
from pydantic_settings import BaseSettings

# TODO: adjust paths for deployment as docker image
lua_communication_script_path = "../lua/communication_layer.lua"
typesystem_path = "../resources/typesystem.xml"
models_path = "../resources/models"


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


class ModelConfig(BaseModel):
    # the name of the model (must match folder name in models_path)
    name: str
    # the type of the model, either "DL"
    type: Literal["DL"]
    # value similarity matchers
    similarity_matchers: List[str]
    # the path to the model file (relative to models_path)
    nn_model_file: str = Field(default="model.weights.h5", alias="model_file")


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


@lru_cache(maxsize=3)
def get_model(model_name: str) -> DLMatchingModel | NSMatchingModel:
    folder_path = f"{models_path}/{model_name}"
    if not os.path.exists(folder_path):
        raise ValueError(f"Model '{model_name}' not found.")
    # load model config
    config_path = f"{folder_path}/config.json"
    if not os.path.exists(config_path):
        raise ValueError(f"Model '{model_name}' invalid: missing config.json")
    model_config: ModelConfig
    with open(config_path, "r") as f:
        model_config = ModelConfig.model_validate_json(f.read())
    # load model
    model_file_path = f"{folder_path}/{model_config.nn_model_file}"
    if not os.path.exists(model_file_path):
        raise ValueError(f"Model '{model_name}' invalid: missing model file '{model_config.nn_model_file}'")
    if len(model_config.similarity_matchers) == 0:
        raise ValueError(f"Model '{model_name}' invalid: no similarity matchers specified")
    similarity_map: SimilarityMap = SimilarityMap({
        "value": model_config.similarity_matchers
    })
    model: DLMatchingModel
    if model_config.type == "DL":
        model = DLMatchingModel(similarity_map)
    elif model_config.type == "NS":
        raise ValueError(f"Model '{model_name}' invalid: model type 'NS' not supported yet")
    else:
        raise ValueError(f"Model '{model_name}' invalid: unknown model type '{model_config.type}'")
    model.load_weights(model_file_path)
    return model

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
