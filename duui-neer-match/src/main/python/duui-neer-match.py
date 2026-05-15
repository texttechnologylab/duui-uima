import logging
import os.path
from functools import lru_cache
from typing import Annotated, Any, List, Optional, Literal, Union

import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse
from neer_match.matching_model import DLMatchingModel, NSMatchingModel
from neer_match.similarity_map import SimilarityMap
from pydantic import BaseModel, Field
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
    # limit: Optional[int] = None
    # the batch size for processing (if supported by the model)
    batch_size: int = 16
    # the model to use for matching
    model: str = "example1"


class DocumentEntity(BaseModel):
    # the text of the entity
    text: str
    # enity id
    entity_id: Union[str, int]


class DuuiRequest(BaseModel):
    pipeline_id: str
    # The entities to be matched, as list of strings
    entities: List[DocumentEntity]


class DuuiFinalizeRequest(BaseModel):
    pipeline_id: str
    clear_storage: bool = True
    # Matching properties
    properties: NeerMatchProperties


class MatchPrediction(BaseModel):
    document_1_entity: DocumentEntity
    document_2_entity: DocumentEntity
    # the similarity score between 0 and 1
    score: float


class MatchResult(BaseModel):
    document_1_index: int
    document_2_index: int
    predictions: List[MatchPrediction]


class DuuiResponse(BaseModel):
    pipeline_id: str
    stored_index: int
    stored_count: int

class DuuiFinalizeResponse(BaseModel):
    pipeline_id: str
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
    # Initialize model with dummy data to build the model architecture, then load weights
    model.predict(pd.DataFrame({"value": ["example"]}), pd.DataFrame({"value": ["example"]}))
    model.load_weights(model_file_path)
    return model


class PipelineDocument(BaseModel):
    entities: List[DocumentEntity]


class PipelineEntity(BaseModel):
    pipeline_id: str
    documents: List[PipelineDocument]


class PipelineStorage:
    pipelines: dict[str, PipelineEntity]

    def __init__(self):
        self.pipelines = {}

    def store(self, pipeline_id: str, entities: List[DocumentEntity]) -> int:
        if pipeline_id not in self.pipelines:
            self.pipelines[pipeline_id] = PipelineEntity(pipeline_id=pipeline_id, documents=[])
        stored_index = len(self.pipelines[pipeline_id].documents)
        self.pipelines[pipeline_id].documents.append(PipelineDocument(entities=entities))
        return stored_index

    def get(self, pipeline_id: str) -> PipelineEntity:
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline '{pipeline_id}' not found.")
        return self.pipelines[pipeline_id]

    def clear(self, pipeline_id: str):
        if pipeline_id in self.pipelines:
            del self.pipelines[pipeline_id]


pipeline_storage = PipelineStorage()

# process duui request
@app.post("/v1/process")
def post_process(request: DuuiRequest) -> DuuiResponse:
    stored_index = pipeline_storage.store(request.pipeline_id, request.entities)
    return DuuiResponse(
        pipeline_id=request.pipeline_id,
        stored_index=stored_index,
        stored_count=len(request.entities)
    )

@app.post("/v1/finalize")
def post_finalize(request: DuuiFinalizeRequest) -> DuuiFinalizeResponse:
    pipeline_entity = pipeline_storage.get(request.pipeline_id)
    model = get_model(request.properties.model)
    results: List[MatchResult] = []
    for i, doc1 in enumerate(pipeline_entity.documents):
        for j, doc2 in enumerate(pipeline_entity.documents):
            if i >= j:
                continue
            predictions_np: np.ndarray = model.predict(
                pd.DataFrame({"value": [e.text for e in doc1.entities]}),
                pd.DataFrame({"value": [e.text for e in doc2.entities]}),
                batch_size=request.properties.batch_size,
                verbose=1
            )
            predictions: List[MatchPrediction] = []
            for idx, row in enumerate(predictions_np):
                score = row[0]
                if request.properties.threshold is not None and score < request.properties.threshold:
                    continue
                left_idx = idx // len(doc2.entities)
                right_idx = idx % len(doc2.entities)
                left_entity = doc1.entities[left_idx]
                right_entity = doc2.entities[right_idx]
                predictions.append(MatchPrediction(
                    document_1_entity=left_entity,
                    document_2_entity=right_entity,
                    score=score
                ))
            results.append(MatchResult(
                document_1_index=i,
                document_2_index=j,
                predictions=predictions
            ))
    if request.clear_storage:
        pipeline_storage.clear(request.pipeline_id)
    return DuuiFinalizeResponse(
        pipeline_id=request.pipeline_id,
        results=results
    )
