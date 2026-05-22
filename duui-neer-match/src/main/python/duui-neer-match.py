import logging
import os.path
from functools import lru_cache
from typing import Annotated, Any, Dict, List, Optional, Literal, Tuple, Union

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
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=settings.log_level,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.info("TTLab Neer Match Annotator")
logger.info("Name: %s", settings.annotator_name)
logger.info("Version: %s", settings.annotator_version)


class ModelEntityPropertyConfig(BaseModel):
    # the type of the property
    property_type: Literal["text", "numeric"]
    # the similarity matchers to use for this property
    similarity_matchers: List[str]


class ModelConfig(BaseModel):
    # the name of the model (must match folder name in models_path)
    name: str
    # the type of the model, either "DL"
    type: Literal["DL"]
    # the properties of the entities
    entity_properties: Dict[str, ModelEntityPropertyConfig]
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
    # optional properties of the entity
    properties: Optional[Dict[str, Union[str, int, float]]] = None


class DuuiRequest(BaseModel):
    pipeline_id: str
    # the selected annotation classes
    selection: str
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
def get_model(
    model_name: str,
) -> Tuple[Union[DLMatchingModel, NSMatchingModel], ModelConfig]:
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
        raise ValueError(
            f"Model '{model_name}' invalid: missing model file '{model_config.nn_model_file}'"
        )
    if (
        "text" not in model_config.entity_properties
        or model_config.entity_properties["text"].property_type != "text"
        or len(model_config.entity_properties["text"].similarity_matchers) == 0
    ):
        raise ValueError(
            f"Model '{model_name}' invalid: missing required 'text' property with at least one similarity matcher"
        )
    similarity_map: SimilarityMap = SimilarityMap(
        {
            prop_name: prop_config.similarity_matchers
            for prop_name, prop_config in model_config.entity_properties.items()
        }
    )
    model: DLMatchingModel
    if model_config.type == "DL":
        model = DLMatchingModel(similarity_map)
    elif model_config.type == "NS":
        raise ValueError(
            f"Model '{model_name}' invalid: model type 'NS' not supported yet"
        )
    else:
        raise ValueError(
            f"Model '{model_name}' invalid: unknown model type '{model_config.type}'"
        )
    # Initialize model with dummy data to build the model architecture, then load weights
    initialization_data: Dict[str, Any] = {}
    for prop_name, prop_config in model_config.entity_properties.items():
        if prop_config.property_type == "text":
            initialization_data[prop_name] = ["dummy"]
        elif prop_config.property_type == "numeric":
            initialization_data[prop_name] = [0]
        else:
            raise ValueError(
                f"Model '{model_name}' invalid: unknown property type '{prop_config.property_type}' for property '{prop_name}'"
            )
    model.predict(pd.DataFrame(initialization_data), pd.DataFrame(initialization_data))
    model.load_weights(model_file_path)
    return model, model_config


class PipelineDocument(BaseModel):
    entities: List[DocumentEntity]


class PipelineEntity(BaseModel):
    pipeline_id: str
    selection: str
    documents: List[PipelineDocument]


class PipelineStorage:
    pipelines: dict[str, PipelineEntity]

    def __init__(self):
        self.pipelines = {}

    def store(
        self, pipeline_id: str, entities: List[DocumentEntity], selection: str
    ) -> int:
        if pipeline_id not in self.pipelines:
            self.pipelines[pipeline_id] = PipelineEntity(
                pipeline_id=pipeline_id, selection=selection, documents=[]
            )
        pipeline_entity = self.pipelines[pipeline_id]
        if pipeline_entity.selection != selection:
            raise ValueError(
                f"Pipeline '{pipeline_id}' selection mismatch: expected '{pipeline_entity.selection}', got '{selection}'"
            )
        stored_index = len(pipeline_entity.documents)
        pipeline_entity.documents.append(PipelineDocument(entities=entities))
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
    stored_index = pipeline_storage.store(
        request.pipeline_id, request.entities, request.selection
    )
    return DuuiResponse(
        pipeline_id=request.pipeline_id,
        stored_index=stored_index,
        stored_count=len(request.entities),
    )


def build_dataframe(
    entities: List[DocumentEntity], selection: str, supported_properties: List[str]
) -> pd.DataFrame:
    columns: dict[str, List[Union[str, int, float]]] = {
        "text": [e.text for e in entities]
    }

    def select_property(
        prop_name: str, default: Union[str, int, float]
    ) -> List[Union[str, int, float]]:
        return [
            (
                e.properties[prop_name]
                if e.properties and prop_name in e.properties
                else default
            )
            for e in entities
        ]

    if selection == "de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity":
        if "value" in supported_properties:
            columns["value"] = select_property("value", "")
        if "identifier" in supported_properties:
            columns["identifier"] = select_property("identifier", "")
    elif selection == "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token":
        if "lemma" in supported_properties:
            columns["lemma"] = select_property("lemma", "")
        if "pos" in supported_properties:
            columns["pos"] = select_property("pos", "")
        if "form" in supported_properties:
            columns["form"] = select_property("form", "")
        if "stem" in supported_properties:
            columns["stem"] = select_property("stem", "")

    if any(prop not in columns for prop in supported_properties):
        raise ValueError(
            f"Some properties required by the model are not supported for selection '{selection}'. Missing properties: {[prop for prop in supported_properties if prop not in columns]}"
        )
    return pd.DataFrame(columns)


@app.post("/v1/finalize")
def post_finalize(request: DuuiFinalizeRequest) -> DuuiFinalizeResponse:
    pipeline_entity = pipeline_storage.get(request.pipeline_id)
    model, model_config = get_model(request.properties.model)
    results: List[MatchResult] = []
    supported_properties = [
        key for key in model_config.entity_properties.keys() if key != "text"
    ]
    dataframes = [
        build_dataframe(doc.entities, pipeline_entity.selection, supported_properties)
        for doc in pipeline_entity.documents
    ]
    for i, doc1 in enumerate(pipeline_entity.documents):
        for j, doc2 in enumerate(pipeline_entity.documents):
            if i >= j:
                continue
            predictions_np: np.ndarray = model.predict(
                dataframes[i],
                dataframes[j],
                batch_size=request.properties.batch_size,
                verbose=1,
            )
            predictions: List[MatchPrediction] = []
            for idx, row in enumerate(predictions_np):
                score = row[0]
                if (
                    request.properties.threshold is not None
                    and score < request.properties.threshold
                ):
                    continue
                left_idx = idx // len(doc2.entities)
                right_idx = idx % len(doc2.entities)
                left_entity = doc1.entities[left_idx]
                right_entity = doc2.entities[right_idx]
                predictions.append(
                    MatchPrediction(
                        document_1_entity=left_entity,
                        document_2_entity=right_entity,
                        score=score,
                    )
                )
            results.append(
                MatchResult(
                    document_1_index=i, document_2_index=j, predictions=predictions
                )
            )
    if request.clear_storage:
        pipeline_storage.clear(request.pipeline_id)
    return DuuiFinalizeResponse(pipeline_id=request.pipeline_id, results=results)
