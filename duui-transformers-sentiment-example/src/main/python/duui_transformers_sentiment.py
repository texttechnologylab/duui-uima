import logging
from platform import python_version
from sys import version as sys_version
from time import time
from typing import List, Optional

import torch
from cassis import load_typesystem
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseSettings, BaseModel
from transformers import pipeline, __version__ as transformers_version


class Settings(BaseSettings):
    """
    Tool settings, this is used to configure the tool using environment variables given to Docker
    """

    # Name of annotator
    annotator_name: str

    # Version of annotator
    annotator_version: str

    # Log level
    log_level: Optional[str]

    class Config:
        """
        Extra settings configuration
        """

        # Prefix for environment variables, note that env vars have to be provided fully uppercase
        env_prefix = 'ttlab_duui_transformers_sentiment_example_'


class DUUICapability(BaseModel):
    """
    Provides information about the capabilities of the annotator, this can be accessed via the "capability" field in /v1/documentation endpoint.
    """

    # List of supported languages by the annotator
    supported_languages: List[str]

    # Are results on same inputs reproducible without side effects?
    reproducible: bool


class DUUIDocumentation(BaseModel):
    """
    Provides information about the annotator, this can be accessed via the /v1/documentation endpoint.
    """

    # Name of this annotator
    annotator_name: str

    # Version of this annotator
    version: str

    # Annotator implementation language (Python, Java, ...)
    implementation_lang: Optional[str]

    # Optional map of additional meta data
    meta: Optional[dict]

    # Docker container id, if any
    docker_container_id: Optional[str]

    # Optional map of supported parameters
    parameters: Optional[dict]

    # Capabilities of this annotator
    capability: DUUICapability

    # Analysis engine XML, if available
    implementation_specific: Optional[str]


class UimaAnnotationMeta(BaseModel):
    """
    Metadata that is added to each annotation, this is used to track the annotator and model version.
    """

    # Name and version of the annotator
    name: str
    version: str

    # Name and version of the internal model used
    modelName: str
    modelVersion: str


class UimaDocumentModification(BaseModel):
    """
    Metadata that is added to the document once per tool, this can be used to track changes to the document.
    """

    # User that modified the document, at the moment this is always "DUUI" but should be set by DUUI internally automatically to the actual user
    user: str

    # Timestamp of the modification in seconds since epoch
    timestamp: int

    # Comment about the modification, e.g. this could contain the name and version of the tool, more relevant if the user can modify the comment later
    comment: str


class DUUIRequest(BaseModel):
    """
    This is the request sent by DUUI to this tool, i.e. the input data. This is beeing created by the Lua transformation and is thus specific to the tool.
    """

    # The full text to be analyzed
    text: str

    # The language of the text
    lang: str

    # The length of the document
    doc_len: int


class DUUIResponse(BaseModel):
    """
    This is the response of this tool back to DUUI, i.e. the output data. This is beeing transformed back to UIMA/CAS by Lua and is thus specific to the tool.
    """

    # The sentiment label, i.e. -1, 0 or 1 showing the polarity of the sentiment
    sentiment_label: int

    # The sentiment score, i.e. the confidence of the sentiment
    sentiment_score: float

    # Metadata
    meta: Optional[UimaAnnotationMeta]
    modification_meta: Optional[UimaDocumentModification]


# Initialize settings, this will pull the settings from the environment
settings = Settings()

# Set up logging
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)
logger.info("TTLab DUUI Transformers Sentiment Example")
logger.info("Name: %s", settings.annotator_name)
logger.info("Version: %s", settings.annotator_version)

# Load the type system
typesystem_filename = 'src/main/resources/TypeSystemSentiment.xml'
logger.info("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    logger.debug("Base typesystem:")
    logger.debug(typesystem.to_xml())

# Load the Lua communication layer
lua_communication_script_filename = "src/main/lua/duui_transformers_sentiment.lua"
logger.info("Loading Lua communication script from \"%s\"", lua_communication_script_filename)
with open(lua_communication_script_filename, 'rb') as f:
    lua_communication_script = f.read().decode("utf-8")
    logger.debug("Lua communication script:")
    logger.debug(lua_communication_script)

# Load the model
model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    revision="f3e34b6c30bf27b6649f72eca85d0bbe79df1e55",
    top_k=3
)

# Start the FastAPI app and provide some meta information, these are accessible via the /docs endpoint
app = FastAPI(
    title=settings.annotator_name,
    description="Transformers-based sentiment analysis for TTLab DUUI",
    version=settings.annotator_version,
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "TTLab Team",
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
    """
    This is a DUUI API endpoint that needs to be present in every tool.
    :return: The Lua communication script
    """
    return lua_communication_script


@app.get("/v1/documentation")
def get_documentation() -> DUUIDocumentation:
    """
    This is a DUUI API endpoint that needs to be present in every tool.
    :return: Documentation about the annotator in a structured format
    """
    capabilities = DUUICapability(
        supported_languages=["ar", "en", "fr", "de", "hi", "it", "sp", "pt"],
        reproducible=True
    )

    documentation = DUUIDocumentation(
        annotator_name=settings.annotator_name,
        version=settings.annotator_version,
        implementation_lang="Python",
        meta={
            "python_version": python_version(),
            "python_version_full": sys_version,
            "transformers_version": transformers_version,
            "torch_version": torch.__version__,
        },
        docker_container_id="[TODO]",
        parameters={
            "model_name": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        },
        capability=capabilities,
        implementation_specific=None,
    )

    return documentation


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
    This is a DUUI API endpoint that needs to be present in every tool. This is the main processing endpoint, that will be called for each document. It receives the data produced by the Lua transformation script and returns the processed data, that will then be transformed back by Lua. Note that the data handling is specific for each tool.
    :param request: The request object containing the data transformed by Lua.
    :return: The processed data.
    """
    meta = None
    modification_meta = None
    sentiment_label = None
    sentiment_score = None

    try:
        modification_timestamp_seconds = int(time())

        logger.debug("Received:")
        logger.debug(request)

        # Run the sentiment analysis
        result = model(
            request.text,
            truncation=True,
            padding=True,
            max_length=512
        )
        logger.debug(result)

        # get the top sentiment label, i.e. "Positive"
        sentiment_score = result[0][0]["score"]

        # map the label to the sentiment type, i.e. to -1, 0 or 1
        label = result[0][0]["label"]
        sentiment_mapping = {
            "Positive": 1,
            "Neutral": 0,
            "Negative": -1
        }
        sentiment_label = sentiment_mapping[label]

        # Add annotation metadata, this is used to track the annotator and model version
        meta = UimaAnnotationMeta(
            name=settings.annotator_name,
            version=settings.annotator_version,
            modelName="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            modelVersion="f3e34b6c30bf27b6649f72eca85d0bbe79df1e55",
        )

        # Add document modification info, this can be useful to track changes to the document
        modification_meta_comment = f"{settings.annotator_name} ({settings.annotator_version})"
        modification_meta = UimaDocumentModification(
            user="DUUI",
            timestamp=modification_timestamp_seconds,
            comment=modification_meta_comment
        )

    except Exception as ex:
        logger.exception(ex)

    # Return the response back to DUUI where it will be transformed using Lua
    return DUUIResponse(
        sentiment_label=sentiment_label,
        sentiment_score=sentiment_score,
        meta=meta,
        modification_meta=modification_meta
    )
