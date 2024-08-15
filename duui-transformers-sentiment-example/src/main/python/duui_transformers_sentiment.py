import logging
from typing import Optional

from cassis import load_typesystem
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseSettings, BaseModel
from transformers import pipeline


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


# Initialize settings, this will pull the settings from the environment
settings = Settings()

# Set up logging using the log level provided in the settings
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
    sentiment_label = None
    sentiment_score = None

    try:
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

    except Exception as ex:
        logger.exception(ex)

    # Return the response back to DUUI where it will be transformed using Lua
    return DUUIResponse(
        sentiment_label=sentiment_label,
        sentiment_score=sentiment_score
    )
