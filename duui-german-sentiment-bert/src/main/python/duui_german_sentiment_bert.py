from typing import List

from fastapi import FastAPI, Response
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.utils import get_openapi
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel

import gc
import torch

from germansentiment import SentimentModel

import uvicorn

class Sentence(BaseModel):
    text: str
    iBegin: int
    iEnd: int


class SentimentBert(BaseModel):
    sentiment: int
    iBegin: int
    iEnd: int
    probabilityPositive: float
    probabilityNeutral: float
    probabilityNegative: float


# Request sent by DUUI
# Note, this is transformed by the Lua script
class DUUIRequest(BaseModel):
    # The sentences to process
    sentences: List[Sentence]
    doc_text: str


# Response of this annotator
# Note, this is transformed by the Lua script
class DUUIResponse(BaseModel):
    # List of Sentiment
    sentiments: List[SentimentBert]


# model.predict_sentiment() returns the probabilities in a format like this:
# [["positive", 0.9], ["neutral", 0.05], ["negative", 0.05]]
# This function reformats this into a tuple of the three probabilities (0.9, 0.05, 0.05).
def getProbabilitiesOutOfList(probabilities: [[str, float]]) -> (float, float, float):
    positive, neutral, negative = 0, 0, 0
    for probability in probabilities:
        if probability[0] == "positive":
            positive = probability[1]
        elif probability[0] == "neutral":
            neutral = probability[1]
        elif probability[0] == "negative":
            negative = probability[1]
        else:
            raise ValueError("Unexpected sentiment name")
    return positive, neutral, negative

# Creates an instance of the SentimentModel.
model = SentimentModel()
def analyse(doc_text, doc_length, sentences):

    processed_sentences = []

    # A dict to map the string based sentiment results of the model.predict_sentiment() to the doubles: 1, 0, -1.
    class_sentiment = {
        "positive": 1,
        "neutral": 0,
        "negative": -1
    }

    # Sentences that are processed in one run without freeing the graphic memory
    # To big sizes like 1000 can lead to this error: Torch.cuda.OutOfMemoryError: CUDA out of memory
    # Low step sizes like 1 are slower.
    step_size = 10
    steps = (len(sentences)/step_size)
    if not steps.is_integer():
        steps = int(steps) + 1
    else:
        steps = int(steps)

    for step in range(steps):
        # The number of iterations in the step
        iterations_in_step = min(step_size, len(sentences) - (step*step_size))

        sentences_in_step = sentences[step*step_size:step*step_size+iterations_in_step]

        classes, probabilities = model.predict_sentiment([sentence.text for sentence in sentences_in_step],
                                                         output_probabilities=True)

        for i in range(iterations_in_step):
            positive, neutral, negative = getProbabilitiesOutOfList(probabilities[i])
            processed_sentences.append(SentimentBert(
                iBegin=sentences_in_step[i].iBegin,
                iEnd=sentences_in_step[i].iEnd,
                sentiment=class_sentiment[classes[i]],
                probabilityPositive=positive,
                probabilityNeutral=neutral,
                probabilityNegative=negative,
            ))

        # Clears graphic memory
        torch.cuda.empty_cache()

    # Get sentiment for the hole text.
    classes, probabilities = model.predict_sentiment([doc_text], output_probabilities=True)
    positive, neutral, negative = getProbabilitiesOutOfList(probabilities[0])
    processed_sentences.append(SentimentBert(
        iBegin=0,
        iEnd=doc_length,
        sentiment=class_sentiment[classes[0]],
        probabilityPositive=positive,
        probabilityNeutral=neutral,
        probabilityNegative=negative,
    ))

    # Clears graphic memory
    torch.cuda.empty_cache()

    return processed_sentences


communication = ''

# Start fastapi
app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url="/redoc",
)

# Get input and output of the annotator
@app.get("/v1/details/input_output")
def get_input_output() -> JSONResponse:
    json_item = {
        "inputs": ["de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"],
        "outputs": ["org.texttechnologylab.annotation.SentimentBert"]
    }
    return json_item


# Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = 'dkpro-core-types.xml'
with open(typesystem_filename, 'rb') as f:
    typesystem = f.read()
# Get typesystem of this annotator
@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    return Response(
        content=typesystem,
        media_type="application/xml"
    )


# Load the Lua communication script
communication = "communication.lua"
with open(communication, 'rb') as f:
    communication = f.read().decode("utf-8")

# Return Lua communication script
@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return communication


# Process request from DUUI
@app.post("/v1/process")
def post_process(request: DUUIRequest) -> DUUIResponse:
    doc_text = request.doc_text

    length = len(doc_text)

    sentences = request.sentences

    sentiments = analyse(doc_text, length, sentences)

    # Return data as JSON
    return DUUIResponse(
        sentiments=sentiments
    )

# Documentation for api
openapi_schema = get_openapi(
    title="GermanSentimentBertDUUI",
    description="GermanSentimentBert implementation for TTLab DUUI",
    version="0.1",
    routes=app.routes
)

# Extra Documentation not supported by FastApi
# https://fastapi.tiangolo.com/how-to/extending-openapi/#self-hosting-javascript-and-css-for-docs
# https://spec.openapis.org/oas/v3.1.0#infoObject
openapi_schema["info"]["contact"] = {"name": "TTLab Team", "url": "https://texttechnologylab.org", "email": "abrami@em.uni-frankfurt.de"}
openapi_schema["info"]["termsOfService"] = "https://www.texttechnologylab.org/legal_notice/"
openapi_schema["info"]["license"] = {"name": "AGPL", "url": "http://www.gnu.org/licenses/agpl-3.0.en.html"}
app.openapi_schema = openapi_schema

# For starting the script locally
if __name__ == "__main__":
    uvicorn.run("duui_german_sentiment_bert:app", host="0.0.0.0", port=9714, workers=1)
