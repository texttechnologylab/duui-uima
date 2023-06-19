from typing import List

from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel

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
    doc_length: int
    doc_text: str


# Response of this annotator
# Note, this is transformed by the Lua script
class DUUIResponse(BaseModel):
    # List of Sentiment
    sentiments: List[SentimentBert]


def getProbabilitiesOutOfList(probabilities: [[str, float]]) -> (float, float, float):
    positive, neutral, negative  = 0, 0, 0
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


model = SentimentModel()
def analyse(doc_text, doc_length, sentences):
    #https://huggingface.co/oliverguhr/german-sentiment-bert
    processed_sentences = []

    classes, probabilities = model.predict_sentiment([sentence.text for sentence in sentences], output_probabilities=True)
    class_sentiment = {
        "positive": 1,
        "neutral": 0,
        "negative": -1
    }

    for i in range(len(sentences)):
        positive, neutral, negative = getProbabilitiesOutOfList(probabilities[i])
        processed_sentences.append(SentimentBert(
            iBegin=sentences[i].iBegin,
            iEnd=sentences[i].iEnd,
            sentiment=class_sentiment[classes[i]],
            probabilityPositive=positive,
            probabilityNeutral=neutral,
            probabilityNegative=negative,
        ))

    classes, probabilities = model.predict_sentiment([doc_text], output_probabilities=True)
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

    return processed_sentences


communication = ''

# Start fastapi
# TODO openapi types are not shown?
# TODO self host swagger files: https://fastapi.tiangolo.com/advanced/extending-openapi/#self-hosting-javascript-and-css-for-docs
app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title="GermanSentimentBertDUUI",
    description="GermanSentimentBert implementation for TTLab DUUI",
    version="0.1",
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "TTLab Team",
        "url": "https://texttechnologylab.org",
        "email": "abrami@em.uni-frankfurt.de",
    },
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)

# Get input / output of the annotator
@app.get("/v1/details/input_output")
def get_input_output() -> JSONResponse:
    json_item = {
        "inputs": ["de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"],
        "outputs": ["org.texttechnologylab.annotation.SentimentBert"]
    }
    return json_item


# Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = '../static/dkpro-core-types.xml'
with open(typesystem_filename, 'rb') as f:
    typesystem = f.read()
# Get typesystem of this annotator
@app.get("/v1/typesystem")
def get_typesystem() -> Response:

    # TODO remove cassis dependency, as only needed for typesystem at the moment?
    return Response(
        content=typesystem,
        media_type="application/xml"
    )


# Load the Lua communication script
communication = "../static/communication.lua"
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
    length = request.doc_length
    sentences = request.sentences
    sentiments = analyse(doc_text, length, sentences)
    print("Analyse fertig!")

    # Return data as JSON
    return DUUIResponse(
        sentiments=sentiments
    )

# out for Docker-Image
# if __name__ == "__main__":
#     uvicorn.run("duui_german_sentiment_bert:app", host="0.0.0.0", port=9716, workers=1)
