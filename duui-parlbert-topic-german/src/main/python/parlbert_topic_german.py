from typing import List

from fastapi import FastAPI, Response
from fastapi.openapi.utils import get_openapi
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel

from transformers import pipeline

import uvicorn


class Sentence(BaseModel):
    text: str
    iBegin: int
    iEnd: int


# A label containing the label and its zero-short score
class Label(BaseModel):
    label: str
    score: float
    iBegin: int
    iEnd: int


# Request sent by DUUI
# Note, this is transformed by the Lua script
class DUUIRequest(BaseModel):
    doc_text: str
    sentences: List[Sentence]


# Response of this annotator
# Note, this is transformed by the Lua script
class DUUIResponse(BaseModel):
    # List of Sentiment
    labels: List[Label]


# Creates an instance of the pipeline.
# Device = 0 allows the pipeline to use the gpu, -1 forces cpu usage
try:
    classifier = pipeline("text-classification", model="chkla/parlbert-topic-german", top_k=None, device=0)
except RuntimeError as e:
    print("RuntimeError while instantiating the pipeline.")
    print("Retrying with CPU")
    classifier = pipeline("text-classification", model="chkla/parlbert-topic-german", top_k=None, device=-1)


def analyse_part(text, iBegin, iEnd):
    analyzed_labels = []
    try:
        result = classifier(text)[0]
        for i in range(len(result)):
            label = Label(
                label=result[i]["label"],
                score=result[i]["score"],
                iBegin=iBegin,
                iEnd=iEnd
            )
            analyzed_labels.append(label)
    except RuntimeError as e:
        # To catch the following exception: RuntimeError: The size of tensor a (28206) must match the size of tensor b (512) at non-singleton dimension 1
        # That occurs when the text is too big.
        print(f"RuntimeError: Skipping the part of speech from index {iBegin} to {iEnd}.")

    return analyzed_labels


def analyse(doc_text, sentences):

    analyzed_labels = []

    labels = analyse_part(doc_text, 0, len(doc_text))
    analyzed_labels.extend(labels)

    for sentence in sentences:
        labels = analyse_part(sentence.text, sentence.iBegin, sentence.iEnd)
        analyzed_labels.extend(labels)

    return analyzed_labels


# Start fastapi
app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
)

# Get input and output of the annotator
@app.get("/v1/details/input_output")
def get_input_output() -> JSONResponse:
    json_item = {
        "inputs": ["de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"],
        "outputs": ["org.hucompute.textimager.uima.type.category.CategoryCoveredTagged"]
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
    sentences = request.sentences

    analysed_labels = analyse(doc_text, sentences)

    # Return data as JSON
    return DUUIResponse(
        labels=analysed_labels
    )


# Documentation for api
openapi_schema = get_openapi(
    title="parlbert-topic-german",
    description="A implementation of the chkla/parlbert-topic-german Modell for TTLab DUUI",
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
    uvicorn.run("parlbert_topic_german:app", host="0.0.0.0", port=9714, workers=1)
