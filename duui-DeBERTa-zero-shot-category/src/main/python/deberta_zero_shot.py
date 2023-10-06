from typing import List

from fastapi import FastAPI, Response
from fastapi.openapi.utils import get_openapi
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel

from transformers import pipeline

import uvicorn


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
    labels: List[str]


# Response of this annotator
# Note, this is transformed by the Lua script
class DUUIResponse(BaseModel):
    # List of Sentiment
    labels: List[Label]


# Creates an instance of the pipeline.
# Device = 0 allows the pipeline to use the gpu, -1 forces cpu usage
try:
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", device=0)
except RuntimeError as e:
    print("RuntimeError while instantiating the pipeline.")
    print("Retrying with CPU")
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", device=-1)

def analyse(doc_text, labels):
    analyzed_labels = []

    text_length = len(doc_text)

    result = classifier(doc_text, labels, multi_label=True)

    labels = result["labels"]
    scores = result["scores"]

    for i in range(len(labels)):
        analyzed_labels.append(Label(label=labels[i], score=scores[i], iBegin=0, iEnd=text_length))

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
        "inputs": [""],
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
    labels = request.labels

    analysed_labels = analyse(doc_text, labels)

    # Return data as JSON
    return DUUIResponse(
        labels=analysed_labels
    )


# Documentation for api
openapi_schema = get_openapi(
    title="DeBERTa-Zero-Shot Classification",
    description="A implementation of the MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli Modell for TTLab DUUI",
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
    uvicorn.run("deberta_zero_shot:app", host="0.0.0.0", port=9714, workers=1)
