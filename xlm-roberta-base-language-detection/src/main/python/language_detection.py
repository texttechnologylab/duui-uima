from typing import List

from fastapi import FastAPI, Response
from fastapi.openapi.utils import get_openapi
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel

from transformers import pipeline

import uvicorn


class Language(BaseModel):
    language: str
    score: float
    iBegin: int
    iEnd: int


# Represents a part of a speech like a sentence or a paragraph or even the hole text.
class PartOfSpeech(BaseModel):
    text: str
    iBegin: int
    iEnd: int


# Request sent by DUUI
# Note, this is transformed by the Lua script
class DUUIRequest(BaseModel):
    part_of_speeches: List[PartOfSpeech]
    top_k: int


# Response of this annotator
# Note, this is transformed by the Lua script
class DUUIResponse(BaseModel):
    languages: List[Language]


# Creates an instance of the pipeline.
# Device = 0 allows the pipeline to use the gpu, -1 forces cpu usage
try:
    language_detector = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection", top_k=None, device=0)
except RuntimeError as e:
    print("RuntimeError while instantiating the pipeline.")
    print("Retrying with CPU")
    language_detector = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection", top_k=None, device=-1)


def analyse(part_of_speeches: List[PartOfSpeech], top_k):
    languages: List[Language] = list()

    for part_of_speech in part_of_speeches:
        try:
            detected_languages = language_detector(part_of_speech.text)[0]

            if top_k > 0:
                detected_languages = detected_languages[:top_k]

            for detected_language in detected_languages:
                language = Language(
                    language=detected_language["label"],
                    score=detected_language["score"],
                    iBegin=part_of_speech.iBegin,
                    iEnd=part_of_speech.iEnd,
                )
                languages.append(language)
        except RuntimeError as e:
            # To catch the following exception: RuntimeError: The size of tensor a (28206) must match the size of tensor b (512) at non-singleton dimension 1
            # That occurs when the text is too big.
            print(f"RuntimeError: Skipping the part of speech from index {part_of_speech.iBegin} to {part_of_speech.iEnd}.")

    return languages


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
        "inputs": [],
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

    languages = analyse(request.part_of_speeches, request.top_k)

    # Return data as JSON
    return DUUIResponse(
        languages=languages
    )


# Documentation for api
openapi_schema = get_openapi(
    title="xlm-roberta-base-language-detection ",
    description="A implementation of the papluca/xlm-roberta-base-language-detection for TTLab DUUI",
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
    uvicorn.run("language_detection:app", host="0.0.0.0", port=9714, workers=1)
