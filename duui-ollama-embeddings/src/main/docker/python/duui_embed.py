from typing import Dict, List, Optional
import json
import torch
from cassis import *
from fastapi import FastAPI, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings
from starlette.responses import JSONResponse
from sentence_transformers import SentenceTransformer
import requests
import math


class Embeddings(BaseModel):
    embeddings: List[float]
    begin: int
    end: int

# Request sent by DUUI
class DUUIRequest(BaseModel):
    apiUrl: str
    text: str
    model: str
    apiKey: str
    chunkSize: int

# Response of this annotator
class DUUIResponse(BaseModel):
    # List of annotated:
    # - embeddings
    embeddings: List[Embeddings]
    source: str

# Documentation response
class DUUIDocumentation(BaseModel):
    # Name of this annotator
    annotator_name: str
    # Version of this annotator
    version: str
    # Annotator implementation language (Python, Java, ...)
    implementation_lang: str


class Settings(BaseSettings):
    # Name of the Model
    model_name: Optional[str] = "base"
    duui_tool_name: Optional[str] = "mxbai-embed-large-v1"
    duui_tool_version: Optional[str] = "1.0"


settings = Settings()

# Start fastapi
app = FastAPI(
    docs_url="/api",
    redoc_url=None,
    title="Jina Embeddings V4",
    description="Embedding generation for TTLab DUUI",
    version="1.0",
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "Daniel Bundan",
        "url": "https://bundan.me",
        "email": "s1486849@stud.uni-frankfurt.de",
    },
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)

# Load the Lua communication script
communication = "communication.lua"
with open(communication, 'rb') as f:
    communication = f.read().decode("utf-8")


# Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = 'typesystem.xml'
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)


# Get input / output of the annotator
@app.get("/v1/details/input_output")
def get_input_output() -> JSONResponse:
    json_item = {
        "inputs": [ ],
        "outputs": ["org.texttechnologylab.uima.type.Embedding"]
    }

    json_compatible_item_data = jsonable_encoder(json_item)
    return JSONResponse(content=json_compatible_item_data)


# Get typesystem of this annotator
@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    xml = typesystem.to_xml()
    xml_content = xml.encode("utf-8")

    return Response(
        content=xml_content,
        media_type="application/xml"
    )


# Return Lua communication script
@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return communication


# Return documentation info
@app.get("/v1/documentation")
def get_documentation() -> DUUIDocumentation:

    documentation = DUUIDocumentation(
        annotator_name=settings.duui_tool_name,
        version=settings.duui_tool_version,
        implementation_lang="Python",
    )
    return documentation


model = None
def getModel():
    global model

    if model is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1").to(device)

    return model


def split_into_chunks(text: str, chunk_size: int) -> List[tuple]:
    """Split text into (start_pos, chunk_text) tuples, never cutting inside a word."""
    result = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            result.append((start, text[start:]))
            break
        # If the cut lands inside a word, step back to the last space
        if text[end] != ' ':
            space_pos = text.rfind(' ', start, end)
            if space_pos != -1:
                end = space_pos
        result.append((start, text[start:end]))
        # Skip the space so the next chunk doesn't begin with one
        start = end + 1 if text[end] == ' ' else end
    return result


# Process request from DUUI
@app.post("/v1/process")
def post_process(request: DUUIRequest) -> DUUIResponse:

    chunk_data = split_into_chunks(request.text, request.chunkSize)
    chunks = [chunk_text for _, chunk_text in chunk_data]

    embeddings = []

    if not request.model:
        queries = ['Represent this sentence for searching relevant passages: ' + c for c in chunks]
        with torch.no_grad():
            results = getModel().encode(queries)
        for (begin, chunk_text), vec in zip(chunk_data, results):
            embeddings.append(Embeddings(
                begin=begin,
                end=begin + len(chunk_text) - 1,
                embeddings=vec.tolist()
            ))
    else:
        model_name = request.model
        api_key = request.apiKey

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        payload = {"model": model_name, "input": chunks}
        response = requests.post(request.apiUrl, json=payload, headers=headers)
        response.raise_for_status()

        result_embeddings = response.json()["embeddings"]
        for (begin, chunk_text), vec in zip(chunk_data, result_embeddings):
            embeddings.append(Embeddings(
                begin=begin,
                end=begin + len(chunk_text) - 1,
                embeddings=vec
            ))

    return DUUIResponse(embeddings=embeddings, source=model_name if request.model else "mixedbread-ai/mxbai-embed-large-v1")


#if __name__ == "__main__":
#  uvicorn.run("duui_embed:app", host="0.0.0.0", port=9714, workers=1)
