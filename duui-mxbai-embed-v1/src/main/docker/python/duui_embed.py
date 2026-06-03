from typing import Dict, List, Optional, Union
import json
import torch
import uvicorn
from cassis import *
from fastapi import FastAPI, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings
from starlette.responses import JSONResponse
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaEmbeddings
from openai import OpenAI as OpenAIClient
import math


class Embeddings(BaseModel):
    embeddings: List[float]
    begin: int
    end: int

# Request sent by DUUI
# Note, this is transformed by the Lua script
class DUUIRequest(BaseModel):
    # sentences
    text: str
    ollamaConfig: Optional[Dict] = None
    chunkSize: int

    # DUUI passes all params as strings; this validator coerces a JSON string
    # (with or without surrounding braces) into a dict transparently.
    @field_validator("ollamaConfig", mode="before")
    @classmethod
    def parse_ollama_config(cls, v):
        if not isinstance(v, str):
            return v
        v = v.strip()
        for candidate in (v, "{" + v + "}"):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        raise ValueError(f"ollamaConfig could not be parsed as a JSON object: {v!r}")

# Response of this annotator
# Note, this is transformed by the Lua script
class DUUIResponse(BaseModel):
    # List of annotated:
    # - embeddings
    embeddings: List[Embeddings]

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


# settings + cache
settings = Settings()
lru_cache_with_size = lru_cache(maxsize=3)

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
    # TODO remove cassis dependency, as only needed for typesystem at the moment?
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

_ollama_model = None
_openai_clients: Dict = {}

def getEmbedding(text: str, ollamaConfig: Dict) -> List[float]:
    """Return an embedding vector for *text* using the configured backend.

    If *ollamaConfig* contains an ``apiKey`` the server is assumed to expose an
    OpenAI-compatible ``/v1/embeddings`` endpoint (e.g. TTLab LLM gateway).
    Otherwise the native Ollama ``/api/embed`` protocol is used.
    """
    global _ollama_model, _openai_clients

    model    = ollamaConfig["model"]
    base_url = ollamaConfig.get("url") or ollamaConfig.get("base_url", "")
    api_key  = ollamaConfig.get("apiKey") or ollamaConfig.get("api_key")

    if api_key:
        # ── OpenAI-compatible endpoint ──────────────────────────────────────
        # Use the URL as-is; the caller must supply the correct base URL
        # (typically ending in /v1) so the client can append /embeddings.
        url = base_url.rstrip("/")
        cache_key = (url, api_key)
        if cache_key not in _openai_clients:
            _openai_clients[cache_key] = OpenAIClient(base_url=url, api_key=api_key)
        response = _openai_clients[cache_key].embeddings.create(model=model, input=text)
        return response.data[0].embedding
    else:
        # ── Native Ollama endpoint (/api/embed) ─────────────────────────────
        if _ollama_model is None:
            kwargs: Dict = {"model": model}
            if base_url:
                kwargs["base_url"] = base_url
            _ollama_model = OllamaEmbeddings(**kwargs)
        return _ollama_model.embed_query(text)

# Process request from DUUI
@app.post("/v1/process")
def post_process(request: DUUIRequest) -> DUUIResponse:

    embeddings = []
    chunkSize = request.chunkSize
    chunk_count = math.ceil(len(request.text) / chunkSize)

    for chunk_i in range(chunk_count):
        text = request.text[chunk_i * chunkSize:chunk_i * chunkSize + chunkSize]

        if request.ollamaConfig is None:
            query = 'Represent this sentence for searching relevant passages: ' + text
            docs = [query]
            with torch.no_grad():
                result = getModel().encode(docs)
            # Embedding is of dimensionality 1024
            embeddings.append(Embeddings(
                begin=chunk_i * chunkSize,
                end=chunk_i * chunkSize + len(text) - 1,
                embeddings=result[0].tolist()
            ))
        else:
            result = getEmbedding(text, request.ollamaConfig)
            embeddings.append(Embeddings(
                begin=chunk_i * chunkSize,
                end=chunk_i * chunkSize + len(text) - 1,
                embeddings=result
            ))


    return DUUIResponse(
        embeddings=embeddings
    )


#if __name__ == "__main__":
#  uvicorn.run("duui_jina:app", host="0.0.0.0", port=9714, workers=1)
