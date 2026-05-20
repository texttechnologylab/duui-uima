from typing import List, Optional
import uvicorn
from cassis import *
from fastapi import FastAPI, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from starlette.responses import JSONResponse
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaEmbeddings
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
    ollamaConfig: Optional[Union[Dict, None]]
    chunkSize: int

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

ollama_model = None
def getOllamaModel(ollamaConfig: Union[Dict, None]):
    global ollama_model

    if ollama_model is None:
        ollama_model = OllamaEmbeddings(**ollamaConfig)

    return ollama_model

# Process request from DUUI
@app.post("/v1/process")
def post_process(request: DUUIRequest) -> DUUIResponse:

    embeddings = []
    chunk_count = math.ceil(len(request.text) / chunkSize)

    for chunk_i in range(chunk_count):
        text = request.text[chunk_i * chunkSize:chunk_i * chunkSize + chunkSize]

        if request.ollamaConfig == null:
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
            result = getOllamaModel(request.ollamaConfig).embed_query(text)
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
