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

class Text(BaseModel):
    text: str
    begin: int
    end: int

class Embeddings(BaseModel):
    embeddings: List[float]
    begin: int
    end: int

# Request sent by DUUI
# Note, this is transformed by the Lua script
class DUUIRequest(BaseModel):
    # sentences
    texts: List[Text]

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

#config = {"name": settings.model_name}
config = {"name": "base"}


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
        "inputs": ["de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"],
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
        model = SentenceTransformer("jinaai/jina-embeddings-v4", model_kwargs={'default_task': 'retrieval'}, trust_remote_code=True)
        
    return model

# Process request from DUUI
@app.post("/v1/process")
def post_process(request: DUUIRequest) -> DUUIResponse:
        
    embeddings = []
    
    for i in range(len(request.texts)):
        
        text = request.texts[i].text
        begin = request.texts[i].begin
        end = request.texts[i].end
            
        generated_embeddings = getModel().encode(text)
                    
        embeddings.append(Embeddings(
            begin=begin,
            end=end,
            embeddings=generated_embeddings
        ))
    
    return DUUIResponse(
        embeddings=embeddings
    )


#if __name__ == "__main__":
#  uvicorn.run("duui_jina:app", host="0.0.0.0", port=9714, workers=1)
