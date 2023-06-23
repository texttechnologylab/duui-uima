from typing import List
import uvicorn
from cassis import *
from fastapi import FastAPI, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, BaseSettings
from starlette.responses import JSONResponse
from functools import lru_cache


# Paraphrase
from paraphraser import AutoParaphraser


class Sentence(BaseModel):
    """
    Models: de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence
    """

    begin: int
    end: int
    coveredText: str


def stdl(lst: list) -> List[Sentence]:
    sent_list = []
    for i in range(len(lst)):
        sent_list.append(Sentence(**{"begin": lst[i]["begin"],
                                     "end": lst[i]["end"],
                                     "coveredText": lst[i]["coveredText"]}))
    return sent_list

class Paraphrase(BaseModel):
    begin: int
    end: int
    paraphrased_text: str


# Request sent by DUUI
# Note, this is transformed by the Lua script
class DUUIRequest(BaseModel):
    # The text to process
    sentences: List[Sentence]


# Response of this annotator
# Note, this is transformed by the Lua script
class DUUIResponse(BaseModel):
    # List of Paraphrases
    paraphrases: List[List[Paraphrase]]


# Documentation response
class TextImagerDocumentation(BaseModel):
    # Name of this annotator
    annotator_name: str
    # Version of this annotator
    version: str
    # Annotator implementation language (Python, Java, ...)
    implementation_lang: str


class Settings(BaseSettings):
    # Name of the Model
    model_name: str

    # use gpu
    cuda: int

    # cuda_id
    gpu_id: int

    # meta data
    textimager_para_annotator_name: str
    textimager_para_annotator_version: str


# settings + cache
settings = Settings()
lru_cache_with_size = lru_cache(maxsize=3)

config = {"cuda": bool(settings.cuda),
          "gpu_id": settings.gpu_id,
          "model_name": settings.model_name}


@lru_cache_with_size
def load_paraphraser(**kwargs):
    # loads a Paraphrasing-Model
    return AutoParaphraser(**kwargs)


# Start fastapi
# TODO openapi types are not shown?
# TODO self host swagger files: https://fastapi.tiangolo.com/advanced/extending-openapi/#self-hosting-javascript-and-css-for-docs
app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title="Paraphraser",
    description="Paraphraser implementation for TTLab TextImager DUUI",
    version="0.1",
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "TTLab Team",
        "url": "https://texttechnologylab.org",
        "email": "leon.hammerla@gmx.de",
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
typesystem_filename = 'dkpro-core-types.xml'
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)


# Get input / output of the annotator
@app.get("/v1/details/input_output")
def get_input_output() -> JSONResponse:
    json_item = {
        "inputs": ["de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"],
        "outputs": ["org.texttechnologylab.annotation.Paraphrase"]
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
def get_documentation() -> TextImagerDocumentation:

    documentation = TextImagerDocumentation(
        annotator_name=settings.textimager_para_annotator_name,
        version=settings.textimager_para_annotator_version,
        implementation_lang="Python",
    )
    return documentation



# Process request from DUUI
@app.post("/v1/process")
def post_process(request: DUUIRequest) -> DUUIResponse:

    # load paraphraser-model
    paraphraser = load_paraphraser(**config)
    # get input sentences
    candidate_sentences = request.sentences
    # candidate_sentences = [candidate for candidate in candidate_sentences if len(candidate.coveredText) > 10]
    results = []
    # iterate over candidate base-sentences
    for candidate in candidate_sentences:
        paraphrases = paraphraser.generate(candidate.coveredText)
        results.append([Paraphrase(**{"begin": candidate.begin,
                                      "end": candidate.end,
                                      "paraphrased_text": para}) for para in paraphrases])

    # Return data as JSON
    return DUUIResponse(
        paraphrases=results
    )


if __name__ == "__main__":
    uvicorn.run("duui_paraphraser:app", host="0.0.0.0", port=9714, workers=1)