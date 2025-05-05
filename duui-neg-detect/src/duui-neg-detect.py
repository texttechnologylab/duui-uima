import os
from io import BytesIO
from typing import List, Optional, Any
import uvicorn
from cassis import *
from fastapi import FastAPI, Response, UploadFile, File, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import PlainTextResponse
from pydantic_settings import BaseSettings
from pydantic import BaseModel
from starlette.responses import JSONResponse
from functools import lru_cache
from neg_detect import Pipeline, CueBertInference, ScopeBertInference

from neg_detect_utils import Token, Sentence, Negation


BP = os.path.realpath(os.path.join(os.path.realpath(__file__), "../.."))


# Request sent by DUUI
# Note, this is transformed by the Lua script
class DUUIRequest(BaseModel):
    sentences: Optional[List[Sentence]] = None
    tokens: Optional[List[List[Token]]] = None

# Response of this annotator
# Note, this is transformed by the Lua script
class DUUIResponse(BaseModel):
    negations: Optional[List[Negation]] = None

    accepted: bool


# Documentation response
class TextImagerDocumentation(BaseModel):
    # Name of this annotator
    annotator_name: str
    # Version of this annotator
    version: str
    # Annotator implementation language (Python, Java, ...)
    implementation_lang: str


class Settings(BaseSettings):
    neg_detect_version: float = 0.1
    cuda: str = "cuda:0"
    cue_detection: bool = True
    scope_detection: bool = True

# settings + cache
settings = Settings()
lru_cache_with_size = lru_cache(maxsize=2)


config = {"cuda": settings.cuda,
          "cue_detection": settings.cue_detection,
          "scope_detection": settings.scope_detection}


@lru_cache_with_size
def load_pipeline(**kwargs) -> Pipeline:
    # loads a trankit-Model
    if kwargs.get("cuda") is not None:
        cuda = kwargs.get("cuda")
    else:
        cuda = "cuda:0"

    components = []
    if kwargs.get("cue_detection") is not None:
        if kwargs["cue_detection"]:
            components.append(CueBertInference)
    else:
        components.append(CueBertInference)

    if kwargs.get("scope_detection") is not None:
        if kwargs["scope_detection"]:
            components.append(ScopeBertInference)
    else:
        components.append(ScopeBertInference)
    return Pipeline(components=components)


# Start fastapi
# TODO openapi types are not shown?
# TODO self host swagger files: https://fastapi.tiangolo.com/advanced/extending-openapi/#self-hosting-javascript-and-css-for-docs
app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title="Neg-Detect Component",
    description="Neg-Detect Component for TTLab TextImager DUUI",
    version="0.1",
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "TTLab Team",
        "url": "https://texttechnologylab.org",
        "email": "hammerla@em.uni-frankfurt.de",
    },
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)

# Load the Lua communication script
communication = f"{BP}/src/communication.lua"
with open(communication, 'rb') as f:
    communication = f.read().decode("utf-8")


# Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = f'{BP}/src/dkpro-core-types.xml'
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)


# Get input / output of the annotator
@app.get("/v1/details/input_output")
def get_input_output() -> JSONResponse:
    json_item = {
        "inputs": ["de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence",
                   "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"],
        "outputs": [
                    "org.texttechnologylab.annotation.negation.CompleteNegation"]
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
        annotator_name=f"Neg-Detect-Version: {settings.neg_detect_version}",
        version=str(settings.neg_detect_version),
        implementation_lang="Python",
    )
    return documentation


# Process request from DUUI
@app.post("/v1/process")
def process(request: DUUIRequest) -> DUUIResponse:
    if config.get("cuda") is not None:
        cuda = config.get("cuda")
    else:
        cuda = "cuda:0"

    input_tokens = [[tok.coveredText for tok in sent] for sent in request.tokens]

    pipeline = load_pipeline(**config)

    result = pipeline.run(input_tokens, device=cuda)

    negations = []
    for idx, seq in enumerate(result):
        tok_seq, label_seq = seq
        if "C" in label_seq:
            current_cue = None
            scopes = []
            foci = []
            events = []

            for jdx, label in enumerate(label_seq):
                if label == "C":
                    if current_cue is None:
                        current_cue = (idx, jdx)
                    else:
                        negations.append((current_cue, scopes, foci, events))
                        scopes = []
                        foci = []
                        events = []
                        current_cue = (idx, jdx)
                elif label == "S":
                    scopes.append((idx, jdx))
                elif label == "F":
                    foci.append((idx, jdx))
                elif label == "E":
                    events.append((idx, jdx))
                else:
                    pass
            negations.append((current_cue, scopes, foci, events))

    negations = [Negation(cue=request.tokens[neg[0][0]][neg[0][1]],
                          scope=[request.tokens[sco[0]][sco[1]] for sco in neg[1]],
                          event=[request.tokens[ev[0]][ev[1]] for ev in neg[3]],
                          focus=[request.tokens[foc[0]][foc[1]] for foc in neg[2]]) for neg in negations]

    return DUUIResponse(negations=negations,
                        accepted=True)



if __name__ == "__main__":
    uvicorn.run("duui-neg-detect:app", host="0.0.0.0", port=9714, workers=1)