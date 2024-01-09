from typing import List, Optional
import uvicorn
from cassis import *
from fastapi import FastAPI, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, BaseSettings
from starlette.responses import JSONResponse
from functools import lru_cache
import whisper
import base64

# Token
class AudioToken(BaseModel):
    """
    org.texttechnologylab.annotation.type.AudioToken
    """
    begin: int
    end: int
    timeStart: float
    timeEnd: float
    text: str


# Request sent by DUUI
# Note, this is transformed by the Lua script
class DUUIRequest(BaseModel):
    # audio in base64
    audio: str


# Response of this annotator
# Note, this is transformed by the Lua script
class DUUIResponse(BaseModel):
    # List of annotated:
    # - audiotoken
    audio_token: List[AudioToken]

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
    model_name: Optional[str]


# settings + cache
settings = Settings()
lru_cache_with_size = lru_cache(maxsize=3)

#config = {"name": settings.model_name}
config = {"name": "base"}


@lru_cache_with_size
def load_pipeline(**kwargs):
    # loads a trankit-Model
    return whisper.load_model(**kwargs)



# Start fastapi
# TODO openapi types are not shown?
# TODO self host swagger files: https://fastapi.tiangolo.com/advanced/extending-openapi/#self-hosting-javascript-and-css-for-docs
app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title="Whisper",
    description="Whisper Implementation for TTLab DUUI",
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
        "inputs": [],
        "outputs": ["org.texttechnologylab.annotation.type.AudioToken"]
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



# Process request from DUUI
@app.post("/v1/process")
def post_process(request: DUUIRequest) -> DUUIResponse:
    config = {"name": "base"}
    # load pipeline
    pipeline = load_pipeline(**config)

    results = []

    try:
        with open("tempAudio.mp3", "wb") as f:
            f.write(base64.b64decode(request.audio))
    except Exception as e:
        print(str(e))

    result = pipeline.transcribe(audio="tempAudio.mp3", word_timestamps=True)

    print(result)

    for index, row in result["segments"].iterrows():
        audioStart = row['start']
        audioEnd = row['end']
        text = row['text']

        results.append(AudioToken(
            timeStart=float(audioStart),
            timeEnd=float(audioEnd),
            text=text
        ))

    return DUUIResponse(
        audio_token=results
    )


if __name__ == "__main__":
    uvicorn.run("duui_whisper:app", host="0.0.0.0", port=9714, workers=1)

    # config = {"name": "base"}
    # pipeline = load_pipeline(**config)
    # result = pipeline.transcribe(audio="audio.mp3", word_timestamps=True)
    #
    # # audio = whisper.load_audio("audio.mp3")
    # # audio = whisper.pad_or_trim(audio)
    # #
    # # mel = whisper.log_melspectrogram(audio).to(pipeline.device)
    # #
    # # options = whisper.DecodingOptions()
    # # result = whisper.decode(pipeline, mel, options)
    #
    # print(result["segments"])
