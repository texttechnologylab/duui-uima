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
import whisperx
import base64
import torch

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
    language: str

# Response of this annotator
# Note, this is transformed by the Lua script
class DUUIResponse(BaseModel):
    # List of annotated:
    # - audiotoken
    audio_token: List[AudioToken]
    language: str

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
    title="WhisperX audio transcription",
    description="Audio transcription for TTLab DUUI",
    version="2.1",
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "Daniel Bundan",
        "url": "bundan.me",
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
        
    try:
        with open("tempAudio", "wb") as f:
            f.write(base64.b64decode(request.audio))
    except Exception as e:
        print(str(e))
        
    # Load different pipeline depending on CUDA availability
    asr_options = {"word_timestamps":True}
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"

    if(request.language):
        model = whisperx.load_model("large-v2", device, compute_type=compute_type, language=request.language, asr_options=asr_options)
    else:
        model = whisperx.load_model("large-v2", device, compute_type=compute_type, asr_options=asr_options)


    audio = whisperx.load_audio("tempAudio")
    
    result = model.transcribe(audio, batch_size=16)

    language = request.language
    if(not language):
        language = result["language"]

    alignment_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    aligned_result = whisperx.align(result["segments"], alignment_model, metadata, "tempAudio", device)

    results = []

    current_length = 0

    for word in aligned_result["word_segments"]:

        audio_start = word.get("start")
        audio_end = word.get("end")
        text = word.get("word").strip()
        
        if(audio_start is None or audio_end is None):  # If segment is not spoken out loud, such as '-'
            continue
        
        if((len(text)) == 0 and audio_start == audio_end):  # If segment contains no information
            continue
        
        results.append(AudioToken(
            timeStart=float(audio_start),
            timeEnd=float(audio_end),
            text=text,
            begin=current_length,
            end=current_length + len(text)
        ))
        
        if(len(text) > 0):  
            current_length += len(text) + 1
    
    return DUUIResponse(
        audio_token=results,
        language=language
    )


#if __name__ == "__main__":
#  uvicorn.run("duui_whisperx:app", host="0.0.0.0", port=9714, workers=1)
