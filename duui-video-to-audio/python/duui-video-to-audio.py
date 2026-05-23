from typing import List, Optional, Dict, Union
from time import time
from fastapi import FastAPI, Response
from fastapi.encoders import jsonable_encoder
from cassis import load_typesystem
from functools import lru_cache
from io import BytesIO
import base64
import uvicorn

import warnings
import os

import ffmpeg
import sys

from starlette.responses import PlainTextResponse, JSONResponse


def convert_to_mp3(video_path):
    # Define the output audio path
    audio_path = "tempAudio.mp3"

    # Use ffmpeg to extract audio from the video
    ffmpeg.input(video_path).output(audio_path).run()

    # Read the audio file and convert it to base64
    with open(audio_path, "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

    # Remove the temporary files
    os.remove(video_path)
    os.remove(audio_path)

    return audio_base64


class DUUIRequest(BaseModel):
    # The texts language
    video: str

class DUUIResponse(BaseModel):
    # The texts language
    audio: str
    mimetype: str

app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title= "Video2Audio",
    description="Video To Audio Component",
    version= "1.0",
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "Peter Kannewitz",
        "email": "pk35momo@studserv.uni-leipzig.de",
    },
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)

lua_communication_script_filename = "duui_video_to_audio.lua"

with open(lua_communication_script_filename, 'rb') as f:
    lua_communication_script = f.read().decode("utf-8")

typesystem_filename = 'TypeSystem.xml'
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)


@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    # TODO rimgve cassis dependency, as only needed for typesystem at the moment?
    xml = typesystem.to_xml()
    xml_content = xml.encode("utf-8")

    return Response(
        content=xml_content,
        media_type="application/xml"
    )

# Return Lua communication script
@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return lua_communication_script

# Process request from DUUI
@app.post("/v1/process")
def post_process(request: DUUIRequest):
    # Fetch model-related information
    # this video is base64
    video = request.video

    try:
        # convert base64 to video mp4, and saved it locallz
        with open("tempVideo.mp4", "wb") as f:
                f.write(base64.b64decode(request.video))

        # 1 - read the file
        # 2- extract the audio
        # 3- covert the audio into base64
        # 4- send back base64 audio
        # Convert the video to MP3 and get the base64-encoded audio
        audio_base64 = convert_to_mp3("tempVideo.mp4")

           # Print or send back the base64-encoded audio
        print(audio_base64)

        return DUUIResponse(
             audio = audio_base64,
             mimetype = "audio/mp3"
        )
    except Exception as e:
        print(str(e))


if __name__ == "__main__":
  uvicorn.run("duui-video-to-audio:app", host="0.0.0.0", port=9714, workers=1)
