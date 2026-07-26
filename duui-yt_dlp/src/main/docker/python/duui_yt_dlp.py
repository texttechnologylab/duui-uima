from typing import List
import uvicorn
from cassis import *
from fastapi import FastAPI, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from starlette.responses import JSONResponse
import base64
from yt_dlp import YoutubeDL
import magic
import os
import re

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
    # video in base64
        link: str
        with_youtube_transcription: bool
        transcription_language: str
        with_cookies: str

# Response of this annotator
# Note, this is transformed by the Lua script
class DUUIResponse(BaseModel):
    # List of annotated:
    # - videotoken
    encoded_video: str
    mimetype_video: str
    youtube_audio_token: List[AudioToken]

# Documentation response
class DUUIDocumentation(BaseModel):
    # Name of this annotator
    annotator_name: str
    # Version of this annotator
    version: str
    # Annotator implementation language (Python, Java, ...)
    implementation_lang: str


# Start fastapi
app = FastAPI(
    docs_url="/api",
    redoc_url=None,
    title="Yt dlp video downloader",
    description="YouTube video and transcription downloader for TTLab DUUI",
    version="1..1",
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "Daniel Bundan",
        "url": "https://texttechnologylab.org",
        "email": "danielbundan60@gmail.com",
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
        annotator_name="PyTube",
        version="15.0.0",
        implementation_lang="Python",
    )
    return documentation


AUDIO_PATH = "./youtube_audios"

# Process request from DUUI
@app.post("/v1/process")
def post_process(request: DUUIRequest) -> DUUIResponse:

    if(not os.path.isdir(AUDIO_PATH)):
        os.mkdir(AUDIO_PATH)

    transcription_token = []

    # Download video
    transcription = None
    if(request.link.startswith("https://www.youtube.com/watch") or request.link.startswith("https://youtu.be/")):
        download_result = download_youtube(request.link, request.with_youtube_transcription, request.transcription_language, request.with_cookies)
        video = download_result[0]
        transcription = download_result[1]
    else:  # Not a YouTube video
        return DUUIResponse(
            encoded_video=request.link,
            youtube_audio_token=transcription_token
        )

    try:
        with open(video, "rb") as f:
            video_data = f.read()
        encoded = base64.b64encode(video_data).decode("utf-8")
    except Exception as e:
        print("COULD NOT OPEN AUDIO FILE: " + str(e))

    # Get transcription
    if transcription is None:
        if request.with_youtube_transcription:
            print("Video does not contain any kind of subtitles")
    else:
        p = re.compile("([0-9])+:([0-9])+:([0-9])+\\.([0-9])+ --> ([0-9])+:([0-9])+:([0-9])+\\.([0-9])+")

        with open(transcription, 'r', encoding='UTF-8') as f:
                is_auto_generated = "</c>" in f.read()

        with open(transcription, 'r', encoding='UTF-8') as f:
            start = 0
            end = 0
            text = ""
            found_first = False
            show_block = False
            current_char = 0

            while line := f.readline():
                matched = p.search(line.rstrip())
                if matched is not None:
                    if text and show_block:
                        transcription_token.append(AudioToken(
                            timeStart=float(start),
                            timeEnd=float(end),
                            text=text,
                            begin=current_char,
                            end=current_char + len(text)
                        ))
                        current_char += len(text) + 1

                    text = ""
                    found_first = True
                    show_block = False
                    timestamps = matched.group().split("-->")
                    start = timestamps[0].strip().split(":")
                    start = float("{:.2f}".format(float(start[2]) + float(start[1]) * 60 + float(start[0]) * 3600))
                    end = timestamps[1].strip().split(":")
                    end = float("{:.2f}".format(float(end[2]) + float(end[1]) * 60 + float(end[0]) * 3600))
                elif found_first:
                    line = line.strip()
                    if not line:
                        continue

                    if is_auto_generated and "<c>" in line and "</c>" in line:
                        show_block = True
                        words = line.split("<c>")
                        line_text = ""
                        first_word = True
                        for word in words:
                            if first_word:
                                line_text += word.split(" ")[0] + " "
                                first_word = False
                            else:
                                line_text += word.split("</c>")[0]

                        text = line_text
                    elif not is_auto_generated:
                        show_block = True

                        line_text = line.strip()

                        if not text:
                            text = line_text
                        else:
                            text += " " + line_text
                    elif is_auto_generated:
                        continue



            if text:
                transcription_token.append(AudioToken(
                    timeStart=float(start),
                    timeEnd=float(end),
                    text=text,
                    begin=current_char,
                    end=current_char + len(text)
                ))

    mimetype = magic.from_file(video, mime=True)
    print(mimetype)

    for file in os.listdir():
        if file.endswith(".mp4"):
            os.remove(file)
        if file.endswith(".json"):
            os.remove(file)
        if file.endswith(".vtt"):
            os.remove(file)

    return DUUIResponse(
        encoded_video=encoded,
        mimetype_video=mimetype,
        youtube_audio_token=transcription_token
    )


def download_youtube(link: str, download_trans: bool, transcription_lang: str, cookies: str):

    if download_trans:
        if transcription_lang == "":
            ydl_opts = {
                "writesubtitles": True,
                "writeautomaticsub": True,
                "noplaylist": True,
                "quiet": False,
                "no_warnings": True,
                "ratelimit": 500000,
                "user_agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                "retries": 10,
                "fragment_retries": 10,
                #"subtitleslangs": ["de"],
                "format_sort": ['res:1080', 'ext:mp4:m4a']
            }
        else:
            ydl_opts = {
                "writesubtitles": True,
                "writeautomaticsub": True,
                "noplaylist": True,
                "quiet": False,
                "no_warnings": True,
                "ratelimit": 500000,
                "retries": 10,
                "fragment_retries": 10,
                "user_agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                "subtitleslangs": [transcription_lang],
                "format_sort": ['res:1080', 'ext:mp4:m4a']
            }

    else:
        ydl_opts = {
            'format_sort': ['res:1080', 'ext:mp4:m4a'],
            "noplaylist": True,
            "quiet": False,
            "no_warnings": True,
            "ratelimit": 500000,
            "retries": 10,
            "user_agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            "fragment_retries": 10,
        }

    if cookies is not None and len(cookies) > 0:
        with open("cookies.txt", "w", encoding="utf-8") as file:
            file.write(cookies)

        if os.path.exists("cookies.txt"):
            print("Cookies found")
            ydl_opts["cookiefile"] = "cookies.txt"

    print(ydl_opts)
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download(link)

    transcription = None
    video = None

    for file in os.listdir():
        if file.endswith(".vtt"):
            transcription = file
        if file.endswith(".mp4"):
            video = file

    return (video, transcription)

#if __name__ == "__main__":
#    uvicorn.run("duui_yt_dlp:app", host="0.0.0.0", port=9713, workers=1)
