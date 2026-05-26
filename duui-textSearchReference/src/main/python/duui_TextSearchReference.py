from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Union
import logging
from fastapi import FastAPI, Response
from cassis import load_typesystem
from functools import lru_cache
from threading import Lock
import time
import wikipedia
from extractReferenceText import search_wikidata, search_google, wikipedia_search, wikipedia_text_extract, google_search_words, get_results
import json
# from sp_correction import SentenceBestPrediction

# Settings
# These are automatically loaded from env variables
from starlette.responses import PlainTextResponse

model_lock = Lock()




class Settings(BaseSettings):
    # Name of this annotator
    annotator_name: str
    # Version of this annotator
    annotator_version: str
    # Log level
    log_level: str

# Load settings from env vars
settings = Settings()
# lru_cache_with_size = lru_cache(maxsize=settings.model_cache_size)
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# device = 0 if torch.cuda.is_available() else "cpu"
# logger.info(f'USING {device}')
# Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = 'TypeSystemTextSearchReference.xml'
logger.debug("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    logger.debug("Base typesystem:")
    # logger.debug(typesystem.to_xml())

# Load the Lua communication script
lua_communication_script_filename = "duui_TextSearchReference.lua"
logger.debug("Loading Lua communication script from \"%s\"", lua_communication_script_filename)


# Request sent by DUUI
# Note, this is transformed by the Lua script
class DUUIRequest(BaseModel):
    text : str
    # The texts language
    lang: str
    #
    len: int
    #
    searches: Optional[List] = None
    #
    search_language: str = None
    #
    method: str = None
    #
    search: str
    #


# UIMA type: mark modification of the document
class DocumentModification(BaseModel):
    user: str
    timestamp: int
    comment: str


# UIMA type: adds metadata to each annotation
class AnnotationMeta(BaseModel):
    name: str
    version: str


def fix_unicode_problems(text):
    # fix emoji in python string and prevent json error on response
    # File "/usr/local/lib/python3.8/site-packages/starlette/responses.py", line 190, in render
    # UnicodeEncodeError: 'utf-8' codec can't encode characters in position xx-yy: surrogates not allowed
    clean_text = text.encode('utf-16', 'surrogatepass').decode('utf-16', 'surrogateescape')
    return clean_text


# Response sent by DUUI
# Note, this is transformed by the Lua script
class DUUIResponse(BaseModel):
    meta: AnnotationMeta
    # Modification meta, one per document
    modification_meta: DocumentModification
    # begin_prompts
    references_begin: List[int]
    references_end: List[int]
    references_ids: List[int]
    urls: List[str]
    groups: List[str]
    methods: List[str]
    priorities: List[int]
    summaries: List[str]
    infos: List[str]
    success: List[bool]
    texts: List[str]
    datetimes: List[str]

app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title=settings.annotator_name,
    description="Web-Search for references in the search text",
    # version=settings.model_version,
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "TTLab Team",
        "url": "https://texttechnologylab.org",
        "email": "bagci@em.uni-frankfurt.de",
    },
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)

with open(lua_communication_script_filename, 'rb') as f:
    lua_communication_script = f.read().decode("utf-8")
logger.debug("Lua communication script:")
logger.debug(lua_communication_script_filename)


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
    return lua_communication_script


# Return documentation info
@app.get("/v1/documentation")
def get_documentation():
    return "Test"


# Process request from DUUI
@app.post("/v1/process")
def post_process(request: DUUIRequest):
    # Return data
    # Save modification start time for later
    modification_timestamp_seconds = int(time.time())
    success = []
    texts = []
    datetimes = []
    urls = []
    groups = []
    methods = []
    priorities = []
    summaries = []
    infos = []
    references_begin = []
    references_end = []
    references_ids = []
    try:
        # set meta Informations
        meta = AnnotationMeta(
            name=settings.annotator_name,
            version=settings.annotator_version,
        )
        # Add modification info
        modification_meta_comment = f"{settings.annotator_name} ({settings.annotator_version}))"
        modification_meta = DocumentModification(
            user=settings.annotator_name,
            timestamp=modification_timestamp_seconds,
            comment=modification_meta_comment
        )
        search_language = request.search_language
        search = request.search.lower()
        method = request.method.lower()
        wikipedia.set_lang(search_language)
        for search_i in request.searches:
            id_search  = search_i["id"]
            text_search = search_i["text"]
            begin_search = search_i["begin"]
            end_search = search_i["end"]
            match (method):
                case "wikidata":
                    time_i = time.time()
                    datetime_i = time.strftime("%d-%m-%Y %H:%M:%S", time.localtime(time_i))
                    search_qids = search_wikidata(text_search, search_language)
                    if len(search_qids) > 0:
                        search_result = get_results(search_qids)
                        for qid in search_result:
                            counter = 0
                            for language in search_result[qid]:
                                data_i = search_result[qid][language]
                                description = data_i["Description"]
                                label_i = data_i["Label"]
                                alt_label = data_i["AltLabel"]
                                url_i = data_i["url"]
                                if len(description) > 0:
                                    text_i = " ".join(description)
                                    success.append(True)
                                    texts.append(text_i)
                                    datetimes.append(datetime_i)
                                    urls.append(url_i)
                                    groups.append(search)
                                    methods.append(method)
                                    priorities.append(counter)
                                    counter+=1
                                    summaries.append("")
                                    info_i={}
                                    info_i["search_text"] = text_search
                                    info_i["qid"] = qid
                                    info_i["class"] = search
                                    info_i["method"] = method
                                    info_i["search_language"] = search_language
                                    info_i["Description"] = description
                                    info_i["Label"] = label_i
                                    info_i["AltLabel"] = alt_label
                                    info_i["url"] = url_i
                                    info_json = json.dumps(info_i)
                                    infos.append(info_json)
                                    references_begin.append(begin_search)
                                    references_end.append(end_search)
                                    references_ids.append(id_search)
                case "google":
                    time_i = time.time()
                    datetime_i = time.strftime("%d-%m-%Y %H:%M:%S", time.localtime(time_i))
                    if search=="wikipedia":
                        # search after keywords
                        counter = 1
                        pre_search =  f"site:{search_language}.wikipedia.org"
                        search_keywords = google_search_words([text_search],pre_search, search_language)
                        for keyword in search_keywords:
                            search_result = wikipedia_text_extract(keyword)
                            if "search_text" in search_result:
                                texts.append(search_result["search_text"])
                                success.append(True)
                                info_i = search_result
                            else:
                                texts.append("")
                                success.append(False)
                                info_i = {}
                            info_i["search_text"] = text_search
                            info_i["keyword"] = keyword
                            info_i["class"] = search
                            info_i["method"] = method
                            info_i["search_language"] = search_language
                            info_dumps = json.dumps(info_i)
                            infos.append(info_dumps)
                            references_begin.append(begin_search)
                            references_end.append(end_search)
                            references_ids.append(id_search)
                            if "url" in search_result:
                                urls.append(search_result["url"])
                            else:
                                urls.append("")
                            groups.append(search)
                            methods.append(method)
                            priorities.append(counter)
                            counter += 1
                            if "summary" in search_result:
                                summaries.append(search_result["summary"])
                            else:
                                summaries.append("")
                            datetimes.append(datetime_i)
                case "wikipedia":
                    time_i = time.time()
                    datetime_i = time.strftime("%d-%m-%Y %H:%M:%S", time.localtime(time_i))
                    if search=="wikipedia":
                        # search after keywords
                        counter = 1
                        search_keywords = wikipedia_search([text_search])[0]
                        for keyword in search_keywords:
                            search_result = wikipedia_text_extract(keyword)
                            if "search_text" in search_result:
                                texts.append(search_result["search_text"])
                                success.append(True)
                                info_i = search_result
                            else:
                                texts.append("")
                                success.append(False)
                                info_i = {}
                            info_i["search_text"] = text_search
                            info_i["keyword"] = keyword
                            info_i["class"] = search
                            info_i["method"] = method
                            info_i["search_language"] = search_language
                            info_dumps = json.dumps(info_i)
                            infos.append(info_dumps)
                            references_begin.append(begin_search)
                            references_end.append(end_search)
                            references_ids.append(id_search)
                            if "url" in search_result:
                                urls.append(search_result["url"])
                            else:
                                urls.append("")
                            groups.append(search)
                            methods.append(method)
                            priorities.append(counter)
                            counter += 1
                            if "summary" in search_result:
                                summaries.append(search_result["summary"])
                            else:
                                summaries.append("")
                            datetimes.append(datetime_i)
                case _:
                    logger.error(f"Unknown method: {method}")
                    break
    except Exception as ex:
        logger.exception(ex)
    return DUUIResponse(meta=meta, modification_meta=modification_meta, references_begin=references_begin, references_end=references_end, references_ids=references_ids, urls=urls, groups=groups, methods=methods, priorities=priorities, summaries=summaries, infos=infos, success=success, texts=texts, datetimes=datetimes)
