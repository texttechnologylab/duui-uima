import logging
from datetime import datetime
from platform import python_version
from sys import version as sys_version
from threading import Lock
from time import time

from cassis import load_typesystem
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse
from vaderSentiment import vaderSentiment as vaderSentimentEn
from vaderSentiment_fr import vaderSentiment as vaderSentimentFr
from gervader import vaderSentimentGER

from .duui.reqres import TextImagerResponse, TextImagerRequest
from .duui.sentiment import SentimentSentence, SentimentSelection
from .duui.service import Settings, TextImagerDocumentation, TextImagerCapability
from .duui.uima import *

# TODO get from source?
VADER_EN_VERSION = "3.3.2"
VADER_FR_VERSION = "1.3.4"
VADER_DE_VERSION = "4d6e5a4ba9a294a38b1257999f627b5c0fe15809"

settings = Settings()
model_lock = Lock()
analyzer_cache = {}
supported_languages = sorted(list(["en", "fr", "de"]))

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)
logger.info("TTLab TextImager DUUI Transformers Vader")
logger.info("Name: %s", settings.annotator_name)
logger.info("Version: %s", settings.annotator_version)

typesystem_filename = 'src/main/resources/TypeSystemSentiment.xml'
logger.info("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    logger.debug("Base typesystem:")
    logger.debug(typesystem.to_xml())

lua_communication_script_filename = "src/main/lua/textimager_duui_vader_sentiment.lua"
logger.info("Loading Lua communication script from \"%s\"", lua_communication_script_filename)
with open(lua_communication_script_filename, 'rb') as f:
    lua_communication_script = f.read().decode("utf-8")
    logger.debug("Lua communication script:")
    logger.debug(lua_communication_script)

app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title=settings.annotator_name,
    description="Vader-based sentiment analysis for TTLab TextImager DUUI",
    version=settings.annotator_version,
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "TTLab Team",
        "url": "https://texttechnologylab.org",
        "email": "baumartz@em.uni-frankfurt.de",
    },
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)


@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return lua_communication_script


@app.get("/v1/documentation")
def get_documentation() -> TextImagerDocumentation:
    capabilities = TextImagerCapability(
        supported_languages=supported_languages,
        reproducible=True
    )

    documentation = TextImagerDocumentation(
        annotator_name=settings.annotator_name,
        version=settings.annotator_version,
        implementation_lang="Python",
        meta={
            "python_version": python_version(),
            "python_version_full": sys_version,
            "vader_en_version": VADER_EN_VERSION,
            "vader_fr_version": VADER_FR_VERSION,
            "vader_der_version": VADER_DE_VERSION,
        },
        docker_container_id="[TODO]",
        parameters={
            "model_name": ["vader-en", "vader-fr", "vader-de"]
        },
        capability=capabilities,
        implementation_specific=None,
    )

    return documentation


@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    xml = typesystem.to_xml()
    xml_content = xml.encode("utf-8")

    return Response(
        content=xml_content,
        media_type="application/xml"
    )


def load_analyzer(lang: str):
    with model_lock:
        if lang in analyzer_cache:
            return analyzer_cache[lang]

        elif lang == "fr":
            analyzer = vaderSentimentFr.SentimentIntensityAnalyzer()
            model_name = "vader-fr"
            model_version = VADER_FR_VERSION
        elif lang == "de":
            analyzer = vaderSentimentGER.SentimentIntensityAnalyzer()
            model_name = "vader-de"
            model_version = VADER_DE_VERSION
        else:
            # use en as default
            # TODO error instead?
            analyzer = vaderSentimentEn.SentimentIntensityAnalyzer()
            model_name = "vader-en"
            model_version = VADER_EN_VERSION

        analyzer_cache[lang] = analyzer, model_name, model_version
        return analyzer, model_name, model_version


def fix_unicode_problems(text):
    # fix emoji in python string and prevent json error on response
    # File "/usr/local/lib/python3.8/site-packages/starlette/responses.py", line 190, in render
    # UnicodeEncodeError: 'utf-8' codec can't encode characters in position xx-yy: surrogates not allowed
    #clean_text = text.encode('utf-16', 'surrogatepass').decode('utf-16', 'surrogateescape')

    # try to fix by single char to prevent length change
    clean_text = ''
    for char_ in text:
        try:
            char = char_.encode('utf-8').decode('utf-8')
        except UnicodeEncodeError:
            char = u"\uFFFD"
        clean_text += char
    assert len(clean_text) == len(text)

    return clean_text


@app.post("/v1/process")
def process(request: TextImagerRequest) -> TextImagerResponse:
    processed_selections = []

    dt = datetime.now()
    modification_timestamp_seconds = int(time())

    logger.debug("Received:")
    logger.debug(request)

    analyzer, model_name, model_version = load_analyzer(request.lang)

    for selection in request.selections:
        processed_sentences = []

        for sentence in selection.sentences:
            try:
                vs = analyzer.polarity_scores(
                    fix_unicode_problems(sentence.text)
                )

                processed_sentences.append(SentimentSentence(
                    # remove text content due to unicode problems
                    sentence=UimaSentence(
                        text="",
                        begin=sentence.begin,
                        end=sentence.end
                    ),
                    compound=vs["compound"],
                    pos=vs["pos"],
                    neu=vs["neu"],
                    neg=vs["neg"],
                ))
            except Exception as ex:
                logger.error("Error while processing sentence \"%s\": %s", sentence.text, ex)

        # compute avg for this selection, if >1
        if len(processed_sentences) > 1:
            begin = 0
            end = request.doc_len

            compounds = 0
            poss = 0
            neus = 0
            negs = 0
            for sentence in processed_sentences:
                compounds += sentence.compound
                poss += sentence.pos
                neus += sentence.neu
                negs += sentence.neg

            compound = compounds / len(processed_sentences)
            pos = poss / len(processed_sentences)
            neu = neus / len(processed_sentences)
            neg = negs / len(processed_sentences)

            processed_sentences.append(SentimentSentence(
                sentence=UimaSentence(
                    text="",
                    begin=begin,
                    end=end),
                compound=compound,
                pos=pos,
                neu=neu,
                neg=neg,
            ))

        processed_selections.append(SentimentSelection(
            selection=selection.selection,
            sentences=processed_sentences
        ))

    meta = UimaAnnotationMeta(
        name=settings.annotator_name,
        version=settings.annotator_version,
        modelName=model_name,
        modelVersion=model_version,
    )

    modification_meta_comment = f"{settings.annotator_name} ({settings.annotator_version})"
    modification_meta = UimaDocumentModification(
        user="TextImager",
        timestamp=modification_timestamp_seconds,
        comment=modification_meta_comment
    )

    logger.debug(processed_selections)

    dte = datetime.now()
    print(dte, 'Finished processing', flush=True)
    print('Time elapsed', f'{dte-dt}', flush=True)

    return TextImagerResponse(
        selections=processed_selections,
        meta=meta,
        modification_meta=modification_meta
    )
