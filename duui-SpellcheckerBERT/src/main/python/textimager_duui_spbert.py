from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Union
import logging
from time import time
from fastapi import FastAPI, Response
from cassis import load_typesystem
# from sp_correction import SentenceBestPrediction
from symspellpy import SymSpell
from spellchecker import spellchecker

# Settings
# These are automatically loaded from env variables
from starlette.responses import PlainTextResponse


class Settings(BaseSettings):
    # Name of this annotator
    textimager_spbert_annotator_name: str
    # Version of this annotator
    textimager_spbert_annotator_version: str
    # Log level
    textimager_spbert_log_level: str
    # model_name
    textimager_spbert_model_name: str
    # Name of this annotator
    textimager_spbert_model_version: str


# Load settings from env vars
settings = Settings()

logging.basicConfig(level=settings.textimager_spbert_log_level)
logger = logging.getLogger(__name__)

# Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = 'TypeSystemSPBERT.xml'
logger.debug("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    logger.debug("Base typesystem:")
    logger.debug(typesystem.to_xml())

# Load the Lua communication script
lua_communication_script_filename = "textimager_duui_spbert.lua"
logger.debug("Loading Lua communication script from \"%s\"", lua_communication_script_filename)


# Request sent by DUUI
# Note, this is transformed by the Lua script
class TextImagerRequest(BaseModel):
    # The text to process
    text: str
    # The texts language
    lang: str
    #
    sen: Optional[list] = None
    #
    tokens: Optional[list] = None
    # Optional map/dict of parameters
    # TODO how with lua?
    parameters: Optional[dict] = None


# UIMA type: mark modification of the document
class DocumentModification(BaseModel):
    user: str
    timestamp: int
    comment: str


# UIMA type: adds metadata to each annotation
class AnnotationMeta(BaseModel):
    name: str
    version: str
    modelName: str
    modelVersion: str


# Response sent by DUUI
# Note, this is transformed by the Lua script
class TextImagerResponse(BaseModel):
    # Symspelloutput
    # List of Sentence with every token
    # Every token is a dictionary with following Infos:
    # Symspelloutput right if the token is correct, wrong if the token is incorrect, skipped if the token was skipped, unkownn if token can corrected with Symspell
    # If token is unkown it will be predicted with BERT Three output pos:
    # 1. Best Prediction with BERT MASKED
    # 2. Best Cos-sim with Sentence-Bert and with perdicted words of BERT MASK
    # 3. Option 1 and 2 together
    tokens: List[List[Dict[str, Union[float, str, int, Dict[str, Dict[str, Union[str, float]]]]]]]
    meta: AnnotationMeta
    # Modification meta, one per document
    modification_meta: DocumentModification


app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title=settings.textimager_spbert_annotator_name,
    description="spaCy implementation for TTLab TextImager DUUI",
    version=settings.textimager_spbert_model_version,
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "TTLab Team",
        "url": "https://texttechnologylab.org",
        "email": "baumartz@stud.uni-frankfurt.de",
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
def post_process(request: TextImagerRequest):
    # Return data
    meta = None
    symspell_out = []
    # Save modification start time for later
    modification_timestamp_seconds = int(time())
    try:
        # set meta Informations
        meta = AnnotationMeta(
            name=settings.textimager_spbert_annotator_name,
            version=settings.textimager_spbert_annotator_version,
            modelName=settings.textimager_spbert_model_name,
            modelVersion=settings.textimager_spbert_model_version,
        )
        # Add modification info
        modification_meta_comment = f"{settings.textimager_spbert_annotator_name} ({settings.textimager_spbert_annotator_version}))"
        modification_meta = DocumentModification(
            user=settings.textimager_spbert_annotator_name,
            timestamp=modification_timestamp_seconds,
            comment=modification_meta_comment
        )
        # Get CAS from XMI string
        logger.debug("Received:")
        logger.debug(request)

        # Params, set here to empty dict to allow easier access later
        if request.parameters is None:
            request.parameters = {}

        text = request.text
        lang = request.lang
        sen = request.sen
        token = request.tokens
        document_token_sentences = []
        begin_token_sentences = []
        end_token_sentences = []
        for c, sen_i in enumerate(token):
            document_token_sentences.append([])
            begin_token_sentences.append([])
            end_token_sentences.append([])
            for token_i in sen_i:
                document_token_sentences[c].append(token_i["text"])
                begin_token_sentences[c].append(token_i["begin"])
                end_token_sentences[c].append(token_i["end"])
        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = "de-100k.txt"
        sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        # sen_pred = SentenceBestPrediction("", "bert-base-uncased", "all-mpnet-base-v2", 0)
        for c, sen_i in enumerate(document_token_sentences):
            spell_out = spellchecker(sen_i, begin_token_sentences[c], end_token_sentences[c], sym_spell,
                                     lower_case=True)
            symspell_out.append(spell_out)
            # sen_org = " ".join(sen_i)
            # sen_test, sen_org = sen_pred.mask_sentence(spell_out)
            # sen_pred.set_sen_org(sen_org)
            # pred_sentences = sen_pred.get_Mask_prediction(sen_test)
            # cos_sim_sentences = sen_pred.get_sentence_sim(pred_sentences)
            # symspell_out.append(sen_pred.get_best_word(cos_sim_sentences, spell_out))
    except Exception as ex:
        logger.exception(ex)
    return TextImagerResponse(tokens=symspell_out, meta=meta, modification_meta=modification_meta)
