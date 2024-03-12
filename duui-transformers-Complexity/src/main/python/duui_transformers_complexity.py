from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Union
import logging
from time import time
from fastapi import FastAPI, Response
from cassis import load_typesystem
import torch
from threading import Lock
from functools import lru_cache
from BERT_converter import BertSentence, BertConverter, BertSentenceConverter
import numpy as np
from Complexity import distance, compute_wasserstein_distance, compute_distance_correlation, compute_mahalanobis_distance, compute_bhattacharyya_distance, compute_jensenshannon_distance
# from sp_correction import SentenceBestPrediction
sources = {
    "intfloat/multilingual-e5-base": "https://huggingface.co/intfloat/multilingual-e5-base",
    "google-bert/bert-base-multilingual-cased": "https://huggingface.co/google-bert/bert-base-multilingual-cased",
    "FacebookAI/xlm-roberta-large": "https://huggingface.co/FacebookAI/xlm-roberta-large",
    "facebook/xlm-v-base": "https://huggingface.co/facebook/xlm-v-base",
    "cardiffnlp/twitter-xlm-roberta-base": "https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base",
    "setu4993/LEALLA-small": "https://huggingface.co/setu4993/LEALLA-small",
    "sentence-transformers/LaBSE": "https://huggingface.co/sentence-transformers/LaBSE",
    "Twitter/twhin-bert-large": "https://huggingface.co/Twitter/twhin-bert-large",
    "paraphrase-multilingual-MiniLM-L12-v2": "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "distiluse-base-multilingual-cased-v2": "https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2"
}
languages = {
    "intfloat/multilingual-e5-base": "Multi",
    "google-bert/bert-base-multilingual-cased": "Multi",
    "FacebookAI/xlm-roberta-large": "Multi",
    "facebook/xlm-v-base": "Multi",
    "cardiffnlp/twitter-xlm-roberta-base": "Multi",
    "setu4993/LEALLA-small": "Multi",
    "sentence-transformers/LaBSE": "Multi",
    "Twitter/twhin-bert-large": "Multi",
    "paraphrase-multilingual-MiniLM-L12-v2": "Multi",
    "distiluse-base-multilingual-cased-v2": "Multi"
}
versions = {
    "intfloat/multilingual-e5-base": "d13f1b27baf31030b7fd040960d60d909913633f",
    "google-bert/bert-base-multilingual-cased": "3f076fdb1ab68d5b2880cb87a0886f315b8146f8",
    "FacebookAI/xlm-roberta-large": "c23d21b0620b635a76227c604d44e43a9f0ee389",
    "facebook/xlm-v-base": "68c75dd7733d2640b3a98114e3e94196dc543fe1",
    "cardiffnlp/twitter-xlm-roberta-base": "4c365f1490cb329b52150ad72f922ea467b5f4e6",
    "setu4993/LEALLA-small": "8fadf81fe3979f373ba9922ab616468a4184b266",
    "sentence-transformers/LaBSE": "5513ed8dd44a9878c7d4fe8646d4dd9df2836b7b",
    "Twitter/twhin-bert-large": "2786782c0f659550e3492093e4aab963d495243",
    "paraphrase-multilingual-MiniLM-L12-v2": "43dcf585e1eb6d4ece18c2e0c29474d9c5146b70",
    "distiluse-base-multilingual-cased-v2": "501a2afbd9deb9f028b175cc6060f38bb5055ce4"
}
# Settings
# These are automatically loaded from env variables
from starlette.responses import PlainTextResponse
model_lock = Lock()

class Settings(BaseSettings):
    # Name of this annotator
    complexity_annotator_name: str
    # Version of this annotator
    complexity_annotator_version: str
    # Log level
    complexity_log_level: str
    # # model_name
    # complexity_model_name: str
    # Name of this annotator
    complexity_model_version: str
    #cach_size
    complexity_model_cache_size: int


# Load settings from env vars
settings = Settings()
lru_cache_with_size = lru_cache(maxsize=settings.complexity_model_cache_size)
logging.basicConfig(level=settings.complexity_log_level)
logger = logging.getLogger(__name__)

device = 0 if torch.cuda.is_available() else "cpu"
logger.info(f'USING {device}')
# Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = 'TypeSystemComplexity.xml'
logger.debug("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    logger.debug("Base typesystem:")
    logger.debug(typesystem.to_xml())

# Load the Lua communication script
lua_communication_script_filename = "duui_complexity.lua"
logger.debug("Loading Lua communication script from \"%s\"", lua_communication_script_filename)


# Request sent by DUUI
# Note, this is transformed by the Lua script
class TextImagerRequest(BaseModel):
    #
    lang: str
    #
    model_name: str
    #
    model_art: str
    #
    complexity_compute: str
    #
    embeddings_keep: str
    #
    sentences_i: List[Dict]
    #
    sentences_j: List[Dict]


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
    meta: AnnotationMeta
    # Modification meta, one per document
    modification_meta: DocumentModification
    begin_emb: List[int]
    end_emb: List[int]
    embeddings: List[List]
    len_emb: int
    begin_i: List[int]
    end_i: List[int]
    begin_j: List[int]
    end_j: List[int]
    complexity: List[float]
    art: List[str]
    model_name: str
    model_version: str
    model_source: str
    model_lang: str



app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title=settings.complexity_annotator_name,
    description="Factuality annotator",
    version=settings.complexity_annotator_version,
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

@lru_cache_with_size
def load_model(model_name, model_art):
    match model_art:
        case "Sentence":
            model_i = BertSentenceConverter(model_name, device)
        case "BertSentence":
            model_i = BertSentence(model_name, device)
        case _:
            model_i = BertConverter(model_name, device)
    return model_i


def fix_unicode_problems(text):
    # fix emoji in python string and prevent json error on response
    # File "/usr/local/lib/python3.8/site-packages/starlette/responses.py", line 190, in render
    # UnicodeEncodeError: 'utf-8' codec can't encode characters in position xx-yy: surrogates not allowed
    clean_text = text.encode('utf-16', 'surrogatepass').decode('utf-16', 'surrogateescape')
    return clean_text


def compute_distance(art_i, sentences_i, sentences_j, embeddings):
    begins_i = []
    ends_i = []
    begins_j = []
    ends_j = []
    all_distances = []
    art = []
    pos = []
    for c, sen_i in enumerate(sentences_i):
        begin_i = sen_i["begin"]
        end_i = sen_i["end"]
        begin_j = sentences_j[c]["begin"]
        end_j = sentences_j[c]["end"]
        embedding_i = embeddings[f"{begin_i}_{end_i}"]
        embedding_j = embeddings[f"{begin_j}_{end_j}"]
        match art_i:
            case "euclidean":
                dis_i = distance.euclidean(embedding_i, embedding_j)
            case "cosine":
                dis_i = 1-distance.cosine(embedding_i, embedding_j)
            case "wasserstein":
                dis_i = compute_wasserstein_distance(embedding_i, embedding_j)
            case "distance":
                dis_i = compute_distance_correlation(embedding_i, embedding_j)
            case "jensenshannon":
                dis_i = compute_jensenshannon_distance(embedding_i, embedding_j)
            case "bhattacharyya":
                dis_i = compute_bhattacharyya_distance(embedding_i, embedding_j)
            case "mahalanobis":
                dis_i = compute_mahalanobis_distance(embedding_i, embedding_j)
            case _:
                art_i = "euclidean"
                dis_i = 1-distance.cosine(embedding_i, embedding_j)
        begins_i.append(begin_i)
        ends_i.append(end_i)
        begins_j.append(begin_j)
        ends_j.append(end_j)
        all_distances.append(dis_i)
        art.append(art_i)
        pos.append(c)
    return begins_i, ends_i, begins_j, ends_j, all_distances, art, pos

def process_selection(model_name, model_art, complexities, sentences_i, sentences_j):
    embeddings_dict = {}
    distances = {
        "begin_i": [],
        "end_i": [],
        "begin_j": [],
        "end_j": [],
        "key": [],
        "art": [],
        "complexity": []
    }
    all_sentences = {}
    embeddings = {
        "begin": [],
        "end": [],
        "embedding": [],
    }
    len_emb = 0
    for sen_i in sentences_i:
        begin = sen_i["begin"]
        end = sen_i["end"]
        text = sen_i["text"]
        if f"{begin}_{end}" not in all_sentences:
            all_sentences[f"{begin}_{end}"] = {
                "begin": begin,
                "end": end,
                "text": text
            }
    for sen_i in sentences_j:
        begin = sen_i["begin"]
        end = sen_i["end"]
        text = sen_i["text"]
        if f"{begin}_{end}" not in all_sentences:
            all_sentences[f"{begin}_{end}"] = {
                "begin": begin,
                "end": end,
                "text": text
            }
    all_text = []
    all_keys = list(all_sentences.keys())
    chunk_size = 50
    split_keys = [all_keys[i:i + chunk_size] for i in range(0, len(all_keys), chunk_size)]
    for c, chunk_i in enumerate(split_keys):
        all_text.append([])
        for key_i in chunk_i:
            all_text[c].append(all_sentences[key_i]["text"])
    with model_lock:
        model_i = load_model(model_name, model_art)
        for c, texts in enumerate(all_text):
            embeddings_i = model_i.encode_to_vec(texts)
            keys_i = split_keys[c]
            for index_i, emb_i in enumerate(embeddings_i):
                key_i = keys_i[index_i]
                embeddings["begin"].append(all_sentences[key_i]["begin"])
                embeddings["end"].append(all_sentences[key_i]["end"])
                len_emb = len(emb_i)
                embeddings["embedding"].append(emb_i)
                embeddings_dict[f"{all_sentences[key_i]['begin']}_{all_sentences[key_i]['end']}"] = emb_i
        distances_art = complexities.split(",")
        for distances_i in distances_art:
            distances_i = distances_i.strip().lower()
            begins_i, ends_i, begins_j, ends_j, all_distances, all_art, keys = compute_distance(distances_i, sentences_i, sentences_j, embeddings_dict)
            distances["begin_i"] = distances["begin_i"]+begins_i
            distances["end_i"] = distances["end_i"]+ends_i
            distances["begin_j"] = distances["begin_j"]+begins_j
            distances["end_j"] = distances["end_j"]+ends_j
            distances["complexity"] = distances["complexity"]+all_distances
            distances["art"] = distances["art"]+all_art
            distances["key"] = distances["key"]+keys
    return embeddings, distances, len_emb

# Process request from DUUI
@app.post("/v1/process")
def post_process(request: TextImagerRequest):
    # Return data
    meta = None
    begin = []
    end = []
    len_results = []
    results = []
    factors = []
    # Save modification start time for later
    modification_timestamp_seconds = int(time())
    try:
        model_source = sources[request.model_name]
        model_lang = languages[request.model_name]
        model_version = versions[request.model_name]
        # set meta Informations
        meta = AnnotationMeta(
            name=settings.complexity_annotator_name,
            version=settings.complexity_annotator_version,
            modelName=request.model_name,
            modelVersion=model_version,
        )
        # Add modification info
        modification_meta_comment = f"{settings.complexity_annotator_name} ({settings.complexity_annotator_version}))"
        modification_meta = DocumentModification(
            user=settings.complexity_annotator_name,
            timestamp=modification_timestamp_seconds,
            comment=modification_meta_comment
        )
        mv = ""
        embeddings, distances, len_emb = process_selection(request.model_name, request.model_art, request.complexity_compute, request.sentences_i, request.sentences_j)
    except Exception as ex:
        logger.exception(ex)
    return TextImagerResponse(meta=meta, modification_meta=modification_meta, begin_emb=embeddings["begin"], end_emb=embeddings["end"], embeddings=embeddings["embedding"], len_emb=len_emb, begin_i=distances["begin_i"], end_i = distances["end_i"], begin_j=distances["begin_j"], end_j=distances["end_j"], complexity=distances["complexity"], art=distances["art"], model_name=request.model_name, model_version=model_version, model_source=model_source, model_lang=model_lang)



