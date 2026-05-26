import yaml
from bertopic import BERTopic
import numpy as np
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Union
import logging
from time import time
from fastapi import FastAPI, Response
from cassis import load_typesystem
import torch
from functools import lru_cache

from starlette.responses import PlainTextResponse
import pandas as pd
from collections import defaultdict as dd

class Config(BaseSettings):
    # Name of this annotator
    annotator_name: str
    # Version of this annotator
    annotator_version: str
    # Log level
    log_level: str
    # model_name
    model_name: str
    # Name of this annotator
    model_version: str
    #cach_size
    model_cache_size: int
    # url of the model
    model_source: str
    # language of the model
    model_lang: str


class UimaSentence(BaseModel):
    text: str
    begin: int
    end: int


class UimaSentenceSelection(BaseModel):
    selection: str
    sentences: List[UimaSentence]


# Request sent by DUUI
# Note, this is transformed by the Lua script
class DUUIRequest(BaseModel):
    doc_len: int
    lang: str
    selections: List[UimaSentenceSelection]

# Response sent by DUUI
# Note, this is transformed by the Lua script
class DUUIResponse(BaseModel):
    begin: List[int]
    end: List[int]
    results: List
    factors: List
    len_results: List[int]
    words: List
    words_prob: List
    model_name: str
    model_version: str
    model_lang: str
    model_source: str


def fix_unicode_problems(text):
    # fix emoji in python string and prevent json error on response
    # File "/usr/local/lib/python3.8/site-packages/starlette/responses.py", line 190, in render
    # UnicodeEncodeError: 'utf-8' codec can't encode characters in position xx-yy: surrogates not allowed
    clean_text = text.encode('utf-16', 'surrogatepass').decode('utf-16', 'surrogateescape')
    return clean_text

def get_word_dist(topic_model, texts, topic_token_distr):
    word_dist = dd(lambda: dd())
    analyzer = topic_model.vectorizer_model.build_tokenizer()
    for i, text in enumerate(texts):
        tokens = analyzer(text)

        # Prepare dataframe with results
        df = pd.DataFrame(topic_token_distr[i] / topic_token_distr[i].sum()).T
        #df = pd.DataFrame(topic_token_distr).T

        df.columns = [f"{token}_{i}" for i, token in enumerate(tokens)]
        df.columns = [f"{token}{' ' * i}" for i, token in enumerate(tokens)]
        df.index = list(topic_model.topic_labels_.values())[topic_model._outliers :]
        df = df.loc[(df.sum(axis=1) != 0), :]

        # Filter each row to only include words with probability > 0
        filtered_topics = {topic: row[row > 0] for topic, row in df.iterrows()}

        word_list = []
        prob_list = []
        for topic, words in filtered_topics.items():
            word_list.append([k.strip() for k in words.to_dict().keys()])
            prob_list.append(list(words.to_dict().values()))
        word_dist[i]['words'] = word_list
        word_dist[i]['prob'] = prob_list

    return word_dist

def process_selection(model_name, selection):
    begin = []
    end = []
    results_out = []
    factors = []
    len_results = []
    words = []
    words_prob = []
    for s in selection.sentences:
        s.text = fix_unicode_problems(s.text)

    texts = [
        s.text
        for s in selection.sentences
    ]
    logger.debug("Preprocessed texts:")
    logger.debug(texts)

    topic_model = BERTopic.load(model_name, embedding_model="BAAI/bge-small-en")
    topic_distr, topic_token_distr = topic_model.approximate_distribution(texts, calculate_tokens=True)
    topic_indices_per_doc = [list(np.where(row > 0)[0]) for row in topic_distr]
    word_dists = get_word_dist(topic_model, texts, topic_token_distr)

    for doc_idx, topic_idxs in enumerate(topic_indices_per_doc):
        t = []
        p = []
        w = word_dists[doc_idx]['words']
        wp = word_dists[doc_idx]['prob']
        for topic_idx in topic_idxs:
            t.append(topic_model.custom_labels_[topic_idx])
            p.append(topic_distr[doc_idx][topic_idx])

        sentence_i = selection.sentences[doc_idx]
        begin_i = sentence_i.begin
        end_i = sentence_i.end
        len_i = len(t)
        begin.append(begin_i)
        end.append(end_i)
        results_out.append(t)
        len_results.append(len_i)
        factors.append(p)
        words.append(w)
        words_prob.append(wp)


    output = {
        "begin": begin,
        "end": end,
        "len_results": len_results,
        "results": results_out,
        "factors": factors,
        "words": words,
        "words_prob": words_prob
    }

    return output



config_file = 'config.yaml'
with open(config_file, 'r') as stream:
    config = yaml.safe_load(stream)

config = Config(annotator_name=config['annotation']['name'],annotator_version=str(config['annotation']['version']),
                model_source=config['model']['source'], model_lang=config['model']['lang'], model_version=config['model']['version'],
                model_name=config['model']['name'],model_cache_size=int(config['model']['cache_size']),log_level=config['log_level'],)

lru_cache_with_size = lru_cache(maxsize=config.model_cache_size)
logging.basicConfig(level=config.log_level)
logger = logging.getLogger(__name__)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"
logger.info(f'USING {device}')
# Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = 'TypeSystemUnifiedTopic.xml'
logger.debug("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    logger.debug("Base typesystem:")
    logger.debug(typesystem.to_xml())

app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title=config.annotator_name,
    description="Factuality annotator",
    version=config.annotator_version,
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

# Load the Lua communication script
lua_communication_script_filename = "duui-transformers-berttopic.lua"
logger.debug("Loading Lua communication script from \"%s\"", lua_communication_script_filename)


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
def load_model(model_name):
    pass

@app.post("/v1/process")
def post_process(request: DUUIRequest):
    begin = []
    end = []
    len_results = []
    results = []
    factors = []
    words = []
    words_prob = []
    try:
        for selection in request.selections:
            processed_sentences = process_selection(config.model_name, selection)
            begin = begin + processed_sentences["begin"]
            end = end + processed_sentences["end"]
            len_results = len_results + processed_sentences["len_results"]
            results = results + processed_sentences["results"]
            factors = factors + processed_sentences["factors"]
            words = words + processed_sentences["words"]
            words_prob = words_prob + processed_sentences["words_prob"]
    except Exception as ex:
        logger.exception(ex)
    return DUUIResponse( begin=begin, end=end, results=results, words=words, words_prob=words_prob,
                         len_results=len_results, factors=factors, model_name=config.model_name,
                         model_version=config.model_version, model_source=config.model_source,
                         model_lang=config.model_lang)
