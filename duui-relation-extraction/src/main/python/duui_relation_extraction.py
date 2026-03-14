"""
DUUI Endpunkt
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from relation_extraction import RebelRelationExtraction, KnowGLRelationExtraction, GLiNERRelationExtraction
from functools import lru_cache
from threading import Lock
import torch
from time import time
import os

model_lock = Lock()
device = os.getenv("DEVICE", "cpu")

MODELS = {
    "Babelscape/rebel-large": {
        "path": "Babelscape/rebel-large",
        "class": RebelRelationExtraction
    },
    "ibm-research/knowgl-large": {
        "path": "ibm-research/knowgl-large",
        "class": KnowGLRelationExtraction
    }
}

model_name = os.getenv("MODEL_NAME")

MODEL_PATH = MODELS[model_name]["path"]

model_class = MODELS[model_name]["class"]

class UimaSentence(BaseModel):
    text: str
    begin: int
    end: int

class UimaSentenceSelection(BaseModel):
    selection: str
    sentences: List[UimaSentence]

class DUUIRequest(BaseModel):
    doc_len: int
    lang: str
    selections: List[UimaSentenceSelection]

class DocumentModification(BaseModel):
    user: str
    timestamp: int
    comment: str

class AnnotationMeta(BaseModel):
    name: str
    version: str
    modelName: str
    modelVersion: str

class DUUIResponse(BaseModel):
    meta: AnnotationMeta
    modification_meta: DocumentModification
    begins: List[int]
    ends: List[int]
    relations: List[List[dict]]
    model_name: str
    model_version: str
    model_source: str
    model_lang: str

duui_relation_extraction = FastAPI(title="DUUI Relation Extraction", version="1.0")

@lru_cache()
def load_model():
    return model_class(MODEL_PATH, device)

def process_selection(selection: UimaSentenceSelection):
    begins, ends, relations = [], [], []

    texts = [f"sentence: {s.text}" for s in selection.sentences]
    with model_lock:
        model = load_model()
        results = model.extract_relations(texts)

    for idx, sentence in enumerate(selection.sentences):
        begins.append(sentence.begin)
        ends.append(sentence.end)
        relations.append(results[idx])

    return begins, ends, relations

@duui_relation_extraction.post("/v1/process")
def post_process(request: DUUIRequest):
    modification_timestamp_seconds = int(time())
    begins, ends, relations = [], [], []

    meta = AnnotationMeta(
        name="REBEL Relation Extractor",
        version="1.0",
        modelName="Babelscape/rebel-large",
        modelVersion="1.0"
    )

    modification_meta = DocumentModification(
        user="REBEL RE",
        timestamp=modification_timestamp_seconds,
        comment="Relation Extraction"
    )

    for selection in request.selections:
        b, e, r = process_selection(selection)
        begins.extend(b)
        ends.extend(e)
        relations.extend(r)

    return DUUIResponse(
        meta=meta,
        modification_meta=modification_meta,
        begins=begins,
        ends=ends,
        relations=relations,
        model_name="Babelscape/rebel-large",
        model_version="1.0",
        model_source="HuggingFace",
        model_lang="multilingual"
    )