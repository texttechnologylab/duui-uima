from typing import List, Optional, Dict
from pydantic import BaseModel

UIMA_TYPE_SENTENCE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
UIMA_TYPE_TOKEN = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"


class UimaAnnotationMeta(BaseModel):
    name: str
    version: str
    modelName: str
    modelVersion: str


class UimaDocumentModification(BaseModel):
    user: str
    timestamp: int
    comment: str


class UimaSentence(BaseModel):
    text: str
    begin: int
    end: int


class Token(BaseModel):
    begin: int
    end: int
    ind: int
    write_token: Optional[bool]
    lemma: Optional[str]
    write_lemma: Optional[bool]
    pos: Optional[str]
    pos_coarse: Optional[str]
    write_pos: Optional[bool]
    morph: Optional[str]
    morph_details: Optional[dict]
    write_morph: Optional[bool]
    parent_ind: Optional[int]
    write_dep: Optional[bool]


