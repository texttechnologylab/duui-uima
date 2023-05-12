from typing import List
from pydantic import BaseModel


UIMA_TYPE_SENTENCE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"


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


class UimaSentenceSelection(BaseModel):
    selection: str
    sentences: List[UimaSentence]
