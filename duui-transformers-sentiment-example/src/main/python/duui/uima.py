from typing import List
from pydantic import BaseModel


UIMA_TYPE_SENTENCE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
UIMA_TYPE_SENTIMENT = "org.hucompute.textimager.uima.type.Sentiment"
UIMA_TYPE_SENTIMENT_CATEGORIZED = "org.hucompute.textimager.uima.type.CategorizedSentiment"


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
