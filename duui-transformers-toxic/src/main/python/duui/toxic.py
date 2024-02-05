from typing import List, Dict
from pydantic import BaseModel

from .uima import UimaSentence


class ToxicSentence(BaseModel):
    sentence: UimaSentence
    toxics:List[Dict]


class ToxicSelection(BaseModel):
    selection: str
    sentences: List[ToxicSentence]
