from typing import List, Dict
from pydantic import BaseModel

from .uima import UimaSentence


class TopicSentence(BaseModel):
    sentence: UimaSentence

    topics:List[Dict]


class TopicSelection(BaseModel):
    selection: str
    sentences: List[TopicSentence]
