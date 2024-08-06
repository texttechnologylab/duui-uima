from typing import List, Dict
from pydantic import BaseModel

from .uima import UimaSentence


class SentimentSentence(BaseModel):
    sentence: UimaSentence

    # generated sentiment score based on top predicted label (-1, 0, 1)
    sentiment: float
    # score of top label
    score: float

    # polarity based on softmax results: pos-neg
    # see https://github.com/text-analytics-20/news-sentiment-development
    polarity: float
    # average scores with similarities mapped to pos, neu and neg
    pos: float
    neu: float
    neg: float

    # softmax scores for each label
    details: Dict[str, float]


class SentimentSelection(BaseModel):
    selection: str
    sentences: List[SentimentSentence]
