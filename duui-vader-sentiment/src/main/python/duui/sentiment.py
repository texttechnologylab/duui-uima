from typing import List, Dict
from pydantic import BaseModel

from .uima import UimaSentence


class SentimentSentence(BaseModel):
    sentence: UimaSentence

    # Info from GerVADER:
    # The 'compound' score is computed by summing the valence scores of each word in the lexicon, adjusted
    # according to the rules, and then normalized to be between -1 (most extreme negative) and +1 (most extreme positive).
    # This is the most useful metric if you want a single unidimensional measure of sentiment for a given sentence.
    # Calling it a 'normalized, weighted composite score' is accurate.
    compound: float

    # Info from GerVADER:
    # The 'pos', 'neu', and 'neg' scores are ratios for proportions of text that fall in each category (so these
    # should all add up to be 1... or close to it with float operation).  These are the most useful metrics if
    # you want multidimensional measures of sentiment for a given sentence.
    pos: float
    neu: float
    neg: float


class SentimentSelection(BaseModel):
    selection: str
    sentences: List[SentimentSentence]
