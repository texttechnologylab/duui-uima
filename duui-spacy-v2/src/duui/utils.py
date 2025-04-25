from typing import Optional

import spacy
from fastapi.datastructures import State
from spacy import Language
from spacy.tokens import Doc

from duui.const import SpacyModelName
from duui.models import DuuiSentence, SpacySettings
from duui.settings import SETTINGS


def load_spacy_model(
    settings: SpacySettings,
    model_name: Optional[SpacyModelName] = None,
) -> spacy.Language:
    return spacy.load(
        model_name or settings.resolve_model(),
        exclude=settings.spacy_exclude
        if settings.spacy_exclude is not None
        else SETTINGS.spacy_exclude,
    )


def get_spacy_model(state: State, settings: SpacySettings):
    model_name = settings.resolve_model()
    if model_name in state.models:
        state.lru.remove(model_name)
        state.lru.insert(0, model_name)
        return state.models[model_name]
    else:
        if len(state.lru) >= SETTINGS.max_loaded_models:
            oldest_model_name = state.lru.pop()
            del state.models[oldest_model_name]

        model = load_spacy_model(settings, model_name)
        state.models[model_name] = model
        state.lru.insert(0, model_name)
        return model


def get_doc(nlp: Language, sentence: DuuiSentence) -> Doc:
    with nlp.select_pipes(enable=["tokenizer"]):
        doc = nlp(sentence.text)
        doc.user_data["offset"] = sentence.offset
        return doc
