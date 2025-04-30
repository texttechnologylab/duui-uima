from typing import Optional

import spacy
from fastapi.datastructures import State
from fastapi.logger import logger

from duui.const import SpacyModelName
from duui.models import SpacySettings
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
        if 0 < SETTINGS.max_loaded_models <= len(state.lru):
            oldest_model_name = state.lru.pop()
            logger.info(
                f"Exceeded max_loaded_models={SETTINGS.max_loaded_models}, unloading model: {oldest_model_name}",
            )
            del state.models[oldest_model_name]

        logger.info(f"Loading spaCy model: {model_name}")
        model = load_spacy_model(settings, model_name)
        state.models[model_name] = model
        state.lru.insert(0, model_name)
        return model
