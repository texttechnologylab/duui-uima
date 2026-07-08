from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


TIME_BASE_TYPE = "org.texttechnologylab.annotation.semaf.isotimeml.TimeX3"
TIME_TYPE_MAP: Dict[str, str] = {
    "DATE": "org.texttechnologylab.annotation.semaf.isotimeml.time.Date",
    "TIME": "org.texttechnologylab.annotation.semaf.isotimeml.time.Time",
    "DURATION": "org.texttechnologylab.annotation.semaf.isotimeml.time.Duration",
    "SET": "org.texttechnologylab.annotation.semaf.isotimeml.time.Set",
    "UNKNOWN": TIME_BASE_TYPE,
}

LANGUAGE_TO_CULTURE: Dict[str, str] = {
    "de": "de-de",
    "en": "en-us",
    "es": "es-es",
    "fr": "fr-fr",
    "it": "it-it",
    "pt": "pt-br",
}

LANGUAGE_TO_DUCKLING_LOCALE: Dict[str, str] = {
    "de": "de_DE",
    "en": "en_US",
    "es": "es_ES",
    "fr": "fr_FR",
    "it": "it_IT",
    "pt": "pt_BR",
}

LANGUAGE_TO_TEI2GO_MODEL: Dict[str, str] = {
    "de": "de_tei2go",
    "en": "en_tei2go",
    "es": "es_tei2go",
    "fr": "fr_tei2go",
    "it": "it_tei2go",
    "pt": "pt_tei2go",
}

MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "microsoft": {
        "backend": "microsoft_recognizers_text",
        "model_id": "recognizers-text-suite",
        "model_source": "https://github.com/microsoft/Recognizers-Text",
        "model_lang": "multi",
    },
    "duckling": {
        "backend": "duckling_http",
        "model_id": "duckling",
        "model_source": "https://github.com/facebook/duckling",
        "model_lang": "multi",
    },
    "sutime": {
        "backend": "sutime_http",
        "model_id": "stanford-corenlp-sutime",
        "model_source": "https://stanfordnlp.github.io/CoreNLP/sutime.html",
        "model_lang": "multi",
    },
    "german-gelectra": {
        "backend": "hf_token_classification",
        "model_id": "satyaalmasian/temporal_tagger_German_GELECTRA",
        "model_source": "https://huggingface.co/satyaalmasian/temporal_tagger_German_GELECTRA",
        "model_lang": "de",
    },
        "bert-got-a-date": {
            "backend": "hf_token_classification",
            "model_id": "satyaalmasian/temporal_tagger_BERT_tokenclassifier",
            "model_source": "https://github.com/satya77/Transformer_Temporal_Tagger",
            "model_lang": "en",
        },
    "tei2go": {
        "backend": "spacy_tei2go",
        "model_id": "tei2go",
        "model_source": "https://github.com/hmosousa/tei2go",
        "model_lang": "multi",
    },
    "timexy": {
        "backend": "spacy_timexy",
        "model_id": "timexy",
        "model_source": "https://pypi.org/project/timexy/",
        "model_lang": "multi",
    },
    "hf-token-classification": {
        "backend": "hf_token_classification",
        "model_id": "",
        "model_source": "",
        "model_lang": "multi",
    },
}


class TimeRecognizer(ABC):
    model_name: str
    model_id: str
    model_version: str
    model_source: str
    model_lang: str

    @abstractmethod
    def predict(
        self,
        texts: List[str],
        language: str,
        document_creation_time: Optional[str] = None,
        threshold: float = 0.0,
        batch_size: int = 8,
        duckling_url: Optional[str] = None,
        corenlp_url: Optional[str] = None,
        duckling_timezone: Optional[str] = None,
    ) -> List[List[Dict[str, Any]]]:
        raise NotImplementedError


def resolve_model_name(model_name: str) -> Tuple[str, Dict[str, str]]:
    selected = (model_name or "").strip()
    if not selected:
        selected = "microsoft"

    if selected in MODEL_REGISTRY:
        return selected, MODEL_REGISTRY[selected]

    for alias, cfg in MODEL_REGISTRY.items():
        if selected == cfg.get("model_id") and cfg.get("model_id"):
            return alias, cfg

    # Exact Hugging Face model ids are allowed through the generic backend.
    if "/" in selected:
        cfg = dict(MODEL_REGISTRY["hf-token-classification"])
        cfg["model_id"] = selected
        cfg["model_source"] = f"https://huggingface.co/{selected}"
        return selected, cfg

    supported = sorted(list(MODEL_REGISTRY.keys()))
    raise ValueError(f"Unsupported model_name '{selected}'. Supported values: {', '.join(supported)}")


def apply_model_settings(
    cfg: Dict[str, str],
    *,
    model_specname: Optional[str] = None,
    model_source: Optional[str] = None,
    model_lang: Optional[str] = None,
) -> Dict[str, str]:
    """Apply Docker build-time model settings to a registry entry.

    MODEL_NAME selects the registry alias/backend. MODEL_SPECNAME can override
    the concrete model id/specification stored in the registry. MODEL_SOURCE
    and MODEL_LANG are metadata fields written to the DUUI response.
    """
    configured = dict(cfg)

    specname = (model_specname or "").strip()
    source = (model_source or "").strip()
    lang = (model_lang or "").strip()

    if specname:
        configured["model_id"] = specname
    if source:
        configured["model_source"] = source
    if lang:
        configured["model_lang"] = lang

    return configured


def normalize_timex_type(value: Optional[str]) -> str:
    if not value:
        return "UNKNOWN"

    label = str(value).strip().upper()
    label = re.sub(r"^[BI]-", "", label)

    # TIMEX/TIMEX3 ist generisch, nicht automatisch TIME.
    if label in {"TIMEX", "TIMEX3", "TIME_EXPRESSION"}:
        return "UNKNOWN"

    if "DURATION" in label or label in {"DUR", "PERIOD"}:
        return "DURATION"
    if "SET" in label or label in {"FREQUENCY", "FREQ"}:
        return "SET"
    if "DATERANGE" in label or label in {"DATEPERIOD"}:
        return "DATE"
    if "DATE" in label and "TIME" not in label:
        return "DATE"
    if "DATETIME" in label:
        return "TIME"
    if "TIME" in label:
        return "TIME"

    return "UNKNOWN"


GERMAN_MONTHS = {
    "januar", "jan", "februar", "feb", "märz", "maerz", "mrz",
    "april", "apr", "mai", "juni", "jun", "juli", "jul",
    "august", "aug", "september", "sep", "oktober", "okt",
    "november", "nov", "dezember", "dez",
}

GERMAN_WEEKDAYS = {
    "montag", "dienstag", "mittwoch", "donnerstag",
    "freitag", "samstag", "sonntag",
}

RELATIVE_DATE_WORDS_DE = {
    "heute", "morgen", "übermorgen", "uebermorgen", "gestern", "vorgestern",
}

SET_WORDS_DE = {
    "jeden", "jede", "jeder", "jedes", "täglich", "taeglich",
    "wöchentlich", "woechentlich", "monatlich", "jährlich", "jaehrlich",
}


def infer_timex_type_from_text(text: str, language: str = "de") -> str:
    normalized = text.strip().lower()

    if not normalized:
        return "UNKNOWN"

    tokens = re.findall(r"\w+", normalized, flags=re.UNICODE)
    token_set = set(tokens)

    if token_set & SET_WORDS_DE:
        return "SET"

    if re.search(r"\b\d{1,2}\s*(uhr|:\d{2})\b", normalized):
        return "TIME"

    if re.search(r"\b(sekunde|minuten?|stunden?|tage?|wochen?|monate?|jahre?)\b", normalized):
        if re.search(r"\b\d+\b", normalized):
            return "DURATION"

    if token_set & RELATIVE_DATE_WORDS_DE:
        return "DATE"

    if token_set & GERMAN_MONTHS:
        return "DATE"

    if token_set & GERMAN_WEEKDAYS:
        return "DATE"

    if re.search(r"\b\d{1,2}\.\s*\d{1,2}\.?\s*(\d{2,4})?\b", normalized):
        return "DATE"

    if re.fullmatch(r"\d{4}", normalized):
        return "DATE"

    return "UNKNOWN"


def infer_timex_type_from_timex3(value: Optional[str]) -> str:
    if not value:
        return "UNKNOWN"

    match = re.search(r'type="([^"]+)"', str(value), flags=re.IGNORECASE)
    if not match:
        return "UNKNOWN"

    return normalize_timex_type(match.group(1))

def normalize_resolution_type(value_type: Optional[str]) -> str:
    if not value_type:
        return "UNKNOWN"

    value_type = str(value_type).lower()

    if value_type in {"date", "daterange", "dateperiod"}:
        return "DATE"
    if value_type in {"time"}:
        return "TIME"
    if value_type in {"datetime", "datetimerange", "timeperiod"}:
        return "TIME"
    if value_type == "duration":
        return "DURATION"
    if value_type == "set":
        return "SET"

    return "UNKNOWN"


def time_type_for_timex(timex_type: str) -> str:
    return TIME_TYPE_MAP.get(normalize_timex_type(timex_type), TIME_BASE_TYPE)


def make_time_expression(
    *,
    sentence: str,
    start: int,
    end: int,
    timex_type: Optional[str],
    value: Optional[str],
    score: Optional[float],
    model_name: str,
    model_id: str,
    model_version: str = "",
    raw: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    try:
        start_i = max(0, int(start))
        end_i = min(len(sentence), int(end))
    except Exception:
        return None

    if end_i <= start_i:
        return None

    normalized_type = normalize_timex_type(timex_type)
    covered = sentence[start_i:end_i]

    return {
        "text": covered,
        "label": normalized_type,
        "timex_type": normalized_type,
        "value": value,
        "score": float(score) if score is not None else 1.0,
        "start": start_i,
        "end": end_i,
        "time_type": time_type_for_timex(normalized_type),
        "model_name": model_name,
        "model_id": model_id,
        "model_version": model_version,
        "raw": raw or {},
    }


def deduplicate_and_prefer_longest(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
    for item in items:
        key = (int(item["start"]), int(item["end"]), str(item.get("timex_type", "UNKNOWN")))
        current = unique.get(key)
        if current is None or float(item.get("score", 0.0)) > float(current.get("score", 0.0)):
            unique[key] = item

    ordered = sorted(unique.values(), key=lambda x: (int(x["start"]), -(int(x["end"]) - int(x["start"]))))
    result: List[Dict[str, Any]] = []

    for item in ordered:
        overlaps = [existing for existing in result if not (item["end"] <= existing["start"] or item["start"] >= existing["end"])]
        if not overlaps:
            result.append(item)
            continue

        longest = max(overlaps, key=lambda x: int(x["end"]) - int(x["start"]))
        if int(item["end"]) - int(item["start"]) > int(longest["end"]) - int(longest["start"]):
            result = [existing for existing in result if existing is not longest]
            result.append(item)

    return sorted(result, key=lambda x: (int(x["start"]), int(x["end"])))


class MicrosoftRecognizer(TimeRecognizer):
    def __init__(self, alias: str, cfg: Dict[str, str], model_version: str = ""):
        try:
            from recognizers_date_time import recognize_datetime
        except ImportError as exc:
            raise RuntimeError("Install recognizers-text-suite to use the Microsoft backend.") from exc

        self.recognize_datetime = recognize_datetime
        self.model_name = alias
        self.model_id = cfg["model_id"]
        self.model_version = model_version or "latest"
        self.model_source = cfg.get("model_source", "")
        self.model_lang = cfg.get("model_lang", "multi")

    def predict(
        self,
        texts: List[str],
        language: str,
        document_creation_time: Optional[str] = None,
        threshold: float = 0.0,
        batch_size: int = 8,
        duckling_url: Optional[str] = None,
        corenlp_url: Optional[str] = None,
        duckling_timezone: Optional[str] = None,
    ) -> List[List[Dict[str, Any]]]:
        culture = LANGUAGE_TO_CULTURE.get(language.lower(), language)
        outputs: List[List[Dict[str, Any]]] = []

        for sentence in texts:
            items: List[Dict[str, Any]] = []
            for result in self.recognize_datetime(sentence, culture):
                result_text = getattr(result, "text", "") or ""
                start = int(getattr(result, "start", 0))
                end = start + len(result_text) if result_text else int(getattr(result, "end", start)) + 1
                resolution = getattr(result, "resolution", {}) or {}
                type_name = getattr(result, "type_name", "") or ""
                value = self._extract_value(resolution)
                timex_type = self._map_type(type_name, resolution)
                item = make_time_expression(
                    sentence=sentence,
                    start=start,
                    end=end,
                    timex_type=timex_type,
                    value=value,
                    score=1.0,
                    model_name=self.model_name,
                    model_id=self.model_id,
                    model_version=self.model_version,
                    raw={"type_name": type_name, "resolution": resolution},
                )
                if item is not None:
                    items.append(item)
            outputs.append(deduplicate_and_prefer_longest(items))
        return outputs

    @staticmethod
    def _extract_value(resolution: Dict[str, Any]) -> Optional[str]:
        values = resolution.get("values") or []
        if not values or not isinstance(values[0], dict):
            return None
        first = values[0]
        return first.get("timex") or first.get("value") or first.get("start") or first.get("end")

    @staticmethod
    def _map_type(type_name: str, resolution: Dict[str, Any]) -> str:
        values = resolution.get("values") or []

        # Microsoft liefert oft den zuverlässigsten Typ hier:
        # {'type': 'daterange'} => DATE
        if values and isinstance(values[0], dict):
            from_resolution = normalize_resolution_type(values[0].get("type"))
            if from_resolution != "UNKNOWN":
                return from_resolution

        name = type_name.lower()

        if "duration" in name:
            return "DURATION"
        if "set" in name:
            return "SET"
        if "daterange" in name or "dateperiod" in name:
            return "DATE"
        if "datetimerange" in name:
            return "TIME"
        if "datetime" in name:
            return "TIME"
        if "time" in name:
            return "TIME"
        if "date" in name:
            return "DATE"

        return "UNKNOWN"


class DucklingRecognizer(TimeRecognizer):
    def __init__(self, alias: str, cfg: Dict[str, str], model_version: str = ""):
        self.model_name = alias
        self.model_id = cfg["model_id"]
        self.model_version = model_version or "latest"
        self.model_source = cfg.get("model_source", "")
        self.model_lang = cfg.get("model_lang", "multi")
        self.default_url = ""

    def predict(
        self,
        texts: List[str],
        language: str,
        document_creation_time: Optional[str] = None,
        threshold: float = 0.0,
        batch_size: int = 8,
        duckling_url: Optional[str] = None,
        corenlp_url: Optional[str] = None,
        duckling_timezone: Optional[str] = None,
    ) -> List[List[Dict[str, Any]]]:
        locale = LANGUAGE_TO_DUCKLING_LOCALE.get(language.lower(), "en_US")
        url = (duckling_url or self.default_url).rstrip("/")
        if not url:
            raise RuntimeError(
                "duckling_url is required for the duckling backend. "
                "Pass it with DUUI .withParameter('duckling_url', 'http://duckling:8000')."
            )
        timezone_name = duckling_timezone or "Europe/Berlin"
        outputs: List[List[Dict[str, Any]]] = []

        for sentence in texts:
            response = requests.post(
                f"{url}/parse",
                data={
                    "text": sentence,
                    "locale": locale,
                    "tz": timezone_name,
                    "dims": json.dumps(["time", "duration"]),
                },
                timeout=30,
            )
            response.raise_for_status()
            items: List[Dict[str, Any]] = []
            for raw in response.json():
                dim = raw.get("dim")
                if dim not in {"time", "duration"}:
                    continue
                value = raw.get("value") or {}
                item = make_time_expression(
                    sentence=sentence,
                    start=raw.get("start", 0),
                    end=raw.get("end", 0),
                    timex_type=self._map_type(dim, value),
                    value=self._extract_value(value),
                    score=1.0,
                    model_name=self.model_name,
                    model_id=self.model_id,
                    model_version=self.model_version,
                    raw=raw,
                )
                if item is not None:
                    items.append(item)
            outputs.append(deduplicate_and_prefer_longest(items))
        return outputs

    @staticmethod
    def _map_type(dimension: str, value: Dict[str, Any]) -> str:
        if dimension == "duration":
            return "DURATION"
        if value.get("type") == "interval":
            return "DATE"
        if value.get("grain") in {"second", "minute", "hour"}:
            return "TIME"
        return "DATE"

    @staticmethod
    def _extract_value(value: Dict[str, Any]) -> Optional[str]:
        if "value" in value:
            return str(value["value"])
        if value.get("type") == "interval":
            start = (value.get("from") or {}).get("value")
            end = (value.get("to") or {}).get("value")
            if start or end:
                return f"{start or ''}/{end or ''}"
        normalized = value.get("normalized")
        return str(normalized) if normalized else None


class SutimeRecognizer(TimeRecognizer):
    def __init__(self, alias: str, cfg: Dict[str, str], model_version: str = ""):
        self.model_name = alias
        self.model_id = cfg["model_id"]
        self.model_version = model_version or "latest"
        self.model_source = cfg.get("model_source", "")
        self.model_lang = cfg.get("model_lang", "multi")
        self.default_url = ""

    def predict(
        self,
        texts: List[str],
        language: str,
        document_creation_time: Optional[str] = None,
        threshold: float = 0.0,
        batch_size: int = 8,
        duckling_url: Optional[str] = None,
        corenlp_url: Optional[str] = None,
        duckling_timezone: Optional[str] = None,
    ) -> List[List[Dict[str, Any]]]:
        url = (corenlp_url or self.default_url).rstrip("/")
        if not url:
            raise RuntimeError(
                "corenlp_url is required for the sutime backend. "
                "Pass it with DUUI .withParameter('corenlp_url', 'http://corenlp:9000')."
            )
        outputs: List[List[Dict[str, Any]]] = []
        for sentence in texts:
            properties: Dict[str, Any] = {
                "annotators": "tokenize,ssplit,pos,lemma,ner",
                "outputFormat": "json",
            }
            if document_creation_time:
                properties["sutime.referenceDate"] = document_creation_time[:10]
            if language.lower() == "de":
                properties["tokenize.language"] = "de"

            response = requests.post(
                url,
                params={"properties": json.dumps(properties)},
                data=sentence.encode("utf-8"),
                headers={"Content-Type": "text/plain; charset=utf-8"},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            items: List[Dict[str, Any]] = []
            for sent in data.get("sentences", []):
                for mention in sent.get("entitymentions", []):
                    ner = mention.get("ner")
                    if ner not in {"DATE", "TIME", "DURATION", "SET"}:
                        continue
                    timex = mention.get("timex") or {}
                    item = make_time_expression(
                        sentence=sentence,
                        start=mention.get("characterOffsetBegin", 0),
                        end=mention.get("characterOffsetEnd", 0),
                        timex_type=timex.get("type") or ner,
                        value=timex.get("value") or mention.get("normalizedNER"),
                        score=1.0,
                        model_name=self.model_name,
                        model_id=self.model_id,
                        model_version=self.model_version,
                        raw=mention,
                    )
                    if item is not None:
                        items.append(item)
            outputs.append(deduplicate_and_prefer_longest(items))
        return outputs


class HuggingFaceTimeRecognizer(TimeRecognizer):
    def __init__(self, alias: str, cfg: Dict[str, str], device: str = "cpu", model_version: str = ""):
        try:
            from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
        except ImportError as exc:
            raise RuntimeError("Install transformers and torch to use the Hugging Face backend.") from exc

        model_id = cfg.get("model_id") or os.getenv("HF_TEMPORAL_MODEL_ID", "")
        if not model_id:
            raise ValueError(
                f"The backend '{alias}' needs a Hugging Face model id. "
                "Set MODEL_NAME to an exact Hugging Face id or set HF_TEMPORAL_MODEL_ID."
            )

        self.model_name = alias
        self.model_id = model_id
        self.model_version = model_version or "latest"
        self.model_source = cfg.get("model_source") or f"https://huggingface.co/{model_id}"
        self.model_lang = cfg.get("model_lang", "multi")

        import torch

        self.device = device
        revision = None if self.model_version in {"", "latest"} else self.model_version
        self.model = AutoModelForTokenClassification.from_pretrained(model_id, revision=revision)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, revision=revision)
        self.model.to(device)
        self.model.eval()
        pipe_device = 0 if str(device).startswith("cuda") else -1
        self.pipeline = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=pipe_device,
        )
        self.torch = torch

    def predict(
        self,
        texts: List[str],
        language: str,
        document_creation_time: Optional[str] = None,
        threshold: float = 0.0,
        batch_size: int = 8,
        duckling_url: Optional[str] = None,
        corenlp_url: Optional[str] = None,
        duckling_timezone: Optional[str] = None,
    ) -> List[List[Dict[str, Any]]]:
        with self.torch.no_grad():
            outputs = self.pipeline(texts, batch_size=batch_size)

        sentence_outputs = normalize_sentence_outputs(outputs, len(texts))
        result: List[List[Dict[str, Any]]] = []

        for sentence, raw_items in zip(texts, sentence_outputs):
            items: List[Dict[str, Any]] = []
            for raw in flatten_raw_items(raw_items):
                score = float(raw.get("score", 1.0) or 0.0)
                if score < threshold:
                    continue
                label = raw.get("entity_group") or raw.get("entity") or raw.get("label")
                timex_type = normalize_timex_type(str(label))
                if timex_type == "UNKNOWN":
                    continue
                item = make_time_expression(
                    sentence=sentence,
                    start=raw.get("start", 0),
                    end=raw.get("end", 0),
                    timex_type=timex_type,
                    value=None,
                    score=score,
                    model_name=self.model_name,
                    model_id=self.model_id,
                    model_version=self.model_version,
                    raw=raw,
                )
                if item is not None:
                    items.append(item)
            result.append(deduplicate_and_prefer_longest(items))
        return result


class SpacyTimeRecognizer(TimeRecognizer):
    def __init__(self, alias: str, cfg: Dict[str, str], language: str, model_version: str = ""):
        import importlib
        import spacy

        self.model_name = alias
        self.model_id = cfg["model_id"]
        self.model_version = model_version or "latest"
        self.model_source = cfg.get("model_source", "")
        self.model_lang = cfg.get("model_lang", "multi")

        backend = cfg["backend"]
        if backend == "spacy_tei2go":
            model_name = os.getenv(
                "TEI2GO_MODEL",
                LANGUAGE_TO_TEI2GO_MODEL.get(language.lower(), "de_tei2go"),
            )
        else:
            model_name = os.getenv(
                "SPACY_MODEL",
                "de_core_news_sm" if language.lower() == "de" else "en_core_web_sm",
            )

        self.nlp = self._load_spacy_model(
            importlib=importlib,
            spacy=spacy,
            model_name=model_name,
            backend=backend,
            language=language,
        )

        if backend == "spacy_timexy" and "timexy" not in self.nlp.pipe_names:
            try:
                from timexy import Timexy  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "Install timexy to use the Timexy backend: python -m pip install timexy==0.1.3"
                ) from exc

            timexy_config = {
                "kb_id_type": "timex3",
                "label": "TIMEX",
                "overwrite": False,
            }

            if "ner" in self.nlp.pipe_names:
                self.nlp.add_pipe("timexy", config=timexy_config, before="ner")
            else:
                self.nlp.add_pipe("timexy", config=timexy_config)

    @staticmethod
    def _load_spacy_model(importlib: Any, spacy: Any, model_name: str, backend: str, language: str) -> Any:
        try:
            module = importlib.import_module(model_name)
            return module.load() if hasattr(module, "load") else spacy.load(model_name)

        except ModuleNotFoundError as exc:
            if exc.name != model_name:
                raise RuntimeError(
                    f"spaCy model '{model_name}' could not be imported because dependency "
                    f"{exc.name!r} is missing."
                ) from exc

            try:
                return spacy.load(model_name)
            except OSError as load_exc:
                raise RuntimeError(
                    SpacyTimeRecognizer._missing_spacy_model_message(
                        backend=backend,
                        language=language,
                        model_name=model_name,
                    )
                ) from load_exc

        except ImportError as exc:
            raise RuntimeError(
                f"spaCy model '{model_name}' could not be imported: {exc}"
            ) from exc

        except OSError as exc:
            raise RuntimeError(
                SpacyTimeRecognizer._missing_spacy_model_message(
                    backend=backend,
                    language=language,
                    model_name=model_name,
                )
            ) from exc

    @staticmethod
    def _missing_spacy_model_message(backend: str, language: str, model_name: str) -> str:
        if backend == "spacy_tei2go":
            wheel_url = (
                f"https://huggingface.co/hugosousa/{model_name}/resolve/main/"
                f"{model_name}-any-py3-none-any.whl"
            )
            local_wheel = f"/tmp/{model_name}-0.0.0-py3-none-any.whl"

            return (
                f"TEI2GO spaCy model '{model_name}' is not installed for language "
                f"'{language}'. Install it with:\n"
                f"  curl -L -o {local_wheel} {wheel_url}\n"
                f"  python -m pip install --no-deps {local_wheel}\n"
                "Or set TEI2GO_MODEL to an installed spaCy model package/path."
            )

        return (
            f"spaCy model '{model_name}' is not installed for language '{language}'. "
            f"Install it with:\n"
            f"  python -m spacy download {model_name}\n"
            "Or set SPACY_MODEL to an installed spaCy model package/path."
        )

    def predict(
            self,
            texts: List[str],
            language: str,
            document_creation_time: Optional[str] = None,
            threshold: float = 0.0,
            batch_size: int = 8,
        duckling_url: Optional[str] = None,
        corenlp_url: Optional[str] = None,
        duckling_timezone: Optional[str] = None,
    ) -> List[List[Dict[str, Any]]]:
        outputs: List[List[Dict[str, Any]]] = []

        # Wichtig: explizit über die Eingabetexte iterieren.
        # So bleibt len(outputs) immer gleich len(texts).
        for sentence in texts:
            doc = self.nlp(sentence)
            items: List[Dict[str, Any]] = []

            for span in iter_spacy_temporal_spans(doc):
                label = getattr(span, "label_", "UNKNOWN") or "UNKNOWN"
                value = extract_spacy_value(span)

                timex_type = normalize_timex_type(label)

                # Timexy: label ist oft nur "TIMEX"; echter Typ steht in kb_id_/value.
                if timex_type == "UNKNOWN":
                    timex_type = infer_timex_type_from_timex3(value)

                # TEI2GO: label ist oft "TIMEX", value ist oft None; dann aus Text ableiten.
                if timex_type == "UNKNOWN" and str(label).upper() in {
                    "TIMEX",
                    "TIMEX3",
                    "TIME_EXPRESSION",
                }:
                    timex_type = infer_timex_type_from_text(span.text, language=language)

                if timex_type == "UNKNOWN":
                    continue

                item = make_time_expression(
                    sentence=sentence,
                    start=span.start_char,
                    end=span.end_char,
                    timex_type=timex_type,
                    value=value,
                    score=1.0,
                    model_name=self.model_name,
                    model_id=self.model_id,
                    model_version=self.model_version,
                    raw={"label": label},
                )

                if item is not None:
                    items.append(item)

            outputs.append(deduplicate_and_prefer_longest(items))

        return outputs


def normalize_sentence_outputs(outputs: Any, text_count: int) -> List[Any]:
    if outputs is None:
        sentence_outputs: List[Any] = []
    elif isinstance(outputs, dict):
        sentence_outputs = [outputs]
    elif isinstance(outputs, list):
        if text_count == 1:
            if outputs and all(isinstance(item, dict) for item in outputs):
                sentence_outputs = [outputs]
            else:
                sentence_outputs = outputs
        else:
            sentence_outputs = outputs
    else:
        sentence_outputs = []

    if len(sentence_outputs) < text_count:
        sentence_outputs = sentence_outputs + [None] * (text_count - len(sentence_outputs))
    elif len(sentence_outputs) > text_count:
        sentence_outputs = sentence_outputs[:text_count]
    return sentence_outputs


def flatten_raw_items(raw: Any) -> Iterable[Dict[str, Any]]:
    if raw is None:
        return
    if isinstance(raw, dict):
        if "entities" in raw:
            yield from flatten_raw_items(raw["entities"])
            return
        if any(key in raw for key in ("start", "end", "entity", "entity_group", "label")):
            yield raw
            return
        for value in raw.values():
            yield from flatten_raw_items(value)
        return
    if isinstance(raw, list):
        for item in raw:
            yield from flatten_raw_items(item)


def iter_spacy_temporal_spans(doc: Any) -> Iterable[Any]:
    for entity in getattr(doc, "ents", []):
        yield entity

    for name, spans in getattr(doc, "spans", {}).items():
        if "time" not in str(name).lower() and "timex" not in str(name).lower():
            continue
        for span in spans:
            yield span


def extract_spacy_value(span: Any) -> Optional[str]:
    if getattr(span, "kb_id_", None):
        return str(span.kb_id_)

    custom = getattr(span, "_", None)
    if custom is None:
        return None

    for attribute_name in ["timex_value", "value", "timex3", "normalized", "timex"]:
        try:
            if custom.has(attribute_name):
                value = getattr(custom, attribute_name)
                if value:
                    return str(value)
        except Exception:
            continue
    return None


def create_time_recognizer(
    model_name: str,
    language: str,
    device: str = "cpu",
    model_version: Optional[str] = None,
    model_specname: Optional[str] = None,
    model_source: Optional[str] = None,
    model_lang: Optional[str] = None,
) -> TimeRecognizer:
    alias, cfg = resolve_model_name(model_name)
    cfg = apply_model_settings(
        cfg,
        model_specname=model_specname,
        model_source=model_source,
        model_lang=model_lang,
    )
    backend = cfg["backend"]

    if backend == "microsoft_recognizers_text":
        return MicrosoftRecognizer(alias, cfg, model_version=model_version or "latest")
    if backend == "duckling_http":
        return DucklingRecognizer(alias, cfg, model_version=model_version or "latest")
    if backend == "sutime_http":
        return SutimeRecognizer(alias, cfg, model_version=model_version or "latest")
    if backend == "hf_token_classification":
        return HuggingFaceTimeRecognizer(alias, cfg, device=device, model_version=model_version or "latest")
    if backend in {"spacy_tei2go", "spacy_timexy"}:
        return SpacyTimeRecognizer(alias, cfg, language=language, model_version=model_version or "latest")

    raise ValueError(f"Unsupported backend '{backend}' for model '{model_name}'")


def predict_time(
    model_name: str,
    texts: List[str],
    language: str = "de",
    document_creation_time: Optional[str] = None,
    device: str = "cpu",
    threshold: float = 0.0,
    batch_size: int = 8,
    model_version: str = "latest",
    model_specname: Optional[str] = None,
    model_source: Optional[str] = None,
    model_lang: Optional[str] = None,
    duckling_url: Optional[str] = None,
    corenlp_url: Optional[str] = None,
    duckling_timezone: Optional[str] = None,
) -> List[List[Dict[str, Any]]]:
    recognizer = create_time_recognizer(
        model_name,
        language=language,
        device=device,
        model_version=model_version,
        model_specname=model_specname,
        model_source=model_source,
        model_lang=model_lang,
    )
    return recognizer.predict(
        texts,
        language=language,
        document_creation_time=document_creation_time,
        threshold=threshold,
        batch_size=batch_size,
        duckling_url=duckling_url,
        corenlp_url=corenlp_url,
        duckling_timezone=duckling_timezone,
    )


if __name__ == "__main__":
    examples_de = [
        "Wir treffen uns morgen.",
        "Wir treffen uns morgen um 14 Uhr.",
        "Wir treffen uns morgen um 14 Uhr und danach jeden Montag.",
        "Die Sitzung fand vom 3. bis 5. Mai 2024 statt.",
    ]

    examples_en = [
        "We will meet tomorrow.",
        "We will meet tomorrow at 2 pm.",
        "We will meet tomorrow at 2 pm and then every Monday.",
        "The meeting took place from May 3 to May 5, 2024.",
    ]

    tests = [
        {
            "name": "microsoft",
            "examples": examples_de,
            "language": "de",
            "kwargs": {},
        },
        {
            "name": "tei2go",
            "examples": examples_de,
            "language": "de",
            "kwargs": {},
        },
        {
            "name": "timexy",
            "examples": examples_de,
            "language": "de",
            "kwargs": {},
        },
        {
            "name": "german-gelectra",
            "examples": examples_de,
            "language": "de",
            "kwargs": {},
        },
        {
            "name": "bert-got-a-date",
            "examples": examples_en,
            "language": "en",
            "kwargs": {
                "model_specname": "satyaalmasian/temporal_tagger_BERT_tokenclassifier",
                "model_source": "https://github.com/satya77/Transformer_Temporal_Tagger",
                "model_lang": "en",
            },
        },
        {
            "name": "hf-token-classification",
            "examples": examples_en,
            "language": "en",
            "kwargs": {
                "model_specname": "satyaalmasian/temporal_tagger_BERT_tokenclassifier",
                "model_source": "https://huggingface.co/satyaalmasian/temporal_tagger_BERT_tokenclassifier",
                "model_lang": "en",
            },
        },
        {
            "name": "duckling",
            "examples": examples_de,
            "language": "de",
            "kwargs": {},
        },
        {
            "name": "sutime",
            "examples": examples_de,
            "language": "de",
            "kwargs": {},
        },
    ]

    print("TIME_TYPE_MAP:")
    for key, value in TIME_TYPE_MAP.items():
        print(f"  {key:8} -> {value}")

    print("\n" + "=" * 80)
    print("Running backend tests")

    for test in tests:
        model_name = test["name"]
        examples = test["examples"]
        language = test["language"]
        kwargs = test["kwargs"]

        print("\n" + "=" * 80)
        print(f"Model: {model_name}")
        print(f"Language: {language}")
        print(f"Input length: {len(examples)}")

        try:
            result = predict_time(
                model_name,
                examples,
                language=language,
                document_creation_time="2026-06-09",
                device="cpu",
                threshold=0.0,
                batch_size=4,
                **kwargs,
            )

            print(f"Output length: {len(result)}")

            for i, (text, items) in enumerate(zip(examples, result), start=1):
                print("\n" + "-" * 80)
                print(f"[{i}] Input: {text}")

                if not items:
                    print("  -> []")
                    continue

                for j, item in enumerate(items, start=1):
                    raw = item.get("raw") or {}

                    raw_model_type = (
                        raw.get("type_name")
                        or raw.get("label")
                        or raw.get("dim")
                        or raw.get("entity_group")
                        or raw.get("entity")
                        or raw.get("ner")
                        or None
                    )

                    # For Microsoft/Duckling/SUTime-like raw dicts, try to expose nested type too.
                    raw_resolution_type = None
                    resolution = raw.get("resolution")
                    if isinstance(resolution, dict):
                        values = resolution.get("values") or []
                        if values and isinstance(values[0], dict):
                            raw_resolution_type = values[0].get("type")

                    duckling_value = raw.get("value")
                    if isinstance(duckling_value, dict):
                        raw_resolution_type = duckling_value.get("type") or duckling_value.get("grain")

                    print(f"  Hit {j}:")
                    print(f"    text:              {item.get('text')!r}")
                    print(f"    model_raw_type:    {raw_model_type!r}")
                    print(f"    model_value_type:  {raw_resolution_type!r}")
                    print(f"    normalized_type:   {item.get('timex_type')!r}")
                    print(f"    mapped_uima_type:  {item.get('time_type')!r}")
                    print(f"    value:             {item.get('value')!r}")
                    print(f"    score:             {item.get('score')!r}")
                    print(f"    span:              ({item.get('start')}, {item.get('end')})")

        except Exception as exc:
            print(f"FAILED: {type(exc).__name__}: {exc}")
