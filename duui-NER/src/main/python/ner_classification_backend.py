from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "gliner": {
        "backend": "gliner",
        "model_id": "urchade/gliner_multi-v2.1",
        "model_source": "https://huggingface.co/urchade/gliner_multi-v2.1",
        "model_lang": "multi",
    },
    "gliner2": {
        "backend": "gliner2",
        "model_id": "fastino/gliner2-multi-v1",
        "model_source": "https://huggingface.co/fastino/gliner2-multi-v1",
        "model_lang": "multi",
    },
    "roberta-ner-multilingual": {
        "backend": "hf_token_classification",
        "model_id": "julian-schelb/roberta-ner-multilingual",
        "model_source": "https://huggingface.co/julian-schelb/roberta-ner-multilingual",
        "model_lang": "multi",
    },
    "wikineural-multilingual-ner": {
        "backend": "hf_token_classification",
        "model_id": "Babelscape/wikineural-multilingual-ner",
        "model_source": "https://huggingface.co/Babelscape/wikineural-multilingual-ner",
        "model_lang": "multi",
    },
    "xlm-r-ner-40-lang": {
        "backend": "hf_token_classification",
        "model_id": "nbroad/jplu-xlm-r-ner-40-lang",
        "model_source": "https://huggingface.co/nbroad/jplu-xlm-r-ner-40-lang",
        "model_lang": "multi",
    },
}


def resolve_model_name(model_name: str) -> Tuple[str, Dict[str, str]]:
    """Resolve a short alias or exact HuggingFace model id to a registry entry."""
    model_name = (model_name or "").strip()
    if model_name in MODEL_REGISTRY:
        return model_name, MODEL_REGISTRY[model_name]

    for alias, cfg in MODEL_REGISTRY.items():
        if model_name == cfg["model_id"]:
            return alias, cfg

    supported = sorted(list(MODEL_REGISTRY.keys()) + [cfg["model_id"] for cfg in MODEL_REGISTRY.values()])
    raise ValueError(f"Unsupported model_name '{model_name}'. Supported values: {', '.join(supported)}")


def _entity_label(ent: Dict[str, Any], label_hint: Optional[str]) -> str:
    return str(
        ent.get(
            "label",
            ent.get(
                "entity_group",
                ent.get("entity", ent.get("class", ent.get("type", label_hint or "NamedEntity"))),
            ),
        )
    )


def _entity_score(ent: Dict[str, Any]) -> float:
    try:
        return float(ent.get("score", ent.get("confidence", ent.get("probability", 1.0))) or 0.0)
    except Exception:
        return 0.0


def _entity_start_end(ent: Dict[str, Any]) -> Tuple[int, int]:
    try:
        start = int(ent.get("start", ent.get("start_pos", ent.get("span_start", 0))) or 0)
    except Exception:
        start = 0
    try:
        end = int(ent.get("end", ent.get("end_pos", ent.get("span_end", 0))) or 0)
    except Exception:
        end = 0
    return start, end


def _flatten_raw_entities(raw: Any, label_hint: Optional[str] = None) -> Iterable[Dict[str, Any]]:
    """
    Flatten all known NER output variants into raw entity dicts.

    Supported shapes:
    - [{"text": ..., "start": ..., "end": ..., "label": ...}, ...]
    - {"entities": [{...}, ...]}
    - {"entities": {"taxon": [{...}], "location": [{...}]}}
    - {"taxon": [{...}], "location": [{...}]}  # defensive fallback
    """
    if raw is None:
        return

    if isinstance(raw, list):
        for item in raw:
            yield from _flatten_raw_entities(item, label_hint=label_hint)
        return

    if not isinstance(raw, dict):
        return

    # GLiNER2 format_results=True: {"entities": {label: [entities...]}}
    entities_value = raw.get("entities")
    if isinstance(entities_value, dict):
        for label, entities in entities_value.items():
            yield from _flatten_raw_entities(entities, label_hint=str(label))
        return

    # Alternative wrapper: {"entities": [entities...]}
    if isinstance(entities_value, list):
        yield from _flatten_raw_entities(entities_value, label_hint=label_hint)
        return

    # Normal flat entity dictionary.
    if any(key in raw for key in ("start", "end", "text", "word", "label", "entity_group", "entity")):
        ent = dict(raw)
        if label_hint and not any(k in ent for k in ("label", "entity_group", "entity", "class", "type")):
            ent["label"] = label_hint
        yield ent
        return

    # Defensive fallback for a direct label -> entities dict.
    for label, value in raw.items():
        if isinstance(value, list):
            yield from _flatten_raw_entities(value, label_hint=str(label))


def normalize_entity_output(
        sentence: str,
        entities: Any,
        model_name: str,
        model_id: str,
) -> List[Dict[str, Any]]:
    """Normalize GLiNER, GLiNER2 and HF-like entity outputs to one common format."""
    normalized: List[Dict[str, Any]] = []

    for ent in _flatten_raw_entities(entities):
        if not isinstance(ent, dict):
            continue

        start, end = _entity_start_end(ent)
        label = _entity_label(ent, None)
        score = _entity_score(ent)
        covered = str(ent.get("text", ent.get("word", "")) or "")

        if not covered and 0 <= start < end <= len(sentence):
            covered = sentence[start:end]

        if end <= start and covered:
            # Fallback for outputs that contain text but no valid span.
            idx = sentence.find(covered)
            if idx >= 0:
                start = idx
                end = idx + len(covered)

        if end <= start:
            # DUUI needs valid spans.
            continue

        normalized.append(
            {
                "text": covered,
                "label": label,
                "score": score,
                "start": start,
                "end": end,
                "model_name": model_name,
                "model_id": model_id,
            }
        )

    return normalized


def _normalize_sentence_outputs(outputs: Any, text_count: int) -> List[Any]:
    """Return exactly one raw output object per input sentence where possible."""
    if outputs is None:
        sentence_outputs: List[Any] = []
    elif isinstance(outputs, dict):
        sentence_outputs = [outputs]
    elif isinstance(outputs, list):
        # If there is one input sentence and the model returns a flat entity list,
        # keep that flat list as the single sentence output.
        if text_count == 1:
            if outputs and all(isinstance(item, dict) for item in outputs):
                if not (len(outputs) == 1 and "entities" in outputs[0]):
                    # Could be a flat list of entity dicts.
                    sentence_outputs = [outputs]
                else:
                    sentence_outputs = outputs
            else:
                sentence_outputs = [outputs]
        else:
            sentence_outputs = outputs
    else:
        sentence_outputs = []

    # Pad/truncate defensively so zip(text, sentence_outputs) cannot silently skip
    # sentences or shift later output.
    if len(sentence_outputs) < text_count:
        sentence_outputs = sentence_outputs + [None] * (text_count - len(sentence_outputs))
    elif len(sentence_outputs) > text_count:
        sentence_outputs = sentence_outputs[:text_count]

    return sentence_outputs


class NERClassificationGLiNER:
    def __init__(self, model_name: str = "urchade/gliner_multi-v2.1", device: str = "cuda"):
        from gliner import GLiNER

        self.model_name = "gliner"
        self.model_id = model_name
        self.device = device
        self.model = GLiNER.from_pretrained(model_id=model_name, map_location=device)

    def predict(
            self,
            text: List[str],
            labels: List[str],
            threshold: float = 0.5,
            batch_size: int = 8,
    ) -> List[List[Dict[str, Any]]]:
        with torch.no_grad():
            try:
                outputs = self.model.inference(
                    text,
                    labels,
                    threshold=threshold,
                    multi_label=True,
                    return_class_probs=True,
                    batch_size=batch_size,
                )
            except TypeError:
                outputs = self.model.inference(
                    text,
                    labels,
                    multi_label=True,
                    return_class_probs=True,
                    batch_size=batch_size,
                )

        sentence_outputs = _normalize_sentence_outputs(outputs, len(text))

        return [
            normalize_entity_output(sentence, entities, self.model_name, self.model_id)
            for sentence, entities in zip(text, sentence_outputs)
        ]


class NERClassificationGLiNER2:
    def __init__(self, model_name: str = "fastino/gliner2-multi-v1", device: str = "cuda"):
        from gliner2 import GLiNER2

        self.model_name = "gliner2"
        self.model_id = model_name
        self.device = device
        self.model = GLiNER2.from_pretrained(model_name)
        if hasattr(self.model, "to"):
            self.model.to(device)

    def predict(
            self,
            text: List[str],
            labels: List[str],
            threshold: float = 0.5,
            batch_size: int = 8,
    ) -> List[List[Dict[str, Any]]]:
        with torch.no_grad():
            outputs = self.model.batch_extract_entities(
                text,
                labels,
                batch_size,
                threshold=threshold,
                format_results=True,
                include_confidence=True,
                include_spans=True,
            )

        # GLiNER2 already returns one dict per input sentence when text is a list:
        #   [{"entities": {"taxon": [...], ...}}, ...]
        # Never wrap this full list again.
        sentence_outputs = _normalize_sentence_outputs(outputs, len(text))

        return [
            normalize_entity_output(sentence, sentence_output, self.model_name, self.model_id)
            for sentence, sentence_output in zip(text, sentence_outputs)
        ]


class NERClassification:
    def __init__(self, model_name: str, device: str = "cpu"):
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

        self.model_name = model_name
        self.model_id = model_name
        self.device = device

        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                add_prefix_space=True,
            )
        except TypeError:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        self.model.to(device)
        self.model.eval()

        pipe_device = 0 if str(device).startswith("cuda") else -1
        self.ner_pipeline = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=pipe_device,
        )

    def predict(
            self,
            text: List[str],
            labels: Optional[List[str]] = None,
            threshold: float = 0.5,
            batch_size: int = 8,
    ) -> List[List[Dict[str, Any]]]:
        with torch.no_grad():
            outputs = self.ner_pipeline(text, batch_size=batch_size)

        sentence_outputs = _normalize_sentence_outputs(outputs, len(text))
        return [
            normalize_entity_output(sentence, sentence_output, self.model_name, self.model_id)
            for sentence, sentence_output in zip(text, sentence_outputs)
        ]


def create_ner_classifier(
        model_name: str,
        device: str = "cpu",
        model_version: Optional[str] = None,
):
    """
    Factory used by DUUI. Returns one of the classes from the second code.

    model_version is accepted for API consistency with DUUI metadata. It is not
    passed as a HuggingFace revision here, because MODEL_VERSION is metadata in
    your Docker build setup.
    """
    alias, cfg = resolve_model_name(model_name)
    backend = cfg["backend"]
    model_id = cfg["model_id"]

    if backend == "gliner":
        classifier = NERClassificationGLiNER(model_name=model_id, device=device)
        classifier.model_name = alias
        return classifier

    if backend == "gliner2":
        classifier = NERClassificationGLiNER2(model_name=model_id, device=device)
        classifier.model_name = alias
        return classifier

    if backend == "hf_token_classification":
        classifier = NERClassification(model_name=model_id, device=device)
        classifier.model_name = alias
        return classifier

    raise ValueError(f"Unsupported backend '{backend}' for model '{model_name}'")


def predict_ner(
        model_name: str,
        texts: List[str],
        labels: Optional[List[str]] = None,
        device: str = "cpu",
        threshold: float = 0.5,
        batch_size: int = 8,
) -> List[List[Dict[str, Any]]]:
    """Convenience function for standalone use outside DUUI."""
    classifier = create_ner_classifier(model_name, device=device)
    return classifier.predict(texts, labels or [], threshold=threshold, batch_size=batch_size)


if __name__ == "__main__":
    textes = [
        "Dr. Anna Weber untersuchte für BioFID eine Streuobstwiese bei Frankfurt am Main und einen Buchenwald im Taunus.",
        "Auf der Wiese fand sie Apis mellifera, Bombus terrestris, Papilio machaon und Vanessa atalanta.",
    ]
    labels = ["person", "organization", "location", "date", "event", "product", "taxon", "other"]
    device_i = "cuda" if torch.cuda.is_available() else "cpu"

    for name in ["roberta-ner-multilingual", "wikineural-multilingual-ner"]:
        print(name)
        print(predict_ner(name, textes, labels=labels, device=device_i))