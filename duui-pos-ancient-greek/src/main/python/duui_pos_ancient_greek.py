"""
POS tagger for Ancient Greek, built as a DUUI component for the
TextImager pipeline. Uses a fine-tuned XLM-RoBERTa model (trained
on the UD Perseus treebank) to tag tokens with Universal POS labels.

I did not write this from nothing. The DUUI boilerplate: the endpoint
structure, the Lua communication layer, the typesystem handshake,
is borrowed heavily from the existing TTLab components, especially
the Flair POS tagger and the spaCy sentencizer . I studied those to
understand how DUUI components are supposed to be wired up, then
adapted the skeleton for my own model.

The actual inference logic (tokenisation, subword-to-word alignment)
was written with a lot of help from GitHub Copilot and several rounds
of asking ChatGPT "why is word_ids() returning None for special tokens."

Last meaningful edit: Feb 2026
"""

import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from transformers import AutoModelForTokenClassification, AutoTokenizer

# -- Config --
# BORROWED. This env-var-based config pattern is straight from the
# DUUI emotion and sentiment components.
# I liked it better than the pydantic BaseSettings approach used in
# the spaCy sentencizer, mostly because I don't fully understand how
# pydantic settings auto-loads from env vars and I didn't want to
# debug that on top of everything else.

ANNOTATOR_NAME = os.environ.get(
    "DUUI_POS_AG_ANNOTATOR_NAME", "duui-pos-ancient-greek"
)
ANNOTATOR_VERSION = os.environ.get(
    "DUUI_POS_AG_ANNOTATOR_VERSION", "0.1.0"
)
LOG_LEVEL = os.environ.get("DUUI_POS_AG_LOG_LEVEL", "DEBUG")
MODEL_PATH = os.environ.get(
    "DUUI_POS_AG_MODEL_PATH", "qbnguyen/ancient-greek-pos-xlmr"
)

COMPONENT_ROOT = Path(__file__).parent

logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

# BORROWED. The Flair POS component and the spaCy sentencizer both have
# this. I assume it's for performance but honestly I just copied the
# pattern because it seemed like the right thing to do.
_TYPESYSTEM_XML = (COMPONENT_ROOT / "TypeSystemPOS.xml").read_text("utf-8")
_LUA_SCRIPT = (
    (COMPONENT_ROOT / "duui_pos_ancient_greek.lua").read_text("utf-8")
)

# Punctuation pattern for the tokenizer.
# FRAGILE. I assembled this character class myself by looking at what
# shows up in my Ancient Greek test corpus. The middle dot (·) and the
# Greek question mark (;) are the ones that kept tripping me up.
# There are probably more punctuation marks in Unicode Greek ranges
# that I'm missing. If tokens start looking wrong, check here first.
_PUNCT = r"""[,.:;!?·;()\[\]«»\u201c\u201d\u2018\u2019]+"""

# -- Schemas --
# BORROWED. The request/response schema pattern comes from the DUUI
# components. The Flair tagger uses DkproSentence / DkproPos, the
# emotion component uses UimaSentence, etc. I renamed things to match
# what my component actually does but the shape is the same.
#
# I asked ChatGPT: "what is the difference between a Pydantic BaseModel
# and a regular dataclass" and the answer was helpful enough that I
# stopped worrying and just used BaseModel like everyone else.


class Sentence(BaseModel):
    begin: int
    end: int
    text: str


class PosRequest(BaseModel):
    doc_text: str
    doc_len: int
    lang: str = "grc"
    model_name: Optional[str] = None
    sentences: Optional[list[Sentence]] = None


class TokenPOS(BaseModel):
    begin: int
    end: int
    pos_value: str
    pos_coarse_value: str


class PosResponse(BaseModel):
    tokens: list[TokenPOS]
    model_name: str
    model_version: str
    model_source: str
    model_lang: str
    errors: list[str]


# -- Model loading --

# BORROWED. The lru_cache trick for model loading appears in every
# single DUUI component I looked at. The Flair tagger has a
# configurable cache size, the emotion component uses a lock + cache
# combo. I went with the simplest version: cache one model, no lock.
#
# REVISIT. The emotion and sentiment components use a threading Lock
# around model loading/inference. I'm not doing that because I only
# run one worker (see uvicorn config at the bottom), but if I ever
# scale this up I'll need to add locking. I only half-understand why
# concurrent access to a pytorch model is dangerous.

@lru_cache(maxsize=1)
def load_model(model_path: str):
    logger.info("Loading model from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()
    logger.info("Model loaded successfully on CPU")
    return model, tokenizer


# -- Tokenisation

# SOLID (mostly). I wrote this function myself. It's the part I
# understand best because it's basically text processing, which is
# closer to my wheelhouse than the ML inference stuff.
#
# The idea: split on whitespace, then peel off leading and trailing
# punctuation as separate tokens. I need character offsets because
# DUUI maps annotations back onto the original document by position.
#
# I went through about four versions of this. First attempt used
# spaCy's tokenizer but it was overkill and slow for Ancient Greek.
# Second attempt was a naive whitespace split that broke on «εἶπεν»
# because the guillemets stayed glued to the word. Current version
# handles that.
#
# The _append helper inside the function is a pattern Copilot
# suggested when I kept repeating the dict construction. I would
# have just written it out each time, but this is cleaner.

def tokenize_raw_text(text: str) -> list[dict]:
    """Split *text* into word tokens with character offsets,
    separating leading/trailing punctuation."""
    tokens: list[dict] = []

    def _append(form: str, start: int):
        tokens.append({"form": form, "begin": start, "end": start + len(form)})

    for m in re.finditer(r"\S+", text):
        word, ws = m.group(), m.start()

        # peel off leading punctuation — «, (, [, etc.
        lead = re.match(f"^({_PUNCT})", word)
        if lead:
            _append(lead.group(1), ws)
            ws += lead.end()
            word = word[lead.end() :]
        if not word:
            continue

        # peel off trailing punctuation — same idea, from the right
        trail = re.search(f"({_PUNCT})$", word)
        trail_tok = None
        if trail:
            trail_tok = (trail.group(1), ws + trail.start())
            word = word[: trail.start()]

        if word:
            _append(word, ws)
        if trail_tok:
            _append(*trail_tok)

    return tokens


# -- POS inference --

# COPILOT wrote the first draft of this function. My prompt was
# roughly: "given a list of pre-tokenized words, run them through
# a HuggingFace token classification model and map subword predictions
# back to the original words using word_ids()"
#
# I then rewrote parts of it after spending a long time reading:
# https://huggingface.co/docs/transformers/tasks/token_classification
# and this Stack Overflow answer about word_ids() alignment:
# https://stackoverflow.com/a/75903065
#
# The key thing I learned (from ChatGPT, after staring at wrong output
# for two hours): when you pass is_split_into_words=True, the tokenizer
# may split a single word into multiple subword tokens. word_ids()
# tells you which original word each subword belongs to. We only want
# the prediction for the *first* subword of each word. That's what
# the `seen` set is for. I understand this now but I would not have
# figured it out without help.
#
# FRAGILE. The max_length=256 truncation means very long sentences
# will lose tokens at the end silently. My corpus doesn't have
# sentences that long, but if yours does, raise this. I don't know
# what the actual max is for XLM-RoBERTa.

def predict_pos(
    text: str, offset: int, model, tokenizer
) -> list[TokenPOS]:
    if not text or not text.strip():
        return []

    word_tokens = tokenize_raw_text(text)
    if not word_tokens:
        return []

    words = [t["form"] for t in word_tokens]

    # Tokenize with the model's subword tokenizer.
    # is_split_into_words=True tells it we already split on whitespace.
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )

    # Run inference: no gradient computation needed, we're just predicting
    with torch.no_grad():
        logits = model(**encoding).logits

    # argmax gives us the most likely label index for each subword token
    preds = torch.argmax(logits, dim=-1)[0].tolist()
    word_ids = encoding.word_ids()
    id2label = model.config.id2label

    # Map subword predictions back to our original word tokens.
    # We only take the first subword's prediction for each word.
    # COPILOT. This loop structure is mostly Copilot's. I added the
    # offset arithmetic to make the character positions absolute
    # (relative to the full document, not just this sentence).
    results: list[TokenPOS] = []
    seen: set[int] = set()
    for sw_idx, wid in enumerate(word_ids):
        if wid is None or wid in seen:
            continue
        seen.add(wid)
        tok = word_tokens[wid]
        label = id2label[preds[sw_idx]]
        results.append(
            TokenPOS(
                begin=tok["begin"] + offset,
                end=tok["end"] + offset,
                pos_value=label,
                # I'm setting coarse and fine to the same value because the
                # model only outputs Universal POS tags. The DUUI type system
                # expects both fields. The Flair POS component leaves
                # coarse_value empty (""), but I figured identical values
                # are more informative than blank.
                pos_coarse_value=label,
            )
        )
    return results


# -- Helpers --

# SOLID. Just bundles the response. Nothing clever happening here.
def _make_response(
    tokens: list[TokenPOS],
    model_path: str,
    errors: list[str],
) -> PosResponse:
    return PosResponse(
        tokens=tokens,
        model_name=model_path,
        model_version=ANNOTATOR_VERSION,
        model_source=model_path,
        model_lang="grc",
        errors=errors,
    )


# -- FastAPI --
# BORROWED. The endpoint structure is required by the DUUI protocol.
# Every DUUI component follows this pattern. I copied the skeleton from
# the Flair POS tagger and the spaCy sentencizer, then filled in my
# own details.

app = FastAPI(
    title=ANNOTATOR_NAME,
    version=ANNOTATOR_VERSION,
    description="DUUI component for Ancient Greek POS tagging",
)


# Returns the UIMA type system XML.
# The Flair component returns this with media_type="application/xml",
# but I'm using PlainTextResponse like the simpler components do.
@app.get("/v1/typesystem", response_class=PlainTextResponse)
def get_typesystem():
    return _TYPESYSTEM_XML


@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer():
    return _LUA_SCRIPT


# BORROWED. The documentation endpoint structure is adapted from the
# Flair POS component. The Flair version has a proper TextImagerDocumentation
# Pydantic model with a capabilities field. The spaCy version does too.
# I simplified mine to a plain dict because the emotion component by
# Bagci literally just returns the string "Test" for this endpoint and
# apparently that's fine? So I figured a real dict is already an
# improvement.
#
# REVISIT. Should probably add a TextImagerCapability model like the
# Flair and spaCy components do. Right now this is just a dict.
@app.get("/v1/documentation")
def get_documentation():
    return {
        "annotator_name": ANNOTATOR_NAME,
        "version": ANNOTATOR_VERSION,
        "implementation_lang": "Python",
        "meta": {
            "description": (
                "Part-of-Speech tagger for Ancient Greek using a "
                "fine-tuned XLM-RoBERTa model on UD Perseus treebank."
            ),
            "language": "grc",
            "model": "xlm-roberta-base (fine-tuned)",
            "training_data": "UD_Ancient_Greek-Perseus",
            "tagset": "Universal POS (17 tags)",
        },
        "parameters": {
            "model_name": {
                "type": "string",
                "description": "Path or HF Hub ID for the model",
                "default": MODEL_PATH,
            }
        },
    }


# BORROWED. From the Flair POS component. Maps DUUI input/output types
# so the Java pipeline knows what annotations this component reads and
# produces.
@app.get("/v1/details/input_output")
def get_input_output():
    return {
        "inputs": [
            "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
        ],
        "outputs": [
            "de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS"
        ],
    }


# BORROWED. The overall structure (try to load model, iterate over
# sentences, collect results, catch exceptions into an error list)
# is modelled on the emotion and sentiment components.
# Those components process "selections" of sentences; mine is simpler
# because I only handle one selection type (sentences).
#
# The fallback path (when no sentences are provided) splits on newlines.
# I added this because during testing I kept sending raw text without
# pre-segmented sentences and getting empty results back. Took me
# embarrassingly long to realise the sentence list was just empty.
#
# FRAGILE. The newline fallback uses `cur += len(line) + 1` to track
# character offsets. The +1 is for the newline character itself. This
# will be wrong if the document uses \r\n line endings. I don't think
# Ancient Greek corpora have that problem but I've been wrong before.
@app.post("/v1/process", response_model=PosResponse)
def process(request: PosRequest):
    model_path = request.model_name or MODEL_PATH

    try:
        model, tokenizer = load_model(model_path)
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        return _make_response([], model_path, [f"Model load error: {e}"])

    all_tokens: list[TokenPOS] = []
    errors: list[str] = []

    try:
        if request.sentences:
            for sent in request.sentences:
                all_tokens.extend(
                    predict_pos(sent.text, sent.begin, model, tokenizer)
                )
        else:
            # No pre-segmented sentences, fall back to line-by-line.
            # Not ideal but better than returning nothing.
            cur = 0
            for line in request.doc_text.split("\n"):
                if line.strip():
                    all_tokens.extend(
                        predict_pos(line, cur, model, tokenizer)
                    )
                cur += len(line) + 1  # +1 for the newline character
    except Exception as e:
        logger.error("Inference error: %s", e, exc_info=True)
        errors.append(f"Inference error: {e}")

    return _make_response(all_tokens, model_path, errors)


# -- Entry point --
# SOLID. Standard uvicorn startup. workers=1 because I don't want to
# deal with concurrent model access. Port 9714 was chosen arbitrarily.
# The otehr DUUI components each seem to pick their own port and I just
# made sure mine didn't collide with any of the ones I saw in their
# docker-compose files.

if __name__ == "__main__":
    uvicorn.run(
        "duui_pos_ancient_greek:app",
        host="0.0.0.0",
        port=9714,
        workers=1,
    )