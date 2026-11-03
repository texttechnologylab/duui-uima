"""DUUI component: Stanza (Universal Dependencies) on given sentences and tokens.

POS (UPOS and the language-specific XPOS), lemmas, morphological features and UD dependencies for German and English, with the
Stanza package default_accurate. The words are written as StanzaToken, the sentences as StanzaSentence (a complete layer of its
own), so that the CAS can keep the layers of other tools.

The processing follows docs/preprocess.py (StanzaLocal) of duui-coh-metrix, the reference preprocessing of Coh-Metrix 0.4.0-py:
- Stanza runs on the given tokens (tokenize_pretokenized). Its multi-word token step does not run on given tokens, so German
  preposition + article contractions are split here with a table (zum -> zu dem). Stanza's resplit_mwt was compared on UD German
  GSD/HDT and was worse (80 wrong tokens against 9, duui-coh-metrix/docs/stanza-contraction-split/REPORT.md).
- At most 500 sentences per call, depparse_batch_size 1000.
- A sentence of more than 500 tokens is parsed in pieces of at most 500 tokens, a piece the transformer cannot take (more than
  its 512 subword positions) in halves; every piece becomes a sentence of its own.
"""
import logging
from collections import defaultdict
from functools import lru_cache
from importlib.metadata import version as package_version
from platform import python_version
from sys import version as sys_version
from threading import Lock
from time import time
from typing import List, Optional

import stanza
import torch
from cassis import load_typesystem
from cassis.cas import Utf16CodepointOffsetConverter
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings

STANZA_VERSION = package_version("stanza")

# Stanza package by document language, other languages are not supported
STANZA_PACKAGES = {
    "de": "default_accurate",
    "en": "default_accurate",
}
STANZA_PROCESSORS = "tokenize,pos,lemma,depparse"

# Sentences per Stanza call: Stanza pads its batches to the longest sentence, so the memory grows with the call size
SENTENCES_PER_CALL = 500
DEPPARSE_BATCH_SIZE = 1000
# A longer sentence is parsed in pieces of at most this many tokens (the parser's memory grows with the square of the
# length, the transformer takes at most 512 subword positions)
MAX_WORDS = 500

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Settings(BaseSettings):
    annotator_name: str
    annotator_version: str
    log_level: str

    class Config:
        env_prefix = 'duui_stanza_'


settings = Settings()

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)
logger.info("TTLab DUUI Stanza")
logger.info("Name: %s", settings.annotator_name)
logger.info("Version: %s", settings.annotator_version)

UIMA_TYPE_SENTENCE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
UIMA_TYPE_TOKEN = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"
UIMA_TYPE_STANZA_SENTENCE = "org.texttechnologylab.uima.type.stanza.StanzaSentence"
UIMA_TYPE_STANZA_TOKEN = "org.texttechnologylab.uima.type.stanza.StanzaToken"
UIMA_TYPE_TOKEN_FORM = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.TokenForm"
UIMA_TYPE_POS = "de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS"
UIMA_TYPE_LEMMA = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma"
UIMA_TYPE_MORPH = "de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.morph.MorphologicalFeatures"
UIMA_TYPE_DEPENDENCY = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency"
UIMA_TYPE_ROOT = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.ROOT"


class Span(BaseModel):
    begin: int
    end: int


class TextImagerRequest(BaseModel):
    text: str
    len: int
    lang: str
    sentences: List[Span]
    tokens: List[Span]


class Word(BaseModel):
    begin: int
    end: int
    # the word if it differs from the covered text (the words of a split contraction), else None
    form: Optional[str] = None
    lemma: Optional[str] = None
    upos: Optional[str] = None
    xpos: Optional[str] = None
    # UD feature string, e.g. "Case=Nom|Number=Sing|Number[psor]=Plur"
    feats: Optional[str] = None
    deprel: str
    # index of the head in the words of this sentence, None for the root
    head: Optional[int] = None


class StanzaSentence(BaseModel):
    begin: int
    end: int
    words: List[Word]


class AnnotationMeta(BaseModel):
    name: str
    version: str
    modelName: str
    modelVersion: str


class DocumentModification(BaseModel):
    user: str
    timestamp: int
    comment: str


class TextImagerResponse(BaseModel):
    sentences: List[StanzaSentence]
    meta: AnnotationMeta
    modification_meta: DocumentModification


class TextImagerCapability(BaseModel):
    supported_languages: List[str]
    reproducible: bool


class TextImagerDocumentation(BaseModel):
    annotator_name: str
    version: str
    implementation_lang: Optional[str]
    meta: Optional[dict]
    docker_container_id: Optional[str]
    parameters: Optional[dict]
    capability: TextImagerCapability
    implementation_specific: Optional[str]


class TextImagerInputOutput(BaseModel):
    inputs: List[str]
    outputs: List[str]


with open("TypeSystem.xml", "rb") as f:
    typesystem_xml_content = load_typesystem(f).to_xml().encode("utf-8")

with open("communication.lua", "rb") as f:
    lua_communication_script = f.read().decode("utf-8")

app = FastAPI(
    title=settings.annotator_name,
    description="Stanza (Universal Dependencies) for TTLab DUUI",
    version=settings.annotator_version,
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "TTLab Team - Daniel Baumartz",
        "url": "https://texttechnologylab.org",
        "email": "baumartz@em.uni-frankfurt.de",
    },
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)

# a Stanza pipeline is not shared between concurrent requests
pipeline_lock = Lock()


@lru_cache(maxsize=None)
def get_pipeline(lang: str) -> stanza.Pipeline:
    logger.info("Loading Stanza \"%s/%s\" on %s", lang, STANZA_PACKAGES[lang], DEVICE)
    # the models were downloaded when the image was built, the container runs offline
    return stanza.Pipeline(lang, processors=STANZA_PROCESSORS, package=STANZA_PACKAGES[lang],
                           tokenize_pretokenized=True, download_method=None, depparse_batch_size=DEPPARSE_BATCH_SIZE,
                           use_gpu=DEVICE == "cuda", logging_level="WARN")


def utf16_to_utf8(text):
    # same as in duui-spacy: DUUI <= 1.5.7 sends characters outside the BMP as two lone surrogates, joined here
    clean_text = text.encode('utf-16', 'surrogatepass').decode('utf-16', 'surrogateescape')
    return clean_text


# German preposition + article contractions (STTS APPRART), split into the two words UD German writes ("zum" -> "zu dem").
# From duui-coh-metrix docs/preprocess.py (_DE_CONTRACTIONS): the expansions of UD German GSD and HDT (each of their 16 forms
# has one expansion) plus hinterm, hinters, überm, vors (STTS). Only all-lower or capitalised tokens ("Im" -> "In dem", not
# "IM"). "am" before a superlative stays one word (STTS PTKA, "am besten", as in HDT): next word lower case in -sten and no
# capitalised word after it.
DE_CONTRACTIONS = {
    "am": ("an", "dem"), "ans": ("an", "das"), "aufs": ("auf", "das"), "beim": ("bei", "dem"), "durchs": ("durch", "das"),
    "fürs": ("für", "das"), "hinterm": ("hinter", "dem"), "hinters": ("hinter", "das"), "im": ("in", "dem"), "ins": ("in", "das"),
    "übers": ("über", "das"), "überm": ("über", "dem"), "ums": ("um", "das"), "unterm": ("unter", "dem"),
    "unters": ("unter", "das"), "vom": ("von", "dem"), "vorm": ("vor", "dem"), "vors": ("vor", "das"), "zum": ("zu", "dem"),
    "zur": ("zu", "der"),
}


def split_contractions(words: list) -> list:
    """[(word, token index)] for the words of one sentence's tokens (preprocess.py split_contractions)"""
    out = []
    for i, w in enumerate(words):
        low = w.lower()
        parts = DE_CONTRACTIONS.get(low) if w in (low, low.capitalize()) else None
        if parts and low == "am":
            nxt, after = (words[i + 1] if i + 1 < len(words) else ""), (words[i + 2] if i + 2 < len(words) else "")
            if nxt[:1].islower() and nxt.endswith("sten") and not after[:1].isupper():
                parts = None
        if parts:
            out.extend([(parts[0].capitalize() if w[0].isupper() else parts[0], i), (parts[1], i)])
        else:
            out.append((w, i))
    return out


def pieces(toks: list, words: list, size: int) -> list:
    """a sentence's tokens and words (word, token index) in consecutive pieces of at most `size` tokens; the words of a split
    contraction stay with their token (preprocess.py _pieces)"""
    out = []
    for g in range(0, len(toks), size):
        out.append((toks[g:g + size], [(w, ti - g) for w, ti in words if g <= ti < g + size]))
    return out


def parse(nlp: stanza.Pipeline, items: list) -> list:
    """[(tokens, words, span)] -> [(tokens, words, span, Stanza sentence)]; a sentence the transformer cannot take (more than
    its 512 subword positions) is halved until its pieces fit (preprocess.py StanzaLocal._parse)"""
    try:
        parsed = nlp([[w for w, _ in words] for _, words, _ in items])
        return [(toks, words, span, s) for (toks, words, span), s in zip(items, parsed.sentences, strict=True)]
    except RuntimeError as e:
        if "existing size (512)" not in str(e):
            raise
    if len(items) > 1:
        return [x for item in items for x in parse(nlp, [item])]
    toks, words, span = items[0]
    if len(toks) < 2:
        raise RuntimeError(f"a single token at {span} is longer than the parser's 512 positions")
    return [x for pt, pw in pieces(toks, words, (len(toks) + 1) // 2) for x in parse(nlp, [(pt, pw, span)])]


@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return lua_communication_script


@app.get("/v1/documentation")
def get_documentation() -> TextImagerDocumentation:
    return TextImagerDocumentation(
        annotator_name=settings.annotator_name,
        version=settings.annotator_version,
        implementation_lang="Python",
        meta={
            "python_version": python_version(),
            "python_version_full": sys_version,
            "stanza_version": STANZA_VERSION,
            "stanza_packages": STANZA_PACKAGES,
            "stanza_processors": STANZA_PROCESSORS,
            "sentences_per_call": SENTENCES_PER_CALL,
            "depparse_batch_size": DEPPARSE_BATCH_SIZE,
            "max_words": MAX_WORDS,
            "contractions": "German: split with a table (DE_CONTRACTIONS); English: as tokenized",
            "device": DEVICE,
        },
        docker_container_id="[TODO]",
        parameters={},
        capability=TextImagerCapability(supported_languages=sorted(STANZA_PACKAGES), reproducible=True),
        implementation_specific=None,
    )


@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    return Response(content=typesystem_xml_content, media_type="application/xml")


@app.get("/v1/details/input_output")
def get_input_output() -> TextImagerInputOutput:
    return TextImagerInputOutput(
        inputs=[UIMA_TYPE_SENTENCE, UIMA_TYPE_TOKEN],
        outputs=[UIMA_TYPE_STANZA_SENTENCE, UIMA_TYPE_STANZA_TOKEN, UIMA_TYPE_TOKEN_FORM, UIMA_TYPE_POS, UIMA_TYPE_LEMMA,
                 UIMA_TYPE_MORPH, UIMA_TYPE_DEPENDENCY, UIMA_TYPE_ROOT],
    )


@app.post("/v1/process")
def post_process(request: TextImagerRequest) -> TextImagerResponse:
    modification_timestamp_seconds = int(time())

    if request.lang not in STANZA_PACKAGES:
        raise HTTPException(status_code=400, detail=f"Language \"{request.lang}\" is not supported, only: {', '.join(sorted(STANZA_PACKAGES))}")
    if request.tokens and not request.sentences:
        raise HTTPException(status_code=400, detail="Sentence annotations are required (the component parses given sentences)")

    # CAS offsets are UTF-16 code units, Python strings use code points
    text = utf16_to_utf8(request.text)
    utf16_converter = Utf16CodepointOffsetConverter()
    utf16_converter.create_offset_mapping(text)
    to_py = utf16_converter.external_to_python
    to_cas = utf16_converter.python_to_external

    # the tokens of each sentence: those inside its span, in text order (a token crossing a sentence boundary is in none)
    tokens = sorted(((to_py(t.begin), to_py(t.end)) for t in request.tokens), key=lambda t: (t[0], -t[1]))
    items, j = [], 0
    for sentence in sorted(request.sentences, key=lambda s: (s.begin, -s.end)):
        sb, se = to_py(sentence.begin), to_py(sentence.end)
        while j < len(tokens) and tokens[j][0] < sb:
            j += 1
        toks = []
        while j < len(tokens) and tokens[j][1] <= se:
            toks.append(tokens[j])
            j += 1
        if not toks:
            continue
        texts = [text[b:e] for b, e in toks]
        words = split_contractions(texts) if request.lang == "de" else [(w, i) for i, w in enumerate(texts)]
        # a sentence longer than MAX_WORDS tokens is cut into pieces, each parsed (and written) as a sentence of its own
        items.extend((pt, pw, (sb, se)) for pt, pw in pieces(toks, words, MAX_WORDS))

    parsed = []
    with pipeline_lock:
        nlp = get_pipeline(request.lang)
        for k in range(0, len(items), SENTENCES_PER_CALL):
            parsed.extend(parse(nlp, items[k:k + SENTENCES_PER_CALL]))

    # the pieces of each given sentence, in order: the first keeps the sentence's begin, the last its end
    groups = defaultdict(list)
    for toks, words, span, s in parsed:
        groups[span].append((toks, words, s))

    sentences = []
    for span in sorted(groups):
        group = groups[span]
        if len(group) > 1:
            logger.warning("A sentence of %d tokens at %d-%d was parsed in %d pieces, each written as a sentence",
                           sum(len(g[0]) for g in group), span[0], span[1], len(group))
        for i, (toks, words, s) in enumerate(group):
            if [w.text for w in s.words] != [w for w, _ in words]:
                raise RuntimeError(f"Stanza changed the words of the sentence at {span}")
            out_words = []
            for (word, ti), w in zip(words, s.words):
                b, e = toks[ti]
                out_words.append(Word(
                    begin=to_cas(b),
                    end=to_cas(e),
                    form=word if word != text[b:e] else None,
                    lemma=w.lemma,
                    upos=w.upos,
                    xpos=w.xpos,
                    feats=w.feats,
                    deprel=w.deprel,
                    head=None if w.head == 0 else w.head - 1,
                ))
            begin = span[0] if i == 0 else toks[0][0]
            end = span[1] if i == len(group) - 1 else toks[-1][1]
            sentences.append(StanzaSentence(begin=to_cas(begin), end=to_cas(end), words=out_words))

    package = STANZA_PACKAGES[request.lang]
    logger.info("%d sentences, %d words, %d seconds", len(sentences), sum(len(s.words) for s in sentences),
                int(time()) - modification_timestamp_seconds)

    return TextImagerResponse(
        sentences=sentences,
        meta=AnnotationMeta(
            name=settings.annotator_name,
            version=settings.annotator_version,
            modelName=f"Stanza {request.lang}/{package}",
            modelVersion=STANZA_VERSION,
        ),
        modification_meta=DocumentModification(
            user=settings.annotator_name,
            timestamp=modification_timestamp_seconds,
            comment=f"{settings.annotator_name} ({settings.annotator_version}), Stanza ({STANZA_VERSION}) {request.lang}/{package}",
        ),
    )
