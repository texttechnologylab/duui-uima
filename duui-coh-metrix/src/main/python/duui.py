import csv
import logging
import pyphen
import textstat
import math

from platform import python_version
from sys import version as sys_version
from time import time
from typing import List, Optional, Dict, Tuple, Any

from cassis import load_typesystem
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, validator
from pydantic_settings import BaseSettings
from lexicalrichness import LexicalRichness
from lexical_diversity import lex_div as ld
from similarity.normalized_levenshtein import NormalizedLevenshtein
from itertools import combinations
from collections import defaultdict, Counter
from nltk.corpus import wordnet as wn
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from germanetpy.germanet import Germanet
from germanetpy.synset import WordCategory
from pathlib import Path

import numpy as np

class Settings(BaseSettings):
    annotator_name: str
    annotator_version: str
    log_level: str
    germanet_path: Optional[str] = None

    class Config:
        env_prefix = 'duui_coh_metrix_'


settings = Settings()

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)
logger.info("TTLab TextImager DUUI Coh-Metrix")
logger.info("Name: %s", settings.annotator_name)
logger.info("Version: %s", settings.annotator_version)

TEXTIMAGER_ANNOTATOR_OUTPUT_TYPES = [
]

TEXTIMAGER_ANNOTATOR_INPUT_TYPES = [
    "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence",
    "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token",
    "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma",
]

SUPPORTED_LANGS = [
    # all
]

class Token(BaseModel):
    begin: int
    end: int
    text: str
    pos_value: str   # spacy tag_ -> language specific https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html https://homepage.ruhr-uni-bochum.de/stephen.berman/Korpuslinguistik/Tagsets-STTS.html
    pos_coarse: str  # spacy pos_ -> Universal Dependencies https://universaldependencies.org/u/pos/index.html
    lemma: str
    is_alpha: bool
    is_punct: bool
    dep_type: str
    morph_person: Optional[str] = ""
    morph_number: Optional[str] = ""
    morph_tense: Optional[str] = ""
    vector: Optional[List[float]] = None
    has_vector: bool


class NounChunk(BaseModel):
    begin: int
    end: int


class Sentence(BaseModel):
    begin: int
    end: int
    text: str
    tokens: List[Token]


class Paragraph(BaseModel):
    begin: int
    end: int
    text: str
    sentences: List[Sentence]


class TextImagerRequest(BaseModel):
    language: str
    text: str
    paragraphs: List[Paragraph]
    noun_chunks: List[NounChunk]


class AnnotationMeta(BaseModel):
    name: str
    version: str
    modelName: str
    modelVersion: str


class DocumentModification(BaseModel):
    user: str
    timestamp: int
    comment: str


class Index(BaseModel):
    index: int
    type_name: str
    label_ttlab: Optional[str] = None
    label_v3: Optional[str] = None
    label_v2: Optional[str] = None
    description: str
    value: Optional[float]  # can be None if not applicable or on error
    error: Optional[str]    # fill with error message if applicable
    version: Optional[str] = None

    @validator('value')
    def value_must_be_finite(cls, v):
        if v is not None and (math.isinf(v) or math.isnan(v)):
            print("Validating value:", v)
            return None
        return v


class TextImagerResponse(BaseModel):
    indices: List[Index]
    meta: Optional[AnnotationMeta]
    modification_meta: Optional[DocumentModification]


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


typesystem_filename = 'src/main/resources/TypeSystem.xml'
logger.debug("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    typesystem_xml_content = typesystem.to_xml().encode("utf-8")
    logger.debug("Base typesystem:")
    logger.debug(typesystem_xml_content)

lua_communication_script_filename = "src/main/lua/communication.lua"
logger.debug("Loading Lua communication script from \"%s\"", lua_communication_script_filename)
with open(lua_communication_script_filename, 'rb') as f:
    lua_communication_script = f.read().decode("utf-8")
    logger.debug("Lua communication script:")
    logger.debug(lua_communication_script_filename)

if settings.germanet_path:
    gnp = Path(settings.germanet_path)
    if gnp.is_dir() and any(gnp.iterdir()):
        logger.info("Loading GermaNet from \"%s\"", settings.germanet_path)
        germanet = Germanet(settings.germanet_path)
    else:
        logger.warning("GermaNet path defined as \"%s\", but empty or non-existing. Metrics based on GermaNet will return -1", settings.germanet_path)
        germanet = None
else:
    logger.warning("No GermaNet path defined. Metrics based on GermaNet will return -1")
    germanet = None

app = FastAPI(
    title=settings.annotator_name,
    description="TTLab TextImager DUUI Coh-Metrix",
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


@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return lua_communication_script


@app.get("/v1/documentation")
def get_documentation() -> TextImagerDocumentation:
    capabilities = TextImagerCapability(
        supported_languages=SUPPORTED_LANGS,
        reproducible=True
    )

    documentation = TextImagerDocumentation(
        annotator_name=settings.annotator_name,
        version=settings.annotator_version,
        implementation_lang="Python",
        meta={
            "python_version": python_version(),
            "python_version_full": sys_version,
        },
        docker_container_id="[TODO]",
        parameters={},
        capability=capabilities,
        implementation_specific=None,
    )

    return documentation


@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    return Response(
        content=typesystem_xml_content,
        media_type="application/xml"
    )


@app.get("/v1/details/input_output")
def get_input_output() -> TextImagerInputOutput:
    return TextImagerInputOutput(
        inputs=TEXTIMAGER_ANNOTATOR_INPUT_TYPES,
        outputs=TEXTIMAGER_ANNOTATOR_OUTPUT_TYPES
    )

# ============================================================================
# DESCRIPTIVE (DES*) — "How big is the text?"
# Basic counts: paragraphs, sentences, words, syllables, letters, plus means
# and standard deviations. These are the most reliable Coh-Metrix outputs;
# they are just counts over spaCy tokens.
# ============================================================================

# LAY: How many paragraphs are in the text?
# Reliable.
def cm_despc(paragraphs: List[Paragraph]) -> Optional[float]:
    # Paragraph count, number of paragraphs
    return len(paragraphs)

# LAY: How many sentences are in the text?
# Reliable.
def cm_dessc(paragraphs: List[Paragraph]) -> Optional[float]:
    # Sentence count, number of sentences
    return sum([len(p.sentences) for p in paragraphs])

# LAY: How many words (excluding punctuation) are in the text?
# Reliable.
def cm_deswc(paragraphs: List[Paragraph]) -> Optional[float]:
    # Word count, number of words.
    # M6 fix: exclude punctuation to match _count_metrics["total_tokens"] and
    # spec ("words taken from leaves of the sentence parse trees").
    return sum(1 for p in paragraphs for s in p.sentences for t in s.tokens if not t.is_punct)

# LAY: On average, how many sentences are in one paragraph?
# ↑ Higher = longer paragraphs. Reliable.
def cm_despl(paragraphs: List[Paragraph]) -> Optional[float]:
    # Paragraph length, number of sentences, mean
    return np.mean([len(p.sentences) for p in paragraphs])

# LAY: How uneven are paragraph lengths? (spread around the mean)
# ↑ Higher = more variable paragraph sizes. Reliable.
def cm_despld(paragraphs: List[Paragraph]) -> Optional[float]:
    # Paragraph length, number of sentences, standard deviation
    return np.std([len(p.sentences) for p in paragraphs])

# LAY: On average, how many words are in one sentence?
# ↑ Higher = longer sentences (often harder to read). Reliable.
def cm_dessl(paragraphs: List[Paragraph]) -> Optional[float]:
    # Sentence length, number of words, mean. M6 fix: exclude punctuation.
    return np.mean([sum(1 for t in s.tokens if not t.is_punct)
                    for p in paragraphs for s in p.sentences])

# LAY: How uneven are sentence lengths?
# ↑ Higher = more variable sentence sizes. Reliable.
def cm_dessld(paragraphs: List[Paragraph]) -> Optional[float]:
    # Sentence length, number of words, standard deviation. M6 fix: exclude punctuation.
    return np.std([sum(1 for t in s.tokens if not t.is_punct)
                   for p in paragraphs for s in p.sentences])

pyphens = {
    "en": pyphen.Pyphen(lang='en'),
    "de": pyphen.Pyphen(lang='de'),
}

def _syllables_count(tokens: List[Token], lang: str) -> List[int]:
    syllables_counts = [
        len(pyphens[lang].positions(token.text))+1
        for token in tokens
    ]
    return syllables_counts

# LAY: On average, how many syllables per word?
# ↑ Higher = longer, more complex words. Reliable.
def cm_deswlsy(tokens: List[Token], lang: str) -> Optional[float]:
    # Word length, number of syllables, mean
    return np.mean(_syllables_count(tokens, lang))

# LAY: How uneven is the syllable count across words?
# ↑ Higher = mix of short and long words. Reliable.
def cm_deswlsyd(tokens: List[Token], lang: str) -> Optional[float]:
    # Word length, number of syllables, standard deviation
    return np.std(_syllables_count(tokens, lang))

# LAY: On average, how many letters per word?
# ↑ Higher = longer words. Reliable.
def cm_deswllt(paragraphs: List[Paragraph]) -> Optional[float]:
    # Word length, number of letters, mean
    text_letters = []
    for p in paragraphs:
        for s in p.sentences:
            for t in s.tokens:
                text_letters.append(len(''.join(c for c in t.text if c.isalpha())))
    return np.mean(text_letters)

# LAY: How uneven is the letter count across words?
# ↑ Higher = mix of short and long words. Reliable.
def cm_deswlltd(paragraphs: List[Paragraph]) -> Optional[float]:
    # Word length, number of letters, standard deviation
    text_letters = []
    for p in paragraphs:
        for s in p.sentences:
            for t in s.tokens:
                text_letters.append(len(''.join(c for c in t.text if c.isalpha())))
    return np.std(text_letters)

ud_noun_pos = {"NOUN", "PROPN"}
# H6 fix: DET was previously in this set which caused pronoun-overlap (and thus
# argument-overlap) to approach 1.0 for any normal text because determiners
# like "the"/"der"/"die"/"das" appear in nearly every sentence. Spec (Ch. 4,
# §Referential cohesion) specifies pronouns ("he/he"), not determiners.
ud_pronouns_pos = {"PRON"}
ud_content_pos = {"NOUN", "VERB", "ADJ", "ADV"}

def _noun_overlap(sentence_a: Sentence, sentence_b: Sentence) -> int:
    nouns_a = set([t.text for t in sentence_a.tokens if t.pos_coarse and t.pos_coarse in ud_noun_pos])
    nouns_b = set([t.text for t in sentence_b.tokens if t.pos_coarse and t.pos_coarse in ud_noun_pos])
    return len(nouns_a.intersection(nouns_b))

def _argument_overlap(sentence_a: Sentence, sentence_b: Sentence) -> int:
    nouns_a = set([t.lemma for t in sentence_a.tokens if t.pos_coarse and t.pos_coarse in ud_noun_pos])
    nouns_b = set([t.lemma for t in sentence_b.tokens if t.pos_coarse and t.pos_coarse in ud_noun_pos])
    noun_overlap = len(nouns_a.intersection(nouns_b))

    promouns_a = set([t.text for t in sentence_a.tokens if t.pos_coarse and t.pos_coarse in ud_pronouns_pos])
    promouns_b = set([t.text for t in sentence_b.tokens if t.pos_coarse and t.pos_coarse in ud_pronouns_pos])
    pronoun_overlap = len(promouns_a.intersection(promouns_b))

    return noun_overlap + pronoun_overlap

def _stem_overlap(sentence_nouns: Sentence, sentence_contents: Sentence) -> int:
    nouns_a = set([t.lemma for t in sentence_nouns.tokens if t.pos_coarse and t.pos_coarse in ud_noun_pos])
    nouns_b = set([t.lemma for t in sentence_contents.tokens if t.pos_coarse and t.pos_coarse in ud_content_pos])
    return len(nouns_a.intersection(nouns_b))

def _word_overlap(sentence_a: Sentence, sentence_b: Sentence) -> float:
    # H5 fix: denominator must be the union of CONTENT-word lemmas only, not all
    # tokens (function words, determiners, punctuation were previously included,
    # which deflated overlap ratios). Spec: Ch. 4 §Referential cohesion -
    # "proportion of explicit content words that overlap between pairs of sentences".
    nouns_a = set([t.lemma for t in sentence_a.tokens if t.pos_coarse and t.pos_coarse in ud_content_pos])
    nouns_b = set([t.lemma for t in sentence_b.tokens if t.pos_coarse and t.pos_coarse in ud_content_pos])
    overlap = len(nouns_a.intersection(nouns_b))

    all_words = nouns_a | nouns_b

    return overlap / len(all_words) if len(all_words) > 0 else 0.0

# ============================================================================
# REFERENTIAL COHESION (CRF*) — "Do sentences keep talking about the same things?"
# Measures how often nouns, pronouns, stems, or any content words reappear
# across sentences. Higher = tighter textual cohesion. Reliable.
# ============================================================================

# LAY: Do adjacent sentences share at least one noun?
# ↑ Higher = nouns recur across neighbours. Reliable.
def cm_crfno1(sentences: List[Sentence]) -> Optional[float]:
    # Noun overlap, adjacent sentences, binary, mean
    noun_overlap_per_sentence = []
    for sind in range(len(sentences)):
        if sind == 0:
            continue
        current_sentence = sentences[sind]
        previous_sentence = sentences[sind-1]
        noun_overlap = min(1, _noun_overlap(current_sentence, previous_sentence))
        noun_overlap_per_sentence.append(noun_overlap)
    return np.mean(noun_overlap_per_sentence)

# LAY: Do any two sentences in the text share at least one noun?
# ↑ Higher = noun recurrence across the whole text. Reliable.
def cm_crfnoa(sentences: List[Sentence]) -> Optional[float]:
    # Noun overlap, all sentences, binary, mean
    noun_overlap_per_sentence = []
    for sinda in range(len(sentences)):
        for sindb in range(len(sentences)):
            if sindb <= sinda:
                continue
            sentence_a = sentences[sinda]
            sentence_b = sentences[sindb]
            noun_overlap = min(1, _noun_overlap(sentence_a, sentence_b))
            noun_overlap_per_sentence.append(noun_overlap)
    return np.mean(noun_overlap_per_sentence)

# LAY: Do adjacent sentences share a noun or pronoun (an "argument")?
# ↑ Higher = same entities referred to across neighbours. Reliable.
def cm_crfao1(sentences: List[Sentence]) -> Optional[float]:
    # Argument overlap, adjacent sentences, binary, mean
    argument_overlap_per_sentence = []
    for sind in range(len(sentences)):
        if sind == 0:
            continue
        current_sentence = sentences[sind]
        previous_sentence = sentences[sind-1]
        argument_overlap = min(1, _argument_overlap(current_sentence, previous_sentence))
        argument_overlap_per_sentence.append(argument_overlap)
    return np.mean(argument_overlap_per_sentence)

# LAY: Do any two sentences share a noun or pronoun?
# ↑ Higher = entity recurrence throughout text. Reliable.
def cm_crfaoa(sentences: List[Sentence]) -> Optional[float]:
    # Argument overlap, all sentences, binary, mean
    argument_overlap_per_sentence = []
    for sinda in range(len(sentences)):
        for sindb in range(len(sentences)):
            if sindb <= sinda:
                continue
            sentence_a = sentences[sinda]
            sentence_b = sentences[sindb]
            argument_overlap = min(1, _argument_overlap(sentence_a, sentence_b))
            argument_overlap_per_sentence.append(argument_overlap)
    return np.mean(argument_overlap_per_sentence)

# LAY: Do adjacent sentences share a word stem (e.g. "running"/"runs")?
# ↑ Higher = same word families recur. Reliable.
def cm_crfso1(sentences: List[Sentence]) -> Optional[float]:
    # Stem overlap, adjacent sentences, binary, mean
    stem_overlap_per_sentence = []
    for sind in range(len(sentences)):
        if sind == 0:
            continue
        current_sentence = sentences[sind]
        previous_sentence = sentences[sind-1]
        stem_overlap = min(1, _stem_overlap(current_sentence, previous_sentence))
        stem_overlap_per_sentence.append(stem_overlap)
    return np.mean(stem_overlap_per_sentence)

# LAY: Do any two sentences share a word stem?
# ↑ Higher = word-family recurrence throughout text. Reliable.
def cm_crfsoa(sentences: List[Sentence]) -> Optional[float]:
    # Stem overlap, all sentences, binary, mean
    stem_overlap_per_sentence = []
    for sinda in range(len(sentences)):
        for sindb in range(len(sentences)):
            if sindb <= sinda:
                continue
            sentence_a = sentences[sinda]
            sentence_b = sentences[sindb]
            stem_overlap = min(1, _stem_overlap(sentence_a, sentence_b))
            stem_overlap_per_sentence.append(stem_overlap)
    return np.mean(stem_overlap_per_sentence)

# LAY: What share of content words is shared between adjacent sentences? (mean)
# ↑ Higher = tighter local cohesion. Reliable.
def cm_crfcwo1(sentences: List[Sentence]) -> Optional[float]:
    # Content word overlap, adjacent sentences, proportional, mean
    word_overlap_per_sentence = []
    for sind in range(len(sentences)):
        if sind == 0:
            continue
        current_sentence = sentences[sind]
        previous_sentence = sentences[sind-1]
        word_overlap = min(1, _word_overlap(current_sentence, previous_sentence))
        word_overlap_per_sentence.append(word_overlap)
    return np.mean(word_overlap_per_sentence)

# LAY: How uneven is the content-word overlap between adjacent sentences?
# ↑ Higher = some pairs repeat heavily, others not at all. Reliable.
def cm_crfcwo1d(sentences: List[Sentence]) -> Optional[float]:
    # Content word overlap, adjacent sentences, proportional, standard deviation
    word_overlap_per_sentence = []
    for sind in range(len(sentences)):
        if sind == 0:
            continue
        current_sentence = sentences[sind]
        previous_sentence = sentences[sind-1]
        word_overlap = min(1, _word_overlap(current_sentence, previous_sentence))
        word_overlap_per_sentence.append(word_overlap)
    return np.std(word_overlap_per_sentence)

# LAY: What share of content words is shared across all sentence pairs? (mean)
# ↑ Higher = global cohesion. Reliable.
def cm_crfcwoa(sentences: List[Sentence]) -> Optional[float]:
    # Content word overlap, all sentences, proportional, mean
    word_overlap_per_sentence = []
    for sinda in range(len(sentences)):
        for sindb in range(len(sentences)):
            if sindb <= sinda:
                continue
            sentence_a = sentences[sinda]
            sentence_b = sentences[sindb]
            word_overlap = min(1, _word_overlap(sentence_a, sentence_b))
            word_overlap_per_sentence.append(word_overlap)
    return np.mean(word_overlap_per_sentence)

# LAY: How uneven is the global content-word overlap across sentence pairs?
# ↑ Higher = mix of tight and loose cohesion. Reliable.
def cm_crfcwoad(sentences: List[Sentence]) -> Optional[float]:
    # Content word overlap, all sentences, proportional, mean
    word_overlap_per_sentence = []
    for sinda in range(len(sentences)):
        for sindb in range(len(sentences)):
            if sindb <= sinda:
                continue
            sentence_a = sentences[sinda]
            sentence_b = sentences[sindb]
            word_overlap = min(1, _word_overlap(sentence_a, sentence_b))
            word_overlap_per_sentence.append(word_overlap)
    return np.std(word_overlap_per_sentence)

def _lexical_diversity_tokens(tokens: List[Token]) -> Tuple[List[str], List[str], List[str]]:
    # M4 fix: Appendix A #46 specifies LDTTRc uses "content word LEMMAS", not
    # surface forms. Return an extra lemma list for LDTTRc; keep surface-form
    # lists for LDTTRa/LDMTLDa/LDVOCDa ("all words", spec is surface).
    tokens_alpha = [token.text.lower() for token in tokens if token.is_alpha]
    tokens_content = [token.text.lower() for token in tokens if token.pos_coarse in ud_content_pos and token.is_alpha]
    tokens_content_lemma = [
        (token.lemma.lower() if token.lemma else token.text.lower())
        for token in tokens
        if token.pos_coarse in ud_content_pos and token.is_alpha
    ]
    return tokens_alpha, tokens_content, tokens_content_lemma

# ============================================================================
# LEXICAL DIVERSITY (LD*) — "How varied is the vocabulary?"
# Ratios of unique words to total words. High values mean the author uses many
# different words; low values mean the same words recur often. Reliable.
# ============================================================================

# LAY: Ratio of unique content-word lemmas to total content words.
# ↑ Higher = richer content vocabulary. Reliable.
def cm_ldttrc(tokens: List[Token]) -> Optional[float]:
    # M4: use content-word LEMMAS (Appendix A #46).
    _, _, tokens_content_lemma = _lexical_diversity_tokens(tokens)
    return ld.ttr(tokens_content_lemma)

# LAY: Ratio of unique words to total words (all alphabetic words).
# ↑ Higher = more varied vocabulary overall. Reliable.
def cm_ldttra(tokens: List[Token]) -> Optional[float]:
    tokens_alpha, _, _ = _lexical_diversity_tokens(tokens)
    return ld.ttr(tokens_alpha)

# LAY: MTLD — vocabulary diversity, robust to text length.
# ↑ Higher = more varied vocabulary (not inflated by length). Reliable.
def cm_ldmtlda(tokens: List[Token]) -> Optional[float]:
    tokens_alpha, _, _ = _lexical_diversity_tokens(tokens)
    return ld.mtld(tokens_alpha)

# LAY: VOCD — vocabulary diversity via random-sample curve fitting.
# ↑ Higher = more varied vocabulary. Reliable.
def cm_ldvocda(tokens: List[Token]) -> Optional[float]:
    tokens_alpha, _, _ = _lexical_diversity_tokens(tokens)
    lex = LexicalRichness(tokens_alpha, preprocessor=None, tokenizer=None)
    return lex.vocd()

# ============================================================================
# SYNTACTIC COMPLEXITY (SYN*) — "How complex and consistent is sentence structure?"
# Left-embeddedness, noun-phrase size, sentence-pair edit distance, and
# structural similarity. SYNSTRUT* is an approximation (we use dependency
# rather than constituency parses); the others are reliable.
# ============================================================================

# LAY: Average number of words before the main verb of a sentence.
# ↑ Higher = more left-embedded clauses, harder to read. Reliable.
def cm_synle(sentences: List[Sentence]) -> Optional[float]:
    deps = [[token.dep_type for token in sent.tokens] for sent in sentences]

    word_counts = []
    counter_start = 0
    # M1 fix (partial): accept multiple ROOT conventions. DKPro/Stanford interop
    # emits "--", spaCy/UD emits "ROOT"/"root". Previously only "--" was matched,
    # which made SYNLE silently return 0 under UD-style pipelines.
    # TODO(M1): verify which marker the current DUUI spaCy component emits; this
    # wider match is safe because none of these strings collide with any real
    # non-root dep label.
    _ROOT_MARKERS = ("--", "ROOT", "root")
    for sent in deps:
        root = [c for c, token in enumerate(sent) if token in _ROOT_MARKERS]
        if root:
            root_index = (counter_start+root[0]) - counter_start
            word_counts.append(root_index)
        counter_start += len(sent)

    return np.mean(word_counts) if word_counts else 0

ud_tiger_dep_mapping_de = {
    "de": {
        "AMOD": ["NK", "ADC"],
        "COMPOUND": ["NK"],
        "PREP": ["MO", "AC", "PG"]
    }
}

# LAY: Average number of modifiers (adjectives, compounds, PPs) per noun phrase.
# ↑ Higher = denser noun phrases, more complex syntax. Reliable.
def cm_synnp(sentences: List[Sentence], noun_chunks: List[NounChunk], lang: str) -> Optional[float]:
    # H2 fix: previously compared noun_chunk.begin/end to sentence.begin/end for
    # equality, which never matched (chunks are sub-sentence spans), so this
    # index always returned 0. Now use containment and count modifier-type deps
    # for tokens that fall inside the chunk span.
    dep_map_en = ["AMOD", "COMPOUND", "PREP"]
    if lang == "de":
        dep_map = []
        for dep_en in dep_map_en:
            dep_map.extend(ud_tiger_dep_mapping_de["de"][dep_en])
    else:
        dep_map = dep_map_en

    modifier_counts = []
    for noun_chunk in noun_chunks:
        for sentence in sentences:
            if sentence.begin <= noun_chunk.begin and noun_chunk.end <= sentence.end:
                chunk_tokens = [tok for tok in sentence.tokens
                                if tok.begin >= noun_chunk.begin and tok.end <= noun_chunk.end]
                modifiers = [tok for tok in chunk_tokens if tok.dep_type in dep_map]
                modifier_counts.append(len(modifiers))
                break

    return np.mean(modifier_counts) if modifier_counts else 0

# LAY: How different are adjacent sentences in their POS-tag sequences?
# ↑ Higher = consecutive sentences have very different grammatical shapes. Reliable.
def cm_synmedpos(sentences: List[Sentence]) -> Optional[float]:
    # H1 fix: compute mean normalized edit distance between CONSECUTIVE SENTENCES'
    # POS-tag sequences (spec: Ch. 4, "distance ... between consecutive sentences"),
    # not between adjacent individual tokens as done previously.
    # Approximation: character-level Levenshtein on space-joined POS strings.
    sent_pos_strs = [" ".join(token.pos_coarse for token in sent.tokens) for sent in sentences]
    normalized_levenshtein = NormalizedLevenshtein()
    pos_dists = []
    for i in range(len(sent_pos_strs) - 1):
        pos_dists.append(normalized_levenshtein.distance(sent_pos_strs[i], sent_pos_strs[i + 1]))
    return np.mean(pos_dists) if pos_dists else 0

# LAY: How different are adjacent sentences word-for-word?
# ↑ Higher = neighbours share fewer surface words. Reliable.
def cm_synmedwrd(sentences: List[Sentence]) -> Optional[float]:
    # H1 fix: see cm_synmedpos. Edit distance over consecutive sentence texts.
    sent_word_strs = [" ".join(token.text for token in sent.tokens) for sent in sentences]
    normalized_levenshtein = NormalizedLevenshtein()
    word_dists = []
    for i in range(len(sent_word_strs) - 1):
        word_dists.append(normalized_levenshtein.distance(sent_word_strs[i], sent_word_strs[i + 1]))
    return np.mean(word_dists) if word_dists else 0

# LAY: How different are adjacent sentences at the lemma level?
# ↑ Higher = neighbours share fewer word stems. Reliable.
def cm_synmedlem(sentences: List[Sentence]) -> Optional[float]:
    # H1 fix: see cm_synmedpos. Edit distance over consecutive sentence lemmas.
    sent_lemma_strs = [" ".join(token.lemma for token in sent.tokens) for sent in sentences]
    normalized_levenshtein = NormalizedLevenshtein()
    lemma_dists = []
    for i in range(len(sent_lemma_strs) - 1):
        lemma_dists.append(normalized_levenshtein.distance(sent_lemma_strs[i], sent_lemma_strs[i + 1]))
    return np.mean(lemma_dists) if lemma_dists else 0

# NOTE(M9): SYNSTRUT is a documented approximation of Coh-Metrix's constituency-
# based tree similarity. The spec (Ch. 4, Fig. 4.1) computes the maximum common
# subtree over constituency parses (NP, VP, S, DT, NN, VBD, PRP, …). Here we
# operate on (dep_label, pos_tag) node sets because the DUUI pipeline emits
# dependency parses, not constituency trees. Jaccard over these tuple-nodes
# preserves the intended signal (structural similarity) but values are not
# numerically comparable to the original Coh-Metrix output.
def _compute_tree_similarity(nodes1, nodes2):
    common = nodes1 & nodes2
    total = len(nodes1) + len(nodes2) - len(common)
    return len(common) / total if total else 0.0

def _get_tree_nodes(deps, poses, puncts):
    # Each node is a tuple of dependency + POS tag
    nodes = set()
    for token_pos, token_deps, token_punct in zip(poses, deps, puncts):
        if not token_punct:
            nodes.add((token_deps, token_pos))
    return nodes

def _get_tree_nodes_paragraphs(deps, poses, puncts):
    nodes = set()
    for sent_pos, sent_deps, sent_puncts in zip(poses, deps, puncts):
        for pos, dep, punct in zip(sent_pos, sent_deps, sent_puncts):
            if not punct:
                nodes.add((dep, pos))
    return nodes

# LAY: How similar are adjacent sentences in grammatical structure?
# ↑ Higher = neighbours share grammatical patterns. Approximate (uses
#   dependency parse; original uses constituency parse — see NOTE(M9)).
def cm_synstruta(sentences: List[Sentence]) -> Optional[float]:
    poses = [[token.pos_coarse for token in sent.tokens] for sent in sentences]
    deps = [[token.dep_type for token in sent.tokens] for sent in sentences]
    puncts = [[token.is_punct for token in sent.tokens] for sent in sentences]

    similarities = []
    for i in range(len(deps) - 1):
        sim = _compute_tree_similarity(
            _get_tree_nodes(deps[i], poses[i], puncts[i]),
            _get_tree_nodes(deps[i + 1], poses[i + 1], puncts[i + 1])
        )
        similarities.append(sim)

    return np.mean(similarities) if similarities else 0.0

# LAY: How similar are any two paragraphs in grammatical structure?
# ↑ Higher = consistent syntactic style across paragraphs. Approximate (see NOTE(M9)).
def cm_synstrutt(paragraphs: List[Paragraph]) -> Optional[float]:
    poses_paragraph = []
    deps_paragraph = []
    puncts_paragraph = []
    for par in paragraphs:
        poses_sent = []
        deps_sent = []
        puncts_sent = []
        for sent in par.sentences:
            poses_sent.append([token.pos_coarse for token in sent.tokens])
            deps_sent.append([token.dep_type for token in sent.tokens])
            puncts_sent.append([token.is_punct for token in sent.tokens])
        poses_paragraph.append(poses_sent)
        deps_paragraph.append(deps_sent)
        puncts_paragraph.append(puncts_sent)

    similarities = []
    # M8 fix: was range(len - 1), which dropped the last paragraph from all
    # pair combinations. Spec (SYNSTRUTt) wants all paragraph pairs.
    for s1, s2 in combinations(range(len(deps_paragraph)), 2):
        sim = _compute_tree_similarity(
            _get_tree_nodes_paragraphs(deps_paragraph[s1], poses_paragraph[s1], puncts_paragraph[s1]),
            _get_tree_nodes_paragraphs(deps_paragraph[s2], poses_paragraph[s2], puncts_paragraph[s2])
        )
        similarities.append(sim)

    return np.mean(similarities) if similarities else 0.0

def _count_metrics(sentences: List[Sentence], noun_chunks: List[NounChunk], lang: str) -> Dict[str, int]:
    token_pos = [[token.pos_coarse for token in sent.tokens] for sent in sentences]
    words = [[token.text for token in sent.tokens] for sent in sentences]
    tags = [[token.pos_value for token in sent.tokens] for sent in sentences]
    deps = [[token.dep_type for token in sent.tokens] for sent in sentences]
    lemmas = [[token.lemma for token in sent.tokens] for sent in sentences]
    puncts = [[token.is_punct for token in sent.tokens] for sent in sentences]

    count_metrics_dict = {
        "total_tokens": 0,
        "total_sentences": 0,
        "noun_phrase_count": 0,
        "verb_count": 0,
        "adverb_count": 0,
        "prep_count": 0,
        "passive_sentences": 0,
        "neg_count": 0,
        "gerund_count": 0,
        "infinitive_count": 0
    }

    count_metrics_dict["noun_phrase_count"] = len(noun_chunks)

    # H3: passive detection must be language-specific.
    #  - English (spec Ch. 4, \u00a7Syntactic complexity): sentence has an auxpass
    #    or nsubjpass dependency.
    #  - German: \"Vorgangspassiv\" = form of werden + past participle (VVPP);
    #    \"Zustandspassiv\" = form of sein + past participle.
    de_passive_aux_lemmas = {"werden", "sein"}

    for c, tokens in enumerate(token_pos):
        is_passive = False
        for j, token_pos_i in enumerate(tokens):
            match token_pos_i:
                case "VERB":
                    count_metrics_dict["verb_count"] += 1
                case "ADV":
                    count_metrics_dict["adverb_count"] += 1
                case "ADP":
                    count_metrics_dict["prep_count"] += 1
            word_i = words[c][j]
            tag_i = tags[c][j]
            dep_i = deps[c][j]
            lemma_i = lemmas[c][j]
            if lang == "de":
                # German passive: auxiliary werden/sein co-occurring with a
                # past participle (VVPP) in the same sentence.
                if lemma_i.lower() in de_passive_aux_lemmas and "VVPP" in tags[c]:
                    is_passive = True
            else:
                # English passive: look for auxpass / nsubjpass deps.
                if dep_i in ("AUXPASS", "NSUBJPASS", "auxpass", "nsubjpass"):
                    is_passive = True
            if dep_i == "NEG" or dep_i == "NG":
                count_metrics_dict["neg_count"] += 1
            # H11: VBG is the English gerund (\"-ing\"). German has no gerund
            # equivalent, so leave count at 0 for DE (previously VVPP/ADJD were
            # counted, but VVPP is past participle and ADJD is adverbial
            # adjective \u2013 not gerunds).
            if lang != "de" and tag_i == "VBG":
                count_metrics_dict["gerund_count"] += 1
            if 0 < j < len(tokens) - 1:
                # DE = VVINF
                #if (word_i.lower() == "to" or word_i.lower() == "zu") and (tags[c][j+1] == "VB" or tags[c][j+1] == "VVINF") and token_pos[c][j+1] == "VERB":
                #    count_metrics_dict["infinitive_count"] += 1
                if word_i.lower() == "to" or word_i.lower() == "zu":
                    if j + 1 < len(tags[c]) and j + 1 < len(token_pos[c]):
                        if (tags[c][j+1] in ["VB", "VVINF"]) and token_pos[c][j+1] == "VERB":
                            count_metrics_dict["infinitive_count"] += 1
                if tags[c][j] == "VM" and token_pos[c][j] == "VERB":
                    for offset in range(1, 3):
                        next_idx = j + offset
                        if next_idx < len(tags[c]) and next_idx < len(token_pos[c]):
                            if tags[c][next_idx] == "VVINF" and token_pos[c][next_idx] == "VERB":
                                count_metrics_dict["infinitive_count"] += 1
                                break
        if is_passive:
            count_metrics_dict["passive_sentences"] += 1

    # iterate over puncts count falses
    count_metrics_dict["total_tokens"] = len([token for sublist in puncts for token in sublist if not token])
    count_metrics_dict["total_sentences"] = len(token_pos)

    return count_metrics_dict

# ============================================================================
# SYNTACTIC PATTERN DENSITY (DR*) — "Which grammatical constructions appear,
# and how often?" All values are incidences (per 1,000 words). Reliable
# counts, modulo one language caveat: DRGERUND returns 0 for German (no direct
# gerund equivalent — see NOTE(H11)). DRPVAL is a per-sentence proportion.
# ============================================================================

# LAY: How many noun phrases per 1,000 words?
# ↑ Higher = more noun-heavy text. Reliable.
def cm_drnp(sentences: List[Sentence], noun_chunks: List[NounChunk], lang: str, metrics: Optional[Dict[str, int]] = None) -> Optional[float]:
    # H12: DRNP is an INCIDENCE (per 1000 words), not a raw ratio.
    # L2: accept precomputed dict to avoid recomputing 8 times per request.
    if metrics is None:
        metrics = _count_metrics(sentences, noun_chunks, lang)
    return _incidence(metrics["noun_phrase_count"], metrics["total_tokens"])

# LAY: How many verbs per 1,000 words?
# ↑ Higher = more action/verb-driven text. Reliable.
def cm_drvp(sentences: List[Sentence], noun_chunks: List[NounChunk], lang: str, metrics: Optional[Dict[str, int]] = None) -> Optional[float]:
    # H12: DRVP incidence per 1000 words.
    if metrics is None:
        metrics = _count_metrics(sentences, noun_chunks, lang)
    return _incidence(metrics["verb_count"], metrics["total_tokens"])

# LAY: How many adverbs per 1,000 words?
# ↑ Higher = more descriptive modifiers. Reliable.
def cm_drap(sentences: List[Sentence], noun_chunks: List[NounChunk], lang: str, metrics: Optional[Dict[str, int]] = None) -> Optional[float]:
    # H12: DRAP incidence per 1000 words.
    if metrics is None:
        metrics = _count_metrics(sentences, noun_chunks, lang)
    return _incidence(metrics["adverb_count"], metrics["total_tokens"])

# LAY: How many prepositions per 1,000 words?
# ↑ Higher = more phrase-rich syntax. Reliable.
def cm_drpp(sentences: List[Sentence], noun_chunks: List[NounChunk], lang: str, metrics: Optional[Dict[str, int]] = None) -> Optional[float]:
    # H12: DRPP incidence per 1000 words.
    if metrics is None:
        metrics = _count_metrics(sentences, noun_chunks, lang)
    return _incidence(metrics["prep_count"], metrics["total_tokens"])

# LAY: Proportion of sentences in passive voice (spec: per-sentence, not per 1000 words).
# ↑ Higher = more passive/impersonal style. Reliable.
def cm_drpval(sentences: List[Sentence], noun_chunks: List[NounChunk], lang: str, metrics: Optional[Dict[str, int]] = None) -> Optional[float]:
    # DRPVAL: passive voice density. Spec (Ch.4) defines as incidence of
    # passive sentences per 1000 words (Appendix A index 77).
    if metrics is None:
        metrics = _count_metrics(sentences, noun_chunks, lang)
    return _incidence(metrics["passive_sentences"], metrics["total_tokens"])

# LAY: How many negations (not/nicht/kein/...) per 1,000 words?
# ↑ Higher = more negation. Reliable.
def cm_drneg(sentences: List[Sentence], noun_chunks: List[NounChunk], lang: str, metrics: Optional[Dict[str, int]] = None) -> Optional[float]:
    # H12: DRNEG incidence per 1000 words.
    if metrics is None:
        metrics = _count_metrics(sentences, noun_chunks, lang)
    return _incidence(metrics["neg_count"], metrics["total_tokens"])

# LAY: How many English gerunds ("-ing" verb forms) per 1,000 words?
# Language-limited: DE=0 (German has no gerund equivalent — NOTE(H11)).
def cm_drgerund(sentences: List[Sentence], noun_chunks: List[NounChunk], lang: str, metrics: Optional[Dict[str, int]] = None) -> Optional[float]:
    # H12: DRGERUND incidence per 1000 words.
    if metrics is None:
        metrics = _count_metrics(sentences, noun_chunks, lang)
    return _incidence(metrics["gerund_count"], metrics["total_tokens"])

# LAY: How many infinitive verb forms (to/zu + VERB) per 1,000 words?
# ↑ Higher = more infinitive constructions. Reliable.
def cm_drinf(sentences: List[Sentence], noun_chunks: List[NounChunk], lang: str, metrics: Optional[Dict[str, int]] = None) -> Optional[float]:
    # H12: DRINF incidence per 1000 words.
    if metrics is None:
        metrics = _count_metrics(sentences, noun_chunks, lang)
    return _incidence(metrics["infinitive_count"], metrics["total_tokens"])

def _incidence(count, total_words):
    # Coh-Metrix convention (Ch. 4 §Word information): "relative frequency of
    # each word category by counting the number of instances of the category
    # per 1,000 words of text, called incidence scores." Used for all DR*, WRD*
    # (noun/verb/adj/adv/pronoun), and Situation Model verb incidences
    # (SMCAUSv, SMCAUSvp, SMINTEp — see H15). Returns 0 when total_words is 0
    # to avoid ZeroDivisionError on empty documents; callers typically guard
    # this by checking for empty input upstream.
    return (count / total_words) * 1000 if total_words > 0 else 0

def _count_words(poses: List[List[str]], words: List[List[str]], pronouns_category) -> Dict[str, int]:
    total_words  = 0
    for sent in words:
        for word in sent:
            if word.isalpha():
                total_words += 1
    counters = {
        "noun": 0,
        "verb": 0,
        "adj": 0,
        "adv": 0,
        "pronoun_total": 0,
        "prp1s": 0,  # first-person singular pronouns
        "prp1p": 0,  # first-person plural pronouns
        "prp2": 0,  # second-person pronouns
        "prp3s": 0,  # third-person singular pronouns
        "prp3p": 0  # third-person plural pronouns
    }
    for i, sent in enumerate(poses):
        for j, pos in enumerate(sent):
            word_i = words[i][j].lower()
            match pos:
                case "NOUN":
                    counters["noun"] += 1
                case "VERB":
                    counters["verb"] += 1
                case "ADJ":
                    counters["adj"] += 1
                case "ADV":
                    counters["adv"] += 1
                case "PRON":
                    counters["pronoun_total"] += 1
                    if "prp1s" in pronouns_category:
                        if word_i in pronouns_category["prp1s"]:
                            counters["prp1s"] += 1
                    if "prp1p" in pronouns_category:
                        if word_i in pronouns_category["prp1p"]:
                            counters["prp1p"] += 1
                    if "prp2" in pronouns_category:
                        if word_i in pronouns_category["prp2"]:
                            counters["prp2"] += 1
                    if "prp3s" in pronouns_category:
                        if word_i in pronouns_category["prp3s"]:
                            counters["prp3s"] += 1
                    if "prp3p" in pronouns_category:
                        if word_i in pronouns_category["prp3p"]:
                            counters["prp3p"] += 1
    counter_1000 = {
        "noun": 0,
        "verb": 0,
        "adj": 0,
        "adv": 0,
        "pronoun_total": 0,
        "prp1s": 0,  # first-person singular pronouns
        "prp1p": 0,  # first-person plural pronouns
        "prp2": 0,  # second-person pronouns
        "prp3s": 0,  # third-person singular pronouns
        "prp3p": 0   # third-person plural pronouns
    }
    for key in counters:
        counter_1000[key] = _incidence(counters[key], total_words)
    return counter_1000

def _get_morhological_features(poses: List[List[str]], words: List[List[str]], morph_Person, morph_Number) -> defaultdict[Any, set]:
    pronouns_by_category = defaultdict(set)
    for i, sent in enumerate(poses):
        for  j, pos in enumerate(sent):
            if pos != "PRON":
                continue
            person = morph_Person[i][j]
            number = morph_Number[i][j]

            person = person[0] if person else "Unknown"
            number = number[0] if number else "Unknown"
            if person == "2":
                key = "prp2"
            else:
                key = f"prp{person.lower()}{'s' if number == 'S' else 'p'}"
            pronouns_by_category[key].add(words[i][j].lower())

    return pronouns_by_category

def _wrd_precompute(sentences: List[Sentence]) -> Dict[str, int]:
    # L11 helper: compute the shared WRD* per-1000 counter dict once so the
    # 10 cm_wrd* wrappers don't re-iterate tokens and morphology features.
    token_pos = [[token.pos_coarse for token in sent.tokens] for sent in sentences]
    words = [[token.text for token in sent.tokens] for sent in sentences]
    morph_person = [[token.morph_person for token in sent.tokens] for sent in sentences]
    morph_number = [[token.morph_number for token in sent.tokens] for sent in sentences]
    pronouns_by_category = _get_morhological_features(token_pos, words, morph_person, morph_number)
    return _count_words(token_pos, words, pronouns_by_category)

# ============================================================================
# WORD INFORMATION (WRD*) — "What kind of words are used?"
# Three sub-groups:
#   (a) POS & pronoun incidences — counts per 1,000 words. Reliable.
#   (b) MRC psycholinguistic ratings (AoA, familiarity, concreteness,
#       imageability, meaningfulness). English-only — DE returns None. NOTE(H8).
#   (c) Polysemy & hypernymy via WordNet (en) / GermaNet (de). Values are
#       not directly comparable across languages. NOTE(H9).
#   (d) CELEX word-frequency indices are stubbed None; Wikipedia-sample
#       alternatives WRDFRQ*_wiki10000 are emitted. NOTE(L9).
# ============================================================================

# LAY: How many nouns per 1,000 words?
# ↑ Higher = noun-dense informational style. Reliable.
def cm_wrdnoun(sentences: List[Sentence], counts: Optional[Dict[str, int]] = None) -> Optional[float]:
    # L11: accept precomputed dict to avoid recomputing 10 times per request.
    if counts is None:
        counts = _wrd_precompute(sentences)
    return counts["noun"]

# LAY: How many verbs per 1,000 words?
# ↑ Higher = action-heavy style. Reliable.
def cm_wrdverb(sentences: List[Sentence], counts: Optional[Dict[str, int]] = None) -> Optional[float]:
    if counts is None:
        counts = _wrd_precompute(sentences)
    return counts["verb"]

# LAY: How many adjectives per 1,000 words?
# ↑ Higher = more descriptive/modifier-heavy style. Reliable.
def cm_wrdadj(sentences: List[Sentence], counts: Optional[Dict[str, int]] = None) -> Optional[float]:
    if counts is None:
        counts = _wrd_precompute(sentences)
    return counts["adj"]

# LAY: How many adverbs per 1,000 words?
# ↑ Higher = more manner/degree modifiers. Reliable.
def cm_wrdadv(sentences: List[Sentence], counts: Optional[Dict[str, int]] = None) -> Optional[float]:
    if counts is None:
        counts = _wrd_precompute(sentences)
    return counts["adv"]

# LAY: How many pronouns per 1,000 words?
# ↑ Higher = more reference/less explicit naming. Reliable.
def cm_wrdpro(sentences: List[Sentence], counts: Optional[Dict[str, int]] = None) -> Optional[float]:
    if counts is None:
        counts = _wrd_precompute(sentences)
    return counts["pronoun_total"]

# LAY: First-person singular pronouns (I/me/my — ich/mir/mein) per 1,000 words.
# ↑ Higher = personal/narrative voice. Reliable.
def cm_wrdprp1s(sentences: List[Sentence], counts: Optional[Dict[str, int]] = None) -> Optional[float]:
    if counts is None:
        counts = _wrd_precompute(sentences)
    return counts["prp1s"]

# LAY: First-person plural pronouns (we/us — wir/uns) per 1,000 words.
# ↑ Higher = group/collective voice. Reliable.
def cm_wrdprp1p(sentences: List[Sentence], counts: Optional[Dict[str, int]] = None) -> Optional[float]:
    if counts is None:
        counts = _wrd_precompute(sentences)
    return counts["prp1p"]

# LAY: Second-person pronouns (you — du/ihr/Sie) per 1,000 words.
# ↑ Higher = direct address to the reader. Reliable.
def cm_wrdprp2(sentences: List[Sentence], counts: Optional[Dict[str, int]] = None) -> Optional[float]:
    if counts is None:
        counts = _wrd_precompute(sentences)
    return counts["prp2"]

# LAY: Third-person singular pronouns (he/she/it — er/sie/es) per 1,000 words.
# ↑ Higher = more narration about a single entity. Reliable.
def cm_wrdprp3s(sentences: List[Sentence], counts: Optional[Dict[str, int]] = None) -> Optional[float]:
    if counts is None:
        counts = _wrd_precompute(sentences)
    return counts["prp3s"]

# LAY: Third-person plural pronouns (they — sie/ihnen) per 1,000 words.
# ↑ Higher = more narration about groups. Reliable.
def cm_wrdprp3p(sentences: List[Sentence], counts: Optional[Dict[str, int]] = None) -> Optional[float]:
    if counts is None:
        counts = _wrd_precompute(sentences)
    return counts["prp3p"]

def _load_mrc_database():
    filepath = "src/main/resources/mrc_psycholinguistic_database.csv"
    mrc_dict = {}
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=',')
        for row in reader:
            word = row['Word'].lower()
            meaningful_colorado = int(row['Meaningfulness: Coloradao Norms']) if row['Meaningfulness: Coloradao Norms'] else None
            meaningful_pavio = int(row['Meaningfulness: Pavio Norms']) if row[
                'Meaningfulness: Pavio Norms'] else None
            meaningful = None
            if meaningful_colorado is not None and meaningful_pavio is not None:
                meaningful = meaningful_colorado + meaningful_pavio
            elif meaningful_colorado is not None:
                meaningful = meaningful_colorado
            elif meaningful_pavio is not None:
                meaningful = meaningful_pavio
            mrc_dict[word] = {
                'AoA': int(row['Age of Acquisition Rating']) if row['Age of Acquisition Rating'] else None,
                'Familiarity': int(row['Familiarity']) if row['Familiarity'] else None,
                'Concreteness': int(row['Concreteness']) if row['Concreteness'] else None,
                'Imageability': int(row['Imageability']) if row['Imageability'] else None,
                'Meaningfulness': meaningful
            }
    return mrc_dict

mrc_dict = _load_mrc_database()

def _get_content_words(sentences: List[Sentence]):
    words = [[token.text for token in sent.tokens] for sent in sentences]
    poses = [[token.pos_coarse for token in sent.tokens] for sent in sentences]

    words_flatten = [word for sublist in words for word in sublist]
    poses_flatten = [pos for sublist in poses for pos in sublist]

    content_words = [
        word.lower()
        for word, pos
        in zip(words_flatten, poses_flatten)
        if pos in ud_content_pos and word.isalpha()
    ]

    return content_words, words, poses

def _get_content_words_per_sentence(sentences: List[Sentence]):
    words = [[token.text for token in sent.tokens] for sent in sentences]
    poses = [[token.pos_coarse for token in sent.tokens] for sent in sentences]

    content_words = []
    for swords, sposes in zip(words, poses):
        content_words_sent = [
            word.lower()
            for word, pos
            in zip(swords, sposes)
            if pos in ud_content_pos and word.isalpha()
        ]
        content_words.append(content_words_sent)

    return content_words, words, poses

def _average_rating(words, mrc_dict, key):
    ratings = [
        mrc_dict[w][key]
        for w
        in words
        if w in mrc_dict and mrc_dict[w][key] is not None
    ]
    if not ratings:
        return None
    return sum(ratings) / len(ratings)

# LAY: Average age at which content words are typically learned (MRC norms).
# ↑ Higher = later-acquired, harder words. English-only (DE=None — NOTE(H8)).
def cm_wrdaoac(sentences: List[Sentence], lang: str) -> Optional[float]:
    # H8: MRC Psycholinguistic Database is English-only.
    if lang != "en":
        return None
    content_words, _, _ = _get_content_words(sentences)
    if not content_words:
        return None
    return _average_rating(content_words, mrc_dict, 'AoA')

# LAY: Average subjective familiarity rating of content words (MRC norms).
# ↑ Higher = more familiar, easier words. English-only (DE=None — NOTE(H8)).
def cm_wrdfamc(sentences: List[Sentence], lang: str) -> Optional[float]:
    if lang != "en":
        return None
    content_words, _, _ = _get_content_words(sentences)
    if not content_words:
        return None
    return _average_rating(content_words, mrc_dict, 'Familiarity')

# LAY: Average concreteness of content words (MRC norms).
# ↑ Higher = more concrete, sensory words. English-only (DE=None — NOTE(H8)).
def cm_wrdcncc(sentences: List[Sentence], lang: str) -> Optional[float]:
    if lang != "en":
        return None
    content_words, _, _ = _get_content_words(sentences)
    if not content_words:
        return None
    return _average_rating(content_words, mrc_dict, 'Concreteness')

# LAY: Average imageability of content words (MRC norms).
# ↑ Higher = easier to form a mental picture. English-only (DE=None — NOTE(H8)).
def cm_wrdimgc(sentences: List[Sentence], lang: str) -> Optional[float]:
    if lang != "en":
        return None
    content_words, _, _ = _get_content_words(sentences)
    if not content_words:
        return None
    return _average_rating(content_words, mrc_dict, 'Imageability')

# LAY: Average subjective meaningfulness of content words (MRC/Colorado norms).
# ↑ Higher = stronger semantic associations. English-only (DE=None — NOTE(H8)).
def cm_wrdmeac(sentences: List[Sentence], lang: str) -> Optional[float]:
    if lang != "en":
        return None
    content_words, _, _ = _get_content_words(sentences)
    if not content_words:
        return None
    return _average_rating(content_words, mrc_dict, 'Meaningfulness')

def _get_polysemy(word, lang: str = "en"):
    # H9: dispatch by language. WordNet for English; GermaNet for German when
    # configured. For unsupported languages (or missing GermaNet) return None
    # so callers treat the word as not-covered instead of silently using the
    # English WordNet for German/etc.
    if lang == "en":
        synsets = wn.synsets(word)
        return len(synsets)
    if lang == "de":
        if germanet is None:
            return None
        try:
            return len(germanet.get_synsets_by_orthform(word))
        except Exception:
            return None
    return None

def _get_max_hypernym_depth(word, pos=None, lang: str = "en"):
    # H9: dispatch by language (see _get_polysemy).
    if lang == "en":
        synsets = wn.synsets(word, pos=pos) if pos else wn.synsets(word)
        if not synsets:
            return None
        depths = [synset.max_depth() for synset in synsets]
        return max(depths)
    if lang == "de":
        if germanet is None:
            return None
        try:
            word_cat = None
            if pos == "NOUN":
                word_cat = WordCategory.nomen
            elif pos == "VERB":
                word_cat = WordCategory.verben
            synsets = germanet.get_synsets_by_orthform(word)
            if word_cat is not None:
                synsets = [s for s in synsets if s.word_category == word_cat]
            if not synsets:
                return None
            depths = [s.min_depth() if hasattr(s, "min_depth") else 0 for s in synsets]
            return max(depths) if depths else None
        except Exception:
            return None
    return None

# LAY: Average polysemy: how many senses each content word has.
# ↑ Higher = more ambiguous vocabulary. Approximate for DE (GermaNet) — NOTE(H9).
def cm_wrdpolc(sentences: List[Sentence], lang: str) -> Optional[float]:
    polysemies = []

    content_word, _, _ = _get_content_words(sentences)
    for word in content_word:
        poly = _get_polysemy(word, lang=lang)
        if poly is None or poly <= 0:
            continue
        polysemies.append(poly)

    return np.mean(polysemies) if polysemies else None

def _calc_wrdhyp(sentences: List[Sentence], lang: str) -> Dict[str, Optional[float]]:
    hypernym_nouns = []
    hypernym_verbs = []

    content_word, words, poses = _get_content_words(sentences)
    set_content_word = set(content_word)
    for i, sent in enumerate(words):
        for j, word in enumerate(sent):
            pos = poses[i][j]
            if word.lower() in set_content_word and (pos=="NOUN" or pos=="VERB"):
                # For EN pass WordNet POS; for DE pass coarse POS string so
                # dispatch can map to a GermaNet WordCategory.
                if lang == "en":
                    pos_in = wn.NOUN if pos == "NOUN" else wn.VERB
                else:
                    pos_in = pos
                hyp = _get_max_hypernym_depth(word.lower(), pos=pos_in, lang=lang)
                if hyp is not None:
                    if pos == "NOUN":
                        hypernym_nouns.append(hyp)
                    elif pos == "VERB":
                        hypernym_verbs.append(hyp)

    # Average hypernymy for nouns and verbs separately
    hypn_avg = sum(hypernym_nouns) / len(hypernym_nouns) if hypernym_nouns else None
    hypv_avg = sum(hypernym_verbs) / len(hypernym_verbs) if hypernym_verbs else None
    # Combined hypernymy average
    combined = hypernym_nouns + hypernym_verbs
    hypnv_avg = sum(combined) / len(combined) if combined else None

    return {
        "WRDHYPn": hypn_avg,
        "WRDHYPv": hypv_avg,
        "WRDHYPnv": hypnv_avg
    }

# LAY: Average hypernymy depth of nouns (how specific/abstract).
# ↑ Higher = more specific nouns (deeper in the WordNet tree). Approximate for DE — NOTE(H9).
def cm_wrdhypn(sentences: List[Sentence], lang: str) -> Optional[float]:
    return _calc_wrdhyp(sentences, lang)["WRDHYPn"]

# LAY: Average hypernymy depth of verbs.
# ↑ Higher = more specific verbs. Approximate for DE (GermaNet) — NOTE(H9).
def cm_wrdhypv(sentences: List[Sentence], lang: str) -> Optional[float]:
    return _calc_wrdhyp(sentences, lang)["WRDHYPv"]

# LAY: Average hypernymy depth of nouns and verbs combined.
# ↑ Higher = more specific content words overall. Approximate for DE — NOTE(H9).
def cm_wrdhypnv(sentences: List[Sentence], lang: str) -> Optional[float]:
    return _calc_wrdhyp(sentences, lang)["WRDHYPnv"]

# List has been generated by ChatGPT 4o based on the Coh-Metrix index definitions
connectives_list = {
    "Causal": {
        "en": {"because", "since", "so", "therefore", "thus", "as", "due to", "consequently", "hence"},
        "de": {"weil", "da", "denn", "also", "deshalb", "daher", "somit", "folglich", "aus", "wegen"}
    },
    "Logical": {
        "en": {"and", "or", "either", "neither", "not only", "but also"},
        "de": {"und", "oder", "entweder", "weder", "nicht nur", "sondern auch"}
    },
    "Adversative": {
        "en": {"although", "though", "whereas", "while", "however", "nevertheless", "but", "on the other hand"},
        "de": {"obwohl", "während", "hingegen", "jedoch", "trotzdem", "aber", "andererseits"}
    },
    "Temporal": {
        "en": {"when", "before", "after", "until", "since", "as soon as"},
        "de": {"wenn", "bevor", "nachdem", "bis", "seit", "sobald"}
    },
    "Expanded": {
        "en": {"at first", "eventually", "finally", "meanwhile", "in the meantime", "subsequently", "thereafter"},
        "de": {"zuerst", "schließlich", "endlich", "mittlerweile", "inzwischen", "anschließend", "danach"}
    },
    "Additive": {
        "en": {"and", "also", "in addition", "moreover", "furthermore", "besides"},
        "de": {"und", "auch", "außerdem", "zudem", "darüber hinaus", "ferner"}
    },
    "Positive": {
        "en": {"also", "moreover", "likewise", "similarly", "in addition"},
        "de": {"auch", "ebenso", "außerdem", "ebenso wie", "darüber hinaus"}
    },
    "Negative": {
        "en": {"however", "but", "on the contrary", "yet", "although", "nevertheless"},
        "de": {"jedoch", "aber", "hingegen", "dennoch", "obwohl", "trotzdem"}
    }
}

all_connectives_list = {
    "en": set(),
    "de": set()
}
for category, connectives in connectives_list.items():
    for lang, words in connectives.items():
        all_connectives_list[lang].update(words)

# H4 fix: previous impl used `text.count(conn)` which produced massive
# false-positive matches (e.g., counted \"since\" inside \"princess\", or
# German \"und\" inside \"Bund/Stunde\"). Now we tokenize on whitespace and do
# case-insensitive whole-word/phrase matching on a stripped-punct token stream.
# H10 (partial): the Additive/Positive lists overlap heavily with Logical/etc.
# by design of Coh-Metrix (CNCAll sums unique occurrences), so we dedupe
# occurrences for CNCAll by matching each token position at most once.
import re as _re_conn

def _normalize_tokens_for_connectives(text: str) -> List[str]:
    # Lowercase, split on whitespace, strip surrounding punctuation from each token.
    return [_re_conn.sub(r"^[^\w]+|[^\w]+$", "", t.lower()) for t in text.split()]

def _count_phrase_occurrences(tokens: List[str], phrase: str) -> int:
    phrase_tokens = phrase.lower().split()
    if not phrase_tokens:
        return 0
    n = len(phrase_tokens)
    count = 0
    for i in range(len(tokens) - n + 1):
        if tokens[i:i + n] == phrase_tokens:
            count += 1
    return count

def _count_connectives_in_doc(text: str, connectives_set) -> int:
    # H4 fix: proper whole-word/phrase counting via tokenization.
    tokens = _normalize_tokens_for_connectives(text)
    count = 0
    for conn in connectives_set:
        count += _count_phrase_occurrences(tokens, conn)
    return count

def _count_connectives(text: str, lang: str, total_words: int) -> Dict[str, float]:
    count_list = {
        "CNCAll": _count_connectives_in_doc(text, all_connectives_list[lang]),
        "CNCCaus": _count_connectives_in_doc(text, connectives_list["Causal"][lang]),
        "CNCLogic": _count_connectives_in_doc(text, connectives_list["Logical"][lang]),
        "CNCADC": _count_connectives_in_doc(text, connectives_list["Adversative"][lang]),
        "CNCTemp": _count_connectives_in_doc(text, connectives_list["Temporal"][lang]),
        "CNCTempX": _count_connectives_in_doc(text, connectives_list["Expanded"][lang]),
        "CNCAdd": _count_connectives_in_doc(text, connectives_list["Additive"][lang]),
        "CNCPos": _count_connectives_in_doc(text, connectives_list["Positive"][lang]),
        "CNCNeg": _count_connectives_in_doc(text, connectives_list["Negative"][lang])
    }
    count_list_per_1000_words = {k: (v / total_words * 1000) if total_words > 0 else 0 for k, v in count_list.items()}
    return count_list_per_1000_words

# ============================================================================
# CONNECTIVES (CNC*) — "How often are connecting words (and, because, however)
# used?" All are incidences per 1,000 words. Approximate: the word lists were
# LLM-generated and contain some double-counting across categories (NOTE(H10)).
# ============================================================================

# LAY: All connectives combined, per 1,000 words.
# ↑ Higher = more explicitly connected prose. Approximate (NOTE(H10)).
def cm_cncall(text: str, lang: str, tokens_count: int, connectives: Optional[Dict[str, float]] = None) -> Optional[float]:
    # L1: accept precomputed dict to avoid recomputing 9 times per request.
    if connectives is None:
        connectives = _count_connectives(text, lang, tokens_count)
    return connectives["CNCAll"]

# LAY: Causal connectives (because, since, therefore…) per 1,000 words.
# ↑ Higher = more cause-effect signalling. Approximate (NOTE(H10)).
def cm_cnccaus(text: str, lang: str, tokens_count: int, connectives: Optional[Dict[str, float]] = None) -> Optional[float]:
    if connectives is None:
        connectives = _count_connectives(text, lang, tokens_count)
    return connectives["CNCCaus"]

# LAY: Logical connectives (and, or, either…) per 1,000 words.
# ↑ Higher = more logical coordination. Approximate (NOTE(H10)).
def cm_cnclogic(text: str, lang: str, tokens_count: int, connectives: Optional[Dict[str, float]] = None) -> Optional[float]:
    if connectives is None:
        connectives = _count_connectives(text, lang, tokens_count)
    return connectives["CNCLogic"]

# LAY: Adversative/contrastive connectives (but, however, although…) per 1,000 words.
# ↑ Higher = more contrastive discourse. Approximate (NOTE(H10)).
def cm_cncadc(text: str, lang: str, tokens_count: int, connectives: Optional[Dict[str, float]] = None) -> Optional[float]:
    if connectives is None:
        connectives = _count_connectives(text, lang, tokens_count)
    return connectives["CNCADC"]

# LAY: Temporal connectives (when, before, after…) per 1,000 words.
# ↑ Higher = more time-sequence signalling. Approximate (NOTE(H10)).
def cm_cnctemp(text: str, lang: str, tokens_count: int, connectives: Optional[Dict[str, float]] = None) -> Optional[float]:
    if connectives is None:
        connectives = _count_connectives(text, lang, tokens_count)
    return connectives["CNCTemp"]

# LAY: Expanded temporal connectives (at first, finally, meanwhile…) per 1,000 words.
# ↑ Higher = richer narrative time markers. Approximate (NOTE(H10)).
def cm_cnctempx(text: str, lang: str, tokens_count: int, connectives: Optional[Dict[str, float]] = None) -> Optional[float]:
    if connectives is None:
        connectives = _count_connectives(text, lang, tokens_count)
    return connectives["CNCTempX"]

# LAY: Additive connectives (and, also, moreover, furthermore…) per 1,000 words.
# ↑ Higher = more information stacking. Approximate (NOTE(H10)).
def cm_cncadd(text: str, lang: str, tokens_count: int, connectives: Optional[Dict[str, float]] = None) -> Optional[float]:
    if connectives is None:
        connectives = _count_connectives(text, lang, tokens_count)
    return connectives["CNCAdd"]

# LAY: Positive connectives (also, likewise, similarly…) per 1,000 words.
# ↑ Higher = more reinforcement/agreement signalling. Approximate (NOTE(H10)).
def cm_cncpos(text: str, lang: str, tokens_count: int, connectives: Optional[Dict[str, float]] = None) -> Optional[float]:
    if connectives is None:
        connectives = _count_connectives(text, lang, tokens_count)
    return connectives["CNCPos"]

# LAY: Negative connectives (however, but, yet, on the contrary…) per 1,000 words.
# ↑ Higher = more contrast/opposition signalling. Approximate (NOTE(H10)).
def cm_cncneg(text: str, lang: str, tokens_count: int, connectives: Optional[Dict[str, float]] = None) -> Optional[float]:
    if connectives is None:
        connectives = _count_connectives(text, lang, tokens_count)
    return connectives["CNCNeg"]

def _get_paragraph_token_vectors(paragraphs: List[Paragraph]) -> Tuple[List[List[List[List[float]]]], List[List[List[str]]]]:
    token_vectors = []
    token_words = []
    for p in paragraphs:
        vectors = []
        words = []
        for s in p.sentences:
            vectors.append([
                token.vector if token.has_vector else None
                for token in s.tokens
            ])
            words.append([token.text for token in s.tokens])
        token_vectors.append(vectors)
        token_words.append(words)
    return token_vectors, token_words

def _sentence_vector(token_has_vector, words, tokens_vector_length: int):
    vectors = []
    for j, word in enumerate(words):
        vector_i = token_has_vector[j]
        if word.isalpha() and vector_i is not None:
            vectors.append(vector_i)
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(tokens_vector_length)

# NOTE(M5): LSA approximation. The original Coh-Metrix LSA indices are built on
# a ~300-dim LSA space trained on the TASA corpus (Landauer et al., 2007). This
# implementation uses the pipeline's spaCy word vectors, averages per sentence/
# paragraph, and applies sklearn TruncatedSVD to 100 dimensions (see also L3).
# Absolute values differ from the original; rank-ordering is expected to be
# similar on the same text. No TASA-trained weights are shipped with this
# container.
def _reduce_dimensionality(vectors, n_components=100):
    # L12 fix: TruncatedSVD requires n_components < min(n_samples, n_features).
    # Previous code only capped at n_features-1, which caused ValueError on
    # short texts (e.g. 2 sentences × 300 features requesting 100 components).
    # The outer try/except swallowed the error and nulled all 8 LSA indices.
    if vectors.shape[0] == 0:
        return vectors
    n_samples = vectors.shape[0]
    n_features = vectors.shape[1]
    effective = min(n_components, n_samples - 1, n_features - 1)
    if effective < 1:
        # Not enough samples/features to reduce meaningfully; return as-is.
        return vectors
    svd = TruncatedSVD(n_components=effective)
    reduced = svd.fit_transform(vectors)
    return reduced

# NOTE(LSA-GivenNew): LSAGN / LSAGNd (Appendix A indices 44–45) measure how
# much of each sentence's meaning is already "given" by preceding sentences.
# Hempelmann et al. (2005) / Landauer et al. (2007) define it geometrically:
# project the current sentence vector v_i onto the linear subspace spanned by
# previous sentence vectors {v_0..v_{i-1}}; the projection's length is the
# "given" component G, the orthogonal residual's length is the "new" component
# N, and the index returned is G / (G + N) ∈ [0, 1]. The first sentence has no
# prior context, so G = 0 and the ratio is 0. QR decomposition of the basis
# matrix gives an orthonormal Q; Q @ (Q.T @ v) is the orthogonal projection.
# Caveat: the implementation uses spaCy word vectors (see NOTE(M5)), not TASA-
# trained LSA, so absolute values differ from the reference Coh-Metrix output.
# @see docs/review-report-v3.html (M5, L3) for the LSA approximation context.
def _project_onto_hyperplane(v, basis):
    if basis.shape[0] == 0:
        return np.zeros_like(v), v
    Q, _ = np.linalg.qr(basis.T)
    p = Q @ (Q.T @ v)
    return p, v - p

def _lsa_given_new_for_vectors(vectors):
    results = []
    for i in range(len(vectors)):
        current_vec = vectors[i]
        if i == 0:
            G = 0
            N = np.linalg.norm(current_vec)
        else:
            basis = vectors[:i]
            p, perp = _project_onto_hyperplane(current_vec, basis)
            G = np.linalg.norm(p)
            N = np.linalg.norm(perp)
        given_new_ratio = G / (G + N) if (G + N) > 0 else 0
        results.append(given_new_ratio)
    return np.array(results)

def _lsa_cohesion_indices(vec_per_paragraph_sentences: List[List[List[List[float]]]], words: List[List[List[str]]], tokens_vector_length: int, n_components) -> Dict[str, Any]:
    all_sentences = []
    sentence_vectors = []
    paragraph_vectors = []
    sentences_per_paragraph = []
    for c, para in enumerate(vec_per_paragraph_sentences):
        sentences_per_paragraph.append(len(para))
        all_sentences.extend(para)
        sent_vecs = np.array([_sentence_vector(sent, sent_words, tokens_vector_length) for sent, sent_words in zip(para, words[c]) if sent])
        sentence_vectors.append(sent_vecs)
        # paragraph vector = mean of sentence vectors
        if len(sent_vecs) > 0:
            paragraph_vectors.append(np.mean(sent_vecs, axis=0))
        else:
            paragraph_vectors.append(np.zeros(tokens_vector_length))

    # Concatenate all sentence vectors
    sentence_vectors_all = np.vstack(sentence_vectors) if sentence_vectors else np.empty((0, tokens_vector_length))
    paragraph_vectors = np.array(paragraph_vectors)

    # Reduce dimensionality (LSA)
    sentence_vectors_reduced = _reduce_dimensionality(sentence_vectors_all, n_components)
    paragraph_vectors_reduced = _reduce_dimensionality(paragraph_vectors, n_components)

    # --- LSA similarity between adjacent sentences ---
    adj_sent_sim = []
    for i in range(len(sentence_vectors_reduced) - 1):
        sim = cosine_similarity([sentence_vectors_reduced[i]], [sentence_vectors_reduced[i + 1]])[0][0]
        adj_sent_sim.append(sim)
    adj_sent_sim = np.array(adj_sent_sim)

    # --- LSA similarity between all sentence pairs in paragraphs ---
    all_sent_pairs_sim = []
    idx = 0
    for count in sentences_per_paragraph:
        if count > 1:
            sent_vecs = sentence_vectors_reduced[idx:idx + count]
            sim_matrix = cosine_similarity(sent_vecs)
            # Take upper triangle excluding diagonal
            triu_indices = np.triu_indices(count, k=1)
            sims = sim_matrix[triu_indices]
            all_sent_pairs_sim.extend(sims)
        idx += count
    all_sent_pairs_sim = np.array(all_sent_pairs_sim)

    # --- LSA similarity between adjacent paragraphs ---
    adj_para_sim = []
    for i in range(len(paragraph_vectors_reduced) - 1):
        sim = cosine_similarity([paragraph_vectors_reduced[i]], [paragraph_vectors_reduced[i + 1]])[0][0]
        adj_para_sim.append(sim)
    adj_para_sim = np.array(adj_para_sim)

    given_new_ratios = _lsa_given_new_for_vectors(sentence_vectors_reduced)

    return {
        'LSASS1': np.mean(adj_sent_sim) if adj_sent_sim.size > 0 else np.nan,
        'LSASS1d': np.std(adj_sent_sim) if adj_sent_sim.size > 0 else np.nan,
        'LSASSp': np.mean(all_sent_pairs_sim) if all_sent_pairs_sim.size > 0 else np.nan,
        'LSASSpd': np.std(all_sent_pairs_sim) if all_sent_pairs_sim.size > 0 else np.nan,
        'LSAPP1': np.mean(adj_para_sim) if adj_para_sim.size > 0 else np.nan,
        'LSAPP1d': np.std(adj_para_sim) if adj_para_sim.size > 0 else np.nan,
        'LSAGN': np.mean(given_new_ratios) if given_new_ratios.size > 0 else np.nan,
        'LSAGNd': np.std(given_new_ratios) if given_new_ratios.size > 0 else np.nan,
    }

# ============================================================================
# LSA (LSA*) — "Are ideas in nearby sentences similar in meaning?"
# Latent Semantic Analysis overlap scores. 0 = unrelated, 1 = identical.
# Approximate: uses spaCy word vectors + TruncatedSVD(100) instead of the
# original TASA-trained 300-dim space (NOTE(M5)). Rank-order should track,
# absolute values are not comparable to published Coh-Metrix output.
# ============================================================================

# LAY: Average meaning-similarity between adjacent sentences.
# ↑ Higher = smoother topic flow. Approximate (NOTE(M5)).
# ============================================================================
# LSA (LSA*) — "Are ideas in nearby sentences similar in meaning?"
# Latent Semantic Analysis overlap scores. 0 = unrelated, 1 = identical.
# Approximate: uses spaCy word vectors + TruncatedSVD(100) instead of the
# original TASA-trained 300-dim space (NOTE(M5)). Rank-order should track,
# absolute values are not comparable to published Coh-Metrix output.
# ============================================================================

# LAY: Average meaning-similarity between adjacent sentences.
# ↑ Higher = smoother topic flow. Approximate (NOTE(M5)).
def cm_lsass1(lsa_indices: Dict[str, Any]) -> Optional[float]:
    return lsa_indices["LSASS1"]

# LAY: How uneven is the meaning-similarity of adjacent sentences?
# ↑ Higher = bumpy topic transitions. Approximate (NOTE(M5)).
# LAY: How uneven is the meaning-similarity of adjacent sentences?
# ↑ Higher = bumpy topic transitions. Approximate (NOTE(M5)).
def cm_lsass1d(lsa_indices: Dict[str, Any]) -> Optional[float]:
    return lsa_indices["LSASS1d"]

# LAY: Average meaning-similarity among all sentences within each paragraph.
# ↑ Higher = internally coherent paragraphs. Approximate (NOTE(M5)).
# LAY: Average meaning-similarity among all sentences within each paragraph.
# ↑ Higher = internally coherent paragraphs. Approximate (NOTE(M5)).
def cm_lsassp(lsa_indices: Dict[str, Any]) -> Optional[float]:
    return lsa_indices["LSASSp"]

# LAY: How uneven is the within-paragraph meaning-similarity?
# ↑ Higher = mix of tight and loose paragraphs. Approximate (NOTE(M5)).
# LAY: How uneven is the within-paragraph meaning-similarity?
# ↑ Higher = mix of tight and loose paragraphs. Approximate (NOTE(M5)).
def cm_lsasspd(lsa_indices: Dict[str, Any]) -> Optional[float]:
    return lsa_indices["LSASSpd"]

# LAY: Average meaning-similarity between adjacent paragraphs.
# ↑ Higher = smooth flow across paragraphs. Approximate (NOTE(M5)).
# LAY: Average meaning-similarity between adjacent paragraphs.
# ↑ Higher = smooth flow across paragraphs. Approximate (NOTE(M5)).
def cm_lsapp1(lsa_indices: Dict[str, Any]) -> Optional[float]:
    return lsa_indices["LSAPP1"]

# LAY: How uneven is the meaning-similarity of adjacent paragraphs?
# ↑ Higher = abrupt topic shifts between paragraphs. Approximate (NOTE(M5)).
# LAY: How uneven is the meaning-similarity of adjacent paragraphs?
# ↑ Higher = abrupt topic shifts between paragraphs. Approximate (NOTE(M5)).
def cm_lsapp1d(lsa_indices: Dict[str, Any]) -> Optional[float]:
    return lsa_indices["LSAPP1d"]

# LAY: How much NEW information each sentence adds, on average ("given/new" ratio).
# ↑ Higher = each sentence introduces more novel content. Approximate (NOTE(M5), NOTE(LSA-GivenNew)).
# LAY: How much NEW information each sentence adds, on average ("given/new" ratio).
# ↑ Higher = each sentence introduces more novel content. Approximate (NOTE(M5), NOTE(LSA-GivenNew)).
def cm_lsagn(lsa_indices: Dict[str, Any]) -> Optional[float]:
    return lsa_indices["LSAGN"]

# LAY: How uneven is the given/new ratio across sentences?
# ↑ Higher = mix of reinforcing and information-dense sentences. Approximate.
# LAY: How uneven is the given/new ratio across sentences?
# ↑ Higher = mix of reinforcing and information-dense sentences. Approximate.
def cm_lsagnd(lsa_indices: Dict[str, Any]) -> Optional[float]:
    return lsa_indices["LSAGNd"]

def _load_word_frequencies(path: str, lowercase_words=True) -> Dict[str, int]:
    word_freq = {}
    with open(path, "r", encoding="utf-8") as file:
        skipped_first_line = False
        for line in file:
            if not skipped_first_line:
                skipped_first_line = True
                continue
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                word, freq = parts
                if lowercase_words:
                    word = word.lower()
                word_freq[word] = int(freq)
    return word_freq

all_word_frequencies_map = {
    "en": {
        "wiki-20220301-sample10000": _load_word_frequencies("src/main/resources/word_frequencies_en_enwiki-20220301-sample10000.csv"),
    },
    "de": {
        "wiki-20220301-sample10000": _load_word_frequencies("src/main/resources/word_frequencies_de_dewiki-20220301-sample10000.csv"),
    }
}

# NOTE(L9): Word-frequency resource used here is
# wiki-20220301-sample10000 (~10k Wikipedia articles) rather than the
# original CELEX reference corpus (17.9M words, paywalled). Absolute
# frequency values and their logs differ from the reference Coh-Metrix
# output; rank-ordering among content words is expected to be broadly
# similar. The standard V3 WRDFRQc/a/mc outputs are stubbed (None) on
# purpose (see L6); the Wikipedia-backed alternatives are exposed with a
# `label_ttlab` suffix.
# ============================================================================
# WORD FREQUENCY (WRDFRQ*) & L2 READABILITY (RDL2)
# Approximate: the original Coh-Metrix uses the CELEX lexical database, which
# is paywalled. This implementation uses a 10,000-article Wikipedia sample
# (NOTE(L9)). Standard outputs WRDFRQc/a/mc remain None; the _wiki10000
# suffix variants are emitted instead. RDL2 is the "L2 Readability" composite
# formula (Crossley et al. 2008) and is emitted twice — using SYNSTRUTa and
# using SYNSTRUTt — because the spec does not disambiguate (NOTE(L7)).
# ============================================================================

# LAY: Average frequency of content words in everyday text.
# ↑ Higher = uses common words. Approximate (Wikipedia sample — NOTE(L9)).
# ============================================================================
# WORD FREQUENCY (WRDFRQ*) & L2 READABILITY (RDL2)
# Approximate: the original Coh-Metrix uses the CELEX lexical database, which
# is paywalled. This implementation uses a 10,000-article Wikipedia sample
# (NOTE(L9)). Standard outputs WRDFRQc/a/mc remain None; the _wiki10000
# suffix variants are emitted instead. RDL2 is the "L2 Readability" composite
# formula (Crossley et al. 2008) and is emitted twice — using SYNSTRUTa and
# using SYNSTRUTt — because the spec does not disambiguate (NOTE(L7)).
# ============================================================================

# LAY: Average frequency of content words in everyday text.
# ↑ Higher = uses common words. Approximate (Wikipedia sample — NOTE(L9)).
def cm_wrdfrqc(sentences: List[Sentence], lang: str, frequencies_source: str) -> Optional[float]:
    word_frequencies_map = all_word_frequencies_map[lang][frequencies_source]
    content_words, _, _ = _get_content_words(sentences)
    word_frequencies = [
        word_frequencies_map.get(word, 0)
        for word in content_words
    ]
    return np.mean(word_frequencies) if word_frequencies else 0.0

# LAY: Average log-frequency of ALL words in everyday text.
# ↑ Higher = text leans on common vocabulary. Approximate (NOTE(L9)).
# LAY: Average log-frequency of ALL words in everyday text.
# ↑ Higher = text leans on common vocabulary. Approximate (NOTE(L9)).
def cm_wrdfrqa(tokens: List[Token], lang: str, frequencies_source: str) -> Optional[float]:
    word_frequencies_map = all_word_frequencies_map[lang][frequencies_source]
    word_frequencies = [
        word_frequencies_map.get(word.text, 0)
        for word in tokens
    ]
    log_word_frequencies = [
        np.log(freq + 1e-5)  # smoothing to avoid log(0)
        for freq in word_frequencies
    ]
    return np.mean(log_word_frequencies) if log_word_frequencies else 0.0

# LAY: Average MINIMUM log-frequency among content words (rarest word per sentence).
# ↑ Higher = even the rarest content words are reasonably common. Approximate (NOTE(L9)).
# LAY: Average MINIMUM log-frequency among content words (rarest word per sentence).
# ↑ Higher = even the rarest content words are reasonably common. Approximate (NOTE(L9)).
def cm_wrdfrqmc(sentences: List[Sentence], lang: str, frequencies_source: str) -> Optional[float]:
    # CELEX Log minimum frequency for content words, mean
    # -> across sentences
    word_frequencies_map = all_word_frequencies_map[lang][frequencies_source]
    content_words, _, _ = _get_content_words_per_sentence(sentences)
    sentence_min_frequencies = []
    for sentence in content_words:
        try:
            word_frequencies = [
                word_frequencies_map.get(word, 0)
                for word in sentence
            ]
            log_word_frequencies = [
                np.log(freq + 1e-5)  # smoothing to avoid log(0)
                for freq in word_frequencies
            ]
            min_freq = np.min(log_word_frequencies)
            sentence_min_frequencies.append(min_freq)
        except:
            # ignore problems when processing sentences
            pass
    return np.mean(sentence_min_frequencies) if sentence_min_frequencies else 0.0

# LAY: Second-language readability composite score (Crossley et al. 2008).
# ↑ Higher = easier for L2 learners. Approximate (NOTE(L7) — emitted twice).
# LAY: Second-language readability composite score (Crossley et al. 2008).
# ↑ Higher = easier for L2 learners. Approximate (NOTE(L7) — emitted twice).
def cm_rdl2(crfcwo1: float, synstrut: float, wrdfrqmc: float) -> Optional[float]:
    # RDL2: Coh-Metrix L2 Readability (Appendix A index 106).
    # Formula + constants from Crossley, Greenfield & McNamara (2008), as cited
    # in the Coh-Metrix 3.0 spec (Ch. 5 §Readability / L2 readability formula):
    #   RDL2 = -45.032 + 52.230·CRFCWO1 + 61.306·SYNSTRUT + 22.205·WRDFRQmc
    # Note the spec writes "SYNSTRUT" without disambiguation; this service
    # computes two variants (RDL2_synstruta using SYNSTRUTa, RDL2_synstrutt
    # using SYNSTRUTt) — see NOTE(L7).
    # L8 fix: make the None-path explicit instead of relying on the caller's
    # try/except to swallow a TypeError from None * float.
    if crfcwo1 is None or synstrut is None or wrdfrqmc is None:
        return None
    l2 = -45.032 + (52.230 * crfcwo1) + (61.306 * synstrut) + (22.205 * wrdfrqmc)
    return l2

def _sm_get_data(sentences: List[Sentence]):
    words = []
    tags = []
    morph_tense = []
    lemmas = []
    poses = []
    vectors = []
    for sent in sentences:
        # For each sentence, get the vectors of the tokens
        vectors.append([token.vector if token.has_vector else None for token in sent.tokens])
        words.append([token.text for token in sent.tokens])
        tags.append([token.pos_value for token in sent.tokens])
        morph_tense.append([token.morph_tense for token in sent.tokens])
        lemmas.append([token.lemma for token in sent.tokens])
        poses.append([token.pos_coarse for token in sent.tokens])

    return words, tags, morph_tense, lemmas, poses, vectors

def count_verbs(poses: List[List[str]], words: List[List[str]], lemmas: List[List[str]], tags: List[List[str]], morph_tense: List[List[str]], lang,causal_practical_set):
    counters = {
        "causal_verbs": 0,
        "intentional_verbs": 0,
        "causal_particles": 0,
        "intentional_particles": 0
    }
    tenses = []

    for i, sent in enumerate(poses):
        for j, pos in enumerate(sent):
            word_i = words[i][j].lower()
            lemma_i = lemmas[i][j].lower()
            tag_i = tags[i][j]
            morph_i = morph_tense[i][j]
            if pos == "VERB":
                if lemma_i in causal_practical_set["causal_verbs"]:
                    counters["causal_verbs"] += 1
                if lemma_i in causal_practical_set["intentional_verbs"]:
                    counters["intentional_verbs"] += 1
            if word_i in causal_practical_set["causal_particles"]:
                counters["causal_particles"] += 1
            if word_i in causal_practical_set["intentional_particles"]:
                counters["intentional_particles"] += 1

            if lang=="en":
                match tag_i:
                    case "VBD" | "VBN":
                        tenses.append("past")
                    case "VB" | "VBP" | "VBZ":
                        tenses.append("present")
                    case "VBG":
                        tenses.append("progressive")
                    case "MD":
                        tenses.append("modal")
                    case _:
                        tenses.append("other")
            elif lang=="de":
                if morph_i:
                    tenses.append(morph_i[0].lower())
                else:
                    tenses.append("other")
            else:
                tenses.append("other")
    return {
        "counter": counters,
        "tenses": tenses
    }

def _get_hyponyms(synset):
    hypos = set()
    for hypo in synset.hyponyms():
        hypos.add(hypo)
        hypos |= _get_hyponyms(hypo)
    return hypos

def _get_verb_lemmas_for_synset(synset):
    verbs = set(synset.lemma_names())
    for hypo in _get_hyponyms(synset):
        verbs |= set(hypo.lemma_names())
    return {v.replace('_', ' ') for v in verbs}

def _germanet_all_hyponyms(synset) -> set:
    """Recursively collect all GermaNet hyponyms of a synset.
    The germanetpy API exposes direct hyponyms either as a property
    (`direct_hyponyms`) or via `causes`/`entailments` \u2014 we probe both.
    """
    hypos = set()
    # germanetpy Synset: `direct_hyponyms` is a set attribute.
    direct = getattr(synset, "direct_hyponyms", None)
    if direct is None:
        return hypos
    try:
        direct_iter = list(direct)
    except TypeError:
        return hypos
    for hypo in direct_iter:
        if hypo in hypos:
            continue
        hypos.add(hypo)
        hypos |= _germanet_all_hyponyms(hypo)
    return hypos

def _germanet_expand_verb_lemmas(orthforms) -> set:
    """Given one or more German verb lemmas, return the transitive closure of
    hyponym lemmas across all matching verb synsets in GermaNet. Returns an
    empty set on any failure (caller should use a seed fallback)."""
    if germanet is None:
        return set()
    lemmas: set = set()
    try:
        for ortho in orthforms:
            synsets = germanet.get_synsets_by_orthform(ortho)
            for syn in synsets:
                if getattr(syn, "word_category", None) is not None and \
                        syn.word_category != WordCategory.verben:
                    continue
                # include lemmas of this synset
                for lex in getattr(syn, "lexunits", []) or []:
                    of = getattr(lex, "orthform", None)
                    if of:
                        lemmas.add(of.lower())
                # recurse over hyponyms
                for hypo in _germanet_all_hyponyms(syn):
                    for lex in getattr(hypo, "lexunits", []) or []:
                        of = getattr(lex, "orthform", None)
                        if of:
                            lemmas.add(of.lower())
    except Exception as ex:
        logger.warning("GermaNet expansion failed: %s", ex)
        return set()
    return lemmas

# M11: precompute German causal/intentional verb sets from GermaNet at module
# load, mirroring the English WordNet expansion. Falls back to the previous
# hardcoded seeds if GermaNet is not available or the expansion fails.
# Dropped "folgen" from the causal seed \u2014 it means "to follow" and is not
# semantically causal.
_DE_CAUSAL_SEED = {"verursachen", "bewirken", "ausl\u00f6sen", "hervorrufen", "brechen",
                   "frieren", "schlagen", "bewegen", "treffen", "ausbrechen", "entdecken"}
_DE_INTENTIONAL_SEED = {"beabsichtigen", "planen", "wollen", "vorhaben", "kontaktieren",
                        "fallenlassen", "gehen", "sprechen", "kaufen", "erz\u00e4hlen",
                        "fahren", "entscheiden"}

_de_causal_verbs_expanded = _germanet_expand_verb_lemmas({"verursachen", "bewirken", "ausl\u00f6sen"})
_de_intentional_verbs_expanded = _germanet_expand_verb_lemmas({"beabsichtigen", "planen"})
if not _de_causal_verbs_expanded:
    _de_causal_verbs_expanded = set(_DE_CAUSAL_SEED)
    logger.info("DE causal verbs: using %d hardcoded seeds (GermaNet expansion unavailable)",
                len(_de_causal_verbs_expanded))
else:
    # always keep the seed terms as part of the set
    _de_causal_verbs_expanded |= _DE_CAUSAL_SEED
    logger.info("DE causal verbs: %d entries (GermaNet expansion)", len(_de_causal_verbs_expanded))
if not _de_intentional_verbs_expanded:
    _de_intentional_verbs_expanded = set(_DE_INTENTIONAL_SEED)
    logger.info("DE intentional verbs: using %d hardcoded seeds (GermaNet expansion unavailable)",
                len(_de_intentional_verbs_expanded))
else:
    _de_intentional_verbs_expanded |= _DE_INTENTIONAL_SEED
    logger.info("DE intentional verbs: %d entries (GermaNet expansion)", len(_de_intentional_verbs_expanded))

def _causal_practical_verbs_intentional(lang: str):
    cause_synset = wn.synset('cause.v.01')
    causal_verbs_en = _get_verb_lemmas_for_synset(cause_synset)
    intend_synset = wn.synset('intend.v.01')
    plan_synset = wn.synset('plan.v.01')
    intentional_verbs_en = _get_verb_lemmas_for_synset(intend_synset) | _get_verb_lemmas_for_synset(plan_synset)

    causal_particles_en_seed = ["because", "therefore", "since", "so", "thus", "hence", "in order to"]
    intentional_particles_en_seed = ["want", "need", "plan", "intend", "decide"]

    causal_particles_de_seed = ["weil", "deshalb", "daher", "darum", "folglich", "infolgedessen", "aus diesem Grund"]
    intentional_particles_de_seed = ["wollen", "planen", "beabsichtigen", "versuchen", "vorhaben"]

    # M11 fix: precomputed GermaNet-expanded sets (with seed fallback) replace
    # the previous 10-word hardcoded lists.
    causal_verbs_de = _de_causal_verbs_expanded
    intentional_verbs_de = _de_intentional_verbs_expanded

    if lang == "en":
        return {
            "causal_verbs": causal_verbs_en,
            "intentional_verbs": intentional_verbs_en,
            "causal_particles": causal_particles_en_seed,
            "intentional_particles": intentional_particles_en_seed
        }
    elif lang == "de":
        return {
            "causal_verbs": causal_verbs_de,
            "intentional_verbs": intentional_verbs_de,
            "causal_particles": causal_particles_de_seed,
            "intentional_particles": intentional_particles_de_seed
        }
    else:
        # TODO: no causal/intentional verb resources available for other languages
        return None

# ============================================================================
# SITUATION MODEL (SM*) — "Do verbs show causal / intentional relationships?"
# Counts and ratios for causal and intentional verbs + particles, verb overlap
# in meaning, and tense/aspect cohesion. Approximate:
#   • Verb lists are curated seeds (en) + GermaNet hyponym expansion (de) — NOTE(M11).
#   • SMCAUSlsa uses approximate LSA vectors — NOTE(M5).
#   • SMCAUSwn is reliable for en, partial for de (GermaNet) — NOTE(H9).
#   • SMTEMP interpretation is spec-ambiguous — NOTE(M2).
# ============================================================================

# LAY: Causal verbs (cause, break, move, …) per 1,000 words.
# ↑ Higher = more cause-describing verbs. Approximate (NOTE(M11)).
def cm_smcausv(sentences: List[Sentence], lang: str) -> Optional[float]:
    # SMCAUSv: Causal verb incidence (per 1000 words)
    # Spec: Coh-Metrix 3.0 Appendix A, index 59; Chapter 4 §Situation model
    causal_set = _causal_practical_verbs_intentional(lang)
    if causal_set is None:
        return None
    words, tags, morph_tense, lemmas, poses, _ = _sm_get_data(sentences)
    count_tenses = count_verbs(poses, words, lemmas, tags, morph_tense, lang, causal_set)
    total_words = sum(len(s) for s in poses)
    count_causal_verb = count_tenses["counter"]["causal_verbs"]
    return _incidence(count_causal_verb, total_words)

# LAY: Causal verbs + causal connectives (because, in order to…) per 1,000 words.
# ↑ Higher = more explicit causal structure. Approximate (NOTE(M11), NOTE(H10)).
def cm_smcausvp(sentences: List[Sentence], lang: str) -> Optional[float]:
    # SMCAUSvp: Causal verbs and causal particles incidence (per 1000 words)
    # Spec: Coh-Metrix 3.0 Appendix A, index 60; Chapter 4: "causal verbs plus
    # causal particles (SMCAUSvp: e.g., both causal verbs and connectives such
    # as because, in order to)". Earlier impl summed intentional_verbs by mistake.
    causal_set = _causal_practical_verbs_intentional(lang)
    if causal_set is None:
        return None
    words, tags, morph_tense, lemmas, poses, _ = _sm_get_data(sentences)
    count_tenses = count_verbs(poses, words, lemmas, tags, morph_tense, lang, causal_set)
    total_words = sum(len(s) for s in poses)
    count_causal_verb = count_tenses["counter"]["causal_verbs"]
    count_causal_particle = count_tenses["counter"]["causal_particles"]
    return _incidence(count_causal_verb + count_causal_particle, total_words)

# LAY: Intentional verbs (contact, drop, plan, want…) per 1,000 words.
# ↑ Higher = more goal/intention language. Approximate (NOTE(M11)).
def cm_smintep(sentences: List[Sentence], lang: str) -> Optional[float]:
    # SMINTEp: Intentional verbs incidence (per 1000 words)
    # Spec: Coh-Metrix 3.0 Appendix A, index 61; Chapter 4: "intentional verbs
    # (SMINTEp: e.g., contact, drop, walk, talk)". Earlier impl returned
    # intentional_particles raw count by mistake.
    causal_set = _causal_practical_verbs_intentional(lang)
    if causal_set is None:
        return None
    words, tags, morph_tense, lemmas, poses, _ = _sm_get_data(sentences)
    count_tenses = count_verbs(poses, words, lemmas, tags, morph_tense, lang, causal_set)
    total_words = sum(len(s) for s in poses)
    count_intentional_verb = count_tenses["counter"]["intentional_verbs"]
    return _incidence(count_intentional_verb, total_words)

# LAY: Ratio of causal particles (connectives) to causal verbs.
# ↑ Higher = causality signalled more by connectives than verbs. Approximate.
def cm_smcausr(sentences: List[Sentence], lang: str) -> Optional[float]:
    causal_set = _causal_practical_verbs_intentional(lang)
    if causal_set is None:
        return None
    words, tags, morph_tense, lemmas, poses, _ = _sm_get_data(sentences)
    count_tenses = count_verbs(poses, words, lemmas, tags, morph_tense, lang, causal_set)
    count_causal_verb = count_tenses["counter"]["causal_verbs"]
    count_causal_particle = count_tenses["counter"]["causal_particles"]
    SMCAUSr = count_causal_particle / count_causal_verb if count_causal_verb > 0 else 0
    return np.round(SMCAUSr, 3)

# LAY: Ratio of intentional particles to intentional verbs.
# ↑ Higher = intentions signalled more by particles than verbs. Approximate.
def cm_sminter(sentences: List[Sentence], lang: str) -> Optional[float]:
    causal_set = _causal_practical_verbs_intentional(lang)
    if causal_set is None:
        return None
    words, tags, morph_tense, lemmas, poses, _ = _sm_get_data(sentences)
    count_tenses = count_verbs(poses, words, lemmas, tags, morph_tense, lang, causal_set)
    count_intentional_particle = count_tenses["counter"]["intentional_particles"]
    count_intentional_verb = count_tenses["counter"]["intentional_verbs"]
    SMINTEr = count_intentional_particle / count_intentional_verb if count_intentional_verb > 0 else 0
    return np.round(SMINTEr, 3)

def get_SMCAUSlsa(poses: List[List[str]], vectors: List[List[List[Any]]]):
    all_verbs = []
    for i, sent in enumerate(poses):
        for j, pos in enumerate(sent):
            if pos == "VERB":
                if vectors[i][j] is not None:
                    all_verbs.append(vectors[i][j])

    # M3 fix: previously used adjacent pairs only. Spec description is identical
    # to SMCAUSwn, and LSASSp also uses all pairs \u2014 harmonize SMCAUSlsa to
    # all unordered verb pairs.
    cos_similarities = []
    for i in range(len(all_verbs)):
        vec_i = all_verbs[i]
        for j in range(i + 1, len(all_verbs)):
            vec_j = all_verbs[j]
            if vec_i is not None and vec_j is not None:
                denom = np.linalg.norm(vec_i) * np.linalg.norm(vec_j)
                if denom == 0:
                    continue
                cos_sim = np.dot(vec_i, vec_j) / denom
                cos_similarities.append(cos_sim)

    return np.mean(cos_similarities) if cos_similarities else 0.0

# LAY: Average meaning-similarity of all verb pairs via LSA vectors.
# ↑ Higher = verbs share semantic field. Approximate (NOTE(M5)).
def cm_smcauslsa(sentences: List[Sentence]) -> Optional[float]:
    _, _, _, _, poses, vectors = _sm_get_data(sentences)
    SMCAUSlsa = get_SMCAUSlsa(poses, vectors)
    return np.round(SMCAUSlsa, 3)

def get_SMCAUSwn(poses: List[List[str]], word_lemma: List[List[str]], lang: str):
    if lang=="en":
        verbs_lemma = []
        syn_overlap_count = 0
        total_pairs = 0
        for i, sent in enumerate(poses):
            for j, pos in enumerate(sent):
                if pos == "VERB":
                    lemma = word_lemma[i][j].lower()
                    verbs_lemma.append(lemma)
        for i, lemma in enumerate(verbs_lemma):
            synsets_i = wn.synsets(lemma, pos=wn.VERB)
            for j in range(i + 1, len(verbs_lemma)):
                synsets_j = wn.synsets(verbs_lemma[j], pos=wn.VERB)
                total_pairs = total_pairs + 1
                if synsets_i and synsets_j and set(synsets_i).intersection(synsets_j):
                    syn_overlap_count += 1
        SMCAUSwn = syn_overlap_count / total_pairs if total_pairs > 0 else 0
    elif lang=="de":
        if  germanet is None:
            logger.warning("GermaNet not available")
            SMCAUSwn = -1.0
        else:
            verbs_lemma = []
            syn_overlap_count = 0
            total_pairs = 0
            for i, sent in enumerate(poses):
                for j, pos in enumerate(sent):
                    if pos == "VERB":
                        lemma = word_lemma[i][j].lower()
                        verbs_lemma.append(lemma)
            for i, lemma in enumerate(verbs_lemma):
                synsets_i = set(filter(lambda ss: ss.word_category==WordCategory.verben, germanet.get_synsets_by_orthform(lemma)))
                for j in range(i + 1, len(verbs_lemma)):
                    synsets_j = set(filter(lambda ss: ss.word_category==WordCategory.verben, germanet.get_synsets_by_orthform(verbs_lemma[j])))
                    total_pairs = total_pairs + 1
                    if synsets_i and synsets_j and synsets_i.intersection(synsets_j):
                        syn_overlap_count += 1
            SMCAUSwn = syn_overlap_count / total_pairs if total_pairs > 0 else 0
    else:
        SMCAUSwn = -1.0
    return SMCAUSwn

# LAY: Share of verb pairs that share a WordNet synset (semantic overlap).
# ↑ Higher = verbs cluster around shared meanings. Partial for DE (NOTE(H9)).
def cm_smcauswn(sentences: List[Sentence], lang) -> Optional[float]:
    _, _, _, lemmas, poses, vectors = _sm_get_data(sentences)
    SMCAUSwn = get_SMCAUSwn(poses, lemmas, lang)
    return np.round(SMCAUSwn, 3)

# LAY: How consistently tense/aspect is preserved across verbs in the text.
# ↑ Higher = more tense stability (typically indicates narrative cohesion).
#   Approximate: spec is ambiguous about the exact formula (NOTE(M2)).
def cm_smtemp(sentences: List[Sentence], lang: str) -> Optional[float]:
    # NOTE(M2): Interpretation of "temporal cohesion decreases as tense/aspect
    # shifts are encountered" as the proportion of verbs with the dominant
    # tense. This is a reasonable monotonic proxy: a text in a single tense
    # scores 1.0; a text with uniformly distributed tenses scores ~1/k. An
    # alternative would be 1 - (adjacent-transition rate). The original
    # Coh-Metrix algorithm is not publicly specified at the formula level.
    causal_set = _causal_practical_verbs_intentional(lang)
    if causal_set is None:
        return None
    words, tags, morph_tense, lemmas, poses, _ = _sm_get_data(sentences)
    count_tenses = count_verbs(
        poses,
        words,
        lemmas,
        tags,
        morph_tense,
        lang,
        causal_set
    )
    tense_distribution = Counter(count_tenses["tenses"])
    tenses = count_tenses["tenses"]
    dominant_tense_freq = tense_distribution.most_common(1)[0][1] if tense_distribution else 0
    SMTEMP = dominant_tense_freq / len(tenses) if len(tenses) > 0 else 0
    return np.round(SMTEMP, 3)

@app.post("/v1/process")
def post_process(request: TextImagerRequest) -> TextImagerResponse:
    modification_timestamp_seconds = int(time())

    indices = []
    meta = None
    modification_meta = None

    try:
        textstat.set_lang(request.language)

        sentences = []
        for p in request.paragraphs:
            sentences.extend(p.sentences)

        tokens = []
        for s in sentences:
            tokens.extend(s.tokens)
        tokens_count = len(tokens)

        tokens_vector_length = 0
        for t in tokens:
            if t.has_vector:
                tokens_vector_length = len(t.vector)
                break

        ### Descriptive

        # DESPC
        try:
            despc = cm_despc(request.paragraphs)
            despc_error = None
        except Exception as e:
            logger.error("Error calculating DESPC: %s", e)
            despc = None
            despc_error = str(e)
        indices.append(Index(
            index=1,
            type_name="Descriptive",
            label_v3="DESPC",
            label_v2="READNP",
            description="Paragraph count, number of paragraphs",
            value=despc,
            error=despc_error
        ))

        # DESSC
        try:
            dessc = cm_dessc(request.paragraphs)
            dessc_error = None
        except Exception as e:
            logger.error("Error calculating DESSC: %s", e)
            dessc = None
            dessc_error = str(e)
        indices.append(Index(
            index=2,
            type_name="Descriptive",
            label_v3="DESSC",
            label_v2="READNS",
            description="Sentence count, number of sentences",
            value=dessc,
            error=dessc_error
        ))

        # DESWC
        try:
            deswc = cm_deswc(request.paragraphs)
            deswc_error = None
        except Exception as e:
            logger.error("Error calculating DESWC: %s", e)
            deswc = None
            deswc_error = str(e)
        indices.append(Index(
            index=3,
            type_name="Descriptive",
            label_v3="DESWC",
            label_v2="READNW",
            description="Word count, number of words",
            value=deswc,
            error=deswc_error
        ))

        # DESPL
        try:
            despl = cm_despl(request.paragraphs)
            despl_error = None
        except Exception as e:
            logger.error("Error calculating DESPL: %s", e)
            despl = None
            despl_error = str(e)
        indices.append(Index(
            index=4,
            type_name="Descriptive",
            label_v3="DESPL",
            label_v2="READAPL",
            description="Paragraph length, number of sentences, mean",
            value=despl,
            error=despl_error
        ))

        # DESPLd
        try:
            despld = cm_despld(request.paragraphs)
            despld_error = None
        except Exception as e:
            logger.error("Error calculating DESPLd: %s", e)
            despld = None
            despld_error = str(e)
        indices.append(Index(
            index=5,
            type_name="Descriptive",
            label_v3="DESPLd",
            label_v2="n/a",
            description="Paragraph length, number of sentences, standard deviation",
            value=despld,
            error=despld_error
        ))

        # DESSL
        try:
            dessl = cm_dessl(request.paragraphs)
            dessl_error = None
        except Exception as e:
            logger.error("Error calculating DESSL: %s", e)
            dessl = None
            dessl_error = str(e)
        indices.append(Index(
            index=6,
            type_name="Descriptive",
            label_v3="DESSL",
            label_v2="READASL",
            description="Sentence length, number of words, mean",
            value=dessl,
            error=dessl_error
        ))

        # DESSLd
        try:
            dessld = cm_dessld(request.paragraphs)
            dessld_error = None
        except Exception as e:
            logger.error("Error calculating DESSLd: %s", e)
            dessld = None
            dessld_error = str(e)
        indices.append(Index(
            index=7,
            type_name="Descriptive",
            label_v3="DESSLd",
            label_v2="n/a",
            description="Sentence length, number of words, standard deviation",
            value=dessld,
            error=dessld_error
        ))

        # DESWLsy
        try:
            deswlsy = cm_deswlsy(tokens, request.language)
            deswlsy_error = None
        except Exception as e:
            logger.error("Error calculating DESWLsy: %s", e)
            deswlsy = None
            deswlsy_error = str(e)
        indices.append(Index(
            index=8,
            type_name="Descriptive",
            label_v3="DESWLsy",
            label_v2="READASW",
            description="Word length, number of syllables, mean",
            value=deswlsy,
            error=deswlsy_error
        ))

        # DESWLsyd
        try:
            deswlsyd = cm_deswlsyd(tokens, request.language)
            deswlsyd_error = None
        except Exception as e:
            logger.error("Error calculating DESWLsyd: %s", e)
            deswlsyd = None
            deswlsyd_error = str(e)
        indices.append(Index(
            index=9,
            type_name="Descriptive",
            label_v3="DESWLsyd",
            label_v2="n/a",
            description="Word length, number of syllables, standard deviation",
            value=deswlsyd,
            error=deswlsyd_error
        ))

        # DESWLlt
        try:
            deswllt = cm_deswllt(request.paragraphs)
            deswllt_error = None
        except Exception as e:
            logger.error("Error calculating DESWLlt: %s", e)
            deswllt = None
            deswllt_error = str(e)
        indices.append(Index(
            index=10,
            type_name="Descriptive",
            label_v3="DESWLlt",
            label_v2="n/a",
            description="Word length, number of letters, mean",
            value=deswllt,
            error=deswllt_error
        ))

        # DESWLltd
        try:
            deswlltd = cm_deswlltd(request.paragraphs)
            deswlltd_error = None
        except Exception as e:
            logger.error("Error calculating DESWLltd: %s", e)
            deswlltd = None
            deswlltd_error = str(e)
        indices.append(Index(
            index=11,
            type_name="Descriptive",
            label_v3="DESWLltd",
            label_v2="n/a",
            description="Word length, number of letters, standard deviation",
            value=deswlltd,
            error=deswlltd_error
        ))

        ### Text Easability Principal Component Scores

        # PCNARz
        try:
            #pcnarz = cm_pcnarz(sentences)
            pcnarz = None
            pcnarz_error = "Not implemented: Text Easability PC scores require LSA model + regression weights trained on TASA corpus (not available in this container)"
        except Exception as e:
            logger.error("Error calculating PCNARz: %s", e)
            pcnarz = None
            pcnarz_error = str(e)
        indices.append(Index(
            index=12,
            type_name="Text Easability Principal Component Scores",
            label_v3="PCNARz",
            label_v2="n/a",
            description="Text Easability PC Narrativity, z score",
            value=pcnarz,
            error=pcnarz_error
        ))

        # PCNARp
        try:
            #pcnarp = cm_pcnarp(sentences)
            pcnarp = None
            pcnarp_error = "Not implemented: Text Easability PC scores require LSA model + regression weights trained on TASA corpus (not available in this container)"
        except Exception as e:
            logger.error("Error calculating PCNARp: %s", e)
            pcnarp = None
            pcnarp_error = str(e)
        indices.append(Index(
            index=13,
            type_name="Text Easability Principal Component Scores",
            label_v3="PCNARp",
            label_v2="n/a",
            description="Text Easability PC Narrativity, percentile",
            value=pcnarp,
            error=pcnarp_error
        ))

        # PCSYNz
        try:
            #pcsynz = cm_pcsynz(sentences)
            pcsynz = None
            pcsynz_error = "Not implemented: Text Easability PC scores require LSA model + regression weights trained on TASA corpus (not available in this container)"
        except Exception as e:
            logger.error("Error calculating PCSYNz: %s", e)
            pcsynz = None
            pcsynz_error = str(e)
        indices.append(Index(
            index=14,
            type_name="Text Easability Principal Component Scores",
            label_v3="PCSYNz",
            label_v2="n/a",
            description="Text Easability PC Syntactic simplicity, z score",
            value=pcsynz,
            error=pcsynz_error
        ))

        # PCSYNp
        try:
            #pcsynp = cm_pcsynp(sentences)
            pcsynp = None
            pcsynp_error = "Not implemented: Text Easability PC scores require LSA model + regression weights trained on TASA corpus (not available in this container)"
        except Exception as e:
            logger.error("Error calculating PCSYNp: %s", e)
            pcsynp = None
            pcsynp_error = str(e)
        indices.append(Index(
            index=15,
            type_name="Text Easability Principal Component Scores",
            label_v3="PCSYNp",
            label_v2="n/a",
            description="Text Easability PC Syntactic simplicity, percentile",
            value=pcsynp,
            error=pcsynp_error
        ))

        # PCCNCz
        try:
            #pccncz = cm_pccncz(sentences)
            pccncz = None
            pccncz_error = "Not implemented: Text Easability PC scores require LSA model + regression weights trained on TASA corpus (not available in this container)"
        except Exception as e:
            logger.error("Error calculating PCCNCz: %s", e)
            pccncz = None
            pccncz_error = str(e)
        indices.append(Index(
            index=16,
            type_name="Text Easability Principal Component Scores",
            label_v3="PCCNCz",
            label_v2="n/a",
            description="Text Easability PC Word concreteness, z score",
            value=pccncz,
            error=pccncz_error
        ))

        # PCCNCp
        try:
            #pccncp = cm_pccncp(sentences)
            pccncp = None
            pccncp_error = "Not implemented: Text Easability PC scores require LSA model + regression weights trained on TASA corpus (not available in this container)"
        except Exception as e:
            logger.error("Error calculating PCCNCp: %s", e)
            pccncp = None
            pccncp_error = str(e)
        indices.append(Index(
            index=17,
            type_name="Text Easability Principal Component Scores",
            label_v3="PCCNCp",
            label_v2="n/a",
            description="Text Easability PC Word concreteness, percentile",
            value=pccncp,
            error=pccncp_error
        ))

        # PCREFz
        try:
            #pcrefz = cm_pcrefz(sentences)
            pcrefz = None
            pcrefz_error = "Not implemented: Text Easability PC scores require LSA model + regression weights trained on TASA corpus (not available in this container)"
        except Exception as e:
            logger.error("Error calculating PCREFz: %s", e)
            pcrefz = None
            pcrefz_error = str(e)
        indices.append(Index(
            index=18,
            type_name="Text Easability Principal Component Scores",
            label_v3="PCREFz",
            label_v2="n/a",
            description="Text Easability PC Referential cohesion, z score",
            value=pcrefz,
            error=pcrefz_error
        ))

        # PCREFp
        try:
            #pcrefp = cm_pcrefp(sentences)
            pcrefp = None
            pcrefp_error = "Not implemented: Text Easability PC scores require LSA model + regression weights trained on TASA corpus (not available in this container)"
        except Exception as e:
            logger.error("Error calculating PCREFp: %s", e)
            pcrefp = None
            pcrefp_error = str(e)
        indices.append(Index(
            index=19,
            type_name="Text Easability Principal Component Scores",
            label_v3="PCREFp",
            label_v2="n/a",
            description="Text Easability PC Referential cohesion, percentile",
            value=pcrefp,
            error=pcrefp_error
        ))

        # PCDCz
        try:
            #pcdcz = cm_pcdcz(sentences)
            pcdcz = None
            pcdcz_error = "Not implemented: Text Easability PC scores require LSA model + regression weights trained on TASA corpus (not available in this container)"
        except Exception as e:
            logger.error("Error calculating PCDCz: %s", e)
            pcdcz = None
            pcdcz_error = str(e)
        indices.append(Index(
            index=20,
            type_name="Text Easability Principal Component Scores",
            label_v3="PCDCz",
            label_v2="n/a",
            description="Text Easability PC Deep cohesion, z score",
            value=pcdcz,
            error=pcdcz_error
        ))

        # PCDCp
        try:
            #pcdcp = cm_pcdcp(sentences)
            pcdcp = None
            pcdcp_error = "Not implemented: Text Easability PC scores require LSA model + regression weights trained on TASA corpus (not available in this container)"
        except Exception as e:
            logger.error("Error calculating PCDCp: %s", e)
            pcdcp = None
            pcdcp_error = str(e)
        indices.append(Index(
            index=21,
            type_name="Text Easability Principal Component Scores",
            label_v3="PCDCp",
            label_v2="n/a",
            description="Text Easability PC Deep cohesion, percentile",
            value=pcdcp,
            error=pcdcp_error
        ))

        # PCVERBz
        try:
            #pcverbz = cm_pcverbz(sentences)
            pcverbz = None
            pcverbz_error = "Not implemented: Text Easability PC scores require LSA model + regression weights trained on TASA corpus (not available in this container)"
        except Exception as e:
            logger.error("Error calculating PCVERBz: %s", e)
            pcverbz = None
            pcverbz_error = str(e)
        indices.append(Index(
            index=22,
            type_name="Text Easability Principal Component Scores",
            label_v3="PCVERBz",
            label_v2="n/a",
            description="Text Easability PC Verb cohesion, z score",
            value=pcverbz,
            error=pcverbz_error
        ))

        # PCVERBp
        try:
            #pcverbp = cm_pcverbp(sentences)
            pcverbp = None
            pcverbp_error = "Not implemented: Text Easability PC scores require LSA model + regression weights trained on TASA corpus (not available in this container)"
        except Exception as e:
            logger.error("Error calculating PCVERBp: %s", e)
            pcverbp = None
            pcverbp_error = str(e)
        indices.append(Index(
            index=23,
            type_name="Text Easability Principal Component Scores",
            label_v3="PCVERBp",
            label_v2="n/a",
            description="Text Easability PC Verb cohesion, percentile",
            value=pcverbp,
            error=pcverbp_error
        ))

        # PCCONNz
        try:
            #pcconnz = cm_pcconnz(sentences)
            pcconnz = None
            pcconnz_error = "Not implemented: Text Easability PC scores require LSA model + regression weights trained on TASA corpus (not available in this container)"
        except Exception as e:
            logger.error("Error calculating PCCONNz: %s", e)
            pcconnz = None
            pcconnz_error = str(e)
        indices.append(Index(
            index=24,
            type_name="Text Easability Principal Component Scores",
            label_v3="PCCONNz",
            label_v2="n/a",
            description="Text Easability PC Connectivity, z score",
            value=pcconnz,
            error=pcconnz_error
        ))

        # PCCONNp
        try:
            #pcconnp = cm_pcconnp(sentences)
            pcconnp = None
            pcconnp_error = "Not implemented: Text Easability PC scores require LSA model + regression weights trained on TASA corpus (not available in this container)"
        except Exception as e:
            logger.error("Error calculating PCCONNp: %s", e)
            pcconnp = None
            pcconnp_error = str(e)
        indices.append(Index(
            index=25,
            type_name="Text Easability Principal Component Scores",
            label_v3="PCCONNp",
            label_v2="n/a",
            description="Text Easability PC Connectivity, percentile",
            value=pcconnp,
            error=pcconnp_error
        ))

        # PCTEMPz
        try:
            #pctempz = cm_pctempz(sentences)
            pctempz = None
            pctempz_error = "Not implemented: Text Easability PC scores require LSA model + regression weights trained on TASA corpus (not available in this container)"
        except Exception as e:
            logger.error("Error calculating PCTEMPz: %s", e)
            pctempz = None
            pctempz_error = str(e)
        indices.append(Index(
            index=26,
            type_name="Text Easability Principal Component Scores",
            label_v3="PCTEMPz",
            label_v2="n/a",
            description="Text Easability PC Temporality, z score",
            value=pctempz,
            error=pctempz_error
        ))

        # PCTEMPp
        try:
            #pctempp = cm_pctempp(sentences)
            pctempp = None
            pctempp_error = "Not implemented: Text Easability PC scores require LSA model + regression weights trained on TASA corpus (not available in this container)"
        except Exception as e:
            logger.error("Error calculating PCTEMPp: %s", e)
            pctempp = None
            pctempp_error = str(e)
        indices.append(Index(
            index=27,
            type_name="Text Easability Principal Component Scores",
            label_v3="PCTEMPp",
            label_v2="n/a",
            description="Text Easability PC Temporality, percentile",
            value=pctempp,
            error=pctempp_error
        ))

        ### Referential Cohesion

        # CRFNO1
        try:
            crfno1 = cm_crfno1(sentences)
            crfno1_error = None
        except Exception as e:
            logger.error("Error calculating CRFNO1: %s", e)
            crfno1 = None
            crfno1_error = str(e)
        indices.append(Index(
            index=28,
            type_name="Referential Cohesion",
            label_v3="CRFNO1",
            label_v2="CRFBN1um",
            description="Noun overlap, adjacent sentences, binary, mean",
            value=crfno1,
            error=crfno1_error
        ))

        # CRFAO1
        try:
            crfao1 = cm_crfao1(sentences)
            crfao1_error = None
        except Exception as e:
            logger.error("Error calculating CRFAO1: %s", e)
            crfao1 = None
            crfao1_error = str(e)
        indices.append(Index(
            index=29,
            type_name="Referential Cohesion",
            label_v3="CRFAO1",
            label_v2="CRFBA1um",
            description="Argument overlap, adjacent sentences, binary, mean",
            value=crfao1,
            error=crfao1_error
        ))

        # CRFSO1
        try:
            crfso1 = cm_crfso1(sentences)
            crfso1_error = None
        except Exception as e:
            logger.error("Error calculating CRFSO1: %s", e)
            crfso1 = None
            crfso1_error = str(e)
        indices.append(Index(
            index=30,
            type_name="Referential Cohesion",
            label_v3="CRFSO1",
            label_v2="CRFBS1um",
            description="Stem overlap, adjacent sentences, binary, mean",
            value=crfso1,
            error=crfso1_error
        ))

        # CRFNOa
        try:
            crfnoa = cm_crfnoa(sentences)
            crfnoa_error = None
        except Exception as e:
            logger.error("Error calculating CRFNOa: %s", e)
            crfnoa = None
            crfnoa_error = str(e)
        indices.append(Index(
            index=31,
            type_name="Referential Cohesion",
            label_v3="CRFNOa",
            label_v2="CRFBNaum",
            description="Noun overlap, all sentences, binary, mean",
            value=crfnoa,
            error=crfnoa_error
        ))

        # CRFAOa
        try:
            crfaoa = cm_crfaoa(sentences)
            crfaoa_error = None
        except Exception as e:
            logger.error("Error calculating CRFAOa: %s", e)
            crfaoa = None
            crfaoa_error = str(e)
        indices.append(Index(
            index=32,
            type_name="Referential Cohesion",
            label_v3="CRFAOa",
            label_v2="CRFBAaum",
            description="Argument overlap, all sentences, binary, mean",
            value=crfaoa,
            error=crfaoa_error
        ))

        # CRFSOa
        try:
            crfsoa = cm_crfsoa(sentences)
            crfsoa_error = None
        except Exception as e:
            logger.error("Error calculating CRFSOa: %s", e)
            crfsoa = None
            crfsoa_error = str(e)
        indices.append(Index(
            index=33,
            type_name="Referential Cohesion",
            label_v3="CRFSOa",
            label_v2="CRFBSaum",
            description="Stem overlap, all sentences, binary, mean",
            value=crfsoa,
            error=crfsoa_error
        ))

        # CRFCWO1
        try:
            crfcwo1 = cm_crfcwo1(sentences)
            crfcwo1_error = None
        except Exception as e:
            logger.error("Error calculating CRFCWO1: %s", e)
            crfcwo1 = None
            crfcwo1_error = str(e)
        indices.append(Index(
            index=34,
            type_name="Referential Cohesion",
            label_v3="CRFCWO1",
            label_v2="CRFPC1um",
            description="Content word overlap, adjacent sentences, proportional, mean",
            value=crfcwo1,
            error=crfcwo1_error
        ))

        # CRFCWO1d
        try:
            crfcwo1d = cm_crfcwo1d(sentences)
            crfcwo1d_error = None
        except Exception as e:
            logger.error("Error calculating CRFCWO1d: %s", e)
            crfcwo1d = None
            crfcwo1d_error = str(e)
        indices.append(Index(
            index=35,
            type_name="Referential Cohesion",
            label_v3="CRFCWO1d",
            label_v2="n/a",
            description="Content word overlap, adjacent sentences, proportional, standard deviation",
            value=crfcwo1d,
            error=crfcwo1d_error
        ))

        # CRFCWOa
        try:
            crfcwoa = cm_crfcwoa(sentences)
            crfcwoa_error = None
        except Exception as e:
            logger.error("Error calculating CRFCWOa: %s", e)
            crfcwoa = None
            crfcwoa_error = str(e)
        indices.append(Index(
            index=36,
            type_name="Referential Cohesion",
            label_v3="CRFCWOa",
            label_v2="CRFPCaum",
            description="Content word overlap, all sentences, proportional, mean",
            value=crfcwoa,
            error=crfcwoa_error
        ))

        # CRFCWOad
        try:
            crfcwoad = cm_crfcwoad(sentences)
            crfcwoad_error = None
        except Exception as e:
            logger.error("Error calculating CRFCWOad: %s", e)
            crfcwoad = None
            crfcwoad_error = str(e)
        indices.append(Index(
            index=37,
            type_name="Referential Cohesion",
            label_v3="CRFCWOad",
            label_v2="n/a",
            description="Content word overlap, all sentences, proportional, standard deviation",
            value=crfcwoad,
            error=crfcwoad_error
        ))

        ### Lexical Diversity

        try:
            lsa_n_components = 100
            token_vectors, token_words = _get_paragraph_token_vectors(request.paragraphs)
            lsa_indices = _lsa_cohesion_indices(token_vectors, token_words, tokens_vector_length, lsa_n_components)
            lsa_error = None
        except Exception as e:
            logger.error("Error calculating LSA: %s", e)
            lsa_indices = None
            lsa_error = str(e)

        # LSASS1
        try:
            lsass1 = cm_lsass1(lsa_indices)
            lsass1_error = None
        except Exception as e:
            logger.error("Error calculating LSASS1: %s", e)
            lsass1 = None
            # M7 fix: guard against None + str when the outer LSA step succeeded.
            lsass1_error = ((lsa_error + "\n") if lsa_error else "") + str(e)
        indices.append(Index(
            index=38,
            type_name="LSA",
            label_ttlab="LSASS1_spacy",
            label_v3="LSASS1",
            label_v2="LSAassa",
            description="LSA overlap, adjacent sentences, mean",
            value=lsass1,
            error=lsass1_error
        ))

        # LSASS1d
        try:
            lsass1d = cm_lsass1d(lsa_indices)
            lsass1d_error = None
        except Exception as e:
            logger.error("Error calculating LSASS1d: %s", e)
            lsass1d = None
            lsass1d_error = ((lsa_error + "\n") if lsa_error else "") + str(e)  # M7
        indices.append(Index(
            index=39,
            type_name="LSA",
            label_ttlab="LSASS1d_spacy",
            label_v3="LSASS1d",
            label_v2="LSAassd",
            description="LSA overlap, adjacent sentences, standard deviation",
            value=lsass1d,
            error=lsass1d_error
        ))

        # LSASSp
        try:
            lsassp = cm_lsassp(lsa_indices)
            lsassp_error = None
        except Exception as e:
            logger.error("Error calculating LSASSp: %s", e)
            lsassp = None
            lsassp_error = ((lsa_error + "\n") if lsa_error else "") + str(e)  # M7
        indices.append(Index(
            index=40,
            type_name="LSA",
            label_ttlab="LSASSp_spacy",
            label_v3="LSASSp",
            label_v2="LSApssa",
            description="LSA overlap, all sentences in paragraph, mean",
            value=lsassp,
            error=lsassp_error
        ))

        # LSASSpd
        try:
            lsasspd = cm_lsasspd(lsa_indices)
            lsasspd_error = None
        except Exception as e:
            logger.error("Error calculating LSASSpd: %s", e)
            lsasspd = None
            lsasspd_error = ((lsa_error + "\n") if lsa_error else "") + str(e)  # M7
        indices.append(Index(
            index=41,
            type_name="LSA",
            label_ttlab="LSASSpd_spacy",
            label_v3="LSASSpd",
            label_v2="LSApssd",
            description="LSA overlap, all sentences in paragraph, standard deviation",
            value=lsasspd,
            error=lsasspd_error
        ))

        # LSAPP1
        try:
            lsapp1 = cm_lsapp1(lsa_indices)
            lsapp1_error = None
        except Exception as e:
            logger.error("Error calculating LSAPP1: %s", e)
            lsapp1 = None
            lsapp1_error = ((lsa_error + "\n") if lsa_error else "") + str(e)  # M7
        indices.append(Index(
            index=42,
            type_name="LSA",
            label_ttlab="LSAPP1_spacy",
            label_v3="LSAPP1",
            label_v2="LSAppa",
            description="LSA overlap, adjacent paragraphs, mean",
            value=lsapp1,
            error=lsapp1_error
        ))

        # LSAPP1d
        try:
            lsapp1d = cm_lsapp1d(lsa_indices)
            lsapp1d_error = None
        except Exception as e:
            logger.error("Error calculating LSAPP1d: %s", e)
            lsapp1d = None
            lsapp1d_error = ((lsa_error + "\n") if lsa_error else "") + str(e)  # M7
        indices.append(Index(
            index=43,
            type_name="LSA",
            label_ttlab="LSAPP1d_spacy",
            label_v3="LSAPP1d",
            label_v2="LSAppd",
            description="LSA overlap, adjacent paragraphs, standard deviation",
            value=lsapp1d,
            error=lsapp1d_error
        ))

        # LSAGN
        try:
            lsagn = cm_lsagn(lsa_indices)
            lsagn_error = None
        except Exception as e:
            logger.error("Error calculating LSAGN: %s", e)
            lsagn = None
            lsagn_error = ((lsa_error + "\n") if lsa_error else "") + str(e)  # M7
        indices.append(Index(
            index=44,
            type_name="LSA",
            label_ttlab="LSAGN_spacy",
            label_v3="LSAGN",
            label_v2="LSAGN",
            description="LSA given/new, sentences, mean",
            value=lsagn,
            error=lsagn_error
        ))

        # LSAGNd
        try:
            lsagnd = cm_lsagnd(lsa_indices)
            lsagnd_error = None
        except Exception as e:
            logger.error("Error calculating LSAGNd: %s", e)
            lsagnd = None
            lsagnd_error = ((lsa_error + "\n") if lsa_error else "") + str(e)  # M7
        indices.append(Index(
            index=45,
            type_name="LSA",
            label_ttlab="LSAGNd_spacy",
            label_v3="LSAGNd",
            label_v2="n/a",
            description="LSA given/new, sentences, standard deviation",
            value=lsagnd,
            error=lsagnd_error
        ))

        ### Lexical Diversity

        # LDTTRc
        try:
            ldttrc = cm_ldttrc(tokens)
            ldttrc_error = None
        except Exception as e:
            logger.error("Error calculating LDTTRc: %s", e)
            ldttrc = None
            ldttrc_error = str(e)
        indices.append(Index(
            index=46,
            type_name="Lexical Diversity",
            label_v3="LDTTRc",
            label_v2="TYPTOKc",
            description="Lexical diversity, type-token ratio, content word lemmas",
            value=ldttrc,
            error=ldttrc_error
        ))

        # LDTTRa
        try:
            ldttra = cm_ldttra(tokens)
            ldttra_error = None
        except Exception as e:
            logger.error("Error calculating LDTTRa: %s", e)
            ldttra = None
            ldttra_error = str(e)
        indices.append(Index(
            index=47,
            type_name="Lexical Diversity",
            label_v3="LDTTRa",
            label_v2="n/a",
            description="Lexical diversity, type-token ratio, all words",
            value=ldttra,
            error=ldttra_error
        ))

        # LDMTLDa
        try:
            ldmtlda = cm_ldmtlda(tokens)
            ldmtlda_error = None
        except Exception as e:
            logger.error("Error calculating LDMTLDa: %s", e)
            ldmtlda = None
            ldmtlda_error = str(e)
        indices.append(Index(
            index=48,
            type_name="Lexical Diversity",
            label_v3="LDMTLDa",
            label_v2="LEXDIVTD",
            description="Lexical diversity, MTLD, all words",
            value=ldmtlda,
            error=ldmtlda_error
        ))

        # LDVOCDa
        try:
            ldvocda = cm_ldvocda(tokens)
            ldvocda_error = None
        except Exception as e:
            logger.error("Error calculating LDVOCDa: %s", e)
            ldvocda = None
            ldvocda_error = str(e)
        indices.append(Index(
            index=49,
            type_name="Lexical Diversity",
            label_v3="LDVOCDa",
            label_v2="LEXDIVVD",
            description="Lexical diversity, VOCD, all words",
            value=ldvocda,
            error=ldvocda_error
        ))

        ### Connectives

        # L1: compute connective counts once and reuse across all 9 CNC* indices.
        try:
            _cnc_counts = _count_connectives(request.text, request.language, tokens_count)
        except Exception as e:
            logger.error("Error precomputing connective counts: %s", e)
            _cnc_counts = None

        # CNCAll
        try:
            cncall = cm_cncall(request.text, request.language, tokens_count, connectives=_cnc_counts)
            cncall_error = None
        except Exception as e:
            logger.error("Error calculating CNCAll: %s", e)
            cncall = None
            cncall_error = str(e)
        indices.append(Index(
            index=50,
            type_name="Connectives",
            label_v3="CNCAll",
            label_v2="CONi",
            description="All connectives incidence",
            value=cncall,
            error=cncall_error
        ))

        # CNCCaus
        try:
            cnccaus = cm_cnccaus(request.text, request.language, tokens_count, connectives=_cnc_counts)
            cnccaus_error = None
        except Exception as e:
            logger.error("Error calculating CNCCaus: %s", e)
            cnccaus = None
            cnccaus_error = str(e)
        indices.append(Index(
            index=51,
            type_name="Connectives",
            label_v3="CNCCaus",
            label_v2="CONCAUSi",
            description="Causal connectives incidence",
            value=cnccaus,
            error=cnccaus_error
        ))

        # CNCLogic
        try:
            cnclogic = cm_cnclogic(request.text, request.language, tokens_count, connectives=_cnc_counts)
            cnclogic_error = None
        except Exception as e:
            logger.error("Error calculating CNCLogic: %s", e)
            cnclogic = None
            cnclogic_error = str(e)
        indices.append(Index(
            index=52,
            type_name="Connectives",
            label_v3="CNCLogic",
            label_v2="CONLOGi",
            description="Logical connectives incidence",
            value=cnclogic,
            error=cnclogic_error
        ))

        # CNCADC
        try:
            cncadc = cm_cncadc(request.text, request.language, tokens_count, connectives=_cnc_counts)
            cncadc_error = None
        except Exception as e:
            logger.error("Error calculating CNCADC: %s", e)
            cncadc = None
            cncadc_error = str(e)
        indices.append(Index(
            index=53,
            type_name="Connectives",
            label_v3="CNCADC",
            label_v2="CONADVCONi",
            description="Adversative and contrastive connectives incidence",
            value=cncadc,
            error=cncadc_error
        ))

        # CNCTemp
        try:
            cnctemp = cm_cnctemp(request.text, request.language, tokens_count, connectives=_cnc_counts)
            cnctemp_error = None
        except Exception as e:
            logger.error("Error calculating CNCTemp: %s", e)
            cnctemp = None
            cnctemp_error = str(e)
        indices.append(Index(
            index=54,
            type_name="Connectives",
            label_v3="CNCTemp",
            label_v2="CONTEMPi",
            description="Temporal connectives incidence",
            value=cnctemp,
            error=cnctemp_error
        ))

        # CNCTempx
        try:
            cnctempx = cm_cnctempx(request.text, request.language, tokens_count, connectives=_cnc_counts)
            cnctempx_error = None
        except Exception as e:
            logger.error("Error calculating CNCTempx: %s", e)
            cnctempx = None
            cnctempx_error = str(e)
        indices.append(Index(
            index=55,
            type_name="Connectives",
            label_v3="CNCTempx",
            label_v2="CONTEMPEXi",
            description="Expanded temporal connectives incidence",
            value=cnctempx,
            error=cnctempx_error
        ))

        # CNCAdd
        try:
            cncadd = cm_cncadd(request.text, request.language, tokens_count, connectives=_cnc_counts)
            cncadd_error = None
        except Exception as e:
            logger.error("Error calculating CNCAdd: %s", e)
            cncadd = None
            cncadd_error = str(e)
        indices.append(Index(
            index=56,
            type_name="Connectives",
            label_v3="CNCAdd",
            label_v2="CONADDi",
            description="Additive connectives incidence",
            value=cncadd,
            error=cncadd_error
        ))

        # CNCPos
        try:
            cncpos = cm_cncpos(request.text, request.language, tokens_count, connectives=_cnc_counts)
            cncpos_error = None
        except Exception as e:
            logger.error("Error calculating CNCPos: %s", e)
            cncpos = None
            cncpos_error = str(e)
        indices.append(Index(
            index=57,
            type_name="Connectives",
            label_v3="CNCPos",
            label_v2="n/a",
            description="Positive connectives incidence",
            value=cncpos,
            error=cncpos_error
        ))

        # CNCNeg
        try:
            cncneg = cm_cncneg(request.text, request.language, tokens_count, connectives=_cnc_counts)
            cncneg_error = None
        except Exception as e:
            logger.error("Error calculating CNCNeg: %s", e)
            cncneg = None
            cncneg_error = str(e)
        indices.append(Index(
            index=58,
            type_name="Connectives",
            label_v3="CNCNeg",
            label_v2="n/a",
            description="Negative connectives incidence",
            value=cncneg,
            error=cncneg_error
        ))

        ### Situation Model

        # SMCAUSv
        try:
            smcausv = cm_smcausv(sentences, request.language)
            smcausv_error = None
        except Exception as e:
            logger.error("Error calculating SMCAUSv: %s", e)
            smcausv = None
            smcausv_error = str(e)
        indices.append(Index(
            index=59,
            type_name="Situation Model",
            label_v3="SMCAUSv",
            label_v2="CAUSV",
            description="Causal verb incidence",
            value=smcausv,
            error=smcausv_error
        ))

        # SMCAUSvp
        try:
            smcausvp = cm_smcausvp(sentences, request.language)
            smcausvp_error = None
        except Exception as e:
            logger.error("Error calculating SMCAUSvp: %s", e)
            smcausvp = None
            smcausvp_error = str(e)
        indices.append(Index(
            index=60,
            type_name="Situation Model",
            label_v3="SMCAUSvp",
            label_v2="CAUSVP",
            description="Causal verbs and causal particles incidence",
            value=smcausvp,
            error=smcausvp_error
        ))

        # SMINTEp
        try:
            smintep = cm_smintep(sentences, request.language)
            smintep_error = None
        except Exception as e:
            logger.error("Error calculating SMINTEp: %s", e)
            smintep = None
            smintep_error = str(e)
        indices.append(Index(
            index=61,
            type_name="Situation Model",
            label_v3="SMINTEp",
            label_v2="INTEi",
            description="Intentional verbs incidence",
            value=smintep,
            error=smintep_error
        ))

        # SMCAUSr
        try:
            smcausr = cm_smcausr(sentences, request.language)
            smcausr_error = None
        except Exception as e:
            logger.error("Error calculating SMCAUSr: %s", e)
            smcausr = None
            smcausr_error = str(e)
        indices.append(Index(
            index=62,
            type_name="Situation Model",
            label_v3="SMCAUSr",
            label_v2="CAUSC",
            description="Ratio of causal particles to causal verbs",
            value=smcausr,
            error=smcausr_error
        ))

        # SMINTEr
        try:
            sminter = cm_sminter(sentences, request.language)
            sminter_error = None
        except Exception as e:
            logger.error("Error calculating SMINTEr: %s", e)
            sminter = None
            sminter_error = str(e)
        indices.append(Index(
            index=63,
            type_name="Situation Model",
            label_v3="SMINTEr",
            label_v2="INTEC",
            description="Ratio of intentional particles to intentional verbs",
            value=sminter,
            error=sminter_error
        ))

        # SMCAUSlsa
        try:
            smcauslsa = cm_smcauslsa(sentences)
            smcauslsa_error = None
        except Exception as e:
            logger.error("Error calculating SMCAUSlsa: %s", e)
            smcauslsa = None
            smcauslsa_error = str(e)
        indices.append(Index(
            index=64,
            type_name="Situation Model",
            label_v3="SMCAUSlsa",
            label_v2="CAUSLSA",
            description="LSA verb overlap",
            value=smcauslsa,
            error=smcauslsa_error
        ))

        # SMCAUSwn
        try:
            smcauswn = cm_smcauswn(sentences, request.language)
            smcauswn_error = None
        except Exception as e:
            logger.error("Error calculating SMCAUSwn: %s", e)
            smcauswn = None
            smcauswn_error = str(e)
        indices.append(Index(
            index=65,
            type_name="Situation Model",
            label_v3="SMCAUSwn",
            label_v2="CAUSWN",
            description="WordNet verb overlap",
            value=smcauswn,
            error=smcauswn_error
        ))

        # SMTEMP
        try:
            smtemp = cm_smtemp(sentences, request.language)
            smtemp_error = None
        except Exception as e:
            logger.error("Error calculating SMTEMP: %s", e)
            smtemp = None
            smtemp_error = str(e)
        indices.append(Index(
            index=66,
            type_name="Situation Model",
            label_v3="SMTEMP",
            label_v2="TEMPta",
            description="Temporal cohesion, tense and aspect repetition, mean",
            value=smtemp,
            error=smtemp_error
        ))

        ### Syntactic Complexity

        # SYNLE
        try:
            synle = cm_synle(sentences)
            synle_error = None
        except Exception as e:
            logger.error("Error calculating SYNLE: %s", e)
            synle = None
            synle_error = str(e)
        indices.append(Index(
            index=67,
            type_name="Syntactic Complexity",
            label_v3="SYNLE",
            label_v2="SYNLE",
            description="Left embeddedness, words before main verb, mean",
            value=synle,
            error=synle_error
        ))

        # SYNNP
        try:
            synnp = cm_synnp(sentences, request.noun_chunks, request.language)
            synnp_error = None
        except Exception as e:
            logger.error("Error calculating SYNNP: %s", e)
            synnp = None
            synnp_error = str(e)
        indices.append(Index(
            index=68,
            type_name="Syntactic Complexity",
            label_v3="SYNNP",
            label_v2="SYNNP",
            description="Number of modifiers per noun phrase, mean",
            value=synnp,
            error=synnp_error
        ))

        # SYNMEDpos
        try:
            synmedpos = cm_synmedpos(sentences)
            synmedpos_error = None
        except Exception as e:
            logger.error("Error calculating SYNMEDpos: %s", e)
            synmedpos = None
            synmedpos_error = str(e)
        indices.append(Index(
            index=69,
            type_name="Syntactic Complexity",
            label_v3="SYNMEDpos",
            label_v2="MEDwtm",
            description="Minimal Edit Distance, part of speech",
            value=synmedpos,
            error=synmedpos_error
        ))

        # SYNMEDwrd
        try:
            synmedwrd = cm_synmedwrd(sentences)
            synmedwrd_error = None
        except Exception as e:
            logger.error("Error calculating SYNMEDwrd: %s", e)
            synmedwrd = None
            synmedwrd_error = str(e)
        indices.append(Index(
            index=70,
            type_name="Syntactic Complexity",
            label_v3="SYNMEDwrd",
            label_v2="MEDawm",
            description="Minimal Edit Distance, all words",
            value=synmedwrd,
            error=synmedwrd_error
        ))

        # SYNMEDlem
        try:
            synmedlem = cm_synmedlem(sentences)
            synmedlem_error = None
        except Exception as e:
            logger.error("Error calculating SYNMEDlem: %s", e)
            synmedlem = None
            synmedlem_error = str(e)
        indices.append(Index(
            index=71,
            type_name="Syntactic Complexity",
            label_v3="SYNMEDlem",
            label_v2="MEDalm",
            description="Minimal Edit Distance, lemmas",
            value=synmedlem,
            error=synmedlem_error
        ))

        # SYNSTRUTa
        try:
            synstruta = cm_synstruta(sentences)
            synstruta_error = None
        except Exception as e:
            logger.error("Error calculating SYNSTRUTa: %s", e)
            synstruta = None
            synstruta_error = str(e)
        indices.append(Index(
            index=72,
            type_name="Syntactic Complexity",
            label_v3="SYNSTRUTa",
            label_v2="STRUTa",
            description="Sentence syntax similarity, adjacent sentences, mean",
            value=synstruta,
            error=synstruta_error
        ))

        # SYNSTRUTt
        try:
            synstrutt = cm_synstrutt(request.paragraphs)
            synstrutt_error = None
        except Exception as e:
            logger.error("Error calculating SYNSTRUTt: %s", e)
            synstrutt = None
            synstrutt_error = str(e)
        indices.append(Index(
            index=73,
            type_name="Syntactic Complexity",
            label_v3="SYNSTRUTt",
            label_v2="STRUTt",
            description="Sentence syntax similarity, all combinations, across paragraphs, mean",
            value=synstrutt,
            error=synstrutt_error
        ))

        ### Syntactic Pattern Density

        # L2: compute count_metrics once and reuse across all 8 DR* indices.
        try:
            _dr_metrics = _count_metrics(sentences, request.noun_chunks, request.language)
        except Exception as e:
            logger.error("Error precomputing DR metrics: %s", e)
            _dr_metrics = None

        # DRNP
        try:
            drnp = cm_drnp(sentences, request.noun_chunks, request.language, metrics=_dr_metrics)
            drnp_error = None
        except Exception as e:
            logger.error("Error calculating DRNP: %s", e)
            drnp = None
            drnp_error = str(e)
        indices.append(Index(
            index=74,
            type_name="Syntactic Pattern Density",
            label_v3="DRNP",
            label_v2="n/a",
            description="Noun phrase density, incidence",
            value=drnp,
            error=drnp_error
        ))

        # DRVP
        try:
            drvp = cm_drvp(sentences, request.noun_chunks, request.language, metrics=_dr_metrics)
            drvp_error = None
        except Exception as e:
            logger.error("Error calculating DRVP: %s", e)
            drvp = None
            drvp_error = str(e)
        indices.append(Index(
            index=75,
            type_name="Syntactic Pattern Density",
            label_v3="DRVP",
            label_v2="n/a",
            description="Verb phrase density, incidence",
            value=drvp,
            error=drvp_error
        ))

        # DRAP
        try:
            drap = cm_drap(sentences, request.noun_chunks, request.language, metrics=_dr_metrics)
            drap_error = None
        except Exception as e:
            logger.error("Error calculating DRAP: %s", e)
            drap = None
            drap_error = str(e)
        indices.append(Index(
            index=76,
            type_name="Syntactic Pattern Density",
            label_v3="DRAP",
            label_v2="n/a",
            description="Adverbial phrase density, incidence",
            value=drap,
            error=drap_error
        ))

        # DRPP
        try:
            drpp = cm_drpp(sentences, request.noun_chunks, request.language, metrics=_dr_metrics)
            drpp_error = None
        except Exception as e:
            logger.error("Error calculating DRPP: %s", e)
            drpp = None
            drpp_error = str(e)
        indices.append(Index(
            index=77,
            type_name="Syntactic Pattern Density",
            label_v3="DRPP",
            label_v2="n/a",
            description="Preposition phrase density, incidence",
            value=drpp,
            error=drpp_error
        ))

        # DRPVAL
        try:
            drpval = cm_drpval(sentences, request.noun_chunks, request.language, metrics=_dr_metrics)
            drpval_error = None
        except Exception as e:
            logger.error("Error calculating DRPVAL: %s", e)
            drpval = None
            drpval_error = str(e)
        indices.append(Index(
            index=78,
            type_name="Syntactic Pattern Density",
            label_v3="DRPVAL",
            label_v2="AGLSPSVi",
            description="Agentless passive voice density, incidence",
            value=drpval,
            error=drpval_error
        ))

        # DRNEG
        try:
            drneg = cm_drneg(sentences, request.noun_chunks, request.language, metrics=_dr_metrics)
            drneg_error = None
        except Exception as e:
            logger.error("Error calculating DRNEG: %s", e)
            drneg = None
            drneg_error = str(e)
        indices.append(Index(
            index=79,
            type_name="Syntactic Pattern Density",
            label_v3="DRNEG",
            label_v2="DENNEGi",
            description="Negation density, incidence",
            value=drneg,
            error=drneg_error
        ))

        # DRGERUND
        try:
            drgerund = cm_drgerund(sentences, request.noun_chunks, request.language, metrics=_dr_metrics)
            drgerund_error = None
        except Exception as e:
            logger.error("Error calculating DRGERUND: %s", e)
            drgerund = None
            drgerund_error = str(e)
        indices.append(Index(
            index=80,
            type_name="Syntactic Pattern Density",
            label_v3="DRGERUND",
            label_v2="GERUNDi",
            description="Gerund density, incidence",
            value=drgerund,
            error=drgerund_error
        ))

        # DRINF
        try:
            drinf = cm_drinf(sentences, request.noun_chunks, request.language, metrics=_dr_metrics)
            drinf_error = None
        except Exception as e:
            logger.error("Error calculating DRINF: %s", e)
            drinf = None
            drinf_error = str(e)
        indices.append(Index(
            index=81,
            type_name="Syntactic Pattern Density",
            label_v3="DRINF",
            label_v2="INFi",
            description="Infinitive density, incidence",
            value=drinf,
            error=drinf_error
        ))

        ### Word Information

        # L11: compute WRD* counter dict once and reuse across all 10 WRD* indices.
        try:
            _wrd_counts = _wrd_precompute(sentences)
        except Exception as e:
            logger.error("Error precomputing WRD counts: %s", e)
            _wrd_counts = None

        # WRDNOUN
        try:
            wrdnoun = cm_wrdnoun(sentences, counts=_wrd_counts)
            wrdnoun_error = None
        except Exception as e:
            logger.error("Error calculating WRDNOUN: %s", e)
            wrdnoun = None
            wrdnoun_error = str(e)
        indices.append(Index(
            index=82,
            type_name="Word Information",
            label_v3="WRDNOUN",
            label_v2="NOUNi",
            description="Noun incidence",
            value=wrdnoun,
            error=wrdnoun_error
        ))

        # WRDVERB
        try:
            wrdverb = cm_wrdverb(sentences, counts=_wrd_counts)
            wrdverb_error = None
        except Exception as e:
            logger.error("Error calculating WRDVERB: %s", e)
            wrdverb = None
            wrdverb_error = str(e)
        indices.append(Index(
            index=83,
            type_name="Word Information",
            label_v3="WRDVERB",
            label_v2="VERBi",
            description="Verb incidence",
            value=wrdverb,
            error=wrdverb_error
        ))

        # WRDADJ
        try:
            wrdadj = cm_wrdadj(sentences, counts=_wrd_counts)
            wrdadj_error = None
        except Exception as e:
            logger.error("Error calculating WRDADJ: %s", e)
            wrdadj = None
            wrdadj_error = str(e)
        indices.append(Index(
            index=84,
            type_name="Word Information",
            label_v3="WRDADJ",
            label_v2="ADJi",
            description="Adjective incidence",
            value=wrdadj,
            error=wrdadj_error
        ))

        # WRDADV
        try:
            wrdadv = cm_wrdadv(sentences, counts=_wrd_counts)
            wrdadv_error = None
        except Exception as e:
            logger.error("Error calculating WRDADV: %s", e)
            wrdadv = None
            wrdadv_error = str(e)
        indices.append(Index(
            index=85,
            type_name="Word Information",
            label_v3="WRDADV",
            label_v2="ADVi",
            description="Adverb incidence",
            value=wrdadv,
            error=wrdadv_error
        ))

        # WRDPRO
        try:
            wrdpro = cm_wrdpro(sentences, counts=_wrd_counts)
            wrdpro_error = None
        except Exception as e:
            logger.error("Error calculating WRDPRO: %s", e)
            wrdpro = None
            wrdpro_error = str(e)
        indices.append(Index(
            index=86,
            type_name="Word Information",
            label_v3="WRDPRO",
            label_v2="DENPRPi",
            description="Pronoun incidence",
            value=wrdpro,
            error=wrdpro_error
        ))

        # WRDPRP1s
        try:
            wrdprp1s = cm_wrdprp1s(sentences, counts=_wrd_counts)
            wrdprp1s_error = None
        except Exception as e:
            logger.error("Error calculating WRDPRP1s: %s", e)
            wrdprp1s = None
            wrdprp1s_error = str(e)
        indices.append(Index(
            index=87,
            type_name="Word Information",
            label_v3="WRDPRP1s",
            label_v2="n/a",
            description="First-person singular pronoun incidence",
            value=wrdprp1s,
            error=wrdprp1s_error
        ))

        # WRDPRP1p
        try:
            wrdprp1p = cm_wrdprp1p(sentences, counts=_wrd_counts)
            wrdprp1p_error = None
        except Exception as e:
            logger.error("Error calculating WRDPRP1p: %s", e)
            wrdprp1p = None
            wrdprp1p_error = str(e)
        indices.append(Index(
            index=88,
            type_name="Word Information",
            label_v3="WRDPRP1p",
            label_v2="n/a",
            description="First-person plural pronoun incidence",
            value=wrdprp1p,
            error=wrdprp1p_error
        ))

        # WRDPRP2
        try:
            wrdprp2 = cm_wrdprp2(sentences, counts=_wrd_counts)
            wrdprp2_error = None
        except Exception as e:
            logger.error("Error calculating WRDPRP2: %s", e)
            wrdprp2 = None
            wrdprp2_error = str(e)
        indices.append(Index(
            index=89,
            type_name="Word Information",
            label_v3="WRDPRP2",
            label_v2="PRO2i",
            description="Second-person pronoun incidence",
            value=wrdprp2,
            error=wrdprp2_error
        ))

        # WRDPRP3s
        try:
            wrdprp3s = cm_wrdprp3s(sentences, counts=_wrd_counts)
            wrdprp3s_error = None
        except Exception as e:
            logger.error("Error calculating WRDPRP3s: %s", e)
            wrdprp3s = None
            wrdprp3s_error = str(e)
        indices.append(Index(
            index=90,
            type_name="Word Information",
            label_v3="WRDPRP3s",
            label_v2="n/a",
            description="Third-person singular pronoun incidence",
            value=wrdprp3s,
            error=wrdprp3s_error
        ))

        # WRDPRP3p
        try:
            wrdprp3p = cm_wrdprp3p(sentences, counts=_wrd_counts)
            wrdprp3p_error = None
        except Exception as e:
            logger.error("Error calculating WRDPRP3p: %s", e)
            wrdprp3p = None
            wrdprp3p_error = str(e)
        indices.append(Index(
            index=91,
            type_name="Word Information",
            label_v3="WRDPRP3p",
            label_v2="n/a",
            description="Third-person plural pronoun incidence",
            value=wrdprp3p,
            error=wrdprp3p_error
        ))

        # WRDFRQc
        try:
            # wrdfrqc = cm_wrdfrqc(sentences, request.language, "celex")
            wrdfrqc = None
            wrdfrqc_error = None
        except Exception as e:
            logger.error("Error calculating WRDFRQc: %s", e)
            wrdfrqc = None
            wrdfrqc_error = str(e)
        indices.append(Index(
            index=92,
            type_name="Word Information",
            label_v3="WRDFRQc",
            label_v2="FRCLacwm",
            description="CELEX word frequency for content words, mean",
            value=wrdfrqc,
            error=wrdfrqc_error
        ))

        # WRDFRQa
        try:
            # wrdfrqa = cm_wrdfrqa(tokens, request.language, "celex")
            wrdfrqa = None
            wrdfrqa_error = None
        except Exception as e:
            logger.error("Error calculating WRDFRQa: %s", e)
            wrdfrqa = None
            wrdfrqa_error = str(e)
        indices.append(Index(
            index=93,
            type_name="Word Information",
            label_v3="WRDFRQa",
            label_v2="FRCLaewm",
            description="CELEX Log frequency for all words, mean",
            value=wrdfrqa,
            error=wrdfrqa_error
        ))

        # WRDFRQmc
        try:
            # wrdfrqmc = cm_wrdfrqmc(sentences, request.language, "celex")
            wrdfrqmc = None
            wrdfrqmc_error = None
        except Exception as e:
            logger.error("Error calculating WRDFRQmc: %s", e)
            wrdfrqmc = None
            wrdfrqmc_error = str(e)
        indices.append(Index(
            index=94,
            type_name="Word Information",
            label_v3="WRDFRQmc",
            label_v2="FRCLmcsm",
            description="CELEX Log minimum frequency for content words, mean",
            value=wrdfrqmc,
            error=wrdfrqmc_error
        ))

        # WRDFRQc
        try:
            wrdfrqc = cm_wrdfrqc(sentences, request.language, "wiki-20220301-sample10000")
            wrdfrqc_error = None
        except Exception as e:
            logger.error("Error calculating WRDFRQc_wiki10000: %s", e)
            wrdfrqc = None
            wrdfrqc_error = str(e)
        indices.append(Index(
            index=92,
            type_name="Word Information",
            label_ttlab="WRDFRQc_wiki10000",
            label_v3="WRDFRQc",
            label_v2="FRCLacwm",
            description="Wikipedia word frequency for content words, mean",
            version="wiki-20220301-sample10000",
            value=wrdfrqc,
            error=wrdfrqc_error
        ))

        # WRDFRQa
        try:
            wrdfrqa = cm_wrdfrqa(tokens, request.language, "wiki-20220301-sample10000")
            wrdfrqa_error = None
        except Exception as e:
            logger.error("Error calculating WRDFRQa_wiki10000: %s", e)
            wrdfrqa = None
            wrdfrqa_error = str(e)
        indices.append(Index(
            index=93,
            type_name="Word Information",
            label_ttlab="WRDFRQa_wiki10000",
            label_v3="WRDFRQa",
            label_v2="FRCLaewm",
            description="Wikipedia Log frequency for all words, mean",
            version="wiki-20220301-sample10000",
            value=wrdfrqa,
            error=wrdfrqa_error
        ))

        # WRDFRQmc
        try:
            wrdfrqmc = cm_wrdfrqmc(sentences, request.language, "wiki-20220301-sample10000")
            wrdfrqmc_error = None
        except Exception as e:
            logger.error("Error calculating WRDFRQmc_wiki10000: %s", e)
            wrdfrqmc = None
            wrdfrqmc_error = str(e)
        indices.append(Index(
            index=94,
            type_name="Word Information",
            label_ttlab="WRDFRQmc_wiki10000",
            label_v3="WRDFRQmc",
            label_v2="FRCLmcsm",
            description="Wikipedia Log minimum frequency for content words, mean",
            version="wiki-20220301-sample10000",
            value=wrdfrqmc,
            error=wrdfrqmc_error
        ))

        # WRDAOAc
        try:
            wrdaoac = cm_wrdaoac(sentences, request.language)
            wrdaoac_error = None
        except Exception as e:
            logger.error("Error calculating WRDAOAc: %s", e)
            wrdaoac = None
            wrdaoac_error = str(e)
        indices.append(Index(
            index=95,
            type_name="Word Information",
            label_v3="WRDAOAc",
            label_v2="WRDAacwm",
            description="Age of acquisition for content words, mean",
            value=wrdaoac,
            error=wrdaoac_error
        ))

        # WRDFAMc
        try:
            wrdfamc = cm_wrdfamc(sentences, request.language)
            wrdfamc_error = None
        except Exception as e:
            logger.error("Error calculating WRDFAMc: %s", e)
            wrdfamc = None
            wrdfamc_error = str(e)
        indices.append(Index(
            index=96,
            type_name="Word Information",
            label_v3="WRDFAMc",
            label_v2="WRDFacwm",
            description="Familiarity for content words, mean",
            value=wrdfamc,
            error=wrdfamc_error
        ))

        # WRDCNCc
        try:
            wrdcncc = cm_wrdcncc(sentences, request.language)
            wrdcncc_error = None
        except Exception as e:
            logger.error("Error calculating WRDCNCc: %s", e)
            wrdcncc = None
            wrdcncc_error = str(e)
        indices.append(Index(
            index=97,
            type_name="Word Information",
            label_v3="WRDCNCc",
            label_v2="WRDCacwm",
            description="Concreteness for content words, mean",
            value=wrdcncc,
            error=wrdcncc_error
        ))

        # WRDIMGc
        try:
            wrdimgc = cm_wrdimgc(sentences, request.language)
            wrdimgc_error = None
        except Exception as e:
            logger.error("Error calculating WRDIMGc: %s", e)
            wrdimgc = None
            wrdimgc_error = str(e)
        indices.append(Index(
            index=98,
            type_name="Word Information",
            label_v3="WRDIMGc",
            label_v2="WRDIacwm",
            description="Imagability for content words, mean",
            value=wrdimgc,
            error=wrdimgc_error
        ))

        # WRDMEAc
        try:
            wrdmeac = cm_wrdmeac(sentences, request.language)
            wrdmeac_error = None
        except Exception as e:
            logger.error("Error calculating WRDMEAc: %s", e)
            wrdmeac = None
            wrdmeac_error = str(e)
        indices.append(Index(
            index=99,
            type_name="Word Information",
            label_v3="WRDMEAc",
            label_v2="WRDMacwm",
            description="Meaningfulness, Colorado norms, content words, mean",
            value=wrdmeac,
            error=wrdmeac_error
        ))

        # WRDPOLc
        try:
            wrdpolc = cm_wrdpolc(sentences, request.language)
            wrdpolc_error = None
        except Exception as e:
            logger.error("Error calculating WRDPOLc: %s", e)
            wrdpolc = None
            wrdpolc_error = str(e)
        indices.append(Index(
            index=100,
            type_name="Word Information",
            label_v3="WRDPOLc",
            label_v2="POLm",
            description="Polysemy for content words, mean",
            value=wrdpolc,
            error=wrdpolc_error
        ))

        # WRDHYPn
        try:
            wrdhypn = cm_wrdhypn(sentences, request.language)
            wrdhypn_error = None
        except Exception as e:
            logger.error("Error calculating WRDHYPn: %s", e)
            wrdhypn = None
            wrdhypn_error = str(e)
        indices.append(Index(
            index=101,
            type_name="Word Information",
            label_v3="WRDHYPn",
            label_v2="HYNOUNaw",
            description="Hypernymy for nouns, mean",
            value=wrdhypn,
            error=wrdhypn_error
        ))

        # WRDHYPv
        try:
            wrdhypv = cm_wrdhypv(sentences, request.language)
            wrdhypv_error = None
        except Exception as e:
            logger.error("Error calculating WRDHYPv: %s", e)
            wrdhypv = None
            wrdhypv_error = str(e)
        indices.append(Index(
            index=102,
            type_name="Word Information",
            label_v3="WRDHYPv",
            label_v2="HYVERBaw",
            description="Hypernymy for verbs, mean",
            value=wrdhypv,
            error=wrdhypv_error
        ))

        # WRDHYPnv
        try:
            wrdhypnv = cm_wrdhypnv(sentences, request.language)
            wrdhypnv_error = None
        except Exception as e:
            logger.error("Error calculating WRDHYPnv: %s", e)
            wrdhypnv = None
            wrdhypnv_error = str(e)
        indices.append(Index(
            index=103,
            type_name="Word Information",
            label_v3="WRDHYPnv",
            label_v2="HYPm",
            description="Hypernymy for nouns and verbs, mean",
            value=wrdhypnv,
            error=wrdhypnv_error
        ))

        ### Readability

        # NOTE(L4): RDFRE/RDFKGL formulas are spec-correct (Flesch Reading Ease /
        # Flesch-Kincaid Grade Level). Token/syllable counts use textstat's
        # tokenizer, not the original Charniak constituency parser, so absolute
        # values may differ slightly from the reference Coh-Metrix output on the
        # same text. Relative ordering is expected to be consistent.

        # RDFRE
        try:
            rdfre = textstat.flesch_reading_ease(request.text)
            rdfre_error = None
        except Exception as e:
            logger.error("Error calculating RDFRE: %s", e)
            rdfre = None
            rdfre_error = str(e)
        indices.append(Index(
            index=104,
            type_name="Readability",
            label_ttlab="RDFRE_textstat",
            label_v3="RDFRE",
            label_v2="READFRE",
            description="Flesch Reading Ease",
            value=rdfre,
            error=rdfre_error
        ))

        # RDFKGL
        try:
            rdfkgl = textstat.flesch_kincaid_grade(request.text)
            rdfkgl_error = None
        except Exception as e:
            logger.error("Error calculating RDFKGL: %s", e)
            rdfkgl = None
            rdfkgl_error = str(e)
        indices.append(Index(
            index=105,
            type_name="Readability",
            label_ttlab="RDFKGL_textstat",
            label_v3="RDFKGL",
            label_v2="READFKGL",
            description="Flesch–Kincaid Grade Level",
            value=rdfkgl,
            error=rdfkgl_error
        ))

        # RDFOG
        try:
            rdfog = textstat.gunning_fog(request.text)
            rdfog_error = None
        except Exception as e:
            logger.error("Error calculating RDFOG: %s", e)
            rdfog = None
            rdfog_error = str(e)
        indices.append(Index(
            index=1001,
            type_name="Readability",
            label_ttlab="RDFOG_textstat",
            description="Gunning Fog Index",
            value=rdfog,
            error=rdfog_error
        ))

        # RDSMOG
        try:
            rdsmog = textstat.smog_index(request.text)
            rdsmog_error = None
        except Exception as e:
            logger.error("Error calculating RDSMOG: %s", e)
            rdsmog = None
            rdsmog_error = str(e)
        indices.append(Index(
            index=1002,
            type_name="Readability",
            label_ttlab="RDSMOG_textstat",
            description="SMOG Grade",
            value=rdsmog,
            error=rdsmog_error
        ))

        # RDARI
        try:
            rdari = textstat.automated_readability_index(request.text)
            rdari_error = None
        except Exception as e:
            logger.error("Error calculating RDARI: %s", e)
            rdari = None
            rdari_error = str(e)
        indices.append(Index(
            index=1003,
            type_name="Readability",
            label_ttlab="RDARI_textstat",
            description="Automated Readability Index",
            value=rdari,
            error=rdari_error
        ))

        # RDCLI
        try:
            rdcli = textstat.coleman_liau_index(request.text)
            rdcli_error = None
        except Exception as e:
            logger.error("Error calculating RDCLI: %s", e)
            rdcli = None
            rdcli_error = str(e)
        indices.append(Index(
            index=1004,
            type_name="Readability",
            label_ttlab="RDCLI_textstat",
            description="Coleman–Liau Index",
            value=rdcli,
            error=rdcli_error
        ))

        # RDLW
        try:
            rdlw = textstat.linsear_write_formula(request.text)
            rdlw_error = None
        except Exception as e:
            logger.error("Error calculating RDLW: %s", e)
            rdlw = None
            rdlw_error = str(e)
        indices.append(Index(
            index=1005,
            type_name="Readability",
            label_ttlab="RDLW_textstat",
            description="Linsear Write Formula",
            value=rdlw,
            error=rdlw_error
        ))

        # RDDCRS
        try:
            rddcrs = textstat.dale_chall_readability_score(request.text)
            rddcrs_error = None
        except Exception as e:
            logger.error("Error calculating RDDCRS: %s", e)
            rddcrs = None
            rddcrs_error = str(e)
        indices.append(Index(
            index=1006,
            type_name="Readability",
            label_ttlab="RDDCRS_textstat",
            description="Dale-Chall Readability Score",
            value=rddcrs,
            error=rddcrs_error
        ))

        # RDSPACHE
        try:
            rdspache = textstat.spache_readability(request.text)
            rdspache_error = None
        except Exception as e:
            logger.error("Error calculating RDSPACHE: %s", e)
            rdspache = None
            rdspache_error = str(e)
        indices.append(Index(
            index=1007,
            type_name="Readability",
            label_ttlab="RDSPACHE_textstat",
            description="Spache Readability Formula",
            value=rdspache,
            error=rdspache_error
        ))

        # RDWSTF1
        try:
            rdwstf = textstat.wiener_sachtextformel(request.text, variant=1)
            rdwstf_error = None
        except Exception as e:
            logger.error("Error calculating RDWSTF1: %s", e)
            rdwstf = None
            rdwstf_error = str(e)
        indices.append(Index(
            index=1008,
            type_name="Readability",
            label_ttlab="RDWSTF1_textstat",
            description="Wiener Sachtextformel 1",
            value=rdwstf,
            error=rdwstf_error
        ))

        # RDWSTF2
        try:
            rdwstf = textstat.wiener_sachtextformel(request.text, variant=2)
            rdwstf_error = None
        except Exception as e:
            logger.error("Error calculating RDWSTF2: %s", e)
            rdwstf = None
            rdwstf_error = str(e)
        indices.append(Index(
            index=1009,
            type_name="Readability",
            label_ttlab="RDWSTF2_textstat",
            description="Wiener Sachtextformel 2",
            value=rdwstf,
            error=rdwstf_error
        ))

        # RDWSTF3
        try:
            rdwstf = textstat.wiener_sachtextformel(request.text, variant=3)
            rdwstf_error = None
        except Exception as e:
            logger.error("Error calculating RDWSTF3: %s", e)
            rdwstf = None
            rdwstf_error = str(e)
        indices.append(Index(
            index=1010,
            type_name="Readability",
            label_ttlab="RDWSTF3_textstat",
            description="Wiener Sachtextformel 3",
            value=rdwstf,
            error=rdwstf_error
        ))

        # RDWSTF4
        try:
            rdwstf = textstat.wiener_sachtextformel(request.text, variant=4)
            rdwstf_error = None
        except Exception as e:
            logger.error("Error calculating RDWSTF4: %s", e)
            rdwstf = None
            rdwstf_error = str(e)
        indices.append(Index(
            index=1011,
            type_name="Readability",
            label_ttlab="RDWSTF4_textstat",
            description="Wiener Sachtextformel 4",
            value=rdwstf,
            error=rdwstf_error
        ))

        # RDL2
        try:
            rdl2 = cm_rdl2(crfcwo1, synstruta, wrdfrqmc)
            rdl2_error = None
        except Exception as e:
            logger.error("Error calculating RDL2: %s", e)
            rdl2 = None
            rdl2_error = str(e)
        indices.append(Index(
            index=106,
            type_name="Readability",
            label_ttlab="RDL2_synstruta",
            label_v3="RDL2",
            label_v2="L2",
            description="Coh-Metrix L2 Readability",
            value=rdl2,
            error=rdl2_error
        ))

        # RDL2
        try:
            rdl2 = cm_rdl2(crfcwo1, synstrutt, wrdfrqmc)
            rdl2_error = None
        except Exception as e:
            logger.error("Error calculating RDL2: %s", e)
            rdl2 = None
            rdl2_error = str(e)
        indices.append(Index(
            index=106,
            type_name="Readability",
            label_ttlab="RDL2_synstrutt",
            label_v3="RDL2",
            label_v2="L2",
            description="Coh-Metrix L2 Readability",
            value=rdl2,
            error=rdl2_error
        ))

        meta = AnnotationMeta(
            name=settings.annotator_name,
            version=settings.annotator_version,
            modelName=settings.annotator_name,
            modelVersion=settings.annotator_version
        )

        modification_meta = DocumentModification(
            user=settings.annotator_name,
            timestamp=modification_timestamp_seconds,
            comment=f"{settings.annotator_name} ({settings.annotator_version})"
        )

    except Exception as ex:
        logger.exception(ex)

    logger.debug(meta)
    logger.debug(modification_meta)

    duration = int(time()) - modification_timestamp_seconds
    logger.info("Processed in %d seconds", duration)

    response = TextImagerResponse(
        indices=indices,
        meta=meta,
        modification_meta=modification_meta,
    )
    print(response)
    return response
