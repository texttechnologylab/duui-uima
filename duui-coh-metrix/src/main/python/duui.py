import csv
import logging
import pyphen
import textstat
import math

from platform import python_version
from sys import version as sys_version
from time import time
from typing import List, Optional, Dict, Tuple, Set, Any

from cassis import load_typesystem
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, validator
from pydantic_settings import BaseSettings
from lexicalrichness import LexicalRichness
from lexical_diversity import lex_div as ld
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from germanetpy.germanet import Germanet
from germanetpy.synset import WordCategory
from pathlib import Path
from functools import lru_cache

import numpy as np

class Settings(BaseSettings):
    annotator_name: str
    annotator_version: str
    log_level: str
    germanet_path: Optional[str] = None
    lsa_use_truncated_svd: bool = False
    lsa_svd_components: int = 100

    class Config:
        env_prefix = 'duui_coh_metrix_'


settings = Settings()

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)
logger.info("TTLab TextImager DUUI Coh-Metrix")
logger.info("Name: %s", settings.annotator_name)
logger.info("Version: %s", settings.annotator_version)

# LSA configuration
logger.info(
    "LSA TruncatedSVD: %s",
    "enabled" if settings.lsa_use_truncated_svd else "disabled"
)

if settings.lsa_use_truncated_svd:
    logger.info(
        "LSA TruncatedSVD components: %d",
        settings.lsa_svd_components
    )

TEXTIMAGER_ANNOTATOR_OUTPUT_TYPES = [
]

TEXTIMAGER_ANNOTATOR_INPUT_TYPES = [
    "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence",
    "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token",
    "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma",
    "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency",
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
    head_index: Optional[int] = None
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
        # FV1 fix: exclude punctuation to keep consistency
        for token in tokens if not token.is_punct
    ]
    return syllables_counts

# LAY: On average, how many syllables per word?
# ↑ Higher = longer, more complex words. Reliable.
def cm_deswlsy(tokens: List[Token], lang: str) -> Optional[float]:
    # Word length, number of syllables, mean
    # FV1 fix: exclude punctuation to keep consistency
    return np.mean(_syllables_count(tokens, lang))

# LAY: How uneven is the syllable count across words?
# ↑ Higher = mix of short and long words. Reliable.
def cm_deswlsyd(tokens: List[Token], lang: str) -> Optional[float]:
    # Word length, number of syllables, standard deviation
    # FV1 fix: exclude punctuation to keep consistency
    return np.std(_syllables_count(tokens, lang))

# LAY: On average, how many letters per word?
# ↑ Higher = longer words. Reliable.
def cm_deswllt(paragraphs: List[Paragraph]) -> Optional[float]:
    # Word length, number of letters, mean
    # FV1 fix: exclude punctuation to keep consistency
    text_letters = []
    for p in paragraphs:
        for s in p.sentences:
            for t in s.tokens:
                if not t.is_punct:
                    text_letters.append(len(''.join(c for c in t.text if c.isalpha())))
    return np.mean(text_letters)

# LAY: How uneven is the letter count across words?
# ↑ Higher = mix of short and long words. Reliable.
def cm_deswlltd(paragraphs: List[Paragraph]) -> Optional[float]:
    # Word length, number of letters, standard deviation
    # FV1 fix: exclude punctuation to keep consistency
    text_letters = []
    for p in paragraphs:
        for s in p.sentences:
            for t in s.tokens:
                if not t.is_punct:
                    text_letters.append(len(''.join(c for c in t.text if c.isalpha())))
    return np.std(text_letters)

ud_noun_pos = {"NOUN", "PROPN"}
# H6 fix: DET was previously in this set which caused pronoun-overlap (and thus
# argument-overlap) to approach 1.0 for any normal text because determiners
# like "the"/"der"/"die"/"das" appear in nearly every sentence. Spec (Ch. 4,
# §Referential cohesion) specifies pronouns ("he/he"), not determiners.
ud_pronouns_pos = {"PRON"}
ud_content_pos = {"NOUN","PROPN", "VERB", "ADJ", "ADV"}
# FV3 Fix: Use the Coh-Metrix content-word classes for CRFCWO,
# including pronouns and proper nouns.
ud_stem_content_pos = {"NOUN","PROPN","VERB","ADJ","ADV","PRON",}

def _noun_overlap(sentence_a: Sentence, sentence_b: Sentence) -> int:
    nouns_a = set([t.text for t in sentence_a.tokens if t.pos_coarse and t.pos_coarse in ud_noun_pos])
    nouns_b = set([t.text for t in sentence_b.tokens if t.pos_coarse and t.pos_coarse in ud_noun_pos])
    return len(nouns_a.intersection(nouns_b))

def _argument_overlap(sentence_a: Sentence, sentence_b: Sentence) -> int:
    nouns_a = set([t.lemma for t in sentence_a.tokens if t.pos_coarse and t.pos_coarse in ud_noun_pos])
    nouns_b = set([t.lemma for t in sentence_b.tokens if t.pos_coarse and t.pos_coarse in ud_noun_pos])
    noun_overlap = len(nouns_a.intersection(nouns_b))

    pronouns_a = {
        t.text.lower()
        for t in sentence_a.tokens
        if t.pos_coarse in ud_pronouns_pos
    }

    pronouns_b = {
        t.text.lower()
        for t in sentence_b.tokens
        if t.pos_coarse in ud_pronouns_pos
    }
    pronoun_overlap = len(pronouns_a.intersection(pronouns_b))

    return noun_overlap + pronoun_overlap

# FV3 Fix: Stem overlap compares noun lemmas in the current sentence
# with content-word lemmas in the previous sentence and also includes
# matching pronouns as specified for Coh-Metrix stem overlap.
def _stem_overlap(
    sentence_nouns: Sentence,
    sentence_contents: Sentence
) -> int:
    nouns = {
        token.lemma.lower()
        for token in sentence_nouns.tokens
        if token.pos_coarse in ud_noun_pos
        and token.lemma
    }

    content_words = {
        token.lemma.lower()
        for token in sentence_contents.tokens
        if token.pos_coarse in ud_stem_content_pos
        and token.lemma
    }

    noun_content_overlap = len(
        nouns.intersection(content_words)
    )

    pronouns_a = {
        token.text.lower()
        for token in sentence_nouns.tokens
        if token.pos_coarse in ud_pronouns_pos
    }

    pronouns_b = {
        token.text.lower()
        for token in sentence_contents.tokens
        if token.pos_coarse in ud_pronouns_pos
    }

    pronoun_overlap = len(
        pronouns_a.intersection(pronouns_b)
    )

    return noun_content_overlap + pronoun_overlap

# FV3 Fix: CRFCWO measures the proportion of explicit surface-form
# content words in the current sentence that also occur in the previous sentence.
def _word_overlap(
    current_sentence: Sentence,
    previous_sentence: Sentence
) -> float:
    current_content_words = [
        token.text.lower()
        for token in current_sentence.tokens
        if token.pos_coarse in ud_stem_content_pos
        and token.is_alpha
    ]

    previous_content_words = {
        token.text.lower()
        for token in previous_sentence.tokens
        if token.pos_coarse in ud_stem_content_pos
        and token.is_alpha
    }

    if not current_content_words:
        return 0.0

    overlap = sum(
        1
        for word in current_content_words
        if word in previous_content_words
    )
    # FV3 Fix: Normalize overlap by the number of content words in the
    # current sentence instead of using the union of both sentences.
    return overlap / len(current_content_words)

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
# FV3 Fix: Apply the same temporal direction as local CRFCWO:
# compare each later sentence against each earlier sentence.
def cm_crfsoa(sentences: List[Sentence]) -> Optional[float]:
    # Stem overlap, all sentences, binary, mean
    stem_overlap_per_sentence = []
    for sinda in range(len(sentences)):
        for sindb in range(len(sentences)):
            if sindb <= sinda:
                continue
            previous_sentence = sentences[sinda]
            current_sentence = sentences[sindb]
            stem_overlap = min(1, _stem_overlap(current_sentence,previous_sentence))
            stem_overlap_per_sentence.append(stem_overlap)
    return np.mean(stem_overlap_per_sentence)

# LAY: What share of content words is shared between adjacent sentences? (mean)
# ↑ Higher = tighter local cohesion. Reliable.
def cm_crfcwo1(sentences: List[Sentence]) -> Optional[float]:
    word_overlap_per_sentence = []
    for sind in range(1, len(sentences)):
        current_sentence = sentences[sind]
        previous_sentence = sentences[sind - 1]
        word_overlap = _word_overlap(
            current_sentence,
            previous_sentence
        )
        word_overlap_per_sentence.append(word_overlap)
    return np.mean(word_overlap_per_sentence)

# LAY: How uneven is the content-word overlap between adjacent sentences?
# ↑ Higher = some pairs repeat heavily, others not at all. Reliable.
def cm_crfcwo1d(sentences: List[Sentence]) -> Optional[float]:
    word_overlap_per_sentence = []
    for sind in range(1, len(sentences)):
        current_sentence = sentences[sind]
        previous_sentence = sentences[sind - 1]
        word_overlap = _word_overlap(
            current_sentence,
            previous_sentence
        )
        word_overlap_per_sentence.append(word_overlap)
    return np.std(word_overlap_per_sentence)

# LAY: What share of content words is shared across all sentence pairs? (mean)
# ↑ Higher = global cohesion. Reliable.
# FV3 Fix: Apply the same temporal direction as local CRFCWO:
# compare each later sentence against each earlier sentence.
def cm_crfcwoa(sentences: List[Sentence]) -> Optional[float]:
    word_overlap_per_sentence = []
    for sinda in range(len(sentences)):
        for sindb in range(len(sentences)):
            if sindb <= sinda:
                continue
            previous_sentence = sentences[sinda]
            current_sentence = sentences[sindb]
            word_overlap = _word_overlap(
                current_sentence,
                previous_sentence
            )
            word_overlap_per_sentence.append(word_overlap)
    return np.mean(word_overlap_per_sentence)

# LAY: How uneven is the global content-word overlap across sentence pairs?
# ↑ Higher = mix of tight and loose cohesion. Reliable.
# FV3 Fix: Apply the same temporal direction as local CRFCWO:
# compare each later sentence against each earlier sentence.
def cm_crfcwoad(sentences: List[Sentence]) -> Optional[float]:
    word_overlap_per_sentence = []
    for sinda in range(len(sentences)):
        for sindb in range(len(sentences)):
            if sindb <= sinda:
                continue
            previous_sentence = sentences[sinda]
            current_sentence = sentences[sindb]
            word_overlap = _word_overlap(
                current_sentence,
                previous_sentence
            )
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
# ↑ Higher = more left-embedded clauses, harder to read.
def cm_synle(
    sentences: List[Sentence],
    lang: str
) -> Optional[float]:

    lang = (lang or "").strip().lower()

    word_counts = []

    _ROOT_MARKERS = {
        "--",
        "ROOT",
        "root"
    }

    for sentence in sentences:

        if lang == "de":
            # German-specific correction:
            # auxiliary/modal ROOTs are resolved to the lexical
            # main verb of the main verbal complex.
            main_verb_index = _de_synle_main_verb_index(
                sentence
            )

        else:
            # English: keep existing ROOT-based approximation,
            # which passed the diagnostic cases.
            main_verb_index = None

            for token_index, token in enumerate(
                sentence.tokens
            ):
                if token.dep_type in _ROOT_MARKERS:
                    main_verb_index = token_index
                    break

                # Defensive fallback for self-headed ROOT conventions.
                if token.head_index == token_index:
                    main_verb_index = token_index
                    break

        if main_verb_index is None:
            continue

        # Count non-punctuation words occurring before the selected
        # main-verb token. We retain the ORIGINAL sentence-local token
        # indices because head_index refers to this token sequence.
        words_before_main_verb = sum(
            1
            for token_index, token
            in enumerate(sentence.tokens)
            if token_index < main_verb_index
            and not token.is_punct
        )

        word_counts.append(
            words_before_main_verb
        )

    return (
        np.mean(word_counts)
        if word_counts
        else 0
    )

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

def _sequence_levenshtein(seq1, seq2):
    # FV1 fix: Coh-Metrix MED compares linguistic sequence elements
    # (POS tags, words, or lemmas), not individual characters of a joined string.
    # Therefore, insertion, deletion, and substitution operate on complete
    # sequence elements with unit cost.
    if len(seq1) < len(seq2):
        seq1, seq2 = seq2, seq1

    previous_row = list(range(len(seq2) + 1))

    for i, item1 in enumerate(seq1, start=1):
        current_row = [i]

        for j, item2 in enumerate(seq2, start=1):
            insertion = current_row[j - 1] + 1
            deletion = previous_row[j] + 1
            substitution = previous_row[j - 1] + (
                0 if item1 == item2 else 1
            )

            current_row.append(
                min(insertion, deletion, substitution)
            )

        previous_row = current_row

    return previous_row[-1]


def _normalized_sequence_edit_distance(seq1, seq2):
    # FV1 fix: normalize the sequence-level Levenshtein distance by
    # the length of the longer linguistic sequence. This keeps MED in [0, 1]
    # and follows the intended element-level comparison between sentences.
    max_len = max(len(seq1), len(seq2))

    if max_len == 0:
        return np.nan

    distance = _sequence_levenshtein(seq1, seq2)

    return distance / max_len

# LAY: How different are adjacent sentences in their POS-tag sequences?
# ↑ Higher = consecutive sentences have very different grammatical shapes. Reliable.
def cm_synmedpos(sentences: List[Sentence]) -> Optional[float]:
    # H1 fix: compute mean normalized edit distance between CONSECUTIVE SENTENCES'
    # POS-tag sequences (spec: Ch. 4, "distance ... between consecutive sentences"),
    # not between adjacent individual tokens as done previously.
    #
    # FV2 fix: compare complete POS tags as sequence elements instead of
    # applying character-level Levenshtein to space-joined POS strings.
    # Example: replacing NOUN with PRON counts as one substitution, independent
    # of the number of characters in the POS labels.
    sent_pos_sequences = [
        [token.pos_coarse for token in sent.tokens]
        for sent in sentences
    ]

    pos_dists = []

    for i in range(len(sent_pos_sequences) - 1):
        pos_dists.append(
            _normalized_sequence_edit_distance(
                sent_pos_sequences[i],
                sent_pos_sequences[i + 1]
            )
        )

    return np.mean(pos_dists) if pos_dists else 0

# LAY: How different are adjacent sentences word-for-word?
# ↑ Higher = neighbours share fewer surface words. Reliable.
def cm_synmedwrd(sentences: List[Sentence]) -> Optional[float]:
    # H1 fix: compute mean normalized edit distance between CONSECUTIVE SENTENCES'
    # word sequences, following the Coh-Metrix MED definition.
    #
    # FV2 fix: compare complete words as sequence elements instead of
    # characters of a joined sentence string. Replacing one word therefore
    # counts as one substitution regardless of the word's character length.
    sent_word_sequences = [
        [token.text for token in sent.tokens]
        for sent in sentences
    ]

    word_dists = []

    for i in range(len(sent_word_sequences) - 1):
        word_dists.append(
            _normalized_sequence_edit_distance(
                sent_word_sequences[i],
                sent_word_sequences[i + 1]
            )
        )

    return np.mean(word_dists) if word_dists else 0

# LAY: How different are adjacent sentences at the lemma level?
# ↑ Higher = neighbours share fewer word stems. Reliable.
def cm_synmedlem(sentences: List[Sentence]) -> Optional[float]:
    # H1 fix: compute mean normalized edit distance between CONSECUTIVE SENTENCES'
    # lemma sequences, following the Coh-Metrix MED definition.
    #
    # FV2 fix: compare complete lemmas as sequence elements instead of
    # characters of a joined lemma string. This ensures that inflectional
    # variants mapped to the same lemma do not introduce artificial distance.
    sent_lemma_sequences = [
        [token.lemma for token in sent.tokens]
        for sent in sentences
    ]

    lemma_dists = []

    for i in range(len(sent_lemma_sequences) - 1):
        lemma_dists.append(
            _normalized_sequence_edit_distance(
                sent_lemma_sequences[i],
                sent_lemma_sequences[i + 1]
            )
        )

    return np.mean(lemma_dists) if lemma_dists else 0

# NOTE(M9): SYNSTRUT is a documented approximation of Coh-Metrix's
# constituency-tree similarity. Coh-Metrix compares constituency parse trees
# using a maximum-common-tree approach. This implementation reconstructs
# ordered dependency trees from (dep_type, pos_coarse) nodes and head_index.
# Parent-child relations, sibling order, and repeated structures are preserved.
# The largest common ordered dependency structure is approximated recursively
# and normalized as:
#
#   common / (size_tree1 + size_tree2 - common)
#
# SYNSTRUTa compares adjacent sentences; SYNSTRUTt compares sentence pairs
# belonging to different paragraphs. Because dependency trees are used instead
# of the original constituency trees, the resulting values are not numerically
# identical to the original Coh-Metrix output.
def _normalize_tree_dep(dep_type: str) -> str:
    if dep_type in {"--", "ROOT", "root"}:
        return "ROOT"

    return dep_type or ""


def _build_dependency_tree(sentence: Sentence):
    tokens = sentence.tokens

    # FV3 fix: preserve the original token indices because head_index refers
    # to the complete, unfiltered sentence token list.
    valid_indices = {
        i
        for i, token in enumerate(tokens)
        if not token.is_punct
    }

    if not valid_indices:
        return None

    children = {
        i: []
        for i in valid_indices
    }

    roots = []

    def _resolve_non_punct_head(token_index: int):
        token = tokens[token_index]
        head_index = token.head_index

        visited = {token_index}

        while head_index is not None:
            if head_index < 0 or head_index >= len(tokens):
                return None

            # Self-reference denotes ROOT.
            if head_index == token_index:
                return None

            if head_index in visited:
                return None

            visited.add(head_index)

            if head_index in valid_indices:
                return head_index

            # Head is punctuation: continue towards its governor.
            next_head = tokens[head_index].head_index

            if next_head == head_index:
                return None

            head_index = next_head

        return None

    for token_index in sorted(valid_indices):
        parent_index = _resolve_non_punct_head(token_index)

        if parent_index is None:
            roots.append(token_index)
        else:
            children[parent_index].append(token_index)

    # Preserve sentence order among siblings.
    for child_indices in children.values():
        child_indices.sort()

    def _build_node(token_index: int, path=None):
        if path is None:
            path = set()

        # Defensive protection against malformed dependency cycles.
        if token_index in path:
            return None

        new_path = path | {token_index}
        token = tokens[token_index]

        label = (
            _normalize_tree_dep(token.dep_type),
            token.pos_coarse
        )

        child_trees = []

        for child_index in children[token_index]:
            child_tree = _build_node(
                child_index,
                new_path
            )

            if child_tree is not None:
                child_trees.append(child_tree)

        return (
            label,
            tuple(child_trees)
        )

    root_trees = []

    for root_index in sorted(roots):
        root_tree = _build_node(root_index)

        if root_tree is not None:
            root_trees.append(root_tree)

    if not root_trees:
        return None

    # Virtual sentence root. Coh-Metrix operates on complete parse trees;
    # this provides an equivalent common starting point for dependency trees.
    return (
        ("__ROOT__", "__ROOT__"),
        tuple(root_trees)
    )


@lru_cache(maxsize=4096)
def _tree_size(tree) -> int:
    _, children = tree

    return 1 + sum(
        _tree_size(child)
        for child in children
    )


@lru_cache(maxsize=16384)
def _maximum_common_tree_size(tree1, tree2) -> int:
    label1, children1 = tree1
    label2, children2 = tree2

    if label1 != label2:
        return 0

    n = len(children1)
    m = len(children2)

    # DP table for the optimal matching of the two child sequences.
    dp = [
        [0] * (m + 1)
        for _ in range(n + 1)
    ]

    for i in range(1, n + 1):
        for j in range(1, m + 1):

            matched_subtree = _maximum_common_tree_size(
                children1[i - 1],
                children2[j - 1]
            )

            dp[i][j] = max(
                dp[i - 1][j],
                dp[i][j - 1],
                dp[i - 1][j - 1] + matched_subtree
            )

    # +1 for the matching current node.
    return 1 + dp[n][m]

def _clear_synstrut_caches() -> None:
    _tree_size.cache_clear()
    _maximum_common_tree_size.cache_clear()

def _compute_tree_similarity(
    tree1,
    tree2
) -> float:
    if tree1 is None or tree2 is None:
        return 0.0

    size1 = _tree_size(tree1)
    size2 = _tree_size(tree2)

    common_size = _maximum_common_tree_size(
        tree1,
        tree2
    )

    denominator = (
        size1
        + size2
        - common_size
    )

    if denominator == 0:
        return 0.0

    return common_size / denominator


# LAY: Mean syntactic similarity between adjacent sentences.
# ↑ Higher = consecutive sentences use more similar syntactic structures.
# Approximate for Coh-Metrix 3.0 because dependency trees are used instead
# of the original constituency parse trees.
def cm_synstruta(
    sentences: List[Sentence]
) -> float:

    similarities = []

    for i in range(len(sentences) - 1):
        tree1 = _build_dependency_tree(
            sentences[i]
        )

        tree2 = _build_dependency_tree(
            sentences[i + 1]
        )

        if tree1 is None or tree2 is None:
            continue

        similarity = _compute_tree_similarity(
            tree1,
            tree2
        )

        similarities.append(similarity)

    return (
        np.mean(similarities)
        if similarities
        else 0.0
    )


# LAY: Mean syntactic similarity between all sentence pairs that belong
# to different paragraphs.
# ↑ Higher = syntactic structures remain similar across paragraph boundaries.
# Approximate for Coh-Metrix 3.0 because dependency trees are used instead
# of the original constituency parse trees.
def cm_synstrutt(
    paragraphs: List[Paragraph]
) -> float:

    paragraph_trees = []

    for paragraph in paragraphs:
        trees = []

        for sentence in paragraph.sentences:
            tree = _build_dependency_tree(
                sentence
            )

            if tree is not None:
                trees.append(tree)

        paragraph_trees.append(trees)

    similarities = []

    # FV3 fix: SYNSTRUTt compares sentence pairs ACROSS paragraphs,
    # not aggregated paragraph structures.
    for paragraph_i in range(
        len(paragraph_trees) - 1
    ):
        for paragraph_j in range(
            paragraph_i + 1,
            len(paragraph_trees)
        ):
            for tree1 in paragraph_trees[paragraph_i]:
                for tree2 in paragraph_trees[paragraph_j]:

                    similarity = _compute_tree_similarity(
                        tree1,
                        tree2
                    )

                    similarities.append(similarity)

    return (
        np.mean(similarities)
        if similarities
        else 0.0
    )

_DE_NEGATION_LEMMAS = {
    "nicht",
    "kein",
    "keiner",
    "niemand",
    "nichts",
    "nie",
    "niemals",
    "nirgends",
    "nirgendwo",
    "keinesfalls",
    "keineswegs",
    "mitnichten",
    "nein",
    "weder",
}

_EN_NEGATION_LEMMAS = {
    "not",
    "no",
    "nobody",
    "nothing",
    "none",
    "never",
    "neither",
    "nowhere",
}

_DE_INFINITIVE_TAGS = {
    "VVINF",   # lexical verb infinitive
    "VAINF",   # auxiliary infinitive
    "VMINF",   # modal infinitive
    "VVIZU",   # zu-infinitive, if emitted as a combined STTS tag
}

_DE_PASSIVE_AUX_LEMMAS = {"werden", "sein"}

def _get_head_index(sentence: Sentence, token_index: int) -> Optional[int]:
    if token_index < 0 or token_index >= len(sentence.tokens):
        return None

    head_index = sentence.tokens[token_index].head_index

    if head_index is None:
        return None
    if head_index < 0 or head_index >= len(sentence.tokens):
        return None
    if head_index == token_index:
        return None

    return head_index


def _head_chain_reaches(
    sentence: Sentence,
    start_index: int,
    target_indices: set[int],
    max_hops: int = 8,
) -> bool:
    current = start_index
    visited = set()

    for _ in range(max_hops):
        if current in visited:
            return False
        visited.add(current)

        head_index = _get_head_index(sentence, current)
        if head_index is None:
            return False

        if head_index in target_indices:
            return True

        current = head_index

    return False


def _is_adverbial_modifier(token: Token, lang: str) -> bool:
    if token.pos_coarse != "ADV":
        return False

    dep = (token.dep_type or "").strip()

    if lang == "de":
        return dep == "MO"

    return dep.lower() in {"advmod"}


def _count_adverbial_phrases(sentence: Sentence, lang: str) -> int:
    count = 0

    for i, token in enumerate(sentence.tokens):
        if not _is_adverbial_modifier(token, lang):
            continue

        head_index = _get_head_index(sentence, i)

        # If this ADV/MO modifies another ADV/MO, it belongs to the same
        # adverbial phrase and is not counted as a separate phrase head.
        if head_index is not None:
            head = sentence.tokens[head_index]
            if _is_adverbial_modifier(head, lang):
                continue

        count += 1

    return count


def _german_passive_participle_indices(sentence: Sentence) -> set[int]:
    passive_participles: set[int] = set()

    for i, token in enumerate(sentence.tokens):
        if token.pos_value != "VVPP":
            continue
        if token.pos_coarse not in {"VERB", "AUX"}:
            continue

        current = i
        visited = set()

        for _ in range(8):
            if current in visited:
                break
            visited.add(current)

            head_index = _get_head_index(sentence, current)
            if head_index is None:
                break

            head = sentence.tokens[head_index]
            head_lemma = (head.lemma or head.text or "").lower()

            if head.pos_coarse == "AUX" and head_lemma in _DE_PASSIVE_AUX_LEMMAS:
                if head_lemma == "werden":
                    passive_participles.add(i)
                elif head_lemma == "sein" and token.dep_type == "PD":
                    passive_participles.add(i)
                break

            current = head_index

    return passive_participles


def _german_passive_has_explicit_agent(
    sentence: Sentence,
    passive_participles: set[int],
) -> bool:
    if not passive_participles:
        return False

    for i, token in enumerate(sentence.tokens):
        dep = (token.dep_type or "").upper()
        lemma = (token.lemma or token.text or "").lower()

        # Strong TIGER passive-subject / agent signal.
        if dep == "SBP":
            if _head_chain_reaches(sentence, i, passive_participles):
                return True

        # Diagnostic approximation for "durch den Jungen".
        if (
            token.pos_coarse == "ADP"
            and lemma == "durch"
            and dep == "MO"
            and _head_chain_reaches(sentence, i, passive_participles)
        ):
            return True

    return False


def _english_passive_status(sentence: Sentence) -> Tuple[bool, bool]:
    passive_deps = {
        "auxpass",
        "nsubjpass",
        "csubjpass",
    }
    agent_deps = {"agent"}

    is_passive = any(
        (token.dep_type or "").lower() in passive_deps
        for token in sentence.tokens
    )

    has_agent = any(
        (token.dep_type or "").lower() in agent_deps
        for token in sentence.tokens
    )

    return is_passive, has_agent


def _is_infinitive_token(sentence: Sentence, token_index: int, lang: str) -> bool:
    token = sentence.tokens[token_index]

    if token.pos_coarse not in {"VERB", "AUX"}:
        return False

    if lang == "de":
        if token.pos_value not in _DE_INFINITIVE_TAGS:
            return False

        # Prevent known false positives: finite verb tagged VVINF but with tense.
        if (token.morph_tense or "").strip():
            return False

        return True

    # English
    if token.pos_value != "VB":
        return False

    # "to + VB"
    if token_index > 0:
        prev = sentence.tokens[token_index - 1]
        if (prev.lemma or prev.text or "").lower() == "to":
            return True

    # FV3 fix: bare infinitive with modal auxiliary.
    # In spaCy, the modal AUX is attached TO the lexical VB.
    # Example: "can sleep" -> can(AUX/MD) --AUX--> sleep(VERB/VB)
    for child_index, child in enumerate(sentence.tokens):
        if (
                child_index != token_index
                and child.head_index == token_index
                and child.pos_coarse == "AUX"
                and child.pos_value == "MD"
        ):
            return True

    return False


def _count_metrics(
    sentences: List[Sentence],
    noun_chunks: List[NounChunk],
    lang: str
) -> Dict[str, int]:
    count_metrics_dict = {
        "total_tokens": 0,
        "total_sentences": len(sentences),
        "noun_phrase_count": len(noun_chunks),

        # legacy/raw counters retained
        "verb_count": 0,
        "adverb_count": 0,
        "prep_count": 0,
        "passive_sentences": 0,

        # V3-oriented structural counters
        "verb_phrase_count": 0,
        "adverbial_phrase_count": 0,
        "agentless_passive_sentences": 0,

        "neg_count": 0,
        "gerund_count": 0,
        "infinitive_count": 0,
    }

    for sentence in sentences:
        # DRAP: phrase-level count from dependency hierarchy.
        count_metrics_dict["adverbial_phrase_count"] += _count_adverbial_phrases(
            sentence, lang
        )

        # DRPVAL: sentence-level passive + explicit-agent distinction.
        if lang == "de":
            passive_participles = _german_passive_participle_indices(sentence)
            is_passive = bool(passive_participles)
            has_explicit_agent = _german_passive_has_explicit_agent(
                sentence, passive_participles
            )
        else:
            is_passive, has_explicit_agent = _english_passive_status(sentence)

        if is_passive:
            count_metrics_dict["passive_sentences"] += 1

            if not has_explicit_agent:
                count_metrics_dict["agentless_passive_sentences"] += 1

        for i, token in enumerate(sentence.tokens):
            if not token.is_punct:
                count_metrics_dict["total_tokens"] += 1

            # Legacy lexical counters.
            if token.pos_coarse == "VERB":
                count_metrics_dict["verb_count"] += 1
            elif token.pos_coarse == "ADV":
                count_metrics_dict["adverb_count"] += 1
            elif token.pos_coarse == "ADP":
                count_metrics_dict["prep_count"] += 1

            # DRVP approximation:
            # constituency VP projections are approximated by verbal heads.
            # AUX is intentionally included; diagnostic AUX+VERB constructions
            # therefore contribute two verbal projections instead of one.
            if token.pos_coarse in {"VERB", "AUX"}:
                count_metrics_dict["verb_phrase_count"] += 1

            dep = (token.dep_type or "").upper()
            lemma = (token.lemma or token.text or "").lower()

            # DRNEG: dependency signal + conservative German lexical coverage.
            if lang == "de":
                is_negation = (
                    dep in {"NEG", "NG"}
                    or lemma in _DE_NEGATION_LEMMAS
                )
            else:
                is_negation = (
                        dep in {"NEG", "NG"}
                        or lemma in _EN_NEGATION_LEMMAS
                )
            if is_negation:
                count_metrics_dict["neg_count"] += 1

            # DRGERUND: English-only VBG approximation.
            if lang != "de" and token.pos_value == "VBG":
                count_metrics_dict["gerund_count"] += 1

            # DRINF
            if _is_infinitive_token(sentence, i, lang):
                count_metrics_dict["infinitive_count"] += 1

    return count_metrics_dict

# ============================================================================
# SYNTACTIC PATTERN DENSITY (DR*) — "Which grammatical constructions appear,
# and how often?"
# The implemented measures are incidences per 1,000 non-punctuation words.
# Constituency-based Coh-Metrix patterns are approximated from the available
# dependency/POS annotations where necessary.
#
# DRGERUND is English-specific and returns None for German because German has
# no directly corresponding grammatical gerund category.
# ============================================================================

# LAY: How many noun phrases per 1,000 words?
# ↑ Higher = more noun-heavy text. Reliable.
def cm_drnp(
    sentences: List[Sentence],
    noun_chunks: List[NounChunk],
    lang: str,
    metrics: Optional[Dict[str, int]] = None
) -> Optional[float]:
    # H12: DRNP incidence per 1000 words.
    if metrics is None:
        metrics = _count_metrics(sentences, noun_chunks, lang)
    return _incidence(metrics["noun_phrase_count"], metrics["total_tokens"])

# LAY: How many verb-phrase approximations occur per 1,000 words?
# ↑ Higher = greater density of verbal structures.
# Constituency VP nodes are approximated by VERB/AUX verbal projections.
def cm_drvp(
    sentences: List[Sentence],
    noun_chunks: List[NounChunk],
    lang: str,
    metrics: Optional[Dict[str, int]] = None
) -> Optional[float]:
    # H12: DRVP incidence per 1000 words.
    # FV3 approximation: verbal phrase/projection density, not raw VERB density.
    if metrics is None:
        metrics = _count_metrics(sentences, noun_chunks, lang)
    return _incidence(metrics["verb_phrase_count"], metrics["total_tokens"])

# LAY: How many adverbial-phrase approximations occur per 1,000 words?
# ↑ Higher = greater density of adverbial structures.
# Maximal dependency-based adverbial structures approximate constituency ADVPs.
def cm_drap(
    sentences: List[Sentence],
    noun_chunks: List[NounChunk],
    lang: str,
    metrics: Optional[Dict[str, int]] = None
) -> Optional[float]:
    # H12: DRAP incidence per 1000 words.
    # FV3 approximation: maximal dependency-based adverbial phrases.
    if metrics is None:
        metrics = _count_metrics(sentences, noun_chunks, lang)
    return _incidence(metrics["adverbial_phrase_count"], metrics["total_tokens"])

# LAY: How many prepositional-phrase approximations occur per 1,000 words?
# ↑ Higher = greater density of prepositional structures.
# The current approximation uses ADP/preposition heads rather than
# constituency PP nodes.
def cm_drpp(
    sentences: List[Sentence],
    noun_chunks: List[NounChunk],
    lang: str,
    metrics: Optional[Dict[str, int]] = None
) -> Optional[float]:
    # H12: DRPP incidence per 1000 words.
    # Retained: ADP/preposition incidence is the current V3 approximation.
    if metrics is None:
        metrics = _count_metrics(sentences, noun_chunks, lang)
    return _incidence(metrics["prep_count"], metrics["total_tokens"])

# LAY: How many agentless passive constructions occur per 1,000 words?
# ↑ Higher = more agentless passive/impersonal constructions.
# Dependency-based approximation of the Coh-Metrix agentless passive index.
def cm_drpval(
    sentences: List[Sentence],
    noun_chunks: List[NounChunk],
    lang: str,
    metrics: Optional[Dict[str, int]] = None
) -> Optional[float]:
    # DRPVAL: agentless passive voice incidence per 1,000 words.
    # Passive detection and explicit-agent detection are language-specific
    # dependency-based approximations.
    if metrics is None:
        metrics = _count_metrics(sentences, noun_chunks, lang)
    return _incidence(
        metrics["agentless_passive_sentences"],
        metrics["total_tokens"]
    )

# LAY: How many negations (not/nicht/kein/...) per 1,000 words?
# ↑ Higher = more negation. Reliable.
def cm_drneg(
    sentences: List[Sentence],
    noun_chunks: List[NounChunk],
    lang: str,
    metrics: Optional[Dict[str, int]] = None
) -> Optional[float]:
    # H12: DRNEG incidence per 1000 words.
    if metrics is None:
        metrics = _count_metrics(sentences, noun_chunks, lang)
    return _incidence(metrics["neg_count"], metrics["total_tokens"])

# LAY: How many English gerunds ("-ing" verb forms) per 1,000 words?
# Language-limited: DE=0 (German has no gerund equivalent — NOTE(H11)).
def cm_drgerund(
    sentences: List[Sentence],
    noun_chunks: List[NounChunk],
    lang: str,
    metrics: Optional[Dict[str, int]] = None
) -> Optional[float]:
    # H12: DRGERUND incidence per 1000 words.
    # German has no direct grammatical equivalent of the English -ing gerund.
    # None means "not applicable", not "zero gerunds observed".
    if lang == "de":
        return None

    if metrics is None:
        metrics = _count_metrics(sentences, noun_chunks, lang)

    return _incidence(metrics["gerund_count"], metrics["total_tokens"])

# LAY: How many infinitive verb forms (to/zu + VERB) per 1,000 words?
# ↑ Higher = more infinitive constructions. Reliable.
def cm_drinf(
    sentences: List[Sentence],
    noun_chunks: List[NounChunk],
    lang: str,
    metrics: Optional[Dict[str, int]] = None
) -> Optional[float]:
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

def _normalize_morph_person(
    value: Optional[str]
) -> Optional[str]:

    if not value:
        return None

    value = str(value).strip().lower()

    if value in {"1", "1st", "first"}:
        return "1"

    if value in {"2", "2nd", "second"}:
        return "2"

    if value in {"3", "3rd", "third"}:
        return "3"

    return None


def _normalize_morph_number(
    value: Optional[str]
) -> Optional[str]:

    if not value:
        return None

    value = str(value).strip().lower()

    if value in {
        "sing",
        "singular",
        "sg",
        "s",
    }:
        return "sing"

    if value in {
        "plur",
        "plural",
        "pl",
        "p",
    }:
        return "plur"

    return None

# FV4 fix:
# WRD incidence scores use the same word denominator as DESWC:
# all non-punctuation tokens.
#
# Pronoun subcategories are classified per token from morphological
# Person/Number annotations rather than from sets of surface forms.
# This prevents ambiguous forms such as German "sie" from being assigned
# to several person/number categories simultaneously.
def _count_words(
    sentences: List[Sentence]
) -> Dict[str, float]:

    counters = {
        "noun": 0,
        "verb": 0,
        "adj": 0,
        "adv": 0,
        "pronoun_total": 0,
        "prp1s": 0,
        "prp1p": 0,
        "prp2": 0,
        "prp3s": 0,
        "prp3p": 0,
    }

    total_words = 0

    for sentence in sentences:
        for token in sentence.tokens:

            # Use the same word basis as DESWC and the other
            # incidence-based indices: all non-punctuation tokens.
            if token.is_punct:
                continue

            total_words += 1

            pos = token.pos_coarse

            if pos in {"NOUN", "PROPN"}:
                counters["noun"] += 1

            elif pos == "VERB":
                counters["verb"] += 1

            elif pos == "ADJ":
                counters["adj"] += 1

            elif pos == "ADV":
                counters["adv"] += 1

            elif pos == "PRON":
                counters["pronoun_total"] += 1

                person = _normalize_morph_person(
                    token.morph_person
                )

                number = _normalize_morph_number(
                    token.morph_number
                )

                # First person
                if person == "1":
                    if number == "sing":
                        counters["prp1s"] += 1
                    elif number == "plur":
                        counters["prp1p"] += 1

                # Second person:
                # Coh-Metrix has one combined second-person index.
                elif person == "2":
                    counters["prp2"] += 1

                # Third person
                elif person == "3":
                    if number == "sing":
                        counters["prp3s"] += 1
                    elif number == "plur":
                        counters["prp3p"] += 1

    return {
        key: _incidence(value, total_words)
        for key, value in counters.items()
    }

def _wrd_precompute(
    sentences: List[Sentence]
) -> Dict[str, float]:
    return _count_words(sentences)

# ============================================================================
# WORD INFORMATION (WRD*) — "What kind of words are used?"
# Three sub-groups:
#   (a) POS & pronoun incidences — counts per 1,000 words. Reliable.
#   (b) Psycholinguistic ratings (AoA, familiarity, concreteness,
#       imageability, meaningfulness). English uses the processed MRC norms;
#       German uses the project-specific translated/aggregated approximation.
#       German and English values therefore do not represent identical lexical
#       resources and should be interpreted accordingly.
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

# FV1 Fix: Included support for german language, respective database is
# now loaded, depending on the text language.
@lru_cache(maxsize=2)
def _load_mrc_database(lang: str) -> Dict[str, Dict[str, Optional[float]]]:
    if lang == "de":
        filepath = "src/main/resources/mrc_psycholinguistic_database_de.csv"
    else:
        filepath = "src/main/resources/mrc_psycholinguistic_database.csv"

    mrc_dict = {}

    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=',')

        for row in reader:
            word = row['Word'].lower()

            # FV2 fix: WRDMEAc is defined specifically by the MRC Colorado
            # meaningfulness norms. Paivio meaningfulness values must therefore
            # not be combined with, or used as a fallback for, the Colorado rating.
            meaningful_colorado = (
                float(row['Meaningfulness: Coloradao Norms'])
                if row['Meaningfulness: Coloradao Norms']
                else None
            )

            mrc_dict[word] = {
                'AoA': float(row['Age of Acquisition Rating'])
                if row['Age of Acquisition Rating'] else None,

                'Familiarity': float(row['Familiarity'])
                if row['Familiarity'] else None,

                'Concreteness': float(row['Concreteness'])
                if row['Concreteness'] else None,

                'Imageability': float(row['Imageability'])
                if row['Imageability'] else None,

                'Meaningfulness': meaningful_colorado
            }

    return mrc_dict

# FV1 fix: Removed old mrc_dict creation
#mrc_dict = _load_mrc_database()

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
# ↑ Higher = later-acquired, harder words.
# FV1 Fix: Reworked Function for german language Support.
def cm_wrdaoac(sentences: List[Sentence], lang: str, mrc_dict: Optional[Dict[str, float]] = None) -> Optional[float]:
    # H8: MRC Psycholinguistic Database is English-only.
    if mrc_dict is None:
        mrc_dict = _load_mrc_database(lang)
    content_words, _, _ = _get_content_words(sentences)
    if not content_words:
        return None
    return _average_rating(content_words, mrc_dict, 'AoA')

# LAY: Average subjective familiarity rating of content words (MRC norms).
# ↑ Higher = more familiar, easier words.
# FV1 Fix: Reworked Function for german language Support.
def cm_wrdfamc(sentences: List[Sentence], lang: str, mrc_dict: Optional[Dict[str, float]] = None) -> Optional[float]:
    if mrc_dict is None:
        mrc_dict = _load_mrc_database(lang)
    content_words, _, _ = _get_content_words(sentences)
    if not content_words:
        return None
    return _average_rating(content_words, mrc_dict, 'Familiarity')

# LAY: Average concreteness of content words (MRC norms).
# ↑ Higher = more concrete, sensory words.
# FV1 Fix: Reworked Function for german language Support.
def cm_wrdcncc(sentences: List[Sentence], lang: str, mrc_dict: Optional[Dict[str, float]] = None) -> Optional[float]:
    if mrc_dict is None:
        mrc_dict = _load_mrc_database(lang)
    content_words, _, _ = _get_content_words(sentences)
    if not content_words:
        return None
    return _average_rating(content_words, mrc_dict, 'Concreteness')

# LAY: Average imageability of content words (MRC norms).
# ↑ Higher = easier to form a mental picture.
# FV1 Fix: Reworked Function for german language Support.
def cm_wrdimgc(sentences: List[Sentence], lang: str, mrc_dict: Optional[Dict[str, float]] = None) -> Optional[float]:
    if mrc_dict is None:
        mrc_dict = _load_mrc_database(lang)
    content_words, _, _ = _get_content_words(sentences)
    if not content_words:
        return None
    return _average_rating(content_words, mrc_dict, 'Imageability')

# LAY: Average subjective meaningfulness of content words (MRC/Colorado norms).
# ↑ Higher = stronger semantic associations.
# FV1 Fix: Reworked Function for german language Support.
# FV3 FIX: Uses the Colorado meaningfulness field only; Paivio meaningfulness is not
# combined with or substituted for the Colorado rating.
def cm_wrdmeac(sentences: List[Sentence], lang: str, mrc_dict: Optional[Dict[str, float]] = None) -> Optional[float]:
    if mrc_dict is None:
        mrc_dict = _load_mrc_database(lang)
    content_words, _, _ = _get_content_words(sentences)
    if not content_words:
        return None
    return _average_rating(content_words, mrc_dict, 'Meaningfulness')

def _get_lexical_lookup_form(
    token: Token,
    lang: str
) -> Optional[str]:
    if token.pos_coarse not in ud_content_pos:
        return None

    if not token.is_alpha:
        return None

    lemma = (token.lemma or "").strip()

    if not lemma or lemma == "--":
        lemma = token.text.strip()

    if not lemma:
        return None

    # Princeton WordNet entries are normalized to lowercase.
    if lang == "en":
        return lemma.lower()

    # Preserve German lemma casing, especially for nouns.
    # Individual GermaNet-based measures decide separately whether lookup should
    # be case-sensitive; polysemy currently preserves case, whereas the hypernym
    # lookup uses ignorecase=True.
    return lemma


def _iter_content_lemmas(
    sentences: List[Sentence],
    lang: str
):
    for sentence in sentences:
        for token in sentence.tokens:
            lemma = _get_lexical_lookup_form(token, lang)

            if lemma is not None:
                yield lemma, token.pos_coarse

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
            # FV2 refinement: German lemma casing is preserved by
            # _get_lexical_lookup_form(). Keep the GermaNet lookup
            # case-sensitive here to avoid conflating lexically distinct
            # German forms that differ only in capitalization.
            return len(
                germanet.get_synsets_by_orthform(word)
            )
        except Exception:
            return None

    return None

def _get_max_hypernym_depth(word, pos=None, lang: str = "en"):
    # H9: dispatch by language (see _get_polysemy).
    # FV2 Fix: Set ignorecase for germanet to prevent not finding words because
    # of upper/lowercase start
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
            synsets = germanet.get_synsets_by_orthform(word, ignorecase=True)
            if word_cat is not None:
                synsets = [s for s in synsets if s.word_category == word_cat]
            if not synsets:
                return None
            depths = []

            for synset in synsets:
                paths = synset.hypernym_paths()

                if not paths:
                    continue

                max_depth = max(
                    len(path) - 1
                    for path in paths
                )

                depths.append(max_depth)
            return max(depths) if depths else None
        except Exception:
            return None
    return None

# LAY: Average polysemy: how many senses each content word has.
# ↑ Higher = more ambiguous vocabulary. Approximate for DE (GermaNet) — NOTE(H9).
def cm_wrdpolc(
    sentences: List[Sentence],
    lang: str
) -> Optional[float]:
    polysemies = []

    for lemma, _ in _iter_content_lemmas(sentences, lang):
        polysemy = _get_polysemy(
            lemma,
            lang=lang
        )

        # Preserve the current coverage policy:
        # uncovered words are not included as zero values.
        if polysemy is None or polysemy <= 0:
            continue

        polysemies.append(polysemy)

    return np.mean(polysemies) if polysemies else None

def _calc_wrdhyp(
    sentences: List[Sentence],
    lang: str
) -> Dict[str, Optional[float]]:
    hypernym_nouns = []
    hypernym_verbs = []

    for lemma, pos in _iter_content_lemmas(sentences, lang):
        if pos not in {"NOUN", "VERB"}:
            continue

        if lang == "en":
            pos_for_lookup = (
                wn.NOUN
                if pos == "NOUN"
                else wn.VERB
            )
        else:
            # _get_max_hypernym_depth maps these values
            # to the corresponding GermaNet WordCategory.
            pos_for_lookup = pos

        depth = _get_max_hypernym_depth(
            lemma,
            pos=pos_for_lookup,
            lang=lang
        )

        if depth is None:
            continue

        if pos == "NOUN":
            hypernym_nouns.append(depth)
        else:
            hypernym_verbs.append(depth)

    hypn_avg = (
        sum(hypernym_nouns) / len(hypernym_nouns)
        if hypernym_nouns
        else None
    )

    hypv_avg = (
        sum(hypernym_verbs) / len(hypernym_verbs)
        if hypernym_verbs
        else None
    )

    combined = hypernym_nouns + hypernym_verbs

    hypnv_avg = (
        sum(combined) / len(combined)
        if combined
        else None
    )

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

_CONNECTIVE_CATEGORY_TO_INDEX = {
    "Causal": "CNCCaus",
    "Logical": "CNCLogic",
    "Adversative": "CNCADC",
    "Temporal": "CNCTemp",
    "Expanded": "CNCTempX",
    "Additive": "CNCAdd",
    "Positive": "CNCPos",
    "Negative": "CNCNeg",
}


def _build_connective_patterns(
    lang: str
) -> Dict[Tuple[str, ...], Set[str]]:
    patterns: Dict[Tuple[str, ...], Set[str]] = defaultdict(set)

    for category, language_lists in connectives_list.items():
        for expression in language_lists.get(lang, set()):

            pattern = tuple(
                expression.lower().split()
            )

            if not pattern:
                continue

            patterns[pattern].add(category)

    return patterns


def _find_non_overlapping_connectives(
    text: str,
    lang: str
) -> List[Tuple[int, int, Tuple[str, ...], Set[str]]]:
    tokens = _normalize_tokens_for_connectives(text)
    patterns = _build_connective_patterns(lang)

    candidates = []

    # Find all possible occurrences first.
    for pattern, categories in patterns.items():
        pattern_length = len(pattern)

        for start_index in range(
            len(tokens) - pattern_length + 1
        ):
            end_index = start_index + pattern_length

            if tuple(tokens[start_index:end_index]) == pattern:
                candidates.append(
                    (
                        start_index,
                        end_index,
                        pattern,
                        categories
                    )
                )
    candidates.sort(
        key=lambda match: (
            -(match[1] - match[0]),
            match[0]
        )
    )

    occupied_indices: Set[int] = set()
    accepted_matches = []

    for (
        start_index,
        end_index,
        pattern,
        categories
    ) in candidates:

        span = set(
            range(start_index, end_index)
        )
        if span.intersection(occupied_indices):
            continue

        accepted_matches.append(
            (
                start_index,
                end_index,
                pattern,
                categories
            )
        )

        occupied_indices.update(span)
    accepted_matches.sort(
        key=lambda match: match[0]
    )

    return accepted_matches

def _count_connectives(
    text: str,
    lang: str,
    total_words: int
) -> Dict[str, float]:

    raw_counts = {
        "CNCAll": 0,
        "CNCCaus": 0,
        "CNCLogic": 0,
        "CNCADC": 0,
        "CNCTemp": 0,
        "CNCTempX": 0,
        "CNCAdd": 0,
        "CNCPos": 0,
        "CNCNeg": 0,
    }

    matches = _find_non_overlapping_connectives(
        text,
        lang
    )

    for _, _, _, categories in matches:
        raw_counts["CNCAll"] += 1
        for category in categories:
            index_name = _CONNECTIVE_CATEGORY_TO_INDEX.get(
                category
            )

            if index_name is not None:
                raw_counts[index_name] += 1

    return {
        key: _incidence(value, total_words)
        for key, value in raw_counts.items()
    }

# ============================================================================
# CONNECTIVES (CNC*) — "How often are connecting expressions used?"
# All values are incidences per 1,000 words.
#
# The connective inventories are project-specific approximations of the
# Coh-Metrix connective categories. Matching is case-insensitive and
# longest-expression-first. Overlapping shorter expressions are suppressed.
# Each accepted surface occurrence contributes once to CNCAll, while the same
# complete expression may legitimately contribute to multiple connective
# categories when it is listed in more than one category.
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

# NOTE(M5): LSA approximation. The original Coh-Metrix LSA indices are based
# on an LSA semantic space trained on the TASA corpus. This implementation
# instead uses the word vectors provided by the upstream spaCy component and
# averages them to obtain sentence and paragraph vectors.
#
# Optional per-document TruncatedSVD dimensionality reduction is available
# through the lsa_use_truncated_svd setting but is disabled by default.
# Therefore, the default implementation operates directly in the spaCy vector
# space. The resulting values are an approximation and are not numerically
# equivalent to the original TASA-based Coh-Metrix LSA scores.
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
# prior context, so G = 0 and the ratio is 0. # The subspace of the preceding
# sentence vectors is determined using an SVD.
# Only singular vectors belonging to the numerical rank of the basis are
# retained. The current sentence vector is projected onto this orthonormal
# basis; the projection represents the "given" component and the orthogonal
# residual represents the "new" component. Rank-aware projection prevents
# linearly dependent preceding vectors from introducing artificial dimensions.
# Caveat: the implementation uses spaCy word vectors (see NOTE(M5)), not TASA-
# trained LSA, so absolute values differ from the reference Coh-Metrix output.
# @see docs/review-report-v3.html (M5, L3) for the LSA approximation context.
# FV3 Fix: Build the Given-New hyperplane from the numerical rank of
# previous vectors so dependent vectors cannot add artificial dimensions.
def _project_onto_hyperplane(v, basis):
    if basis.shape[0] == 0:
        return np.zeros_like(v), v

    basis_t = np.asarray(basis, dtype=float).T

    U, singular_values, _ = np.linalg.svd(
        basis_t,
        full_matrices=False
    )

    if singular_values.size == 0:
        return np.zeros_like(v), v

    tolerance = (
        max(basis_t.shape)
        * np.finfo(singular_values.dtype).eps
        * singular_values[0]
    )

    rank = np.sum(singular_values > tolerance)

    if rank == 0:
        return np.zeros_like(v), v

    Q = U[:, :rank]

    projection = Q @ (Q.T @ v)
    perpendicular = v - projection

    return projection, perpendicular

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

def _lsa_cohesion_indices(vec_per_paragraph_sentences: List[List[List[List[float]]]], words: List[List[List[str]]], tokens_vector_length: int, n_components, use_truncated_svd: bool = False) -> Dict[str, Any]:
    sentence_vectors = []
    paragraph_vectors = []
    sentences_per_paragraph = []
    for c, para in enumerate(vec_per_paragraph_sentences):
        sentences_per_paragraph.append(len(para))
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
    if use_truncated_svd:
        sentence_vectors_transformed = _reduce_dimensionality(
            sentence_vectors_all,
            n_components
        )
        paragraph_vectors_transformed = _reduce_dimensionality(
            paragraph_vectors,
            n_components
        )
    else:
        sentence_vectors_transformed = sentence_vectors_all
        paragraph_vectors_transformed = paragraph_vectors

    # --- LSA similarity between adjacent sentences ---
    adj_sent_sim = []
    for i in range(len(sentence_vectors_transformed) - 1):
        sim = cosine_similarity([sentence_vectors_transformed[i]], [sentence_vectors_transformed[i + 1]])[0][0]
        adj_sent_sim.append(sim)
    adj_sent_sim = np.array(adj_sent_sim)

    # --- LSA similarity between all sentence pairs in paragraphs ---
    all_sent_pairs_sim = []
    idx = 0
    for count in sentences_per_paragraph:
        if count > 1:
            sent_vecs = sentence_vectors_transformed[idx:idx + count]
            sim_matrix = cosine_similarity(sent_vecs)
            # Take upper triangle excluding diagonal
            triu_indices = np.triu_indices(count, k=1)
            sims = sim_matrix[triu_indices]
            all_sent_pairs_sim.extend(sims)
        idx += count
    all_sent_pairs_sim = np.array(all_sent_pairs_sim)

    # --- LSA similarity between adjacent paragraphs ---
    adj_para_sim = []
    for i in range(len(paragraph_vectors_transformed) - 1):
        sim = cosine_similarity([paragraph_vectors_transformed[i]], [paragraph_vectors_transformed[i + 1]])[0][0]
        adj_para_sim.append(sim)
    adj_para_sim = np.array(adj_para_sim)

    given_new_ratios = _lsa_given_new_for_vectors(sentence_vectors_transformed)

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
def cm_lsass1d(lsa_indices: Dict[str, Any]) -> Optional[float]:
    return lsa_indices["LSASS1d"]

# LAY: Average meaning-similarity among all sentences within each paragraph.
# ↑ Higher = internally coherent paragraphs. Approximate (NOTE(M5)).
def cm_lsassp(lsa_indices: Dict[str, Any]) -> Optional[float]:
    return lsa_indices["LSASSp"]

# LAY: How uneven is the within-paragraph meaning-similarity?
# ↑ Higher = mix of tight and loose paragraphs. Approximate (NOTE(M5)).
def cm_lsasspd(lsa_indices: Dict[str, Any]) -> Optional[float]:
    return lsa_indices["LSASSpd"]

# LAY: Average meaning-similarity between adjacent paragraphs.
# ↑ Higher = smooth flow across paragraphs. Approximate (NOTE(M5)).
def cm_lsapp1(lsa_indices: Dict[str, Any]) -> Optional[float]:
    return lsa_indices["LSAPP1"]

# LAY: How uneven is the meaning-similarity of adjacent paragraphs?
# ↑ Higher = abrupt topic shifts between paragraphs. Approximate (NOTE(M5)).
def cm_lsapp1d(lsa_indices: Dict[str, Any]) -> Optional[float]:
    return lsa_indices["LSAPP1d"]

# LAY: How much of each sentence is already given by previous sentences?
# ↑ Higher = more previously given information and greater semantic cohesion. Approximate (NOTE(M5)).
def cm_lsagn(lsa_indices: Dict[str, Any]) -> Optional[float]:
    return lsa_indices["LSAGN"]

# LAY: How uneven is the given/new ratio across sentences?
# ↑ Higher = greater variation in how much prior information sentences reuse. Approximate (NOTE(M5)).
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
                # FV1 fix: Adjustment for identical Words
                word_freq[word] = word_freq.get(word, 0) + int(freq)
    return word_freq

def _normalize_word_frequencies_per_million(
        word_counts: Dict[str, int]
) -> Dict[str, float]:
    # FV2 fix: the current Wikipedia resource contains absolute token counts,
    # whereas Coh-Metrix word-frequency indices use corpus-normalized lexical
    # frequencies. Restrict the temporary approximation to alphabetic word
    # forms and convert their counts to occurrences per million word tokens.
    lexical_counts = {
        word: count
        for word, count in word_counts.items()
        if word.isalpha() and count > 0
    }

    total_word_count = sum(lexical_counts.values())

    if total_word_count == 0:
        return {}

    factor = 1_000_000 / total_word_count

    return {
        word: count * factor
        for word, count in lexical_counts.items()
    }

all_word_frequencies_map = {
    "en": {
        "wiki-20220301-sample10000": _normalize_word_frequencies_per_million(
            _load_word_frequencies(
                "src/main/resources/word_frequencies_en_enwiki-20220301-sample10000.csv"
            )
        ),
    },
    "de": {
        "wiki-20220301-sample10000": _normalize_word_frequencies_per_million(
            _load_word_frequencies(
                "src/main/resources/word_frequencies_de_dewiki-20220301-sample10000.csv"
            )
        ),
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
def cm_wrdfrqc(
        sentences: List[Sentence],
        lang: str,
        frequencies_source: str
) -> Optional[float]:
    word_frequencies_map = all_word_frequencies_map[lang][frequencies_source]
    content_words, _, _ = _get_content_words(sentences)

    # FV2 fix: Coh-Metrix excludes words that are not covered by the
    # frequency lexicon instead of assigning an artificial frequency of 0.
    word_frequencies = [
        word_frequencies_map[word]
        for word in content_words
        if word in word_frequencies_map
    ]

    return np.mean(word_frequencies) if word_frequencies else 0.0

# LAY: Average log-frequency of ALL words in everyday text.
# ↑ Higher = text leans on common vocabulary. Approximate (NOTE(L9)).
def cm_wrdfrqa(
        tokens: List[Token],
        lang: str,
        frequencies_source: str
) -> Optional[float]:
    word_frequencies_map = all_word_frequencies_map[lang][frequencies_source]

    # FV2 fix: WRDFRQa operates on lexical word tokens. Exclude punctuation
    # and other non-alphabetic tokens and ignore words not covered by the
    # reference frequency resource instead of assigning them frequency 0.
    word_frequencies = [
        word_frequencies_map[token.text.lower()]
        for token in tokens
        if token.is_alpha
        and token.text.lower() in word_frequencies_map
    ]

    # FV2 approximation: use base-10 logarithms for the normalized
    # frequency values to approximate the published Coh-Metrix scale.
    log_word_frequencies = [
        np.log10(freq)
        for freq in word_frequencies
        if freq > 0
    ]

    return np.mean(log_word_frequencies) if log_word_frequencies else 0.0

# LAY: Average MINIMUM log-frequency among content words (rarest word per sentence).
# ↑ Higher = even the rarest content words are reasonably common. Approximate (NOTE(L9)).
def cm_wrdfrqmc(
        sentences: List[Sentence],
        lang: str,
        frequencies_source: str
) -> Optional[float]:
    # CELEX Log minimum frequency for content words, mean
    # -> across sentences
    word_frequencies_map = all_word_frequencies_map[lang][frequencies_source]
    content_words, _, _ = _get_content_words_per_sentence(sentences)

    sentence_min_frequencies = []

    for sentence in content_words:
        # FV2 fix: ignore OOV content words instead of assigning frequency 0.
        # Only frequencies actually covered by the reference resource can
        # contribute to the minimum frequency of a sentence.
        word_frequencies = [
            word_frequencies_map[word]
            for word in sentence
            if word in word_frequencies_map
        ]

        if not word_frequencies:
            continue

        log_word_frequencies = [
            np.log10(freq)
            for freq in word_frequencies
            if freq > 0
        ]

        if not log_word_frequencies:
            continue

        sentence_min_frequencies.append(
            np.min(log_word_frequencies)
        )

    return (
        np.mean(sentence_min_frequencies)
        if sentence_min_frequencies
        else 0.0
    )

# LAY: Second-language readability composite score (Crossley et al. 2008).
# ↑ Higher = easier for L2 learners. Approximate (NOTE(L7) — emitted twice).
def cm_rdl2(crfcwo1: float, synstrut: float, wrdfrqmc: float) -> Optional[float]:
    # RDL2: Coh-Metrix L2 Readability (Appendix A index 106).
    # Formula:
    #
    #   RDL2 = -45.032
    #          + 52.230 * CRFCWO1
    #          + 61.306 * SYNSTRUT
    #          + 22.205 * WRDFRQmc
    #
    # The published formula labels the syntactic predictor only as "SYNSTRUT".
    # Two project variants are therefore retained:
    #
    #   RDL2_synstruta -> uses adjacent-sentence syntax similarity (SYNSTRUTa)
    #   RDL2_synstrutt -> uses across-paragraph syntax similarity (SYNSTRUTt)
    #
    # Available source evidence indicates that the adjacent-sentence measure
    # (SYNSTRUTa) is the closer approximation to the original RDL2 predictor.
    # RDL2_synstrutt is retained as an additional project-specific variant.
    # Both values remain approximations because SYNSTRUT is dependency-based and
    # WRDFRQmc uses the project Wikipedia frequency resource instead of CELEX.
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

# FV1 fix: Changed token counter to exclude punctuation.
def _count_non_punct_tokens(sentences: List[Sentence]) -> int:
    # Use the same non-punctuation word count as DESWC to maintain
    # consistent word counting across incidence-based indices.
    return sum(
        1
        for sent in sentences
        for token in sent.tokens
        if not token.is_punct
    )

def _count_phrase_matches(
    sentence_words: List[str],
    expressions: List[str]
) -> int:

    words = [word.lower() for word in sentence_words]

    patterns = [
        tuple(expression.lower().split())
        for expression in expressions
        if expression and expression.strip()
    ]

    # Longest expressions first.
    patterns.sort(key=len, reverse=True)

    count = 0
    i = 0

    while i < len(words):
        matched_length = 0

        for pattern in patterns:
            length = len(pattern)

            if tuple(words[i:i + length]) == pattern:
                count += 1
                matched_length = length
                break

        if matched_length > 0:
            i += matched_length
        else:
            i += 1

    return count

def count_verbs(
    poses,
    words,
    lemmas,
    causal_practical_set
):
    counters = {
        "causal_verbs": 0,
        "intentional_verbs": 0,
        "causal_particles": 0,
        "intentional_particles": 0,
    }
    # FV3 fix:
    # Count causal/intentional particles as lexical expressions rather
    # than individual tokens. This is required for multi-token expressions
    # such as "in order to" and "aus diesem Grund".
    for i, sent in enumerate(poses):
        counters["causal_particles"] += _count_phrase_matches(
            words[i],
            causal_practical_set["causal_particles"]
        )

        counters["intentional_particles"] += _count_phrase_matches(
            words[i],
            causal_practical_set["intentional_particles"]
        )

        for j, pos in enumerate(sent):
            if pos != "VERB":
                continue

            lemma = lemmas[i][j].lower()

            if lemma in causal_practical_set["causal_verbs"]:
                counters["causal_verbs"] += 1

            if lemma in causal_practical_set["intentional_verbs"]:
                counters["intentional_verbs"] += 1

    return counters

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

@lru_cache(maxsize=2)
def _causal_practical_verbs_intentional(
    lang: str
):
    if lang == "en":
        cause_synset = wn.synset(
            "cause.v.01"
        )

        causal_verbs_en = _get_verb_lemmas_for_synset(
            cause_synset
        )

        intend_synset = wn.synset(
            "intend.v.01"
        )

        plan_synset = wn.synset(
            "plan.v.01"
        )

        intentional_verbs_en = (
            _get_verb_lemmas_for_synset(
                intend_synset
            )
            |
            _get_verb_lemmas_for_synset(
                plan_synset
            )
        )

        return {
            "causal_verbs": causal_verbs_en,
            "intentional_verbs": intentional_verbs_en,
            "causal_particles": [
                "because",
                "therefore",
                "since",
                "so",
                "thus",
                "hence",
                "in order to",
            ],
            "intentional_particles": [
                "in order not to",
                "in order to",
                "so as not to",
                "so as to",
                "in order that",
                "lest",
            ],
        }

    if lang == "de":
        return {
            "causal_verbs": _de_causal_verbs_expanded,
            "intentional_verbs": _de_intentional_verbs_expanded,
            "causal_particles": [
                "weil",
                "deshalb",
                "daher",
                "darum",
                "folglich",
                "infolgedessen",
                "aus diesem Grund",
            ],
            "intentional_particles": [
                "um nicht zu",
                "um zu",
                "auf dass",
                "auf daß",
            ],
        }

    return None

@lru_cache(maxsize=10000)
def _is_german_verb(word: str) -> bool:
    if germanet is None or not word:
        return False
    word = word.strip().lower()
    if not word:
        return False
    try:
        synsets = germanet.get_synsets_by_orthform(word)
        return any((synset.word_category == WordCategory.verben for synset in synsets))
    except Exception:
        return False

# STTS verb tags used by the German spaCy pipeline.

_DE_FINITE_TAGS = {
    "VVFIN",  # lexical finite verb
    "VAFIN",  # finite auxiliary
    "VMFIN",  # finite modal
}

_DE_PARTICIPLE_TAGS = {
    "VVPP",
    "VAPP",
    "VMPP",
}

_DE_VERB_TAGS = (
    _DE_FINITE_TAGS
    | _DE_INFINITIVE_TAGS
    | _DE_PARTICIPLE_TAGS
)

# Dependency relations used to reconstruct German verbal complexes.

_DE_VERBAL_LINK_DEPS = {
    "oc",
    "aux",
    "auxpass",
    "aux:pass",
    "pd",
}

# German auxiliary forms used as lexical fallback.

_DE_HABEN_FORMS = {
    "habe", "hast", "hat", "haben", "habt",
    "hatte", "hattest", "hatten", "hattet",
    "gehabt",
}

_DE_SEIN_FORMS = {
    "bin", "bist", "ist", "sind", "seid",
    "war", "warst", "waren", "wart",
    "sei", "seien",
    "sein",
    "gewesen",
}

_DE_WERDEN_FORMS = {
    "werde", "wirst", "wird", "werden", "werdet",
    "wurde", "wurdest", "wurden", "wurdet",
    "worden", "geworden",
}

_DE_WUERDE_FORMS = {
    "würde",
    "würdest",
    "würden",
    "würdet",
}

_DE_PRESENT_FORMS = {
    "bin", "bist", "ist", "sind", "seid",
    "habe", "hast", "hat", "haben", "habt",
    "werde", "wirst", "wird", "werden", "werdet",
}

_DE_PAST_FORMS = {
    "war", "warst", "waren", "wart",
    "hatte", "hattest", "hatten", "hattet",
    "wurde", "wurdest", "wurden", "wurdet",
}

# Lexical Ersatzinfinitiv triggers; modal infinitives are identified via VMINF.

_DE_ERSATZINFINITIV_LEMMAS = {
    "lassen",
    "sehen",
    "hören",
    "brauchen",
}

# Project-level lexical approximation of semantic completion.

_DE_COMPLETION_LEMMAS = {
    "aufessen",
    "austrinken",
    "aufbrauchen",
    "abschließen",
    "beenden",
    "vollenden",
    "fertigstellen",
    "erledigen",
    "fertigmachen",
}

_DE_COMPLETION_PARTICLE_VERBS = {
    ("auf", "essen"),
    ("aus", "trinken"),
    ("auf", "brauchen"),
    ("ab", "schließen"),
}

_DE_COMPLETION_NEGATION_WORDS = {
    "nicht",
    "nie",
    "niemals",
    "keinesfalls",
}

_DE_COMPLETION_APPROXIMATION_WORDS = {
    "fast",
    "beinahe",
    "halb",
    "teilweise",
    "kaum",
}

_DE_COMPLETION_SCOPE_BLOCKERS = {
    "versuchen",
    "beginnen",
    "anfangen",
    "planen",
    "beabsichtigen",
    "vorhaben",
    "wollen",
}

def _de_norm_dep(token) -> str:
    return (token.dep_type or '').strip().lower()

def _de_norm_text(token) -> str:
    return (token.text or '').strip().lower()

def _de_norm_lemma(token) -> str:
    lemma = (token.lemma or '').strip().lower()
    if lemma:
        return lemma
    return _de_norm_text(token)

def _de_is_root_token(token, token_index: int) -> bool:
    dep = _de_norm_dep(token)
    return dep in {'root', '--'} or token.head_index == token_index

def _de_find_root_index(sentence: Sentence) -> Optional[int]:
    for i, token in enumerate(sentence.tokens):
        if _de_is_root_token(token, i):
            return i
    return None

def _de_is_finite(token) -> bool:
    return token.pos_value in _DE_FINITE_TAGS

def _de_is_infinitive(token) -> bool:
    return token.pos_value in _DE_INFINITIVE_TAGS

def _de_is_participle(token) -> bool:
    return token.pos_value in _DE_PARTICIPLE_TAGS

def _de_is_verbish(token) -> bool:
    if token.pos_value in _DE_VERB_TAGS:
        return True
    return token.pos_coarse in {'VERB', 'AUX'}

def _de_is_haben(token) -> bool:
    lemma = _de_norm_lemma(token)
    text = _de_norm_text(token)
    return lemma == 'haben' or text in _DE_HABEN_FORMS

def _de_is_sein(token) -> bool:
    lemma = _de_norm_lemma(token)
    text = _de_norm_text(token)
    return lemma == 'sein' or text in _DE_SEIN_FORMS

def _de_is_werden(token) -> bool:
    lemma = _de_norm_lemma(token)
    text = _de_norm_text(token)
    return lemma == 'werden' or text in _DE_WERDEN_FORMS

def _de_is_gewesen(token) -> bool:
    return _de_norm_text(token) == 'gewesen' and _de_is_participle(token)

def _de_is_worden(token) -> bool:
    return _de_norm_text(token) == 'worden' and _de_is_participle(token)

def _de_get_verbal_component(sentence: Sentence, seed_index: int) -> Set[int]:
    tokens = sentence.tokens
    if not 0 <= seed_index < len(tokens):
        return set()
    component = {seed_index}
    changed = True
    while changed:
        changed = False
        for i, token in enumerate(tokens):
            if i in component:
                continue
            if not _de_is_verbish(token):
                continue
            head_index = token.head_index
            if head_index is not None and head_index in component and (_de_norm_dep(token) in _DE_VERBAL_LINK_DEPS):
                component.add(i)
                changed = True
                continue
            for component_index in list(component):
                component_token = tokens[component_index]
                if component_token.head_index == i and _de_norm_dep(component_token) in _DE_VERBAL_LINK_DEPS:
                    component.add(i)
                    changed = True
                    break
    return component

def _de_get_main_component(sentence: Sentence) -> Set[int]:
    root_index = _de_find_root_index(sentence)
    if root_index is not None:
        component = _de_get_verbal_component(sentence, root_index)
        if component:
            return component
    return set()

# SYNLE: lexical German verb tags.
# In auxiliary/modal constructions, Coh-Metrix's "main verb" is
# approximated by the lexical verb belonging to the main verbal complex.
_DE_SYNLE_LEXICAL_VERB_TAGS = {
    "VVFIN",
    "VVINF",
    "VVIZU",
    "VVPP",
}


def _de_synle_distance_to_root(
    sentence: Sentence,
    token_index: int,
    root_index: int,
    max_hops: int = 16
) -> Optional[int]:
    if token_index == root_index:
        return 0

    current = token_index
    visited = set()

    for distance in range(1, max_hops + 1):
        if current in visited:
            return None

        visited.add(current)

        head_index = _get_head_index(
            sentence,
            current
        )

        if head_index is None:
            return None

        if head_index == root_index:
            return distance

        current = head_index

    return None


def _de_synle_main_verb_index(
    sentence: Sentence
) -> Optional[int]:
    root_index = _de_find_root_index(sentence)

    if root_index is None:
        return None

    root_token = sentence.tokens[root_index]

    # Simple lexical predicate:
    # "Der Hund läuft."
    if root_token.pos_value in _DE_SYNLE_LEXICAL_VERB_TAGS:
        return root_index

    component = _de_get_main_component(sentence)

    if not component:
        return root_index

    lexical_candidates = [
        token_index
        for token_index in component
        if sentence.tokens[token_index].pos_value
        in _DE_SYNLE_LEXICAL_VERB_TAGS
    ]

    if lexical_candidates:
        ranked_candidates = []

        for token_index in lexical_candidates:
            distance = _de_synle_distance_to_root(
                sentence,
                token_index,
                root_index
            )

            if distance is not None:
                ranked_candidates.append(
                    (distance, token_index)
                )

        if ranked_candidates:
            # Shortest dependency path first.
            # Token index is only a deterministic tie-breaker.
            ranked_candidates.sort(
                key=lambda item: (
                    item[0],
                    item[1]
                )
            )

            return ranked_candidates[0][1]

    # No lexical VV* verb found:
    # retain ROOT, e.g. copular constructions.
    return root_index

def _de_get_finite_tense(token) -> Optional[str]:
    if not _de_is_finite(token):
        return None
    text = _de_norm_text(token)
    if text in _DE_WUERDE_FORMS:
        return None
    morph_tense = (token.morph_tense or '').strip()
    if morph_tense == 'Past':
        return 'PAST'
    if morph_tense == 'Pres':
        return 'PRESENT'
    if text in _DE_PAST_FORMS:
        return 'PAST'
    if text in _DE_PRESENT_FORMS:
        return 'PRESENT'
    return None

def _de_find_finite_anchor(sentence: Sentence, component: Set[int]) -> Optional[int]:
    root_index = _de_find_root_index(sentence)
    if root_index is not None and root_index in component and _de_is_finite(sentence.tokens[root_index]):
        return root_index
    for i in sorted(component):
        if _de_is_finite(sentence.tokens[i]):
            return i
    return None

def _de_is_future_component(sentence: Sentence, component: Set[int], finite_index: int) -> bool:
    finite = sentence.tokens[finite_index]
    if not _de_is_werden(finite):
        return False
    finite_tense = _de_get_finite_tense(finite)
    if finite_tense != 'PRESENT':
        return False
    for i in component:
        if i == finite_index:
            continue
        if _de_is_infinitive(sentence.tokens[i]):
            return True
    return False

def _de_is_ersatzinfinitiv_trigger(token) -> bool:
    if token.pos_value == 'VMINF':
        return True
    return _de_norm_lemma(token) in _DE_ERSATZINFINITIV_LEMMAS

def _de_has_ersatzinfinitiv(sentence: Sentence, component: Set[int]) -> bool:
    infinitives = [sentence.tokens[i] for i in component if _de_is_infinitive(sentence.tokens[i])]
    if len(infinitives) < 2:
        return False
    return any((_de_is_ersatzinfinitiv_trigger(token) for token in infinitives))

def _de_participle_relations(sentence: Sentence, component: Set[int]):
    result = []
    for i in component:
        token = sentence.tokens[i]
        if not _de_is_participle(token):
            continue
        result.append((i, token, _de_norm_dep(token)))
    return result

def _de_detect_perfect_component(sentence: Sentence, component: Set[int], finite_index: int):
    tokens = sentence.tokens
    finite = tokens[finite_index]
    participles = _de_participle_relations(sentence, component)
    has_participle = bool(participles)
    if _de_is_haben(finite):
        if has_participle:
            return (True, 'high', 'haben_perfect')
        if _de_has_ersatzinfinitiv(sentence, component):
            return (True, 'high', 'ersatzinfinitiv_perfect')
        return (False, 'high', 'haben_nonperfect')
    if _de_is_sein(finite):
        if any((_de_is_gewesen(token) for _, token, _ in participles)):
            return (True, 'high', 'sein_gewesen_perfect')
        if any((_de_is_worden(token) for _, token, _ in participles)):
            return (True, 'high', 'passive_perfect_worden')
        if not participles:
            return (False, 'high', 'sein_nonperfect')
        deps = {dep for _, _, dep in participles}
        if 'oc' in deps:
            return (True, 'high', 'sein_perfect_oc')
        if deps and deps <= {'pd'}:
            return (False, 'high', 'stative_passive_pd')
        return (None, 'low', 'sein_participle_ambiguous')
    if _de_is_future_component(sentence, component, finite_index):
        nonfinite_haben = [tokens[i] for i in component if i != finite_index and _de_is_infinitive(tokens[i]) and _de_is_haben(tokens[i])]
        if nonfinite_haben:
            if has_participle or _de_has_ersatzinfinitiv(sentence, component):
                return (True, 'high', 'future_perfect_haben')
        nonfinite_sein = [tokens[i] for i in component if i != finite_index and _de_is_infinitive(tokens[i]) and _de_is_sein(tokens[i])]
        if nonfinite_sein:
            if any((_de_is_worden(token) or _de_is_gewesen(token) for _, token, _ in participles)):
                return (True, 'high', 'future_perfect_sein_explicit')
            deps = {dep for _, _, dep in participles}
            if 'oc' in deps:
                return (True, 'high', 'future_perfect_sein_oc')
            if deps and deps <= {'pd'}:
                return (False, 'high', 'future_stative_passive')
            if participles:
                return (None, 'low', 'future_sein_participle_ambiguous')
        return (False, 'high', 'future_nonperfect')
    return (False, 'high', 'nonperfect_construction')

def _detect_german_grammatical_aspect(sentence: Sentence):
    component = _de_get_main_component(sentence)
    if not component:
        return ('UNKNOWN', 'none', None)
    finite_index = _de_find_finite_anchor(sentence, component)
    if finite_index is None:
        return ('UNKNOWN', 'none', None)
    is_perfect, confidence, evidence = _de_detect_perfect_component(sentence, component, finite_index)
    if is_perfect is True:
        return ('PERFECT', confidence, evidence)
    if is_perfect is False:
        return ('IMPERFECT', confidence, evidence)
    return ('UNKNOWN', confidence, evidence)

def _detect_german_tense(sentence: Sentence):
    component = _de_get_main_component(sentence)
    if not component:
        return ('UNKNOWN', 'none', None)
    finite_index = _de_find_finite_anchor(sentence, component)
    if finite_index is None:
        return ('UNKNOWN', 'none', None)
    finite = sentence.tokens[finite_index]
    if _de_norm_text(finite) in _DE_WUERDE_FORMS:
        return ('UNKNOWN', 'low', 'conditional_wuerde')
    if _de_is_future_component(sentence, component, finite_index):
        return ('FUTURE', 'high', 'future_construction')
    finite_tense = _de_get_finite_tense(finite)
    if finite_tense == 'PAST':
        return ('PAST', 'high', 'finite_past')
    if finite_tense == 'PRESENT':
        return ('PRESENT', 'high', 'finite_present')
    for i in component:
        token = sentence.tokens[i]
        tense = _de_get_finite_tense(token)
        if tense == 'PAST':
            return ('PAST', 'medium', 'finite_past_fallback')
        if tense == 'PRESENT':
            return ('PRESENT', 'medium', 'finite_present_fallback')
    return ('UNKNOWN', 'none', None)

def _de_is_nominalized_infinitive(token) -> bool:
    text = _de_norm_text(token)
    lemma = _de_norm_lemma(token)
    if not text:
        return False
    candidates = {text, lemma}
    for candidate in candidates:
        if not candidate:
            continue
        try:
            if _is_german_verb(candidate):
                return True
        except Exception:
            pass
    return False

def _de_find_progressive_nominal(sentence: Sentence, prep_index: int) -> Optional[int]:
    tokens = sentence.tokens
    for i, token in enumerate(tokens):
        if token.head_index != prep_index:
            continue
        if _de_is_nominalized_infinitive(token):
            return i
    upper = min(len(tokens), prep_index + 4)
    for i in range(prep_index + 1, upper):
        token = tokens[i]
        if token.pos_coarse == 'PUNCT':
            break
        if _de_is_nominalized_infinitive(token):
            return i
    return None

def _detect_german_progressive(sentence: Sentence):
    tokens = sentence.tokens
    has_sein = any((_de_is_sein(token) for token in tokens if _de_is_verbish(token)))
    if not has_sein:
        return (False, 'none', None)
    for i, token in enumerate(tokens):
        text = _de_norm_text(token)
        if text not in {'am', 'beim'}:
            continue
        nominal_index = _de_find_progressive_nominal(sentence, i)
        if nominal_index is None:
            continue
        if text == 'am':
            return (True, 'high', 'am_progressive')
        if text == 'beim':
            return (True, 'medium', 'beim_progressive')
    return (False, 'none', None)

def _de_has_completion_blocker(sentence: Sentence) -> bool:
    tokens = sentence.tokens
    for token in tokens:
        text = _de_norm_text(token)
        lemma = _de_norm_lemma(token)
        dep = _de_norm_dep(token)
        if dep in {'neg', 'ng'} or text in _DE_COMPLETION_NEGATION_WORDS:
            return True
        if text in _DE_COMPLETION_APPROXIMATION_WORDS:
            return True
        if token.pos_value in {'VMFIN', 'VMINF'}:
            return True
        if lemma in _DE_COMPLETION_SCOPE_BLOCKERS:
            return True
        if text in {'wenn', 'falls', 'sofern'}:
            return True
    return False

def _de_has_completion_predicate(sentence: Sentence) -> bool:
    tokens = sentence.tokens
    for token in tokens:
        lemma = _de_norm_lemma(token)
        if lemma in _DE_COMPLETION_LEMMAS:
            return True
    for i, token in enumerate(tokens):
        dep = _de_norm_dep(token)
        if dep != 'svp':
            continue
        particle = _de_norm_text(token)
        head_index = token.head_index
        if head_index is None or not 0 <= head_index < len(tokens):
            continue
        head = tokens[head_index]
        base = _de_norm_lemma(head)
        if (particle, base) in _DE_COMPLETION_PARTICLE_VERBS:
            return True
        combined = particle + base
        if combined in _DE_COMPLETION_LEMMAS:
            return True
    return False

def _detect_german_completion(sentence: Sentence):
    if _de_has_completion_blocker(sentence):
        return (False, 'high', 'completion_blocked')
    if not _de_has_completion_predicate(sentence):
        return (False, 'none', None)
    tense, _, _ = _detect_german_tense(sentence)
    grammatical_aspect, _, _ = _detect_german_grammatical_aspect(sentence)
    if grammatical_aspect == 'PERFECT':
        return (True, 'high', 'bounded_perfect')
    if tense == 'PAST':
        return (True, 'high', 'bounded_past')
    return (False, 'high', 'bounded_but_not_completed')

def _detect_german_aspect(sentence: Sentence):
    progressive, progressive_confidence, progressive_evidence = _detect_german_progressive(sentence)
    if progressive:
        return ('IN_PROGRESS', progressive_confidence, progressive_evidence)
    completed, completion_confidence, completion_evidence = _detect_german_completion(sentence)
    if completed:
        return ('COMPLETED', completion_confidence, completion_evidence)
    return ('UNMARKED', 'none', None)

# Penn Treebank verb tags used by the English spaCy pipeline.

_EN_MODAL_TAG = "MD"

_EN_PROGRESSIVE_TAG = "VBG"

_EN_PARTICIPLE_TAG = "VBN"

_EN_AUX_DEPS = {
    "aux",
    "auxpass",
    "aux:pass",
    "cop",
}

# Ramm/TMV: will/shall are treated as future.

_EN_FUTURE_MODALS = {
    "will",
    "shall",
}

# Ramm/TMV: would/should/could/might are retained as conditional.

_EN_CONDITIONAL_MODALS = {
    "would",
    "should",
    "could",
    "might",
}

def _en_normalize_dep(token) -> str:
    return (token.dep_type or '').strip().lower()

def _en_normalize_lemma(token) -> str:
    return (token.lemma or token.text or '').strip().lower()

def _en_is_root_token(token, token_index: int) -> bool:
    dep = _en_normalize_dep(token)
    return dep in {'root', '--'} or token.head_index == token_index

def _en_find_root_index(sentence: Sentence) -> Optional[int]:
    for i, token in enumerate(sentence.tokens):
        if _en_is_root_token(token, i):
            return i
    return None

def _en_is_auxiliary_relation(token) -> bool:
    return _en_normalize_dep(token) in _EN_AUX_DEPS

def _en_get_main_chain_indices(sentence: Sentence) -> List[int]:
    tokens = sentence.tokens
    root_index = _en_find_root_index(sentence)
    if root_index is None:
        return []
    chain = {root_index}
    changed = True
    while changed:
        changed = False
        for i, token in enumerate(tokens):
            if i in chain:
                continue
            if token.head_index in chain and _en_is_auxiliary_relation(token):
                chain.add(i)
                changed = True
    return sorted(chain)

def _en_chain_tokens(sentence: Sentence):
    return [(i, sentence.tokens[i]) for i in _en_get_main_chain_indices(sentence)]

def _en_classify_modal(token) -> Optional[str]:
    if token.pos_value != _EN_MODAL_TAG:
        return None
    lemma = _en_normalize_lemma(token)
    if lemma in _EN_FUTURE_MODALS:
        return 'FUTURE'
    if lemma in _EN_CONDITIONAL_MODALS:
        return 'CONDITIONAL'
    return 'PRESENT'

def _detect_english_tense(sentence: Sentence):
    tokens = sentence.tokens
    root_index = _en_find_root_index(sentence)
    if root_index is None:
        return ('UNKNOWN', 'none', None)
    chain = _en_chain_tokens(sentence)
    for _, token in chain:
        modal_class = _en_classify_modal(token)
        if modal_class == 'FUTURE':
            return ('FUTURE', 'high', 'future_modal')
        if modal_class == 'CONDITIONAL':
            return ('CONDITIONAL', 'high', 'conditional_modal')
        if modal_class == 'PRESENT':
            return ('PRESENT', 'medium', 'present_modal')
    for i, token in chain:
        if i == root_index:
            continue
        tag = token.pos_value
        if tag == 'VBD':
            return ('PAST', 'high', 'finite_aux_past')
        if tag in {'VBP', 'VBZ'}:
            return ('PRESENT', 'high', 'finite_aux_present')
    root = tokens[root_index]
    if root.pos_value == 'VBD':
        return ('PAST', 'high', 'finite_root_past')
    if root.pos_value in {'VBP', 'VBZ'}:
        return ('PRESENT', 'high', 'finite_root_present')
    return ('UNKNOWN', 'none', None)

def _detect_english_grammatical_aspect(sentence: Sentence):
    tokens = sentence.tokens
    root_index = _en_find_root_index(sentence)
    if root_index is None:
        return ('NONE', 'none', None)
    chain = _en_chain_tokens(sentence)
    root = tokens[root_index]
    has_have_aux = False
    has_be_aux = False
    has_progressive_being = False
    for i, token in chain:
        if i == root_index:
            continue
        if not _en_is_auxiliary_relation(token):
            continue
        lemma = _en_normalize_lemma(token)
        if lemma == 'have':
            has_have_aux = True
        if lemma == 'be':
            has_be_aux = True
            if token.pos_value == 'VBG':
                has_progressive_being = True
    active_progressive = root.pos_value == _EN_PROGRESSIVE_TAG and has_be_aux
    passive_progressive = root.pos_value == _EN_PARTICIPLE_TAG and has_progressive_being
    is_progressive = active_progressive or passive_progressive
    is_perfect = has_have_aux
    if is_perfect and is_progressive:
        return ('BOTH', 'high', 'perfect_progressive')
    if is_progressive:
        return ('PROGRESSIVE', 'high', 'progressive')
    if is_perfect:
        return ('PERFECT', 'high', 'perfect')
    return ('NONE', 'none', None)

def _detect_english_aspect(sentence: Sentence):
    grammatical_aspect, confidence, evidence = _detect_english_grammatical_aspect(sentence)
    if grammatical_aspect == 'PROGRESSIVE':
        return ('IN_PROGRESS', confidence, 'progressive')
    if grammatical_aspect == 'BOTH':
        return ('IN_PROGRESS', confidence, 'perfect_progressive')
    if grammatical_aspect == 'PERFECT':
        return ('COMPLETED', confidence, 'perfect')
    return ('UNMARKED', 'none', None)

# ============================================================================
# SITUATION MODEL (SM*) — causal, intentional, semantic, and temporal cohesion
# approximations.
#
#   • EN causal/intentional verb inventories are derived from selected WordNet
#     synsets and their hyponyms.
#   • DE inventories use GermaNet expansion with documented seed fallbacks.
#   • Causal and intentional particle inventories are project-specific lexical
#     approximations.
#   • SMCAUSlsa uses spaCy semantic vectors rather than the original TASA LSA.
#   • SMCAUSwn uses WordNet for EN and GermaNet for DE.
#   • SMTEMP is a language-specific dependency/morphology-based approximation
#     of tense/aspect repetition across adjacent sentences.
# ============================================================================

# LAY: Causal verbs (cause, break, move, …) per 1,000 words.
# ↑ Higher = more cause-describing verbs. Approximate (NOTE(M11)).
def cm_smcausv(sentences: List[Sentence], lang: str) -> Optional[float]:
    # SMCAUSv: Causal verb incidence (per 1000 words)
    # Spec: Coh-Metrix 3.0 Appendix A, index 59; Chapter 4 §Situation model
    causal_set = _causal_practical_verbs_intentional(lang)
    if causal_set is None:
        return None
    words, _, _, lemmas, poses, _ = _sm_get_data(
        sentences
    )
    counts = count_verbs(
        poses,
        words,
        lemmas,
        causal_set
    )
    # FV1 fix: Use the same non-punctuation word count as DESWC to maintain
    # consistent word counting across incidence-based indices.
    total_words = _count_non_punct_tokens(sentences)
    return _incidence(
        counts["causal_verbs"],
        total_words
    )

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
    words, _, _, lemmas, poses, _ = _sm_get_data(
        sentences
    )
    counts = count_verbs(
        poses,
        words,
        lemmas,
        causal_set
    )
    # FV1 fix: Use the same non-punctuation word count as DESWC to maintain
    # consistent word counting across incidence-based indices.
    total_words = _count_non_punct_tokens(sentences)
    return _incidence(
        counts["causal_verbs"]
        + counts["causal_particles"],
        total_words
    )

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
    words, _, _, lemmas, poses, _ = _sm_get_data(
        sentences
    )
    counts = count_verbs(
        poses,
        words,
        lemmas,
        causal_set
    )
    # FV1 fix: Use the same non-punctuation word count as DESWC to maintain
    # consistent word counting across incidence-based indices.
    total_words = _count_non_punct_tokens(sentences)
    return _incidence(
        counts["intentional_verbs"],
        total_words
    )

# LAY: Ratio of causal particles (connectives) to causal verbs.
# ↑ Higher = causality signalled more by connectives than verbs. Approximate.
def cm_smcausr(sentences: List[Sentence], lang: str) -> Optional[float]:
    causal_set = _causal_practical_verbs_intentional(lang)
    if causal_set is None:
        return None
    words, _, _, lemmas, poses, _ = _sm_get_data(
        sentences
    )
    counts = count_verbs(
        poses,
        words,
        lemmas,
        causal_set
    )
    causal_verbs = counts["causal_verbs"]

    ratio = (
        counts["causal_particles"] / causal_verbs
        if causal_verbs > 0
        else 0
    )

    return np.round(ratio, 3)

# LAY: Ratio of intentional particles to intentional verbs.
# ↑ Higher = intentions signalled more by particles than verbs. Approximate.
def cm_sminter(sentences: List[Sentence], lang: str) -> Optional[float]:
    causal_set = _causal_practical_verbs_intentional(lang)
    if causal_set is None:
        return None
    words, _, _, lemmas, poses, _ = _sm_get_data(
        sentences
    )
    counts = count_verbs(
        poses,
        words,
        lemmas,
        causal_set
    )
    intentional_verbs = counts["intentional_verbs"]

    ratio = (
        counts["intentional_particles"] / intentional_verbs
        if intentional_verbs > 0
        else 0
    )

    return np.round(ratio, 3)

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

# Values excluded from tense/aspect comparisons.
_SMTEMP_UNKNOWN_VALUES = {
    None,
    "",
    "UNKNOWN",
}

def _get_sentence_temporal_state(sentence: Sentence, lang: str):
    lang = (lang or '').strip().lower()
    if lang == 'de':
        tense, _, _ = _detect_german_tense(sentence)
        aspect, _, _ = _detect_german_aspect(sentence)
        return (tense, aspect)
    if lang == 'en':
        tense, _, _ = _detect_english_tense(sentence)
        aspect, _, _ = _detect_english_aspect(sentence)
        return (tense, aspect)
    return ('UNKNOWN', 'UNKNOWN')

def _smtemp_dimension_score(value_a, value_b):
    if value_a in _SMTEMP_UNKNOWN_VALUES or value_b in _SMTEMP_UNKNOWN_VALUES:
        return None
    return 1.0 if value_a == value_b else 0.0

def _smtemp_pair_score(state_a, state_b):
    tense_a, aspect_a = state_a
    tense_b, aspect_b = state_b
    dimension_scores = []
    tense_score = _smtemp_dimension_score(tense_a, tense_b)
    if tense_score is not None:
        dimension_scores.append(tense_score)
    aspect_score = _smtemp_dimension_score(aspect_a, aspect_b)
    if aspect_score is not None:
        dimension_scores.append(aspect_score)
    if not dimension_scores:
        return None
    return sum(dimension_scores) / len(dimension_scores)

# LAY: Temporal cohesion based on tense/aspect repetition across adjacent sentences.

def cm_smtemp(sentences: List[Sentence], lang: str) -> Optional[float]:
    if not sentences or len(sentences) < 2:
        return 0.0
    lang = (lang or '').strip().lower()
    states = [_get_sentence_temporal_state(sentence, lang) for sentence in sentences]
    pair_scores = []
    for state_a, state_b in zip(states, states[1:]):
        score = _smtemp_pair_score(state_a, state_b)
        if score is not None:
            pair_scores.append(score)
    if not pair_scores:
        return 0.0
    smtemp = sum(pair_scores) / len(pair_scores)
    return round(smtemp, 3)

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

        # FV1 fix: Exclude punctuation for overall consistancy
        tokens_count = sum(1 for t in tokens if not t.is_punct)

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
            token_vectors, token_words = _get_paragraph_token_vectors(request.paragraphs)
            lsa_indices = _lsa_cohesion_indices(
                token_vectors,
                token_words,
                tokens_vector_length,
                n_components=settings.lsa_svd_components,
                use_truncated_svd=settings.lsa_use_truncated_svd
            )
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
            synle = cm_synle(sentences, request.language)
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
        finally:
            _clear_synstrut_caches()
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

        # FV1 fix:  load mrc once and reuse across all 5 WRD* indices.
        try:
            _mrc_dict = _load_mrc_database(request.language)
        except Exception as e:
            logger.error("Error precomputing MRC dict: %s", e)
            _mrc_dict = None

        # WRDAOAc
        # FV1 fix: Include mrc_dict in function call if loaded successfully
        try:
            wrdaoac = cm_wrdaoac(sentences, request.language, mrc_dict=_mrc_dict)
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
        # FV1 fix: Include mrc_dict in function call if loaded successfully
        try:
            wrdfamc = cm_wrdfamc(sentences, request.language, mrc_dict=_mrc_dict)
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
        # FV1 fix: Include mrc_dict in function call if loaded successfully
        try:
            wrdcncc = cm_wrdcncc(sentences, request.language, mrc_dict=_mrc_dict)
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
        # FV1 fix: Include mrc_dict in function call if loaded successfully
        try:
            wrdimgc = cm_wrdimgc(sentences, request.language, mrc_dict=_mrc_dict)
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
        # FV1 fix: Include mrc_dict in function call if loaded successfully
        try:
            wrdmeac = cm_wrdmeac(sentences, request.language, mrc_dict=_mrc_dict)
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

        # Precompute hypernymy values once for WRDHYPn, WRDHYPv and WRDHYPnv.
        try:
            _wrdhyp = _calc_wrdhyp(
                sentences,
                request.language
            )
            _wrdhyp_error = None
        except Exception as e:
            logger.error(
                "Error calculating WRDHYP indices: %s",
                e
            )

            _wrdhyp = {
                "WRDHYPn": None,
                "WRDHYPv": None,
                "WRDHYPnv": None,
            }

            _wrdhyp_error = str(e)

        # WRDHYPn
        wrdhypn = _wrdhyp["WRDHYPn"]
        wrdhypn_error = _wrdhyp_error
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
        wrdhypv = _wrdhyp["WRDHYPv"]
        wrdhypv_error = _wrdhyp_error
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
        wrdhypnv = _wrdhyp["WRDHYPnv"]
        wrdhypnv_error = _wrdhyp_error
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

        # NOTE(L4): RDFRE and RDFKGL are currently calculated through textstat on the
        # raw document text. Coh-Metrix 3.0 defines these indices from its own DESSL
        # and DESWLsy measures. The present implementation is therefore retained as a
        # textstat-based approximation and is not claimed to be numerically identical
        # to the original Coh-Metrix computation. The V3 labels are retained
        # provisionally for compatibility and may be revised in a later pass.

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
    return response
