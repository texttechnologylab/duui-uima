"""The computation of the Coh-Metrix indices (Coh-Metrix 0.4.0-py core, PR #305): German and English read in Universal
Dependencies. `compute(request, wanted)` gives {key: (value, error)} for the wanted index keys and runs only the steps they
need (KEY_BLOCK, BLOCK_DEPS). Labels, variants, the Text Easability components and the service are in duui.py. Resources are
read at import from the paths in `settings` (environment variables DUUI_COH_METRIX_*_PATH of the container)."""
import bisect
import csv
import json
import re
import logging
import threading
import textstat
import math

from contextvars import ContextVar
from typing import List, Optional, Dict, Tuple, Set, Any

from pydantic import BaseModel
import os
try:
    from . import ud_mapping
except ImportError:  # imported outside the package (tests)
    import ud_mapping
from lexicalrichness import LexicalRichness
# lexical_diversity (0.1.1) finds its lemma list with pkg_resources.resource_filename, which setuptools >= 81 no longer ships; a stand-in
# for that one function (resurrection, 2026-10-22), so that nothing in the environment has to be downgraded
import sys as _sys, types as _types
if "pkg_resources" not in _sys.modules:
    try:
        import pkg_resources  # noqa: F401
    except ImportError:
        import importlib.resources as _ir
        _pr = _types.ModuleType("pkg_resources")
        _pr.resource_filename = lambda package, name: str(_ir.files(package) / name)
        _sys.modules["pkg_resources"] = _pr
from lexical_diversity import lex_div as ld
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.decomposition import TruncatedSVD
from germanetpy.germanet import Germanet
from germanetpy.synset import WordCategory
from pathlib import Path
from functools import lru_cache

import numpy as np

RESOURCES = Path(__file__).resolve().parent.parent / "resources"
import nltk as _nltk
_nltk.data.path.insert(0, str(RESOURCES / "nltk_data"))

# NLTK's WordNet reader is not thread-safe: it loads lazily, and synset lookups seek and read on shared file handles, so
# concurrent requests read each other's lines. The reader is loaded completely here, and its methods that read the files during
# a request run under one lock (everything else, and every index that does not use WordNet, stays concurrent).
try:
    wn.ensure_loaded()
except LookupError:   # without the NLTK data (outside the image): the WordNet indices fail on their own
    pass
else:
    _WORDNET_LOCK = threading.RLock()
    for _name in ("synset_from_pos_and_offset", "lemma_count", "lemma_from_key", "index_sense", "all_eng_synsets",
                  "_data_file", "_load_lang_data"):
        def _locked(*args, _method=getattr(wn, _name), **kwargs):
            with _WORDNET_LOCK:
                return _method(*args, **kwargs)
        setattr(wn, _name, _locked)

class Settings:
    """the container's settings (environment variables DUUI_COH_METRIX_*), with defaults for running without it"""
    def __init__(self):
        env = lambda k, d=None: os.environ.get("DUUI_COH_METRIX_" + k.upper(), d)
        self.germanet_path = env("germanet_path", "/data/GermaNet/GN_V180_XML")
        self.dimlex_path = env("dimlex_path", "/data/CORE/export/t0/dimlex/DimLex.xml")
        # English connectives (2026-10-29): DiMLex-Eng and the connective weights estimated on GUM
        self.dimlex_en_path = env("dimlex_en_path", "/data/CORE/export/t0/cm_resources/en_dimlex/en_dimlex.xml")
        # the weights are ours (counts from GUM's CC BY 4.0 annotations) and ship in resources/ since 2026-11-02
        self.dimlex_en_weights_path = env("dimlex_en_weights_path", str(RESOURCES / "gum_weights_en.csv"))
        # German word norms: read from a path, never copied into the component (Köper: CC BY-NC-SA 4.0;
        # Birchenough: no licence stated); a missing file leaves its indices None
        self.koeper_path = env("koeper_path", "/data/CORE/export/t0/cm_resources/koeper_affective_norms/affective_norms.txt.gz")
        self.birchenough_path = env("birchenough_path",
                                    "/data/CORE/export/t0/cm_resources/birchenough_aoa/13428_2016_718_MOESM1_ESM.csv")
        # German word frequencies: SUBTLEX-DE per million, read from a path like the norms
        self.subtlex_de_path = env("subtlex_de_path", "/data/CORE/export/t0/cm_resources/subtlex_de/SUBTLEXde_SUBTLWF.csv")
        # English word frequencies (2026-10-29): SUBTLEX-US per million, read from a path like SUBTLEX-DE
        # English word norms (2026-10-29): read from paths like the German ones (no licence stated)
        self.brysbaert_path = env("brysbaert_path",
                                  "/data/CORE/export/t0/cm_resources/brysbaert_concreteness/brysbaert_concreteness.csv")
        self.kuperman_path = env("kuperman_path", "/data/CORE/export/t0/cm_resources/kuperman_aoa/kuperman_aoa.csv")
        self.subtlex_us_path = env("subtlex_us_path",
                                   "/data/CORE/export/t0/cm_resources/subtlex_us/unzipped/SUBTLEXus74286wordstextversion.txt")
        # the German Text Easability model (2026-10-29): ours, fitted on reference corpus v1 (easability/), shipped in resources/;
        # since 0.3.0 (2026-10-31) refitted on the corpus as annotated by the review code, new id ("new"); since 0.4.0
        # (2026-11-01) refitted on the corpus as 0.4.0 annotates it, refv1-ud-rr-cr2 ("2. refa-en-rr-gn-cr2")
        self.easability_model_path = env("easability_model_path",
                                         str(RESOURCES / "easability" / "text_easability_de_refv1-ud-rr-cr2.json"))
        # the English one (2026-10-30): fitted on reference corpus A, components named by congruence with the German model; since
        # 0.3.0 refitted likewise and named after the refitted German model; since 0.4.0 refa-en-rr-gn-cr2. The models of 0.2.0
        # and 0.3.0 stay in resources/ (refv1-ud-rr, refa-en-rr-gn; refv1-ud-rr-cr, refa-en-rr-gn-cr) for old outputs: set the
        # paths to them.
        self.easability_model_path_en = env("easability_model_path_en",
                                            str(RESOURCES / "easability" / "text_easability_en_refa-en-rr-gn-cr2.json"))
        self.lsa_use_truncated_svd = str(env("lsa_use_truncated_svd", "false")).lower() == "true"
        self.lsa_svd_components = int(env("lsa_svd_components", "100"))


settings = Settings()

logger = logging.getLogger(__name__)

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



SUPPORTED_LANGS = ["de", "en"]   # was [] ("all"): the syllables, the norms and every rule since the UD ports are German or English
# Language codes the core accepts: before, only case and spaces were normalised, so "de-DE" or "deu"
# ran with the German label check, UD handling and rules switched off, and nothing said so.
_LANGUAGE_CODES = {"de": "de", "deu": "de", "ger": "de", "german": "de", "en": "en", "eng": "en", "english": "en"}


def _language(request) -> str:
    """the request's language as "de" or "en" ("de-DE", "deu", "EN" ...); other languages are refused"""
    code = (request.language or "").strip().lower().replace("_", "-").split("-")[0]
    if code not in _LANGUAGE_CODES:
        raise ValueError(f"language {request.language!r} is not supported: the core computes German and English (de, en)")
    return _LANGUAGE_CODES[code]

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
    morph_verbform: Optional[str] = ""   # UD VerbForm (2026-10-29: gerunds, VerbForm=Ger)
    morph_number_psor: Optional[str] = ""   # UD Number[psor], the possessor's number (2026-10-31: German possessives)
    morph_mood: Optional[str] = ""       # UD Mood (2026-10-31: German Konjunktiv II in SMTEMP)
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
    vectors: Optional[str] = None   # the spaCy model whose word vectors the tokens carry (2026-10-29; names the LSA indices' vectors)












# Annotation sufficiency helpers. A finite zero is only meaningful if the
# annotation layer required by the metric is actually present. These helpers
# deliberately operate per feature family; they are not a global request gate.
def _non_punct_tokens(sentence: Sentence) -> List[Token]:
    return [token for token in sentence.tokens if not token.is_punct]


def _has_complete_pos_annotations(sentence: Sentence) -> bool:
    tokens = _non_punct_tokens(sentence)
    return all(
        bool((token.pos_coarse or "").strip())
        and (token.pos_coarse or "").strip() != "--"
        for token in tokens
    )


def _has_complete_fine_pos_annotations(sentence: Sentence) -> bool:
    return _has_complete_pos_annotations(sentence) and all(
        bool((token.pos_value or "").strip())
        and (token.pos_value or "").strip() != "--"
        for token in _non_punct_tokens(sentence)
    )


def _has_complete_dependency_annotations(sentence: Sentence) -> bool:
    tokens = _non_punct_tokens(sentence)
    return all(
        bool((token.dep_type or "").strip())
        and token.head_index is not None
        for token in tokens
    )


def _has_usable_lemma(token: Token) -> bool:
    return bool((token.lemma or "").strip()) and token.lemma.strip() != "--"


def _has_analyzable_verbal_predicate(sentence: Sentence) -> bool:
    """True if tense/aspect can be grounded in an observed verbal token."""
    return any(
        token.pos_coarse in {"VERB", "AUX"}
        for token in _non_punct_tokens(sentence)
    )






if settings.germanet_path:
    gnp = Path(settings.germanet_path)
    if gnp.is_dir() and any(gnp.iterdir()):
        logger.info("Loading GermaNet from \"%s\"", settings.germanet_path)
        try:
            germanet = Germanet(settings.germanet_path)
        except Exception as e:      # a folder without GermaNet (user, 2026-10-25): the GermaNet-based indices are None (NaN), the run goes on
            logger.warning("GermaNet could not be loaded from \"%s\" (%s: %s). Metrics based on GermaNet will return None",
                           settings.germanet_path, type(e).__name__, str(e)[:200])
            germanet = None
    else:
        logger.warning("GermaNet path defined as \"%s\", but empty or non-existing. Metrics based on GermaNet will return None", settings.germanet_path)
        germanet = None
else:
    logger.warning("No GermaNet path defined. Metrics based on GermaNet will return None")
    germanet = None

# Fix B3 (2026-10-28, README): fallbacks for German content lemmas GermaNet does not list, used by
# WRDPOLc and WRDHYPn/v/nv, which skipped them (5.3 % of the noun tokens of the fix sample; the fallbacks recover 73.8 % of those):
# - old spelling -> current spelling from GermaNet's own old_orthform / old_orthvar ("Fluß" -> "Fluss", "Prozeß" -> "Prozess");
# - a noun -> its compound head (_de_compound_head: "Tankrabatt" -> "Rabatt"). Against GermaNet's compound annotation (5,000
#   annotated noun compounds, each treated as unlisted): the annotated head for 94.1 %. Such a compound counts as one sense in
#   WRDPOLc (97.5 % of GermaNet's compounds have one, mean 1.03; their heads 2.65) and with its head's depth in WRDHYP*. Depth
#   against GermaNet: head + 1 exact for 43.5 % but 1.2 levels too deep on average, head + 0 exact for 12.9 %, 0.25 too deep on
#   average (mean |difference| 1.64 / 1.70). On the fix sample head + 1 lowered WRDHYPn / WRDHYPnv against CLEAR (.272 -> .260,
#   .166 -> .148), head + 0 kept them (.269, .160) while WRDPOLc gained (.320 -> .335); head + 0 was tested and
#   decide on CLEAR (2026-10-28): head + 0.
_GN_NOUN_FORMS: Set[str] = set()
_GN_OLD_SPELLINGS: Dict[str, str] = {}
if germanet is not None:
    for _lu in germanet.lexunits.values():
        if _lu.synset.word_category == WordCategory.nomen:
            _GN_NOUN_FORMS.add(_lu.orthform)
        for _old in (_lu.old_orthform, _lu.old_orthvar):
            if _old and _old != _lu.orthform:
                _GN_OLD_SPELLINGS.setdefault(_old, _lu.orthform)


def _de_compound_head(word: str) -> Optional[str]:
    """the head of a noun compound GermaNet does not list: after the last hyphen, the longest proper suffix of at least 3
    letters that, capitalised, is a GermaNet noun; none for a nominalised verb or participle"""
    part = word.rsplit("-", 1)[-1]
    if part != word and part[:1].isupper() and part in _GN_NOUN_FORMS:
        return part
    # no head for a nominalised verb (its lower case is a GermaNet verb: "Joggen" gave "Gen", "Heizen"
    # "Zen") or participle (the stem + "en" is one: "Demonstrierende" gave "Ende", "Mitarbeitender" "Tender"). Gold as above
    # (5,000 annotated compounds): the annotated head stays at .941; GermaNet nouns without a compound
    # annotation, treated as unlisted, get a head .519 -> .513. A minimum of 4 letters for the head ("Natürlich" gives "Ich",
    # "Raus" "Aus") was tried and dropped: the annotated head fell to .917 (Hohlweg -> Weg, Fernbus -> Bus, Zwischenruf -> Ruf)
    low = part.lower()
    participle = re.match(r"(.+?)ende[rns]?$", low)
    if _is_german_verb(low) or (participle and _is_german_verb(participle.group(1) + "en")):
        return None
    for i in range(1, len(part) - 2):
        cand = part[i].upper() + part[i + 1:]
        if cand in _GN_NOUN_FORMS:
            return cand
    return None


def _de_germanet_fallback(lemma: str, pos: str) -> Optional[Tuple[str, str]]:
    """(form to look up, "old spelling" | "compound") for a lemma GermaNet does not list, or None"""
    if lemma in _GN_OLD_SPELLINGS:
        return _GN_OLD_SPELLINGS[lemma], "old spelling"
    if pos == "NOUN" and lemma[:1].isupper():
        head = _de_compound_head(lemma)
        if head:
            return head, "compound"
    return None

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
    valid_paragraphs = [
        paragraph
        for paragraph in paragraphs
        if (paragraph.text or "").strip()
        or any(sentence.tokens for sentence in paragraph.sentences)
    ]
    if not valid_paragraphs:
        return None
    return len(valid_paragraphs)

# LAY: How many sentences are in the text?
# Reliable.
def cm_dessc(paragraphs: List[Paragraph]) -> Optional[float]:
    # Sentence count, number of sentences
    sentence_count = sum(
        1
        for paragraph in paragraphs
        for sentence in paragraph.sentences
        if (sentence.text or "").strip() or sentence.tokens
    )
    return sentence_count if sentence_count > 0 else None

# LAY: How many words (excluding punctuation) are in the text?
# Reliable.
def cm_deswc(paragraphs: List[Paragraph]) -> Optional[float]:
    # Word count, number of words.
    # M6 fix: exclude punctuation to match _count_metrics["total_tokens"] and
    # spec ("words taken from leaves of the sentence parse trees").
    word_count = sum(
        1
        for paragraph in paragraphs
        for sentence in paragraph.sentences
        for token in sentence.tokens
        if not token.is_punct
    )
    return word_count if word_count > 0 else None

# LAY: On average, how many sentences are in one paragraph?
# ↑ Higher = longer paragraphs. Reliable.
def cm_despl(paragraphs: List[Paragraph]) -> Optional[float]:
    # Paragraph length, number of sentences, mean
    sentence_counts = [
        sum(1 for sentence in paragraph.sentences
            if (sentence.text or "").strip() or sentence.tokens)
        for paragraph in paragraphs
        if (paragraph.text or "").strip()
        or any(sentence.tokens for sentence in paragraph.sentences)
    ]
    return (
        np.mean(sentence_counts)
        if sentence_counts and sum(sentence_counts) > 0
        else np.nan
    )

# LAY: How uneven are paragraph lengths? (spread around the mean)
# ↑ Higher = more variable paragraph sizes. Reliable.
def cm_despld(paragraphs: List[Paragraph]) -> Optional[float]:
    # Paragraph length, number of sentences, standard deviation
    sentence_counts = [
        sum(1 for sentence in paragraph.sentences
            if (sentence.text or "").strip() or sentence.tokens)
        for paragraph in paragraphs
        if (paragraph.text or "").strip()
        or any(sentence.tokens for sentence in paragraph.sentences)
    ]
    return (
        np.std(sentence_counts)
        if sentence_counts and sum(sentence_counts) > 0
        else np.nan
    )

# LAY: On average, how many words are in one sentence?
# ↑ Higher = longer sentences (often harder to read). Reliable.
def cm_dessl(paragraphs: List[Paragraph]) -> Optional[float]:
    # Sentence length, number of words, mean. M6 fix: exclude punctuation.
    sentence_lengths = [
        sum(1 for token in sentence.tokens if not token.is_punct)
        for paragraph in paragraphs
        for sentence in paragraph.sentences
        if (sentence.text or "").strip() or sentence.tokens
    ]
    return (
        np.mean(sentence_lengths)
        if sentence_lengths and sum(sentence_lengths) > 0
        else np.nan
    )

# LAY: How uneven are sentence lengths?
# ↑ Higher = more variable sentence sizes. Reliable.
def cm_dessld(paragraphs: List[Paragraph]) -> Optional[float]:
    # Sentence length, number of words, standard deviation. M6 fix: exclude punctuation.
    sentence_lengths = [
        sum(1 for token in sentence.tokens if not token.is_punct)
        for paragraph in paragraphs
        for sentence in paragraph.sentences
        if (sentence.text or "").strip() or sentence.tokens
    ]
    return (
        np.std(sentence_lengths)
        if sentence_lengths and sum(sentence_lengths) > 0
        else np.nan
    )


# German syllables: hyphenation points + 1 (pyphen) undercount German syllables, since hyphenation never splits
# off a single vowel (Abend 1, Idee 1, Energie 2). Now: the vowel nuclei of the spelling, with ä, ö, ü and y as vowels and the digraphs
# below as one nucleus ("io" too: the glide of Region, Radio, Millionen); an all-capital word of 2-5 letters is read letter by letter
# (USA, SPD: one syllable per letter, Ypsilon three). Gold: the German Wiktionary's IPA (syllable nuclei), on the words of the
# fix sample: exact for 98.1 % of the tokens (pyphen 93.6 %), mean error -0.01 syllables per token (pyphen -0.03);
#. English keeps pyphen.
_DE_SYLLABLE_DIGRAPHS = ("äu", "ai", "au", "ay", "ei", "ey", "eu", "ie", "aa", "ee", "oo", "io")
_DE_VOWELS = set("aeiouäöüy")


def _de_syllables(word: str) -> int:
    # a hyphenated word counts its parts ("US-Präsident" 5, was 4); a word of 2-5 letters with at least
    # two capitals is read letter by letter like an all-capital one ("AfD" 3, was 1; "GmbH" 4); "u" after "q" is no nucleus
    # ("Quelle" 2, was 3). Against the Wiktionary gold: exact .967 -> .974 of types, .981 -> .983 of tokens. A rule giving
    # "-ien" two nuclei was tried and dropped: the gold reads the i as a glide (Medien, Spanien, Familien) and it hit "Parteien"
    parts = [p for p in word.split("-") if p]
    if len(parts) > 1:
        return sum(_de_syllables(p) for p in parts)
    if word.isalpha() and 2 <= len(word) <= 5 and sum(c.isupper() for c in word) >= 2:
        return sum(3 if c in "YÝ" else 1 for c in word)
    w, n, i = word.lower(), 0, 0
    while i < len(w):
        if w[i] == "u" and i > 0 and w[i - 1] == "q":
            i += 1
        elif w[i] in _DE_VOWELS:
            n += 1
            i += 2 if w[i:i + 2] in _DE_SYLLABLE_DIGRAPHS else 1
        else:
            i += 1
    return max(1, n)


# English syllables (2026-10-29): pyphen undercounts English as it did German. English spelling is too
# irregular for a rule alone, so: the CMU Pronouncing Dictionary (resources/nltk_data; its first pronunciation, one syllable per
# stressed or unstressed vowel), and a spelling rule for words it lacks (_en_rule_syllables). Clitics split off by the tokenizer
# (n't, 's, 'll, 're, 've, 'd, 'm: UD words of their own) count the syllables they add to their host: CMU(host + clitic) -
# CMU(host) where both are listed ("did" + "n't" adds 1, "do" + "n't" adds 0), else n't adds 1 after a consonant and 's after a
# sibilant, the others 0. Gold:.
_EN_CLITICS = {"n't", "'s", "'ll", "'re", "'ve", "'d", "'m"}
_EN_CMU: Optional[Dict[str, int]] = None


def _en_cmu() -> Dict[str, int]:
    global _EN_CMU
    if _EN_CMU is None:
        from nltk.corpus import cmudict
        _EN_CMU = {w: sum(ph[-1].isdigit() for ph in prons[0]) for w, prons in cmudict.dict().items()}
    return _EN_CMU


def _en_rule_syllables(word: str) -> int:
    """vowel groups of the spelling, adjusted for silent final -e, -ed and -es; all-capital words of 2-5 letters letter by letter"""
    if word.isalpha() and word.isupper() and 2 <= len(word) <= 5:
        return sum(3 if c == "W" else 1 for c in word)
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 1
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and n > 1:
        n -= 1                                               # silent final e (make, fire), not table, free
    elif w.endswith("ed") and not w.endswith(("ted", "ded")) and n > 1:
        n -= 1                                               # walked, not wanted
    elif w.endswith("es") and not re.search(r"(s|x|z|ch|sh|ge|ce)es$", w) and n > 1:
        n -= 1                                               # makes, not boxes, judges
    return max(1, n)


def _en_syllables(word: str, host: Optional[str]) -> int:
    low = word.lower().replace("’", "'")
    cmu = _en_cmu()
    if low in _EN_CLITICS and host:
        h = host.lower().replace("’", "'")
        if h + low in cmu and h in cmu:
            return max(0, cmu[h + low] - cmu[h])
        if low == "n't":
            return 0 if h[-1:] in "aeiouy" else 1
        if low == "'s":
            return 1 if re.search(r"(s|x|z|ch|sh|ge|ce)$", h) else 0
        return 0
    if low in cmu:
        return cmu[low]
    return _en_rule_syllables(word)


def _contraction_continuation(tokens: List[Token], k: int) -> bool:
    """True for the second (third ...) word of a split contraction: German "im" = "in" + "dem" since the UD port, the words share
    the token's span (preprocess.StanzaLocal)"""
    return k > 0 and tokens[k].begin == tokens[k - 1].begin and tokens[k].end == tokens[k - 1].end


def _surface_words(paragraphs: List[Paragraph]) -> List[Tuple[Token, str, bool]]:
    """(token, surface form, continuation) of every non-punctuation word in text order. The words of a split contraction share
    one surface form: the first carries it, the others are continuations."""
    out = []
    for p in paragraphs:
        for s in p.sentences:
            words = [t for t in s.tokens if not t.is_punct]
            for k, t in enumerate(words):
                cont = _contraction_continuation(words, k)
                surface = t.text
                if not cont and k + 1 < len(words) and _contraction_continuation(words, k + 1):
                    surface = s.text[t.begin - s.begin:t.end - s.begin] or t.text
                out.append((t, surface, cont))
    return out


def _syllables_count(tokens: List[Token], lang: str) -> List[int]:
    return _memo("syllables", (tokens,), lang, lambda: _syllables_count_uncached(tokens, lang))


def _syllables_count_uncached(tokens: List[Token], lang: str) -> List[int]:
    if lang == "de":
        # a split contraction keeps its surface's syllables once, on its first word ("in" for "im"; every German contraction has
        # the syllables of its first word); the further words count 0, as English clitics count only what they add (code
        # review 2026-10-31; before, "im" counted 2 syllables as "in" + "dem")
        words = [token for token in tokens if not token.is_punct]
        return [0 if _contraction_continuation(words, k) else _de_syllables(t.text) for k, t in enumerate(words)]
    if lang == "en":
        counts, host = [], None
        for token in tokens:
            if token.is_punct:
                continue
            counts.append(_en_syllables(token.text, host))
            host = token.text
        return counts
    raise ValueError(f"no syllable counting for language {lang!r} (German and English only)")

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
    # a split contraction counts its surface's letters once (before "im" had 5, as "in" + "dem")
    text_letters = [0 if cont else sum(c.isalpha() for c in surface) for _, surface, cont in _surface_words(paragraphs)]
    return np.mean(text_letters)

# LAY: How uneven is the letter count across words?
# ↑ Higher = mix of short and long words. Reliable.
def cm_deswlltd(paragraphs: List[Paragraph]) -> Optional[float]:
    # Word length, number of letters, standard deviation
    # FV1 fix: exclude punctuation to keep consistency
    text_letters = [0 if cont else sum(c.isalpha() for c in surface) for _, surface, cont in _surface_words(paragraphs)]
    return np.std(text_letters)

# Readability formulas: computed from the request's own units instead of textstat on the raw text, which split
# sentences by regex (dropping those of two words or fewer), took words from whitespace and counted syllables with pyphen. Now: the
# sentences of DESSC, the words of DESWC (non-punctuation tokens), the syllables of DESWLsy (_syllables_count), the letters of DESWLlt;
# so RDFRE and RDFKGL follow from DESSL and DESWLsy as Coh-Metrix 3.0 defines them. Formula constants, thresholds and easy-word lists
# as in textstat 0.7.12 (backend/metrics; for German its resources/de list of 2,579 words): FRE with Amstad's German constants;
# polysyllabic (SMOG, Wiener Sachtextformel) = 3 or more syllables; difficult = not an easy word and 3 or more syllables (Fog), 2 or
# more (Spache), any (Dale-Chall); Linsear Write on the first 100 words and the sentences they reach. ARI counts letters and digits per
# word, Coleman-Liau and the Wiener long words (more than 6) letters only.
_FRE_CONSTANTS = {"de": (180.0, 1.0, 58.5), "en": (206.835, 1.015, 84.6)}
_EASY_WORDS: Dict[str, Set[str]] = {}


def _easy_words(lang: str) -> Set[str]:
    if lang not in _EASY_WORDS:
        with open(Path(textstat.__file__).parent / "resources" / lang / "easy_words.txt", encoding="utf-8") as fh:
            _EASY_WORDS[lang] = {line.strip() for line in fh}
    return _EASY_WORDS[lang]


def _readability_scores(paragraphs: List[Paragraph], lang: str) -> Dict[str, float]:
    sentences = [[t for t in s.tokens if not t.is_punct] for p in paragraphs for s in p.sentences
                 if (s.text or "").strip() or s.tokens]
    words = [t for s in sentences for t in s]
    syllables = _syllables_count(words, lang)
    # a split German contraction counts its surface's letters once and is looked up as its surface
    # ("zum", not "zu" + "dem"); an English clitic ("n't", "'s") is not a word of its own for the easy-word lists, its host is
    # looked up with it ("don't") where the list has the pair
    surface = _surface_words(paragraphs)
    letters = [0 if cont else sum(c.isalpha() for c in sf) for _, sf, cont in surface]
    alnum = [0 if cont else sum(c.isalnum() for c in sf) for _, sf, cont in surface]
    easy = _easy_words(lang)
    clitic = lambda t: (t.text or "").lower().replace("’", "'") in _EN_CLITICS
    hard = []
    for k, (t, sf, cont) in enumerate(surface):
        prev, nxt = surface[k - 1][0] if k else None, surface[k + 1][0] if k + 1 < len(surface) else None
        if cont or (lang == "en" and clitic(t) and prev is not None and prev.end == t.begin):
            hard.append(False)
        elif (lang == "en" and nxt is not None and nxt.begin == t.end and clitic(nxt)
              and (sf + nxt.text).lower().replace("’", "'") in easy):
            hard.append(False)
        else:
            hard.append(sf.lower() not in easy)
    n_words, n_sent = len(words), len(sentences)
    pct = lambda n: 100 * n / n_words
    asl, asw = n_words / n_sent, sum(syllables) / n_words
    ms, es = pct(sum(n >= 3 for n in syllables)), pct(sum(n == 1 for n in syllables))
    iw, pdw = pct(sum(n > 6 for n in letters)), pct(sum(hard))
    # Linsear Write: the first 100 words and the sentences they reach
    lw_sent, before = 0, 0
    for s in sentences:
        lw_sent += 1 if s and before < 100 else 0
        before += len(s)
    lw = sum(3 if n >= 3 else 1 for n in syllables[:100]) / lw_sent
    base, sl_weight, sy_weight = _FRE_CONSTANTS[lang]
    return {
        "FRE": base - sl_weight * asl - sy_weight * asw,
        "FKGL": 0.39 * asl + 11.8 * asw - 15.59,
        "FOG": 0.4 * (asl + pct(sum(h and n >= 3 for h, n in zip(hard, syllables)))),
        "SMOG": 1.043 * math.sqrt(30 * sum(n >= 3 for n in syllables) / n_sent) + 3.1291,
        "ARI": 4.71 * sum(alnum) / n_words + 0.5 * asl - 21.43,
        "CLI": 0.058 * pct(sum(letters)) - 0.296 * pct(n_sent) - 15.8,
        "LW": (lw - 2 if lw <= 20 else lw) / 2,
        "DCRS": 0.1579 * pdw + 0.0496 * asl + (3.6365 if pdw > 5 else 0),
        "SPACHE": 0.141 * asl + 0.086 * pct(sum(h and n >= 2 for h, n in zip(hard, syllables))) + 0.839,
        "WSTF1": 0.1935 * ms + 0.1672 * asl + 0.1297 * iw - 0.0327 * es - 0.875,
        "WSTF2": 0.2007 * ms + 0.1682 * asl + 0.1373 * iw - 2.779,
        "WSTF3": 0.2963 * ms + 0.1905 * asl - 1.1144,
        "WSTF4": 0.2744 * ms + 0.2656 * asl - 1.693,
    }

# A word, for lexical diversity, content-word overlap, the word norms, polysemy / hypernymy and the frequencies:
# letters, or letters joined by hyphens. Before, Python's str.isalpha (as spaCy's is_alpha, which DUUI used), so
# hyphenated compounds ("Bundes-Regierung", "E-Mail"; 2.9 % of German content words, 8.2 % in Nachrichtenleicht) dropped out,
# although DESWC counts them, and the hyphen rule of fix B3 (_de_compound_head) could not be reached. Everything else stays as
# it was: numbers, symbols, abbreviations with periods, words with digits, English clitics ("n't", "'s"). The LSA sentence
# vectors keep their alphabetic tokens (_get_paragraph_token_vectors).
def _is_word_text(text: str) -> bool:
    return bool(text) and all(part.isalpha() for part in text.split("-"))


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

# speed (resurrection, 2026-10-22): the overlap indices over all sentence pairs (CRF*a) recomputed each sentence's annotation check
# and word sets once per pair (1.4 million pairs on a page of 1,668 sentences). They are now computed once per sentence and request,
# with the same expressions as before; compute resets the cache. The entry keeps the sentence itself, so a reused id cannot hit.
class _PerRequestDict:
    """a dict per request: concurrent requests (FastAPI runs each in a thread with its own context) do not see each other's
    entries; compute() resets it before and after each text"""

    def __init__(self, name: str):
        self._var: ContextVar = ContextVar(name, default=None)

    def _dict(self) -> dict:
        d = self._var.get()
        if d is None:
            d = {}
            self._var.set(d)
        return d

    def reset(self) -> None:
        self._var.set({})

    def clear(self) -> None:
        self._dict().clear()

    def get(self, key, default=None):
        return self._dict().get(key, default)

    def __getitem__(self, key):
        return self._dict()[key]

    def __setitem__(self, key, value):
        self._dict()[key] = value

    def __contains__(self, key):
        return key in self._dict()


_SENTENCE_FEATURES = _PerRequestDict("_SENTENCE_FEATURES")   # {id(sentence): (sentence, features)}

# speed: steps several indices share run once per text: the DiMLex pass of a sentence (CNC*,
# SMCAUSvp, SMCAUSr: three times before), the situation-model data and verb counts (SMCAUSv, SMCAUSvp, SMINTEp, SMCAUSr,
# SMINTEr, SMCAUSlsa, SMCAUSwn: seven and five times), the syllables (DESWLsy, DESWLsyd), the lexical-diversity words (four LD*)
# and the all-pairs content-word overlap (CRFCWOa, CRFCWOad). No caller changes what it gets. compute resets the memo
# before and after each text; an entry keeps its key objects, so a reused id cannot hit.
_TEXT_MEMO = _PerRequestDict("_TEXT_MEMO")   # {key: (key objects, value)}


def _memo(name: str, objects: tuple, extra, compute):
    key = (name, extra) + tuple(id(o) for o in objects)
    hit = _TEXT_MEMO.get(key)
    if hit is not None and all(a is b for a, b in zip(hit[0], objects)):
        return hit[1]
    value = compute()
    _TEXT_MEMO[key] = (objects, value)
    return value


def _sentence_features(sentence: Sentence) -> dict:
    hit = _SENTENCE_FEATURES.get(id(sentence))
    if hit is not None and hit[0] is sentence:
        return hit[1]
    tokens = sentence.tokens
    non_punct = _non_punct_tokens(sentence)
    # the word sets come from the non-punctuation tokens, as the usability checks (German UD tags "%" as
    # NOUN, so "5 %" ... "3 %" made a noun overlap); nouns compare in lower case (English "Turbulence" / "turbulence"); content
    # words are words with a letter (_is_word_text)
    content_words = [t.text.lower() for t in non_punct if t.pos_coarse in ud_stem_content_pos and _is_word_text(t.text)]
    f = {
        "pos_complete": _has_complete_pos_annotations(sentence),
        "noun_texts": {t.text.lower() for t in non_punct if t.pos_coarse in ud_noun_pos},
        "noun_lemmas_usable": not any(t.pos_coarse in ud_noun_pos and not _has_usable_lemma(t) for t in non_punct),
        "noun_lemmas": {t.lemma for t in non_punct if t.pos_coarse in ud_noun_pos and t.lemma and t.lemma != "--"},
        "pronouns": {t.text.lower() for t in non_punct if t.pos_coarse in ud_pronouns_pos},
        "content_lemmas_usable": not any(t.pos_coarse in ud_content_pos.union(ud_noun_pos) and not _has_usable_lemma(t)
                                         for t in non_punct),
        "stem_nouns": {t.lemma.lower() for t in non_punct if t.pos_coarse in ud_noun_pos and t.lemma},
        "stem_contents": {t.lemma.lower() for t in non_punct if t.pos_coarse in ud_content_pos and t.lemma},
        "content_words": content_words,
        "content_word_set": set(content_words),
    }
    _SENTENCE_FEATURES[id(sentence)] = (sentence, f)
    return f


def _noun_overlap(sentence_a: Sentence, sentence_b: Sentence) -> Optional[int]:
    fa, fb = _sentence_features(sentence_a), _sentence_features(sentence_b)
    if not (fa["pos_complete"] and fb["pos_complete"]):
        return None
    return len(fa["noun_texts"].intersection(fb["noun_texts"]))

def _argument_overlap(sentence_a: Sentence, sentence_b: Sentence) -> Optional[int]:
    fa, fb = _sentence_features(sentence_a), _sentence_features(sentence_b)
    if not (fa["pos_complete"] and fb["pos_complete"]):
        return None
    if not (fa["noun_lemmas_usable"] and fb["noun_lemmas_usable"]):
        return None
    noun_overlap = len(fa["noun_lemmas"].intersection(fb["noun_lemmas"]))
    pronoun_overlap = len(fa["pronouns"].intersection(fb["pronouns"]))

    return noun_overlap + pronoun_overlap

# FV3 Fix: Stem overlap compares noun lemmas in the current sentence
# with content-word lemmas in the previous sentence and also includes
# matching pronouns as specified for Coh-Metrix stem overlap.
def _stem_overlap(
    sentence_nouns: Sentence,
    sentence_contents: Sentence
) -> Optional[int]:
    fn, fc = _sentence_features(sentence_nouns), _sentence_features(sentence_contents)
    if not (fn["pos_complete"] and fc["pos_complete"]):
        return None
    if not (fn["content_lemmas_usable"] and fc["content_lemmas_usable"]):
        return None
    noun_content_overlap = len(
        fn["stem_nouns"].intersection(fc["stem_contents"])
    )
    pronoun_overlap = len(
        fn["pronouns"].intersection(fc["pronouns"])
    )

    return noun_content_overlap + pronoun_overlap

# FV3 Fix: CRFCWO measures the proportion of explicit surface-form
# content words in the current sentence that also occur in the previous sentence.
def _word_overlap(
    current_sentence: Sentence,
    previous_sentence: Sentence
) -> Optional[float]:
    fc, fp = _sentence_features(current_sentence), _sentence_features(previous_sentence)
    if not (fc["pos_complete"] and fp["pos_complete"]):
        return None
    current_content_words = fc["content_words"]
    previous_content_words = fp["content_word_set"]

    if not current_content_words:
        return None

    overlap = sum(
        1
        for word in current_content_words
        if word in previous_content_words
    )
    # FV3 Fix: Normalize overlap by the number of content words in the
    # current sentence instead of using the union of both sentences.
    return overlap / len(current_content_words)

# speed (resurrection, 2026-10-23): the indices over all sentence pairs (CRFNOa, CRFAOa, CRFSOa, CRFCWOa, CRFCWOad) computed
# as sparse products of sentence x word matrices, one earlier sentence a at a time against all sentences b, so the values come in
# the order of the original loops (a outer, b > a inner). A pair counts under the same checks as in _noun_overlap,
# _argument_overlap, _stem_overlap and _word_overlap; the overlaps are exact counts (sums of 1.0), and the word-overlap share is
# the same division.
def _word_matrices(families: List[List[Any]], counts: List[bool]):
    """one sparse sentence x word matrix per family of per-sentence word collections, over one shared vocabulary; an entry is 1.0
    per distinct word, or (counts=True) the number of times the word occurs"""
    from collections import Counter
    from scipy.sparse import csr_matrix
    vocab: Dict[str, int] = {}
    parts = []
    for family, count in zip(families, counts):
        indptr, idx, data = [0], [], []
        for words in family:
            items = Counter(words).items() if count else ((w, 1) for w in set(words))
            for w, k in items:
                idx.append(vocab.setdefault(w, len(vocab)))
                data.append(float(k))
            indptr.append(len(idx))
        parts.append((data, idx, indptr))
    shape = (len(families[0]), max(1, len(vocab)))
    return [csr_matrix((np.asarray(d, dtype=np.float64), np.asarray(i, dtype=np.int64), np.asarray(p, dtype=np.int64)), shape=shape)
            for d, i, p in parts]


def _pair_rows(earlier, later, block: int = 256):
    """for every sentence a, in order: the dense vector over all sentences b of earlier[a] . later[b]"""
    later_t = later.T.tocsc()
    for a0 in range(0, earlier.shape[0], block):
        products = (earlier[a0:a0 + block] @ later_t).toarray()
        for k in range(products.shape[0]):
            yield a0 + k, products[k]


class _Moments:
    """count, mean and SD (ddof 0) of values that come in parts, without keeping them (the
    all-pairs indices joined their n(n - 1) / 2 values first, 155 MB per index at 6,218 sentences). The mean is the sum over the
    count as in np.mean, identical for 0 / 1 values, whose sums are exact; the SD merges the parts' squared deviations (Chan,
    Golub & LeVeque 1979), the same as np.std up to rounding."""

    def __init__(self):
        self.n, self.total, self.m2 = 0, 0.0, 0.0

    def add(self, values) -> None:
        k = values.size
        if not k:
            return
        part_total = float(values.sum())
        part_mean = part_total / k
        part_m2 = float(np.square(values - part_mean).sum())
        if self.n:
            delta = part_mean - self.total / self.n
            part_m2 += delta * delta * self.n * k / (self.n + k)
        self.n, self.total, self.m2 = self.n + k, self.total + part_total, self.m2 + part_m2

    def mean(self) -> float:
        return self.total / self.n if self.n else np.nan

    def std(self) -> float:
        return math.sqrt(self.m2 / self.n) if self.n else np.nan


def _word_overlap_all_pairs(sentences: List[Sentence]) -> _Moments:
    """_word_overlap(current b, previous a) for all pairs a < b, without the pairs it returns None for: their count, mean and SD
    (once per text for CRFCWOa and CRFCWOad)"""
    return _memo("word_overlap_all_pairs", (sentences,), None, lambda: _word_overlap_all_pairs_uncached(sentences))


def _word_overlap_all_pairs_uncached(sentences: List[Sentence]) -> _Moments:
    feats = [_sentence_features(s) for s in sentences]
    ok = np.array([f["pos_complete"] for f in feats], dtype=bool)
    n_words = np.array([len(f["content_words"]) for f in feats], dtype=np.float64)
    keep = ok & (n_words > 0)
    previous, current = _word_matrices([[f["content_word_set"] for f in feats], [f["content_words"] for f in feats]], [False, True])
    moments = _Moments()
    for a, row in _pair_rows(previous, current):
        if ok[a]:
            sel = keep[a + 1:]
            moments.add(row[a + 1:][sel] / n_words[a + 1:][sel])
    return moments

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
        noun_overlap = _noun_overlap(current_sentence, previous_sentence)
        if noun_overlap is not None:
            noun_overlap_per_sentence.append(min(1, noun_overlap))
    return np.mean(noun_overlap_per_sentence)

# LAY: Do any two sentences in the text share at least one noun?
# ↑ Higher = noun recurrence across the whole text. Reliable.
def cm_crfnoa(sentences: List[Sentence]) -> Optional[float]:
    # Noun overlap, all sentences, binary, mean
    # speed (resurrection): _noun_overlap for all pairs a < b at once (see _pair_rows)
    feats = [_sentence_features(s) for s in sentences]
    ok = np.array([f["pos_complete"] for f in feats], dtype=bool)
    (nouns,) = _word_matrices([[f["noun_texts"] for f in feats]], [False])
    noun_overlap = _Moments()
    for a, row in _pair_rows(nouns, nouns):
        if ok[a]:
            noun_overlap.add(np.minimum(1, row[a + 1:][ok[a + 1:]]))
    return noun_overlap.mean()

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
        argument_overlap = _argument_overlap(current_sentence, previous_sentence)
        if argument_overlap is not None:
            argument_overlap_per_sentence.append(min(1, argument_overlap))
    return np.mean(argument_overlap_per_sentence)

# LAY: Do any two sentences share a noun or pronoun?
# ↑ Higher = entity recurrence throughout text. Reliable.
def cm_crfaoa(sentences: List[Sentence]) -> Optional[float]:
    # Argument overlap, all sentences, binary, mean
    # speed (resurrection): _argument_overlap for all pairs a < b at once (see _pair_rows)
    feats = [_sentence_features(s) for s in sentences]
    ok = np.array([f["pos_complete"] and f["noun_lemmas_usable"] for f in feats], dtype=bool)
    lemmas, pronouns = _word_matrices([[f["noun_lemmas"] for f in feats], [f["pronouns"] for f in feats]], [False, False])
    argument_overlap = _Moments()
    for (a, row_l), (_, row_p) in zip(_pair_rows(lemmas, lemmas), _pair_rows(pronouns, pronouns)):
        if ok[a]:
            argument_overlap.add(np.minimum(1, (row_l + row_p)[a + 1:][ok[a + 1:]]))
    return argument_overlap.mean()

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
        stem_overlap = _stem_overlap(current_sentence, previous_sentence)
        if stem_overlap is not None:
            stem_overlap_per_sentence.append(min(1, stem_overlap))
    return np.mean(stem_overlap_per_sentence)

# LAY: Do any two sentences share a word stem?
# ↑ Higher = word-family recurrence throughout text. Reliable.
# FV3 Fix: Apply the same temporal direction as local CRFCWO:
# compare each later sentence against each earlier sentence.
def cm_crfsoa(sentences: List[Sentence]) -> Optional[float]:
    # Stem overlap, all sentences, binary, mean
    # speed (resurrection): _stem_overlap(current b, previous a) for all pairs a < b at once (see _pair_rows): the noun lemmas of
    # the later sentence against the content lemmas of the earlier one, plus the shared pronouns
    feats = [_sentence_features(s) for s in sentences]
    ok = np.array([f["pos_complete"] and f["content_lemmas_usable"] for f in feats], dtype=bool)
    contents, nouns, pronouns = _word_matrices([[f["stem_contents"] for f in feats], [f["stem_nouns"] for f in feats],
                                                [f["pronouns"] for f in feats]], [False, False, False])
    stem_overlap = _Moments()
    for (a, row_s), (_, row_p) in zip(_pair_rows(contents, nouns), _pair_rows(pronouns, pronouns)):
        if ok[a]:
            stem_overlap.add(np.minimum(1, (row_s + row_p)[a + 1:][ok[a + 1:]]))
    return stem_overlap.mean()

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
        if word_overlap is not None:
            word_overlap_per_sentence.append(word_overlap)
    return (
        np.mean(word_overlap_per_sentence)
        if word_overlap_per_sentence
        else np.nan
    )

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
        if word_overlap is not None:
            word_overlap_per_sentence.append(word_overlap)
    return (
        np.std(word_overlap_per_sentence)
        if word_overlap_per_sentence
        else np.nan
    )

# LAY: What share of content words is shared across all sentence pairs? (mean)
# ↑ Higher = global cohesion. Reliable.
# FV3 Fix: Apply the same temporal direction as local CRFCWO:
# compare each later sentence against each earlier sentence.
def cm_crfcwoa(sentences: List[Sentence]) -> Optional[float]:
    return _word_overlap_all_pairs(sentences).mean()

# LAY: How uneven is the global content-word overlap across sentence pairs?
# ↑ Higher = mix of tight and loose cohesion. Reliable.
# FV3 Fix: Apply the same temporal direction as local CRFCWO:
# compare each later sentence against each earlier sentence.
def cm_crfcwoad(sentences: List[Sentence]) -> Optional[float]:
    return _word_overlap_all_pairs(sentences).std()

def _lexical_diversity_tokens(tokens: List[Token]) -> Tuple[List[str], List[str], List[str]]:
    return _memo("lexical_diversity", (tokens,), None, lambda: _lexical_diversity_tokens_uncached(tokens))


def _lexical_diversity_tokens_uncached(tokens: List[Token]) -> Tuple[List[str], List[str], List[str]]:
    # M4 fix: Appendix A #46 specifies LDTTRc uses "content word LEMMAS", not
    # surface forms. Return an extra lemma list for LDTTRc; keep surface-form
    # lists for LDTTRa/LDMTLDa/LDVOCDa ("all words", spec is surface).
    # words: tokens with a letter (_is_word_text, before str.isalpha)
    tokens_alpha = [token.text.lower() for token in tokens if _is_word_text(token.text)]
    tokens_content = [token.text.lower() for token in tokens if token.pos_coarse in ud_content_pos and _is_word_text(token.text)]
    tokens_content_lemma = [
        (token.lemma.lower() if token.lemma else token.text.lower())
        for token in tokens
        if token.pos_coarse in ud_content_pos and _is_word_text(token.text)
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
    if not tokens_content_lemma:
        return None
    return ld.ttr(tokens_content_lemma)

# LAY: Ratio of unique words to total words (all alphabetic words).
# ↑ Higher = more varied vocabulary overall. Reliable.
def cm_ldttra(tokens: List[Token]) -> Optional[float]:
    tokens_alpha, _, _ = _lexical_diversity_tokens(tokens)
    if not tokens_alpha:
        return None
    return ld.ttr(tokens_alpha)

# LAY: MTLD — vocabulary diversity, robust to text length.
# ↑ Higher = more varied vocabulary (not inflated by length). Reliable.
def cm_ldmtlda(tokens: List[Token]) -> Optional[float]:
    tokens_alpha, _, _ = _lexical_diversity_tokens(tokens)
    # no word repeated: no MTLD factor completes and lexical_diversity returns 0, one repeat more gives ~120;
    # undefined
    if not tokens_alpha or len(set(tokens_alpha)) == len(tokens_alpha):
        return None
    return ld.mtld(tokens_alpha)

# LAY: VOCD — vocabulary diversity via random-sample curve fitting.
# ↑ Higher = more varied vocabulary. Reliable.
_VOCD_LOCK = threading.Lock()


def cm_ldvocda(tokens: List[Token]) -> Optional[float]:
    tokens_alpha, _, _ = _lexical_diversity_tokens(tokens)
    # no word repeated: the curve cannot be fitted and vocd returns D of about 10^6; undefined
    if not tokens_alpha or len(set(tokens_alpha)) == len(tokens_alpha):
        return None
    lex = LexicalRichness(tokens_alpha, preprocessor=None, tokenizer=None)
    # lexicalrichness seeds and draws from Python's global random generator (random.seed(42), random.sample): concurrent
    # requests would interleave their draws and the value would no longer be reproducible
    with _VOCD_LOCK:
        return lex.vocd()

# ============================================================================
# SYNTACTIC COMPLEXITY (SYN*) — "How complex and consistent is sentence structure?"
# Left-embeddedness, noun-phrase size, sentence-pair edit distance, and
# structural similarity. SYNSTRUT* is an approximation (we use dependency
# rather than constituency parses); the others are reliable.
# ============================================================================

_DE_SYNLE_FINITE_TAGS = {"VVFIN", "VAFIN", "VMFIN", "VVIMP", "VAIMP"}


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
        # Coh-Metrix SYNLE is approximated from the sentence parse root.
        # Do not replace an auxiliary/modal root with a language-specific
        # lexical verb: that would change the original index definition.
        main_verb_index = None

        for token_index, token in enumerate(sentence.tokens):
            if (token.dep_type or "").strip() in _ROOT_MARKERS:
                main_verb_index = token_index
                break

            # Defensive fallback for self-headed ROOT conventions.
            if token.head_index == token_index:
                main_verb_index = token_index
                break

        if main_verb_index is None:
            continue

        # German (UD port, 2026-10-28): in TIGER the root of a main clause is its finite verb (the left bracket after the
        # Vorfeld); UD makes the content word the root ("Er hat das Buch gelesen": gelesen; "Das ist wichtig": wichtig) and
        # attaches the finite auxiliary or copula to it. So the finite verb is taken: the root if finite, else its finite
        # aux / aux:pass / cop dependent; a root without one (verbless sentence) stays. The lexical verb usually ends a German
        # clause (right bracket), so counting to it would count almost the whole clause (decision 2026-11-02; our analogy).
        # English (decision 2026-11-02): the words before the lexical verb, as in Graesser, McNamara & Kulikowich (2011,
        # p. 226): "At any time during the last 12 months, were you or any member of your household enrolled in ..." has "16
        # [words] before the main verb (i.e., 'enrolled')", the auxiliary "were" among them. That is the UD root if it is a
        # verb; in a copula clause ("The cake is good": root "good") the copula, the clause's only verb.
        if lang == "de":
            finite_tags = _DE_SYNLE_FINITE_TAGS
            if sentence.tokens[main_verb_index].pos_value not in finite_tags:
                main_verb_index = next(
                    (j for j, t in enumerate(sentence.tokens)
                     if t.head_index == main_verb_index and j != main_verb_index
                     and (t.dep_type or "").lower() in ("aux", "aux:pass", "cop") and t.pos_value in finite_tags),
                    main_verb_index)
        elif sentence.tokens[main_verb_index].pos_coarse != "VERB":
            main_verb_index = next(
                (j for j, t in enumerate(sentence.tokens)
                 if t.head_index == main_verb_index and j != main_verb_index and (t.dep_type or "").lower() == "cop"),
                main_verb_index)

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
        else np.nan
    )

def label_scheme(request) -> str:
    """the dependency scheme of the request: "UD", or "TIGER" (de) / "ClearNLP" (en), spaCy's labels, which compute() maps to
    UD as best effort (ud_mapping); UD version 1 input is refused (ValueError)"""
    return ud_mapping.label_scheme(_language(request), (t.dep_type for p in request.paragraphs for s in p.sentences
                                                         for t in s.tokens))


# Text Easability components (2026-10-29): the German model fitted on reference
# corpus v1 with this core's indices (easability/build_model.py, redundancy rule: 45 measures),
# applied as duui-coh-metrix's cm_text_easability applies a model: z = standardized inputs @ weights, the percentile from the
# reference corpus's quantiles; a missing input is replaced by the reference mean only where the reference had missing values,
# otherwise the components are undefined. Inputs are matched by label_internal, so other resources (e.g. other word vectors)
# leave the components undefined instead of scoring them against the wrong reference. English (2026-10-30):
# the model fitted on reference corpus A, its components
# named by congruence with the German model; one model per language, each checked to be of its language.
def _load_easability_model(path: str, lang: str) -> Optional[dict]:
    try:
        model = json.loads(Path(path).read_text())
    except OSError:
        logger.warning('Text Easability model (%s) not found at "%s": the PC* indices will return None', lang, path)
        return None
    if model.get("language", "de") != lang:
        raise ValueError(f'Text Easability model "{path}" is for language {model.get("language")!r}, not {lang!r}')
    for m in model["inputs"]:
        m["_parts"] = m["labels_internal"]
        if not (isinstance(m.get("scale"), (int, float)) and math.isfinite(m["scale"]) and m["scale"] > 0):
            raise ValueError(f'Text Easability model "{path}": input {m["_parts"]} has scale {m.get("scale")!r}')
    model["_weights"] = np.asarray([c["weights"] for c in model["components"]]).T
    logger.info('Text Easability model "%s": %d inputs, %d reference texts', model["model_id"], len(model["inputs"]),
                model["documents"])
    return model


easability_models = {"de": _load_easability_model(settings.easability_model_path, "de"),
                     "en": _load_easability_model(settings.easability_model_path_en, "en")}


def _easability_scores(values: Dict[str, Optional[float]], model: dict) -> Tuple[Dict[str, float], Optional[str]]:
    """z score and percentile of every component for one text's index values ({label_internal: value})."""
    standardized = []
    for m in model["inputs"]:
        # an input whose label was not computed at all (another resource, e.g. other word vectors) leaves the components
        # undefined, as documented above; before it was imputed where the reference corpus had missing values
        #
        absent = [label for label in m["_parts"] if label not in values]
        if absent:
            return {}, f"input {' + '.join(absent)} not computed (another resource than the model's?)"
        parts = [values.get(label) for label in m["_parts"]]
        if any(v is None or math.isnan(v) for v in parts):
            if m["missing_documents"] == 0:
                return {}, f"input {' + '.join(m['_parts'])} missing (never missing in the reference corpus)"
            standardized.append(0.0)
        else:
            standardized.append((sum(parts) - m["mean"]) / m["scale"])
    z = np.asarray(standardized) @ model["_weights"]
    scores = {}
    for c, zc in zip(model["components"], z):
        q = c["percentile_quantiles"]
        scores[c["label_v3_z"]] = float(zc)
        scores[c["label_v3_p"]] = float(np.interp(zc, q, np.linspace(0, 100, len(q))))
    return scores, None

# Noun chunks from the UD tree, for German and (since 2026-10-29) English. spaCy's German noun-chunk iterator (spacy/lang/de/syntax_iterators.py) reads TIGER
# labels, so the chunks are built here with its logic on UD relations (also for DUUI input from an upstream UD parser, which
# brings no chunks): a noun, proper noun or pronoun in one of the relations below (TIGER: sb, oa, da, og, nk, mo, ag, cj, pd,
# app, root) spans from the leftmost word of its phrase to itself, extended over a close apposition on its right (UD flat,
# flat:name; TIGER nk: "Stadt Berlin"); a candidate whose phrase starts inside the previous chunk is skipped (as spaCy). The
# phrase of a noun: itself and the subtrees of its phrase-internal dependents (_UD_NP_INTERNAL); UD makes the predicate of a
# copular clause its head ("Er ist ein guter Lehrer"), so its subject and copula are not part of it. A preposition (case) or
# coordinator (cc) at the start is left out (TIGER does not attach them to the noun). English: spaCy's English iterator
# (spacy/lang/en/syntax_iterators.py) uses the same idea on ClearNLP labels (nsubj, nsubjpass, dobj, dative, pobj, appos, attr,
# conj, ROOT; the noun's left edge to the noun), so the same UD rule serves both; "all the children" adds det:predet.
# Code review 2026-10-31: "nmod:poss" was a head relation too. An English possessor ("their wattled homes": their = nmod:poss)
# then became a chunk of its own and the possessed noun was skipped as nested (107 of 2,464 chunks of the 20 English test
# texts); spaCy's English iterator has no possessor label. It stays phrase-internal (_UD_NP_INTERNAL).
_UD_NP_DEPS = {"nsubj", "nsubj:pass", "obj", "iobj", "obl", "obl:agent", "obl:arg", "nmod", "appos", "conj", "root"}
_UD_NP_INTERNAL = {"det", "det:poss", "det:predet", "amod", "nummod", "nmod", "nmod:poss", "compound", "case", "cc", "flat",
                   "flat:name"}


def _ud_noun_chunks(sentences: List[Sentence]) -> List[NounChunk]:
    chunks = []
    for sentence in sentences:
        toks = sentence.tokens
        n = len(toks)
        kids = [[] for _ in range(n)]
        for j, t in enumerate(toks):
            h = t.head_index
            if h is not None and 0 <= h < n and h != j:
                kids[h].append(j)

        # the leftmost word of every subtree, bottom-up without recursion (a deep tree, e.g. a long nmod
        # chain, exceeded Python's recursion limit, and as the chunks are built before the first index the output was empty);
        # the words of a head cycle, which a tree cannot have, keep their own position
        left = list(range(n))
        order = [j for j in range(n) if toks[j].head_index is None or not 0 <= toks[j].head_index < n or toks[j].head_index == j]
        k = 0
        while k < len(order):
            order.extend(kids[order[k]])
            k += 1
        for j in reversed(order):
            for c in kids[j]:
                left[j] = min(left[j], left[c])

        dep = [(t.dep_type or "").lower() for t in toks]
        prev_end = -1
        for j, t in enumerate(toks):
            if t.pos_coarse not in ("NOUN", "PROPN", "PRON") or dep[j] not in _UD_NP_DEPS:
                continue
            start = min([j] + [left[c] for c in kids[j] if dep[c] in _UD_NP_INTERNAL])
            if start <= prev_end:
                continue
            end = j
            for c in kids[j]:
                if c > j and toks[c].pos_coarse in ("NOUN", "PROPN") and dep[c] in ("flat", "flat:name"):
                    end = max(end, c)
            while start < j and dep[start] in ("case", "cc", "punct"):
                start += 1
            prev_end = end
            chunks.append(NounChunk(begin=toks[start].begin, end=toks[end].end))
    return chunks


def _chunk_head(chunk_indices: List[int], tokens: List[Token]) -> Optional[int]:
    """the head of a noun chunk: its token whose head is outside the chunk (or itself, the root); of several, the rightmost
    noun or pronoun, else the rightmost one"""
    inside = set(chunk_indices)
    heads = [i for i in chunk_indices if tokens[i].head_index is None or tokens[i].head_index == i
             or tokens[i].head_index not in inside]
    if not heads:
        return None
    nominal = [i for i in heads if tokens[i].pos_coarse in ("NOUN", "PROPN", "PRON")]
    return (nominal or heads)[-1]


# SYNNP, "Number of modifiers per noun phrase, mean" (App. A). Graesser et al. (2004, p. 198): "A modifier is an optional element
# describing the property of the head of a phrase ... the NP the lovely little girl is an NP that has three modifiers and a
# head" - the determiner counts. Our operationalisation (decision 2026-11-02, C'): per noun chunk, the words that depend
# directly on the chunk's head, without punctuation ("the lovely little girl" 3, "a very big dog" 2: "very" describes "big",
# "the old man's house" 1: the possessor). The same rule for every language and for mapped (best-effort) input.
def cm_synnp(sentences: List[Sentence], noun_chunks: List[NounChunk], lang: str) -> Optional[float]:

    if noun_chunks and not all(
        _has_complete_pos_annotations(sentence)
        and _has_complete_dependency_annotations(sentence)
        for sentence in sentences
    ):
        return None

    # speed: the sentence of a chunk by bisection (the scan of every sentence per chunk took
    # 4.25 s at 2,000 sentences). With the sentences' starts and ends ascending (text order), the scan's hit, the first sentence
    # that contains the chunk, can only be the first one that ends at or after the chunk's end; otherwise the scan as before
    ends = [sentence.end for sentence in sentences]
    ordered = all(s.begin <= t.begin and s.end <= t.end for s, t in zip(sentences, sentences[1:]))
    modifier_counts = []
    for noun_chunk in noun_chunks:
        k = bisect.bisect_left(ends, noun_chunk.end) if ordered else 0
        for sentence in (sentences[k:k + 1] if ordered else sentences):
            if sentence.begin <= noun_chunk.begin and noun_chunk.end <= sentence.end:
                chunk = [i for i, tok in enumerate(sentence.tokens)
                         if tok.begin >= noun_chunk.begin and tok.end <= noun_chunk.end]
                head = _chunk_head(chunk, sentence.tokens)
                if head is not None:
                    modifier_counts.append(sum(1 for i in chunk if i != head and sentence.tokens[i].head_index == head
                                               and not sentence.tokens[i].is_punct))
                break

    return np.mean(modifier_counts) if modifier_counts else np.nan

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
    pos_dists = []

    for first, second in zip(sentences, sentences[1:]):
        if not (_has_complete_pos_annotations(first)
                and _has_complete_pos_annotations(second)):
            continue
        distance = _normalized_sequence_edit_distance(
            [token.pos_coarse for token in first.tokens],
            [token.pos_coarse for token in second.tokens]
        )
        if np.isfinite(distance):
            pos_dists.append(distance)

    return np.mean(pos_dists) if pos_dists else np.nan

# LAY: How different are adjacent sentences word-for-word?
# ↑ Higher = neighbours share fewer surface words. Reliable.
def cm_synmedwrd(sentences: List[Sentence]) -> Optional[float]:
    # H1 fix: compute mean normalized edit distance between CONSECUTIVE SENTENCES'
    # word sequences, following the Coh-Metrix MED definition.
    #
    # FV2 fix: compare complete words as sequence elements instead of
    # characters of a joined sentence string. Replacing one word therefore
    # counts as one substitution regardless of the word's character length.
    word_dists = []

    for first, second in zip(sentences, sentences[1:]):
        if not first.tokens or not second.tokens:
            continue
        distance = _normalized_sequence_edit_distance(
            [token.text for token in first.tokens],
            [token.text for token in second.tokens]
        )
        if np.isfinite(distance):
            word_dists.append(distance)

    return np.mean(word_dists) if word_dists else np.nan

# LAY: How different are adjacent sentences at the lemma level?
# ↑ Higher = neighbours share fewer word stems. Reliable.
def cm_synmedlem(sentences: List[Sentence]) -> Optional[float]:
    # H1 fix: compute mean normalized edit distance between CONSECUTIVE SENTENCES'
    # lemma sequences, following the Coh-Metrix MED definition.
    #
    # FV2 fix: compare complete lemmas as sequence elements instead of
    # characters of a joined lemma string. This ensures that inflectional
    # variants mapped to the same lemma do not introduce artificial distance.
    lemma_dists = []

    for first, second in zip(sentences, sentences[1:]):
        if not first.tokens or not second.tokens:
            continue
        if any(
            not _has_usable_lemma(token)
            for sentence in (first, second)
            for token in _non_punct_tokens(sentence)
        ):
            continue
        distance = _normalized_sequence_edit_distance(
            [token.lemma for token in first.tokens],
            [token.lemma for token in second.tokens]
        )
        if np.isfinite(distance):
            lemma_dists.append(distance)

    return np.mean(lemma_dists) if lemma_dists else np.nan

# SYNSTRUT is a documented approximation of Coh-Metrix's
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

        token = tokens[token_index]
        is_explicit_root = (
            _normalize_tree_dep(token.dep_type) == "ROOT"
            or token.head_index == token_index
        )

        if parent_index is None:
            # Missing, invalid, or cyclic governor information is not a valid
            # additional root. Without the required dependency structure the
            # complete sentence tree is not computable.
            if not is_explicit_root:
                return None
            roots.append(token_index)
        else:
            children[parent_index].append(token_index)

    if len(roots) != 1:
        return None

    # A valid dependency tree must cover every non-punctuation token exactly
    # once from its single ROOT. This rejects detached cyclic islands that
    # were previously ignored and made an incomplete tree look valid.
    reached = set()
    active_path = set()

    def _visit(token_index: int) -> bool:
        if token_index in active_path or token_index in reached:
            return False
        active_path.add(token_index)
        reached.add(token_index)
        for child_index in children[token_index]:
            if not _visit(child_index):
                return False
        active_path.remove(token_index)
        return True

    if not _visit(roots[0]) or reached != valid_indices:
        return None

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


class _TreeInterner:
    """the dependency trees of one text, interned: one integer per distinct subtree (its label and its children's integers),
    with its size; the common-subtree sizes are memoised on pairs of integers (reviewer B's
    prototype). Before, two lru_caches hashed the nested tuples again on every lookup, and the one of 16,384 pairs was full from
    about 250 sentences on (SYNSTRUTt 12 s on a 1,392-sentence text, quadratic). One per text, shared by SYNSTRUTa and
    SYNSTRUTt; nothing global, so requests running at the same time cannot see each other's tables."""

    def __init__(self):
        self.ids: Dict[tuple, int] = {}
        self.kids: List[tuple] = []
        self.labels: List[Any] = []
        self.sizes: List[int] = []
        self.common: Dict[Tuple[int, int], int] = {}

    def intern(self, tree) -> int:
        label, children = tree
        kids = tuple(self.intern(child) for child in children)
        key = (label, kids)
        i = self.ids.get(key)
        if i is None:
            i = self.ids[key] = len(self.kids)
            self.kids.append(kids)
            self.labels.append(label)
            self.sizes.append(1 + sum(self.sizes[k] for k in kids))
        return i

    def common_size(self, a: int, b: int) -> int:
        """the size of the largest common subtree of the trees a and b, rooted at both roots, as before: 0 if the root labels
        differ, else 1 + the best order-preserving matching of the children; a tree with itself has its own size, and the
        measure is symmetric"""
        if a == b:
            return self.sizes[a]
        labels = self.labels
        if labels[a] != labels[b]:
            return 0
        key = (a, b) if a < b else (b, a)
        hit = self.common.get(key)
        if hit is not None:
            return hit
        children2 = self.kids[b]
        m = len(children2)
        previous = [0] * (m + 1)
        for x in self.kids[a]:
            current = [0] * (m + 1)
            label = labels[x]
            for j, y in enumerate(children2, 1):
                best = previous[j] if previous[j] > current[j - 1] else current[j - 1]
                if labels[y] == label:              # a pair with other labels matches 0 and cannot beat previous[j]
                    matched = previous[j - 1] + self.common_size(x, y)
                    if matched > best:
                        best = matched
                current[j] = best
            previous = current
        value = self.common[key] = 1 + previous[m]
        return value

    def similarity(self, a: int, b: int) -> float:
        """common / (size a + size b - common), as before"""
        common_size = self.common_size(a, b)
        denominator = self.sizes[a] + self.sizes[b] - common_size
        return np.nan if denominator == 0 else common_size / denominator


# LAY: Mean syntactic similarity between adjacent sentences.
# ↑ Higher = consecutive sentences use more similar syntactic structures.
# Approximate for Coh-Metrix 3.0 because dependency trees are used instead
# of the original constituency parse trees.
def cm_synstruta(
    sentences: List[Sentence],
    trees: Optional[_TreeInterner] = None
) -> float:

    trees = _TreeInterner() if trees is None else trees
    similarities = []
    # each sentence's tree built and interned once (before twice, as left and as right neighbour)
    ids = [None if tree is None else trees.intern(tree) for tree in map(_build_dependency_tree, sentences)]

    for i in range(len(sentences) - 1):
        if ids[i] is None or ids[i + 1] is None:
            continue

        similarities.append(trees.similarity(ids[i], ids[i + 1]))

    return (
        np.mean(similarities)
        if similarities
        else np.nan
    )


# LAY: Mean syntactic similarity between all sentence pairs that belong
# to different paragraphs.
# ↑ Higher = syntactic structures remain similar across paragraph boundaries.
# Approximate for Coh-Metrix 3.0 because dependency trees are used instead
# of the original constituency parse trees.
def cm_synstrutt(
    paragraphs: List[Paragraph],
    trees: Optional[_TreeInterner] = None
) -> float:

    trees = _TreeInterner() if trees is None else trees
    # per paragraph how often each distinct (interned) tree occurs: each distinct pair is compared once
    # and weighed by the number of sentence pairs it stands for; the mean is the same up to rounding
    paragraph_trees = []

    for paragraph in paragraphs:
        counts = defaultdict(int)

        for sentence in paragraph.sentences:
            tree = _build_dependency_tree(
                sentence
            )

            if tree is not None:
                counts[trees.intern(tree)] += 1

        paragraph_trees.append(counts)

    total, pairs = 0.0, 0

    # FV3 fix: SYNSTRUTt compares sentence pairs ACROSS paragraphs,
    # not aggregated paragraph structures.
    for paragraph_i in range(
        len(paragraph_trees) - 1
    ):
        for paragraph_j in range(
            paragraph_i + 1,
            len(paragraph_trees)
        ):
            for a, count_a in paragraph_trees[paragraph_i].items():
                for b, count_b in paragraph_trees[paragraph_j].items():
                    total += count_a * count_b * trees.similarity(a, b)
                    pairs += count_a * count_b

    return total / pairs if pairs else np.nan

# Fix C2 (2026-10-28), against UD German GSD + HDT (all parts): the gold marks negation as Polarity=Neg ("nicht" 4,547) and
# PronType=Neg ("kein" 1,299, "nichts" 216, "niemand" 46, "keinerlei" 33, "keineswegs" 3); the list had all of these except
# "keinerlei", now added. Its other entries (nie, niemals, nirgends, nirgendwo, weder, keinesfalls, mitnichten, nein) are
# negations the gold leaves unmarked; they parallel English's never, nowhere, neither and stay.
_DE_NEGATION_LEMMAS = {
    "nicht",
    "kein",
    "keiner",
    "keinerlei",
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


def _is_adverbial_modifier(token: Token, lang: str) -> bool:
    # German: UD advmod since the UD port (2026-10-28; TIGER: MO), as for English
    if (token.dep_type or "").strip().lower() != "advmod":
        return False
    if token.pos_coarse == "ADV":
        return True
    # a German adjective used as an adverb ("er läuft schnell": ADJD as advmod) heads an adverbial phrase as an English -ly
    # adverb does (before only ADV, as upstream: 705 such phrases against 3,478 in 200 German texts)
    return lang == "de" and token.pos_coarse == "ADJ" and token.pos_value == "ADJD"


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


# Passive (DRPVAL): one rule for both languages since the UD port (2026-10-28). A sentence is passive when it has a passive
# auxiliary or subject: UD aux:pass, nsubj:pass, csubj:pass (German GSD: werden and the "sein" passive, "ist geschlossen";
# English ClearNLP until its UD port of 2026-10-29: auxpass, nsubjpass, csubjpass). TIGER had no passive relation, so German followed the head chain of a
# participle (VVPP) up to "werden", or to "sein" when the participle was a predicate (PD); removed with the agent check
# (SBP, "durch" as MO), which nothing called.
def _passive_status(sentence: Sentence) -> Tuple[bool, bool]:
    passive_deps = {"aux:pass", "nsubj:pass", "csubj:pass"}   # ClearNLP's auxpass, nsubjpass, csubjpass until 2026-10-29
    agent_deps = {"obl:agent"}

    is_passive = any(
        (token.dep_type or "").lower() in passive_deps
        for token in sentence.tokens
    )

    has_agent = any(
        (token.dep_type or "").lower() in agent_deps
        for token in sentence.tokens
    )

    return is_passive, has_agent


def _de_stative_passive(sentence: Sentence) -> bool:
    """German "sein" + participle (Zustandspassiv, "ist geschlossen"): "sein" as cop or aux:pass on a word tagged VVPP. The old
    TIGER rule counted it (participle as PD of "sein"), UD German HDT marks it aux:pass, Stanza (trained on GSD) often reads it as
    copula + adjective. Against the gold (2026-10-28), passive sentences on Stanza's analysis: agreement
    GSD .970 -> .968, HDT .943 -> .962 (the old pipeline .960 / .968)."""
    tokens = sentence.tokens
    for token in tokens:
        if (token.lemma or "").lower() == "sein" and (token.dep_type or "").lower() in ("cop", "aux:pass"):
            h = token.head_index
            if h is not None and 0 <= h < len(tokens) and tokens[h].pos_value == "VVPP":
                return True
    return False


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

    # English: an infinitive is a VB with UD VerbForm=Inf, i.e. every infinitive form, as the German
    # rule counts every infinitive tag: to-infinitives, bare infinitives after modals and do-support, and after let / make /
    # help; imperatives are VerbForm=Fin. The old rule came from spaCy, where the modal hangs on the VB; in UD a copula or
    # auxiliary "be" / "have" after a modal hangs on the predicate with the modal ("can be happy", "must be done"), so those
    # were missed, as was a split "to" ("to really leave"; 233 of the 300 infinitives of the 20 English test texts were
    # found). Requests without VerbForm (other UD parsers): "to" as mark, or a modal / do auxiliary on the verb or, for a
    # copula or auxiliary verb, on its head left of it.
    if token.pos_value != "VB":
        return False
    verbform = (token.morph_verbform or "").strip()
    if verbform:
        return verbform == "Inf"
    toks = sentence.tokens
    lower = lambda t: (t.lemma or t.text or "").strip().lower()
    marks = lambda t: (t.dep_type or "").strip().lower() == "mark" and lower(t) == "to"
    auxiliary = lambda t: (t.pos_coarse == "AUX" and (t.pos_value == "MD" or lower(t) == "do")
                           and (t.dep_type or "").strip().lower() in {"aux", "aux:pass"})
    if any(i != token_index and t.head_index == token_index and (marks(t) or auxiliary(t)) for i, t in enumerate(toks)):
        return True
    head = token.head_index
    if (token.dep_type or "").strip().lower() in {"cop", "aux", "aux:pass"} and head is not None and head != token_index:
        return any(i < token_index and i != head and t.head_index == head and (marks(t) or auxiliary(t))
                   for i, t in enumerate(toks))
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
        "neg_count": 0,
        "gerund_count": 0,
        "infinitive_count": 0,
        "has_pos_annotations": all(
            _has_complete_pos_annotations(sentence)
            for sentence in sentences
        ) if sentences else False,
        "has_fine_pos_annotations": all(
            _has_complete_fine_pos_annotations(sentence)
            for sentence in sentences
        ) if sentences else False,
        "has_dependency_annotations": all(
            _has_complete_dependency_annotations(sentence)
            for sentence in sentences
        ) if sentences else False,
    }

    for sentence in sentences:
        # DRAP: phrase-level count from dependency hierarchy.
        count_metrics_dict["adverbial_phrase_count"] += _count_adverbial_phrases(
            sentence, lang
        )

        # DRPVAL: agentless passive voice density, as the index is named and described in Coh-Metrix 3.0 ("Agentless passive
        # voice density"; McNamara et al. 2014). Code review 2026-10-31: until then every passive sentence counted (TTLab's
        # documentation counted its example with "vom Hund"), and the agent check was computed and dropped. A sentence counts
        # once, if it is passive and has no agent (UD obl:agent: "vom Autor", "by the committee").
        is_passive, has_agent = _passive_status(sentence)
        if lang == "de" and not is_passive:
            is_passive = _de_stative_passive(sentence)

        if is_passive and not has_agent:
            count_metrics_dict["passive_sentences"] += 1

        for i, token in enumerate(sentence.tokens):
            if not token.is_punct:
                count_metrics_dict["total_tokens"] += 1

            # Legacy lexical counters.
            if token.pos_coarse == "VERB":
                count_metrics_dict["verb_count"] += 1
            elif token.pos_coarse == "ADV":
                count_metrics_dict["adverb_count"] += 1
            elif token.pos_coarse == "ADP" and (token.dep_type or "").strip().lower() != "compound:prt":
                # a separable verb particle (German "ruft ... an", English "give up") is tagged ADP but heads no
                # prepositional phrase
                count_metrics_dict["prep_count"] += 1

            # DRVP approximation:
            # constituency VP projections are approximated by verbal heads.
            # AUX is intentionally included; diagnostic AUX+VERB constructions
            # therefore contribute two verbal projections instead of one.
            if token.pos_coarse in {"VERB", "AUX"}:
                count_metrics_dict["verb_phrase_count"] += 1

            lemma = (token.lemma or token.text or "").lower()

            # DRNEG: the negation lists.
            # German since the UD port (2026-10-28): the lexical list alone; UD has no negation relation ("nicht" is advmod with
            # Polarity=Neg), and TIGER's NG was "nicht", which the list has.
            # English since its UD port (2026-10-29): the list alone too (ClearNLP's neg was "not" / "n't", lemma "not", in it).
            is_negation = lemma in (_DE_NEGATION_LEMMAS if lang == "de" else _EN_NEGATION_LEMMAS)
            if is_negation:
                count_metrics_dict["neg_count"] += 1

            # DRGERUND (English): every -ing verb form (Penn tag VBG; decision 2026-11-02). Book ch. 4: "the relative frequency of
            # the use of the gerund (DRGERUND; in its -ing form)"; so progressive participles and participial modifiers count
            # too: the index measures -ing forms, not only gerunds in the narrow sense (on the UD English test sets every VBG
            # gives 342 / 506 words where the gold has 93 / 168 gerunds, EWT / GUM; Maxim Konca's count).
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
    if not metrics["has_pos_annotations"]:
        return None
    if not noun_chunks and any(
        token.pos_coarse in {"NOUN", "PROPN", "PRON"}
        for sentence in sentences
        for token in _non_punct_tokens(sentence)
    ):
        # With an observed nominal candidate, an empty noun-chunk layer is not
        # evidence of zero noun phrases. Verbless/non-nominal inputs can still
        # produce a genuine zero without a global noun-chunk requirement.
        return None
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
    if not metrics["has_pos_annotations"]:
        return None
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
    if not (metrics["has_pos_annotations"]
            and metrics["has_dependency_annotations"]):
        return None
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
    if not metrics["has_pos_annotations"]:
        return None
    return _incidence(metrics["prep_count"], metrics["total_tokens"])

# LAY: How many agentless passive constructions occur per 1,000 words?
# ↑ Higher = more agentless passive/impersonal constructions.
# Dependency-based approximation of the Coh-Metrix DRPVAL index.
def cm_drpval(
    sentences: List[Sentence],
    noun_chunks: List[NounChunk],
    lang: str,
    metrics: Optional[Dict[str, int]] = None
) -> Optional[float]:
    if metrics is None:
        metrics = _count_metrics(sentences, noun_chunks, lang)
    if not (metrics["has_pos_annotations"]
            and metrics["has_dependency_annotations"]):
        return None
    return _incidence(
        metrics["passive_sentences"],
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
    if not metrics["has_dependency_annotations"]:
        return None
    return _incidence(metrics["neg_count"], metrics["total_tokens"])

# LAY: How many English gerunds ("-ing" verb forms) per 1,000 words?
# Language-limited: DE=0 (German has no gerund equivalent —).
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

    # -ing forms are read from the Penn tag (VBG), so the fine POS tags are needed
    if not metrics["has_fine_pos_annotations"]:
        return None

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
    if not (metrics["has_fine_pos_annotations"]
            and metrics["has_dependency_annotations"]):
        return None
    return _incidence(metrics["infinitive_count"], metrics["total_tokens"])

def _incidence(count, total_words):
    # Coh-Metrix convention (Ch. 4 §Word information): "relative frequency of
    # each word category by counting the number of instances of the category
    # per 1,000 words of text, called incidence scores." Used for all DR*, WRD*
    # (noun/verb/adj/adv/pronoun), and Situation Model verb incidences
    # (SMCAUSv, SMCAUSvp, SMINTEp — see H15). A missing word denominator is
    # not a calculation and therefore becomes None at the final index.
    return (count / total_words) * 1000 if total_words > 0 else None

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
) -> Dict[str, Optional[float]]:

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

    # Without a complete coarse-POS layer an apparent zero category count is
    # not an observation. Keep all POS-incidence end values non-computable.
    if not sentences or not all(
        _has_complete_pos_annotations(sentence)
        for sentence in sentences
    ):
        return {key: None for key in counters}

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

            elif pos == "PRON" or (pos == "DET" and token.pos_value == "PPOSAT"):
                # German possessive determiners ("mein", "unser"; STTS PPOSAT, UPOS DET) count like the English possessives
                # ("my", "our"; UPOS PRON), with the possessor's person and number, as the comment of WRDPRP1s says (I/me/my —
                # ich/mir/mein). Code review 2026-10-31: before, UPOS DET kept them out (24.2 per 1,000 words in German CLEAR),
                # and German possessive pronouns ("meiner", PPOSS) took the agreement number instead of the possessor's.
                counters["pronoun_total"] += 1

                possessive = token.pos_value in ("PPOSAT", "PPOSS")
                person = _normalize_morph_person(
                    _possessor_person(token) if possessive else token.morph_person
                )

                number = _normalize_morph_number(
                    _possessor_number(token) if possessive else token.morph_number
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

# The possessor's number of a German possessive: UD Number[psor] where the annotator gives it (Stanza: "unser" Plur, "mein"
# Sing; none for the ambiguous "ihr", her / their); requests without it (made before 2026-10-31): the
# unambiguous lemmas
_DE_POSSESSOR_NUMBER = {"mein": "Sing", "dein": "Sing", "sein": "Sing", "unser": "Plur", "euer": "Plur"}
# its person: from the lemma, which fixes it for every possessive but "ihr" (her / their / formal Ihr), there from the
# annotator (Stanza tagged "deine" Person=1)
_DE_POSSESSOR_PERSON = {"mein": "1", "unser": "1", "dein": "2", "euer": "2", "sein": "3"}


def _possessor_person(token: Token) -> Optional[str]:
    return _DE_POSSESSOR_PERSON.get((token.lemma or "").strip().lower()) or token.morph_person


def _possessor_number(token: Token) -> Optional[str]:
    psor = (token.morph_number_psor or "").strip()
    return psor or _DE_POSSESSOR_NUMBER.get((token.lemma or "").strip().lower())


def _wrd_precompute(
    sentences: List[Sentence]
) -> Dict[str, Optional[float]]:
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
#       not directly comparable across languages..
#   (d) CELEX word-frequency indices are stubbed None; Wikipedia-sample
#       alternatives WRDFRQ*_wiki10000 are emitted..
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
# FV4 Fix: Treat 0 values as None values, since they represent a missing rating inside the MRC databse.
@lru_cache(maxsize=2)
def _load_mrc_database(
    lang: str
) -> Dict[str, Dict[str, Optional[float]]]:

    if lang == "de":
        filepath = str(RESOURCES) + "/mrc_psycholinguistic_database_de.csv"
    else:
        filepath = str(RESOURCES) + "/mrc_psycholinguistic_database.csv"

    def _parse_mrc_rating(value) -> Optional[float]:
        if value is None:
            return None

        value = str(value).strip()

        if not value:
            return None

        try:
            rating = float(value)
        except (TypeError, ValueError):
            return None

        # MRC uses 0 as a missing-value sentinel.
        if rating <= 0:
            return None

        # Defensive check for NaN / infinity.
        if not math.isfinite(rating):
            return None

        return rating

    mrc_dict = {}

    with open(
        filepath,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(
            file,
            delimiter=","
        )

        for row in reader:

            word = (
                row.get("Word") or ""
            ).strip().lower()

            if not word:
                continue

            # a word with several rows keeps the first rating of each kind (the last row won, and
            # where it was unrated "voice", "sunshine", "verdict" ... lost their familiarity and imageability)
            ratings = {
                "AoA": _parse_mrc_rating(
                    row.get(
                        "Age of Acquisition Rating"
                    )
                ),

                "Familiarity": _parse_mrc_rating(
                    row.get(
                        "Familiarity"
                    )
                ),

                "Concreteness": _parse_mrc_rating(
                    row.get(
                        "Concreteness"
                    )
                ),

                "Imageability": _parse_mrc_rating(
                    row.get(
                        "Imageability"
                    )
                ),

                "Meaningfulness": _parse_mrc_rating(
                    row.get(
                        "Meaningfulness: Coloradao Norms"
                    )
                ),
            }
            known = mrc_dict.setdefault(word, ratings)
            for key, value in ratings.items():
                if known[key] is None:
                    known[key] = value

    return mrc_dict

# FV1 fix: Removed old mrc_dict creation
#mrc_dict = _load_mrc_database

# German word norms (README): native norms where they exist, one source per index,
# looked up by lemma (exact case first, since Köper keeps nouns and verbs apart, "Essen" / "essen"; then lower case). The
# translated MRC table rated 12-31 % of the content-word tokens of the fix sample as looked up before (lower-cased word form).
# - Concreteness, Imageability: Köper & Schulte im Walde (2016, LREC), AbstConc and IMG (0-10; automatically extended from
#   human ratings to 351,617 lemmas): 88.7 % of the tokens; on shared words rho .83 / .72 with the translated MRC values.
# - AoA: Birchenough, Davies & Connelly (2016, BRM), rated age of acquisition in years, 3,259 words: 36.0 %; rho .71.
# - Familiarity, Meaningfulness: no native norms; the translated MRC table, now by lemma (31 -> 43 %, 24 -> 34 %).
_GERMAN_NORM_SOURCES = {"AoA": "birchenough", "Familiarity": "mrctranslate", "Concreteness": "koeper",
                        "Imageability": "koeper", "Meaningfulness": "mrctranslate"}


@lru_cache(maxsize=1)
def _load_german_norms() -> Dict[str, Dict[str, Optional[float]]]:
    import gzip
    norms: Dict[str, Dict[str, Optional[float]]] = defaultdict(dict)
    for word, ratings in _load_mrc_database("de").items():
        norms[word]["Familiarity"] = ratings["Familiarity"]
        norms[word]["Meaningfulness"] = ratings["Meaningfulness"]
    if settings.koeper_path and Path(settings.koeper_path).is_file():
        with gzip.open(settings.koeper_path, "rt", encoding="utf-8") as fh:
            next(fh)
            for line in fh:
                word, abst_conc, _, img, _ = line.rstrip("\n").split("\t")
                norms[word]["Concreteness"] = float(abst_conc)
                norms[word]["Imageability"] = float(img)
    else:
        logger.warning("Köper norms not found at \"%s\": WRDCNCc and WRDIMGc will return None for German", settings.koeper_path)
    if settings.birchenough_path and Path(settings.birchenough_path).is_file():
        with open(settings.birchenough_path, encoding="latin-1", newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    aoa = float(row["AoAestimate"])
                except (TypeError, ValueError):
                    continue
                norms[row["Word"].strip()].setdefault("AoA", aoa)
    else:
        logger.warning("Birchenough AoA norms not found at \"%s\": WRDAOAc will return None for German", settings.birchenough_path)
    return dict(norms)


# English word norms (2026-10-29: one source per index, by coverage):
# - Concreteness: Brysbaert, Warriner & Kuperman (2014, BRM), Conc.M (1-5), 37,058 single words: 87.3 % of the CLEAR originals'
#   word tokens (word forms) against MRC's 70.8 %.
# - AoA: Kuperman, Stadthagen-Gonzalez & Brysbaert (2012, BRM), rated age of acquisition in years, 31,105 words: 76.4 % against
#   MRC's 8.1 %.
# - Imageability, Familiarity, Meaningfulness: the original MRC database (the Glasgow norms cover 22.0 % only); German familiarity
#   and meaningfulness are the translated MRC, so both languages share these data.
# Looked up by lemma since 2026-10-29 (before: lower-cased word forms), exact case first, then lower case, as German.
_ENGLISH_NORM_SOURCES = {"AoA": "kuperman", "Familiarity": "", "Concreteness": "brysbaert", "Imageability": "", "Meaningfulness": ""}


@lru_cache(maxsize=1)
def _load_english_norms() -> Dict[str, Dict[str, Optional[float]]]:
    norms: Dict[str, Dict[str, Optional[float]]] = defaultdict(dict)
    for word, ratings in _load_mrc_database("en").items():
        for key in ("Imageability", "Familiarity", "Meaningfulness"):
            norms[word][key] = ratings[key]
    for key, path, word_col, value_col, name, index in (
            ("Concreteness", settings.brysbaert_path, "Word", "Conc.M", "Brysbaert concreteness norms", "WRDCNCc"),
            ("AoA", settings.kuperman_path, "Word", "Rating.Mean", "Kuperman AoA norms", "WRDAOAc")):
        if not path or not Path(path).is_file():
            logger.warning("%s not found at \"%s\": %s will return None for English", name, path, index)
            continue
        with open(path, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if key == "Concreteness" and row.get("Bigram") not in ("0", None):
                    continue                                 # two-word expressions
                try:
                    value = float(row[value_col])
                except (TypeError, ValueError):
                    continue                                 # Kuperman's unrated words ("#N/A")
                norms[row[word_col].strip()][key] = value
    return dict(norms)


def _load_norms(lang: str) -> Dict[str, Dict[str, Optional[float]]]:
    # other languages: no norms (they got the English MRC table)
    return _load_german_norms() if lang == "de" else _load_english_norms() if lang == "en" else {}


def _get_content_lemmas(sentences: List[Sentence]) -> List[Tuple[str, bool]]:
    """(lemma, may be looked up in lower case) of the content words (same selection as _get_content_words: content UPOS, a word
    with a letter). Proper nouns are not looked up in lower case (English "US" was rated as "us",
    "Turkey" as "turkey"; German "Essen" the city as "essen")."""
    return [((token.lemma or token.text), token.pos_coarse != "PROPN") for sent in sentences for token in sent.tokens
            if token.pos_coarse in ud_content_pos and _is_word_text(token.text)]

# English contraction stems that SoMaJo (en_PTB) splits off, with the word they stand for ("ca" of
# "can't" got the frequency of the abbreviation "ca", 2 per million against 5,247 for "can")
_EN_SPLIT_STEMS = {("ca", "n't"): "can", ("wo", "n't"): "will", ("sha", "n't"): "shall", ("gon", "na"): "going",
                   ("wan", "na"): "want"}


def _word_forms(toks: List[Token]) -> List[str]:
    """the tokens' texts, a split English contraction stem replaced by its word"""
    out = []
    for k, t in enumerate(toks):
        nxt = toks[k + 1] if k + 1 < len(toks) else None
        stem = (t.text.lower(), nxt.text.lower().replace("’", "'")) if nxt is not None and nxt.begin == t.end else None
        out.append(_EN_SPLIT_STEMS.get(stem, t.text))
    return out


def _get_content_words(sentences: List[Sentence]):
    words = [_word_forms(sent.tokens) for sent in sentences]
    poses = [[token.pos_coarse for token in sent.tokens] for sent in sentences]

    words_flatten = [word for sublist in words for word in sublist]
    poses_flatten = [pos for sublist in poses for pos in sublist]

    content_words = [
        word.lower()
        for word, pos
        in zip(words_flatten, poses_flatten)
        if pos in ud_content_pos and _is_word_text(word)
    ]

    return content_words, words, poses

def _get_content_words_per_sentence(sentences: List[Sentence]):
    words = [_word_forms(sent.tokens) for sent in sentences]
    poses = [[token.pos_coarse for token in sent.tokens] for sent in sentences]

    content_words = []
    for swords, sposes in zip(words, poses):
        content_words_sent = [
            word.lower()
            for word, pos
            in zip(swords, sposes)
            if pos in ud_content_pos and _is_word_text(word)
        ]
        content_words.append(content_words_sent)

    return content_words, words, poses

def _norm_forms(word: str, lower: bool, german: bool) -> List[str]:
    """the spellings a norm table is searched under, in order: the word, in lower case (German lemmas, fix B1; not for proper
    nouns), and for German the new spelling of an old one ("Rußland" -> "Russland", GermaNet's list) and all of these with
    "ss" for "ß" (the Köper table writes "gross", "Strasse", so "groß", "heißen", "Straße" were never
    found)"""
    forms = [word] + ([word.lower()] if lower else [])
    if not german:
        return forms
    new = _GN_OLD_SPELLINGS.get(word)
    if new:
        forms += [new] + ([new.lower()] if lower else [])
    return forms + [f.replace("ß", "ss") for f in forms if "ß" in f]


def _average_rating(words, mrc_dict, key):
    # words: strings, or (word, may be looked up in lower case, German spellings) from _norm_words; the first spelling with a
    # rating counts (_norm_forms)
    ratings = []
    for item in words:
        w, lower, german = item if isinstance(item, tuple) else (item, True, False)
        r = None
        for form in _norm_forms(w, lower, german):
            r = (mrc_dict.get(form) or {}).get(key)
            if r is not None:
                break
        if r is not None:
            ratings.append(r)
    if not ratings:
        return None
    return sum(ratings) / len(ratings)


def _norm_words(sentences: List[Sentence], lang: str) -> list:
    """the words the norm indices look up: (lemma, may be looked up in lower case, German spellings) of the content words
    (German since fix B1, English since 2026-10-29), else the lower-cased content words as before"""
    if lang in ("de", "en"):
        return [(w, lower, lang == "de") for w, lower in _get_content_lemmas(sentences)]
    return _get_content_words(sentences)[0]

# LAY: Average age at which content words are typically learned (MRC norms).
# ↑ Higher = later-acquired, harder words.
# FV1 Fix: Reworked Function for german language Support.
def cm_wrdaoac(sentences: List[Sentence], lang: str, mrc_dict: Optional[Dict[str, float]] = None) -> Optional[float]:
    # H8: MRC Psycholinguistic Database is English-only.
    if mrc_dict is None:
        mrc_dict = _load_norms(lang)
    content_words = _norm_words(sentences, lang)
    if not content_words:
        return None
    return _average_rating(content_words, mrc_dict, 'AoA')

# LAY: Average subjective familiarity rating of content words (MRC norms).
# ↑ Higher = more familiar, easier words.
# FV1 Fix: Reworked Function for german language Support.
def cm_wrdfamc(sentences: List[Sentence], lang: str, mrc_dict: Optional[Dict[str, float]] = None) -> Optional[float]:
    if mrc_dict is None:
        mrc_dict = _load_norms(lang)
    content_words = _norm_words(sentences, lang)
    if not content_words:
        return None
    return _average_rating(content_words, mrc_dict, 'Familiarity')

# LAY: Average concreteness of content words (MRC norms).
# ↑ Higher = more concrete, sensory words.
# FV1 Fix: Reworked Function for german language Support.
def cm_wrdcncc(sentences: List[Sentence], lang: str, mrc_dict: Optional[Dict[str, float]] = None) -> Optional[float]:
    if mrc_dict is None:
        mrc_dict = _load_norms(lang)
    content_words = _norm_words(sentences, lang)
    if not content_words:
        return None
    return _average_rating(content_words, mrc_dict, 'Concreteness')

# LAY: Average imageability of content words (MRC norms).
# ↑ Higher = easier to form a mental picture.
# FV1 Fix: Reworked Function for german language Support.
def cm_wrdimgc(sentences: List[Sentence], lang: str, mrc_dict: Optional[Dict[str, float]] = None) -> Optional[float]:
    if mrc_dict is None:
        mrc_dict = _load_norms(lang)
    content_words = _norm_words(sentences, lang)
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
        mrc_dict = _load_norms(lang)
    content_words = _norm_words(sentences, lang)
    if not content_words:
        return None
    return _average_rating(content_words, mrc_dict, 'Meaningfulness')

def _get_lexical_lookup_form(
    token: Token,
    lang: str
) -> Optional[str]:
    if token.pos_coarse not in ud_content_pos:
        return None

    if not _is_word_text(token.text):   # before str.isalpha, so hyphenated compounds never got here
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

@lru_cache(maxsize=16384)
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
            # _get_lexical_lookup_form. Keep the GermaNet lookup
            # case-sensitive here to avoid conflating lexically distinct
            # German forms that differ only in capitalization.
            return len(
                germanet.get_synsets_by_orthform(word)
            )
        except Exception as e:   # logged since 2026-10-31; before an error looked like "word not covered"
            logger.warning("GermaNet polysemy lookup of %r failed: %s", word, e)
            return None

    return None

@lru_cache(maxsize=16384)
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
        except Exception as e:   # logged since 2026-10-31
            logger.warning("GermaNet hypernym lookup of %r failed: %s", word, e)
            return None
    return None

# LAY: Average polysemy: how many senses each content word has.
# ↑ Higher = more ambiguous vocabulary. Approximate for DE (GermaNet) —.
def cm_wrdpolc(
    sentences: List[Sentence],
    lang: str
) -> Optional[float]:
    polysemies = []

    for lemma, pos in _iter_content_lemmas(sentences, lang):
        polysemy = _get_polysemy(
            lemma,
            lang=lang
        )

        # fix B3: a lemma GermaNet does not list, by its current spelling or as a compound of one sense
        if lang == "de" and germanet is not None and not polysemy:
            fallback = _de_germanet_fallback(lemma, pos)
            if fallback is not None:
                polysemy = _get_polysemy(fallback[0], lang=lang) if fallback[1] == "old spelling" else 1

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

        # fix B3: a lemma GermaNet does not list, by its current spelling or with its compound head's depth
        if depth is None and lang == "de" and germanet is not None:
            fallback = _de_germanet_fallback(lemma, pos)
            if fallback is not None:
                head_depth = _get_max_hypernym_depth(fallback[0], pos=pos_for_lookup, lang=lang)
                if head_depth is not None:
                    depth = head_depth    # head depth + 0 (2026-10-28, see the fix B3 comment at _GN_NOUN_FORMS)

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
# ↑ Higher = more specific nouns (deeper in the WordNet tree). Approximate for DE —.
def cm_wrdhypn(sentences: List[Sentence], lang: str) -> Optional[float]:
    return _calc_wrdhyp(sentences, lang)["WRDHYPn"]

# LAY: Average hypernymy depth of verbs.
# ↑ Higher = more specific verbs. Approximate for DE (GermaNet) —.
def cm_wrdhypv(sentences: List[Sentence], lang: str) -> Optional[float]:
    return _calc_wrdhyp(sentences, lang)["WRDHYPv"]

# LAY: Average hypernymy depth of nouns and verbs combined.
# ↑ Higher = more specific content words overall. Approximate for DE —.
def cm_wrdhypnv(sentences: List[Sentence], lang: str) -> Optional[float]:
    return _calc_wrdhyp(sentences, lang)["WRDHYPnv"]

# The connective lists of duui-coh-metrix 0.2.0 (variant CNC*_LLMlist, decision 2026-11-02), counted as in 0.2.0 with
# `_count_connectives_llmlist` on the text. The main CNC* indices count DiMLex connectives (see "CONNECTIVES from DiMLex"
# below). Problems found (German): "aus" (a preposition) was 52%
# of the causal hits, "da"/"bis"/"seit"/"während" mostly not connectives, frequent connectives missing
#.
# List has been generated by ChatGPT 4o based on the Coh-Metrix index definitions
connectives_llmlist = {
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
    # Preserve punctuation/newline boundaries so a multi-word connective can
    # never be assembled across sentence or clause boundaries. At the same
    # time, punctuation attached to a word is split off, so "run,but" still
    # exposes the valid single-word connective "but".
    boundary = "__CONNECTIVE_BOUNDARY__"
    normalized = []
    for match in _re_conn.finditer(
        r"\w+(?:['’]\w+)*|[^\w\s]+|\s+",
        (text or "").lower(),
    ):
        item = match.group(0)
        if item.isspace():
            if "\n" in item or "\r" in item:
                normalized.append(boundary)
        elif _re_conn.fullmatch(r"\w+(?:['’]\w+)*", item):
            normalized.append(item)
        else:
            normalized.append(boundary)
    return normalized

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

    for category, language_lists in connectives_llmlist.items():
        for expression in language_lists.get(lang, set()):

            pattern = tuple(
                token
                for token in _normalize_tokens_for_connectives(expression)
                if token != "__CONNECTIVE_BOUNDARY__"
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

def _count_connectives_llmlist(
    text: str,
    lang: str,
    total_words: int
) -> Dict[str, float]:
    # the 0.2.0 connective counting (variant CNC*_LLMlist), unchanged

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
# CONNECTIVES from DiMLex (2026-10-27: new lists with a rule that sorts out the wrong readings)
#
# Lexicon: DiMLex, the German connective lexicon (github.com/discourse-lab/dimlex, CC BY-NC-SA 4.0), read from
# settings.dimlex_path (DUUI_COH_METRIX_DIMLEX_PATH; licensed like GermaNet: never copied into this folder or an image).
# Without it, and for languages other than German (no DiMLex-Eng here), the CNC* indices and SMCAUSvp/SMCAUSr are None.
#
# Which candidate is a connective: the rules of GAT's connectives.py (2026-10-06; `load_dimlex`, `candidates`, `decide`),
# ported to this module's tokens (STTS tags, heads within the sentence; TIGER dependencies until the UD port of 2026-10-28,
# UD German GSD conventions as Stanza writes them since), with the two changes below:
# - subordinator (DiMLex subj, postp) tagged KOUS/KOUI/PWAV: only if its clause is an adverbial, not a complement (UD ccomp,
#   csubj, csubj:pass, and acl for dass/ob/wie/wo/wann: "die Frage, ob ..."; TIGER was oc, sb, op, ag, and re for those forms);
#   a coordinated clause (conj) takes the function of its first conjunct (new with UD);
# - coordinator tagged KON: sentence-initially or between clauses (UD: the conjunct it is attached to as cc is a verb or the
#   predicate of a copular clause; TIGER: a cj child is a verb); "denn" always;
# - preposition: never (prepositions take noun phrases, not propositions: "aus Asien", "bis zu", "seit 2010");
# - entry without other readings (DiMLex non_conn = 0): always;
# - ambiguous adverb: weighted with DiMLex's share of connective readings in its annotated examples ("auch" .75, "nur" 0);
# - phrase or two-part form ("aus diesem Grund", "zwar ... aber"): when all parts occur in order in the sentence.
# An occurrence matched by several entries of the same spelling counts once, with the entry of the highest weight.
# Two changes to GAT's rule (2026-10-27):
# - PWAV (relative/interrogative adverb: "wo", "wie", "warum", "wonach", "wodurch") counts only when its clause is an adverbial
#   (UD advcl, TIGER mo) or a continuative relative clause on a verb ("..., wodurch er gewann"; UD acl or acl:relcl, TIGER rc,
#   with a verb, or a copular predicate, as head), not in a question
#   ("Wo investiert er?"), an indirect question or a relative clause on a noun ("Beispiele, wo ..."; "Berichte, wonach ...").
#   The tag does not separate the readings, so it is weighted like an ambiguous adverb ("wie" .02, "wo" .075, "wodurch" .72).
#   GAT counted every PWAV that is not a complement with weight 1 (its docstring asks for mo, its code does not).
# - Ambiguous adverb: not counted as a verb particle or predicate ("dabei haben", "dabei sein", DiMLex's non-connective example
#   of "dabei"; UD compound:prt, or the adverb has a cop child; TIGER svp, pd), nor when it modifies a word instead of the
#   clause (its head is no verb: "noch nicht", "nur noch", "noch einige Jahre"; in UD the predicate of a copular clause counts
#   as the clause: "Das ist deshalb wichtig"), unless DiMLex marks the entry as a possible focus particle (focuspart 1: "auch
#   Kinder").
#
# Categories: DiMLex's PDTB 3 senses (level 2, shares of its corpus frequencies) mapped to the Coh-Metrix categories
# (CONNECTIVE_SENSES). A connective counts in a category, with its weight, when at least CONNECTIVE_MAJORITY of its senses fall
# into that category (its majority reading). CNCTemp takes temporal connectives in a conjunction reading (tagged
# KOUS/KOUI/PWAV/KON: "als", "nachdem", "bevor"), CNCTempX those in any other reading (adverbs, phrases: "danach", "dann").
# CNCAll: every counted connective, whatever its sense (also condition, purpose, manner, ...).
# ============================================================================

CONNECTIVE_SENSES = {
    "CNCCaus": {"Contingency.Cause"},
    "CNCLogic": {"Expansion.Conjunction", "Expansion.Disjunction"},
    "CNCADC": {"Comparison.Contrast", "Comparison.Concession"},
    "CNCTemp": {"Temporal.Synchronous", "Temporal.Asynchronous"},
    "CNCTempX": {"Temporal.Synchronous", "Temporal.Asynchronous"},
    "CNCAdd": {"Expansion.Conjunction"},
    "CNCPos": {"Expansion.Conjunction", "Expansion.Equivalence"},
    "CNCNeg": {"Comparison.Contrast", "Comparison.Concession", "Contingency.Negative-condition"},
}
CONNECTIVE_MAJORITY = 0.5
_DIMLEX_SUBORDINATOR_TAGS = {"KOUS", "KOUI", "PWAV"}
_DIMLEX_ADVERB_TAGS = {"ADV", "PROAV", "PAV", "KON"}
_DIMLEX_ADVERB_CATS = {"konnadv", "padv", "einzel", "v2emb", "apci", "appo"}
_DIMLEX_COMPLEMENT_DEPS = {"ccomp", "csubj", "csubj:pass"}
# "acl" marks a complement clause of a noun for these ("die Frage, ob ...", "die Tatsache, dass ..."; TIGER: "re" for the
# extraposed ones, "Es ist wichtig, dass ..."), otherwise a relative clause
_DIMLEX_CORRELATE_COMPLEMENTS = {"dass", "ob", "wie", "wo", "wann"}
_CONJUNCTION_TAGS = _DIMLEX_SUBORDINATOR_TAGS | {"KON"}


def _load_dimlex(path) -> Optional[List[dict]]:
    """DiMLex entries as GAT's connectives.load_dimlex reads them (no download: a missing file gives None)."""
    import xml.etree.ElementTree as ET
    from collections import Counter
    if not path or not Path(path).is_file():
        logger.warning("DiMLex not found at \"%s\": the CNC* indices and SMCAUSvp, SMCAUSr will return None", path)
        return None
    entries = []
    for e in ET.parse(path).getroot().findall(".//entry"):
        cats = {c.text.strip() for c in e.findall("syn/cat") if c.text}
        senses = Counter()
        for s in e.findall(".//pdtb3_relation"):
            if s.get("sense"):
                senses[s.get("sense")] += int(s.get("freq") or 0)
        if senses and sum(senses.values()) == 0:
            senses = Counter({k: 1 for k in senses})
        level2 = Counter()
        for k, v in senses.items():
            level2[".".join(k.split(".")[:2])] += v
        total = sum(level2.values())
        orths = []
        for o in e.findall("orths/orth"):
            # a comma is a word of its own, as in the tokens ("vorausgesetzt, dass", "für den Fall,
            # dass" and 9 more forms kept "vorausgesetzt," as one word and could never match; "dadurch , dass" could)
            parts = [p.text.lower().replace(",", " , ").split() for p in o.findall("part") if p.text]
            orths.append((o.get("type"), parts))
        nc = e.find("ambiguity/non_conn")
        ambiguous = nc is not None and (nc.text or "").strip() == "1"
        p_conn = None
        if ambiguous and nc.get("freq") and nc.get("anno_N") and int(nc.get("anno_N")) > 0:
            p_conn = max(0.0, 1 - int(nc.get("freq")) / int(nc.get("anno_N")))
        fp = e.find("focuspart")
        entries.append({"id": e.get("id"), "word": e.get("word"), "cats": cats,
                        "ambiguous": ambiguous, "p_conn": p_conn,
                        "focuspart": fp is not None and (fp.text or "").strip() == "1",
                        "senses": {k: v / total for k, v in level2.items()} if total else {},
                        "orths": orths})
    return entries


dimlex_entries = _load_dimlex(settings.dimlex_path)
dimlex_sha256 = None
_dimlex_singles: Dict[str, List[dict]] = defaultdict(list)
_dimlex_multi: List[Tuple[dict, str, list]] = []
if dimlex_entries is not None:
    import hashlib as _hashlib
    dimlex_sha256 = _hashlib.sha256(Path(settings.dimlex_path).read_bytes()).hexdigest()
    for _e in dimlex_entries:
        for _typ, _parts in _e["orths"]:
            if _typ == "cont" and len(_parts) == 1 and len(_parts[0]) == 1:
                _dimlex_singles[_parts[0][0]].append(_e)
            if _typ == "discont" or len(_parts[0]) > 1:
                _dimlex_multi.append((_e, _typ, _parts))
    logger.info("DiMLex: %d entries from \"%s\" (sha256 %s)", len(dimlex_entries), settings.dimlex_path, dimlex_sha256[:12])


def _dimlex_candidates(words: List[str], singles: Optional[Dict[str, List[dict]]] = None,
                       multi: Optional[List[Tuple[dict, str, list]]] = None) -> List[dict]:
    """Connective candidates of one sentence: phrases and two-part forms first, then words (GAT's `candidates`). The lexicon
    is German DiMLex unless another one's single-word and multi-word forms are given (English, 2026-10-29)."""
    singles = _dimlex_singles if singles is None else singles
    multi_all = _dimlex_multi if multi is None else multi
    used, out = set(), []

    def find(seq, start=0):
        for i in range(start, len(words) - len(seq) + 1):
            if words[i:i + len(seq)] == seq and not used.intersection(range(i, i + len(seq))):
                return i
        return None

    present = set(words)
    multi = [(e, typ, parts) for e, typ, parts in multi_all
             if present.issuperset(w for p in parts for w in p)]
    multi.sort(key=lambda x: -sum(len(p) for p in x[2]))
    for e, typ, parts in multi:
        if typ == "discont":
            i = find(parts[0])
            j = find(parts[1], i + len(parts[0])) if i is not None else None
            if i is None or j is None:
                continue
            span = list(range(i, i + len(parts[0]))) + list(range(j, j + len(parts[1])))
        else:
            i = find(parts[0])
            if i is None:
                continue
            span = list(range(i, i + len(parts[0])))
        used.update(span)
        out.append({"entry": e, "start": span[0], "form": " ... ".join(" ".join(p) for p in parts),
                    "kind": "two-part" if typ == "discont" else "phrase"})
    for i, w in enumerate(words):
        if i in used or w not in singles:
            continue
        for e in {x["id"]: x for x in singles[w]}.values():
            out.append({"entry": e, "start": i, "form": w, "kind": "word"})
    return out


def _dimlex_decide(tokens: List[Token], c: dict) -> Tuple[float, str]:
    """Weight of the connective reading (1, 0, or DiMLex's share for ambiguous adverbs and PWAV) and why (GAT's `decide`
    with the two changes of 2026-10-27)."""
    if c["kind"] != "word":
        return 1.0, c["kind"]
    e, i = c["entry"], c["start"]
    t = tokens[i]
    cats = e["cats"]

    def head(k):
        h = tokens[k].head_index
        return k if h is None else h

    def children(k):
        return [x for j, x in enumerate(tokens) if j != k and x.head_index == k]

    def clausal(k):
        """a verb, or the predicate of a copular clause (UD makes the predicate the head: "Das ist deshalb wichtig")"""
        return tokens[k].pos_coarse in ("VERB", "AUX") or any(x.dep_type == "cop" for x in children(k))

    def clause_function(k):
        """relation of the clause headed by k; a coordinated clause (conj) takes its first conjunct's"""
        seen = set()
        while tokens[k].dep_type == "conj" and k not in seen and head(k) != k:
            seen.add(k)
            k = head(k)
        return tokens[k].dep_type, k

    first = i == next((j for j, x in enumerate(tokens) if not x.is_punct), 0)
    if t.pos_value in _DIMLEX_SUBORDINATOR_TAGS and cats & {"subj", "postp"}:
        dep, k = clause_function(head(i))
        if dep in _DIMLEX_COMPLEMENT_DEPS or (dep == "acl" and c["form"] in _DIMLEX_CORRELATE_COMPLEMENTS):
            return 0.0, f"complement clause ({dep})"
        if t.pos_value == "PWAV":
            # Not in GAT (2026-10-27): relative/interrogative adverb, see the comment block above.
            if not (dep == "advcl" or (dep in ("acl", "acl:relcl") and clausal(head(k)))):
                return 0.0, f"relative or interrogative clause ({dep})"
            if e["ambiguous"] and e["p_conn"] is not None:
                return e["p_conn"], "relative adverb (weight from DiMLex)"
        return 1.0, "subordinator"
    if t.pos_value == "KON":
        if c["form"] == "denn" or first:
            return 1.0, "coordinator"
        if t.dep_type == "cc" and clausal(head(i)):
            return 1.0, "coordinator (clauses)"
        return 0.0, "coordinator (not clauses)"
    if t.pos_value in ("APPR", "APPO", "APPRART") or cats == {"praep"}:
        return 0.0, "preposition"
    if not e["ambiguous"]:
        return 1.0, "entry without other readings"
    if cats & _DIMLEX_ADVERB_CATS and (t.pos_value in _DIMLEX_ADVERB_TAGS or (t.pos_value == "ADJD" and t.dep_type == "advmod")):
        # Not in GAT (2026-10-27): verb particle, predicate, or modifier of a word, see the comment block above.
        if t.dep_type == "compound:prt" or any(x.dep_type == "cop" for x in children(i)):
            return 0.0, f"verb particle or predicate ({t.dep_type})"
        if not e["focuspart"] and not clausal(head(i)):
            return 0.0, f"modifies a word ({tokens[head(i)].pos_coarse})"
        if e["p_conn"] is None:
            return 1.0, "adverb"
        return e["p_conn"], "adverb (weight from DiMLex)"
    return 0.0, f"other reading ({t.pos_value})"


def _dimlex_sentence_connectives(sentence: Sentence) -> List[Tuple[float, dict, bool]]:
    """(weight, DiMLex entry, conjunction reading) of every counted connective of one sentence."""
    tokens = sentence.tokens
    # DiMLex spells contractions as one word ("im Übrigen", "zum Beispiel"), the UD words split them ("in dem", "zu dem"; UD port,
    # 2026-10-28): the words sharing one token span are matched as the token's surface form, and a match is placed on the first
    # of them
    best = {}
    surface, first_word = [], []
    for j, t in enumerate(tokens):
        if j and t.begin == tokens[j - 1].begin and t.end == tokens[j - 1].end:
            continue
        span = sentence.text[t.begin - sentence.begin:t.end - sentence.begin]
        surface.append((span if span and span != t.text and j + 1 < len(tokens) and tokens[j + 1].begin == t.begin
                        else t.text).lower())
        first_word.append(j)
    for c in _dimlex_candidates(surface):
        c = {**c, "start": first_word[c["start"]]}
        weight, _ = _dimlex_decide(tokens, c)
        key = (c["start"], c["form"])
        if key not in best or weight > best[key][0]:
            best[key] = (weight, c)
    return [(weight, c["entry"], tokens[c["start"]].pos_value in _CONJUNCTION_TAGS)
            for weight, c in best.values() if weight > 0]


# ============================================================================
# ENGLISH CONNECTIVES from DiMLex-Eng (2026-10-29)
#
# Lexicon: DiMLex-Eng (Das, Scheffler, Bourgonje & Stede 2018; github.com/discourse-lab/en_dimlex, CC BY-NC-SA 4.0), read from
# settings.dimlex_en_path. Its senses are PDTB 2 relations with PDTB frequencies; they are mapped to the PDTB 3 level 2 names of
# German DiMLex (_pdtb3_level2), so CONNECTIVE_SENSES serves both languages.
# Which candidate is a connective: German's rules, in UD terms (Stanza's English tags: UPOS, Penn XPOS):
# - subordinator (SCONJ): only if its clause is no complement (ccomp, csubj, csubj:pass; acl for that / whether / if);
# - relative / interrogative adverb (XPOS WRB: when, where, why, how): only in an adverbial clause or a continuative relative
#   clause on a verb, weighted (German PWAV);
# - coordinator (CCONJ): sentence-initially or between clauses;
# - preposition (ADP): never ("since 1990", "after lunch");
# - adverb (ADV): not as a verb particle or predicate, only when it modifies a clause, weighted;
# - phrase or two-part form: when all parts occur in order.
# Weights (option B): German DiMLex has the share of connective readings of its ambiguous entries; DiMLex-Eng has none. They are
# estimated on GUM's discourse annotation (the GUM Discourse Treebank, PDTB-style, in the MISC column of UD English GUM): among
# the occurrences of a word in that reading that pass the rules above, the share GUM marks as an explicit connective (train and
# dev parts; the test part is kept for the evaluation). Built by into
# settings.dimlex_en_weights_path; used where GUM has at least _EN_WEIGHT_MIN_N occurrences, else 1 (as German entries without
# a share). The table derives from the GUM annotations (CC BY 4.0, Zeldes 2017; only the texts of GUM have their own licences,
# some non-commercial, and the table contains no text).
# ============================================================================

_PDTB2_TO_3 = {"temporal.synchrony": "Temporal.Synchronous", "temporal.synchronous": "Temporal.Synchronous",
               "temporal.asynchronous": "Temporal.Asynchronous", "contingency.cause": "Contingency.Cause",
               "contingency.pragmatic cause": "Contingency.Cause", "contingency.condition": "Contingency.Condition",
               "contingency.conditon": "Contingency.Condition", "contingency.pragmatic condition": "Contingency.Condition",
               "comparison.contrast": "Comparison.Contrast", "comparison.pragmatic contrast": "Comparison.Contrast",
               "comparison.concession": "Comparison.Concession", "comparison.pragmatic concession": "Comparison.Concession",
               "expansion.conjunction": "Expansion.Conjunction", "expansion.list": "Expansion.Conjunction",
               "expansion.instantiation": "Expansion.Instantiation", "expansion.alternative": "Expansion.Disjunction",
               "expansion.exception": "Expansion.Exception",
               # level 1 only: Comparison and Temporal can be mapped, since every category holding one of their level-2 senses
               # holds both; Contingency and Expansion stay level 1 (they count in CNCAll only)
               "comparison": "Comparison.Contrast", "temporal": "Temporal.Asynchronous"}
_EN_WEIGHT_MIN_N = 10
_EN_CORRELATE_COMPLEMENTS = {"that", "whether", "if"}


def _pdtb3_level2(sense: str) -> str:
    """PDTB 2 sense (as DiMLex-Eng writes it) -> PDTB 3 level 2 name (German DiMLex's)."""
    low = sense.strip().lower()
    if low.startswith("expansion.restatement"):
        return "Expansion.Equivalence" if low.endswith("equivalence") else "Expansion.Level-of-detail"
    if low.startswith("expansion.alternative.chosen"):
        return "Expansion.Substitution"
    level2 = ".".join(low.split(".")[:2])
    return _PDTB2_TO_3.get(level2, _PDTB2_TO_3.get(low, sense.split(".")[0]))


def _load_dimlex_en(path) -> Optional[List[dict]]:
    import xml.etree.ElementTree as ET
    from collections import Counter
    if not path or not Path(path).is_file():
        logger.warning("DiMLex-Eng not found at \"%s\": the English CNC* indices and SMCAUSvp, SMCAUSr will return None", path)
        return None
    entries = []
    for e in ET.parse(path).getroot().findall(".//entry"):
        senses = Counter()
        for r in e.iter("pdtb2_relation"):
            if r.get("sense"):
                senses[_pdtb3_level2(r.get("sense"))] += int(r.get("pdtb_freq") or 0)
        if senses and sum(senses.values()) == 0:
            senses = Counter({k: 1 for k in senses})
        total = sum(senses.values())
        orths = []
        for o in e.findall("orths/orth"):
            parts = [p.text.lower().replace(",", " , ").split() for p in o.findall("part") if p.text]
            if parts:
                orths.append((o.get("type"), parts))
        entries.append({"id": e.get("id"), "word": e.get("word"), "cats": {c.text.strip() for c in e.findall("syn/cat") if c.text},
                        "senses": {k: v / total for k, v in senses.items()} if total else {}, "orths": orths})
    return entries


def _load_dimlex_en_weights(path) -> Dict[Tuple[str, str], float]:
    """(form, reading) -> share of connective readings on GUM, where GUM has at least _EN_WEIGHT_MIN_N occurrences"""
    if not path or not Path(path).is_file():
        logger.warning("DiMLex-Eng weights not found at \"%s\": ambiguous English adverbs count with weight 1", path)
        return {}
    out = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if int(row["n"]) >= _EN_WEIGHT_MIN_N:
                out[(row["form"], row["reading"])] = float(row["p_conn"])
    return out


dimlex_en_entries = _load_dimlex_en(settings.dimlex_en_path)
dimlex_en_weights = _load_dimlex_en_weights(settings.dimlex_en_weights_path) if dimlex_en_entries is not None else {}
dimlex_en_sha256 = None
_dimlex_en_singles: Dict[str, List[dict]] = defaultdict(list)
_dimlex_en_multi: List[Tuple[dict, str, list]] = []
if dimlex_en_entries is not None:
    import hashlib as _hashlib
    dimlex_en_sha256 = _hashlib.sha256(Path(settings.dimlex_en_path).read_bytes()).hexdigest()
    for _e in dimlex_en_entries:
        for _typ, _parts in _e["orths"]:
            if _typ == "cont" and len(_parts) == 1 and len(_parts[0]) == 1:
                _dimlex_en_singles[_parts[0][0]].append(_e)
            if _typ == "discont" or len(_parts[0]) > 1:
                _dimlex_en_multi.append((_e, _typ, _parts))
    logger.info("DiMLex-Eng: %d entries from \"%s\" (sha256 %s); %d GUM weights", len(dimlex_en_entries), settings.dimlex_en_path,
                dimlex_en_sha256[:12], len(dimlex_en_weights))
dimlex_en_weights_sha256 = (_hashlib.sha256(Path(settings.dimlex_en_weights_path).read_bytes()).hexdigest()
                            if dimlex_en_weights else None)


def _dimlex_en_decide(tokens: List[Token], c: dict, weights: Optional[Dict[Tuple[str, str], float]] = None
                      ) -> Tuple[float, str, Optional[str]]:
    """Weight of the connective reading, why, and the reading the GUM weights are kept for ("adv", "wrb"; None otherwise).
    weights: the GUM table (default: the loaded one; {} for weight 1, for the weight estimation)."""
    weights = dimlex_en_weights if weights is None else weights
    if c["kind"] != "word":
        return 1.0, c["kind"], None
    i = c["start"]
    t = tokens[i]

    def head(k):
        h = tokens[k].head_index
        return k if h is None else h

    def children(k):
        return [x for j, x in enumerate(tokens) if j != k and x.head_index == k]

    def clausal(k):
        return tokens[k].pos_coarse in ("VERB", "AUX") or any(x.dep_type == "cop" for x in children(k))

    def clause_function(k):
        seen = set()
        while tokens[k].dep_type == "conj" and k not in seen and head(k) != k:
            seen.add(k)
            k = head(k)
        return tokens[k].dep_type, k

    first = i == next((j for j, x in enumerate(tokens) if not x.is_punct), 0)
    if t.pos_value == "WRB":
        dep, k = clause_function(head(i))
        if not (dep == "advcl" or (dep in ("acl", "acl:relcl") and clausal(head(k)))):
            return 0.0, f"relative or interrogative clause ({dep})", None
        return weights.get((c["form"], "wrb"), 1.0), "relative adverb", "wrb"
    if t.pos_coarse == "SCONJ":
        dep, k = clause_function(head(i))
        if dep in _DIMLEX_COMPLEMENT_DEPS or (dep == "acl" and c["form"] in _EN_CORRELATE_COMPLEMENTS):
            return 0.0, f"complement clause ({dep})", None
        return 1.0, "subordinator", None
    if t.pos_coarse == "CCONJ":
        if first:
            return 1.0, "coordinator", None
        if t.dep_type == "cc" and clausal(head(i)):
            return 1.0, "coordinator (clauses)", None
        return 0.0, "coordinator (not clauses)", None
    if t.pos_coarse == "ADP":
        return 0.0, "preposition", None
    if t.pos_coarse == "ADV":
        if t.dep_type == "compound:prt" or any(x.dep_type == "cop" for x in children(i)):
            return 0.0, f"verb particle or predicate ({t.dep_type})", None
        if not clausal(head(i)):
            return 0.0, f"modifies a word ({tokens[head(i)].pos_coarse})", None
        return weights.get((c["form"], "adv"), 1.0), "adverb", "adv"
    return 0.0, f"other reading ({t.pos_coarse})", None


def _dimlex_en_sentence_connectives(sentence: Sentence, weights: Optional[Dict[Tuple[str, str], float]] = None
                                    ) -> List[Tuple[float, dict, bool]]:
    """(weight, DiMLex-Eng entry, conjunction reading) of every counted connective of one English sentence."""
    tokens = sentence.tokens
    best = {}
    for c in _dimlex_candidates([(t.text or "").lower().replace("’", "'") for t in tokens], _dimlex_en_singles, _dimlex_en_multi):
        weight, _, _ = _dimlex_en_decide(tokens, c, weights)
        key = (c["start"], c["form"])
        if key not in best or weight > best[key][0]:
            best[key] = (weight, c)
    return [(weight, c["entry"], tokens[c["start"]].pos_coarse in ("SCONJ", "CCONJ") or tokens[c["start"]].pos_value == "WRB")
            for weight, c in best.values() if weight > 0]


def _sentence_connectives(sentence: Sentence, lang: str) -> List[Tuple[float, dict, bool]]:
    return _memo("connectives", (sentence,), lang, lambda: _dimlex_sentence_connectives(sentence) if lang == "de"
                 else _dimlex_en_sentence_connectives(sentence))


def _connective_lexicon(lang: str):
    return dimlex_entries if lang == "de" else dimlex_en_entries if lang == "en" else None


def _dimlex_category_share(entry: dict, index: str) -> float:
    return sum(v for k, v in entry["senses"].items() if k in CONNECTIVE_SENSES[index])


def _count_connectives(
    sentences: List[Sentence],
    lang: str,
    total_words: int,
    assignment: str = "majority",
) -> Optional[Dict[str, float]]:
    """CNC* incidences from DiMLex (German) or DiMLex-Eng (English, 2026-10-29); None without the lexicon or for another
    language. `assignment` "proportional" (weight x category share) exists only for the impact comparison of 2026-10-27."""
    if _connective_lexicon(lang) is None:
        return None
    raw = {"CNCAll": 0.0, **{index: 0.0 for index in CONNECTIVE_SENSES}}
    for sentence in sentences:
        for weight, entry, conjunction in _sentence_connectives(sentence, lang):
            raw["CNCAll"] += weight
            for index in CONNECTIVE_SENSES:
                if (index == "CNCTemp" and not conjunction) or (index == "CNCTempX" and conjunction):
                    continue
                share = _dimlex_category_share(entry, index)
                if assignment == "proportional":
                    raw[index] += weight * share
                elif share >= CONNECTIVE_MAJORITY - 1e-9:
                    raw[index] += weight
    return {index: _incidence(value, total_words) for index, value in raw.items()}


def _dimlex_causal_particles(sentences: List[Sentence], lang: str) -> Optional[float]:
    """Causal particles of SMCAUSvp and SMCAUSr: the (weighted) number of the connectives CNCCaus counts."""
    if _connective_lexicon(lang) is None:
        return None
    return sum(weight for sentence in sentences for weight, entry, _ in _sentence_connectives(sentence, lang)
               if _dimlex_category_share(entry, "CNCCaus") >= CONNECTIVE_MAJORITY - 1e-9)


# ============================================================================
# CONNECTIVES (CNC*) — "How often are connecting expressions used?"
# All values are incidences per 1,000 words, counted from DiMLex (see above).
# ============================================================================

# LAY: All connectives combined, per 1,000 words.
# ↑ Higher = more explicitly connected prose. Approximate.
def cm_cncall(sentences: List[Sentence], lang: str, tokens_count: int, connectives: Optional[Dict[str, float]] = None) -> Optional[float]:
    # L1: accept precomputed dict to avoid recomputing 9 times per request.
    if connectives is None:
        connectives = _count_connectives(sentences, lang, tokens_count)
    return None if connectives is None else connectives["CNCAll"]

# LAY: Causal connectives (because, since, therefore…) per 1,000 words.
# ↑ Higher = more cause-effect signalling. Approximate.
def cm_cnccaus(sentences: List[Sentence], lang: str, tokens_count: int, connectives: Optional[Dict[str, float]] = None) -> Optional[float]:
    if connectives is None:
        connectives = _count_connectives(sentences, lang, tokens_count)
    return None if connectives is None else connectives["CNCCaus"]

# LAY: Logical connectives (and, or, either…) per 1,000 words.
# ↑ Higher = more logical coordination. Approximate.
def cm_cnclogic(sentences: List[Sentence], lang: str, tokens_count: int, connectives: Optional[Dict[str, float]] = None) -> Optional[float]:
    if connectives is None:
        connectives = _count_connectives(sentences, lang, tokens_count)
    return None if connectives is None else connectives["CNCLogic"]

# LAY: Adversative/contrastive connectives (but, however, although…) per 1,000 words.
# ↑ Higher = more contrastive discourse. Approximate.
def cm_cncadc(sentences: List[Sentence], lang: str, tokens_count: int, connectives: Optional[Dict[str, float]] = None) -> Optional[float]:
    if connectives is None:
        connectives = _count_connectives(sentences, lang, tokens_count)
    return None if connectives is None else connectives["CNCADC"]

# LAY: Temporal connectives (when, before, after…) per 1,000 words.
# ↑ Higher = more time-sequence signalling. Approximate.
def cm_cnctemp(sentences: List[Sentence], lang: str, tokens_count: int, connectives: Optional[Dict[str, float]] = None) -> Optional[float]:
    if connectives is None:
        connectives = _count_connectives(sentences, lang, tokens_count)
    return None if connectives is None else connectives["CNCTemp"]

# LAY: Expanded temporal connectives (at first, finally, meanwhile…) per 1,000 words.
# ↑ Higher = richer narrative time markers. Approximate.
def cm_cnctempx(sentences: List[Sentence], lang: str, tokens_count: int, connectives: Optional[Dict[str, float]] = None) -> Optional[float]:
    if connectives is None:
        connectives = _count_connectives(sentences, lang, tokens_count)
    return None if connectives is None else connectives["CNCTempX"]

# LAY: Additive connectives (and, also, moreover, furthermore…) per 1,000 words.
# ↑ Higher = more information stacking. Approximate.
def cm_cncadd(sentences: List[Sentence], lang: str, tokens_count: int, connectives: Optional[Dict[str, float]] = None) -> Optional[float]:
    if connectives is None:
        connectives = _count_connectives(sentences, lang, tokens_count)
    return None if connectives is None else connectives["CNCAdd"]

# LAY: Positive connectives (also, likewise, similarly…) per 1,000 words.
# ↑ Higher = more reinforcement/agreement signalling. Approximate.
def cm_cncpos(sentences: List[Sentence], lang: str, tokens_count: int, connectives: Optional[Dict[str, float]] = None) -> Optional[float]:
    if connectives is None:
        connectives = _count_connectives(sentences, lang, tokens_count)
    return None if connectives is None else connectives["CNCPos"]

# LAY: Negative connectives (however, but, yet, on the contrary…) per 1,000 words.
# ↑ Higher = more contrast/opposition signalling. Approximate.
def cm_cncneg(sentences: List[Sentence], lang: str, tokens_count: int, connectives: Optional[Dict[str, float]] = None) -> Optional[float]:
    if connectives is None:
        connectives = _count_connectives(sentences, lang, tokens_count)
    return None if connectives is None else connectives["CNCNeg"]

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
    for vector, word in zip(token_has_vector, words):
        if not word.isalpha() or vector is None:
            continue

        vector_array = np.asarray(vector, dtype=float)

        if (
            vector_array.ndim != 1
            or vector_array.size != tokens_vector_length
            or not np.all(np.isfinite(vector_array))
        ):
            continue

        vectors.append(vector_array)

    if not vectors:
        return None

    sentence_vector = np.mean(vectors, axis=0)

    if (
        not np.all(np.isfinite(sentence_vector))
        or np.linalg.norm(sentence_vector) == 0
    ):
        return None

    return sentence_vector


def _safe_cosine_similarity(vector_a, vector_b) -> Optional[float]:
    if vector_a is None or vector_b is None:
        return None

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return None

    # speed: sklearn's cosine_similarity (row-normalise, then dot) without its per-call input validation (~1 ms per pair)
    a = np.asarray(vector_a, dtype=np.float64)
    b = np.asarray(vector_b, dtype=np.float64)
    similarity = np.dot(a / np.sqrt(np.dot(a, a)), b / np.sqrt(np.dot(b, b)))

    return float(similarity) if np.isfinite(similarity) else None

# LSA approximation. The original Coh-Metrix LSA indices are based
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
    # (option lsa_use_truncated_svd, off by default) with as many vectors as components or fewer the
    # reduction loses nothing (their rank is at most their number), so the vectors stay as they are, which gives the cosines of a
    # lossless reduction. Before, the cap at n - 1 dropped a dimension: two sentences or two paragraphs were always similar 1.0.
    if n_samples <= n_components:
        return vectors
    effective = min(n_components, n_samples - 1, n_features - 1)
    if effective < 1:
        # Not enough samples/features to reduce meaningfully; return as-is.
        return vectors
    svd = TruncatedSVD(
        n_components=effective,
        random_state=0
    )
    reduced = svd.fit_transform(vectors)
    return reduced

# LSAGN / LSAGNd (Appendix A indices 44–45) measure how
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
# Caveat: the implementation uses spaCy word vectors (see), not TASA-
# trained LSA, so absolute values differ from the reference Coh-Metrix output.
# @see docs/review-report-v3.html (M5, L3) for the LSA approximation context.
# FV3 Fix: Build the Given-New hyperplane from the numerical rank of
# previous vectors so dependent vectors cannot add artificial dimensions.
def _svd(a):
    """np.linalg.svd (LAPACK gesdd); where it does not converge, scipy's slower but more robust gesvd driver (2026-10-25
    to the fallback the server session proposed): the failure depends on the CPU / BLAS build (7 of 4,041 pages on the workstation,
    none in the Apptainer image) and used to leave all eight LSA indices of the text None"""
    try:
        return np.linalg.svd(a, full_matrices=False)
    except np.linalg.LinAlgError:
        import scipy.linalg
        logger.info("numpy SVD did not converge (%s matrix); retried with scipy gesvd", a.shape)
        return scipy.linalg.svd(a, full_matrices=False, lapack_driver="gesvd")


def _project_onto_hyperplane(v, basis):
    if basis.shape[0] == 0:
        return np.zeros_like(v), v

    basis_t = np.asarray(basis, dtype=float).T

    U, singular_values, _ = _svd(basis_t)

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
    # speed (resurrection, 2026-10-23): _project_onto_hyperplane took the SVD of all preceding vectors at every sentence (a d x i
    # matrix, so quadratic work over a text). The left singular vectors and singular values of [v_0 .. v_{i-1}] are those of
    # [U S | v_{i-1}] with U, S from the step before (the same B B^T), a d x (d + 1) matrix at most; all singular directions are
    # carried on, and the rank cut-off is the original's (max(d, i) * eps * largest singular value).
    # speed: once the preceding sentences span the whole vector space (rank = dimension, from about
    # 300 sentences on), every further sentence is wholly given, ratio 1, and the SVD per sentence is skipped (15.4 s -> 2.7 s on
    # 928 sentences; the computed ratios differ from 1 by at most 2.3e-15)
    results = []
    carried = None
    full = False
    for i in range(len(vectors)):
        current_vec = vectors[i]
        if full:
            results.append(1.0)
            continue
        if i == 0:
            # the first sentence has no prior text, so nothing in it is given: G = 0 and G / (N + G) = 0 (book ch. 4, "how much
            # given versus new information exists in each sentence in a text, compared with the content of prior text
            # information, for example, G/(N+G)"; decision 2026-11-02). Its 0 enters the mean, so the mean depends on the
            # number of sentences (about 1/n; Maxim Konca reports rho .71 with the sentence count, .36 without the first
            # sentence); a one-sentence text gives 0.
            results.append(0.0)
            continue
        else:
            previous = np.asarray(vectors[i - 1], dtype=float)[:, None]
            stacked = previous if carried is None else np.hstack([carried, previous])
            U, singular_values, _ = _svd(stacked)
            carried = U * singular_values
            tolerance = max(len(current_vec), i) * np.finfo(singular_values.dtype).eps * singular_values[0]
            rank = np.sum(singular_values > tolerance)
            if rank == 0:
                p, perp = np.zeros_like(current_vec), current_vec
            else:
                Q = U[:, :rank]
                p = Q @ (Q.T @ current_vec)
                perp = current_vec - p
            G = np.linalg.norm(p)
            N = np.linalg.norm(perp)
            full = rank == len(current_vec)
        given_new_ratio = G / (G + N) if (G + N) > 0 else np.nan
        results.append(given_new_ratio)
    return np.array(results)

def _lsa_cohesion_indices(vec_per_paragraph_sentences: List[List[List[List[float]]]], words: List[List[List[str]]], tokens_vector_length: int, n_components, use_truncated_svd: bool = False) -> Dict[str, Any]:
    if tokens_vector_length <= 0:
        return {
            "LSASS1": np.nan,
            "LSASS1d": np.nan,
            "LSASSp": np.nan,
            "LSASSpd": np.nan,
            "LSAPP1": np.nan,
            "LSAPP1d": np.nan,
            "LSAGN": np.nan,
            "LSAGNd": np.nan,
        }

    # Keep one entry for every original sentence. None marks a sentence that
    # has no usable semantic vector and prevents later sentences/paragraphs
    # from being shifted into the wrong comparison window.
    sentence_vectors_by_paragraph = []
    paragraph_vectors = []

    for paragraph_index, paragraph in enumerate(vec_per_paragraph_sentences):
        paragraph_words = (
            words[paragraph_index]
            if paragraph_index < len(words)
            else []
        )
        paragraph_sentence_vectors = []

        for sentence_index, sentence_vectors in enumerate(paragraph):
            sentence_words = (
                paragraph_words[sentence_index]
                if sentence_index < len(paragraph_words)
                else []
            )
            paragraph_sentence_vectors.append(
                _sentence_vector(
                    sentence_vectors,
                    sentence_words,
                    tokens_vector_length
                )
            )

        sentence_vectors_by_paragraph.append(paragraph_sentence_vectors)

        valid_sentence_vectors = [
            vector
            for vector in paragraph_sentence_vectors
            if vector is not None
        ]

        if valid_sentence_vectors:
            paragraph_vector = np.mean(valid_sentence_vectors, axis=0)
            paragraph_vectors.append(
                paragraph_vector
                if np.linalg.norm(paragraph_vector) > 0
                else None
            )
        else:
            paragraph_vectors.append(None)

    def _transform_optional_vectors(optional_vectors):
        valid_positions = [
            index
            for index, vector in enumerate(optional_vectors)
            if vector is not None
        ]

        if not valid_positions:
            return [None] * len(optional_vectors)

        matrix = np.vstack([
            optional_vectors[index]
            for index in valid_positions
        ])

        transformed = (
            _reduce_dimensionality(matrix, n_components)
            if use_truncated_svd
            else matrix
        )

        result = [None] * len(optional_vectors)
        for position, vector in zip(valid_positions, transformed):
            result[position] = vector
        return result

    flattened_sentence_vectors = [
        vector
        for paragraph in sentence_vectors_by_paragraph
        for vector in paragraph
    ]
    flattened_sentence_vectors_transformed = _transform_optional_vectors(
        flattened_sentence_vectors
    )

    sentence_vectors_transformed_by_paragraph = []
    offset = 0
    for paragraph in sentence_vectors_by_paragraph:
        count = len(paragraph)
        sentence_vectors_transformed_by_paragraph.append(
            flattened_sentence_vectors_transformed[offset:offset + count]
        )
        offset += count

    paragraph_vectors_transformed = _transform_optional_vectors(
        paragraph_vectors
    )

    # --- LSA similarity between adjacent original sentences ---
    adj_sent_sim = []
    for vector_a, vector_b in zip(
        flattened_sentence_vectors_transformed,
        flattened_sentence_vectors_transformed[1:]
    ):
        similarity = _safe_cosine_similarity(vector_a, vector_b)
        if similarity is not None:
            adj_sent_sim.append(similarity)

    # --- LSA similarity between all usable sentence pairs in paragraphs ---
    # speed (resurrection, 2026-10-23): _safe_cosine_similarity for all pairs i < j of a paragraph at once, in loop order
    all_sent_pairs = _Moments()       # without the array of all pairs
    for paragraph in sentence_vectors_transformed_by_paragraph:
        usable = [
            np.asarray(vector, dtype=np.float64)
            for vector in paragraph
            if vector is not None and np.linalg.norm(vector) != 0
        ]
        if len(usable) < 2:
            continue
        matrix = np.vstack(usable)
        matrix = matrix / np.sqrt(np.einsum("ij,ij->i", matrix, matrix))[:, None]
        for first_index in range(len(usable) - 1):
            similarities = matrix[first_index + 1:] @ matrix[first_index]
            all_sent_pairs.add(similarities[np.isfinite(similarities)])

    # --- LSA similarity between adjacent original paragraphs ---
    adj_para_sim = []
    for vector_a, vector_b in zip(
        paragraph_vectors_transformed,
        paragraph_vectors_transformed[1:]
    ):
        similarity = _safe_cosine_similarity(vector_a, vector_b)
        if similarity is not None:
            adj_para_sim.append(similarity)

    usable_given_new_vectors = [
        vector
        for vector in flattened_sentence_vectors_transformed
        if vector is not None and np.linalg.norm(vector) > 0
    ]

    if usable_given_new_vectors:
        given_new_ratios = _lsa_given_new_for_vectors(
            np.vstack(usable_given_new_vectors)
        )
        given_new_ratios = given_new_ratios[
            np.isfinite(given_new_ratios)
        ]
    else:
        given_new_ratios = np.array([])

    adj_sent_sim = np.asarray(adj_sent_sim)
    adj_para_sim = np.asarray(adj_para_sim)

    return {
        'LSASS1': np.mean(adj_sent_sim) if adj_sent_sim.size > 0 else np.nan,
        'LSASS1d': np.std(adj_sent_sim) if adj_sent_sim.size > 0 else np.nan,
        'LSASSp': all_sent_pairs.mean(),
        'LSASSpd': all_sent_pairs.std(),
        'LSAPP1': np.mean(adj_para_sim) if adj_para_sim.size > 0 else np.nan,
        'LSAPP1d': np.std(adj_para_sim) if adj_para_sim.size > 0 else np.nan,
        'LSAGN': np.mean(given_new_ratios) if given_new_ratios.size > 0 else np.nan,
        'LSAGNd': np.std(given_new_ratios) if given_new_ratios.size > 0 else np.nan,
    }

# LAY: Average meaning-similarity between adjacent sentences.
# ↑ Higher = smoother topic flow. Approximate.
# ============================================================================
# LSA (LSA*) — "Are ideas in nearby sentences similar in meaning?"
# Latent Semantic Analysis overlap scores. 0 = unrelated, 1 = identical.
# Approximate: uses spaCy word vectors + TruncatedSVD(100) instead of the
# original TASA-trained 300-dim space. Rank-order should track,
# absolute values are not comparable to published Coh-Metrix output.
# ============================================================================

# LAY: Average meaning-similarity between adjacent sentences.
# ↑ Higher = smoother topic flow. Approximate.
def cm_lsass1(lsa_indices: Dict[str, Any]) -> Optional[float]:
    return lsa_indices["LSASS1"]

# LAY: How uneven is the meaning-similarity of adjacent sentences?
# ↑ Higher = bumpy topic transitions. Approximate.
def cm_lsass1d(lsa_indices: Dict[str, Any]) -> Optional[float]:
    return lsa_indices["LSASS1d"]

# LAY: Average meaning-similarity among all sentences within each paragraph.
# ↑ Higher = internally coherent paragraphs. Approximate.
def cm_lsassp(lsa_indices: Dict[str, Any]) -> Optional[float]:
    return lsa_indices["LSASSp"]

# LAY: How uneven is the within-paragraph meaning-similarity?
# ↑ Higher = mix of tight and loose paragraphs. Approximate.
def cm_lsasspd(lsa_indices: Dict[str, Any]) -> Optional[float]:
    return lsa_indices["LSASSpd"]

# LAY: Average meaning-similarity between adjacent paragraphs.
# ↑ Higher = smooth flow across paragraphs. Approximate.
def cm_lsapp1(lsa_indices: Dict[str, Any]) -> Optional[float]:
    return lsa_indices["LSAPP1"]

# LAY: How uneven is the meaning-similarity of adjacent paragraphs?
# ↑ Higher = abrupt topic shifts between paragraphs. Approximate.
def cm_lsapp1d(lsa_indices: Dict[str, Any]) -> Optional[float]:
    return lsa_indices["LSAPP1d"]

# LAY: How much of each sentence is already given by previous sentences?
# ↑ Higher = more previously given information and greater semantic cohesion. Approximate.
def cm_lsagn(lsa_indices: Dict[str, Any]) -> Optional[float]:
    return lsa_indices["LSAGN"]

# LAY: How uneven is the given/new ratio across sentences?
# ↑ Higher = greater variation in how much prior information sentences reuse. Approximate.
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

def _load_subtlex_de(path: str) -> Dict[str, float]:
    """SUBTLEX-DE (Brysbaert et al. 2011) frequencies per million, as CTAP ships them (word TAB per-million, no header, one entry per
    form in its most frequent spelling): keyed in lower case, as the core looks words up. A missing file gives {} (German
    frequency indices None, with a warning)."""
    if not path or not Path(path).is_file():
        logger.warning("SUBTLEX-DE not found at \"%s\": the German WRDFRQ* and RDL2 indices will return None", path)
        return {}
    freq: Dict[str, float] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2 and parts[0].isalpha():
                try:
                    freq[parts[0].lower()] = freq.get(parts[0].lower(), 0.0) + float(parts[1])
                except ValueError:
                    continue
    return freq


def _load_subtlex_us(path: str) -> Dict[str, float]:
    """SUBTLEX-US (Brysbaert & New 2009) frequencies per million (column SUBTLWF of the tab-separated text version with header),
    keyed in lower case as SUBTLEX-DE; forms differing only in case are summed. The list splits words at apostrophes ("don",
    "t"), so the clitics of the English tokenizer (n't, 's) are not in it. A missing file gives {} (English WRDFRQ* and RDL2
    None, with a warning)."""
    if not path or not Path(path).is_file():
        logger.warning("SUBTLEX-US not found at \"%s\": the English WRDFRQ* and RDL2 indices will return None", path)
        return {}
    freq: Dict[str, float] = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        word_col, wf_col = header.index("Word"), header.index("SUBTLWF")
        for line in fh:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) > wf_col and parts[word_col].isalpha():
                try:
                    freq[parts[word_col].lower()] = freq.get(parts[word_col].lower(), 0.0) + float(parts[wf_col])
                except ValueError:
                    continue
    return freq


# German frequencies: SUBTLEX-DE, a frequency norm validated on German word recognition (Brysbaert et al.
# 2011), instead of the 10,000-article Wikipedia sample; the formulas stay. Coverage on the fix sample (word forms, lower case):
# content words 86.1 % (Wikipedia sample 91.8 %), all words 92.7 % (95.7 %). English keeps the Wikipedia sample.
# English (2026-10-29): SUBTLEX-US, the counterpart of SUBTLEX-DE, instead of the Wikipedia sample.
_FREQUENCY_SOURCE = {"de": "subtlex-de", "en": "subtlex-us"}

class _FrequencySources(dict):
    """{source: {word: per million}} of one language. The Wikipedia sample, which neither language uses since SUBTLEX (German
    2026-10-28, English 2026-10-29), is read only when asked for (both samples were built
    in every process, about 2 s and 40 MB)."""

    def __init__(self, lang: str, loaded: dict):
        super().__init__(loaded)
        self.lang = lang

    def __missing__(self, source):
        if source != "wiki-20220301-sample10000":
            raise KeyError(source)
        name = f"word_frequencies_{self.lang}_{self.lang}wiki-20220301-sample10000.csv"
        self[source] = _normalize_word_frequencies_per_million(_load_word_frequencies(str(RESOURCES) + "/" + name))
        return self[source]


all_word_frequencies_map = {
    "en": _FrequencySources("en", {"subtlex-us": _load_subtlex_us(settings.subtlex_us_path)}),
    "de": _FrequencySources("de", {"subtlex-de": _load_subtlex_de(settings.subtlex_de_path)}),
}

# Word-frequency resource used here is
# wiki-20220301-sample10000 (~10k Wikipedia articles; German: SUBTLEX-DE since 2026-10-28) rather than the
# original CELEX reference corpus (17.9M words, paywalled). Absolute
# frequency values and their logs differ from the reference Coh-Metrix
# output; rank-ordering among content words is expected to be broadly
# similar. Upstream also emitted the V3 WRDFRQc/a/mc as stubs (None); they are gone since 2026-10-29 (no placeholders), and the
# indices carry the public suffix _corpus (_labels).
# ============================================================================
# WORD FREQUENCY (WRDFRQ*) & L2 READABILITY (RDL2)
# Approximate: the original Coh-Metrix uses the CELEX lexical database, which
# is paywalled. This implementation uses a 10,000-article Wikipedia sample
#. Standard outputs WRDFRQc/a/mc remain None; the _wiki10000
# suffix variants are emitted instead. RDL2 is the "L2 Readability" composite
# formula (Crossley et al. 2008); until 2026-10-28 it was emitted twice (SYNSTRUTa,
# SYNSTRUTt), since then only with SYNSTRUTa, as the paper uses (see cm_rdl2).
# ============================================================================

# LAY: Average frequency of content words in everyday text.
# ↑ Higher = uses common words. Approximate (Wikipedia sample —).
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

    # log10 since 2026-10-28 ("WRDFRQc switch to log"), like WRDFRQa and WRDFRQmc. Coh-Metrix 3.0 describes WRDFRQc
    # as the raw "CELEX word frequency for content words, mean" (McNamara et al. 2014, Appendix A), but Graesser et al. (2004,
    # p. 197) argue for the logarithm: reading time falls linearly with log frequency, and raw frequencies are dominated by a
    # few very frequent words. Before: the mean of the raw per-million frequencies.
    log_word_frequencies = [np.log10(freq) for freq in word_frequencies if freq > 0]

    return np.mean(log_word_frequencies) if log_word_frequencies else None

# LAY: Average log-frequency of ALL words in everyday text.
# ↑ Higher = text leans on common vocabulary. Approximate.
def cm_wrdfrqa(
        tokens: List[Token],
        lang: str,
        frequencies_source: str
) -> Optional[float]:
    word_frequencies_map = all_word_frequencies_map[lang][frequencies_source]

    # FV2 fix: WRDFRQa operates on lexical word tokens. Exclude punctuation
    # and other non-alphabetic tokens and ignore words not covered by the
    # reference frequency resource instead of assigning them frequency 0.
    # words with a letter (_is_word_text) and split contraction stems as their word (_word_forms)
    forms = [form.lower() for form in _word_forms(tokens)]
    word_frequencies = [
        word_frequencies_map[form]
        for token, form in zip(tokens, forms)
        if _is_word_text(token.text)
        and form in word_frequencies_map
    ]

    # FV2 approximation: use base-10 logarithms for the normalized
    # frequency values to approximate the published Coh-Metrix scale.
    log_word_frequencies = [
        np.log10(freq)
        for freq in word_frequencies
        if freq > 0
    ]

    return np.mean(log_word_frequencies) if log_word_frequencies else None

# LAY: Average MINIMUM log-frequency among content words (rarest word per sentence).
# ↑ Higher = even the rarest content words are reasonably common. Approximate.
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
        else None
    )

# LAY: Second-language readability composite score (Crossley et al. 2008).
# ↑ Higher = easier for L2 learners. Approximate.
def cm_rdl2(crfcwo1: float, synstrut: float, wrdfrq: float) -> Optional[float]:
    # RDL2: Coh-Metrix L2 Readability (Appendix A index 106).
    # Formula (Crossley, Greenfield & McNamara 2008, TESOL Quarterly 42(3), Table 2 and p. 486):
    #
    #   RDL2 = -45.032
    #          + 52.230 * content word overlap, adjacent sentences (CRFCWO1)
    #          + 61.306 * sentence syntax similarity, "sentence to sentence, adjacent, mean" (p. 482: SYNSTRUTa)
    #          + 22.205 * "the CELEX frequency score for written words" (p. 481)
    #
    # Since 2026-10-28: only the adjacent variant (RDL2_synstruta; the former RDL2_synstrutt used SYNSTRUTt, which
    # the paper does not), and the frequency term is WRDFRQc (mean log frequency of the content words) instead of WRDFRQmc
    # (log minimum per sentence): the paper names neither, and its mean of 2.349 on the Bormuth passages (Table 1) looks like a
    # mean log frequency; on the fix sample WRDFRQc gave CLEAR rho .507 vs .469, AUC simple > standard tagesschau .998 vs .989,
    # dlf .989 vs .987. An approximation in any case: SYNSTRUT is dependency-based, and the weights were fitted on English
    # CELEX values, while the frequencies here are SUBTLEX-DE (German, fix B2) or the Wikipedia sample (English).
    # L8 fix: make the None-path explicit instead of relying on the caller's
    # try/except to swallow a TypeError from None * float.
    if crfcwo1 is None or synstrut is None or wrdfrq is None:
        return None
    l2 = -45.032 + (52.230 * crfcwo1) + (61.306 * synstrut) + (22.205 * wrdfrq)
    return l2

def _sm_get_data(sentences: List[Sentence]):
    return _memo("sm_data", (sentences,), None, lambda: _sm_get_data_uncached(sentences))


def _sm_get_data_uncached(sentences: List[Sentence]):
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


def _has_complete_sm_verb_annotations(sentences: List[Sentence]) -> bool:
    """Check only the annotation layers required by inventory verb counts."""
    return bool(sentences) and all(
        _has_complete_pos_annotations(sentence)
        and all(
            token.pos_coarse != "VERB" or _has_usable_lemma(token)
            for token in _non_punct_tokens(sentence)
        )
        for sentence in sentences
    )

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
    causal_practical_set,
    sentences: Optional[List[Sentence]] = None,
):
    return _memo("count_verbs", (poses, words, lemmas, causal_practical_set, sentences), None,
                 lambda: _count_verbs_uncached(poses, words, lemmas, causal_practical_set, sentences))


def _count_verbs_uncached(
    poses,
    words,
    lemmas,
    causal_practical_set,
    sentences: Optional[List[Sentence]] = None,
):
    counters = {
        "causal_verbs": 0,
        "intentional_verbs": 0,
        "intentional_particles": 0,
    }
    # FV3 fix:
    # Count intentional particles as lexical expressions rather
    # than individual tokens. This is required for multi-token expressions
    # such as "in order to".
    # Causal particles (SMCAUSvp, SMCAUSr) are DiMLex connectives since 2026-10-27: _dimlex_causal_particles.
    # German "um ... zu" by its tag: "um" as KOUI; the phrase "um zu" only matched when nothing stood
    # between ("um zu helfen", not "um ihr zu helfen"), so SMINTEr was 0 in 293 of 300 German texts.
    language = causal_practical_set.get("language")
    particle_tags = causal_practical_set.get("intentional_particle_tags", set())
    for i, sent in enumerate(poses):
        counters["intentional_particles"] += _count_phrase_matches(
            words[i],
            causal_practical_set["intentional_particles"]
        )
        if particle_tags and sentences is not None and i < len(sentences):
            counters["intentional_particles"] += sum(
                1 for t in sentences[i].tokens if ((t.text or "").lower(), t.pos_value) in particle_tags)

        for j, pos in enumerate(sent):
            if pos != "VERB":
                continue

            lemma = (lemmas[i][j] or "").strip().lower()
            lemma_candidates = {lemma} if lemma else set()

            # Reconstruct dependency-linked separable/phrasal verbs for the
            # lexical inventories. German "löst ... aus" must be able to match
            # "auslösen"; English particles can match WordNet forms such as
            # "break down". Each verbal token still contributes at most once.
            # Code review 2026-10-31: a verb with a particle is looked up only as the particle verb (German particle + lemma,
            # English "lemma particle"). Before, the bare base verb matched too ("geht ... aus" counted as gehen, "schlägt ...
            # vor" as schlagen, while "ausgeht" did not), and the German joins were tried for English as well.
            if sentences is not None and i < len(sentences) and lemma:
                sentence = sentences[i]
                particles = [(child.lemma or child.text or "").strip().lower() for child_index, child in enumerate(sentence.tokens)
                             if child_index != j and child.head_index == j
                             and (child.dep_type or "").strip().lower() == "compound:prt"]   # UD particle verbs
                particles = [p for p in particles if p]
                if particles:
                    if language == "de":
                        lemma_candidates = {lemma if lemma.startswith(p) else p + lemma for p in particles}
                    else:
                        lemma_candidates = {lemma + " " + p for p in particles}

            if lemma_candidates.intersection(
                causal_practical_set["causal_verbs"]
            ):
                counters["causal_verbs"] += 1

            if lemma_candidates.intersection(
                causal_practical_set["intentional_verbs"]
            ):
                counters["intentional_verbs"] += 1

    return counters

def _get_hyponyms(synset, visited=None):
    if visited is None:
        visited = set()

    if synset in visited:
        return set()

    visited.add(synset)
    hypos = set()

    for hypo in synset.hyponyms():
        if hypo in visited:
            continue
        hypos.add(hypo)
        hypos |= _get_hyponyms(hypo, visited)

    return hypos

def _get_verb_lemmas_for_synset(synset):
    verbs = set(synset.lemma_names())
    for hypo in _get_hyponyms(synset):
        verbs |= set(hypo.lemma_names())
    return {v.replace('_', ' ') for v in verbs}

def _germanet_all_hyponyms(synset, visited=None) -> set:
    if visited is None:
        visited = set()

    if synset in visited:
        return set()

    visited.add(synset)
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
        if hypo in visited:
            continue
        hypos.add(hypo)
        hypos |= _germanet_all_hyponyms(hypo, visited)
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
# Without GermaNet (no path, an empty or missing folder, or a folder without GermaNet) the German verb inventories are the hand-
# written seed lists, as in duui-coh-metrix 0.2.0; the indices built on them (SMCAUSv, SMCAUSvp, SMINTEp, SMCAUSr, SMINTEr) name
# them in their label (_SEEDlist instead of _GN, README rule 7; decision 2026-11-02). With GermaNet: the expansion plus the seeds.
GERMAN_VERB_INVENTORY = "GN" if (_de_causal_verbs_expanded and _de_intentional_verbs_expanded) else "SEEDlist"
if GERMAN_VERB_INVENTORY == "SEEDlist":
    _de_causal_verbs_expanded = set(_DE_CAUSAL_SEED)
    _de_intentional_verbs_expanded = set(_DE_INTENTIONAL_SEED)
    logger.info("DE causal and intentional verbs: GermaNet expansion unavailable, seed lists (%d, %d entries)",
                len(_de_causal_verbs_expanded), len(_de_intentional_verbs_expanded))
else:
    _de_causal_verbs_expanded |= _DE_CAUSAL_SEED
    _de_intentional_verbs_expanded |= _DE_INTENTIONAL_SEED
    logger.info("DE causal verbs: %d, intentional verbs: %d entries (GermaNet expansion)",
                len(_de_causal_verbs_expanded), len(_de_intentional_verbs_expanded))

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
            # OBSOLETE since 2026-10-27, kept for comparison: causal particles are DiMLex connectives now
            # (_dimlex_causal_particles; no English lexicon here, so SMCAUSvp/SMCAUSr are None for English)
            "causal_particles_llmlist": [
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
            "language": "en",
        }

    if lang == "de":
        if _de_causal_verbs_expanded is None or _de_intentional_verbs_expanded is None:
            return None                 # no GermaNet: the verb-based situation-model indices are None (see above)
        return {
            "causal_verbs": _de_causal_verbs_expanded,
            "intentional_verbs": _de_intentional_verbs_expanded,
            # OBSOLETE since 2026-10-27, kept for comparison: causal particles are DiMLex connectives now
            # (_dimlex_causal_particles)
            "causal_particles_llmlist": [
                "weil",
                "deshalb",
                "daher",
                "darum",
                "folglich",
                "infolgedessen",
                "aus diesem Grund",
            ],
            # "um ... zu" by the tag of "um" (KOUI), whatever stands between
            "intentional_particles": [
                "auf dass",
                "auf daß",
            ],
            "intentional_particle_tags": {("um", "KOUI")},
            "language": "de",
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

# Dependency relations used to reconstruct German verbal complexes. UD since the UD port (2026-10-28): the auxiliaries, the
# passive auxiliary and the copula hang on the content word, and a verbal xcomp (infinitive: "hat gehen lassen", "versucht zu
# gehen") joins the complex, as TIGER's oc did; TIGER: oc, aux, pd. A complement clause (ccomp, TIGER oc) is no longer joined.

_DE_VERBAL_LINK_DEPS = {
    "aux",
    "aux:pass",
    "cop",
    "xcomp",
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
    # the passive auxiliary with an infinitive clause ("werden wir dazu ermutigt, ... zu betrachten") is no future
    #
    if (finite.dep_type or "").strip().lower() == "aux:pass":
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
    # "sein" + participle: perfect ("ist gegangen") or state ("ist geschlossen"). UD (since the port, 2026-10-28) marks it on
    # "sein", which hangs on the participle: aux = perfect, aux:pass (the "sein" passive, UD German GSD) or cop = state; TIGER
    # marked it on the participle (oc = perfect, pd = state).
    if _de_is_sein(finite):
        if any((_de_is_gewesen(token) for _, token, _ in participles)):
            return (True, 'high', 'sein_gewesen_perfect')
        if any((_de_is_worden(token) for _, token, _ in participles)):
            return (True, 'high', 'passive_perfect_worden')
        if not participles:
            return (False, 'high', 'sein_nonperfect')
        sein_dep = _de_norm_dep(finite)
        if sein_dep == 'aux':
            return (True, 'high', 'sein_perfect_aux')
        if sein_dep in ('aux:pass', 'cop'):
            return (False, 'high', 'stative_passive')
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
            sein_deps = {_de_norm_dep(token) for token in nonfinite_sein}
            if participles and 'aux' in sein_deps:
                return (True, 'high', 'future_perfect_sein_aux')
            if participles and sein_deps <= {'aux:pass', 'cop'}:
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
    # Konjunktiv as in the TMV annotator (Ramm, Loáiciga, Friedrich & Fraser 2017, Table 3; decision 2026-11-02): German has no
    # conditional tense, the Konjunktiv is a mood, and the tense is that of the form: "(er) würde arbeiten / gearbeitet haben"
    # future (futureI+II), "(er) arbeite / arbeitete" present, "(er) habe / hätte gearbeitet" past. So "würde" + infinitive is
    # FUTURE, the Konjunktiv II ("könnte", "wäre"; UD Mood=Sub with Tense=Past) PRESENT, and as a perfect (haben / sein +
    # participle: "hätte gearbeitet", "wäre gegangen") PAST. (0.2.0 read "würde" as UNKNOWN and the Konjunktiv II as PAST.)
    if _de_norm_text(finite) in _DE_WUERDE_FORMS:
        return ('FUTURE', 'high', 'future_wuerde')
    if _de_is_future_component(sentence, component, finite_index):
        return ('FUTURE', 'high', 'future_construction')
    if (finite.morph_mood or '').strip() == 'Sub' and (finite.morph_tense or '').strip() == 'Past':
        perfect = (_de_is_haben(finite) or _de_is_sein(finite)) and any(
            _de_is_participle(sentence.tokens[i]) for i in component if i != finite_index)
        return ('PAST', 'high', 'konjunktiv_ii_perfect') if perfect else ('PRESENT', 'high', 'konjunktiv_ii')
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

def _de_is_nominalized_infinitive(token) -> Optional[bool]:
    """Classify the nominal head of an ``am/beim`` construction.

    ``True`` means that the head can be identified as a nominalized verb,
    ``False`` means that the available lexical resource identifies no verb,
    and ``None`` means that the decision requires GermaNet but the resource is
    unavailable.  The three-way result prevents missing lexical evidence from
    being treated as evidence for an ordinary nominal construction.
    """
    text = _de_norm_text(token)
    lemma = _de_norm_lemma(token)
    if not text:
        return False

    if token.pos_value in _DE_VERB_TAGS or token.pos_coarse in {'VERB', 'AUX'}:
        return True

    if germanet is None:
        return None

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

def _de_is_progressive_nominal_candidate(token) -> bool:
    return (
        token.pos_value == 'NN'
        or token.pos_coarse == 'NOUN'
        or token.pos_value in _DE_VERB_TAGS
        or token.pos_coarse in {'VERB', 'AUX'}
    )

def _de_find_progressive_nominal_candidate(
    sentence: Sentence,
    prep_index: int,
) -> Optional[int]:
    tokens = sentence.tokens
    # UD (since the port, 2026-10-28): the preposition is a case dependent of its noun ("am Arbeiten": an <- Arbeiten)
    head_index = tokens[prep_index].head_index
    if head_index is not None and 0 <= head_index < len(tokens) and head_index != prep_index \
            and _de_is_progressive_nominal_candidate(tokens[head_index]):
        return head_index
    for i, token in enumerate(tokens):
        if token.head_index != prep_index:
            continue
        if _de_is_progressive_nominal_candidate(token):
            return i
    upper = min(len(tokens), prep_index + 4)
    for i in range(prep_index + 1, upper):
        token = tokens[i]
        if token.pos_coarse == 'PUNCT':
            break
        if _de_is_progressive_nominal_candidate(token):
            return i
    return None

def _detect_german_progressive(sentence: Sentence):
    tokens = sentence.tokens
    has_sein = any((_de_is_sein(token) for token in tokens if _de_is_verbish(token)))
    if not has_sein:
        return (False, 'none', None)
    for i, token in enumerate(tokens):
        text = _de_norm_text(token)
        # "am"/"beim" are split into "an dem"/"bei dem" since the UD port (2026-10-28): both words keep the span of the
        # contraction, so the pair is recognised by its shared span
        if text in {'an', 'bei'} and i + 1 < len(tokens) and _de_norm_text(tokens[i + 1]) == 'dem' \
                and tokens[i + 1].begin == token.begin and tokens[i + 1].end == token.end:
            text = 'am' if text == 'an' else 'beim'
        if text not in {'am', 'beim'}:
            continue
        nominal_index = _de_find_progressive_nominal_candidate(sentence, i)
        if nominal_index is None:
            continue
        is_nominalized_infinitive = _de_is_nominalized_infinitive(
            tokens[nominal_index]
        )
        if is_nominalized_infinitive is None:
            return (None, 'none', 'missing_germanet_for_progressive')
        if not is_nominalized_infinitive:
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
        # negation: the word list (UD has no negation relation; TIGER's ng was "nicht", which the list has)
        if text in _DE_COMPLETION_NEGATION_WORDS:
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
        if dep != 'compound:prt':                 # separated verb particle (UD; TIGER svp)
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
    if not _has_analyzable_verbal_predicate(sentence):
        return ('UNKNOWN', 'none', 'no_verbal_predicate')
    progressive, progressive_confidence, progressive_evidence = _detect_german_progressive(sentence)
    if progressive is True:
        return ('IN_PROGRESS', progressive_confidence, progressive_evidence)
    if progressive is None:
        return ('UNRESOLVED', progressive_confidence, progressive_evidence)
    # the German perfect (haben / sein + participle; detected by _de_detect_perfect_component, checked against gold, README C1)
    # is COMPLETED, as the English perfect is (before it only fed the completion test, so a Perfekt
    # sentence had the state of a present-tense one, "Er hat gestern gearbeitet" = "Er arbeitet")
    grammatical_aspect, perfect_confidence, perfect_evidence = _detect_german_grammatical_aspect(sentence)
    if grammatical_aspect == 'PERFECT':
        return ('COMPLETED', perfect_confidence, 'perfect')
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
    if not _has_analyzable_verbal_predicate(sentence):
        return ('UNKNOWN', 'none', 'no_verbal_predicate')
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
# ↑ Higher = more cause-describing verbs. Approximate.
def cm_smcausv(sentences: List[Sentence], lang: str) -> Optional[float]:
    # SMCAUSv: Causal verb incidence (per 1000 words)
    # Spec: Coh-Metrix 3.0 Appendix A, index 59; Chapter 4 §Situation model
    causal_set = _causal_practical_verbs_intentional(lang)
    if causal_set is None:
        return None
    if not _has_complete_sm_verb_annotations(sentences):
        return None
    words, _, _, lemmas, poses, _ = _sm_get_data(
        sentences
    )
    counts = count_verbs(
        poses,
        words,
        lemmas,
        causal_set,
        sentences,
    )
    # FV1 fix: Use the same non-punctuation word count as DESWC to maintain
    # consistent word counting across incidence-based indices.
    total_words = _count_non_punct_tokens(sentences)
    return _incidence(
        counts["causal_verbs"],
        total_words
    )

# LAY: Causal verbs + causal connectives (because, in order to…) per 1,000 words.
# ↑ Higher = more explicit causal structure. Approximate.
def _llmlist_causal_particles(words: List[List[str]], causal_set: dict) -> int:
    """the causal particles of the 0.2.0 list (variant _LLMlist), as phrases in each sentence's words"""
    return sum(_count_phrase_matches(sentence_words, causal_set["causal_particles_llmlist"]) for sentence_words in words)


def cm_smcausvp(sentences: List[Sentence], lang: str, particles: str = "dimlex") -> Optional[float]:
    # particles: "dimlex" (DiMLex / DiMLex-Eng causal connectives) or "llmlist" (the 0.2.0 list, variant _LLMlist)
    # SMCAUSvp: Causal verbs and causal particles incidence (per 1000 words)
    # Spec: Coh-Metrix 3.0 Appendix A, index 60; Chapter 4: "causal verbs plus
    # causal particles (SMCAUSvp: e.g., both causal verbs and connectives such
    # as because, in order to)". Earlier impl summed intentional_verbs by mistake.
    causal_set = _causal_practical_verbs_intentional(lang)
    if causal_set is None:
        return None
    if not _has_complete_sm_verb_annotations(sentences):
        return None
    words, _, _, lemmas, poses, _ = _sm_get_data(
        sentences
    )
    causal_particles = (_llmlist_causal_particles(words, causal_set) if particles == "llmlist"
                        else _dimlex_causal_particles(sentences, lang))
    if causal_particles is None:
        return None
    counts = count_verbs(
        poses,
        words,
        lemmas,
        causal_set,
        sentences,
    )
    # FV1 fix: Use the same non-punctuation word count as DESWC to maintain
    # consistent word counting across incidence-based indices.
    total_words = _count_non_punct_tokens(sentences)
    return _incidence(
        counts["causal_verbs"]
        + causal_particles,
        total_words
    )

# LAY: Intentional verbs (contact, drop, plan, want…) per 1,000 words.
# ↑ Higher = more goal/intention language. Approximate.
def cm_smintep(sentences: List[Sentence], lang: str) -> Optional[float]:
    # SMINTEp: Intentional verbs incidence (per 1000 words)
    # Spec: Coh-Metrix 3.0 Appendix A, index 61; Chapter 4: "intentional verbs
    # (SMINTEp: e.g., contact, drop, walk, talk)". Earlier impl returned
    # intentional_particles raw count by mistake.
    causal_set = _causal_practical_verbs_intentional(lang)
    if causal_set is None:
        return None
    if not _has_complete_sm_verb_annotations(sentences):
        return None
    words, _, _, lemmas, poses, _ = _sm_get_data(
        sentences
    )
    counts = count_verbs(
        poses,
        words,
        lemmas,
        causal_set,
        sentences,
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
def cm_smcausr(sentences: List[Sentence], lang: str, particles: str = "dimlex") -> Optional[float]:
    # particles: "dimlex" (DiMLex / DiMLex-Eng causal connectives) or "llmlist" (the 0.2.0 list, variant _LLMlist)
    causal_set = _causal_practical_verbs_intentional(lang)
    if causal_set is None:
        return None
    if not _has_complete_sm_verb_annotations(sentences):
        return None
    words, _, _, lemmas, poses, _ = _sm_get_data(
        sentences
    )
    counts = count_verbs(
        poses,
        words,
        lemmas,
        causal_set,
        sentences,
    )
    causal_verbs = counts["causal_verbs"]

    causal_particles = (_llmlist_causal_particles(words, causal_set) if particles == "llmlist"
                        else _dimlex_causal_particles(sentences, lang))
    if causal_particles is None:
        return None

    # Fix C3 (2026-10-28): the denominator plus 1, as the original defines it (Graesser, McNamara, Louwerse & Cai 2004, BRMIC
    # 36(2), p. 200: "a ratio of causal particles (P) to causal verbs (V). The denominator is incremented by the value of 1 to
    # handle the rare case in which there are zero causal verbs in a text"); before, a text without a causal verb gave None.
    ratio = causal_particles / (causal_verbs + 1)

    return np.round(ratio, 3)

# LAY: Ratio of intentional particles to intentional verbs.
# ↑ Higher = intentions signalled more by particles than verbs. Approximate.
def cm_sminter(sentences: List[Sentence], lang: str) -> Optional[float]:
    causal_set = _causal_practical_verbs_intentional(lang)
    if causal_set is None:
        return None
    if not _has_complete_sm_verb_annotations(sentences):
        return None
    words, _, _, lemmas, poses, _ = _sm_get_data(
        sentences
    )
    counts = count_verbs(
        poses,
        words,
        lemmas,
        causal_set,
        sentences,
    )
    intentional_verbs = counts["intentional_verbs"]

    # Fix C3 (2026-10-28): the denominator plus 1, as for SMCAUSr (Graesser et al. 2004, p. 200, for the causal ratio; the
    # intentional ratio of Coh-Metrix 3.0 by analogy); before, a text without an intentional verb gave None (1,962 of the
    # 4,853 texts of the fix sample)
    ratio = counts["intentional_particles"] / (intentional_verbs + 1)

    return np.round(ratio, 3)

def get_SMCAUSlsa(poses: List[List[str]], vectors: List[List[List[Any]]]):
    valid_verbs_by_dimension = defaultdict(list)
    for i, sent in enumerate(poses):
        for j, pos in enumerate(sent):
            if pos == "VERB":
                if i >= len(vectors) or j >= len(vectors[i]):
                    continue
                vector = vectors[i][j]
                if vector is None:
                    continue
                try:
                    vector_array = np.asarray(vector, dtype=float)
                except (TypeError, ValueError):
                    continue
                if (
                    vector_array.ndim != 1
                    or vector_array.size == 0
                    or not np.all(np.isfinite(vector_array))
                    or np.linalg.norm(vector_array) == 0
                ):
                    continue
                valid_verbs_by_dimension[vector_array.size].append(vector_array)

    if not valid_verbs_by_dimension:
        return None

    # A single malformed token must not dictate the vector space. Use the
    # dimension with the greatest valid coverage; a tie is deterministic.
    selected_dimension = max(
        valid_verbs_by_dimension,
        key=lambda dimension: (
            len(valid_verbs_by_dimension[dimension]),
            dimension,
        ),
    )
    all_verbs = valid_verbs_by_dimension[selected_dimension]

    # M3 fix: previously used adjacent pairs only. Spec description is identical
    # to SMCAUSwn, and LSASSp also uses all pairs \u2014 harmonize SMCAUSlsa to
    # all unordered verb pairs.
    # speed: the pairs as matrix products in blocks of rows, instead of a Python loop over the pairs
    # (18.7 s on 2,158 verbs, quadratic); the same cosines, the mean differs in the last bits only
    n = len(all_verbs)
    if n < 2:
        return None
    unit = np.vstack(all_verbs)
    unit = unit / np.linalg.norm(unit, axis=1)[:, None]
    total, count = 0.0, 0
    for start in range(0, n - 1, 1024):
        stop = min(n - 1, start + 1024)
        block = unit[start:stop] @ unit[start + 1:].T                  # rows start..stop-1, columns start+1..n-1
        values = block[np.arange(start + 1, n)[None, :] > np.arange(start, stop)[:, None]]
        values = values[np.isfinite(values)]
        total += float(values.sum())
        count += values.size
    return total / count if count else None

# LAY: Average meaning-similarity of all verb pairs via LSA vectors.
# ↑ Higher = verbs share semantic field. Approximate.
def cm_smcauslsa(sentences: List[Sentence]) -> Optional[float]:
    _, _, _, _, poses, vectors = _sm_get_data(sentences)
    SMCAUSlsa = get_SMCAUSlsa(poses, vectors)
    return np.round(SMCAUSlsa, 3) if SMCAUSlsa is not None else None

def get_SMCAUSwn(poses: List[List[str]], word_lemma: List[List[str]], lang: str):
    lang = (lang or "").strip().lower()

    if lang not in {"en", "de"}:
        return None

    if lang == "de" and germanet is None:
        logger.warning("GermaNet not available")
        return None

    verbs_lemma = [
        word_lemma[sentence_index][token_index].lower()
        for sentence_index, sentence_poses in enumerate(poses)
        for token_index, pos in enumerate(sentence_poses)
        if pos == "VERB"
        and word_lemma[sentence_index][token_index]
    ]

    if len(verbs_lemma) < 2:
        return None

    try:
        if lang == "en":
            synsets_by_lemma = {
                lemma: set(wn.synsets(lemma, pos=wn.VERB))
                for lemma in set(verbs_lemma)
            }
        else:
            synsets_by_lemma = {
                lemma: {
                    synset
                    for synset in germanet.get_synsets_by_orthform(lemma)
                    if synset.word_category == WordCategory.verben
                }
                for lemma in set(verbs_lemma)
            }
    except Exception as exc:
        logger.warning("Semantic verb lookup failed: %s", exc)
        return None

    # Verbs missing from the lexical resource do not form a valid semantic
    # comparison pair. They must not be treated as verbs with zero overlap.
    covered_synsets = [
        synsets_by_lemma[lemma]
        for lemma in verbs_lemma
        if synsets_by_lemma[lemma]
    ]

    if len(covered_synsets) < 2:
        return None

    # speed: the pairs counted per distinct lemma instead of per token pair (quadratic in the
    # verbs, 1 s on 2,158 verbs): two tokens of one lemma always overlap, two lemmas c1 * c2 times when their synsets intersect;
    # the same integer counts as before
    lemma_counts = defaultdict(int)
    for lemma in verbs_lemma:
        if synsets_by_lemma[lemma]:
            lemma_counts[lemma] += 1
    lemmas = list(lemma_counts)
    total_pairs = len(covered_synsets) * (len(covered_synsets) - 1) // 2
    syn_overlap_count = sum(c * (c - 1) // 2 for c in lemma_counts.values())
    for i, first in enumerate(lemmas):
        for second in lemmas[i + 1:]:
            if synsets_by_lemma[first].intersection(synsets_by_lemma[second]):
                syn_overlap_count += lemma_counts[first] * lemma_counts[second]

    return syn_overlap_count / total_pairs if total_pairs > 0 else None

# LAY: Share of verb pairs that share a WordNet synset (semantic overlap).
# ↑ Higher = verbs cluster around shared meanings. Partial for DE.
def cm_smcauswn(sentences: List[Sentence], lang) -> Optional[float]:
    _, _, _, lemmas, poses, vectors = _sm_get_data(sentences)
    SMCAUSwn = get_SMCAUSwn(poses, lemmas, lang)
    return np.round(SMCAUSwn, 3) if SMCAUSwn is not None else None

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
    # UNRESOLVED denotes a specifically required distinction for which an
    # external lexical resource is missing.  Unlike a generally unknown
    # dimension, it must invalidate this pair instead of silently reducing the
    # calculation to the remaining dimension.
    if 'UNRESOLVED' in {tense_a, aspect_a, tense_b, aspect_b}:
        return None
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
        return None
    lang = (lang or '').strip().lower()
    # German: the progressive (am / beim + nominalised infinitive) is checked with GermaNet; without it those sentences have no
    # aspect and only their tense is compared. The label names it (SMTEMP_LLMlist instead of _LLMlist_GN; decision 2026-11-02)
    states = [_get_sentence_temporal_state(sentence, lang) for sentence in sentences]
    pair_scores = []
    for state_a, state_b in zip(states, states[1:]):
        score = _smtemp_pair_score(state_a, state_b)
        if score is not None:
            pair_scores.append(score)
    if not pair_scores:
        return None
    smtemp = sum(pair_scores) / len(pair_scores)
    return round(smtemp, 3)







def _sha256(path) -> Optional[str]:
    import hashlib
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except (OSError, TypeError):
        return None


# hashed at import, so that the hash is that of the code that runs
_CODE_SHA256 = {"cohmetrix_core.py": _sha256(__file__), "ud_mapping.py": _sha256(ud_mapping.__file__),
                **{f"depedit/{n}": _sha256(os.path.join(ud_mapping._DEPEDIT_DIR, n))
                   for n in ("clearnlp_additions.ini", "stan2uni.ini")}}


def versions_record(language: str) -> dict:
    """what the values depend on beyond the annotators (the version stayed "0.2.0-py"
    across a day of code changes): a hash of the code, the LSA settings, the easability model, which resources were found"""
    lang = _LANGUAGE_CODES.get((language or "").strip().lower().replace("_", "-").split("-")[0], language)
    found = lambda path: bool(path) and Path(path).exists()
    model = settings.easability_model_path if lang == "de" else settings.easability_model_path_en
    resources = ({"germanet": Path(settings.germanet_path).name if germanet is not None else None,
                  "subtlex_de": found(settings.subtlex_de_path), "koeper": found(settings.koeper_path),
                  "birchenough": found(settings.birchenough_path)} if lang == "de" else
                 {"wordnet": wn.get_version(), "subtlex_us": found(settings.subtlex_us_path),
                  "brysbaert": found(settings.brysbaert_path), "kuperman": found(settings.kuperman_path)})
    return {"code_sha256": _CODE_SHA256,
            "lsa": {"truncated_svd": settings.lsa_use_truncated_svd, "svd_components": settings.lsa_svd_components},
            "easability_model": Path(model).name, "easability_model_sha256": _sha256(model), "resources": resources}


def compute(request: TextImagerRequest, wanted: Optional[Set[str]] = None) -> Dict[str, Tuple[Optional[float], Optional[str]]]:
    """the values of the indices `wanted` (core keys, e.g. "DESPC", "RDFOG"; None: all) for one text: {key: (value, error)}.
    Only the blocks the wanted indices need are computed (BLOCK_DEPS). An error inside an index sets that index's error; an
    error outside the indices sets it for every wanted index not computed yet. Labels, variants and the Text Easability
    components are the DUUI layer's (duui.py)."""
    lang = _language(request)   # "de" or "en"
    results: Dict[str, Tuple[Optional[float], Optional[str]]] = {}
    active = _active_blocks(wanted)
    _SENTENCE_FEATURES.reset()
    _TEXT_MEMO.reset()
    try:
        _SENTENCE_FEATURES.clear()
        _TEXT_MEMO.clear()

        # a "sentence" without a word (SaT can cut off "–" or "–«") is no sentence: it counted in DESSC and broke the adjacency
        # of its neighbours; a paragraph left without a sentence goes too
        for p in request.paragraphs:
            p.sentences = [s for s in p.sentences if any(not t.is_punct for t in s.tokens)]
        request.paragraphs = [p for p in request.paragraphs if p.sentences]

        sentences = []
        for p in request.paragraphs:
            sentences.extend(p.sentences)

        tokens = []
        for s in sentences:
            tokens.extend(s.tokens)

        # German (2026-10-28) and English (2026-10-29) are read in UD: relation labels in
        # lower case as UD writes them (an upstream annotator may upper-case them, as the duui-spacy container does), the root as
        # "root". spaCy's TIGER / ClearNLP labels are mapped to UD (best effort, ud_mapping; duui.py marks the indices).
        for t in tokens:
            t.dep_type = (t.dep_type or "").strip().lower()
            if t.dep_type == "--":
                t.dep_type = "root"
        ud_mapping.map_to_ud(sentences, ud_mapping.label_scheme(lang, (t.dep_type for t in tokens)))

        # noun chunks from the UD tree (UD ports: German 2026-10-28, English 2026-10-29; _ud_noun_chunks), whatever the request brings
        # outside any index, so a failure here used to end the request with no index at all; now SYNNP
        # and DRNP carry it
        try:
            noun_chunks = _ud_noun_chunks(sentences)
            noun_chunks_error = None
        except Exception as e:
            logger.error("Error building the noun chunks: %s", e)
            noun_chunks, noun_chunks_error = [], f"noun chunks: {e}"

        # FV1 fix: Exclude punctuation for overall consistency.
        tokens_count = sum(1 for t in tokens if not t.is_punct)
        has_readability_input = tokens_count > 0 and bool(sentences)

        vector_dimension_counts = defaultdict(int)
        for t in tokens:
            if not t.has_vector or t.vector is None:
                continue
            try:
                vector_array = np.asarray(t.vector, dtype=float)
            except (TypeError, ValueError):
                continue
            # the token keeps the array: the LSA and SMCAUSlsa steps converted every vector again from its
            # list; np.asarray of the array is the array itself, with the same values
            t.vector = vector_array
            if (
                vector_array.ndim == 1
                and vector_array.size > 0
                and np.all(np.isfinite(vector_array))
            ):
                vector_dimension_counts[vector_array.size] += 1

        tokens_vector_length = (
            max(
                vector_dimension_counts,
                key=lambda dimension: (
                    vector_dimension_counts[dimension],
                    dimension,
                ),
            )
            if vector_dimension_counts
            else 0
        )

        ### Descriptive

        if "DESPC" in active:
            # DESPC
            try:
                despc = cm_despc(request.paragraphs)
                despc_error = None
            except Exception as e:
                logger.error("Error calculating DESPC: %s", e)
                despc = None
                despc_error = str(e)
            results["DESPC"] = (despc, despc_error)

        if "DESSC" in active:
            # DESSC
            try:
                dessc = cm_dessc(request.paragraphs)
                dessc_error = None
            except Exception as e:
                logger.error("Error calculating DESSC: %s", e)
                dessc = None
                dessc_error = str(e)
            results["DESSC"] = (dessc, dessc_error)

        if "DESWC" in active:
            # DESWC
            try:
                deswc = cm_deswc(request.paragraphs)
                deswc_error = None
            except Exception as e:
                logger.error("Error calculating DESWC: %s", e)
                deswc = None
                deswc_error = str(e)
            results["DESWC"] = (deswc, deswc_error)

        if "DESPL" in active:
            # DESPL
            try:
                despl = cm_despl(request.paragraphs)
                despl_error = None
            except Exception as e:
                logger.error("Error calculating DESPL: %s", e)
                despl = None
                despl_error = str(e)
            results["DESPL"] = (despl, despl_error)

        if "DESPLd" in active:
            # DESPLd
            try:
                despld = cm_despld(request.paragraphs)
                despld_error = None
            except Exception as e:
                logger.error("Error calculating DESPLd: %s", e)
                despld = None
                despld_error = str(e)
            results["DESPLd"] = (despld, despld_error)

        if "DESSL" in active:
            # DESSL
            try:
                dessl = cm_dessl(request.paragraphs)
                dessl_error = None
            except Exception as e:
                logger.error("Error calculating DESSL: %s", e)
                dessl = None
                dessl_error = str(e)
            results["DESSL"] = (dessl, dessl_error)

        if "DESSLd" in active:
            # DESSLd
            try:
                dessld = cm_dessld(request.paragraphs)
                dessld_error = None
            except Exception as e:
                logger.error("Error calculating DESSLd: %s", e)
                dessld = None
                dessld_error = str(e)
            results["DESSLd"] = (dessld, dessld_error)

        if "DESWLsy" in active:
            # DESWLsy
            try:
                deswlsy = cm_deswlsy(tokens, lang)
                deswlsy_error = None
            except Exception as e:
                logger.error("Error calculating DESWLsy: %s", e)
                deswlsy = None
                deswlsy_error = str(e)
            results["DESWLsy"] = (deswlsy, deswlsy_error)

        if "DESWLsyd" in active:
            # DESWLsyd
            try:
                deswlsyd = cm_deswlsyd(tokens, lang)
                deswlsyd_error = None
            except Exception as e:
                logger.error("Error calculating DESWLsyd: %s", e)
                deswlsyd = None
                deswlsyd_error = str(e)
            results["DESWLsyd"] = (deswlsyd, deswlsyd_error)

        if "DESWLlt" in active:
            # DESWLlt
            try:
                deswllt = cm_deswllt(request.paragraphs)
                deswllt_error = None
            except Exception as e:
                logger.error("Error calculating DESWLlt: %s", e)
                deswllt = None
                deswllt_error = str(e)
            results["DESWLlt"] = (deswllt, deswllt_error)

        if "DESWLltd" in active:
            # DESWLltd
            try:
                deswlltd = cm_deswlltd(request.paragraphs)
                deswlltd_error = None
            except Exception as e:
                logger.error("Error calculating DESWLltd: %s", e)
                deswlltd = None
                deswlltd_error = str(e)
            results["DESWLltd"] = (deswlltd, deswlltd_error)

            ### Text Easability Principal Component Scores

            # PCNARz ... PCTEMPp: filled in at the end, once their inputs are computed

            ### Referential Cohesion

        if "CRFNO1" in active:
            # CRFNO1
            try:
                crfno1 = cm_crfno1(sentences)
                crfno1_error = None
            except Exception as e:
                logger.error("Error calculating CRFNO1: %s", e)
                crfno1 = None
                crfno1_error = str(e)
            results["CRFNO1"] = (crfno1, crfno1_error)

        if "CRFAO1" in active:
            # CRFAO1
            try:
                crfao1 = cm_crfao1(sentences)
                crfao1_error = None
            except Exception as e:
                logger.error("Error calculating CRFAO1: %s", e)
                crfao1 = None
                crfao1_error = str(e)
            results["CRFAO1"] = (crfao1, crfao1_error)

        if "CRFSO1" in active:
            # CRFSO1
            try:
                crfso1 = cm_crfso1(sentences)
                crfso1_error = None
            except Exception as e:
                logger.error("Error calculating CRFSO1: %s", e)
                crfso1 = None
                crfso1_error = str(e)
            results["CRFSO1"] = (crfso1, crfso1_error)

        if "CRFNOa" in active:
            # CRFNOa
            try:
                crfnoa = cm_crfnoa(sentences)
                crfnoa_error = None
            except Exception as e:
                logger.error("Error calculating CRFNOa: %s", e)
                crfnoa = None
                crfnoa_error = str(e)
            results["CRFNOa"] = (crfnoa, crfnoa_error)

        if "CRFAOa" in active:
            # CRFAOa
            try:
                crfaoa = cm_crfaoa(sentences)
                crfaoa_error = None
            except Exception as e:
                logger.error("Error calculating CRFAOa: %s", e)
                crfaoa = None
                crfaoa_error = str(e)
            results["CRFAOa"] = (crfaoa, crfaoa_error)

        if "CRFSOa" in active:
            # CRFSOa
            try:
                crfsoa = cm_crfsoa(sentences)
                crfsoa_error = None
            except Exception as e:
                logger.error("Error calculating CRFSOa: %s", e)
                crfsoa = None
                crfsoa_error = str(e)
            results["CRFSOa"] = (crfsoa, crfsoa_error)

        if "CRFCWO1" in active:
            # CRFCWO1
            try:
                crfcwo1 = cm_crfcwo1(sentences)
                crfcwo1_error = None
            except Exception as e:
                logger.error("Error calculating CRFCWO1: %s", e)
                crfcwo1 = None
                crfcwo1_error = str(e)
            results["CRFCWO1"] = (crfcwo1, crfcwo1_error)

        if "CRFCWO1d" in active:
            # CRFCWO1d
            try:
                crfcwo1d = cm_crfcwo1d(sentences)
                crfcwo1d_error = None
            except Exception as e:
                logger.error("Error calculating CRFCWO1d: %s", e)
                crfcwo1d = None
                crfcwo1d_error = str(e)
            results["CRFCWO1d"] = (crfcwo1d, crfcwo1d_error)

        if "CRFCWOa" in active:
            # CRFCWOa
            try:
                crfcwoa = cm_crfcwoa(sentences)
                crfcwoa_error = None
            except Exception as e:
                logger.error("Error calculating CRFCWOa: %s", e)
                crfcwoa = None
                crfcwoa_error = str(e)
            results["CRFCWOa"] = (crfcwoa, crfcwoa_error)

        if "CRFCWOad" in active:
            # CRFCWOad
            try:
                crfcwoad = cm_crfcwoad(sentences)
                crfcwoad_error = None
            except Exception as e:
                logger.error("Error calculating CRFCWOad: %s", e)
                crfcwoad = None
                crfcwoad_error = str(e)
            results["CRFCWOad"] = (crfcwoad, crfcwoad_error)

            ### Lexical Diversity

        if "LSA shared" in active:
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

        if "LSASS1" in active:
            # LSASS1
            try:
                lsass1 = cm_lsass1(lsa_indices)
                lsass1_error = None
            except Exception as e:
                logger.error("Error calculating LSASS1: %s", e)
                lsass1 = None
                # M7 fix: guard against None + str when the outer LSA step succeeded.
                lsass1_error = ((lsa_error + "\n") if lsa_error else "") + str(e)
            results["LSASS1"] = (lsass1, lsass1_error)

        if "LSASS1d" in active:
            # LSASS1d
            try:
                lsass1d = cm_lsass1d(lsa_indices)
                lsass1d_error = None
            except Exception as e:
                logger.error("Error calculating LSASS1d: %s", e)
                lsass1d = None
                lsass1d_error = ((lsa_error + "\n") if lsa_error else "") + str(e)  # M7
            results["LSASS1d"] = (lsass1d, lsass1d_error)

        if "LSASSp" in active:
            # LSASSp
            try:
                lsassp = cm_lsassp(lsa_indices)
                lsassp_error = None
            except Exception as e:
                logger.error("Error calculating LSASSp: %s", e)
                lsassp = None
                lsassp_error = ((lsa_error + "\n") if lsa_error else "") + str(e)  # M7
            results["LSASSp"] = (lsassp, lsassp_error)

        if "LSASSpd" in active:
            # LSASSpd
            try:
                lsasspd = cm_lsasspd(lsa_indices)
                lsasspd_error = None
            except Exception as e:
                logger.error("Error calculating LSASSpd: %s", e)
                lsasspd = None
                lsasspd_error = ((lsa_error + "\n") if lsa_error else "") + str(e)  # M7
            results["LSASSpd"] = (lsasspd, lsasspd_error)

        if "LSAPP1" in active:
            # LSAPP1
            try:
                lsapp1 = cm_lsapp1(lsa_indices)
                lsapp1_error = None
            except Exception as e:
                logger.error("Error calculating LSAPP1: %s", e)
                lsapp1 = None
                lsapp1_error = ((lsa_error + "\n") if lsa_error else "") + str(e)  # M7
            results["LSAPP1"] = (lsapp1, lsapp1_error)

        if "LSAPP1d" in active:
            # LSAPP1d
            try:
                lsapp1d = cm_lsapp1d(lsa_indices)
                lsapp1d_error = None
            except Exception as e:
                logger.error("Error calculating LSAPP1d: %s", e)
                lsapp1d = None
                lsapp1d_error = ((lsa_error + "\n") if lsa_error else "") + str(e)  # M7
            results["LSAPP1d"] = (lsapp1d, lsapp1d_error)

        if "LSAGN" in active:
            # LSAGN
            try:
                lsagn = cm_lsagn(lsa_indices)
                lsagn_error = None
            except Exception as e:
                logger.error("Error calculating LSAGN: %s", e)
                lsagn = None
                lsagn_error = ((lsa_error + "\n") if lsa_error else "") + str(e)  # M7
            results["LSAGN"] = (lsagn, lsagn_error)

        if "LSAGNd" in active:
            # LSAGNd
            try:
                lsagnd = cm_lsagnd(lsa_indices)
                lsagnd_error = None
            except Exception as e:
                logger.error("Error calculating LSAGNd: %s", e)
                lsagnd = None
                lsagnd_error = ((lsa_error + "\n") if lsa_error else "") + str(e)  # M7
            results["LSAGNd"] = (lsagnd, lsagnd_error)

            ### Lexical Diversity

        if "LDTTRc" in active:
            # LDTTRc
            try:
                ldttrc = cm_ldttrc(tokens)
                ldttrc_error = None
            except Exception as e:
                logger.error("Error calculating LDTTRc: %s", e)
                ldttrc = None
                ldttrc_error = str(e)
            results["LDTTRc"] = (ldttrc, ldttrc_error)

        if "LDTTRa" in active:
            # LDTTRa
            try:
                ldttra = cm_ldttra(tokens)
                ldttra_error = None
            except Exception as e:
                logger.error("Error calculating LDTTRa: %s", e)
                ldttra = None
                ldttra_error = str(e)
            results["LDTTRa"] = (ldttra, ldttra_error)

        if "LDMTLDa" in active:
            # LDMTLDa
            try:
                ldmtlda = cm_ldmtlda(tokens)
                ldmtlda_error = None
            except Exception as e:
                logger.error("Error calculating LDMTLDa: %s", e)
                ldmtlda = None
                ldmtlda_error = str(e)
            results["LDMTLDa"] = (ldmtlda, ldmtlda_error)

        if "LDVOCDa" in active:
            # LDVOCDa
            try:
                ldvocda = cm_ldvocda(tokens)
                ldvocda_error = None
            except Exception as e:
                logger.error("Error calculating LDVOCDa: %s", e)
                ldvocda = None
                ldvocda_error = str(e)
            results["LDVOCDa"] = (ldvocda, ldvocda_error)

            ### Connectives

        if "CNC shared" in active:
            # L1: compute connective counts once and reuse across all 9 CNC* indices.
            try:
                _cnc_counts = _count_connectives(sentences, lang, tokens_count)
            except Exception as e:
                logger.error("Error precomputing connective counts: %s", e)
                _cnc_counts = None

        if "CNCAll" in active:
            # CNCAll
            try:
                cncall = cm_cncall(sentences, lang, tokens_count, connectives=_cnc_counts)
                cncall_error = None
            except Exception as e:
                logger.error("Error calculating CNCAll: %s", e)
                cncall = None
                cncall_error = str(e)
            results["CNCAll"] = (cncall, cncall_error)

        if "CNCCaus" in active:
            # CNCCaus
            try:
                cnccaus = cm_cnccaus(sentences, lang, tokens_count, connectives=_cnc_counts)
                cnccaus_error = None
            except Exception as e:
                logger.error("Error calculating CNCCaus: %s", e)
                cnccaus = None
                cnccaus_error = str(e)
            results["CNCCaus"] = (cnccaus, cnccaus_error)

        if "CNCLogic" in active:
            # CNCLogic
            try:
                cnclogic = cm_cnclogic(sentences, lang, tokens_count, connectives=_cnc_counts)
                cnclogic_error = None
            except Exception as e:
                logger.error("Error calculating CNCLogic: %s", e)
                cnclogic = None
                cnclogic_error = str(e)
            results["CNCLogic"] = (cnclogic, cnclogic_error)

        if "CNCADC" in active:
            # CNCADC
            try:
                cncadc = cm_cncadc(sentences, lang, tokens_count, connectives=_cnc_counts)
                cncadc_error = None
            except Exception as e:
                logger.error("Error calculating CNCADC: %s", e)
                cncadc = None
                cncadc_error = str(e)
            results["CNCADC"] = (cncadc, cncadc_error)

        if "CNCTemp" in active:
            # CNCTemp
            try:
                cnctemp = cm_cnctemp(sentences, lang, tokens_count, connectives=_cnc_counts)
                cnctemp_error = None
            except Exception as e:
                logger.error("Error calculating CNCTemp: %s", e)
                cnctemp = None
                cnctemp_error = str(e)
            results["CNCTemp"] = (cnctemp, cnctemp_error)

        if "CNCTempx" in active:
            # CNCTempx
            try:
                cnctempx = cm_cnctempx(sentences, lang, tokens_count, connectives=_cnc_counts)
                cnctempx_error = None
            except Exception as e:
                logger.error("Error calculating CNCTempx: %s", e)
                cnctempx = None
                cnctempx_error = str(e)
            results["CNCTempx"] = (cnctempx, cnctempx_error)

        if "CNCAdd" in active:
            # CNCAdd
            try:
                cncadd = cm_cncadd(sentences, lang, tokens_count, connectives=_cnc_counts)
                cncadd_error = None
            except Exception as e:
                logger.error("Error calculating CNCAdd: %s", e)
                cncadd = None
                cncadd_error = str(e)
            results["CNCAdd"] = (cncadd, cncadd_error)

        if "CNCPos" in active:
            # CNCPos
            try:
                cncpos = cm_cncpos(sentences, lang, tokens_count, connectives=_cnc_counts)
                cncpos_error = None
            except Exception as e:
                logger.error("Error calculating CNCPos: %s", e)
                cncpos = None
                cncpos_error = str(e)
            results["CNCPos"] = (cncpos, cncpos_error)

        if "CNCNeg" in active:
            # CNCNeg
            try:
                cncneg = cm_cncneg(sentences, lang, tokens_count, connectives=_cnc_counts)
                cncneg_error = None
            except Exception as e:
                logger.error("Error calculating CNCNeg: %s", e)
                cncneg = None
                cncneg_error = str(e)
            results["CNCNeg"] = (cncneg, cncneg_error)

            ### Situation Model

        if "SMCAUSv" in active:
            # SMCAUSv
            try:
                smcausv = cm_smcausv(sentences, lang)
                smcausv_error = None
            except Exception as e:
                logger.error("Error calculating SMCAUSv: %s", e)
                smcausv = None
                smcausv_error = str(e)
            results["SMCAUSv"] = (smcausv, smcausv_error)

        if "SMCAUSvp" in active:
            # SMCAUSvp
            try:
                smcausvp = cm_smcausvp(sentences, lang)
                smcausvp_error = None
            except Exception as e:
                logger.error("Error calculating SMCAUSvp: %s", e)
                smcausvp = None
                smcausvp_error = str(e)
            results["SMCAUSvp"] = (smcausvp, smcausvp_error)

        if "SMINTEp" in active:
            # SMINTEp
            try:
                smintep = cm_smintep(sentences, lang)
                smintep_error = None
            except Exception as e:
                logger.error("Error calculating SMINTEp: %s", e)
                smintep = None
                smintep_error = str(e)
            results["SMINTEp"] = (smintep, smintep_error)

        if "SMCAUSr" in active:
            # SMCAUSr
            try:
                smcausr = cm_smcausr(sentences, lang)
                smcausr_error = None
            except Exception as e:
                logger.error("Error calculating SMCAUSr: %s", e)
                smcausr = None
                smcausr_error = str(e)
            results["SMCAUSr"] = (smcausr, smcausr_error)

        if "SMINTEr" in active:
            # SMINTEr
            try:
                sminter = cm_sminter(sentences, lang)
                sminter_error = None
            except Exception as e:
                logger.error("Error calculating SMINTEr: %s", e)
                sminter = None
                sminter_error = str(e)
            results["SMINTEr"] = (sminter, sminter_error)

        if "SMCAUSlsa" in active:
            # SMCAUSlsa
            try:
                smcauslsa = cm_smcauslsa(sentences)
                smcauslsa_error = None
            except Exception as e:
                logger.error("Error calculating SMCAUSlsa: %s", e)
                smcauslsa = None
                smcauslsa_error = str(e)
            results["SMCAUSlsa"] = (smcauslsa, smcauslsa_error)

        if "SMCAUSwn" in active:
            # SMCAUSwn
            try:
                smcauswn = cm_smcauswn(sentences, lang)
                smcauswn_error = None
            except Exception as e:
                logger.error("Error calculating SMCAUSwn: %s", e)
                smcauswn = None
                smcauswn_error = str(e)
            results["SMCAUSwn"] = (smcauswn, smcauswn_error)

        if "SMTEMP" in active:
            # SMTEMP
            try:
                smtemp = cm_smtemp(sentences, lang)
                smtemp_error = None
            except Exception as e:
                logger.error("Error calculating SMTEMP: %s", e)
                smtemp = None
                smtemp_error = str(e)
            results["SMTEMP"] = (smtemp, smtemp_error)

            ### Syntactic Complexity

        if "SYNLE" in active:
            # SYNLE
            try:
                synle = cm_synle(sentences, lang)
                synle_error = None
            except Exception as e:
                logger.error("Error calculating SYNLE: %s", e)
                synle = None
                synle_error = str(e)
            results["SYNLE"] = (synle, synle_error)

        if "SYNNP" in active:
            # SYNNP
            try:
                synnp = cm_synnp(sentences, noun_chunks, lang)
                synnp_error = None
                if noun_chunks_error:
                    synnp, synnp_error = None, noun_chunks_error
            except Exception as e:
                logger.error("Error calculating SYNNP: %s", e)
                synnp = None
                synnp_error = str(e)
            results["SYNNP"] = (synnp, synnp_error)

        if "SYNMEDpos" in active:
            # SYNMEDpos
            try:
                synmedpos = cm_synmedpos(sentences)
                synmedpos_error = None
            except Exception as e:
                logger.error("Error calculating SYNMEDpos: %s", e)
                synmedpos = None
                synmedpos_error = str(e)
            results["SYNMEDpos"] = (synmedpos, synmedpos_error)

        if "SYNMEDwrd" in active:
            # SYNMEDwrd
            try:
                synmedwrd = cm_synmedwrd(sentences)
                synmedwrd_error = None
            except Exception as e:
                logger.error("Error calculating SYNMEDwrd: %s", e)
                synmedwrd = None
                synmedwrd_error = str(e)
            results["SYNMEDwrd"] = (synmedwrd, synmedwrd_error)

        if "SYNMEDlem" in active:
            # SYNMEDlem
            try:
                synmedlem = cm_synmedlem(sentences)
                synmedlem_error = None
            except Exception as e:
                logger.error("Error calculating SYNMEDlem: %s", e)
                synmedlem = None
                synmedlem_error = str(e)
            results["SYNMEDlem"] = (synmedlem, synmedlem_error)

        if "SYNSTRUTa" in active:
            # SYNSTRUTa
            synstrut_trees = _TreeInterner()        # the text's trees, interned once for SYNSTRUTa and SYNSTRUTt
            try:
                synstruta = cm_synstruta(sentences, synstrut_trees)
                synstruta_error = None
            except Exception as e:
                logger.error("Error calculating SYNSTRUTa: %s", e)
                synstruta = None
                synstruta_error = str(e)
            results["SYNSTRUTa"] = (synstruta, synstruta_error)

        if "SYNSTRUTt" in active:
            # SYNSTRUTt
            try:
                synstrutt = cm_synstrutt(request.paragraphs, synstrut_trees)
                synstrutt_error = None
            except Exception as e:
                logger.error("Error calculating SYNSTRUTt: %s", e)
                synstrutt = None
                synstrutt_error = str(e)
            results["SYNSTRUTt"] = (synstrutt, synstrutt_error)

            ### Syntactic Pattern Density

        if "DR shared" in active:
            # L2: compute count_metrics once and reuse across all 8 DR* indices.
            try:
                _dr_metrics = _count_metrics(sentences, noun_chunks, lang)
            except Exception as e:
                logger.error("Error precomputing DR metrics: %s", e)
                _dr_metrics = None

        if "DRNP" in active:
            # DRNP
            try:
                drnp = cm_drnp(sentences, noun_chunks, lang, metrics=_dr_metrics)
                drnp_error = None
                if noun_chunks_error:
                    drnp, drnp_error = None, noun_chunks_error
            except Exception as e:
                logger.error("Error calculating DRNP: %s", e)
                drnp = None
                drnp_error = str(e)
            results["DRNP"] = (drnp, drnp_error)

        if "DRVP" in active:
            # DRVP
            try:
                drvp = cm_drvp(sentences, noun_chunks, lang, metrics=_dr_metrics)
                drvp_error = None
            except Exception as e:
                logger.error("Error calculating DRVP: %s", e)
                drvp = None
                drvp_error = str(e)
            results["DRVP"] = (drvp, drvp_error)

        if "DRAP" in active:
            # DRAP
            try:
                drap = cm_drap(sentences, noun_chunks, lang, metrics=_dr_metrics)
                drap_error = None
            except Exception as e:
                logger.error("Error calculating DRAP: %s", e)
                drap = None
                drap_error = str(e)
            results["DRAP"] = (drap, drap_error)

        if "DRPP" in active:
            # DRPP
            try:
                drpp = cm_drpp(sentences, noun_chunks, lang, metrics=_dr_metrics)
                drpp_error = None
            except Exception as e:
                logger.error("Error calculating DRPP: %s", e)
                drpp = None
                drpp_error = str(e)
            results["DRPP"] = (drpp, drpp_error)

        if "DRPVAL" in active:
            # DRPVAL
            try:
                drpval = cm_drpval(sentences, noun_chunks, lang, metrics=_dr_metrics)
                drpval_error = None
            except Exception as e:
                logger.error("Error calculating DRPVAL: %s", e)
                drpval = None
                drpval_error = str(e)
            results["DRPVAL"] = (drpval, drpval_error)

        if "DRNEG" in active:
            # DRNEG
            try:
                drneg = cm_drneg(sentences, noun_chunks, lang, metrics=_dr_metrics)
                drneg_error = None
            except Exception as e:
                logger.error("Error calculating DRNEG: %s", e)
                drneg = None
                drneg_error = str(e)
            results["DRNEG"] = (drneg, drneg_error)

        if "DRGERUND" in active:
            # DRGERUND
            try:
                drgerund = cm_drgerund(sentences, noun_chunks, lang, metrics=_dr_metrics)
                drgerund_error = None
            except Exception as e:
                logger.error("Error calculating DRGERUND: %s", e)
                drgerund = None
                drgerund_error = str(e)
            results["DRGERUND"] = (drgerund, drgerund_error)

        if "DRINF" in active:
            # DRINF
            try:
                drinf = cm_drinf(sentences, noun_chunks, lang, metrics=_dr_metrics)
                drinf_error = None
            except Exception as e:
                logger.error("Error calculating DRINF: %s", e)
                drinf = None
                drinf_error = str(e)
            results["DRINF"] = (drinf, drinf_error)

            ### Word Information

        if "WRD shared" in active:
            # L11: compute WRD* counter dict once and reuse across all 10 WRD* indices.
            try:
                _wrd_counts = _wrd_precompute(sentences)
            except Exception as e:
                logger.error("Error precomputing WRD counts: %s", e)
                _wrd_counts = None

        if "WRDNOUN" in active:
            # WRDNOUN
            try:
                wrdnoun = cm_wrdnoun(sentences, counts=_wrd_counts)
                wrdnoun_error = None
            except Exception as e:
                logger.error("Error calculating WRDNOUN: %s", e)
                wrdnoun = None
                wrdnoun_error = str(e)
            results["WRDNOUN"] = (wrdnoun, wrdnoun_error)

        if "WRDVERB" in active:
            # WRDVERB
            try:
                wrdverb = cm_wrdverb(sentences, counts=_wrd_counts)
                wrdverb_error = None
            except Exception as e:
                logger.error("Error calculating WRDVERB: %s", e)
                wrdverb = None
                wrdverb_error = str(e)
            results["WRDVERB"] = (wrdverb, wrdverb_error)

        if "WRDADJ" in active:
            # WRDADJ
            try:
                wrdadj = cm_wrdadj(sentences, counts=_wrd_counts)
                wrdadj_error = None
            except Exception as e:
                logger.error("Error calculating WRDADJ: %s", e)
                wrdadj = None
                wrdadj_error = str(e)
            results["WRDADJ"] = (wrdadj, wrdadj_error)

        if "WRDADV" in active:
            # WRDADV
            try:
                wrdadv = cm_wrdadv(sentences, counts=_wrd_counts)
                wrdadv_error = None
            except Exception as e:
                logger.error("Error calculating WRDADV: %s", e)
                wrdadv = None
                wrdadv_error = str(e)
            results["WRDADV"] = (wrdadv, wrdadv_error)

        if "WRDPRO" in active:
            # WRDPRO
            try:
                wrdpro = cm_wrdpro(sentences, counts=_wrd_counts)
                wrdpro_error = None
            except Exception as e:
                logger.error("Error calculating WRDPRO: %s", e)
                wrdpro = None
                wrdpro_error = str(e)
            results["WRDPRO"] = (wrdpro, wrdpro_error)

        if "WRDPRP1s" in active:
            # WRDPRP1s
            try:
                wrdprp1s = cm_wrdprp1s(sentences, counts=_wrd_counts)
                wrdprp1s_error = None
            except Exception as e:
                logger.error("Error calculating WRDPRP1s: %s", e)
                wrdprp1s = None
                wrdprp1s_error = str(e)
            results["WRDPRP1s"] = (wrdprp1s, wrdprp1s_error)

        if "WRDPRP1p" in active:
            # WRDPRP1p
            try:
                wrdprp1p = cm_wrdprp1p(sentences, counts=_wrd_counts)
                wrdprp1p_error = None
            except Exception as e:
                logger.error("Error calculating WRDPRP1p: %s", e)
                wrdprp1p = None
                wrdprp1p_error = str(e)
            results["WRDPRP1p"] = (wrdprp1p, wrdprp1p_error)

        if "WRDPRP2" in active:
            # WRDPRP2
            try:
                wrdprp2 = cm_wrdprp2(sentences, counts=_wrd_counts)
                wrdprp2_error = None
            except Exception as e:
                logger.error("Error calculating WRDPRP2: %s", e)
                wrdprp2 = None
                wrdprp2_error = str(e)
            results["WRDPRP2"] = (wrdprp2, wrdprp2_error)

        if "WRDPRP3s" in active:
            # WRDPRP3s
            try:
                wrdprp3s = cm_wrdprp3s(sentences, counts=_wrd_counts)
                wrdprp3s_error = None
            except Exception as e:
                logger.error("Error calculating WRDPRP3s: %s", e)
                wrdprp3s = None
                wrdprp3s_error = str(e)
            results["WRDPRP3s"] = (wrdprp3s, wrdprp3s_error)

        if "WRDPRP3p" in active:
            # WRDPRP3p
            try:
                wrdprp3p = cm_wrdprp3p(sentences, counts=_wrd_counts)
                wrdprp3p_error = None
            except Exception as e:
                logger.error("Error calculating WRDPRP3p: %s", e)
                wrdprp3p = None
                wrdprp3p_error = str(e)
            results["WRDPRP3p"] = (wrdprp3p, wrdprp3p_error)

            # word frequencies: SUBTLEX-DE for German since fix B2 (2026-10-28), the Wikipedia sample for English
            _freq_source = _FREQUENCY_SOURCE.get(lang, "wiki-20220301-sample10000")
            _freq_name = {"subtlex-de": "SUBTLEX-DE", "subtlex-us": "SUBTLEX-US"}.get(_freq_source, "Wikipedia")

        if "WRDFRQc" in active:
            # WRDFRQc
            try:
                wrdfrqc = cm_wrdfrqc(sentences, lang, _freq_source)
                wrdfrqc_error = None
            except Exception as e:
                logger.error("Error calculating WRDFRQc: %s", e)
                wrdfrqc = None
                wrdfrqc_error = str(e)
            results["WRDFRQc"] = (wrdfrqc, wrdfrqc_error)

        if "WRDFRQa" in active:
            # WRDFRQa
            try:
                wrdfrqa = cm_wrdfrqa(tokens, lang, _freq_source)
                wrdfrqa_error = None
            except Exception as e:
                logger.error("Error calculating WRDFRQa: %s", e)
                wrdfrqa = None
                wrdfrqa_error = str(e)
            results["WRDFRQa"] = (wrdfrqa, wrdfrqa_error)

        if "WRDFRQmc" in active:
            # WRDFRQmc
            try:
                wrdfrqmc = cm_wrdfrqmc(sentences, lang, _freq_source)
                wrdfrqmc_error = None
            except Exception as e:
                logger.error("Error calculating WRDFRQmc: %s", e)
                wrdfrqmc = None
                wrdfrqmc_error = str(e)
            results["WRDFRQmc"] = (wrdfrqmc, wrdfrqmc_error)

            # FV1 fix:  load mrc once and reuse across all 5 WRD* indices.

        if "MRC shared" in active:
            try:
                _mrc_dict = _load_norms(lang)    # German: fix B1
            except Exception as e:
                logger.error("Error precomputing MRC dict: %s", e)
                _mrc_dict = None

        if "WRDAOAc" in active:
            # WRDAOAc
            # FV1 fix: Include mrc_dict in function call if loaded successfully
            try:
                wrdaoac = cm_wrdaoac(sentences, lang, mrc_dict=_mrc_dict)
                wrdaoac_error = None
            except Exception as e:
                logger.error("Error calculating WRDAOAc: %s", e)
                wrdaoac = None
                wrdaoac_error = str(e)
            results["WRDAOAc"] = (wrdaoac, wrdaoac_error)

        if "WRDFAMc" in active:
            # WRDFAMc
            # FV1 fix: Include mrc_dict in function call if loaded successfully
            try:
                wrdfamc = cm_wrdfamc(sentences, lang, mrc_dict=_mrc_dict)
                wrdfamc_error = None
            except Exception as e:
                logger.error("Error calculating WRDFAMc: %s", e)
                wrdfamc = None
                wrdfamc_error = str(e)
            results["WRDFAMc"] = (wrdfamc, wrdfamc_error)

        if "WRDCNCc" in active:
            # WRDCNCc
            # FV1 fix: Include mrc_dict in function call if loaded successfully
            try:
                wrdcncc = cm_wrdcncc(sentences, lang, mrc_dict=_mrc_dict)
                wrdcncc_error = None
            except Exception as e:
                logger.error("Error calculating WRDCNCc: %s", e)
                wrdcncc = None
                wrdcncc_error = str(e)
            results["WRDCNCc"] = (wrdcncc, wrdcncc_error)

        if "WRDIMGc" in active:
            # WRDIMGc
            # FV1 fix: Include mrc_dict in function call if loaded successfully
            try:
                wrdimgc = cm_wrdimgc(sentences, lang, mrc_dict=_mrc_dict)
                wrdimgc_error = None
            except Exception as e:
                logger.error("Error calculating WRDIMGc: %s", e)
                wrdimgc = None
                wrdimgc_error = str(e)
            results["WRDIMGc"] = (wrdimgc, wrdimgc_error)

        if "WRDMEAc" in active:
            # WRDMEAc
            # FV1 fix: Include mrc_dict in function call if loaded successfully
            try:
                wrdmeac = cm_wrdmeac(sentences, lang, mrc_dict=_mrc_dict)
                wrdmeac_error = None
            except Exception as e:
                logger.error("Error calculating WRDMEAc: %s", e)
                wrdmeac = None
                wrdmeac_error = str(e)
            results["WRDMEAc"] = (wrdmeac, wrdmeac_error)

        if "WRDPOLc" in active:
            # WRDPOLc
            try:
                wrdpolc = cm_wrdpolc(sentences, lang)
                wrdpolc_error = None
            except Exception as e:
                logger.error("Error calculating WRDPOLc: %s", e)
                wrdpolc = None
                wrdpolc_error = str(e)
            results["WRDPOLc"] = (wrdpolc, wrdpolc_error)

            # Precompute hypernymy values once for WRDHYPn, WRDHYPv and WRDHYPnv.

        if "HYP shared" in active:
            try:
                _wrdhyp = _calc_wrdhyp(
                    sentences,
                    lang
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

        if "WRDHYPn" in active:
            # WRDHYPn
            wrdhypn = _wrdhyp["WRDHYPn"]
            wrdhypn_error = _wrdhyp_error
            results["WRDHYPn"] = (wrdhypn, wrdhypn_error)

        if "WRDHYPv" in active:
            # WRDHYPv
            wrdhypv = _wrdhyp["WRDHYPv"]
            wrdhypv_error = _wrdhyp_error
            results["WRDHYPv"] = (wrdhypv, wrdhypv_error)

        if "WRDHYPnv" in active:
            # WRDHYPnv
            wrdhypnv = _wrdhyp["WRDHYPnv"]
            wrdhypnv_error = _wrdhyp_error
            results["WRDHYPnv"] = (wrdhypnv, wrdhypnv_error)

            ### Readability

            # RDFRE and RDFKGL were calculated through textstat on the raw
            # document text; Coh-Metrix 3.0 defines them from its own DESSL and DESWLsy.
            # Since fix A2 (2026-10-28) all readability formulas use the request's own
            # sentences, words and syllables (_readability_scores). Labels (user
            # 2026-10-28): RDFRE and RDFKGL now follow Coh-Metrix 3.0 and carry its
            # labels; the 11 formulas outside 3.0 carry "_ext" (upstream rule 7, our
            # LABEL_SUGGESTION.md) instead of "_textstat".

        if "RD shared" in active:
            try:
                _rd = _readability_scores(request.paragraphs, lang) if has_readability_input else {}
                _rd_error = None
            except Exception as e:
                logger.error("Error calculating the readability formulas: %s", e)
                _rd, _rd_error = {}, str(e)

        if "RDFRE" in active:
            # RDFRE
            try:
                rdfre = _rd.get("FRE")
                rdfre_error = _rd_error
            except Exception as e:
                logger.error("Error calculating RDFRE: %s", e)
                rdfre = None
                rdfre_error = str(e)
            results["RDFRE"] = (rdfre, rdfre_error)

        if "RDFKGL" in active:
            # RDFKGL
            try:
                rdfkgl = _rd.get("FKGL")
                rdfkgl_error = _rd_error
            except Exception as e:
                logger.error("Error calculating RDFKGL: %s", e)
                rdfkgl = None
                rdfkgl_error = str(e)
            results["RDFKGL"] = (rdfkgl, rdfkgl_error)

        if "RDFOG" in active:
            # RDFOG
            try:
                rdfog = _rd.get("FOG")
                rdfog_error = _rd_error
            except Exception as e:
                logger.error("Error calculating RDFOG: %s", e)
                rdfog = None
                rdfog_error = str(e)
            results["RDFOG"] = (rdfog, rdfog_error)

        if "RDSMOG" in active:
            # RDSMOG
            try:
                rdsmog = _rd.get("SMOG")
                rdsmog_error = _rd_error
            except Exception as e:
                logger.error("Error calculating RDSMOG: %s", e)
                rdsmog = None
                rdsmog_error = str(e)
            results["RDSMOG"] = (rdsmog, rdsmog_error)

        if "RDARI" in active:
            # RDARI
            try:
                rdari = _rd.get("ARI")
                rdari_error = _rd_error
            except Exception as e:
                logger.error("Error calculating RDARI: %s", e)
                rdari = None
                rdari_error = str(e)
            results["RDARI"] = (rdari, rdari_error)

        if "RDCLI" in active:
            # RDCLI
            try:
                rdcli = _rd.get("CLI")
                rdcli_error = _rd_error
            except Exception as e:
                logger.error("Error calculating RDCLI: %s", e)
                rdcli = None
                rdcli_error = str(e)
            results["RDCLI"] = (rdcli, rdcli_error)

        if "RDLW" in active:
            # RDLW
            try:
                rdlw = _rd.get("LW")
                rdlw_error = _rd_error
            except Exception as e:
                logger.error("Error calculating RDLW: %s", e)
                rdlw = None
                rdlw_error = str(e)
            results["RDLW"] = (rdlw, rdlw_error)

        if "RDDCRS" in active:
            # RDDCRS
            try:
                rddcrs = _rd.get("DCRS")
                rddcrs_error = _rd_error
            except Exception as e:
                logger.error("Error calculating RDDCRS: %s", e)
                rddcrs = None
                rddcrs_error = str(e)
            results["RDDCRS"] = (rddcrs, rddcrs_error)

        if "RDSPACHE" in active:
            # RDSPACHE
            try:
                rdspache = _rd.get("SPACHE")
                rdspache_error = _rd_error
            except Exception as e:
                logger.error("Error calculating RDSPACHE: %s", e)
                rdspache = None
                rdspache_error = str(e)
            results["RDSPACHE"] = (rdspache, rdspache_error)

        if "RDWSTF1" in active:
            # RDWSTF1
            try:
                rdwstf = _rd.get("WSTF1") if lang == "de" else None
                rdwstf_error = _rd_error
            except Exception as e:
                logger.error("Error calculating RDWSTF1: %s", e)
                rdwstf = None
                rdwstf_error = str(e)
            results["RDWSTF1"] = (rdwstf, rdwstf_error)

        if "RDWSTF2" in active:
            # RDWSTF2
            try:
                rdwstf = _rd.get("WSTF2") if lang == "de" else None
                rdwstf_error = _rd_error
            except Exception as e:
                logger.error("Error calculating RDWSTF2: %s", e)
                rdwstf = None
                rdwstf_error = str(e)
            results["RDWSTF2"] = (rdwstf, rdwstf_error)

        if "RDWSTF3" in active:
            # RDWSTF3
            try:
                rdwstf = _rd.get("WSTF3") if lang == "de" else None
                rdwstf_error = _rd_error
            except Exception as e:
                logger.error("Error calculating RDWSTF3: %s", e)
                rdwstf = None
                rdwstf_error = str(e)
            results["RDWSTF3"] = (rdwstf, rdwstf_error)

        if "RDWSTF4" in active:
            # RDWSTF4
            try:
                rdwstf = _rd.get("WSTF4") if lang == "de" else None
                rdwstf_error = _rd_error
            except Exception as e:
                logger.error("Error calculating RDWSTF4: %s", e)
                rdwstf = None
                rdwstf_error = str(e)
            results["RDWSTF4"] = (rdwstf, rdwstf_error)

        if "RDL2" in active:
            # RDL2
            try:
                # 2026-10-28 (user): the frequency term is WRDFRQc (log since then), not WRDFRQmc; RDL2_synstrutt removed
                rdl2 = cm_rdl2(crfcwo1, synstruta, wrdfrqc)
                rdl2_error = None
            except Exception as e:
                logger.error("Error calculating RDL2: %s", e)
                rdl2 = None
                rdl2_error = str(e)
            results["RDL2"] = (rdl2, rdl2_error)


        # ---------------------------------------------------------------- variants (decision 2026-11-02, README rules 4, 7)
        if "WRDFRQ@WIKI" in active:
            # the Wikipedia samples shipped with the component, log10 as the main frequencies
            _wiki = "wiki-20220301-sample10000"
            for _key, _fn, _arg in (("WRDFRQc", cm_wrdfrqc, sentences), ("WRDFRQa", cm_wrdfrqa, tokens),
                                    ("WRDFRQmc", cm_wrdfrqmc, sentences)):
                try:
                    results[_key + "@WIKI"] = (_fn(_arg, lang, _wiki), None)
                except Exception as e:
                    logger.error("Error calculating %s (Wikipedia): %s", _key, e)
                    results[_key + "@WIKI"] = (None, str(e))

        if "RDL2@WIKI" in active:
            try:
                results["RDL2@WIKI"] = (cm_rdl2(crfcwo1, synstruta, results["WRDFRQc@WIKI"][0]), None)
            except Exception as e:
                logger.error("Error calculating RDL2 (Wikipedia): %s", e)
                results["RDL2@WIKI"] = (None, str(e))

        if "CNC@LLMlist" in active:
            # the 0.2.0 lists and counting on the text (connectives_llmlist, _count_connectives_llmlist)
            try:
                _cnc_llm, _cnc_llm_error = _count_connectives_llmlist(request.text, lang, tokens_count), None
            except Exception as e:
                logger.error("Error calculating the connectives (LLM lists): %s", e)
                _cnc_llm, _cnc_llm_error = {}, str(e)
            for _key in ("CNCAll", "CNCCaus", "CNCLogic", "CNCADC", "CNCTemp", "CNCTempx", "CNCAdd", "CNCPos", "CNCNeg"):
                results[_key + "@LLMlist"] = (_cnc_llm.get("CNCTempX" if _key == "CNCTempx" else _key), _cnc_llm_error)

        for _key, _fn in (("SMCAUSvp", cm_smcausvp), ("SMCAUSr", cm_smcausr)):
            if _key + "@LLMlist" in active:
                try:
                    results[_key + "@LLMlist"] = (_fn(sentences, lang, particles="llmlist"), None)
                except Exception as e:
                    logger.error("Error calculating %s (LLM list): %s", _key, e)
                    results[_key + "@LLMlist"] = (None, str(e))

        if "WRD@MRC" in active:
            # the MRC database (German: translated) for age of acquisition, concreteness and imageability
            try:
                _mrc = _load_mrc_database(lang)
                _mrc_error = None
            except Exception as e:
                logger.error("Error loading the MRC database: %s", e)
                _mrc, _mrc_error = None, str(e)
            for _key, _fn in (("WRDAOAc", cm_wrdaoac), ("WRDCNCc", cm_wrdcncc), ("WRDIMGc", cm_wrdimgc)):
                try:
                    results[_key + "@MRC"] = (_fn(sentences, lang, mrc_dict=_mrc), _mrc_error) if _mrc is not None \
                        else (None, _mrc_error)
                except Exception as e:
                    logger.error("Error calculating %s (MRC): %s", _key, e)
                    results[_key + "@MRC"] = (None, str(e))

        if "RD@TEXTSTAT" in active:
            # the readability formulas of the textstat library on the text, as in duui-coh-metrix 0.2.0 (textstat keeps the
            # language in module state, so the calls run under one lock)
            _textstat = {"RDFRE": textstat.flesch_reading_ease, "RDFKGL": textstat.flesch_kincaid_grade,
                         "RDFOG": textstat.gunning_fog, "RDSMOG": textstat.smog_index,
                         "RDARI": textstat.automated_readability_index, "RDCLI": textstat.coleman_liau_index,
                         "RDLW": textstat.linsear_write_formula, "RDDCRS": textstat.dale_chall_readability_score,
                         "RDSPACHE": textstat.spache_readability,
                         **{f"RDWSTF{v}": (lambda text, v=v: textstat.wiener_sachtextformel(text, variant=v)) for v in (1, 2, 3, 4)}}
            with _TEXTSTAT_LOCK:
                textstat.set_lang(lang)
                for _key, _fn in _textstat.items():
                    try:
                        results[_key + "@TEXTSTAT"] = (_fn(request.text) if has_readability_input else None, None)
                    except Exception as e:
                        logger.error("Error calculating %s (textstat): %s", _key, e)
                        results[_key + "@TEXTSTAT"] = (None, str(e))

    except Exception as ex:
        logger.exception(ex)
        for key in (wanted if wanted is not None else KEY_BLOCK):
            results.setdefault(key, (None, f"not computed: {ex}"))
    finally:
        # the per-request caches hold the sentences with their vectors (large for long texts)
        _SENTENCE_FEATURES.reset()
        _TEXT_MEMO.reset()
    return results


# the block of each index key, and the blocks each block needs (derived from the variables they share)
KEY_BLOCK = {
    "DESPC": "DESPC",
    "DESSC": "DESSC",
    "DESWC": "DESWC",
    "DESPL": "DESPL",
    "DESPLd": "DESPLd",
    "DESSL": "DESSL",
    "DESSLd": "DESSLd",
    "DESWLsy": "DESWLsy",
    "DESWLsyd": "DESWLsyd",
    "DESWLlt": "DESWLlt",
    "DESWLltd": "DESWLltd",
    "CRFNO1": "CRFNO1",
    "CRFAO1": "CRFAO1",
    "CRFSO1": "CRFSO1",
    "CRFNOa": "CRFNOa",
    "CRFAOa": "CRFAOa",
    "CRFSOa": "CRFSOa",
    "CRFCWO1": "CRFCWO1",
    "CRFCWO1d": "CRFCWO1d",
    "CRFCWOa": "CRFCWOa",
    "CRFCWOad": "CRFCWOad",
    "LSASS1": "LSASS1",
    "LSASS1d": "LSASS1d",
    "LSASSp": "LSASSp",
    "LSASSpd": "LSASSpd",
    "LSAPP1": "LSAPP1",
    "LSAPP1d": "LSAPP1d",
    "LSAGN": "LSAGN",
    "LSAGNd": "LSAGNd",
    "LDTTRc": "LDTTRc",
    "LDTTRa": "LDTTRa",
    "LDMTLDa": "LDMTLDa",
    "LDVOCDa": "LDVOCDa",
    "CNCAll": "CNCAll",
    "CNCCaus": "CNCCaus",
    "CNCLogic": "CNCLogic",
    "CNCADC": "CNCADC",
    "CNCTemp": "CNCTemp",
    "CNCTempx": "CNCTempx",
    "CNCAdd": "CNCAdd",
    "CNCPos": "CNCPos",
    "CNCNeg": "CNCNeg",
    "SMCAUSv": "SMCAUSv",
    "SMCAUSvp": "SMCAUSvp",
    "SMINTEp": "SMINTEp",
    "SMCAUSr": "SMCAUSr",
    "SMINTEr": "SMINTEr",
    "SMCAUSlsa": "SMCAUSlsa",
    "SMCAUSwn": "SMCAUSwn",
    "SMTEMP": "SMTEMP",
    "SYNLE": "SYNLE",
    "SYNNP": "SYNNP",
    "SYNMEDpos": "SYNMEDpos",
    "SYNMEDwrd": "SYNMEDwrd",
    "SYNMEDlem": "SYNMEDlem",
    "SYNSTRUTa": "SYNSTRUTa",
    "SYNSTRUTt": "SYNSTRUTt",
    "DRNP": "DRNP",
    "DRVP": "DRVP",
    "DRAP": "DRAP",
    "DRPP": "DRPP",
    "DRPVAL": "DRPVAL",
    "DRNEG": "DRNEG",
    "DRGERUND": "DRGERUND",
    "DRINF": "DRINF",
    "WRDNOUN": "WRDNOUN",
    "WRDVERB": "WRDVERB",
    "WRDADJ": "WRDADJ",
    "WRDADV": "WRDADV",
    "WRDPRO": "WRDPRO",
    "WRDPRP1s": "WRDPRP1s",
    "WRDPRP1p": "WRDPRP1p",
    "WRDPRP2": "WRDPRP2",
    "WRDPRP3s": "WRDPRP3s",
    "WRDPRP3p": "WRDPRP3p",
    "WRDFRQc": "WRDFRQc",
    "WRDFRQa": "WRDFRQa",
    "WRDFRQmc": "WRDFRQmc",
    "WRDAOAc": "WRDAOAc",
    "WRDFAMc": "WRDFAMc",
    "WRDCNCc": "WRDCNCc",
    "WRDIMGc": "WRDIMGc",
    "WRDMEAc": "WRDMEAc",
    "WRDPOLc": "WRDPOLc",
    "WRDHYPn": "WRDHYPn",
    "WRDHYPv": "WRDHYPv",
    "WRDHYPnv": "WRDHYPnv",
    "RDFRE": "RDFRE",
    "RDFKGL": "RDFKGL",
    "RDFOG": "RDFOG",
    "RDSMOG": "RDSMOG",
    "RDARI": "RDARI",
    "RDCLI": "RDCLI",
    "RDLW": "RDLW",
    "RDDCRS": "RDDCRS",
    "RDSPACHE": "RDSPACHE",
    "RDWSTF1": "RDWSTF1",
    "RDWSTF2": "RDWSTF2",
    "RDWSTF3": "RDWSTF3",
    "RDWSTF4": "RDWSTF4",
    "RDL2": "RDL2"
}
BLOCK_DEPS = {
    "DESPC": [],
    "DESSC": [],
    "DESWC": [],
    "DESPL": [],
    "DESPLd": [],
    "DESSL": [],
    "DESSLd": [],
    "DESWLsy": [],
    "DESWLsyd": [],
    "DESWLlt": [],
    "DESWLltd": [],
    "CRFNO1": [],
    "CRFAO1": [],
    "CRFSO1": [],
    "CRFNOa": [],
    "CRFAOa": [],
    "CRFSOa": [],
    "CRFCWO1": [],
    "CRFCWO1d": [],
    "CRFCWOa": [],
    "CRFCWOad": [],
    "LSA shared": [],
    "LSASS1": [
        "LSA shared"
    ],
    "LSASS1d": [
        "LSA shared"
    ],
    "LSASSp": [
        "LSA shared"
    ],
    "LSASSpd": [
        "LSA shared"
    ],
    "LSAPP1": [
        "LSA shared"
    ],
    "LSAPP1d": [
        "LSA shared"
    ],
    "LSAGN": [
        "LSA shared"
    ],
    "LSAGNd": [
        "LSA shared"
    ],
    "LDTTRc": [],
    "LDTTRa": [],
    "LDMTLDa": [],
    "LDVOCDa": [],
    "CNC shared": [],
    "CNCAll": [
        "CNC shared"
    ],
    "CNCCaus": [
        "CNC shared"
    ],
    "CNCLogic": [
        "CNC shared"
    ],
    "CNCADC": [
        "CNC shared"
    ],
    "CNCTemp": [
        "CNC shared"
    ],
    "CNCTempx": [
        "CNC shared"
    ],
    "CNCAdd": [
        "CNC shared"
    ],
    "CNCPos": [
        "CNC shared"
    ],
    "CNCNeg": [
        "CNC shared"
    ],
    "SMCAUSv": [],
    "SMCAUSvp": [],
    "SMINTEp": [],
    "SMCAUSr": [],
    "SMINTEr": [],
    "SMCAUSlsa": [],
    "SMCAUSwn": [],
    "SMTEMP": [],
    "SYNLE": [],
    "SYNNP": [],
    "SYNMEDpos": [],
    "SYNMEDwrd": [],
    "SYNMEDlem": [],
    "SYNSTRUTa": [],
    "SYNSTRUTt": [
        "SYNSTRUTa"
    ],
    "DR shared": [],
    "DRNP": [
        "DR shared"
    ],
    "DRVP": [
        "DR shared"
    ],
    "DRAP": [
        "DR shared"
    ],
    "DRPP": [
        "DR shared"
    ],
    "DRPVAL": [
        "DR shared"
    ],
    "DRNEG": [
        "DR shared"
    ],
    "DRGERUND": [
        "DR shared"
    ],
    "DRINF": [
        "DR shared"
    ],
    "WRD shared": [],
    "WRDNOUN": [
        "WRD shared"
    ],
    "WRDVERB": [
        "WRD shared"
    ],
    "WRDADJ": [
        "WRD shared"
    ],
    "WRDADV": [
        "WRD shared"
    ],
    "WRDPRO": [
        "WRD shared"
    ],
    "WRDPRP1s": [
        "WRD shared"
    ],
    "WRDPRP1p": [
        "WRD shared"
    ],
    "WRDPRP2": [
        "WRD shared"
    ],
    "WRDPRP3s": [
        "WRD shared"
    ],
    "WRDPRP3p": [
        "WRD shared"
    ],
    "WRDFRQc": [
        "WRDPRP3p"
    ],
    "WRDFRQa": [
        "WRDPRP3p"
    ],
    "WRDFRQmc": [
        "WRDPRP3p"
    ],
    "MRC shared": [],
    "WRDAOAc": [
        "MRC shared"
    ],
    "WRDFAMc": [
        "MRC shared"
    ],
    "WRDCNCc": [
        "MRC shared"
    ],
    "WRDIMGc": [
        "MRC shared"
    ],
    "WRDMEAc": [
        "MRC shared"
    ],
    "WRDPOLc": [],
    "HYP shared": [],
    "WRDHYPn": [
        "HYP shared"
    ],
    "WRDHYPv": [
        "HYP shared"
    ],
    "WRDHYPnv": [
        "HYP shared"
    ],
    "RD shared": [],
    "RDFRE": [
        "RD shared"
    ],
    "RDFKGL": [
        "RD shared"
    ],
    "RDFOG": [
        "RD shared"
    ],
    "RDSMOG": [
        "RD shared"
    ],
    "RDARI": [
        "RD shared"
    ],
    "RDCLI": [
        "RD shared"
    ],
    "RDLW": [
        "RD shared"
    ],
    "RDDCRS": [
        "RD shared"
    ],
    "RDSPACHE": [
        "RD shared"
    ],
    "RDWSTF1": [
        "RD shared"
    ],
    "RDWSTF2": [
        "RD shared"
    ],
    "RDWSTF3": [
        "RD shared"
    ],
    "RDWSTF4": [
        "RD shared"
    ],
    "RDL2": [
        "CRFCWO1",
        "SYNSTRUTa",
        "WRDFRQc"
    ]
}

# the variants (computed in their own blocks at the end of compute)
KEY_BLOCK.update({f"{k}@WIKI": "WRDFRQ@WIKI" for k in ("WRDFRQc", "WRDFRQa", "WRDFRQmc")})
KEY_BLOCK["RDL2@WIKI"] = "RDL2@WIKI"
KEY_BLOCK.update({f"{k}@LLMlist": "CNC@LLMlist" for k in ("CNCAll", "CNCCaus", "CNCLogic", "CNCADC", "CNCTemp", "CNCTempx",
                                                           "CNCAdd", "CNCPos", "CNCNeg")})
KEY_BLOCK.update({"SMCAUSvp@LLMlist": "SMCAUSvp@LLMlist", "SMCAUSr@LLMlist": "SMCAUSr@LLMlist"})
KEY_BLOCK.update({f"{k}@MRC": "WRD@MRC" for k in ("WRDAOAc", "WRDCNCc", "WRDIMGc")})
KEY_BLOCK.update({f"{k}@TEXTSTAT": "RD@TEXTSTAT" for k in ("RDFRE", "RDFKGL", "RDFOG", "RDSMOG", "RDARI", "RDCLI", "RDLW",
                                                            "RDDCRS", "RDSPACHE", "RDWSTF1", "RDWSTF2", "RDWSTF3", "RDWSTF4")})
BLOCK_DEPS.update({"WRDFRQ@WIKI": [], "RDL2@WIKI": ["CRFCWO1", "SYNSTRUTa", "WRDFRQ@WIKI"], "CNC@LLMlist": [],
                   "SMCAUSvp@LLMlist": [], "SMCAUSr@LLMlist": [], "WRD@MRC": [], "RD@TEXTSTAT": []})

_TEXTSTAT_LOCK = threading.Lock()


def _active_blocks(wanted: Optional[Set[str]]) -> Set[str]:
    """the blocks to run for the wanted keys: their own blocks and everything these need, transitively"""
    todo = list(KEY_BLOCK.values()) if wanted is None else [KEY_BLOCK[k] for k in wanted if k in KEY_BLOCK]
    active = set()
    while todo:
        block = todo.pop()
        if block not in active:
            active.add(block)
            todo.extend(BLOCK_DEPS.get(block, []))
    return active

