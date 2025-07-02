import logging
from platform import python_version
from sys import version as sys_version
from time import time
from typing import List, Optional, Dict, Tuple

from cassis import load_typesystem
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from lexicalrichness import LexicalRichness
from lexical_diversity import lex_div as ld
from similarity.normalized_levenshtein import NormalizedLevenshtein

import numpy as np

class Settings(BaseSettings):
    annotator_name: str
    annotator_version: str
    log_level: str

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
    pos_value: str   # spacy tag_
    pos_coarse: str  # spacy pos_
    lemma: str
    is_alpha: bool
    dep_type: str


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
    label_v3: str
    label_v2: str
    description: str
    value: Optional[float]  # can be None if not applicable or on error
    error: Optional[str]    # fill with error message if applicable


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

def cm_despc(paragraphs: List[Paragraph]) -> Optional[float]:
    # Paragraph count, number of paragraphs
    return len(paragraphs)

def cm_dessc(paragraphs: List[Paragraph]) -> Optional[float]:
    # Sentence count, number of sentences
    return sum([len(p.sentences) for p in paragraphs])

def cm_deswc(paragraphs: List[Paragraph]) -> Optional[float]:
    # Word count, number of words
    return sum([sum([len(s.tokens) for s in p.sentences]) for p in paragraphs])

def cm_despl(paragraphs: List[Paragraph]) -> Optional[float]:
    # Paragraph length, number of sentences, mean
    return np.mean([len(p.sentences) for p in paragraphs])

def cm_despld(paragraphs: List[Paragraph]) -> Optional[float]:
    # Paragraph length, number of sentences, standard deviation
    return np.std([len(p.sentences) for p in paragraphs])

def cm_dessl(paragraphs: List[Paragraph]) -> Optional[float]:
    # Sentence length, number of words, mean
    return np.mean([len(s.tokens) for p in paragraphs for s in p.sentences])

def cm_dessld(paragraphs: List[Paragraph]) -> Optional[float]:
    # Sentence length, number of words, standard deviation
    return np.std([len(s.tokens) for p in paragraphs for s in p.sentences])

def cm_deswlsy(paragraphs: List[Paragraph]) -> Optional[float]:
    # Word length, number of syllables, mean
    # TODO
    return None

def cm_deswlsyd(paragraphs: List[Paragraph]) -> Optional[float]:
    # Word length, number of syllables, standard deviation
    # TODO
    return None

def cm_deswllt(paragraphs: List[Paragraph]) -> Optional[float]:
    # Word length, number of letters, mean
    text_letters = []
    for p in paragraphs:
        for s in p.sentences:
            for t in s.tokens:
                text_letters.append(len(''.join(c for c in t.text if c.isalpha())))
    return np.mean(text_letters)

def cm_deswlltd(paragraphs: List[Paragraph]) -> Optional[float]:
    # Word length, number of letters, standard deviation
    text_letters = []
    for p in paragraphs:
        for s in p.sentences:
            for t in s.tokens:
                text_letters.append(len(''.join(c for c in t.text if c.isalpha())))
    return np.std(text_letters)

def _noun_overlap(sentence_a: Sentence, sentence_b: Sentence) -> int:
    nouns_a = set([t.text for t in sentence_a.tokens if t.pos_value and t.pos_value.startswith('N')])
    nouns_b = set([t.text for t in sentence_b.tokens if t.pos_value and t.pos_value.startswith('N')])
    return len(nouns_a.intersection(nouns_b))

def _argument_overlap(sentence_a: Sentence, sentence_b: Sentence) -> int:
    nouns_a = set([t.lemma for t in sentence_a.tokens if t.pos_value and t.pos_value.startswith('N')])
    nouns_b = set([t.lemma for t in sentence_b.tokens if t.pos_value and t.pos_value.startswith('N')])
    noun_overlap = len(nouns_a.intersection(nouns_b))

    promouns_a = set([t.text for t in sentence_a.tokens if t.pos_value and t.pos_value == 'PPER'])
    promouns_b = set([t.text for t in sentence_b.tokens if t.pos_value and t.pos_value == 'PPER'])
    pronoun_overlap = len(promouns_a.intersection(promouns_b))

    return noun_overlap + pronoun_overlap

def _stem_overlap(sentence_nouns: Sentence, sentence_contents: Sentence) -> int:
    nouns_a = set([t.lemma for t in sentence_nouns.tokens if t.pos_value and t.pos_value.startswith('N')])
    nouns_b = set([t.lemma for t in sentence_contents.tokens if t.pos_value and (
        t.pos_value.startswith('N') or t.pos_value.startswith('V') or t.pos_value.startswith("ADJ") or t.pos_value.startswith("ADV")
    )])
    return len(nouns_a.intersection(nouns_b))

def _word_overlap(sentence_nouns: Sentence, sentence_contents: Sentence) -> float:
    nouns_a = set([t.lemma for t in sentence_nouns.tokens if t.pos_value and t.pos_value.startswith('N')])
    nouns_b = set([t.lemma for t in sentence_contents.tokens if t.pos_value and (
            t.pos_value.startswith('N') or t.pos_value.startswith('V') or t.pos_value.startswith("ADJ") or t.pos_value.startswith("ADV")
    )])
    overlap = len(nouns_a.intersection(nouns_b))

    # TODO check
    all_words = set([t.lemma for t in sentence_nouns.tokens])
    all_words.update(set([t.lemma for t in sentence_contents.tokens]))

    return overlap / len(all_words) if len(all_words) > 0 else 0.0

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

def _lexical_diversity_tokens(tokens: List[Token]) -> Tuple[List[str], List[str]]:
    tokens_alpha = [token.text.lower() for token in tokens if token.is_alpha]
    content_pos = {"NOUN", "VERB", "ADJ", "ADV"}
    tokens_content = [token.text.lower() for token in tokens if token.pos_coarse in content_pos and token.is_alpha]
    return tokens_alpha, tokens_content

def cm_ldttrc(tokens: List[Token]) -> Optional[float]:
    _, tokens_content = _lexical_diversity_tokens(tokens)
    return ld.ttr(tokens_content)

def cm_ldttra(tokens: List[Token]) -> Optional[float]:
    tokens_alpha, _ = _lexical_diversity_tokens(tokens)
    return ld.ttr(tokens_alpha)

def cm_ldmtlda(tokens: List[Token]) -> Optional[float]:
    tokens_alpha, _ = _lexical_diversity_tokens(tokens)
    return ld.mtld(tokens_alpha)

def cm_ldvocda(tokens: List[Token]) -> Optional[float]:
    tokens_alpha, _ = _lexical_diversity_tokens(tokens)
    lex = LexicalRichness(tokens_alpha, preprocessor=None, tokenizer=None)
    return lex.vocd()

def cm_synle(sentences: List[Sentence]) -> Optional[float]:
    deps = [[token.dep_type for token in sent.tokens] for sent in sentences]

    word_counts = []
    counter_start = 0
    for sent in deps:
        root = [c for c, token in enumerate(sent) if token == "--"]  # ROOT in spaCy
        if root:
            root_index = (counter_start+root[0]) - counter_start
            word_counts.append(root_index)
        counter_start += len(sent)

    return np.mean(word_counts) if word_counts else 0

def cm_synnp(sentences: List[Sentence], noun_chunks: List[NounChunk]) -> Optional[float]:
    deps = []
    for noun_chunk in noun_chunks:
        for sentence in sentences:
            if sentence.begin == noun_chunk.begin and sentence.end == noun_chunk.end:
                deps.append([token.dep_type for token in sentence.tokens])
                break

    modifier_counts = []
    for sent in deps:
        modifiers = [tok for tok in sent if tok in ("AMOD", "COMPOUND", "PREP")]
        modifier_counts.append(len(modifiers))

    return np.mean(modifier_counts) if modifier_counts else 0

def cm_synmedpos(tokens: List[Token]) -> Optional[float]:
    poses = [token.pos_coarse for token in tokens]

    normalized_levenshtein = NormalizedLevenshtein()

    pos_dists = []
    for i in range(len(poses)-1):
        pos_i = poses[i]
        pos_j = poses[i+1]
        pos_dists.append(normalized_levenshtein.distance(pos_i, pos_j))

    return np.mean(pos_dists)

def cm_synmedwrd(tokens: List[Token]) -> Optional[float]:
    tokens = [token.text for token in tokens]

    normalized_levenshtein = NormalizedLevenshtein()

    word_dists = []
    for i in range(len(tokens)-1):
        tokens_i = tokens[i]
        tokens_j = tokens[i+1]
        word_dists.append(normalized_levenshtein.distance(tokens_i, tokens_j))

    return np.mean(word_dists)

def cm_synmedlem(tokens: List[Token]) -> Optional[float]:
    lemmas = [token.lemma for token in tokens]

    normalized_levenshtein = NormalizedLevenshtein()

    lemma_dists = []
    for i in range(len(lemmas)-1):
        lemmas_i = lemmas[i]
        lemmas_j = lemmas[i+1]
        lemma_dists.append(normalized_levenshtein.distance(lemmas_i, lemmas_j))

    return np.mean(lemma_dists)

@app.post("/v1/process")
def post_process(request: TextImagerRequest) -> TextImagerResponse:
    modification_timestamp_seconds = int(time())

    indices = []
    meta = None
    modification_meta = None

    try:
        sentences = []
        for p in request.paragraphs:
            sentences.extend(p.sentences)

        tokens = []
        for s in sentences:
            tokens.extend(s.tokens)

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
            deswlsy = cm_deswlsy(request.paragraphs)
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
            deswlsyd = cm_deswlsyd(request.paragraphs)
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
            pcnarz_error = None
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
            pcnarp_error = None
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
            pcsynz_error = None
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
            pcsynp_error = None
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
            pccncz_error = None
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
            pccncp_error = None
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
            pcrefz_error = None
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
            pcrefp_error = None
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
            pcdcz_error = None
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
            pcdcp_error = None
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
            pcverbz_error = None
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
            pcverbp_error = None
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
            pcconnz_error = None
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
            pcconnp_error = None
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
            pctempz_error = None
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
            pctempp_error = None
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

        # LSASS1
        try:
            #lsass1 = cm_lsass1(sentences)
            lsass1 = None
            lsass1_error = None
        except Exception as e:
            logger.error("Error calculating LSASS1: %s", e)
            lsass1 = None
            lsass1_error = str(e)
        indices.append(Index(
            index=38,
            type_name="LSA",
            label_v3="LSASS1",
            label_v2="LSAassa",
            description="LSA overlap, adjacent sentences, mean",
            value=lsass1,
            error=lsass1_error
        ))

        # LSASS1d
        try:
            #lsass1d = cm_lsass1d(sentences)
            lsass1d = None
            lsass1d_error = None
        except Exception as e:
            logger.error("Error calculating LSASS1d: %s", e)
            lsass1d = None
            lsass1d_error = str(e)
        indices.append(Index(
            index=39,
            type_name="LSA",
            label_v3="LSASS1d",
            label_v2="LSAassd",
            description="LSA overlap, adjacent sentences, standard deviation",
            value=lsass1d,
            error=lsass1d_error
        ))

        # LSASSp
        try:
            #lsassp = cm_lsassp(sentences)
            lsassp = None
            lsassp_error = None
        except Exception as e:
            logger.error("Error calculating LSASSp: %s", e)
            lsassp = None
            lsassp_error = str(e)
        indices.append(Index(
            index=40,
            type_name="LSA",
            label_v3="LSASSp",
            label_v2="LSApssa",
            description="LSA overlap, all sentences in paragraph, mean",
            value=lsassp,
            error=lsassp_error
        ))

        # LSASSpd
        try:
            #lsasspd = cm_lsasspd(sentences)
            lsasspd = None
            lsasspd_error = None
        except Exception as e:
            logger.error("Error calculating LSASSpd: %s", e)
            lsasspd = None
            lsasspd_error = str(e)
        indices.append(Index(
            index=41,
            type_name="LSA",
            label_v3="LSASSpd",
            label_v2="LSApssd",
            description="LSA overlap, all sentences in paragraph, standard deviation",
            value=lsasspd,
            error=lsasspd_error
        ))

        # LSAPP1
        try:
            #lsapp1 = cm_lsapp1(sentences)
            lsapp1 = None
            lsapp1_error = None
        except Exception as e:
            logger.error("Error calculating LSAPP1: %s", e)
            lsapp1 = None
            lsapp1_error = str(e)
        indices.append(Index(
            index=42,
            type_name="LSA",
            label_v3="LSAPP1",
            label_v2="LSAppa",
            description="LSA overlap, adjacent paragraphs, mean",
            value=lsapp1,
            error=lsapp1_error
        ))

        # LSAPP1d
        try:
            #lsapp1d = cm_lsapp1d(sentences)
            lsapp1d = None
            lsapp1d_error = None
        except Exception as e:
            logger.error("Error calculating LSAPP1d: %s", e)
            lsapp1d = None
            lsapp1d_error = str(e)
        indices.append(Index(
            index=43,
            type_name="LSA",
            label_v3="LSAPP1d",
            label_v2="LSAppd",
            description="LSA overlap, adjacent paragraphs, standard deviation",
            value=lsapp1d,
            error=lsapp1d_error
        ))

        # LSAGN
        try:
            #lsagn = cm_lsagn(sentences)
            lsagn = None
            lsagn_error = None
        except Exception as e:
            logger.error("Error calculating LSAGN: %s", e)
            lsagn = None
            lsagn_error = str(e)
        indices.append(Index(
            index=44,
            type_name="LSA",
            label_v3="LSAGN",
            label_v2="LSAGN",
            description="LSA given/new, sentences, mean",
            value=lsagn,
            error=lsagn_error
        ))

        # LSAGNd
        try:
            #lsagnd = cm_lsagnd(sentences)
            lsagnd = None
            lsagnd_error = None
        except Exception as e:
            logger.error("Error calculating LSAGNd: %s", e)
            lsagnd = None
            lsagnd_error = str(e)
        indices.append(Index(
            index=45,
            type_name="LSA",
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

        # CNCAll
        try:
            #cncall = cm_cncall(sentences)
            cncall = None
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
            #cnccaus = cm_cnccaus(sentences)
            cnccaus = None
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
            #cnclogic = cm_cnclogic(sentences)
            cnclogic = None
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
            #cncadc = cm_cncadc(sentences)
            cncadc = None
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
            #cnctemp = cm_cnctemp(sentences)
            cnctemp = None
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
            #cnctempx = cm_cnctempx(sentences)
            cnctempx = None
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
            #cncadd = cm_cncadd(sentences)
            cncadd = None
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
            #cncpos = cm_cncpos(sentences)
            cncpos = None
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
            #cncneg = cm_cncneg(sentences)
            cncneg = None
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
            #smcausv = cm_smcausv(sentences)
            smcausv = None
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
            #smcausvp = cm_smcausvp(sentences)
            smcausvp = None
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
            #smintep = cm_smintep(sentences)
            smintep = None
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
            #smcausr = cm_smcausr(sentences)
            smcausr = None
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
            #sminter = cm_sminter(sentences)
            sminter = None
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
            #smcauslsa = cm_smcauslsa(sentences)
            smcauslsa = None
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
            #smcauswn = cm_smcauswn(sentences)
            smcauswn = None
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
            #smtemp = cm_smtemp(sentences)
            smtemp = None
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
            synnp = cm_synnp(sentences, request.noun_chunks)
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
            synmedpos = cm_synmedpos(tokens)
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
            synmedwrd = cm_synmedwrd(tokens)
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
            synmedlem = cm_synmedlem(tokens)
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
            #synstruta = cm_synstruta(sentences)
            synstruta = None
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
            #synstrutt = cm_synstrutt(sentences)
            synstrutt = None
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

        # DRNP
        try:
            #drnp = cm_drnp(sentences)
            drnp = None
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
            #drvp = cm_drvp(sentences)
            drvp = None
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
            #drap = cm_drap(sentences)
            drap = None
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
            #drpp = cm_drpp(sentences)
            drpp = None
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
            #drpval = cm_drpval(sentences)
            drpval = None
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
            #drneg = cm_drneg(sentences)
            drneg = None
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
            #drgerund = cm_drgerund(sentences)
            drgerund = None
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
            #drinf = cm_drinf(sentences)
            drinf = None
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

        # WRDNOUN
        try:
            #wrdnoun = cm_wrdnoun(sentences)
            wrdnoun = None
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
            #wrdverb = cm_wrdverb(sentences)
            wrdverb = None
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
            #wrdadj = cm_wrdadj(sentences)
            wrdadj = None
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
            #wrdadv = cm_wrdadv(sentences)
            wrdadv = None
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
            #wrdpro = cm_wrdpro(sentences)
            wrdpro = None
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
            #wrdprp1s = cm_wrdprp1s(sentences)
            wrdprp1s = None
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
            #wrdprp1p = cm_wrdprp1p(sentences)
            wrdprp1p = None
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
            #wrdprp2 = cm_wrdprp2(sentences)
            wrdprp2 = None
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
            #wrdprp3s = cm_wrdprp3s(sentences)
            wrdprp3s = None
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
            #wrdprp3p = cm_wrdprp3p(sentences)
            wrdprp3p = None
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
            #wrdfrqc = cm_wrdfrqc(sentences)
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
            #wrdfrqa = cm_wrdfrqa(sentences)
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
            #wrdfrqmc = cm_wrdfrqmc(sentences)
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

        # WRDAOAc
        try:
            #wrdaoac = cm_wrdaoac(sentences)
            wrdaoac = None
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
            #wrdfamc = cm_wrdfamc(sentences)
            wrdfamc = None
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
            #wrdcncc = cm_wrdcncc(sentences)
            wrdcncc = None
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
            #wrdimgc = cm_wrdimgc(sentences)
            wrdimgc = None
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
            #wrdmeac = cm_wrdmeac(sentences)
            wrdmeac = None
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
            #wrdpolc = cm_wrdpolc(sentences)
            wrdpolc = None
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
            #wrdhypn = cm_wrdhypn(sentences)
            wrdhypn = None
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
            #wrdhypv = cm_wrdhypv(sentences)
            wrdhypv = None
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
            #wrdhypnv = cm_wrdhypnv(sentences)
            wrdhypnv = None
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

        # RDFRE
        try:
            #rdfre = cm_rdfre(sentences)
            rdfre = None
            rdfre_error = None
        except Exception as e:
            logger.error("Error calculating RDFRE: %s", e)
            rdfre = None
            rdfre_error = str(e)
        indices.append(Index(
            index=104,
            type_name="Readability",
            label_v3="RDFRE",
            label_v2="READFRE",
            description="Flesch Reading Ease",
            value=rdfre,
            error=rdfre_error
        ))

        # RDFKGL
        try:
            #rdfkgl = cm_rdfkgl(sentences)
            rdfkgl = None
            rdfkgl_error = None
        except Exception as e:
            logger.error("Error calculating RDFKGL: %s", e)
            rdfkgl = None
            rdfkgl_error = str(e)
        indices.append(Index(
            index=105,
            type_name="Readability",
            label_v3="RDFKGL",
            label_v2="READFKGL",
            description="Flesch–Kincaid Grade Level",
            value=rdfkgl,
            error=rdfkgl_error
        ))

        # RDL2
        try:
            #rdl2 = cm_rdl2(sentences)
            rdl2 = None
            rdl2_error = None
        except Exception as e:
            logger.error("Error calculating RDL2: %s", e)
            rdl2 = None
            rdl2_error = str(e)
        indices.append(Index(
            index=106,
            type_name="Readability",
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

    return TextImagerResponse(
        indices=indices,
        meta=meta,
        modification_meta=modification_meta,
    )
