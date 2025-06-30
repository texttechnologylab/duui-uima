import logging
from platform import python_version
from sys import version as sys_version
from time import time
from typing import List, Optional

from cassis import load_typesystem
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings

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
    pos: Optional[str]
    lemma: Optional[str]


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
    nouns_a = set([t.text for t in sentence_a.tokens if t.pos and t.pos.startswith('N')])
    nouns_b = set([t.text for t in sentence_b.tokens if t.pos and t.pos.startswith('N')])
    return len(nouns_a.intersection(nouns_b))

def _argument_overlap(sentence_a: Sentence, sentence_b: Sentence) -> int:
    nouns_a = set([t.lemma for t in sentence_a.tokens if t.pos and t.pos.startswith('N')])
    nouns_b = set([t.lemma for t in sentence_b.tokens if t.pos and t.pos.startswith('N')])
    noun_overlap = len(nouns_a.intersection(nouns_b))

    promouns_a = set([t.text for t in sentence_a.tokens if t.pos and t.pos == 'PPER'])
    promouns_b = set([t.text for t in sentence_b.tokens if t.pos and t.pos == 'PPER'])
    pronoun_overlap = len(promouns_a.intersection(promouns_b))

    return noun_overlap + pronoun_overlap

def _stem_overlap(sentence_nouns: Sentence, sentence_contents: Sentence) -> int:
    nouns_a = set([t.lemma for t in sentence_nouns.tokens if t.pos and t.pos.startswith('N')])
    nouns_b = set([t.lemma for t in sentence_contents.tokens if t.pos and (
        t.pos.startswith('N') or t.pos.startswith('V') or t.pos.startswith("ADJ") or t.pos.startswith("ADV")
    )])
    return len(nouns_a.intersection(nouns_b))

def _word_overlap(sentence_nouns: Sentence, sentence_contents: Sentence) -> float:
    nouns_a = set([t.lemma for t in sentence_nouns.tokens if t.pos and t.pos.startswith('N')])
    nouns_b = set([t.lemma for t in sentence_contents.tokens if t.pos and (
            t.pos.startswith('N') or t.pos.startswith('V') or t.pos.startswith("ADJ") or t.pos.startswith("ADV")
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
        # TODO

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
            type_name="Descriptive",
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
            type_name="Descriptive",
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
            type_name="Descriptive",
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
            type_name="Descriptive",
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
            type_name="Descriptive",
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
            type_name="Descriptive",
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
            type_name="Descriptive",
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
            type_name="Descriptive",
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
            type_name="Descriptive",
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
            type_name="Descriptive",
            label_v3="CRFCWOad",
            label_v2="n/a",
            description="Content word overlap, all sentences, proportional, standard deviation",
            value=crfcwoad,
            error=crfcwoad_error
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
