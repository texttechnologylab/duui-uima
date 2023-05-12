from sys import version as sys_version
from platform import python_version
from time import time
from functools import lru_cache
from threading import Lock
from typing import List, Optional
import logging
import spacy
from cassis import load_typesystem
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseSettings, BaseModel


# Settings
# These are automatically loaded from env variables
class Settings(BaseSettings):
    # Name of this annotator
    textimager_spacy_annotator_name: str
    # Version of this annotator
    textimager_spacy_annotator_version: str
    # Log level
    textimager_spacy_log_level: str
    # Model LRU cache size
    textimager_spacy_model_cache_size: int
    # This is set to the model if only one single model is in the Docker image
    textimager_spacy_single_model: Optional[str]
    # This is set to the language of the single model
    textimager_spacy_single_model_lang: Optional[str]


# Load settings from env vars
settings = Settings()

# Init logger
logging.basicConfig(level=settings.textimager_spacy_log_level)
logger = logging.getLogger(__name__)
logger.info("TTLab TextImager DUUI spaCy")
logger.info("Name: %s", settings.textimager_spacy_annotator_name)
logger.info("Version: %s", settings.textimager_spacy_annotator_version)

# Type names needed by this annotator
UIMA_TYPE_SENTENCE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
UIMA_TYPE_TOKEN = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"
UIMA_TYPE_LEMMA = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma"
UIMA_TYPE_POS = "de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS"
UIMA_TYPE_MORPH = "de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.morph.MorphologicalFeatures"
UIMA_TYPE_DEPENDENCY = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency"
UIMA_TYPE_DEPENDENCY_ROOT = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.ROOT"
UIMA_TYPE_NAMED_ENTITY = "de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity"

# Types this annotator produces
# Note: Without any extra meta types or subtypes
TEXTIMAGER_ANNOTATOR_OUTPUT_TYPES = [
    UIMA_TYPE_SENTENCE,
    UIMA_TYPE_TOKEN,
    UIMA_TYPE_LEMMA,
    UIMA_TYPE_POS,
    UIMA_TYPE_MORPH,
    UIMA_TYPE_DEPENDENCY,
    UIMA_TYPE_NAMED_ENTITY
]

# spaCy models
# Supporting the efficient and accurate variants
# Note: Not all models might actually be available in the Docker image!
# TODO test accurate models
SPACY_MODELS = {
    "efficiency": {
        "ca": "ca_core_news_sm",    # Catalan
        "zh": "zh_core_web_sm",     # Chinese
        "da": "da_core_news_sm",    # Danish
        "nl": "nl_core_news_sm",    # Dutch
        "en": "en_core_web_sm",     # English
        "fr": "fr_core_news_sm",    # French
        "de": "de_core_news_sm",    # German
        "el": "el_core_news_sm",    # Greek
        "it": "it_core_news_sm",    # Italian
        "ja": "ja_core_news_sm",    # Japanese
        "lt": "lt_core_news_sm",    # Lithuanian
        "mk": "mk_core_news_sm",    # Macedonian
        "nb": "nb_core_news_sm",    # Norwegian Bokmal
        "pl": "pl_core_news_sm",    # Polish
        "pt": "pt_core_news_sm",    # Portugese
        "ro": "ro_core_news_sm",    # Romanian
        "ru": "ru_core_news_sm",    # Russian
        "es": "es_core_news_sm",    # Spanish
        "xx": "xx_ent_wiki_sm",     # Multi-Language / Unknown Language
    },
    "accuracy": {
        "ca": "ca_core_news_trf",
        "zh": "zh_core_web_trf",
        "da": "da_core_news_trf",
        "nl": "nl_core_news_lg",
        "en": "en_core_web_trf",
        "fr": "fr_dep_news_trf",
        "de": "de_dep_news_trf",
        "el": "el_core_news_lg",
        "it": "it_core_news_lg",
        "ja": "ja_core_news_trf",
        "lt": "lt_core_news_lg",
        "mk": "mk_core_news_lg",
        "nb": "nb_core_news_lg",
        "pl": "pl_core_news_lg",
        "pt": "pt_core_news_lg",
        "ro": "ro_core_news_lg",
        "ru": "ru_core_news_lg",
        "es": "es_dep_news_trf",
        "xx": "xx_sent_ud_sm",
    }
}

# Collect list of supported languages and models
SPACY_SUPPORTED_LANGS = set()
if settings.textimager_spacy_single_model is not None:
    SPACY_SUPPORTED_MODELS = {settings.textimager_spacy_single_model}
    SPACY_SUPPORTED_LANGS = {settings.textimager_spacy_single_model_lang}
    SPACY_SUPPORTED_MODEL_VARIANTS = set()
else:
    SPACY_SUPPORTED_MODELS = set()
    SPACY_SUPPORTED_MODEL_VARIANTS = set(SPACY_MODELS.keys())
    for model_variant in SPACY_MODELS:
        SPACY_SUPPORTED_LANGS.update(SPACY_MODELS[model_variant].keys())
        SPACY_SUPPORTED_MODELS.update(SPACY_MODELS[model_variant].values())

# Provide additional language mappings
# TODO more elaborate mapping on iso codes
SPACY_LANGUAGE_MAPPINGS = {
    "x-unspecified": "xx",
    "arz": "ar",
    "uk": "ru"
}


# Request sent by DUUI
# Note, this is transformed by the Lua script
class TextImagerRequest(BaseModel):
    # The text to process
    text: str
    # The texts language
    lang: str
    # Optional map/dict of parameters
    # TODO how with lua?
    parameters: Optional[dict]


# UIMA type: adds metadata to each annotation
class AnnotationMeta(BaseModel):
    name: str
    version: str
    modelName: str
    modelVersion: str
    spacyVersion: str
    modelLang: str
    modelSpacyVersion: str
    modelSpacyGitVersion: str


# UIMA type: mark modification of the document
class DocumentModification(BaseModel):
    user: str
    timestamp: int
    comment: str


# Token
class Token(BaseModel):
    begin: int
    end: int
    ind: int
    write_token: Optional[bool]
    lemma: Optional[str]
    write_lemma: Optional[bool]
    pos: Optional[str]
    pos_coarse: Optional[str]
    write_pos: Optional[bool]
    morph: Optional[str]
    morph_details: Optional[dict]
    write_morph: Optional[bool]
    parent_ind: Optional[int]
    write_dep: Optional[bool]


# Dependency
class Dependency(BaseModel):
    begin: int
    end: int
    type: str
    flavor: str
    dependent_ind: int
    governor_ind: int
    write_dep: bool


# Sentence
class Sentence(BaseModel):
    begin: int
    end: int
    write_sentence: bool


# Entity
class Entity(BaseModel):
    begin: int
    end: int
    value: str
    write_entity: bool


# Response of this annotator
# Note, this is transformed by the Lua script
class TextImagerResponse(BaseModel):
    # List of sentences
    sentences: List[Sentence]
    # List of tokens
    tokens: List[Token]
    # List of dependencies
    dependencies: List[Dependency]
    # List of entities
    entities: List[Entity]
    # Annotation meta, containing model name, version and more
    # Note: Same for each annotation, so only returned once
    meta: Optional[AnnotationMeta]
    # Modification meta, one per document
    modification_meta: Optional[DocumentModification]


# Capabilities
class TextImagerCapability(BaseModel):
    # List of supported languages by the annotator
    # TODO how to handle language?
    # - ISO 639-1 (two letter codes) as default in meta data
    # - ISO 639-3 (three letters) optionally in extra meta to allow a finer mapping
    supported_languages: List[str]
    # Are results on same inputs reproducible without side effects?
    reproducible: bool


# Documentation response
class TextImagerDocumentation(BaseModel):
    # Name of this annotator
    annotator_name: str
    # Version of this annotator
    version: str
    # Annotator implementation language (Python, Java, ...)
    implementation_lang: Optional[str]
    # Optional map of additional meta data
    meta: Optional[dict]
    # Docker container id, if any
    docker_container_id: Optional[str]
    # Optional map of supported parameters
    parameters: Optional[dict]
    # Capabilities of this annotator
    capability: TextImagerCapability
    # Analysis engine XML, if available
    implementation_specific: Optional[str]


# Get spaCy model from language
def get_spacy_model_name(document_lang, parameters):
    # Directly use specified in parameters, this ignores any model variants!
    if parameters is not None and "model_name" in parameters:
        model_name = parameters["model_name"]
        logger.debug("Model name from parameters: \"%s\"", model_name)
        return model_name

    # Get model variant, use efficient if not in parameters
    model_variant = "efficiency"
    if parameters is not None and "model_variant" in parameters:
        model_variant = parameters["model_variant"]
        logger.debug("Model variant from parameters: \"%s\"", model_variant)

        # Check if variant exits, this is a hard error!
        if model_variant not in SPACY_MODELS:
            raise Exception(f"The specified spaCy model variant \"{model_variant}\" does not exist!")

    # 3) Use multi-language model by default
    if document_lang is None:
        logger.warning("No document language found, using mutli-language")
        document_lang = "xx"
    logger.debug("Document language: \"%s\"", document_lang)

    # Perform language mapping
    if document_lang in SPACY_LANGUAGE_MAPPINGS:
        document_lang = SPACY_LANGUAGE_MAPPINGS[document_lang]
        logger.debug("Mapped language to: \"%s\"", document_lang)

    # Make sure language is supported
    if document_lang not in SPACY_MODELS[model_variant]:
        # Perform strict language checking?
        strict_language_check = parameters["strict_language_check"] if (parameters is not None and "strict_language_check" in parameters) else False
        logger.debug(f"Strict language checking is \"{strict_language_check}\"")

        # Error if language not found and strict checking, else use default multilanguage model
        if strict_language_check:
            raise Exception(f"Document language \"{document_lang}\" could not be mapped to spaCy model!")
        else:
            logger.warning(f"No spaCy model with language \"{document_lang}\" found, using mutli-language")
            document_lang = "xx"

    # 2) Use language from CAS provided in meta data
    model_name = SPACY_MODELS[model_variant][document_lang]
    logger.debug("Mapped model name from document language \"%s\": \"%s\"", document_lang, model_name)
    return model_name


# Create LRU cache with max size for model
lru_cache_with_size = lru_cache(maxsize=settings.textimager_spacy_model_cache_size)

# Lock for model loading
model_load_lock = Lock()

# Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = 'TypeSystemSpacy.xml'
logger.debug("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    logger.debug("Base typesystem:")
    logger.debug(typesystem.to_xml())

# Load the Lua communication script
lua_communication_script_filename = "textimager_duui_spacy.lua"
logger.debug("Loading Lua communication script from \"%s\"", lua_communication_script_filename)
with open(lua_communication_script_filename, 'rb') as f:
    lua_communication_script = f.read().decode("utf-8")
    logger.debug("Lua communication script:")
    logger.debug(lua_communication_script_filename)


# Load/cache spaCy model
@lru_cache_with_size
def load_cache_spacy_model(model_name):
    logger.info("Loading spaCy model \"%s\"...", model_name)
    nlp = spacy.load(model_name)
    logger.info("Finished loading spaCy model \"%s\"", model_name)
    return nlp


# Load spaCy model using LRU cached function
def load_spacy_model(model_name):
    model_load_lock.acquire()

    err = None
    try:
        logger.info("Getting spaCy model \"%s\"...", model_name)
        nlp = load_cache_spacy_model(model_name)
    except Exception as ex:
        nlp = None
        err = str(ex)
        logging.exception("Failed to load spaCy model: %s", ex)

    model_load_lock.release()

    return nlp, err


# Load/cache spaCy sentencizer model
@lru_cache_with_size
def load_cache_spacy_sentencizer_model(model_lang):
    logger.info("Loading spaCy sentencizer model \"%s\"...", model_lang)

    nlp_sents = spacy.blank(model_lang)
    nlp_sents.add_pipe("sentencizer")

    logger.info("Finished loading spaCy sentencizer model \"%s\"", model_lang)
    return nlp_sents


# Load spaCy sentencizer model using LRU cached function
def load_spacy_sentencizer_model(model_lang):
    model_load_lock.acquire()

    err = None
    try:
        logger.info("Getting spaCy sentencizer model \"%s\"...", model_lang)
        nlp = load_cache_spacy_sentencizer_model(model_lang)
    except Exception as ex:
        nlp = None
        err = str(ex)
        logging.exception("Failed to load spaCy sentencizer model: %s", ex)

    model_load_lock.release()

    return nlp, err


# Start fastapi
# TODO openapi types are not shown?
# TODO self host swagger files: https://fastapi.tiangolo.com/advanced/extending-openapi/#self-hosting-javascript-and-css-for-docs
app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title=settings.textimager_spacy_annotator_name,
    description="spaCy implementation for TTLab TextImager DUUI",
    version=settings.textimager_spacy_annotator_version,
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "TTLab Team",
        "url": "https://texttechnologylab.org",
        "email": "baumartz@stud.uni-frankfurt.de",
    },
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)


# Return Lua communication script
@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return lua_communication_script


# Return documentation info
@app.get("/v1/documentation")
def get_documentation() -> TextImagerDocumentation:
    capabilities = TextImagerCapability(
        supported_languages=sorted(list(SPACY_SUPPORTED_LANGS)),
        reproducible=True
    )

    documentation = TextImagerDocumentation(
        annotator_name=settings.textimager_spacy_annotator_name,
        version=settings.textimager_spacy_annotator_version,
        implementation_lang="Python",
        meta={
            "python_version": python_version(),
            "python_version_full": sys_version,
            "spacy_version": spacy.__version__
        },
        # TODO
        docker_container_id="[TODO]",
        parameters={
            # Select model variant, uses efficient by default
            "model_variant": sorted(list(SPACY_SUPPORTED_MODEL_VARIANTS)),
            # Use this specific model, overrules all decisions
            "model_name": sorted(list(SPACY_SUPPORTED_MODELS)),
            # Write the following types, if empty/null all are written
            # Note: All data is always generated by the full document text, dependency is only written if tokens are written too
            "write_types": TEXTIMAGER_ANNOTATOR_OUTPUT_TYPES,
            # Strict language checking, if True the language must be available, on False the multilanguage model is used
            "strict_language_check": [True, False],
            # Split large input texts to prevent dramatic increase of time and resources
            # Note: Splitting is performed on sentence boundaries, if possible, else on "."
            "split_large_texts": [True, False],
            # TODO more zh options
            #"zh_segmenter": [
            #    "char",
            #    "jieba",
            #    "pkuseg"
            #],
            #"zh_pkuseg_model": [
            #    "spacy_ontonotes",
            #    "mixed",
            #    "news",
            #    "web",
            #    "medicine",
            #    "tourism"
            #]
            # TODO more lang options?
        },
        capability=capabilities,
        implementation_specific=None,
    )

    return documentation


# Get typesystem of this annotator
@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    # TODO remove cassis dependency, as only needed for typesystem at the moment?
    xml = typesystem.to_xml()
    xml_content = xml.encode("utf-8")

    return Response(
        content=xml_content,
        media_type="application/xml"
    )


# Process request from DUUI
@app.post("/v1/process")
def post_process(request: TextImagerRequest) -> TextImagerResponse:
    # Return data
    sentences = []
    tokens = []
    dependencies = []
    entities = []
    meta = None
    modification_meta = None

    # Save modification start time for later
    modification_timestamp_seconds = int(time())

    try:
        # Get CAS from XMI string
        logger.debug("Received:")
        logger.debug(request)

        # Params, set here to empty dict to allow easier access later
        if request.parameters is None:
            request.parameters = {}

        # Get spaCy model if not in single model mode
        if settings.textimager_spacy_single_model is None:
            # Resolve model name
            model_name = get_spacy_model_name(request.lang, request.parameters)
        else:
            # In single mode we always use the single specified model!
            model_name = settings.textimager_spacy_single_model
            logger.info("Using single model image: \"%s\"", model_name)
        logger.info("Using spaCy model: \"%s\"", model_name)

        # Load model, this is cached
        nlp, nlp_err = load_spacy_model(model_name)
        if nlp is None:
            raise Exception(f"spaCy model \"{model_name}\" could not be loaded: {nlp_err}")

        # Get meta data on spaCy and used model
        spacy_meta = nlp.meta

        # Split large texts if needed and allowed
        # TODO test splitting!
        texts = None
        texts_meta = None
        text_len = len(request.text)
        logger.debug("spaCy max size: %d", nlp.max_length)
        logger.debug("Text size: %d", text_len)
        if nlp.max_length < text_len:
            # Allow splitting of large texts?
            split_large_texts = (request.parameters["split_large_texts"].lower() != "false") if ("split_large_texts" in request.parameters) else False
            if split_large_texts:
                # Try to split based on sentences first
                model_lang = spacy_meta["lang"]
                logger.info(f"Splitting text into sentences using \"{model_lang}\" sentencizer...")
                try:
                    # Load cached sentencizer model
                    nlp_sents, nlp_sents_err = load_spacy_sentencizer_model(model_lang)
                    if nlp_sents is None:
                        raise Exception(f"spaCy sentencizer model \"{model_lang}\" could not loaded: {nlp_sents_err}")
                    doc_sents = nlp_sents(request.text)
                    texts = []
                    texts_meta = []
                    for sent in doc_sents.sents:
                        texts.append(sent.text)
                        texts_meta.append({
                            "begin": sent.start_char,
                            "end": sent.end_char,
                        })
                except Exception as ex:
                    # Splitting sentences failed, fallback to full text
                    # TODO try to split using "."
                    texts = None
                    texts_meta = None
                    logger.exception("Failed to split sentences: %s", ex)
            else:
                logger.warning("Text is too large, but splitting is disabled, this might be slow to process...")

        # Use full text, if not set
        if texts is None:
            texts = [request.text]
            texts_meta = [{
                "begin": 0,
                "end": len(request.text),
            }]
        logger.info(f"Found {len(texts)} texts to process.")

        # Abort if no texts found
        if len(texts) == 0:
            logger.warning("No texts found, aborting...")
        else:
            # Find max text length
            max_length_new = None
            for text in texts:
                text_len = len(text)
                if nlp.max_length < text_len:
                    if max_length_new is None:
                        max_length_new = text_len + 100
                    else:
                        max_length_new = max(max_length_new, text_len+100)

            # Increase max length, if needed
            max_length_before = None
            if max_length_new is not None:
                logger.info("Increasing spaCy max length %d -> %d", nlp.max_length, max_length_new)
                max_length_before = nlp.max_length
                nlp.max_length = max_length_new

            # Process text with spaCy
            logger.debug("Start processing...")
            docs = list(nlp.pipe(texts))
            logger.debug("Procesed %d texts into %d documents.", len(texts), len(docs))

            # Reset max length, if changed
            if max_length_before is not None:
                logger.info("Resetting spaCy max length to %d", max_length_before)
                nlp.max_length = max_length_before

            # Build a "annotation comment" annotation
            # Can be used for each annotation
            meta = AnnotationMeta(
                    name=settings.textimager_spacy_annotator_name,
                    version=settings.textimager_spacy_annotator_version,
                    modelName=spacy_meta["name"],
                    modelVersion=spacy_meta["version"],
                    spacyVersion=spacy.__version__,
                    modelLang=spacy_meta["lang"],
                    modelSpacyVersion=spacy_meta["spacy_version"],
                    modelSpacyGitVersion=spacy_meta["spacy_git_version"]
                )

            # What types to write?
            if request.parameters is not None and "write_types" in request.parameters and len(request.parameters["write_types"]) > 0:
                write_types = set(request.parameters["write_types"])
                logger.info("Only writing types: %s", ", ".join(write_types))
            else:
                write_types = set(TEXTIMAGER_ANNOTATOR_OUTPUT_TYPES)

            # TODO test splitting in multiple texts
            for doc_meta, doc in zip(texts_meta, docs):
                # Get starting position of this sentence
                doc_begin = doc_meta["begin"]

                # Sentences
                logger.debug("Writing Sentences...")
                try:
                    # Can fail, e.g. with multilang model
                    # TODO add_pipe("sentencizer") seems to work, check later!
                    for sent in doc.sents:
                        sentences.append(Sentence(
                            begin=doc_begin+sent.start_char,
                            end=doc_begin+sent.end_char,
                            write_sentence=UIMA_TYPE_SENTENCE in write_types,
                        ))
                except Exception as ex:
                    logger.exception("Error accessing sentences: %s", ex)

                # Token, Lemma, POS, Morphology
                cas_tokens = {}
                token_ind = 0
                logger.debug("Writing Tokens...")
                for token in doc:
                    if not token.is_space:
                        # Create begin/end
                        token_begin = token.idx
                        token_end = token_begin + len(token)

                        # Extract specific morph features
                        morph_details = {}
                        for feat in token.morph:
                            fields = feat.split("=")
                            if len(fields) != 2:
                                continue
                            feat_key = fields[0].strip()
                            feat_value = fields[1].strip()

                            # TODO check in nlp.meta for missing
                            if feat_key == "Gender":
                                morph_details["gender"] = feat_value
                            elif feat_key == "Number":
                                morph_details["number"] = feat_value
                            elif feat_key == "Case":
                                morph_details["case"] = feat_value
                            elif feat_key == "Degree":
                                morph_details["degree"] = feat_value
                            elif feat_key == "VerbForm":
                                morph_details["verbForm"] = feat_value
                            elif feat_key == "Tense":
                                morph_details["tense"] = feat_value
                            elif feat_key == "Mood":
                                morph_details["mood"] = feat_value
                            elif feat_key == "Voice":  # ?
                                morph_details["voice"] = feat_value
                            elif feat_key == "Definite":
                                morph_details["definiteness"] = feat_value
                            elif feat_key == "Person":
                                morph_details["person"] = feat_value
                            elif feat_key == "Aspect":  # ?
                                morph_details["aspect"] = feat_value
                            elif feat_key == "Animacy":  # ?
                                morph_details["animacy"] = feat_value
                            elif feat_key == "Negative":  # ?
                                morph_details["negative"] = feat_value
                            elif feat_key == "NumType":  # ?
                                morph_details["numType"] = feat_value
                            elif feat_key == "Possessive":  # ?
                                morph_details["possessive"] = feat_value
                            elif feat_key == "PronType":
                                morph_details["pronType"] = feat_value
                            elif feat_key == "Reflex":
                                morph_details["reflex"] = feat_value
                            elif feat_key == "Transitivity":  # ?
                                morph_details["transitivity"] = feat_value

                        # Create token data
                        current_token = Token(
                            begin=doc_begin+token_begin,
                            end=doc_begin+token_end,

                            # Token
                            ind=token_ind,
                            write_token=UIMA_TYPE_TOKEN in write_types,

                            # Lemma
                            lemma=token.lemma_,
                            write_lemma=UIMA_TYPE_LEMMA in write_types,

                            # POS
                            # TODO pos mapping?
                            pos=token.tag_,
                            pos_coarse=token.pos_,
                            write_pos=UIMA_TYPE_POS in write_types,

                            # Morph
                            morph="|".join(token.morph),
                            morph_details=morph_details,
                            write_morph=UIMA_TYPE_MORPH in write_types
                        )
                        tokens.append(current_token)
                        token_ind += 1

                        # Save token info for deps later
                        if token_begin not in cas_tokens:
                            cas_tokens[token_begin] = {}
                        cas_tokens[token_begin][token_end] = current_token

                # Dependency
                # Note: This is only supported if tokens are written!
                logger.debug("Writing Dependencies...")
                for token in doc:
                    if not token.is_space:
                        if not token.head.is_space:
                            # Get CAS token
                            token_begin = token.idx
                            token_end = token_begin + len(token)
                            if token_begin in cas_tokens and token_end in cas_tokens[token_begin]:
                                uima_token = cas_tokens[token_begin][token_end]
                            else:
                                continue

                            # Get head token
                            begin_head = token.head.idx
                            end_head = begin_head + len(token.head)
                            if begin_head in cas_tokens and end_head in cas_tokens[begin_head]:
                                uima_head = cas_tokens[begin_head][end_head]
                            else:
                                continue

                            # Create dependency
                            current_dep = Dependency(
                                begin=doc_begin+token_begin,
                                end=doc_begin+token_end,
                                type=token.dep_.upper(),
                                flavor="basic",
                                dependent_ind=uima_token.ind,
                                governor_ind=uima_head.ind,
                                write_dep=UIMA_TYPE_DEPENDENCY in write_types
                            )
                            dependencies.append(current_dep)

                            # Add reference to token
                            uima_token.parent_ind = uima_head.ind
                            uima_token.write_dep = UIMA_TYPE_DEPENDENCY in write_types

                # Named entities
                logger.debug("Writing Named entities...")
                try:
                    for ent in doc.ents:
                        entities.append(Entity(
                            begin=doc_begin+ent.start_char,
                            end=doc_begin+ent.end_char,
                            value=ent.label_,
                            write_entity=UIMA_TYPE_NAMED_ENTITY in write_types
                        ))
                except Exception as ex:
                    logger.exception("Error accessing named entities: %s", ex)

                # Add modification info
                modification_meta_comment = f"{settings.textimager_spacy_annotator_name} ({settings.textimager_spacy_annotator_version}), spaCy ({spacy.__version__}), {spacy_meta['lang']} {spacy_meta['name']} ({spacy_meta['version']})"
                modification_meta = DocumentModification(
                    user=settings.textimager_spacy_annotator_name,
                    timestamp=modification_timestamp_seconds,
                    comment=modification_meta_comment
                 )

    except Exception as ex:
        logger.exception(ex)

    # Return data as JSON
    return TextImagerResponse(
        sentences=sentences,
        tokens=tokens,
        dependencies=dependencies,
        entities=entities,
        meta=meta,
        modification_meta=modification_meta
    )
