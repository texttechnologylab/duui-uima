from typing import List, Optional
import uvicorn
from cassis import *
from fastapi import FastAPI, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, BaseSettings
from starlette.responses import JSONResponse
from functools import lru_cache
from trankit import Pipeline


# Lemma
class Lemma(BaseModel):
    """
    de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma
    """
    begin: int
    end: int
    value: Optional[str] = None


# Pos
class Pos(BaseModel):
    """
    de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS
    """
    begin: int
    end: int
    PosValue: Optional[str] = None  # xpos
    coarseValue: Optional[str] = None  # upos


# Morph
class MorphUD1(BaseModel):
    """
    de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.morph.MorphologicalFeatures
    """
    begin: int
    end: int

    gender: Optional[str] = None
    case: Optional[str] = None
    number: Optional[str] = None
    degree: Optional[str] = None
    verbForm: Optional[str] = None
    tense: Optional[str] = None
    mood: Optional[str] = None
    voice: Optional[str] = None
    definiteness: Optional[str] = None
    value: Optional[str] = None
    person: Optional[str] = None
    aspect: Optional[str] = None
    animacy: Optional[str] = None
    negative: Optional[str] = None
    numType: Optional[str] = None
    possessive: Optional[str] = None
    pronType: Optional[str] = None
    reflex: Optional[str] = None
    transitivity: Optional[str] = None

    @classmethod
    def from_str(cls, begin: int, end: int, morph_string: Optional[str]) -> "MorphUD1":
        morph_dict = {"begin": begin, "end": end}
        if morph_string is None:
            return cls(**morph_dict)
        else:
            morphs = morph_string.split("|")
            for morph in morphs:
                name, value = morph.split("=")
                morph_dict[name[0].lower()+name[1:]] = value
            return cls(**morph_dict)


# Token
class Token(BaseModel):
    """
    de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token
    """
    begin: int
    end: int
    lemma: Lemma
    pos: Pos
    morph: MorphUD1


# Dependency
class Dependency(BaseModel):
    """
    de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency
    """
    begin: int
    end: int
    DependencyType: str
    flavor: str
    # Governor: Token
    # Dependent: Token
    Governor: int
    Dependent: int


# Entity
class Entity(BaseModel):
    """
    de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity
    """
    begin: int
    end: int
    value: str


# Sentence
class Sentence(BaseModel):
    """
    de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence
    """

    begin: int
    end: int
    coveredText: str


# Request sent by DUUI
# Note, this is transformed by the Lua script
class DUUIRequest(BaseModel):
    # doc-text
    doc_text: str
    # sentences
    sentences: List[Sentence]


# Response of this annotator
# Note, this is transformed by the Lua script
class DUUIResponse(BaseModel):
    # List of annotated:
    # - sentences
    sentences: Optional[List[Sentence]]

    # - token
    token: List[Token]

    # - deps
    deps: List[Dependency]

    # - ners
    ners: List[Entity]


# Documentation response
class TextImagerDocumentation(BaseModel):
    # Name of this annotator
    annotator_name: str
    # Version of this annotator
    version: str
    # Annotator implementation language (Python, Java, ...)
    implementation_lang: str


class Settings(BaseSettings):
    # Name of the Model
    model_name: str

    # use gpu
    cuda: int

    # meta data
    textimager_trankit_annotator_name: str
    textimager_trankit_annotator_version: str


# settings + cache
settings = Settings()
lru_cache_with_size = lru_cache(maxsize=3)

config = {"gpu": bool(settings.cuda),
          "embedding": settings.model_name}


@lru_cache_with_size
def load_pipeline(**kwargs) -> Pipeline:
    # loads a trankit-Model
    return Pipeline('auto', **kwargs)


# Start fastapi
# TODO openapi types are not shown?
# TODO self host swagger files: https://fastapi.tiangolo.com/advanced/extending-openapi/#self-hosting-javascript-and-css-for-docs
app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title="TRANKIT",
    description="Trankit (spacy alternative) Implementation for TTLab TextImager DUUI",
    version="0.1",
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "TTLab Team",
        "url": "https://texttechnologylab.org",
        "email": "leon.hammerla@gmx.de",
    },
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)

# Load the Lua communication script
communication = "communication.lua"
with open(communication, 'rb') as f:
    communication = f.read().decode("utf-8")


# Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = 'dkpro-core-types.xml'
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)


# Get input / output of the annotator
@app.get("/v1/details/input_output")
def get_input_output() -> JSONResponse:
    json_item = {
        "inputs": ["de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"],
        "outputs": ["de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma",
                    "de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS",
                    "de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.morph.MorphologicalFeatures",
                    "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token",
                    "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency",
                    "de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity",
                    "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"]
    }

    json_compatible_item_data = jsonable_encoder(json_item)
    return JSONResponse(content=json_compatible_item_data)


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


# Return Lua communication script
@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return communication


# Return documentation info
@app.get("/v1/documentation")
def get_documentation() -> TextImagerDocumentation:

    documentation = TextImagerDocumentation(
        annotator_name=settings.textimager_trankit_annotator_name,
        version=settings.textimager_trankit_annotator_version,
        implementation_lang="Python",
    )
    return documentation


# Process request from DUUI
@app.post("/v1/process")
def post_process(request: DUUIRequest) -> DUUIResponse:

    # load pipeline
    pipeline = load_pipeline(**config)

    # get input sentences
    if len(request.sentences) > 0:
        # entities, deps, pos, morphs and token
        sents = None
        ners = []
        deps = []
        token = []
        for sent in request.sentences:
            tokens = []
            temp = pipeline(sent.coveredText, is_sent=True)
            beg_prefix = sent.begin
            for tok in temp["tokens"]:
                beg, end = tok["span"]
                beg, end = beg + beg_prefix, end + beg_prefix
                pos = Pos(**{"begin": beg, "end": end, "PosValue": tok.get("xpos"), "coarseValue": tok.get("upos")})
                try:
                    morph = MorphUD1.from_str(begin=beg, end=end, morph_string=tok["feats"])
                except KeyError:
                    morph = MorphUD1.from_str(begin=beg, end=end, morph_string=None)
                lemma = Lemma(**{"begin": beg, "end": end, "value": tok.get("lemma")})
                tokens.append(Token(**{"begin": beg, "end": end, "lemma": lemma, "pos": pos, "morph": morph}))
                if tok.get("ner") != "O" and tok.get("ner") is not None:
                    ners.append(Entity(**{"begin": beg, "end": end, "value": tok["ner"]}))
            for idx, tok in enumerate(temp["tokens"]):
                beg, end = tok["span"]
                beg, end = beg + beg_prefix, end + beg_prefix
                # deps.append(**{"begin": beg, "end": end, "DependencyType": tok["deprel"], "flavor": "basic", "Governor": tokens[tok["head"] - 1], "Dependent": tokens[idx]})
                if tok.get("deprel") is not None and tok.get("head") is not None:
                    deps.append(Dependency(**{"begin": beg, "end": end, "DependencyType": tok["deprel"], "flavor": "basic",
                                   "Governor": len(token) + tok["head"] - 1, "Dependent": len(token) + idx}))
            token.extend(tokens)

        # Return data as JSON
        return DUUIResponse(
            sentences=sents,
            ners=ners,
            deps=deps,
            token=token
        )
    else:
        res = pipeline(request.doc_text)
        # sentences
        sents = []
        # entities, deps, pos, morphs and token
        ners = []
        deps = []
        token = []
        for sent in res["sentences"]:
            tokens = []
            sents.append(Sentence(**{"begin": sent["dspan"][0], "end": sent["dspan"][1], "coveredText": sent["text"]}))
            for tok in sent["tokens"]:
                beg, end = tok["dspan"]
                pos = Pos(**{"begin": beg, "end": end, "PosValue": tok.get("xpos"), "coarseValue": tok.get("upos")})
                try:
                    morph = MorphUD1.from_str(begin=beg, end=end, morph_string=tok["feats"])
                except KeyError:
                    morph = MorphUD1.from_str(begin=beg, end=end, morph_string=None)
                lemma = Lemma(**{"begin": beg, "end": end, "value": tok.get("lemma")})
                tokens.append(Token(**{"begin": beg, "end": end, "lemma": lemma, "pos": pos, "morph": morph}))
                if tok.get("ner") != "O" and tok.get("ner") is not None:
                    ners.append(Entity(**{"begin": beg, "end": end, "value": tok["ner"]}))
            for idx, tok in enumerate(sent["tokens"]):
                beg, end = tok["dspan"]
                # deps.append(**{"begin": beg, "end": end, "DependencyType": tok["deprel"], "flavor": "basic", "Governor": tokens[tok["head"]-1], "Dependent": tokens[idx]})
                if tok.get("deprel") is not None and tok.get("head") is not None:
                    deps.append(Dependency(**{"begin": beg, "end": end, "DependencyType": tok["deprel"], "flavor": "basic",
                                   "Governor": len(token) + tok["head"] - 1, "Dependent": len(token) + idx}))
            token.extend(tokens)
        # Return data as JSON
        return DUUIResponse(
            sentences=sents,
            ners=ners,
            deps=deps,
            token=token
        )


if __name__ == "__main__":
    uvicorn.run("duui_trankit:app", host="0.0.0.0", port=9714, workers=1)


