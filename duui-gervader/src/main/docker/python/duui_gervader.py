from typing import List

from cassis import *
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from gervader import vaderSentimentGER


class Sentence(BaseModel):
    text: str
    iBegin: int
    iEnd: int


class Sentiment(BaseModel):
    # Info from GerVADER:
    # The 'compound' score is computed by summing the valence scores of each word in the lexicon, adjusted
    # according to the rules, and then normalized to be between -1 (most extreme negative) and +1 (most extreme positive).
    # This is the most useful metric if you want a single unidimensional measure of sentiment for a given sentence.
    # Calling it a 'normalized, weighted composite score' is accurate.
    compound: float

    iBegin: int
    iEnd: int

    # Info from GerVADER:
    # The 'pos', 'neu', and 'neg' scores are ratios for proportions of text that fall in each category (so these
    # should all add up to be 1... or close to it with float operation).  These are the most useful metrics if
    # you want multidimensional measures of sentiment for a given sentence.
    pos: float
    neu: float
    neg: float


# Request sent by DUUI
# Note, this is transformed by the Lua script
class DUUIRequest(BaseModel):
    # The sentences to process
    sentences: List[Sentence]
    doc_length: int


# Response of this annotator
# Note, this is transformed by the Lua script
class DUUIResponse(BaseModel):
    # List of Sentiment
    sentiments: List[Sentiment]


analyzer = vaderSentimentGER.SentimentIntensityAnalyzer()


def analyse(sentences, length):
    processed_sentences = []

    last_sentence_end = 0
    for selection in sentences:
        last_sentence_end = max(last_sentence_end, selection.iEnd)
        vs = analyzer.polarity_scores(selection.text)

        processed_sentences.append(Sentiment(
            iBegin=selection.iBegin,
            iEnd=selection.iEnd,
            compound=vs["compound"],
            pos=vs["pos"],
            neu=vs["neu"],
            neg=vs["neg"],
        ))

    # compute avg for this selection, if >1
    print(length, last_sentence_end)
    if len(processed_sentences) > 1:
        begin = 0
        end = last_sentence_end

        compounds = 0
        poss = 0
        neus = 0
        negs = 0
        for sentence in processed_sentences:
            compounds += sentence.compound
            poss += sentence.pos
            neus += sentence.neu
            negs += sentence.neg

        compound = compounds / len(processed_sentences)
        pos = poss / len(processed_sentences)
        neu = neus / len(processed_sentences)
        neg = negs / len(processed_sentences)

        processed_sentences.append(Sentiment(
            iBegin=begin,
            iEnd=end,
            compound=compound,
            pos=pos,
            neu=neu,
            neg=neg,
        ))

    return processed_sentences


communication = ''

# Start fastapi
# TODO openapi types are not shown?
# TODO self host swagger files: https://fastapi.tiangolo.com/advanced/extending-openapi/#self-hosting-javascript-and-css-for-docs
app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title="GerVaderDUUI",
    description="GerVader implementation for TTLab DUUI",
    version="0.1",
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "TTLab Team",
        "url": "https://texttechnologylab.org",
        "email": "abrami@em.uni-frankfurt.de",
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
        "outputs": ["org.hucompute.textimager.uima.type.GerVaderSentiment"]
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


# Process request from DUUI
@app.post("/v1/process")
def post_process(request: DUUIRequest) -> DUUIResponse:
    length = request.doc_length
    sentences = request.sentences
    sentiments = analyse(sentences, length)

    # Return data as JSON
    return DUUIResponse(
        sentiments=sentiments
    )


if __name__ == "__main__":
    uvicorn.run("duui_gervader:app", host="0.0.0.0", port=9715, workers=1)
