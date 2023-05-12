from cassis import *
from fastapi import FastAPI, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel

from summarizer import Summarizer


# Request sent by DUUI
# Note, this is transformed by the Lua script
class DUUIRequest(BaseModel):
    # The sentences to process
    text: str
    doc_length: int


# Response of this annotator
# Note, this is transformed by the Lua script
class DUUIResponse(BaseModel):
    # Abstract
    abstract: str


model = Summarizer()


def analyse(text):
    abstract = model(text, ratio=0.3)
    print(abstract)
    return abstract

# example text
# text = "Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types systems and systems of mixed types."
#
# text2 = requests.get('http://rare-technologies.com/the_matrix_synopsis.txt').text
#
# model = Summarizer()
# print(model(text2, num_sentences=5))
# print(model(text2, ratio=0.3))
# # print(model(text, ratio=0.2))
#
# model = SBertSummarizer('paraphrase-MiniLM-L6-v2')
# print(model(text, num_sentences=3))
#
# # Instantiating the model and tokenizer with gpt-2
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2LMHeadModel.from_pretrained('gpt2')
#
# inputs = tokenizer.batch_encode_plus([text], return_tensors='pt', max_length=512)
# summary_ids = model.generate(inputs['input_ids'], early_stopping=True)
#
# GPT_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
# print(GPT_summary)


communication = ''

# Start fastapi
# TODO openapi types are not shown?
# TODO self host swagger files: https://fastapi.tiangolo.com/advanced/extending-openapi/#self-hosting-javascript-and-css-for-docs
app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title="TextBERTSummaryDUUI",
    description="Text Summery based on BERT implementation for TTLab DUUI",
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
        "inputs": [""],
        "outputs": ["org.texttechnologylab.textimager.uima.type.GerVaderSentiment"]
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
@app.get("/v1/communication_layer", response_class = PlainTextResponse)
def get_communication_layer() -> str:
    return communication


# Process request from DUUI
@app.post("/v1/process")
def post_process(request: DUUIRequest) -> DUUIResponse:
    length = request.doc_length
    text = request.text

    # Return data as JSON
    return DUUIResponse(
        abstract=analyse(text)
    )


# if __name__ == "__main__":
#     uvicorn.run("duui_abstractgenerator:app", host="0.0.0.0", port=9715, workers=1)
