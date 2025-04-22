import logging
from cassis import load_typesystem
from fastapi import FastAPI, Response
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from starlette.responses import PlainTextResponse, JSONResponse
from typing import List, Optional
import re
from mp import get_mp
import pandas as pd
import difflib

class Settings(BaseSettings):
    # Name of this annotator
    annotator_name: str
    # Version of this annotator
    # TODO add these to the settings
    annotator_version: str
    # Log level
    log_level: str


# Speech
class Speech(BaseModel):
    begin: int
    end: int


# Speaker
class Speaker(BaseModel):
    """
    Has to be in line with the features frim the typesystem
    """
    begin: int
    end: int
    label: str
    firstname: Optional[str] = None
    name: Optional[str] = None
    nobility: Optional[str] = None
    title: Optional[str] = None
    role: Optional[str] = None
    party: Optional[str] = None
    party_deducted: Optional[str] = None
    electoral_county: Optional[str] = None
    electoral_county_deducted: Optional[str] = None

def find_mp(speaker: Speaker, mp_df: pd.DataFrame, threshold: float = 0.8) -> Speaker:
    """
    Enriches a Speaker object with party and electoral_county
    by matching speaker.name against mp_df.family_name,
    falling back to a fuzzy match if needed.
    """
    if not speaker.name:
        speaker.party = None
        speaker.electoral_county = None
        return speaker

    name_lower = speaker.name.lower()
    # 1) Exact match
    mask_exact = mp_df["family_name"].str.lower() == name_lower
    if mask_exact.any():
        row = mp_df.loc[mask_exact].iloc[0]
    else:
        # 2) Fuzzy fallback
        candidates = mp_df["family_name"].dropna().unique().tolist()
        closest = difflib.get_close_matches(speaker.name, candidates, n=1, cutoff=threshold)
        if not closest:
            speaker.party = None
            speaker.electoral_county = None
            return speaker
        mask_fuzzy = mp_df["family_name"].str.lower() == closest[0].lower()
        row = mp_df.loc[mask_fuzzy].iloc[0]

    speaker.party_deducted = row.get("Partei") or None
    speaker.electoral_county_deducted = row.get("Wahlkreis") or None
    return speaker

# Load settings from env vars
settings = Settings()
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = 'typesystem.xml'
logger.debug("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    # logger.debug("Base typesystem:")
    # logger.debug(typesystem.to_xml())

# Load the Lua communication script
lua_communication_script_filename = "duui-parliament-segmenter.lua"
logger.debug("Loading Lua communication script from \"%s\"", lua_communication_script_filename)


# Request sent by DUUI
# Note, this is transformed by the Lua script
class DUUIRequest(BaseModel):
    # The texts language
    doc_len: int
    #
    lang: str
    #
    text: str

# Documentation response
class DUUIDocumentation(BaseModel):
    # Name of this annotator
    annotator_name: str

    # Version of this annotator
    version: str

    # Annotator implementation language (Python, Java, ...)
    implementation_lang: Optional[str]


# Response sent by DUUI
# Note, this is transformed by the Lua script
class DUUIResponse(BaseModel):
    speeches: List[Speech]
    speakers: List[Speaker]


app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title=settings.annotator_name,
    description="Identification of structures from plenary minutes/plenary debates",
    version=settings.annotator_version,
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "Marius Liebald, Giuseppe Abrami",
        "url": "https://texttechnologylab.org",
        "email": "marius.liebald@ekhist.uu.se",
    },
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)

with open(lua_communication_script_filename, 'rb') as f:
    lua_communication_script = f.read().decode("utf-8")
logger.debug("Lua communication script:")
logger.debug(lua_communication_script_filename)


# Get typesystem of this annotator
@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    # TODO rimgve cassis dependency, as only needed for typesystem at the moment?
    xml = typesystem.to_xml()
    xml_content = xml.encode("utf-8")

    return Response(
        content=xml_content,
        media_type="application/xml"
    )


# Return Lua communication script
@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return lua_communication_script


# Return documentation info
@app.get("/v1/documentation")
def get_documentation():
    return DUUIDocumentation(
        annotator_name=settings.annotator_name,
        version=settings.annotator_version,
        implementation_lang="Python"
    )


# Get input / output of the annotator
@app.get("/v1/details/input_output")
def get_input_output() -> JSONResponse:
    json_item = {
        "inputs": [],
        "outputs": ["org.texttechnologylab.annotation.parliament.Speaker", "org.texttechnologylab.annotation.parliament.Speech"]
    }

    json_compatible_item_data = jsonable_encoder(json_item)
    return JSONResponse(content=json_compatible_item_data)



# Process request from DUUI
@app.post("/v1/process")
def post_process(request: DUUIRequest):

    text = request.text

    # Get list of members of parliament for respective legislative period
    mp_df = get_mp(legislative_period=2) # has to be adjusted

    # Improved regex pattern with anchors and optional parts
    pattern = re.compile(
        r"""
        ^\s*
        (?:
            (?P<trash1>[^A-ZÄÖÜa-zäöüß]*) # trash
            (?P<firstname>[A-ZÄÖÜ]\.)?\s*? # Initial of first name
            (?P<title>[Dv]r\.|Prof\.)?\s*?  # Optional title
            (?P<nobility>(Graf)?(Freiherr)?\s?v?\.?)?\s*? # Optional nobility indication
            (?P<name>[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)\s*  # Name
            (?:
              \(\s*(?P<party>[A-ZÄÖÜa-zäöüß]{,5})\s*\)   # “(Party)”
              \s*,\s*                                 # comma + optional space
            )?
            (?:
              \(\s*(?P<wkr>[A-ZÄÖÜa-zäöüß]{6,})\s*\)   # “(Wahlkreis)”
              \s*,\s*                                 # comma + optional space
            )?
            (?P<trash2>.{,20}) # trash
            (?P<role>Abgeordneter(?:in)?(?:,\sBerichterstatter)?)  # Role
            :
        |
            (?P<trash_ap>[^A-ZÄÖÜa-zäöüß]*) # trash
            (?P<alt_label>Alterspräsident(?:in)?)\s*
            (?P<title_ap>Dr\.|Prof\.)?\s*?  # Optional title
            (?P<name_ap>[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)\s*  # Name
            :
        |
            (?P<trash_p>[^A-ZÄÖÜa-zäöüß]*) # trash
            (?P<president>Präsident(?:in)?)\s*:
        |
            (?P<trash_vp>[^A-ZÄÖÜa-zäöüß]*) # trash
            (?P<vicepresident>Vizepräsident(?:in)?)\s*
            (?P<title_vp>[Dv]r\.|Prof\.)?\s*?  # Optional title
            (?P<name_vp>[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)\s*  # Name
            :
        |
            (?P<trash_bc>[^A-ZÄÖÜa-zäöüß]*) # trash
            (?P<title_bc>[Dv]r\.|Prof\.)?\s*?  # Optional title
            (?P<nobility_bc>(Graf)?\s?v?\.?)?\s*? # Optional nobility indication
            (?P<name_bc>[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)\s*  # Name
            ,\s*?
            (?P<role_bc>[^:]{,100}?)\s*  # Role
            :
        )
        """,
        re.VERBOSE | re.MULTILINE
    )

    def _get(group: str) -> Optional[str]:
        val = match.group(group)
        if val is None:
            return None
        # strip leading/trailing whitespace
        val = re.sub(r'^\s+|\s+$', '', val)
        val = re.sub(r'\n', ' ', val)
        return val if val else None

    speeches: list[Speech] = []
    speakers: list[Speaker] = []

    matches = list(pattern.finditer(text))

    for i, match in enumerate(matches):
        begin, end = match.span()

        # Standard Abgeordneter
        if match.group("name"):
            speaker = Speaker(
                begin=begin,
                end=end,
                label=f"{_get('name')}, {_get('role')}",
                firstname=_get("firstname"),
                name=_get("name"),
                nobility=_get("nobility"),
                title=_get("title"),
                role=_get("role"),
                party=_get("party"),
                electoral_county=_get("wkr")
            )

            speaker = find_mp(speaker=speaker, mp_df=mp_df)

        # Alterspräsident(in)
        elif match.group("alt_label"):
            speaker = Speaker(
                begin=begin,
                end=end,
                label=_get("alt_label"),
                firstname=None,
                name=_get("name_ap"),
                nobility=None,
                title=_get("title_ap"),
                role=_get("alt_label"),
                party=None,
                electoral_county=None
            )

        # Präsident(in)
        elif match.group("president"):
            speaker = Speaker(
                begin=begin,
                end=end,
                label=_get("president"),
                firstname=None,
                name=_get("president"),
                nobility=None,
                title=None,
                role=_get("president"),
                party=None,
                electoral_county=None
            )

        # Vizepräsident(in)
        elif match.group("vicepresident"):
            speaker = Speaker(
                begin=begin,
                end=end,
                label=_get("vicepresident"),
                firstname=None,
                name=_get("name_vp"),
                nobility=None,
                title=_get("title_vp"),
                role=_get("vicepresident"),
                party=None,
                electoral_county=None
            )

        # Staatsbeamter/BC branch
        elif match.group("role_bc"):
            speaker = Speaker(
                begin=begin,
                end=end,
                label=_get("role_bc"),
                firstname=None,
                name=_get("name_bc"),
                nobility=_get("nobility_bc"),
                title=_get("title_bc"),
                role="Staatsbeamter",
                party=None,
                electoral_county=None
            )

        speakers.append(speaker)

        # Determine the speech text bounds
        speech_start = end
        speech_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)


        speeches.append(Speech(
            begin=end,
            end=speech_end,
            speaker=len(speakers) - 1
        ))

    # Return data as JSON
    return DUUIResponse(
        speeches = speeches,
        speakers = speakers
    )
