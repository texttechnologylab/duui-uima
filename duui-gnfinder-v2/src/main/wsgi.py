import asyncio
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Optional, Self

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import UUID5, BaseModel, Field

GNFINDER_PATH: Final[Path] = Path(
    os.environ.get("GNFINDER_PATH", shutil.which("gnfinder"))
)
STARTUP_DELAY: Final[int] = int(os.environ.get("STARTUP_DELAY", "2"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    gnfinder_process = await asyncio.create_subprocess_exec(GNFINDER_PATH, "-p", "8999")
    try:
        # Wait for the gnfinder server to start
        # If the server exits during this time, raise a RuntimeError
        async with asyncio.timeout(STARTUP_DELAY):
            exit_code = await gnfinder_process.wait()
            if exit_code:
                raise RuntimeError(
                    f"GNFinder server exited unexpectedly with code {exit_code}"
                )
    except TimeoutError:
        pass

    async with httpx.AsyncClient(base_url="http://localhost:8999/") as gnfinder_client:
        try:
            (await gnfinder_client.get("api/v1/ping")).raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=httpx.codes.SERVICE_UNAVAILABLE,
                detail="Could not connect to GNFinder server",
            ) from e

        yield {
            "gnfinder_process": gnfinder_process,
            "gnfinder_client": gnfinder_client,
        }

    try:
        gnfinder_process.terminate()
    except ProcessLookupError:
        # Expected during shutdown, process already terminated by context manager
        pass


app = FastAPI(lifespan=lifespan)


with open("communication_layer.lua", "r") as f:
    lua_communication_layer: str = f.read()


@app.get(
    "/v1/communication_layer",
    response_class=PlainTextResponse,
    description="DUUI API v1: Get the Lua communication layer",
)
def get_communication_layer() -> str:
    return lua_communication_layer


class FinderParams(BaseModel):
    text: Annotated[
        str,
        Field(
            min_length=1,
            examples=["Pomatomus saltator and P. major"],
            description="Contains the text which will be checked for scientific names.",
        ),
    ]

    # bytesOffset: Annotated[
    #     bool,
    #     Field(
    #         description="""This flag changes how the position of a detected name in text is calculated. Normally a name's start and end positions are given as the number of UTF-8 characters from the beginning of the text. If bytesOffset flag is true, the start and end offsets are recalculated in the number of bytes."""
    #     ),
    # ] = False

    # returnContent: Annotated[
    #     bool,
    #     Field(
    #         description="""If this flag is true, the text used for the name detection is returned back. This flag is especially useful if the input was not a plain UTF-8 text and had to be prepared for name-finding. Then the returned content can be used together with start and end fields of detected name-strings to locate the strings in the text."""
    #     ),
    # ] = False

    # uniqueNames: Annotated[
    #     bool,
    #     Field(
    #         description="""If this flag is true, the output returns a list of unique names, instead of a list of all name occurrences. Unique list of names does not provide position information of a name in the text."""
    #     ),
    # ] = False

    ambiguousNames: Annotated[
        bool,
        Field(
            description="""If this flag is true, strings which are simultaneously scientific names and "normal" words are not filtered out from the results. For example generic names like America, Cancer, Cafeteria will be returned in the results."""
        ),
    ] = False

    noBayes: Annotated[
        bool,
        Field(
            description="""If this flag is true, only heuristic algorithms are used for name detection."""
        ),
    ] = False

    oddsDetails: Annotated[
        bool,
        Field(
            description="""If true, the result will contain odds of all features used for calculation of NaiveBayes odds. Odds describe probability of a name to be 'real'. The higher the odds, the higher the probability that a detected name is not a false positive. Odds are calculated by multiplication of the odds of separate features. Odds details explain how the final odds value is calculated."""
        ),
    ] = False

    language: Annotated[
        Literal["eng", "ger", "detect"],
        Field(
            description="""The language of the text. Language value is used for calculation of Bayesian odds. If this parameter is not given, eng is used by default. Currently only English and German languages are supported."""
        ),
    ] = "detect"

    # wordsAround: Annotated[
    #     int,
    #     Field(
    #         ge=0,
    #         description="""Allows to see the context surrounding a name-string. The wordsAround parameter sets the number of words located immediately before or after a detected name. These words are then returned in the output. Default is 0, maximum value is 5.""",
    #     ),
    # ] = 0

    verification: Annotated[
        bool,
        Field(
            description="""When this flag is true, there is an additional verification step for detected names. This step requires internet connection and uses https://verifier.globalnames.org/api/v1 for verification queries."""
        ),
    ] = True

    sources: Annotated[
        list[Annotated[int, Field(gt=0)]],
        Field(
            description="""List of sources used for name detection. The list contains IDs of sources. If the list is empty, all sources are used. The default value is [11], which is the GBIF Backbone Taxonomy. The list of IDs can be found at https://verifier.globalnames.org/data_sources"""
        ),
    ] = Field(default_factory=lambda: [11], examples=[[11]])

    allMatches: Annotated[
        bool,
        Field(
            description="""When this option is true all found results are returned, not only the bestResult. The bestResult field in this case is null, and results field should contain found results of the matches."""
        ),
    ] = False


class FinderScoreDetails(BaseModel):
    """Explains which score properties participated in sorting of the data.

    Properties located higher provide more weight in the sorting.

    All properties vary from 0 to 1, where 0 has no influence on sorting, and 1 has the highest priority for sorting.
    """

    infraSpecificRankScore: Annotated[
        float, Field(description="""Is at maximum if infraspecific ranks match.""")
    ]

    fuzzyLessScore: Annotated[
        float, Field(description="""It is at maximum for exact matches.""")
    ]

    curatedDataScore: Annotated[
        float,
        Field(description="""It is at maximum for data-sources marked as curated."""),
    ]

    authorMatchScore: Annotated[
        float,
        Field(
            description="""It is at maximum if authors and year matched completely."""
        ),
    ]

    acceptedNameScore: Annotated[
        float,
        Field(
            description="""It is at maximum if matched name marked as currently accepted."""
        ),
    ]

    parsingQualityScore: Annotated[
        float,
        Field(
            description="""It is at maximum if no name-parsing problems were detected."""
        ),
    ]


type MatchType = Literal[
    "NoMatch",
    "PartialExact",
    "PartialFuzzy",
    "PartialFuzzyRelaxed",
    "Exact",
    "ExactSpeciesGroup",
    "Fuzzy",
    "FuzzyRelaxed",
    "FuzzySpeciesGroup",
    "FuzzySpeciesGroupRelaxed",
    "Virus",
    "FacetedSearch",
]


class FinderResultData(BaseModel):
    """A successsful match of input name-string to a record in a data-source. ResultData with the highest score appears as a best match. ResultData from selected data-sources are aggregated into results array. Scoring takes the following into account (sorted from the highest impact to the least):
    - infraspecific rank (Aus bus var. cus vs Aus bus f. cus)
    - fuzzy matching edit distance (less is better)
    - authorship matching
    - accepted name vs synonyms (accepted name wins)
    - parsing quality
    """

    dataSourceId: Annotated[
        int,
        Field(
            description="An ID of the data-source where the match was found. This is a local ID at GlobalNames and it has been stable for more than 10 years."
        ),
    ]

    dataSourceTitleShort: Annotated[
        str, Field(description="A simplified/abbreviated title of the data-source.")
    ]

    curation: Annotated[
        str,
        Field(
            description="A curation level for the record's data-source. If data-source is known to be curated, there is less risk of misspelling of the matched name."
        ),
    ]

    recordId: Annotated[
        str,
        Field(
            description="The record ID of the matchedName from the data-source. If the data-source does not provide record ID, the ID is generated by gnames and is prepended with a 'gn_' prefix."
        ),
    ]

    globalId: Optional[
        Annotated[
            str,
            Field(description="A globally unique record ID (UUID, LSID, DOI etc)."),
        ]
    ] = None

    localId: Optional[
        Annotated[
            str,
            Field(
                description="An additional ID that is used internally by the data-source. Quite often such IDs are used by data-sources to generate URLs to their records."
            ),
        ]
    ] = None

    outlink: Optional[
        Annotated[
            str,
            Field(
                description="An external URL provided by the data-source for the record."
            ),
        ]
    ] = None

    entryDate: Annotated[
        str,
        Field(
            description="A date when a record was harvested to gnames. It follows a pattern 'YYYY-MM-DD'."
        ),
    ]

    matchedName: Annotated[
        str, Field(description="A verbatim scientfic name matched to the input.")
    ]

    matchedCardinality: Annotated[
        int,
        Field(
            description="A cardinality value of the matched name. Binomial names have cardinality 2, uninomials have cardinality 1 etc. Cardinality 0 means we cannot determine cardinality of a string."
        ),
    ]

    matchedCanonicalSimple: Annotated[
        str,
        Field(
            description="The most normalized canonical form of a scientific name. It does not contain infraspecific ranks, hybrid signs for named hybrids, authors or years. This canonical form is the best for matching names."
        ),
    ]

    matchedCanonicalFull: Annotated[
        str,
        Field(
            description="A less normalized canonical form of the matched name with more details. This canonical form is better for presentation and creating matching scores."
        ),
    ]

    currentRecordId: Annotated[
        str,
        Field(
            description="If the data-source allows taxonomic resolution of a name, currentRecordId will contain the data-source record ID of the currently accepted name for the matched taxon."
        ),
    ]

    currentName: Annotated[
        str,
        Field(
            description="The currently accepted name for the taxon according to the data-source."
        ),
    ]

    currentCardinality: Annotated[
        int, Field(description="Cardinality of the currently accepted name.")
    ]

    currentCanonicalSimple: Annotated[
        str,
        Field(
            description="The most normalized canonical form of the currently accepted scientific name."
        ),
    ]

    currentCanonicalFull: Annotated[
        str,
        Field(
            description="A less normalized canonical form of the currently accepted scientific name."
        ),
    ]

    isSynonym: Annotated[
        bool,
        Field(
            description="A flag, that is true if the data-source regards matched name as an any kind of synonym to the currently accepted name."
        ),
    ]

    classificationPath: Annotated[
        str,
        Field(
            description="A classification path for the matched taxon provided by the data-source."
        ),
    ]

    classificationRanks: Annotated[
        str, Field(description="The ranks of the classificationPath entries.")
    ]

    editDistance: Annotated[
        int,
        Field(
            ge=0,
            description="Edit distance according to Levenshtein algorithm in case of fuzzy matching. Exact matches would have editDistance at 0.",
        ),
    ]

    editDistanceStem: Optional[
        Annotated[
            int,
            Field(
                ge=0,
                description="Edit distance according to Levenshtein algorithm between stemmed versions of canonical forms of input name and matched name. Stemmed versions are the most normalized forms of scientific names with stripped suffixes and transformation of some characters (v->u, j->i). They are not provided by the API, but can be obtained with gnparser app.",
            ),
        ]
    ] = None

    matchType: Annotated[
        MatchType,
        Field(
            description="A category that describes the matching type between the input name-string and the matched name from the data-source. The categories are:<ul><li>NoMatch</li><li>PartialExact</li><li>PartialFuzzy</li><li>PartialFuzzyRelaxed</li><li>Exact</li><li>ExactSpeciesGroup</li><li>Fuzzy</li><li>FuzzyRelaxed</li><li>FuzzySpeciesGroup</li><li>FuzzySpeciesGroupRelaxed</li><li>Virus</li><li>FacetedSearch</li></ul>"
        ),
    ]

    scoreDetails: FinderScoreDetails


class FinderVerification(BaseModel):
    id: Annotated[
        UUID5,
        Field(
            description="An UUID v5 generated from the input string using 'globalnames.org' DNS domain name as a seed. This UUID can be used as a distributed global unique identifier of the string and can be created independently in any popular programming language."
        ),
    ]

    name: Annotated[
        str,
        Field(
            description="An input name-string entered for the verification, or found by search."
        ),
    ]

    matchType: Annotated[
        Literal[
            "NoMatch", "Exact", "Fuzzy", "PartialExact", "PartialFuzzy", "FacetedSearch"
        ],
        Field(
            description="Indicates the quality of verification. Possible values are: <ul><li>NoMatch (no matched name found)</li><li>Exact (exact match to a canonical form or a verbatim string)</li><li>Fuzzy (fuzzy match to a canonical form)</li><li>PartialExact (match after removing last or middle epithets)</li><li>PartialFuzzy (fuzzy partial match after removing some parts)</li><li>FacetedSearch (match by a faceted search)</li></ul>"
        ),
    ]

    bestResult: Optional[
        Annotated[
            FinderResultData,
            Field(
                description="The best match of the input name-string to a record in a data-source. The bestResult and results fields are mutually exclusive."
            ),
        ]
    ] = None

    results: Optional[
        Annotated[
            list[FinderResultData],
            Field(
                description="All found matched records returned if withAllMatches option is given. The bestResult and results fields are mutually exclusive."
            ),
        ]
    ] = None

    dataSourcesNum: Optional[
        Annotated[
            int,
            Field(
                ge=0,
                description="The number of data-sources where the string had a match to a scientific name.",
            ),
        ]
    ] = None

    curation: Annotated[
        Literal["NotCurated", "AutoCurated", "Curated"],
        Field(
            description="The best curation level from matched data-sources. Possible values are:<ul><li>NotCurated</li><li>AutoCurated</li><li>Curated</li></ul><br>If this flag has a NotCurated value, it does not mean that original data-source is not curated, rather it means the curation level is not known to gnames."
        ),
    ]

    error: Optional[
        Annotated[
            str,
            Field(
                description="Normally an empty string. In case of an unexpected error, it will be shown here. If error is not empty, it means that matching failed because of a bug in the system."
            ),
        ]
    ] = None


class FinderOddsDetails(BaseModel):
    feature: str
    odds: float


class FinderName(BaseModel):
    cardinality: Annotated[
        int,
        Field(
            ge=0,
            le=3,
            description="A cardinality value of the detected name. Binomial names have cardinality 2, uninomials have cardinality 1 etc. Cardinality 0 means we cannot determine cardinality of a string.",
        ),
    ]

    verbatim: Annotated[
        str,
        Field(
            description="Detected name, as it was found in text with new lines, punctuations etc. New lines for better readability are substituted with ' ' character."
        ),
    ]

    name: Annotated[
        str,
        Field(
            description="String representation of a name after cleaning. If a name contains non-standard characters introduced during optical character recognition, they are substituted with '�' character."
        ),
    ]

    oddsLog10: Optional[
        Annotated[
            float,
            Field(
                description="Odds reflect a probability that some feature is true. In our case the higher the odds, the higher is the probability that a detected name is 'real'. Equal odds are a ration 1/1, meaning for every 'true' event there is one 'false' event, the same can be expressed as a probability 0.5. We provide a log base 10 of odds. That means that log10 2 is the same as 100, log1 is the same as 10 log10 0 is the same as 1. If log is negative, it means Bayes algorithm considers an event to be unlikely."
            ),
        ]
    ] = None

    oddsDetails: Optional[
        Annotated[
            list[FinderOddsDetails],
            Field(description="Shows how final odds were calculated."),
        ]
    ] = None

    start: Annotated[
        int,
        Field(
            ge=0,
            description="The offset in UTF-8 characters from the beginning of the text to the start of a name. If withPositionInBytes is true, offset is calculated in bytes instead.",
        ),
    ]

    offsetEnd: Optional[
        Annotated[
            int,
            Field(
                gt=0,
                description="The offset in UTF-8 characters from the beginning of the text to the end of a name. If withPositionInBytes is true, offset is calculated in bytes instead.",
            ),
        ]
    ] = None

    annotNomen: Optional[
        Annotated[
            str,
            Field(
                description="If a name has a nomenclatural annotation (for example 'Aus bus sp. nov') this field will provide the verbatim form of the annotation."
            ),
        ]
    ] = None

    annotNomenType: Optional[
        Annotated[
            Literal["NO_ANNOT", "SP_NOV", "SUBSP_NOV", "COMB_NOV"],
            Field(
                description="Category of the nomenclatural annotation. The same annotation type might have significant variability in literature. This property provides a normalized value of an annotation."
            ),
        ]
    ] = None

    wordsBefore: Optional[
        Annotated[
            list[str],
            Field(
                description="A list of words that precede a name. The number of words is determined by the wordsAround parameter."
            ),
        ]
    ] = None

    wordsAfter: Optional[
        Annotated[
            list[str],
            Field(
                description="A list of words that appear right after a name. The number of words is determined by the wordsAround parameter."
            ),
        ]
    ] = None

    verification: Optional[FinderVerification]


class FinderResult(BaseModel):
    metadata: dict[str, Any]
    names: list[FinderName]


class MetadataType(BaseModel):
    date: str
    version: str
    language: str
    other: Optional[list[tuple[str, str]]] = None


class TaxonType(BaseModel):
    begin: Annotated[int, Field(ge=0)]
    end: Annotated[int, Field(ge=0)]
    value: str
    identifier: str
    cardinality: Annotated[int, Field(ge=0, le=3)]
    oddsLog10: Optional[float] = None
    oddsDetails: Optional[list[FinderOddsDetails]] = None


class TaxonVerifiedType(TaxonType):
    dataSourceId: int
    recordId: str
    globalId: Optional[str] = None
    localId: Optional[str] = None
    outlink: Optional[str] = None
    sortScore: float
    matchedName: str
    matchedCanonicalSimple: str
    matchedCanonicalFull: str
    taxonomicStatus: Literal["Accepted", "Synonym", "N/A"]
    currentName: str
    matchType: MatchType
    editDistance: Annotated[int, Field(ge=0)]

    @classmethod
    def from_finder(cls, name: FinderName, data: FinderResultData) -> Self:
        return cls(
            begin=name.start,
            end=name.offsetEnd if name.offsetEnd else name.start + len(name.verbatim),
            value=data.currentCanonicalSimple,
            identifier=data.outlink if data.outlink else data.recordId,
            cardinality=name.cardinality,
            oddsLog10=name.oddsLog10,
            oddsDetails=name.oddsDetails,
            dataSourceId=data.dataSourceId,
            recordId=data.recordId,
            globalId=data.globalId,
            localId=data.localId,
            outlink=data.outlink,
            sortScore=data.scoreDetails.infraSpecificRankScore,
            matchedName=data.matchedName,
            currentName=data.currentName,
            matchedCanonicalSimple=data.matchedCanonicalSimple,
            matchedCanonicalFull=data.matchedCanonicalFull,
            taxonomicStatus=(
                "Synonym"
                if data.isSynonym
                else "Accepted"
                if data.currentRecordId
                else "N/A"
            ),
            matchType=data.matchType,
            editDistance=data.editDistance,
        )


class DuuiResponse(BaseModel):
    metadata: MetadataType
    results: list[TaxonType | TaxonVerifiedType]


async def fetch_gnfinder_results(
    gnfinder_client: httpx.AsyncClient, params: FinderParams
) -> FinderResult:
    gnfinder_response = await gnfinder_client.post(
        "api/v1/find",
        json={"format": "json"} | params.model_dump(exclude_unset=True),
        timeout=300,
    )
    gnfinder_response.raise_for_status()
    return FinderResult.model_validate(gnfinder_response.json(), strict=False)


@app.post("/api/v1/find", description="GNFinder API v1 find endpoint")
async def api_v1_find(
    params: FinderParams,
    request: Request,
) -> FinderResult:
    return await fetch_gnfinder_results(request.state.gnfinder_client, params)


@app.post("/v1/process", description="DUUI API v1 process endpoint")
async def v1_process(
    params: FinderParams,
    request: Request,
) -> DuuiResponse:
    finder_result = await fetch_gnfinder_results(request.state.gnfinder_client, params)

    metadata = MetadataType(
        date=finder_result.metadata["date"],
        version=finder_result.metadata["gnfinderVersion"],
        language=finder_result.metadata["language"],
        other=[
            (key, str(value))
            for key, value in finder_result.metadata.items()
            if key.startswith("with")
        ],
    )

    results = []
    for name in finder_result.names:
        if name.verification and name.verification.matchType != "NoMatch":
            if name.verification.results:
                for result in name.verification.results:
                    results.append(
                        TaxonVerifiedType.from_finder(name=name, data=result)
                    )
            else:
                result = name.verification.bestResult
                results.append(TaxonVerifiedType.from_finder(name=name, data=result))
        else:
            results.append(
                TaxonType(
                    begin=name.start,
                    end=(
                        name.offsetEnd
                        if name.offsetEnd
                        else name.start + len(name.verbatim)
                    ),
                    value=name.name,
                    identifier=name.verbatim,
                    cardinality=name.cardinality,
                    oddsLog10=name.oddsLog10,
                    oddsDetails=name.oddsDetails,
                )
            )

    return DuuiResponse(metadata=metadata, results=results)
