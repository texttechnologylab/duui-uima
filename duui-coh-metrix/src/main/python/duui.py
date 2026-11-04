"""DUUI layer of Coh-Metrix 0.5.0-py.

The indices are computed by `cohmetrix_core.py`, an unchanged copy of `cohmetrix/core.py` of coh-metrix_resurrection (Coh-Metrix
0.5.0-py: German and English read in Universal Dependencies). This file only adds what DUUI needs, with the interface of
duui-coh-metrix 0.2.0:
- the index catalog with the two-layer labels (`label_v3` public, `label_ttlab` internal with uppercase resource groups; README,
  "Index labels"); the core keeps its own labels, the map between both is `label_map.json` of coh-metrix_resurrection;
- `include_indices` / `exclude_indices` (README, "Selecting indices");
- the token vectors from WordEmbedding annotations or SpacyToken.vector (README, "Vectors");
- the routes of a DUUI component, and `/v1/indices`.
"""
import logging
import re
import threading
import unicodedata
from functools import lru_cache
from platform import python_version
from sys import version as sys_version
from time import time
from typing import Any, Dict, List, Optional, Set, Tuple

import duui_logging
from cassis import load_typesystem
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    annotator_name: str
    annotator_version: str
    log_level: str = "INFO"

    class Config:
        env_prefix = 'duui_coh_metrix_'


settings = Settings()

_configured_log_level = getattr(logging, (settings.log_level or "INFO").strip().upper(), logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(_configured_log_level)
logger.propagate = True
logger.handlers.clear()
duui_logging.install(level=_configured_log_level, capture_warnings=True, keep_stderr=True)

logger.info("TTLab TextImager DUUI Coh-Metrix")
logger.info("Name: %s", settings.annotator_name)
logger.info("Version: %s", settings.annotator_version)

# the core loads its resources at import (paths from the DUUI_COH_METRIX_* variables; a missing file leaves its indices None)
try:
    from . import cohmetrix_core as core
except ImportError:  # imported outside the package (tests)
    import cohmetrix_core as core

# The core keeps per-text caches in module state, so one request at a time (CPU-bound: threads would not run in parallel anyway).
_CORE_LOCK = threading.Lock()

TEXTIMAGER_ANNOTATOR_OUTPUT_TYPES = []

TEXTIMAGER_ANNOTATOR_INPUT_TYPES = [
    "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph",
    "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence",
    "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token",
    "de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS",
    "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma",
    "de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.morph.MorphologicalFeatures",
    "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency",
    "org.texttechnologylab.uima.type.WordEmbedding",
]

SUPPORTED_LANGS = list(core.SUPPORTED_LANGS)

# Index catalog (README, "Index labels"; background: docs/label-scheme.md): (index, type_name, label_v3, label_ttlab template,
# comparability EN, comparability DE, label_v2), in output order. The templates are filled per request with the language's
# resource groups (_GROUPS) and the token vectors ({vectors}: SPACYlg, ...). Comparability says how the value relates to
# Coh-Metrix 3.0: original, counterpart (another resource for the same construct, e.g. GermaNet for WordNet, DiMLex for
# Coh-Metrix's connective lists, native norms for MRC), translated (MRC translated into German), corpus (frequencies from
# another corpus than CELEX), llmlist, emb, refit, ext.
INDEX_CATALOG: List[Tuple[int, str, str, str, str, str, Optional[str]]] = [
    (1, "Descriptive", "DESPC", "DESPC", "original", "original", "READNP"),
    (2, "Descriptive", "DESSC", "DESSC", "original", "original", "READNS"),
    (3, "Descriptive", "DESWC", "DESWC", "original", "original", "READNW"),
    (4, "Descriptive", "DESPL", "DESPL", "original", "original", "READAPL"),
    (5, "Descriptive", "DESPLd", "DESPLd", "original", "original", "n/a"),
    (6, "Descriptive", "DESSL", "DESSL", "original", "original", "READASL"),
    (7, "Descriptive", "DESSLd", "DESSLd", "original", "original", "n/a"),
    (8, "Descriptive", "DESWLsy", "DESWLsy", "original", "original", "READASW"),
    (9, "Descriptive", "DESWLsyd", "DESWLsyd", "original", "original", "n/a"),
    (10, "Descriptive", "DESWLlt", "DESWLlt", "original", "original", "n/a"),
    (11, "Descriptive", "DESWLltd", "DESWLltd", "original", "original", "n/a"),
    (12, "Text Easability Principal Component Scores", "PCNARz", "PCNARz_{easability}", "refit", "refit", "n/a"),
    (13, "Text Easability Principal Component Scores", "PCNARp", "PCNARp_{easability}", "refit", "refit", "n/a"),
    (14, "Text Easability Principal Component Scores", "PCSYNz", "PCSYNz_{easability}", "refit", "refit", "n/a"),
    (15, "Text Easability Principal Component Scores", "PCSYNp", "PCSYNp_{easability}", "refit", "refit", "n/a"),
    (16, "Text Easability Principal Component Scores", "PCCNCz", "PCCNCz_{easability}", "refit", "refit", "n/a"),
    (17, "Text Easability Principal Component Scores", "PCCNCp", "PCCNCp_{easability}", "refit", "refit", "n/a"),
    (18, "Text Easability Principal Component Scores", "PCREFz", "PCREFz_{easability}", "refit", "refit", "n/a"),
    (19, "Text Easability Principal Component Scores", "PCREFp", "PCREFp_{easability}", "refit", "refit", "n/a"),
    (20, "Text Easability Principal Component Scores", "PCDCz", "PCDCz_{easability}", "refit", "refit", "n/a"),
    (21, "Text Easability Principal Component Scores", "PCDCp", "PCDCp_{easability}", "refit", "refit", "n/a"),
    (22, "Text Easability Principal Component Scores", "PCVERBz", "PCVERBz_{easability}", "refit", "refit", "n/a"),
    (23, "Text Easability Principal Component Scores", "PCVERBp", "PCVERBp_{easability}", "refit", "refit", "n/a"),
    (24, "Text Easability Principal Component Scores", "PCCONNz", "PCCONNz_{easability}", "refit", "refit", "n/a"),
    (25, "Text Easability Principal Component Scores", "PCCONNp", "PCCONNp_{easability}", "refit", "refit", "n/a"),
    (26, "Text Easability Principal Component Scores", "PCTEMPz", "PCTEMPz_{easability}", "refit", "refit", "n/a"),
    (27, "Text Easability Principal Component Scores", "PCTEMPp", "PCTEMPp_{easability}", "refit", "refit", "n/a"),
    (28, "Referential Cohesion", "CRFNO1", "CRFNO1", "original", "original", "CRFBN1um"),
    (29, "Referential Cohesion", "CRFAO1", "CRFAO1", "original", "original", "CRFBA1um"),
    (30, "Referential Cohesion", "CRFSO1", "CRFSO1", "original", "original", "CRFBS1um"),
    (31, "Referential Cohesion", "CRFNOa", "CRFNOa", "original", "original", "CRFBNaum"),
    (32, "Referential Cohesion", "CRFAOa", "CRFAOa", "original", "original", "CRFBAaum"),
    (33, "Referential Cohesion", "CRFSOa", "CRFSOa", "original", "original", "CRFBSaum"),
    (34, "Referential Cohesion", "CRFCWO1", "CRFCWO1", "original", "original", "CRFPC1um"),
    (35, "Referential Cohesion", "CRFCWO1d", "CRFCWO1d", "original", "original", "n/a"),
    (36, "Referential Cohesion", "CRFCWOa", "CRFCWOa", "original", "original", "CRFPCaum"),
    (37, "Referential Cohesion", "CRFCWOad", "CRFCWOad", "original", "original", "n/a"),
    (38, "LSA", "LSASS1", "LSASS1_EMB_{vectors}", "emb", "emb", "LSAassa"),
    (39, "LSA", "LSASS1d", "LSASS1d_EMB_{vectors}", "emb", "emb", "LSAassd"),
    (40, "LSA", "LSASSp", "LSASSp_EMB_{vectors}", "emb", "emb", "LSApssa"),
    (41, "LSA", "LSASSpd", "LSASSpd_EMB_{vectors}", "emb", "emb", "LSApssd"),
    (42, "LSA", "LSAPP1", "LSAPP1_EMB_{vectors}", "emb", "emb", "LSAppa"),
    (43, "LSA", "LSAPP1d", "LSAPP1d_EMB_{vectors}", "emb", "emb", "LSAppd"),
    (44, "LSA", "LSAGN", "LSAGN_EMB_{vectors}", "emb", "emb", "LSAGN"),
    (45, "LSA", "LSAGNd", "LSAGNd_EMB_{vectors}", "emb", "emb", "n/a"),
    (46, "Lexical Diversity", "LDTTRc", "LDTTRc", "original", "original", "TYPTOKc"),
    (47, "Lexical Diversity", "LDTTRa", "LDTTRa", "original", "original", "n/a"),
    (48, "Lexical Diversity", "LDMTLDa", "LDMTLDa", "original", "original", "LEXDIVTD"),
    (49, "Lexical Diversity", "LDVOCDa", "LDVOCDa", "original", "original", "LEXDIVVD"),
    (50, "Connectives", "CNCAll", "CNCAll_{connectives}", "counterpart", "counterpart", "CONi"),
    (51, "Connectives", "CNCCaus", "CNCCaus_{connectives}", "counterpart", "counterpart", "CONCAUSi"),
    (52, "Connectives", "CNCLogic", "CNCLogic_{connectives}", "counterpart", "counterpart", "CONLOGi"),
    (53, "Connectives", "CNCADC", "CNCADC_{connectives}", "counterpart", "counterpart", "CONADVCONi"),
    (54, "Connectives", "CNCTemp", "CNCTemp_{connectives}", "counterpart", "counterpart", "CONTEMPi"),
    (55, "Connectives", "CNCTempx", "CNCTempx_{connectives}", "counterpart", "counterpart", "CONTEMPEXi"),
    (56, "Connectives", "CNCAdd", "CNCAdd_{connectives}", "counterpart", "counterpart", "CONADDi"),
    (57, "Connectives", "CNCPos", "CNCPos_{connectives}", "counterpart", "counterpart", "n/a"),
    (58, "Connectives", "CNCNeg", "CNCNeg_{connectives}", "counterpart", "counterpart", "n/a"),
    (59, "Situation Model", "SMCAUSv", "SMCAUSv_{wn}", "original", "counterpart", "CAUSV"),
    (60, "Situation Model", "SMCAUSvp", "SMCAUSvp_{wn}_{connectives}", "counterpart", "counterpart", "CAUSVP"),
    (61, "Situation Model", "SMINTEp", "SMINTEp_{wn}", "original", "counterpart", "INTEi"),
    (62, "Situation Model", "SMCAUSr", "SMCAUSr_{wn}_{connectives}", "counterpart", "counterpart", "CAUSC"),
    (63, "Situation Model", "SMINTEr", "SMINTEr_{wn}", "original", "counterpart", "INTEC"),
    (64, "Situation Model", "SMCAUSlsa", "SMCAUSlsa_EMB_{vectors}", "emb", "emb", "CAUSLSA"),
    (65, "Situation Model", "SMCAUSwn", "SMCAUSwn_{wn}", "original", "counterpart", "CAUSWN"),
    (66, "Situation Model", "SMTEMP", "SMTEMP_{temporal}", "llmlist", "llmlist", "TEMPta"),
    (67, "Syntactic Complexity", "SYNLE", "SYNLE", "original", "original", "SYNLE"),
    (68, "Syntactic Complexity", "SYNNP", "SYNNP", "original", "original", "SYNNP"),
    (69, "Syntactic Complexity", "SYNMEDpos", "SYNMEDpos", "original", "original", "MEDwtm"),
    (70, "Syntactic Complexity", "SYNMEDwrd", "SYNMEDwrd", "original", "original", "MEDawm"),
    (71, "Syntactic Complexity", "SYNMEDlem", "SYNMEDlem", "original", "original", "MEDalm"),
    (72, "Syntactic Complexity", "SYNSTRUTa", "SYNSTRUTa", "original", "original", "STRUTa"),
    (73, "Syntactic Complexity", "SYNSTRUTt", "SYNSTRUTt", "original", "original", "STRUTt"),
    (74, "Syntactic Pattern Density", "DRNP", "DRNP", "original", "original", "n/a"),
    (75, "Syntactic Pattern Density", "DRVP", "DRVP", "original", "original", "n/a"),
    (76, "Syntactic Pattern Density", "DRAP", "DRAP", "original", "original", "n/a"),
    (77, "Syntactic Pattern Density", "DRPP", "DRPP", "original", "original", "n/a"),
    (78, "Syntactic Pattern Density", "DRPVAL", "DRPVAL", "original", "original", "AGLSPSVi"),
    (79, "Syntactic Pattern Density", "DRNEG", "DRNEG_LLMlist", "llmlist", "llmlist", "DENNEGi"),
    (80, "Syntactic Pattern Density", "DRGERUND", "DRGERUND", "original", "original", "GERUNDi"),
    (81, "Syntactic Pattern Density", "DRINF", "DRINF", "original", "original", "INFi"),
    (82, "Word Information", "WRDNOUN", "WRDNOUN", "original", "original", "NOUNi"),
    (83, "Word Information", "WRDVERB", "WRDVERB", "original", "original", "VERBi"),
    (84, "Word Information", "WRDADJ", "WRDADJ", "original", "original", "ADJi"),
    (85, "Word Information", "WRDADV", "WRDADV", "original", "original", "ADVi"),
    (86, "Word Information", "WRDPRO", "WRDPRO", "original", "original", "DENPRPi"),
    (87, "Word Information", "WRDPRP1s", "WRDPRP1s", "original", "original", "n/a"),
    (88, "Word Information", "WRDPRP1p", "WRDPRP1p", "original", "original", "n/a"),
    (89, "Word Information", "WRDPRP2", "WRDPRP2", "original", "original", "PRO2i"),
    (90, "Word Information", "WRDPRP3s", "WRDPRP3s", "original", "original", "n/a"),
    (91, "Word Information", "WRDPRP3p", "WRDPRP3p", "original", "original", "n/a"),
    (92, "Word Information", "WRDFRQc", "WRDFRQc_SUBTLEX", "corpus", "corpus", "FRCLacwm"),
    (93, "Word Information", "WRDFRQa", "WRDFRQa_SUBTLEX", "corpus", "corpus", "FRCLaewm"),
    (94, "Word Information", "WRDFRQmc", "WRDFRQmc_SUBTLEX", "corpus", "corpus", "FRCLmcsm"),
    (95, "Word Information", "WRDAOAc", "WRDAOAc_{aoa}", "counterpart", "counterpart", "WRDAacwm"),
    (96, "Word Information", "WRDFAMc", "WRDFAMc_{mrc}", "original", "translated", "WRDFacwm"),
    (97, "Word Information", "WRDCNCc", "WRDCNCc_{concreteness}", "counterpart", "counterpart", "WRDCacwm"),
    (98, "Word Information", "WRDIMGc", "WRDIMGc_{imagery}", "original", "counterpart", "WRDIacwm"),
    (99, "Word Information", "WRDMEAc", "WRDMEAc_{mrc}", "original", "translated", "WRDMacwm"),
    (100, "Word Information", "WRDPOLc", "WRDPOLc_{wn}", "original", "counterpart", "POLm"),
    (101, "Word Information", "WRDHYPn", "WRDHYPn_{wn}", "original", "counterpart", "HYNOUNaw"),
    (102, "Word Information", "WRDHYPv", "WRDHYPv_{wn}", "original", "counterpart", "HYVERBaw"),
    (103, "Word Information", "WRDHYPnv", "WRDHYPnv_{wn}", "original", "counterpart", "HYPm"),
    (104, "Readability", "RDFRE", "RDFRE", "original", "counterpart", "READFRE"),
    (105, "Readability", "RDFKGL", "RDFKGL", "original", "original", "READFKGL"),
    (1001, "Readability", "RDFOG_ext", "RDFOG_ext", "ext", "ext", None),
    (1002, "Readability", "RDSMOG_ext", "RDSMOG_ext", "ext", "ext", None),
    (1003, "Readability", "RDARI_ext", "RDARI_ext", "ext", "ext", None),
    (1004, "Readability", "RDCLI_ext", "RDCLI_ext", "ext", "ext", None),
    (1005, "Readability", "RDLW_ext", "RDLW_ext", "ext", "ext", None),
    (1006, "Readability", "RDDCRS_ext", "RDDCRS_ext_TEXTSTAT", "ext", "ext", None),
    (1007, "Readability", "RDSPACHE_ext", "RDSPACHE_ext_TEXTSTAT", "ext", "ext", None),
    (1008, "Readability", "RDWSTF1_ext", "RDWSTF1_ext", "ext", "ext", None),
    (1009, "Readability", "RDWSTF2_ext", "RDWSTF2_ext", "ext", "ext", None),
    (1010, "Readability", "RDWSTF3_ext", "RDWSTF3_ext", "ext", "ext", None),
    (1011, "Readability", "RDWSTF4_ext", "RDWSTF4_ext", "ext", "ext", None),
    (106, "Readability", "RDL2", "RDL2_SUBTLEX", "corpus", "corpus", "L2"),
]
_CATALOG_BY_INDEX = {entry[0]: entry for entry in INDEX_CATALOG}

# The public labels and the groups must stay distinct when compared case-insensitively (index selection).
assert len({label_v3.lower() for _, _, label_v3, *_ in INDEX_CATALOG}
           | {type_name.lower() for _, type_name, *_ in INDEX_CATALOG}) \
       == len(INDEX_CATALOG) + len({type_name for _, type_name, *_ in INDEX_CATALOG})

# The resource groups of each language (map to the core's own labels: label_map.json of coh-metrix_resurrection). Every resource
# is named, Coh-Metrix's own included (WN, MRC); the core has no fallback, so a missing resource gives None, not another label.
_GROUPS = {
    "de": {"wn": "GN", "connectives": "DIMLEX", "temporal": "LLMlist_GN", "aoa": "BIRCHENOUGH", "mrc": "MRCtr",
           "concreteness": "KOEPER", "imagery": "KOEPER", "easability": "REFIT_REFv1"},
    "en": {"wn": "WN", "connectives": "DIMLEXeng_GUM", "temporal": "LLMlist", "aoa": "KUPERMAN", "mrc": "MRC",
           "concreteness": "BRYSBAERT", "imagery": "MRC", "easability": "REFIT_REFa"},
}

# version of every index: the computation of Coh-Metrix 0.4.0-py (duui-coh-metrix 0.2.0 had 1.0 for its own computation; the
# values differ for the same text, since this core reads UD and changed most indices, see the core's README)
INDEX_VERSION = "2.0"
# 2.1: the indices whose computation changed in 0.5.0-py, per language (README, "Index version": minor, values change; the user,
# 2026-11-03): SYNNP every word of the noun chunk except its head; English SYNLE the words before the lexical verb; English
# DRGERUND every -ing verb form (Penn VBG), as the original program; the PC* from the refitted easability models (cr3)
_INDEX_VERSION_0_5_0 = "2.1"
_PC_INDICES = frozenset(label_v3 for _, type_name, label_v3, *_ in INDEX_CATALOG
                        if type_name == "Text Easability Principal Component Scores")
_CHANGED_IN_0_5_0 = {"de": frozenset({"SYNNP"}) | _PC_INDICES,
                     "en": frozenset({"SYNNP", "SYNLE", "DRGERUND"}) | _PC_INDICES}


def _index_version(label_v3: str, lang: str) -> str:
    return _INDEX_VERSION_0_5_0 if label_v3 in _CHANGED_IN_0_5_0[lang] else INDEX_VERSION

# The token vectors (README, "Vectors"): their name is a label group, e.g. SPACYlg.
_VECTOR_GROUP_FORMAT = re.compile(r"^[A-Z][A-Z0-9]*[a-z0-9]*$")
_SPACY_MODEL_SIZES = ("sm", "md", "lg", "trf")
_VECTOR_INDICES = frozenset(label_v3 for _, type_name, label_v3, *_ in INDEX_CATALOG
                            if type_name == "LSA" or label_v3 == "SMCAUSlsa")
# The Text Easability models were fitted with these vectors (the core matches the model inputs by its own labels, which name the
# spaCy model); other vectors leave the components None, as other resources do.
_EASABILITY_VECTORS = {"de": ("SPACYlg", "de_core_news_lg"), "en": ("SPACYlg", "en_core_web_lg")}


def _vectors_group(vectors: Optional[str], vectors_model: Optional[str], vectors_source: Optional[str] = None,
                   vectors_available: Optional[str] = None) -> Tuple[str, Optional[str]]:
    # (label group, error); as in duui-coh-metrix 0.2.0
    if vectors_source == "WordEmbedding":
        # the shortName chosen by the producer or user is the group, whatever tool made the vectors
        if not vectors:
            if vectors_available:
                return "INVALID", (f"Token vectors of several vector spaces in the CAS ({vectors_available}), "
                                   "choose one with the parameter vectors")
            return "INVALID", "The WordEmbeddings in the CAS have no shortName"
        name = vectors.strip()
        if not _VECTOR_GROUP_FORMAT.match(name):
            return "INVALID", f"Invalid vectors \"{name}\": expected an uppercase name with an optional lowercase variant"
        if name not in (vectors_available or "").split(","):
            logger.warning("vectors=%s, but the WordEmbeddings in the CAS are %s", name, vectors_available)
        return name, None
    # SpacyToken.vector: SPACY plus the size of the model that produced them (de_core_news_lg -> SPACYlg); plain SPACY if the
    # size is unknown
    name = (vectors or "SPACY").strip()
    if not _VECTOR_GROUP_FORMAT.match(name):
        return "INVALID", f"Invalid vectors \"{name}\": expected an uppercase name with an optional lowercase variant"
    if not name.startswith("SPACY") or name[len("SPACY"):] not in ("",) + _SPACY_MODEL_SIZES:
        return name, f"Vectors \"{name}\" are not supported yet, only SPACY"
    parameter_size = name[len("SPACY"):]
    model_size = (vectors_model or "").strip().rsplit("_", 1)[-1]
    if model_size not in _SPACY_MODEL_SIZES:
        if vectors_model:
            logger.warning("Unknown spaCy model size in \"%s\"", vectors_model)
        elif not parameter_size:
            logger.warning("No spaCy model found for the token vectors, using vectors label SPACY")
        return name, None
    if parameter_size and parameter_size != model_size:
        logger.warning("vectors=%s, but the tokens come from %s; using SPACY%s", name, vectors_model, model_size)
    return "SPACY" + model_size, None


def _ttlab_labels(lang: str, vectors_group: str) -> Dict[str, str]:
    # label_v3 -> label_ttlab for one request
    parts = dict(_GROUPS[lang], vectors=vectors_group)
    return {label_v3: template.format(**parts) for _, _, label_v3, template, *_ in INDEX_CATALOG}


@lru_cache(maxsize=1)
def _known_ttlab_labels() -> frozenset:
    # every label_ttlab a request can produce with spaCy vectors, for the unknown-name warning
    return frozenset(label for lang in _GROUPS for size in ("",) + _SPACY_MODEL_SIZES
                     for label in _ttlab_labels(lang, "SPACY" + size).values())


def _parse_index_names(value: Optional[str]) -> List[str]:
    return [name.strip() for name in (value or "").split(",") if name.strip()]


def _resolve_index_selection(include: Optional[str], exclude: Optional[str], ttlab_labels: Dict[str, str]) -> Set[str]:
    # The selected indices as a set of label_v3 (README, "Selecting indices"); as in duui-coh-metrix 0.2.0. Names are, from least
    # to most specific: "*", a group (type_name), a prefix of label_ttlab ending in "*" (a longer prefix is more specific), a
    # label_v3, a label_ttlab; case-insensitive. For each index the most specific matching name decides; exclude wins on a tie.
    # An empty include means "*". The core computes every index (the Text Easability components need most of them as inputs);
    # the selection decides what is output.
    groups = {type_name.lower() for _, type_name, *_ in INDEX_CATALOG}
    public = {label_v3.lower() for _, _, label_v3, *_ in INDEX_CATALOG}
    known_ttlab = {label.lower() for label in _known_ttlab_labels() | set(ttlab_labels.values())}

    def check(names: List[str]) -> List[str]:
        valid = []
        for name in names:
            key = name.lower()
            if key == "*" or key in groups or key in public or key in known_ttlab or (
                    key.endswith("*") and any(label.startswith(key[:-1]) for label in known_ttlab)):
                valid.append(key)
                continue
            stem = key.split("_")[0]
            candidates = sorted(label_v3 for _, _, label_v3, *_ in INDEX_CATALOG
                                if label_v3.lower() == stem or label_v3.lower().startswith(stem + "_"))
            hint = f" (use {', '.join(candidates)})" if candidates else ""
            logger.warning("Ignoring unknown index name \"%s\"%s", name, hint)
        return valid

    def rank(names: List[str], type_name: str, label_v3: str, label_ttlab: str) -> Optional[Tuple[int, int]]:
        best = None
        for key in names:
            if key == "*":
                match = (0, 0)
            elif key == label_ttlab.lower():
                match = (4, 0)
            elif key == label_v3.lower():
                match = (3, 0)
            elif key.endswith("*") and label_ttlab.lower().startswith(key[:-1]):
                match = (2, len(key) - 1)
            elif key == type_name.lower():
                match = (1, 0)
            else:
                continue
            best = match if best is None else max(best, match)
        return best

    include_names = check(_parse_index_names(include)) if _parse_index_names(include) else ["*"]
    exclude_names = check(_parse_index_names(exclude))
    output = set()
    for _, type_name, label_v3, *_ in INDEX_CATALOG:
        included = rank(include_names, type_name, label_v3, ttlab_labels[label_v3])
        excluded = rank(exclude_names, type_name, label_v3, ttlab_labels[label_v3])
        if included is not None and (excluded is None or included > excluded):
            output.add(label_v3)
    logger.debug("Selected indices: %d", len(output))
    return output


# ------------------------------------------------------------------ request and response
class Token(BaseModel):
    # as the core's Token; is_alpha and is_punct are derived here from the word (the communication layer sends the word: the
    # covered text, or the form of a split contraction)
    begin: int
    end: int
    text: str
    pos_value: str = ""   # language-specific tag (STTS, Penn Treebank)
    pos_coarse: str = ""  # UPOS
    lemma: str = ""
    is_alpha: Optional[bool] = None
    is_punct: Optional[bool] = None
    dep_type: str = ""
    head_index: Optional[int] = None
    morph_person: Optional[str] = ""
    morph_number: Optional[str] = ""
    morph_tense: Optional[str] = ""
    morph_verbform: Optional[str] = ""
    morph_number_psor: Optional[str] = ""
    morph_mood: Optional[str] = ""
    vector: Optional[List[float]] = None
    has_vector: bool = False


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
    # comma-separated index names (README, "Selecting indices")
    include_indices: Optional[str] = None
    exclude_indices: Optional[str] = None
    # source of the token vectors (README, "Vectors"): WordEmbedding (vectors = their shortName, vectors_available = all
    # shortNames in the CAS, comma-separated) or SpacyToken (vectors = the parameter, default SPACY, and the spaCy model of the
    # tokens from the CAS)
    vectors_source: Optional[str] = None
    vectors: Optional[str] = None
    vectors_available: Optional[str] = None
    vectors_model: Optional[str] = None


class AnnotationMeta(BaseModel):
    name: str
    version: str
    modelName: str
    modelVersion: str


class DocumentModification(BaseModel):
    user: str
    timestamp: int
    comment: str


# Index labels (README, "Index labels"; docs/label-scheme.md): label_v3 the public label (Coh-Metrix 3.0, or <NAME>_ext), unique,
# the UIMA type; label_ttlab the internal label with the method and resource groups; label_v2 the Coh-Metrix 2.x label ("n/a"
# where there is none, None for project-only indices); version of the computation.
class Index(BaseModel):
    index: int
    type_name: str
    label_v3: str
    label_ttlab: str
    label_v2: Optional[str] = None
    description: str
    value: Optional[float]
    error: Optional[str]
    version: Optional[str] = None


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


def _is_punct(text: str) -> bool:
    # spaCy's lexical IS_PUNCT (spacy/lang/lex_attrs.py), as the reference preprocessing sets it (cohmetrix/preprocess.py)
    return all(unicodedata.category(c).startswith("P") for c in text)


def _core_request(request: TextImagerRequest, lang: str, vectors_group: str) -> "core.TextImagerRequest":
    # the core's request: the same paragraphs, is_alpha / is_punct from the word, and the vectors under the name the core's
    # labels use (the spaCy model for the vectors the Text Easability model was fitted with, else the group)
    group, model = _EASABILITY_VECTORS[lang]
    paragraphs = []
    for p in request.paragraphs:
        sentences = []
        for s in p.sentences:
            tokens = []
            for t in s.tokens:
                d = t.dict()
                d["is_alpha"] = t.text.isalpha() if t.is_alpha is None else t.is_alpha
                d["is_punct"] = _is_punct(t.text) if t.is_punct is None else t.is_punct
                d["has_vector"] = bool(t.has_vector and t.vector)
                if not d["has_vector"]:
                    d["vector"] = None
                tokens.append(d)
            sentences.append({"begin": s.begin, "end": s.end, "text": s.text, "tokens": tokens})
        paragraphs.append({"begin": p.begin, "end": p.end, "text": p.text, "sentences": sentences})
    return core.TextImagerRequest(language=lang, text=request.text, paragraphs=paragraphs, noun_chunks=[],
                                  vectors=model if vectors_group == group else vectors_group)


# ------------------------------------------------------------------ app
typesystem_filename = 'src/main/resources/TypeSystem.xml'
logger.debug("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    typesystem_xml_content = typesystem.to_xml().encode("utf-8")

lua_communication_script_filename = "src/main/lua/communication.lua"
logger.debug("Loading Lua communication script from \"%s\"", lua_communication_script_filename)
with open(lua_communication_script_filename, 'rb') as f:
    lua_communication_script = f.read().decode("utf-8")

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

duui_logging.add_logging(app, default_logger=__name__)


@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return lua_communication_script


@app.get("/v1/documentation")
def get_documentation() -> TextImagerDocumentation:
    return TextImagerDocumentation(
        annotator_name=settings.annotator_name,
        version=settings.annotator_version,
        implementation_lang="Python",
        meta={
            "python_version": python_version(),
            "python_version_full": sys_version,
            "versions": {lang: core.versions_record(lang) for lang in SUPPORTED_LANGS},
        },
        docker_container_id="[TODO]",
        parameters={
            "include_indices": "comma-separated index groups, label_v3 or label_ttlab names, label_ttlab prefixes "
                               "ending in *, or *; default all",
            "exclude_indices": "comma-separated index groups, label_v3 or label_ttlab names, label_ttlab prefixes "
                               "ending in *; default none",
            "vectors": "token vectors for the LSA indices and SMCAUSlsa: the shortName of the WordEmbedding "
                       "annotations (default: the only one in the CAS), e.g. SPACYlg; it becomes part of the label. "
                       "Without WordEmbeddings: the SpacyToken vectors, SPACY (default; the model size is read from "
                       "the CAS)",
        },
        capability=TextImagerCapability(supported_languages=SUPPORTED_LANGS, reproducible=True),
        implementation_specific=None,
    )


@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    return Response(content=typesystem_xml_content, media_type="application/xml")


@app.get("/v1/details/input_output")
def get_input_output() -> TextImagerInputOutput:
    return TextImagerInputOutput(inputs=TEXTIMAGER_ANNOTATOR_INPUT_TYPES, outputs=TEXTIMAGER_ANNOTATOR_OUTPUT_TYPES)


@app.get("/v1/indices")
def get_indices() -> List[Dict[str, Any]]:
    # The names accepted by include_indices / exclude_indices, and how each index relates to Coh-Metrix 3.0. The vector group
    # depends on the document (SPACYlg, ...), so label_ttlab shows it as *.
    labels = {lang: _ttlab_labels(lang, "*") for lang in _GROUPS}
    return [
        {
            "index": index,
            "type_name": type_name,
            "label_v3": label_v3,
            "label_v2": label_v2,
            "label_ttlab_en": labels["en"][label_v3],
            "label_ttlab_de": labels["de"][label_v3],
            "comparable_en": comparable_en,
            "comparable_de": comparable_de,
            "version_en": _index_version(label_v3, "en"),
            "version_de": _index_version(label_v3, "de"),
        }
        for index, type_name, label_v3, _, comparable_en, comparable_de, label_v2 in INDEX_CATALOG
    ]


@app.post("/v1/process")
def post_process(request: TextImagerRequest) -> TextImagerResponse:
    modification_timestamp_seconds = int(time())

    # German and English only, read in UD: other languages and pre-UD input (spaCy's TIGER / ClearNLP labels) are refused, with
    # the reason (the core's _language and _check_labels)
    try:
        lang = core._language(request)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    vectors_group, vectors_error = _vectors_group(request.vectors, request.vectors_model, request.vectors_source,
                                                  request.vectors_available)
    ttlab_labels = _ttlab_labels(lang, vectors_group)
    output = _resolve_index_selection(request.include_indices, request.exclude_indices, ttlab_labels)
    core_request = _core_request(request, lang, vectors_group)

    with _CORE_LOCK:
        try:
            core_response = core.post_process(core_request)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
    # the core logs an error outside the indices and returns the indices computed so far, without meta; the indices after the
    # failure would be missing without an error (TTLab's review of PR #305), so the request fails, as in the core's CLI
    if core_response.meta is None:
        raise HTTPException(status_code=500, detail="Coh-Metrix stopped before computing every index (see the container log)")

    # unsupported vectors, or no token has a vector (e.g. a spaCy trf model): the vector indices are undefined, not 0
    if vectors_error is None and not output.isdisjoint(_VECTOR_INDICES) and not any(
            t.has_vector and t.vector for p in request.paragraphs for s in p.sentences for t in s.tokens):
        vectors_error = f"No token vectors ({vectors_group})"
        logger.warning("%s: the LSA indices and SMCAUSlsa are undefined", vectors_error)

    indices = []
    for i in core_response.indices:
        index, type_name, label_v3, _, _, _, label_v2 = _CATALOG_BY_INDEX[i.index]
        if label_v3 not in output:
            continue
        value, error = i.value, i.error
        if vectors_error is not None and label_v3 in _VECTOR_INDICES:
            value, error = None, vectors_error
        indices.append(Index(index=index, type_name=type_name, label_v3=label_v3, label_ttlab=ttlab_labels[label_v3],
                             label_v2=label_v2, description=i.description, version=_index_version(label_v3, lang), value=value,
                             error=error))

    meta = AnnotationMeta(name=settings.annotator_name, version=settings.annotator_version,
                          modelName=settings.annotator_name, modelVersion=settings.annotator_version)
    modification_meta = DocumentModification(user=settings.annotator_name, timestamp=modification_timestamp_seconds,
                                             comment=f"{settings.annotator_name} ({settings.annotator_version})")
    logger.info("Processed in %d seconds", int(time()) - modification_timestamp_seconds)
    return TextImagerResponse(indices=indices, meta=meta, modification_meta=modification_meta)
