"""DUUI layer of Coh-Metrix (duui-coh-metrix).

The indices are computed by `cohmetrix_core.py` (the computation of Coh-Metrix 0.4.0-py, German and English read in Universal
Dependencies, PR #305), which only computes: `compute(request, wanted)` gives {key: (value, error)} for the wanted index keys
and runs only the steps they need. This file owns everything around it:
- the index catalog with the two-layer labels (`label_v3` public, `label_ttlab` internal with uppercase resource groups; README,
  "Index labels"), the descriptions, and which indices are emitted;
- `include_indices` / `exclude_indices` (README, "Selecting indices"), mapped to the core keys the selection needs;
- the Text Easability components from the computed inputs (the core's model functions);
- every selected index gets a value or an error;
- the token vectors from WordEmbedding annotations or SpacyToken.vector (README, "Vectors");
- the routes of a DUUI component, and `/v1/indices`.
"""
import logging
from pathlib import Path
import re
import unicodedata
from functools import lru_cache
from platform import python_version
from sys import version as sys_version
from time import time
from collections import defaultdict
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

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
    (59, "Situation Model", "SMCAUSv", "SMCAUSv_{verbs}", "original", "counterpart", "CAUSV"),
    (60, "Situation Model", "SMCAUSvp", "SMCAUSvp_{verbs}_{connectives}", "counterpart", "counterpart", "CAUSVP"),
    (61, "Situation Model", "SMINTEp", "SMINTEp_{verbs}", "original", "counterpart", "INTEi"),
    (62, "Situation Model", "SMCAUSr", "SMCAUSr_{verbs}_{connectives}", "counterpart", "counterpart", "CAUSC"),
    (63, "Situation Model", "SMINTEr", "SMINTEr_{verbs}_LLMlist", "original", "counterpart", "INTEC"),
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
    (1001, "Readability", "RDFOG_ext", "RDFOG_ext_TEXTSTATlist", "ext", "ext", None),
    (1002, "Readability", "RDSMOG_ext", "RDSMOG_ext", "ext", "ext", None),
    (1003, "Readability", "RDARI_ext", "RDARI_ext", "ext", "ext", None),
    (1004, "Readability", "RDCLI_ext", "RDCLI_ext", "ext", "ext", None),
    (1005, "Readability", "RDLW_ext", "RDLW_ext", "ext", "ext", None),
    (1006, "Readability", "RDDCRS_ext", "RDDCRS_ext_TEXTSTATlist", "ext", "ext", None),
    (1007, "Readability", "RDSPACHE_ext", "RDSPACHE_ext_TEXTSTATlist", "ext", "ext", None),
    (1008, "Readability", "RDWSTF1_ext", "RDWSTF1_ext", "ext", "ext", None),
    (1009, "Readability", "RDWSTF2_ext", "RDWSTF2_ext", "ext", "ext", None),
    (1010, "Readability", "RDWSTF3_ext", "RDWSTF3_ext", "ext", "ext", None),
    (1011, "Readability", "RDWSTF4_ext", "RDWSTF4_ext", "ext", "ext", None),
    (106, "Readability", "RDL2", "RDL2_SUBTLEX", "corpus", "corpus", "L2"),
]

# The public labels and the groups must stay distinct when compared case-insensitively (index selection).
assert len({label_v3.lower() for _, _, label_v3, *_ in INDEX_CATALOG}
           | {type_name.lower() for _, type_name, *_ in INDEX_CATALOG}) \
       == len(INDEX_CATALOG) + len({type_name for _, type_name, *_ in INDEX_CATALOG})

# The resource groups of each language. Every resource is named, Coh-Metrix's own included (WN, MRC). What depends on the
# container (GermaNet mounted or not) is filled in by _groups: the German verb inventories are GermaNet or, without it, the
# seed lists (README rule 7), and SMTEMP uses GermaNet for the German progressive only if it is there.
_GROUPS = {
    "de": {"wn": "GN", "connectives": "DIMLEX", "aoa": "BIRCHENOUGH", "mrc": "MRCtr", "concreteness": "KOEPER",
           "imagery": "KOEPER", "easability": "REFIT_REFv1", "clear": "CLEARtr"},
    "en": {"wn": "WN", "verbs": "WN", "connectives": "DIMLEXeng_GUM", "temporal": "LLMlist", "aoa": "KUPERMAN", "mrc": "MRC",
           "concreteness": "BRYSBAERT", "imagery": "MRC", "easability": "REFIT_REFa", "clear": "CLEAR"},
}


def _groups(lang: str, verbs: Optional[str] = None, germanet: Optional[bool] = None) -> Dict[str, str]:
    parts = dict(_GROUPS[lang])
    if lang == "de":
        parts["verbs"] = verbs or core.GERMAN_VERB_INVENTORY
        has_germanet = core.germanet is not None if germanet is None else germanet
        parts["temporal"] = "LLMlist_GN" if has_germanet else "LLMlist"
    return parts


# ------------------------------------------------------------------ the core's results as index records
# The core key of each catalog index (the core computes by these keys: "DESPC", ..., "RDFOG"; PC* are computed here), and the
# description of each index (from the core's records of 0.4.0-py; the frequency indices name their corpus per language).
CORE_KEY_BY_INDEX: Dict[int, str] = {"1": "DESPC", "2": "DESSC", "3": "DESWC", "4": "DESPL", "5": "DESPLd", "6": "DESSL", "7": "DESSLd", "8": "DESWLsy", "9": "DESWLsyd", "10": "DESWLlt", "11": "DESWLltd", "28": "CRFNO1", "29": "CRFAO1", "30": "CRFSO1", "31": "CRFNOa", "32": "CRFAOa", "33": "CRFSOa", "34": "CRFCWO1", "35": "CRFCWO1d", "36": "CRFCWOa", "37": "CRFCWOad", "38": "LSASS1", "39": "LSASS1d", "40": "LSASSp", "41": "LSASSpd", "42": "LSAPP1", "43": "LSAPP1d", "44": "LSAGN", "45": "LSAGNd", "46": "LDTTRc", "47": "LDTTRa", "48": "LDMTLDa", "49": "LDVOCDa", "50": "CNCAll", "51": "CNCCaus", "52": "CNCLogic", "53": "CNCADC", "54": "CNCTemp", "55": "CNCTempx", "56": "CNCAdd", "57": "CNCPos", "58": "CNCNeg", "59": "SMCAUSv", "60": "SMCAUSvp", "61": "SMINTEp", "62": "SMCAUSr", "63": "SMINTEr", "64": "SMCAUSlsa", "65": "SMCAUSwn", "66": "SMTEMP", "67": "SYNLE", "68": "SYNNP", "69": "SYNMEDpos", "70": "SYNMEDwrd", "71": "SYNMEDlem", "72": "SYNSTRUTa", "73": "SYNSTRUTt", "74": "DRNP", "75": "DRVP", "76": "DRAP", "77": "DRPP", "78": "DRPVAL", "79": "DRNEG", "80": "DRGERUND", "81": "DRINF", "82": "WRDNOUN", "83": "WRDVERB", "84": "WRDADJ", "85": "WRDADV", "86": "WRDPRO", "87": "WRDPRP1s", "88": "WRDPRP1p", "89": "WRDPRP2", "90": "WRDPRP3s", "91": "WRDPRP3p", "92": "WRDFRQc", "93": "WRDFRQa", "94": "WRDFRQmc", "95": "WRDAOAc", "96": "WRDFAMc", "97": "WRDCNCc", "98": "WRDIMGc", "99": "WRDMEAc", "100": "WRDPOLc", "101": "WRDHYPn", "102": "WRDHYPv", "103": "WRDHYPnv", "104": "RDFRE", "105": "RDFKGL", "1001": "RDFOG", "1002": "RDSMOG", "1003": "RDARI", "1004": "RDCLI", "1005": "RDLW", "1006": "RDDCRS", "1007": "RDSPACHE", "1008": "RDWSTF1", "1009": "RDWSTF2", "1010": "RDWSTF3", "1011": "RDWSTF4", "106": "RDL2"}
_EASABILITY_NAMES = [("PCNAR", "Narrativity"), ("PCSYN", "Syntactic simplicity"), ("PCCNC", "Word concreteness"),
                     ("PCREF", "Referential cohesion"), ("PCDC", "Deep cohesion"), ("PCVERB", "Semantic overlap"),
                     ("PCCONN", "Connectivity"), ("PCTEMP", "Third-person narration")]
_EASABILITY_INDICES = [(f"{code}{kind}", f"Text Easability PC {name}, {'z score' if kind == 'z' else 'percentile'}")
                       for code, name in _EASABILITY_NAMES for kind in "zp"]
CORE_KEY_BY_INDEX = {int(index): key for index, key in CORE_KEY_BY_INDEX.items()}
CORE_KEY_BY_INDEX.update({number: key for number, (key, _) in enumerate(_EASABILITY_INDICES, start=12)})
EASABILITY_KEYS = frozenset(key for key, _ in _EASABILITY_INDICES)
DESCRIPTIONS: Dict[str, Any] = {
    "DESPC": "Paragraph count, number of paragraphs",
    "DESSC": "Sentence count, number of sentences",
    "DESWC": "Word count, number of words",
    "DESPL": "Paragraph length, number of sentences, mean",
    "DESPLd": "Paragraph length, number of sentences, standard deviation",
    "DESSL": "Sentence length, number of words, mean",
    "DESSLd": "Sentence length, number of words, standard deviation",
    "DESWLsy": "Word length, number of syllables, mean",
    "DESWLsyd": "Word length, number of syllables, standard deviation",
    "DESWLlt": "Word length, number of letters, mean",
    "DESWLltd": "Word length, number of letters, standard deviation",
    "CRFNO1": "Noun overlap, adjacent sentences, binary, mean",
    "CRFAO1": "Argument overlap, adjacent sentences, binary, mean",
    "CRFSO1": "Stem overlap, adjacent sentences, binary, mean",
    "CRFNOa": "Noun overlap, all sentences, binary, mean",
    "CRFAOa": "Argument overlap, all sentences, binary, mean",
    "CRFSOa": "Stem overlap, all sentences, binary, mean",
    "CRFCWO1": "Content word overlap, adjacent sentences, proportional, mean",
    "CRFCWO1d": "Content word overlap, adjacent sentences, proportional, standard deviation",
    "CRFCWOa": "Content word overlap, all sentences, proportional, mean",
    "CRFCWOad": "Content word overlap, all sentences, proportional, standard deviation",
    "LSASS1": "LSA overlap, adjacent sentences, mean",
    "LSASS1d": "LSA overlap, adjacent sentences, standard deviation",
    "LSASSp": "LSA overlap, all sentences in paragraph, mean",
    "LSASSpd": "LSA overlap, all sentences in paragraph, standard deviation",
    "LSAPP1": "LSA overlap, adjacent paragraphs, mean",
    "LSAPP1d": "LSA overlap, adjacent paragraphs, standard deviation",
    "LSAGN": "LSA given/new, sentences, mean (the first sentence counts as 0, so the value depends on the number of sentences)",
    "LSAGNd": "LSA given/new, sentences, standard deviation",
    "LDTTRc": "Lexical diversity, type-token ratio, content word lemmas",
    "LDTTRa": "Lexical diversity, type-token ratio, all words",
    "LDMTLDa": "Lexical diversity, MTLD, all words",
    "LDVOCDa": "Lexical diversity, VOCD, all words",
    "CNCAll": "All connectives incidence",
    "CNCCaus": "Causal connectives incidence",
    "CNCLogic": "Logical connectives incidence",
    "CNCADC": "Adversative and contrastive connectives incidence",
    "CNCTemp": "Temporal connectives incidence",
    "CNCTempx": "Expanded temporal connectives incidence",
    "CNCAdd": "Additive connectives incidence",
    "CNCPos": "Positive connectives incidence",
    "CNCNeg": "Negative connectives incidence",
    "SMCAUSv": "Causal verb incidence",
    "SMCAUSvp": "Causal verbs and causal particles incidence",
    "SMINTEp": "Intentional verbs incidence",
    "SMCAUSr": "Ratio of causal particles to causal verbs",
    "SMINTEr": "Ratio of intentional particles to intentional verbs",
    "SMCAUSlsa": "LSA verb overlap",
    "SMCAUSwn": "WordNet verb overlap",
    "SMTEMP": "Temporal cohesion, tense and aspect repetition, mean",
    "SYNLE": "Left embeddedness, words before main verb, mean",
    "SYNNP": "Number of modifiers per noun phrase, mean",
    "SYNMEDpos": "Minimal Edit Distance, part of speech",
    "SYNMEDwrd": "Minimal Edit Distance, all words",
    "SYNMEDlem": "Minimal Edit Distance, lemmas",
    "SYNSTRUTa": "Sentence syntax similarity, adjacent sentences, mean",
    "SYNSTRUTt": "Sentence syntax similarity, all combinations, across paragraphs, mean",
    "DRNP": "Noun phrase density, incidence",
    "DRVP": "Verb phrase density, incidence",
    "DRAP": "Adverbial phrase density, incidence",
    "DRPP": "Preposition phrase density, incidence",
    "DRPVAL": "Agentless passive voice density, incidence",
    "DRNEG": "Negation density, incidence",
    "DRGERUND": "Gerund density (-ing verb forms), incidence",
    "DRINF": "Infinitive density, incidence",
    "WRDNOUN": "Noun incidence",
    "WRDVERB": "Verb incidence",
    "WRDADJ": "Adjective incidence",
    "WRDADV": "Adverb incidence",
    "WRDPRO": "Pronoun incidence",
    "WRDPRP1s": "First-person singular pronoun incidence",
    "WRDPRP1p": "First-person plural pronoun incidence",
    "WRDPRP2": "Second-person pronoun incidence",
    "WRDPRP3s": "Third-person singular pronoun incidence",
    "WRDPRP3p": "Third-person plural pronoun incidence",
    "WRDFRQc": {
        "de": "SUBTLEX-DE Log frequency for content words, mean",
        "en": "SUBTLEX-US Log frequency for content words, mean"
    },
    "WRDFRQa": {
        "de": "SUBTLEX-DE Log frequency for all words, mean",
        "en": "SUBTLEX-US Log frequency for all words, mean"
    },
    "WRDFRQmc": {
        "de": "SUBTLEX-DE Log minimum frequency for content words, mean",
        "en": "SUBTLEX-US Log minimum frequency for content words, mean"
    },
    "WRDAOAc": "Age of acquisition for content words, mean",
    "WRDFAMc": "Familiarity for content words, mean",
    "WRDCNCc": "Concreteness for content words, mean",
    "WRDIMGc": "Imagability for content words, mean",
    "WRDMEAc": "Meaningfulness, Colorado norms, content words, mean",
    "WRDPOLc": "Polysemy for content words, mean",
    "WRDHYPn": "Hypernymy for nouns, mean",
    "WRDHYPv": "Hypernymy for verbs, mean",
    "WRDHYPnv": "Hypernymy for nouns and verbs, mean",
    "RDFRE": "Flesch Reading Ease",
    "RDFKGL": "Flesch–Kincaid Grade Level",
    "RDFOG": "Gunning Fog Index",
    "RDSMOG": "SMOG Grade",
    "RDARI": "Automated Readability Index",
    "RDCLI": "Coleman–Liau Index",
    "RDLW": "Linsear Write Formula",
    "RDDCRS": "Dale-Chall Readability Score",
    "RDSPACHE": "Spache Readability Formula",
    "RDWSTF1": "Wiener Sachtextformel 1",
    "RDWSTF2": "Wiener Sachtextformel 2",
    "RDWSTF3": "Wiener Sachtextformel 3",
    "RDWSTF4": "Wiener Sachtextformel 4",
    "RDL2": "Coh-Metrix L2 Readability"
}
DESCRIPTIONS.update({key: description for key, description in _EASABILITY_INDICES})

# never emitted, because always None (README rule 9): German has no gerund, the Wiener Sachtextformel is a German formula
NOT_EMITTED = {"de": {"DRGERUND"}, "en": {"RDWSTF1", "RDWSTF2", "RDWSTF3", "RDWSTF4"}}

# The names under which the Text Easability models know their inputs: the core key plus the resource of the language (as
# fitted). A model gets only inputs computed with its resources, otherwise its components are None.
_LSA_KEYS = ["LSASS1", "LSASS1d", "LSASSp", "LSASSpd", "LSAPP1", "LSAPP1d", "LSAGN", "LSAGNd", "SMCAUSlsa"]
_FREQUENCY_KEYS = ["WRDFRQc", "WRDFRQa", "WRDFRQmc", "RDL2"]
_CONNECTIVE_KEYS = ["CNCAll", "CNCCaus", "CNCLogic", "CNCADC", "CNCTemp", "CNCTempx", "CNCAdd", "CNCPos", "CNCNeg"]
_PROJECT_ONLY_KEYS = ["RDFOG", "RDSMOG", "RDARI", "RDCLI", "RDLW", "RDDCRS", "RDSPACHE", "RDWSTF1", "RDWSTF2", "RDWSTF3", "RDWSTF4"]
_GERMANET_KEYS = ["WRDPOLc", "WRDHYPn", "WRDHYPv", "WRDHYPnv", "SMCAUSwn", "SMCAUSv", "SMINTEp", "SMINTEr"]
_MODEL_INPUT_SUFFIX = {
    "de": {**dict.fromkeys(_LSA_KEYS, "spacy"), **dict.fromkeys(_FREQUENCY_KEYS, "subtlexde"),
           **dict.fromkeys(_CONNECTIVE_KEYS, "dimlex"), "SMTEMP": "llmlist_germanet", "DRNEG": "llmlist",
           **dict.fromkeys(_GERMANET_KEYS, "germanet"), "SMCAUSvp": "germanet_dimlex", "SMCAUSr": "germanet_dimlex",
           "WRDAOAc": "birchenough", "WRDCNCc": "koeper", "WRDIMGc": "koeper", "WRDFAMc": "mrctranslate",
           "WRDMEAc": "mrctranslate", **dict.fromkeys(_PROJECT_ONLY_KEYS, "ext")},
    "en": {**dict.fromkeys(_LSA_KEYS, "spacy"), **dict.fromkeys(_FREQUENCY_KEYS, "subtlexus"), "SMTEMP": "llmlist",
           "DRNEG": "llmlist", "WRDCNCc": "brysbaert", "WRDAOAc": "kuperman", **dict.fromkeys(_CONNECTIVE_KEYS, "dimlexeng_gum"),
           "SMCAUSvp": "dimlexeng_gum", "SMCAUSr": "dimlexeng_gum", **dict.fromkeys(_PROJECT_ONLY_KEYS, "ext")},
}


def _model_input_label(key: str, lang: str, vectors: Optional[str]) -> str:
    suffix = _MODEL_INPUT_SUFFIX.get(lang, {}).get(key)
    if suffix == "spacy" and vectors:
        suffix = f"spacy-{vectors}"
    return f"{key}_{suffix}" if suffix else key


def _model_inputs(lang: str) -> Set[str]:
    """the core keys the Text Easability model of the language needs"""
    model = core.easability_models.get(lang)
    if not model:
        return set()
    return {label.split("_", 1)[0] for m in model["inputs"] for label in m["labels_internal"]}


def _easability(results: Dict[str, Tuple[Optional[float], Optional[str]]], lang: str,
                vectors: Optional[str]) -> Dict[str, Tuple[Optional[float], Optional[str]]]:
    """the 16 Text Easability components from the computed inputs (the core's model functions)"""
    model = core.easability_models.get(lang)
    if not model:
        return {key: (None, "no Text Easability model") for key in EASABILITY_KEYS}
    values = {_model_input_label(key, lang, vectors): value for key, (value, _) in results.items()}
    scores, error = core._easability_scores(values, model)
    return {key: (scores.get(key), error) for key in EASABILITY_KEYS}

# The token vectors (README, "Vectors"): their name is a label group, e.g. SPACYlg.
# ------------------------------------------------------------------ records: the indices and their variants
# Variants (decision 2026-11-02): further records of an index computed with another resource or method, each with its own
# label_ttlab (README rules 4, 7) and core key "<key>@<variant>": (index, core key, label_ttlab template, comparability EN,
# comparability DE, languages). They follow their index in the output; users choose with include_indices / exclude_indices.
_ALL = ("de", "en")
VARIANTS: List[Tuple[int, str, str, str, str, Tuple[str, ...]]] = [
    # the Wikipedia samples shipped with the component (0.2.0), log10 as the main frequencies
    *[(i, f"{k}@WIKI", f"{k}_WIKI", "corpus", "corpus", _ALL) for i, k in ((92, "WRDFRQc"), (93, "WRDFRQa"), (94, "WRDFRQmc"))],
    (106, "RDL2@WIKI", "RDL2_WIKI", "corpus", "corpus", _ALL),
    # the LLM-generated connective lists and the 0.2.0 counting
    *[(i, f"{k}@LLMlist", f"{k}_LLMlist", "llmlist", "llmlist", _ALL) for i, k in (
        (50, "CNCAll"), (51, "CNCCaus"), (52, "CNCLogic"), (53, "CNCADC"), (54, "CNCTemp"), (55, "CNCTempx"), (56, "CNCAdd"),
        (57, "CNCPos"), (58, "CNCNeg"))],
    (60, "SMCAUSvp@LLMlist", "SMCAUSvp_{verbs}_LLMlist", "llmlist", "counterpart, llmlist", _ALL),
    (62, "SMCAUSr@LLMlist", "SMCAUSr_{verbs}_LLMlist", "llmlist", "counterpart, llmlist", _ALL),
    # the MRC database (German: translated) where the main record uses other norms
    (95, "WRDAOAc@MRC", "WRDAOAc_{mrc}", "original", "translated", _ALL),
    (97, "WRDCNCc@MRC", "WRDCNCc_{mrc}", "original", "translated", _ALL),
    (98, "WRDIMGc@MRC", "WRDIMGc_{mrc}", "original", "translated", ("de",)),   # English WRDIMGc is MRC already
    # the readability formulas of the textstat library on the text (0.2.0)
    (104, "RDFRE@TEXTSTAT", "RDFRE_TEXTSTAT", "original", "counterpart", _ALL),
    (105, "RDFKGL@TEXTSTAT", "RDFKGL_TEXTSTAT", "original", "original", _ALL),
    *[(i, f"{k}@TEXTSTAT", f"{k}_ext_TEXTSTAT", "ext", "ext", _ALL) for i, k in (
        (1001, "RDFOG"), (1002, "RDSMOG"), (1003, "RDARI"), (1004, "RDCLI"), (1005, "RDLW"), (1006, "RDDCRS"),
        (1007, "RDSPACHE"), (1008, "RDWSTF1"), (1009, "RDWSTF2"), (1010, "RDWSTF3"), (1011, "RDWSTF4"))],
    # the Text Easability models fitted on CLEAR (English) / translated CLEAR (German) with 0.2.0's values: None until they are
    # refitted on the current values (decision 2026-11-02)
    *[(number, f"{key}@CLEAR", f"{key}_REFIT_{{clear}}", "refit", "refit", _ALL)
      for number, (key, _) in enumerate(_EASABILITY_INDICES, start=12)],
]
CLEAR_PENDING = ("the Text Easability model fitted on CLEAR (translated CLEAR for German) has duui-coh-metrix 0.2.0's index "
                 "values as inputs; it is not used until it is refitted on the current ones")
# The main Text Easability models (REFIT_REFv1 / REFIT_REFa: refv1-ud-rr-cr3, refa-en-rr-gn-cr3, PR #306) were fitted on the values
# of Coh-Metrix 0.5.0-py. These inputs are computed differently here (decisions 2026-11-02 and 2026-11-04; measured on 18 texts,
# 2026-11-04): SYNNP (the words on the head of the noun phrase, not every word but the head), LSAGN (the first sentence counts
# 0), German SMTEMP ("würde" + infinitive future, Konjunktiv II the tense of its form). The models are not used while an input
# is listed here: empty the list of a language once its model is refitted on this component's values.
MODEL_INPUTS_CHANGED = {"de": ("LSAGN", "SMTEMP", "SYNNP"), "en": ("LSAGN", "SYNNP")}
REFIT_PENDING = ("not computed: the Text Easability model {model} was fitted on inputs computed differently ({inputs}); it is not "
                 "used until it is refitted on this component's values")

# Best effort without UD (README, "Best effort without UD"): spaCy's TIGER (de) / ClearNLP (en) relations are mapped to UD
# (ud_mapping); the records that read the relations name the mapping as a method group after the public label, e.g.
# SYNNP_TIGERmap, SMCAUSr_CLEARNLPmap_WN_DIMLEXeng_GUM (README rule 7). The keys (2026-11-03): the core keys whose code reads
# dep_type, head_index or the noun chunks (static, through the called functions and the blocks a block needs), together with
# those whose values changed on UD input when the relations were perturbed (every label "dep", or every head the root; six
# texts, de and en): the static analysis misses SYNSTRUTa and RDL2 (read through the dependency trees of SYNSTRUTa).
MAPPED_GROUP = {"TIGER": "TIGERmap", "ClearNLP": "CLEARNLPmap"}
DEPENDENCY_KEYS = frozenset({
    "CNCAll", "CNCCaus", "CNCLogic", "CNCADC", "CNCTemp", "CNCTempx", "CNCAdd", "CNCPos", "CNCNeg",
    "DRNP", "DRVP", "DRAP", "DRPP", "DRPVAL", "DRNEG", "DRGERUND", "DRINF",
    "SMCAUSv", "SMCAUSvp", "SMCAUSr", "SMINTEp", "SMINTEr", "SMTEMP",
    "SYNLE", "SYNNP", "SYNSTRUTa", "SYNSTRUTt", "RDL2", "RDL2@WIKI"})
# the Text Easability models are fitted on UD input; with mapped relations their inputs are not the model's
PC_MAPPED = "not computed: the Text Easability model is fitted on UD input; the dependency relations of this document are mapped from {scheme} (best effort)"


class Record(NamedTuple):
    index: int
    type_name: str
    label_v3: str
    template: str
    comparable_en: str
    comparable_de: str
    label_v2: Optional[str]
    key: str                    # the core key ("WRDFRQc", "WRDFRQc@WIKI")
    languages: Tuple[str, ...]


def _records() -> List[Record]:
    variants = defaultdict(list)
    for index, key, template, comparable_en, comparable_de, languages in VARIANTS:
        variants[index].append((key, template, comparable_en, comparable_de, languages))
    out = []
    for index, type_name, label_v3, template, comparable_en, comparable_de, label_v2 in INDEX_CATALOG:
        out.append(Record(index, type_name, label_v3, template, comparable_en, comparable_de, label_v2,
                          CORE_KEY_BY_INDEX[index], _ALL))
        for key, v_template, v_en, v_de, languages in variants.pop(index, []):
            out.append(Record(index, type_name, label_v3, v_template, v_en, v_de, label_v2, key, languages))
    assert not variants, variants
    return out


RECORDS = _records()
RECORD_BY_KEY = {r.key: r for r in RECORDS}

# version of each record's computation (README, "Index version"): 0.2.0 had "1.0" everywhere. "2.0": the meaning changed
# (another formula, unit or definition); "1.1": the values change, the meaning stays (fixes and edge cases of PR #305, e.g.
# sentences without a word dropped, hyphenated words); "1.0": a new record (a new label_ttlab), or computed as in 0.2.0.
_MAJOR = {"WRDFRQc", "WRDFRQc@WIKI", "RDL2", "RDL2@WIKI", "SYNNP", "DRPVAL", "DRINF", "SMCAUSr", "SMCAUSr@LLMlist",
          "SMINTEr"}
_NEW = ({k for k in CORE_KEY_BY_INDEX.values() if k in _CONNECTIVE_KEYS} | {"SMCAUSvp", "WRDFRQa", "WRDFRQmc", "WRDAOAc",
        "WRDCNCc", "RDFRE", "RDFKGL"} | set(_PROJECT_ONLY_KEYS) | set(EASABILITY_KEYS)
        | {r.key for r in RECORDS if r.key.endswith(("@LLMlist", "@TEXTSTAT", "@CLEAR"))} - {"SMCAUSvp@LLMlist", "SMCAUSr@LLMlist"})
# new in one language only: German WRDIMGc uses Köper & Schulte im Walde's norms, English WRDIMGc stays MRC (looked up by lemma)
_NEW_IN = {"de": {"WRDIMGc"}, "en": set()}
# the version of each record per language (README, "Index version")
VERSIONS = {lang: {r.key: "2.0" if r.key in _MAJOR else "1.0" if r.key in _NEW | _NEW_IN[lang] else "1.1" for r in RECORDS}
            for lang in _ALL}
assert DEPENDENCY_KEYS <= RECORD_BY_KEY.keys(), sorted(DEPENDENCY_KEYS - RECORD_BY_KEY.keys())

# label_ttlab is unique per output (README, "Index labels"): checked for both languages, with and without GermaNet
for _lang in _ALL:
    for _verbs, _germanet in (("GN", True), ("SEEDlist", False)):
        _labels = [r.template.format(**dict(_groups(_lang, _verbs, _germanet), vectors="SPACYlg"))
                   for r in RECORDS if _lang in r.languages]
        assert len(_labels) == len(set(_labels)), sorted(l for l in _labels if _labels.count(l) > 1)


def _description(record: Record, lang: str) -> str:
    base, _, variant = record.key.partition("@")
    if base in EASABILITY_KEYS:
        return _easability_description(base, lang, clear=variant == "CLEAR")
    description = DESCRIPTIONS[base]
    if isinstance(description, dict):
        description = description[lang]
    if variant == "WIKI" and base.startswith("WRDFRQ"):
        return description.replace("SUBTLEX-DE", "Wikipedia").replace("SUBTLEX-US", "Wikipedia")
    suffix = {"WIKI": " (frequencies: Wikipedia sample)", "LLMlist": " (LLM-generated lists)",
              "MRC": " (MRC Psycholinguistic Database" + (", translated)" if lang == "de" else ")"),
              "TEXTSTAT": " (textstat library)"}.get(variant, "")
    return description + suffix


_EASABILITY_BOOK_NAMES = {"PCNAR": "Narrativity", "PCSYN": "Syntactic simplicity", "PCCNC": "Word concreteness",
                          "PCREF": "Referential cohesion", "PCDC": "Deep cohesion", "PCVERB": "Verb cohesion",
                          "PCCONN": "Connectivity", "PCTEMP": "Temporality"}


def _easability_description(key: str, lang: str, clear: bool) -> str:
    """the Coh-Metrix 3.0 name of the component (App. A, ch. 5, Graesser et al. 2011); for the refitted models REFv1 / REFa,
    where a component carries other markers, the model's own name too (decision D2, provisional): the model file's `name`"""
    code, kind = key[:-1], "z score" if key.endswith("z") else "percentile"
    name = _EASABILITY_BOOK_NAMES[code]
    model = None if clear else core.easability_models.get(lang)
    own = next((c["name"] for c in (model or {}).get("components", []) if c.get("label_v3_z") == code + "z"), None)
    if own and own != name:
        name = f"{name} (in this model: {own})"
    return f"Text Easability PC {name}, {kind}"


def _missing_resources(lang: str) -> Dict[str, str]:
    """label group -> the resource of the container that is not there (user-supplied, README "Resources supplied by the
    user"); a record of such a group without a value gets this as its error instead of a bare None"""
    found = lambda path: bool(path) and Path(path).is_file()
    st, missing = core.settings, {}
    if core.germanet is None:
        missing["GN"] = "GermaNet (volume src/main/resources/germanet)"
    if lang == "de":
        checks = (("DIMLEX", core.dimlex_entries is not None, "DiMLex", "DIMLEX_PATH"),
                  ("SUBTLEX", found(st.subtlex_de_path), "SUBTLEX-DE", "SUBTLEX_DE_PATH"),
                  ("KOEPER", found(st.koeper_path), "the Köper & Schulte im Walde norms", "KOEPER_PATH"),
                  ("BIRCHENOUGH", found(st.birchenough_path), "the Birchenough et al. norms", "BIRCHENOUGH_PATH"))
    else:
        checks = (("DIMLEXeng", core.dimlex_en_entries is not None, "DiMLex-Eng", "DIMLEX_EN_PATH"),
                  ("SUBTLEX", found(st.subtlex_us_path), "SUBTLEX-US", "SUBTLEX_US_PATH"),
                  ("BRYSBAERT", found(st.brysbaert_path), "the Brysbaert et al. norms", "BRYSBAERT_PATH"),
                  ("KUPERMAN", found(st.kuperman_path), "the Kuperman et al. norms", "KUPERMAN_PATH"))
    for group, ok, name, env in checks:
        if not ok:
            missing[group] = f"{name} (DUUI_COH_METRIX_{env})"
    return missing


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


def _ttlab_labels(lang: str, vectors_group: str, verbs: Optional[str] = None,
                  germanet: Optional[bool] = None, scheme: str = "UD") -> Dict[str, str]:
    # record key -> label_ttlab for one request (the records of this language); with mapped relations the records that read
    # them get the mapping group after the public label
    parts = dict(_groups(lang, verbs, germanet), vectors=vectors_group)
    labels = {r.key: r.template.format(**parts) for r in RECORDS if lang in r.languages}
    if scheme != "UD":
        for key in DEPENDENCY_KEYS & labels.keys():
            public = RECORD_BY_KEY[key].label_v3
            labels[key] = f"{public}_{MAPPED_GROUP[scheme]}{labels[key][len(public):]}"
    return labels


@lru_cache(maxsize=1)
def _known_ttlab_labels() -> frozenset:
    # every label_ttlab a request can produce with spaCy vectors, with and without GermaNet, for the unknown-name warning
    return frozenset(label for lang in _GROUPS for size in ("",) + _SPACY_MODEL_SIZES
                     for verbs, germanet in (("GN", True), ("SEEDlist", False))
                     for label in _ttlab_labels(lang, "SPACY" + size, verbs, germanet).values())


def _parse_index_names(value: Optional[str]) -> List[str]:
    return [name.strip() for name in (value or "").split(",") if name.strip()]


def _resolve_index_selection(include: Optional[str], exclude: Optional[str], ttlab_labels: Dict[str, str]) -> Set[str]:
    # The selected records as a set of record keys (README, "Selecting indices"); as in duui-coh-metrix 0.2.0. Names are, from least
    # to most specific: "*", a group (type_name), a prefix of label_ttlab ending in "*" (a longer prefix is more specific), a
    # label_v3, a label_ttlab; case-insensitive. For each index the most specific matching name decides; exclude wins on a tie.
    # An empty include means "*". A label_v3 selects every record of the index (its variants too). Only the selected records
    # and what they need are computed.
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
    for r in RECORDS:
        if r.key not in ttlab_labels:
            continue
        included = rank(include_names, r.type_name, r.label_v3, ttlab_labels[r.key])
        excluded = rank(exclude_names, r.type_name, r.label_v3, ttlab_labels[r.key])
        if included is not None and (excluded is None or included > excluded):
            output.add(r.key)
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
    # spaCy's lexical IS_PUNCT (spacy/lang/lex_attrs.py), as the reference preprocessing of Coh-Metrix 0.4.0-py sets it
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
    # The records (indices and their variants): the names accepted by include_indices / exclude_indices, and how each relates
    # to Coh-Metrix 3.0. The vector group depends on the document (SPACYlg, ...), so label_ttlab shows it as *.
    labels = {lang: _ttlab_labels(lang, "*") for lang in _GROUPS}
    return [
        {
            "index": r.index,
            "type_name": r.type_name,
            "label_v3": r.label_v3,
            "label_v2": r.label_v2,
            "label_ttlab_en": labels["en"].get(r.key),
            "label_ttlab_de": labels["de"].get(r.key),
            "comparable_en": r.comparable_en if "en" in r.languages else None,
            "comparable_de": r.comparable_de if "de" in r.languages else None,
            "version_en": VERSIONS["en"][r.key] if "en" in r.languages else None,
            "version_de": VERSIONS["de"][r.key] if "de" in r.languages else None,
        }
        for r in RECORDS
    ]


@app.post("/v1/process")
def post_process(request: TextImagerRequest) -> TextImagerResponse:
    modification_timestamp_seconds = int(time())

    # German and English only, read in UD; spaCy's TIGER / ClearNLP labels are mapped to UD (best effort, README "Best effort
    # without UD"). Other languages and UD version 1 input are refused, with the reason (the core's _language and label_scheme).
    try:
        lang = core._language(request)
        scheme = core.label_scheme(request)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    vectors_group, vectors_error = _vectors_group(request.vectors, request.vectors_model, request.vectors_source,
                                                  request.vectors_available)
    ttlab_labels = _ttlab_labels(lang, vectors_group, scheme=scheme)
    output = {key for key in _resolve_index_selection(request.include_indices, request.exclude_indices, ttlab_labels)
              if key.partition("@")[0] not in NOT_EMITTED.get(lang, ())}
    core_request = _core_request(request, lang, vectors_group)

    # the core keys of the selected records; the Text Easability components (main models) need their model's inputs, the
    # CLEAR ones are not computed, nor the main ones on mapped input or while their model is not refitted (MODEL_INPUTS_CHANGED)
    pending = MODEL_INPUTS_CHANGED.get(lang)
    pc_main = output & EASABILITY_KEYS if scheme == "UD" and not pending else frozenset()
    wanted = {key for key in output if key not in EASABILITY_KEYS and not key.endswith("@CLEAR")}
    if pc_main:
        wanted |= _model_inputs(lang)
    results = core.compute(core_request, wanted)
    if scheme != "UD":
        results.update({key: (None, PC_MAPPED.format(scheme=scheme)) for key in output & EASABILITY_KEYS})
    elif pending:
        model = (core.easability_models.get(lang) or {}).get("model_id", "")
        results.update({key: (None, REFIT_PENDING.format(model=model, inputs=", ".join(pending)))
                        for key in output & EASABILITY_KEYS})
    if pc_main:
        try:
            results.update(_easability(results, lang, core_request.vectors))
        except Exception as e:
            logger.exception(e)
            results.update({key: (None, f"not computed: {e}") for key in EASABILITY_KEYS})
    results.update({key: (None, CLEAR_PENDING) for key in output if key.endswith("@CLEAR")})

    # unsupported vectors, or no token has a vector (e.g. a spaCy trf model): the vector indices are undefined, not 0
    if vectors_error is None and any(RECORD_BY_KEY[k].label_v3 in _VECTOR_INDICES for k in output) and not any(
            t.has_vector and t.vector for p in request.paragraphs for s in p.sentences for t in s.tokens):
        vectors_error = f"No token vectors ({vectors_group})"
        logger.warning("%s: the LSA indices and SMCAUSlsa are undefined", vectors_error)

    missing = _missing_resources(lang)
    indices = []
    for r in RECORDS:
        if r.key not in output:
            continue
        value, error = results.get(r.key, (None, "not computed"))
        if value is None and error is None:
            absent = [missing[g] for g in ttlab_labels[r.key].split("_")[1:] if g in missing]
            if absent:
                error = "resource not available: " + ", ".join(absent)
        if vectors_error is not None and r.label_v3 in _VECTOR_INDICES:
            value, error = None, vectors_error
        indices.append(Index(index=r.index, type_name=r.type_name, label_v3=r.label_v3, label_ttlab=ttlab_labels[r.key],
                             label_v2=r.label_v2, description=_description(r, lang), version=VERSIONS[lang][r.key],
                             value=value, error=error))

    meta = AnnotationMeta(name=settings.annotator_name, version=settings.annotator_version,
                          modelName=settings.annotator_name, modelVersion=settings.annotator_version)
    modification_meta = DocumentModification(user=settings.annotator_name, timestamp=modification_timestamp_seconds,
                                             comment=f"{settings.annotator_name} ({settings.annotator_version})")
    logger.info("Processed in %d seconds", int(time()) - modification_timestamp_seconds)
    return TextImagerResponse(indices=indices, meta=meta, modification_meta=modification_meta)
