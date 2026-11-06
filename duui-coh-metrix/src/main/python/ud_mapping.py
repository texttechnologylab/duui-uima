"""Best effort for input without Universal Dependencies (README, "Best effort without UD"): the core reads UD version 2 relations;
spaCy's German models write TIGER labels, its English models ClearNLP labels. Such input is mapped here to UD, as far as that is
possible without a UD parser, and the indices that read the relations get a label group naming the mapping (duui.py).

- English (ClearNLP): DepEdit (Apache 2.0; Siyao Peng and Amir Zeldes, "All Roads Lead to UD: Converting Stanford and Penn
  Parses to English Universal Dependencies with Multilayer Annotations", LAW-MWE-CxG 2018, https://aclanthology.org/W18-4918/)
  with its published Stanford Dependencies -> UD rules
  (stan2uni.ini, renames and restructuring: prepositions, copulas with a prepositional predicate, conjunctions ...), after our
  rules for the ClearNLP labels Stanford Dependencies does not have (clearnlp_additions.ini: the copula as head of a nominal or
  adjectival predicate, relcl, dative, agent, oprd).
- German (TIGER): a renaming table, partly by the part of speech of the dependent and its head; the tree is kept (TIGER heads:
  the finite auxiliary, the preposition, the copula stay heads).

Only the relation labels and heads change; tokens, POS, lemmas and morphology are spaCy's."""
import os
import threading
from types import SimpleNamespace
from typing import Dict, List, Optional

# the labels that only the pre-UD schemes have (detection; the core's former refusal sets)
TIGER_ONLY_LABELS = {"sb", "oa", "nk", "mo", "oc", "pd", "cj", "cd", "mnr", "ag", "svp", "ng"}
CLEARNLP_ONLY_LABELS = {"dobj", "pobj", "prep", "auxpass", "nsubjpass", "csubjpass", "neg", "attr", "acomp", "npadvmod", "pcomp",
                        "dative", "relcl", "prt", "agent", "oprd", "quantmod"}
UD_V1_ONLY_LABELS = {"dobj", "nsubjpass", "csubjpass", "auxpass", "neg", "name", "mwe", "remnant", "foreign"}

# the scheme of the input by language: "UD", or the pre-UD scheme that gets mapped (its label group in duui.py)
PRE_UD = {"de": ("TIGER", TIGER_ONLY_LABELS), "en": ("ClearNLP", CLEARNLP_ONLY_LABELS)}

_DEPEDIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resources", "depedit")
_ROOT_LABELS = {"", "--", "root"}


def label_scheme(lang: str, labels) -> str:
    """the dependency scheme of the input: "UD", "TIGER" (de) or "ClearNLP" (en); UD version 1 input is refused (ValueError)"""
    labels = {(label or "").strip().lower() for label in labels}
    scheme, only = PRE_UD[lang]
    if labels & only:
        return scheme
    found = labels & UD_V1_ONLY_LABELS
    if found:
        raise ValueError(f"{lang} input with UD version 1 labels ({', '.join(sorted(found))}): the core reads UD version 2 "
                         f"(e.g. duui-stanza) or, as best effort, spaCy's TIGER / ClearNLP labels")
    return "UD"


# --- German: TIGER -> UD, renaming ------------------------------------------------------------------------------------------
# By label; None: by the part of speech (_de_by_pos). Following the UD German treebanks' use of the relations (GSD, HDT).
_TIGER_TO_UD: Dict[str, Optional[str]] = {
    "ac": "case",          # adpositional case marker (second part of a circumposition)
    "adc": "amod",         # adjective component
    "ag": "nmod",          # genitive attribute
    "ams": "obl",          # measure argument of an adjective
    "app": "appos",
    "avc": "advmod",       # adverbial phrase component
    "cc": None,            # comparative complement: obl (nominal) / advcl (clausal)
    "cd": "cc",            # coordinating conjunction
    "cj": "conj",
    "cm": "case",          # comparative conjunction (als, wie)
    "cp": "mark",          # complementizer
    "cvc": "obl",          # collocational verb construction
    "da": "iobj",          # dative object
    "dep": "dep",
    "dm": "discourse",
    "ep": "expl",
    "ju": "cc",            # junctor
    "mnr": "nmod",         # postnominal modifier
    "mo": None,            # modifier: by part of speech
    "ng": "advmod",        # negation
    "nk": None,            # noun kernel element: by part of speech
    "nmc": "flat",         # numerical component
    "oa": "obj",
    "oc": None,            # clausal object / verbal complement: ccomp (finite) / xcomp
    "og": "obj",           # genitive object
    "op": "obl",           # prepositional object
    "par": "parataxis",
    "pd": "xcomp",         # predicate (the copula stays head)
    "pg": "nmod",          # phrasal genitive
    "ph": "expl",          # placeholder
    "pm": "mark",          # morphological particle (zu)
    "pnc": "flat",         # proper noun component
    "punct": "punct",
    "rc": "acl:relcl",
    "re": "dep",           # repeated element
    "rs": "parataxis",     # reported speech
    "sb": "nsubj",
    "sbp": "obl:agent",    # passivised subject (von / durch)
    "svp": "compound:prt",
    "uc": "flat",          # unit component
    "vo": "vocative",
}
_VERBAL = {"VERB", "AUX"}
_NOMINAL = {"NOUN", "PROPN", "PRON", "NUM"}


def _de_by_pos(label: str, token, head) -> str:
    pos = token.pos_coarse or ""
    head_pos = (head.pos_coarse or "") if head is not None else ""
    if label == "nk":
        if pos == "DET":
            return "det"
        if pos == "ADJ":
            return "amod"
        if pos == "NUM":
            return "nummod"
        if pos == "ADV":
            return "advmod"
        if pos in _NOMINAL:
            return "nmod"
        return "dep"
    if label == "mo":
        if pos in ("ADV", "PART", "ADJ"):
            return "advmod"
        if pos == "SCONJ":
            return "mark"
        if pos in _VERBAL:
            return "advcl"
        if pos in ("ADP", "NOUN", "PROPN", "PRON", "NUM"):
            return "obl" if head_pos in _VERBAL or head_pos == "ADJ" else "nmod"
        return "advmod"
    if label == "oc":
        if pos in _VERBAL and (token.morph_verbform or "") != "Fin":
            return "xcomp"
        return "ccomp"
    if label == "cc":
        return "advcl" if pos in _VERBAL else "obl"
    return "dep"


def _map_tiger(sentence) -> None:
    tokens = sentence.tokens
    for i, t in enumerate(tokens):
        label = (t.dep_type or "").strip().lower()
        if label in _ROOT_LABELS or t.head_index is None or t.head_index == i:
            t.dep_type = "root"
            continue
        head = tokens[t.head_index] if 0 <= t.head_index < len(tokens) else None
        ud = _TIGER_TO_UD.get(label, "dep")
        t.dep_type = ud if ud is not None else _de_by_pos(label, t, head)


# --- English: ClearNLP -> UD, DepEdit ---------------------------------------------------------------------------------------
_depedit = None


def _depedit_instance():
    """DepEdit with our ClearNLP rules followed by stan2uni.ini (one ordered rule list), built once. The instance keeps state
    while it runs (run_depedit), and FastAPI runs requests in threads: map_to_ud serialises the calls (_DEPEDIT_LOCK)."""
    global _depedit
    if _depedit is None:
        from depedit import DepEdit
        d = DepEdit(config_file=os.path.join(_DEPEDIT_DIR, "clearnlp_additions.ini"),
                    options=SimpleNamespace(quiet=True, kill=None))
        d.read_config_file(os.path.join(_DEPEDIT_DIR, "stan2uni.ini"))
        _depedit = d
    return _depedit


def _conllu(sentence) -> str:
    lines = []
    for i, t in enumerate(sentence.tokens):
        label = (t.dep_type or "").strip().lower()
        root = label in _ROOT_LABELS or t.head_index is None or t.head_index == i
        tag = t.pos_value or "_"
        # stan2uni.ini matches Penn Treebank tags in the "pos" column (CoNLL-X input), so both columns get the tag
        lines.append("\t".join([str(i + 1), _field(t.text), _field(t.lemma), _field(tag), _field(tag), "_",
                                "0" if root else str(t.head_index + 1), "root" if root else label, "_", "_"]))
    return "\n".join(lines) + "\n"


def _field(value) -> str:
    value = (value or "").replace("\t", " ").replace("\n", " ").strip()
    return value if value else "_"


def _map_clearnlp(sentence) -> None:
    out = _depedit_instance().run_depedit(_conllu(sentence))
    rows = [line.split("\t") for line in out.splitlines() if "\t" in line and not line.startswith("#")]
    if len(rows) != len(sentence.tokens):
        raise ValueError(f"DepEdit returned {len(rows)} tokens for a sentence of {len(sentence.tokens)}")
    for i, (t, row) in enumerate(zip(sentence.tokens, rows)):
        head = int(float(row[6]))
        t.head_index = i if head == 0 else head - 1
        t.dep_type = "root" if head == 0 else row[7]


_DEPEDIT_LOCK = threading.Lock()


def map_to_ud(sentences: List, scheme: str) -> None:
    """map the relation labels (and for ClearNLP the heads) of the sentences in place from the scheme to UD"""
    if scheme == "TIGER":
        for s in sentences:
            _map_tiger(s)
    elif scheme == "ClearNLP":
        with _DEPEDIT_LOCK:
            for s in sentences:
                _map_clearnlp(s)
    elif scheme != "UD":
        raise ValueError(f"unknown dependency scheme {scheme!r}")
