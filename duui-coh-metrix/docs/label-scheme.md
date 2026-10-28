# Coh-Metrix labels: two-layer scheme

Status: decided on 28 October 2026, implemented in version 0.2.0. The rules for adding or changing an index
are in `README.md` under "Index labels"; this document keeps the background. The scheme replaces the
three-layer scheme of 0.1.x (`label_v3`, public `label_ttlab` with deviation suffixes, `label_internal`). It
takes up a colleague's proposal for public labels that name the kind of deviation, and the original
implementer's answers to the open label questions.

## Why

Three layers were too many and confusing: users had to understand both a public TTLab label with
deviation suffixes and an internal label. The new scheme has two layers:

- **Public label:** the Coh-Metrix 3.0 name, as users know it from Coh-Metrix. Short, readable and,
  apart from a few documented cases, unique.
- **Internal label:** unique, with all details on how the value was computed.

## Fields

| Field         | Content                                                                            | Example (German)       |
|---------------|------------------------------------------------------------------------------------|------------------------|
| `label_v3`    | Public label: the V3 name, or `<NAME>_ext` for indices Coh-Metrix 3.0 doesn't have | `WRDPOLc`, `RDFOG_ext` |
| `label_ttlab` | Internal label: the public label plus detail groups                                | `WRDPOLc_GN`           |
| `label_v2`    | The Coh-Metrix 2 name, as a reference; `None` where there is none                  | `WRDPOLc`              |

`label_internal` is removed. `label_ttlab` changes its meaning: it was the public label, now it is the
internal one.

## Label format

```
<PUBLIC>[_<GROUP>...]
```

1. **Public label.** The V3 name exactly as in Coh-Metrix 3.0. The only suffix is `_ext` (lowercase), for
   project-only indices (`index` 1001 and above, `label_v2` is `None`).
2. **Internal label.** Always starts with the public label, so the public label is the internal label up
   to its first uppercase group. Indices without any resource have no groups (`DESWC`).
3. **Groups.** Each group is written like a Coh-Metrix label: an uppercase core with an optional lowercase
   variant (`SPACYlg`, `CLEARtr`, `LLMlist`). No underscores or hyphens inside a group. Order: the method
   first (`REFIT`, `EMB`), then the data the value depends on, in input order.
4. **Name every resource**, including the original one (English `WN`, `MRC`): every lexicon, list, corpus,
   model or library the value depends on. Preprocessing (tokenization, sentence splitting, POS tags,
   parsing, lemmas, our own syllable counting) is not named; it is documented in the CAS metadata of the
   preprocessing components.
5. **No language codes.** Where the resource differs by language, its name says so (`WN`/`GN`,
   `CLEAR`/`CLEARtr`, `MRC`/`MRCtr`). Lists of the same kind (`LLMlist`, `WIKI`) get the same name in
   every language; the document language is in the CAS.
6. **No model names, no resource versions.** For spaCy vectors only the model size goes into the label
   (`SPACYsm`, `SPACYmd`, `SPACYlg`, `SPACYtrf`). A new version of a resource (a new Wikipedia sample, a
   new LLM list) bumps the index `version`, not the label.
7. **What was used at runtime.** If a fallback is used, the internal label names it (`SEEDlist` when
   GermaNet is unavailable).
8. **Derived indices** (RDL2, PC*) name the resources of their inputs.

## Groups

| Group      | Kind   | Meaning                                                                                              |
|------------|--------|------------------------------------------------------------------------------------------------------|
| `REFIT`    | method | Model fitted again on another reference corpus (Text Easability PCA)                                 |
| `EMB`      | method | Vector space instead of the TASA LSA space                                                           |
| `CLEAR`    | data   | CLEAR corpus (reference corpus of the English Text Easability model)                                 |
| `CLEARtr`  | data   | CLEAR translated into German (reference corpus of the German model)                                  |
| `SPACYxx`  | data   | spaCy token vectors from a model of size `xx` (`sm`, `md`, `lg`, `trf`); `SPACY` if the size is unknown |
| `LLMlist`  | data   | Word list generated with an LLM                                                                      |
| `WN`       | data   | WordNet                                                                                              |
| `GN`       | data   | GermaNet                                                                                             |
| `SEEDlist` | data   | Hand-written seed list (fallback without GermaNet)                                                   |
| `MRC`      | data   | MRC Psycholinguistic Database                                                                        |
| `MRCtr`    | data   | MRC translated into German                                                                           |
| `WIKI`     | data   | Word frequencies from a Wikipedia sample instead of CELEX                                            |
| `TEXTSTAT` | data   | Computed with the textstat library (which does its own sentence and word splitting on the raw text) |

## Mapping

| Indices                                                                     | Public      | Internal EN                 | Internal DE                                    | Comparable EN | Comparable DE           |
|-----------------------------------------------------------------------------|-------------|-----------------------------|------------------------------------------------|---------------|-------------------------|
| DES* (11), CRF* (10), LD* (4), SYN* (7), DR* except DRNEG (7), WRDNOUN…WRDPRP3p (10) | V3 | same as public | same as public                                  | original      | original                |
| PC* (16)                                                                    | `PCNARz`    | `PCNARz_REFIT_CLEAR`        | `PCNARz_REFIT_CLEARtr`                         | refit         | refit                   |
| LSA* (8), SMCAUSlsa                                                         | `LSASS1`    | `LSASS1_EMB_SPACYlg`        | `LSASS1_EMB_SPACYlg`                           | emb           | emb                     |
| CNC* (9), SMTEMP, DRNEG                                                     | `CNCAll`    | `CNCAll_LLMlist`            | `CNCAll_LLMlist`                               | llmlist       | llmlist                 |
| SMCAUSv, SMINTEp                                                            | `SMCAUSv`   | `SMCAUSv_WN`                | `SMCAUSv_GN` / `SMCAUSv_SEEDlist`              | original      | counterpart / seedlist  |
| SMCAUSvp, SMCAUSr, SMINTEr                                                  | `SMCAUSvp`  | `SMCAUSvp_WN_LLMlist`       | `SMCAUSvp_GN_LLMlist` / `SMCAUSvp_SEEDlist_LLMlist` | llmlist  | counterpart, llmlist    |
| SMCAUSwn, WRDPOLc, WRDHYPn/v/nv                                             | `WRDPOLc`   | `WRDPOLc_WN`                | `WRDPOLc_GN`                                   | original      | counterpart             |
| WRDAOAc, WRDFAMc, WRDCNCc, WRDIMGc, WRDMEAc                                 | `WRDAOAc`   | `WRDAOAc_MRC`               | `WRDAOAc_MRCtr`                                | original      | translated              |
| WRDFRQc, WRDFRQa, WRDFRQmc                                                  | `WRDFRQc`   | `WRDFRQc_WIKI`              | `WRDFRQc_WIKI`                                 | corpus        | corpus                  |
| RDFRE                                                                       | `RDFRE`     | `RDFRE_TEXTSTAT`            | `RDFRE_TEXTSTAT`                               | original      | counterpart             |
| RDFKGL                                                                      | `RDFKGL`    | `RDFKGL_TEXTSTAT`           | `RDFKGL_TEXTSTAT`                              | original      | original                |
| RDFOG, RDSMOG, RDARI, RDCLI, RDLW, RDDCRS, RDSPACHE, RDWSTF1–4 (11)         | `RDFOG_ext` | `RDFOG_ext_TEXTSTAT`        | `RDFOG_ext_TEXTSTAT`                           | ext           | ext                     |
| RDL2 with SYNSTRUTa                                                         | `RDL2`      | `RDL2_WIKI`                 | `RDL2_WIKI`                                    | corpus        | corpus                  |
| RDL2 with SYNSTRUTt                                                         | `RDL2t_ext` | `RDL2t_ext_WIKI`            | `RDL2t_ext_WIKI`                               | ext           | ext                     |

The rows with several indices show the first one as the example; the others follow the same pattern
(`LSAGNd_EMB_SPACYlg`, `DRNEG_LLMlist`, `WRDHYPv_GN`, ...).

### Comparability

The label names what was used, but no longer says whether the value is comparable to Coh-Metrix 3.0.
That information goes into a per-language column of the index catalog (README table and
`GET /v1/indices`), not into the label:

| Value         | Meaning                                                                                  |
|---------------|------------------------------------------------------------------------------------------|
| `original`    | Same method and resource as Coh-Metrix 3.0; differences come only from preprocessing    |
| `counterpart` | The language's counterpart of the original resource or formula (GermaNet, Amstad)       |
| `translated`  | The original resource, translated                                                        |
| `corpus`      | Another corpus than the original (Wikipedia instead of CELEX)                            |
| `llmlist`     | A word list generated with an LLM instead of the original list                           |
| `emb`         | Another vector space than the TASA LSA space                                             |
| `refit`       | The model fitted again on another reference corpus                                       |
| `seedlist`    | A hand-written fallback list                                                             |
| `ext`         | Not in Coh-Metrix 3.0                                                                    |

German values are never directly comparable to English norms, also where the column says `original`.

### RDL2

The public label must be unique, because it is the UIMA type and the column name in result tables. RDL2
(Crossley, Greenfield & McNamara 2008) combines CELEX word frequency, syntactic similarity and content word
overlap between adjacent sentences. The variant with SYNSTRUTa (adjacent sentences) keeps `RDL2`; the
variant with SYNSTRUTt (all sentences in a paragraph) becomes the project-only index `RDL2t_ext` with its
own `index` number (1012).

### textstat

`TEXTSTAT` names the library, not how it handles each language: language-specific behaviour is a detail of the
library, like the formulas of `WN` are a detail of WordNet. The textstat version is pinned in `requirements.txt`;
an upgrade that changes values bumps the index `version`. The README documents what textstat does per language
(fork https://github.com/felixvoll/textstat, branch `cohmetrix-de`, tag `v0.7.12-post2`; apart from the German
easy-word list and a UTF-8 fix for loading it, identical to upstream textstat 0.7.12):

| Index                           | Language-specific part                                                                   |
|---------------------------------|------------------------------------------------------------------------------------------|
| RDFRE                           | German uses Amstad's constants (180, 1, 58.5)                                            |
| RDFOG, RDDCRS, RDSPACHE         | "Difficult words" from a per-language easy-word list (`resources/<lang>/easy_words.txt`). The German list (2579 words) is the fork's only addition: a translation and manual adaptation of the English Dale-Chall/Spache list |
| RDFKGL, RDSMOG, RDLW, RDWSTF1–4 | Only syllable counting: pyphen for German, CMU Pronouncing Dictionary then pyphen for English |
| RDARI, RDCLI                    | None (characters, words, sentences)                                                      |

textstat gets the raw text and does its own sentence and word splitting, so its counts can differ from ours.

## Vectors

The LSA indices and SMCAUSlsa need token vectors. A new optional parameter `vectors` chooses the source:

- `SPACY` (default): the vectors on `SpacyToken`. The model size is read from the CAS, not from the
  parameter: the `SpacyAnnotatorMetaData` whose `reference` is the first `SpacyToken` gives the model name
  (e.g. `de_core_news_lg`), and its last part is the size (`SPACYlg`). This is the layout of the
  `duui-spacy` component (one metadata annotation per token). If no metadata is found, the group is
  `SPACY` and a warning is logged. If a size is given in the parameter (`SPACYlg`) and the CAS says
  otherwise, the CAS wins and a warning is logged.
- Any other name (later): vectors from a user-supplied embedding annotation. The name goes into the label
  as given and must have the group format (uppercase core, optional lowercase variant, no underscores),
  e.g. `FTcc`. The embedding input itself is designed when that feature comes.

The size matters because a spaCy `sm` model has no static word vectors: spaCy then returns rows of the
contextual `doc.tensor`. A `trf` model returns no token vectors at all (`has_vector` is false).

If no token in the document has a vector, the component logs one warning per document and emits the LSA
indices and SMCAUSlsa with `value=None` (NaN in the CAS) and an `error` such as
`no token vectors (SPACYtrf)`. Single tokens without a vector are normal (out-of-vocabulary words) and
don't cause a warning.

## Selecting indices

`include_indices` and `exclude_indices` (comma-separated, case-insensitive) accept, from least to most
specific:

| Level | Name                 | Example                                                    |
|-------|----------------------|------------------------------------------------------------|
| 0     | `*` (all)            | `*`                                                        |
| 1     | group (`type_name`)  | `Descriptive`                                              |
| 2     | prefix with `*`      | `RDL2*`, `PCNAR*`, `WRDPOLc_G*`; a longer prefix is more specific |
| 3     | public label         | `LSAGN`, `RDFOG_ext`                                       |
| 4     | internal label       | `WRDPOLc_GN`                                               |

- A prefix needs the `*`. Without it, `LSAGN` would also select `LSAGNd`; many labels are prefixes of
  others (`DESWLsy`/`DESWLsyd`, `SMCAUSv`/`SMCAUSvp`, `CNCTemp`/`CNCTempx`, ...).
- Prefixes match the internal label. Since it starts with the public label, `WRDPOLc*` covers both
  languages; `WRDPOLc_G*` only German documents.
- The most specific match wins; exclude wins on a tie; an empty include means `*` (unchanged).
- Internal labels are language-specific: `WRDPOLc_GN` matches nothing in an English document. A warning is
  logged only for names that match nothing in the catalog for any language.
- A prefix is a plain prefix: `RDL*` also selects `RDLW_ext`, so the two RDL2 indices are `RDL2*`.
- Inputs of derived indices are still computed but not output (unchanged).

## Resolved questions

From the original implementer (October 2026):

- The DRNEG negation lists are LLM-generated: `DRNEG_LLMlist`.
- The causal particle lists (SMCAUSvp, SMCAUSr) are LLM-generated: `_LLMlist`.
- The textstat fork changes nothing in `flesch_kincaid_grade` compared with textstat 0.7.12. textstat
  counts German syllables with pyphen (as we do), English syllables with the CMU Pronouncing Dictionary
  and pyphen as fallback. textstat gets the raw text, so its sentence and word splitting can differ from
  ours. Documented in the README ("textstat").

With the public label now always the V3 name, the following no longer affect the public label:

- Who chose the WordNet synsets (`cause.v.01`, `intend.v.01`, `plan.v.01`) and the German seed verbs.
  The original implementer will check; documentation only (`docs/todo.md`).
- The colleague's proposal (`_refit`, `_embedding`, the kinds of vector spaces, the translated MRC) is
  covered by the groups above (`REFIT`, `EMB`, `SPACYxx`, `MRCtr`; public `WRDAOAc`).

## Consequences

- Breaking change for consumers: `labelTTLab` changes meaning, `labelInternal` is removed from the
  TypeSystem, and the UIMA types of the project-only indices become `RDFOG_ext` etc.
- Existing XMIs and the paper evaluation results (`paper_eval/results`, Postgres) carry the old labels.
- The German copy of the Coh-Metrix code can use the public labels directly, since they are the V3 names.
