# Coh-Metrix labels: two-layer scheme

Status: decided on 28 October 2026, implemented in version 0.2.0; updated for 0.3.0 (4 November 2026, section
"0.3.0: main records and variants"). The rules for adding or changing an index
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

| Group      | Kind   | Meaning                                                                                          |
|------------|--------|--------------------------------------------------------------------------------------------------|
| `REFIT`    | method | Model fitted again on another reference corpus (Text Easability PCA)                             |
| `EMB`      | method | Vector space instead of the TASA LSA space                                                       |
| `TIGERmap` | method | Dependency relations mapped from TIGER to UD (spaCy German, best effort)                         |
| `CLEARNLPmap` | method | Dependency relations mapped from ClearNLP to UD (spaCy English, best effort)                  |
| `REFa`     | data   | Reference corpus A (English): 9 registers × 1,000 passages of 250–350 words; reference corpus of the English Text Easability model |
| `REFv1`    | data   | Reference corpus v1 (German): 9 registers × 1,000 passages of 250–350 words; reference corpus of the German Text Easability model |
| `CLEAR`    | data   | CLEAR corpus, reference corpus of the earlier English Text Easability model                      |
| `CLEARtr`  | data   | CLEAR translated into German, reference corpus of the earlier German model                       |
| `SPACYxx`  | data   | spaCy token vectors from a model of size `xx` (`sm`, `md`, `lg`, `trf`); `SPACY` if unknown; other token vectors: their `shortName` (see [Vectors](#vectors)) |
| `DIMLEX`   | data   | DiMLex, lexicon of German discourse connectives                                                  |
| `DIMLEXeng` | data  | DiMLex-Eng, lexicon of English discourse connectives                                             |
| `GUM`      | data   | How often ambiguous English adverbs are connectives, counted on the GUM corpus (`gum_weights_en.csv`) |
| `LLMlist`  | data   | Word list generated with an LLM                                                                  |
| `WN`       | data   | Princeton WordNet                                                                                |
| `GN`       | data   | GermaNet                                                                                         |
| `SEEDlist` | data   | Hand-written seed list (fallback without GermaNet)                                               |
| `MRC`      | data   | MRC Psycholinguistic Database                                                                    |
| `MRCtr`    | data   | MRC translated into German                                                                       |
| `KUPERMAN` | data   | English age-of-acquisition norms (Kuperman et al.)                                               |
| `BIRCHENOUGH` | data | German age-of-acquisition norms (Birchenough et al.)                                            |
| `BRYSBAERT` | data  | English concreteness norms (Brysbaert et al.)                                                    |
| `KOEPER`   | data   | German concreteness and imageability norms (Köper & Schulte im Walde)                            |
| `SUBTLEX`  | data   | Word frequencies from film subtitles (SUBTLEX-US, SUBTLEX-DE) instead of CELEX                   |
| `WIKI`     | data   | Word frequencies from a Wikipedia sample instead of CELEX                                        |
| `TEXTSTAT` | data   | Computed with the textstat library                                                               |
| `TEXTSTATlist` | data | The easy-word list of the textstat library, in the formula of this component                   |

## Mapping

The main record of each index comes first, its variants follow (section "0.3.0: main records and variants"):

| Indices | Public | Internal EN | Internal DE | Comparable EN | Comparable DE |
|---|---|---|---|---|---|
| DES*, CRF*, LD*, SYN*, DR* except DRNEG, WRDNOUN … WRDPRP3p | V3 | same as public | same as public | original | original |
| PC* (16): `None` with an error until the models are refitted on this component's values (the inputs SYNNP, LSAGN and German SMTEMP are computed differently from the fit) | `PCNARz` | `PCNARz_REFIT_REFa` | `PCNARz_REFIT_REFv1` | refit | refit |
| … variant (not computed until refitted, `None` with an error) | | `PCNARz_REFIT_CLEAR` | `PCNARz_REFIT_CLEARtr` | refit | refit |
| LSA* (8), SMCAUSlsa | `LSASS1` | `LSASS1_EMB_SPACYlg` | `LSASS1_EMB_SPACYlg` | emb | emb |
| CNC* (9) | `CNCAll` | `CNCAll_DIMLEXeng_GUM` | `CNCAll_DIMLEX` | counterpart | counterpart |
| … variant | | `CNCAll_LLMlist` | `CNCAll_LLMlist` | llmlist | llmlist |
| SMCAUSv, SMINTEp | `SMCAUSv` | `SMCAUSv_WN` | `SMCAUSv_GN` / `SMCAUSv_SEEDlist` | original | counterpart |
| SMCAUSvp, SMCAUSr | `SMCAUSvp` | `SMCAUSvp_WN_DIMLEXeng_GUM` | `SMCAUSvp_GN_DIMLEX` / `SMCAUSvp_SEEDlist_DIMLEX` | counterpart | counterpart |
| … variant | | `SMCAUSvp_WN_LLMlist` | `SMCAUSvp_GN_LLMlist` / `SMCAUSvp_SEEDlist_LLMlist` | llmlist | counterpart, llmlist |
| SMINTEr | `SMINTEr` | `SMINTEr_WN_LLMlist` | `SMINTEr_GN_LLMlist` / `SMINTEr_SEEDlist_LLMlist` | original | counterpart |
| SMCAUSwn, WRDPOLc, WRDHYPn/v/nv | `WRDPOLc` | `WRDPOLc_WN` | `WRDPOLc_GN` | original | counterpart |
| SMTEMP | `SMTEMP` | `SMTEMP_LLMlist` | `SMTEMP_LLMlist_GN` / `SMTEMP_LLMlist` | llmlist | llmlist |
| DRNEG | `DRNEG` | `DRNEG_LLMlist` | `DRNEG_LLMlist` | llmlist | llmlist |
| WRDFRQc, WRDFRQa, WRDFRQmc | `WRDFRQc` | `WRDFRQc_SUBTLEX` | `WRDFRQc_SUBTLEX` | corpus | corpus |
| … variant | | `WRDFRQc_WIKI` | `WRDFRQc_WIKI` | corpus | corpus |
| WRDAOAc | `WRDAOAc` | `WRDAOAc_KUPERMAN` | `WRDAOAc_BIRCHENOUGH` | counterpart | counterpart |
| … variant | | `WRDAOAc_MRC` | `WRDAOAc_MRCtr` | original | translated |
| WRDCNCc | `WRDCNCc` | `WRDCNCc_BRYSBAERT` | `WRDCNCc_KOEPER` | counterpart | counterpart |
| … variant | | `WRDCNCc_MRC` | `WRDCNCc_MRCtr` | original | translated |
| WRDIMGc | `WRDIMGc` | `WRDIMGc_MRC` | `WRDIMGc_KOEPER` | original | counterpart |
| … variant (German only) | | | `WRDIMGc_MRCtr` | | translated |
| WRDFAMc, WRDMEAc | `WRDFAMc` | `WRDFAMc_MRC` | `WRDFAMc_MRCtr` | original | translated |
| RDFRE | `RDFRE` | `RDFRE` | `RDFRE` | original | counterpart |
| RDFKGL | `RDFKGL` | `RDFKGL` | `RDFKGL` | original | original |
| RDSMOG, RDARI, RDCLI, RDLW, RDWSTF1–4 (all `_ext`) | `RDSMOG_ext` | `RDSMOG_ext` | `RDSMOG_ext` | ext | ext |
| RDFOG, RDDCRS, RDSPACHE (`_ext`; difficult words from textstat's easy-word list) | `RDFOG_ext` | `RDFOG_ext_TEXTSTATlist` | `RDFOG_ext_TEXTSTATlist` | ext | ext |
| … variant of every RD* above: the textstat library | | `RDFRE_TEXTSTAT`, `RDFOG_ext_TEXTSTAT` | `RDFRE_TEXTSTAT`, `RDFOG_ext_TEXTSTAT` | as the main record | as the main record |
| RDL2 | `RDL2` | `RDL2_SUBTLEX` | `RDL2_SUBTLEX` | corpus | corpus |
| … variant | | `RDL2_WIKI` | `RDL2_WIKI` | corpus | corpus |

The rows with several indices show the first one as the example; the others follow the same pattern
(`LSAGNd_EMB_SPACYlg`, `CNCCaus_DIMLEX`, `WRDHYPv_GN`, ...). Not emitted (always `None`):
DRGERUND for German (German has no gerund), RDWSTF1–4 for English (a German formula).

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
combines word frequency, syntactic similarity and content word overlap between adjacent sentences. 0.2.0 had two
records: `RDL2` with SYNSTRUTa and the project-only `RDL2t_ext` (index 1012) with SYNSTRUTt, because "syntactic
similarity" seemed open. **0.3.0 drops `RDL2t_ext`** (decided 2026-11-02): RDL2 uses the adjacent-sentence mean
SYNSTRUTa (Scott A. Crossley, Jerry Greenfield and Danielle S. McNamara, "Assessing Text Readability Using Cognitively
Based Indices", TESOL Quarterly 42(3), 2008, p. 482, https://doi.org/10.1002/j.1545-7249.2008.tb00142.x; Scott A.
Crossley, David B. Allen and Danielle S. McNamara, "Text readability and intuitive simplification: A comparison of
readability formulas", Reading in a Foreign Language 23(1), 2011, p. 91,
https://scholarspace.manoa.hawaii.edu/items/fc754b04-a1ef-4176-885e-e4389a3a27e0/full), and its regression weights are
not valid for SYNSTRUTt. The UIMA type `RDL2t_ext` is removed. The frequency term is now WRDFRQc (SUBTLEX); the
Wikipedia frequencies are the variant `RDL2_WIKI`.

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
- Since 0.2.0, vectors are read from `WordEmbedding` annotations (e.g. written by duui-spacy since 0.8.0, one per
  word, `reference` to the token); their `shortName` is the group, e.g. `SPACYlg`. The name must have the group
  format (uppercase core, optional lowercase variant, no underscores). Details: README, "Vectors".

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
- A prefix is a plain prefix: `RDL*` also selects `RDLW_ext`, so RDL2 with its variants is `RDL2*`.
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
  The original implementer will check; documentation only.
- The colleague's proposal (`_refit`, `_embedding`, the kinds of vector spaces, the translated MRC) is
  covered by the groups above (`REFIT`, `EMB`, `SPACYxx`, `MRCtr`; public `WRDAOAc`).

## 0.3.0: main records and variants

Integration of Coh-Metrix 0.4.0-py (Maxim Konca; UD input, new resources), decided 2026-11-02:

- **Several records per public label.** An index has one main record and may have variants: further records with
  another resource or method, each with its own internal label. The public label (and UIMA type) is shared; the
  internal label stays unique. The variants follow their index in the output and are selected like any internal
  label (`exclude_indices=*_WIKI`).
- **New main resources** from 0.4.0-py, each named by a group: SUBTLEX frequencies (`WRDFRQ*`, `RDL2`), DiMLex /
  DiMLex-Eng with GUM weights for the connectives (`CNC*`, `SMCAUSvp`, `SMCAUSr`), native norms for age of acquisition,
  concreteness and imageability (`KUPERMAN`, `BIRCHENOUGH`, `BRYSBAERT`, `KOEPER`), Text Easability models fitted on new
  reference corpora (`REFIT_REFa`, `REFIT_REFv1`; models `refa-en-rr-gn-cr3` / `refv1-ud-rr-cr3` of PR #306, fitted on the
  values of 0.5.0-py). These are `None` with an error until refitted on this component's values: SYNNP, LSAGN and German
  SMTEMP are computed differently here (decided 2026-11-04). Most of these files are not in the image (licences); without them
  the records are `None` with an error.
- **Kept as variants** (0.2.0's resources): `_WIKI` frequencies (now log10, like SUBTLEX), `_LLMlist` connectives,
  `_MRC` / `_MRCtr` norms, `_TEXTSTAT` readability, `_REFIT_CLEAR` / `_REFIT_CLEARtr` Text Easability (`None` with an
  error until refitted on the current values). The main readability records are computed by the component itself;
  Dale-Chall and Spache use textstat's easy-word list (`TEXTSTATlist`).
- **Best effort without UD.** On spaCy-only input the dependency relations are mapped to UD in the component; the
  records that read them get a method group right after the public label (`SYNNP_TIGERmap`,
  `SMCAUSr_CLEARNLPmap_WN_DIMLEXeng_GUM`); PC* are `None` with an error. Background of rule 4's exception: the mapping
  is done here, not by a preprocessing tool, so the label must say it.
- `RDL2t_ext` removed (section "RDL2").
- Labels of the readability formulas (decided 2026-11-04): variants computed by the textstat library carry
  `TEXTSTAT`; the main records are computed here and name every resource they use, so the three formulas with "difficult
  words" (Fog, Dale-Chall, Spache) carry `TEXTSTATlist`, textstat's easy-word list.
- README rule 9 (no placeholders) gets an exception (decided 2026-11-04): a record whose model waits for its refit is
  emitted as `None` with the reason, so its label stays stable (the PC* records).

## Consequences

- Breaking change for consumers: `labelTTLab` changes meaning, `labelInternal` is removed from the
  TypeSystem, and the UIMA types of the project-only indices become `RDFOG_ext` etc.
- Existing XMIs and the paper evaluation results (`paper_eval/results`, Postgres) carry the old labels.
- The German copy of the Coh-Metrix code can use the public labels directly, since they are the V3 names.
