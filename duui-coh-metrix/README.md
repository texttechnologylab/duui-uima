# Coh-Metrix Text Cohesion Analysis [WIP]

DUUI implementation of Coh-Metrix 3.0 for English and German. The component
computes up to 118 index records over pre-annotated UIMA documents, including
descriptive, referential-cohesion, LSA, lexical-diversity, connective,
situation-model, syntactic-complexity, word-information, and readability
measures. Every index carries its Coh-Metrix 3.0 label and an internal label
that names the resources used where they differ from or replace those of
Coh-Metrix 3.0 (see [Index labels](#index-labels)).

## Prerequisites and preprocessing

The annotator requires a **pre-annotated UIMA CAS** with the following
annotation types:

- `de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph`
- `de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence`
- `org.texttechnologylab.uima.type.spacy.SpacyToken`
- `org.texttechnologylab.uima.type.spacy.SpacyNounChunk`

The validated pipeline uses the following upstream components:

- **DUUI spaCy:**  using spaCy `3.7.2` in the validated
  pipeline. The validated models are `de_core_news_sm` `3.7.0` for German and
  `en_core_web_sm` `3.7.1` for English. The component supplies sentence
  boundaries, tokens, lemmas, universal and language-specific POS tags,
  dependency information, noun chunks, morphology, and token vectors.
- **DUUI Syntok:** `duui-syntok:0.0.3`, configured with
  `write_paragraphs=true` and `write_sentences=false`. In this configuration,
  Syntok supplies paragraph boundaries while the existing spaCy sentence
  annotations are retained.

Token vectors are required for the `LSA*` indices and `SMCAUSlsa` (see
[Vectors](#vectors)). If no usable vector or comparison can be constructed, the
affected result is `None`.

## Bundled and optional resources

### MRC psycholinguistic ratings

The English resource
`mrc_psycholinguistic_database.csv` is downloaded during the Docker build from
the public
[MRC Psycholinguistic Database dataset](https://huggingface.co/datasets/StephanAkkerman/MRC-psycholinguistic-database).
The implementation reads the ratings for age of acquisition, familiarity,
concreteness, imageability, and Colorado meaningfulness. The public source
contains approximately 150,000 lexical records; 9,335 entries have at least
one available target rating and therefore provide usable coverage for these
indices.

`mrc_psycholinguistic_database_de.csv` is a project-specific German derivative.
The English headwords were translated with DeepL, normalized to German lemmas,
and duplicate translations were merged by averaging the available ratings.
The current resource contains
4,275 non-empty unique German entries. It is therefore an approximation and
not an independent German norming study. In both MRC files, a rating of `0` is
treated as the database's missing-value marker and is not included in averages.

### Word frequencies

The files
`word_frequencies_en_enwiki-20220301-sample10000.csv` and
`word_frequencies_de_dewiki-20220301-sample10000.csv` are project resources
derived from 10,000-article samples of the English and German Wikipedia dumps
dated 2022-03-01. Counts are normalized to frequencies per million before the
indices are calculated. Words not covered by the resource are excluded rather
than assigned an artificial frequency of zero.

These resources replace the non-public CELEX data used by the original
Coh-Metrix implementation. Consequently, the corresponding indices carry the
`WIKI` group in their internal label (`WRDFRQc_WIKI`, see
[Index labels](#index-labels)) and their absolute values are not directly
comparable with CELEX-based Coh-Metrix output.

### WordNet and NLTK

English lexical-semantic indices use Princeton WordNet through `nltk==3.10.0`.

### GermaNet

German lexical-semantic indices use GermaNet through `germanetpy==0.2.5`.
GermaNet is not distributed with the image because it is a separately licensed
resource. It must be mounted into the container and configured through
`DUUI_COH_METRIX_GERMANET_PATH`.

Without GermaNet:

- German causal and intentional verb inventories use their documented seed
  lists where this fallback is supported.
- GermaNet-dependent polysemy, hypernymy, and semantic verb-overlap results are
  `None`.
- German `SMTEMP` is computed from spaCy tense/aspect annotations. GermaNet is
  only needed for `am`/`beim` progressive constructions whose noun spaCy does
  not tag as a verb, to decide whether the noun is a nominalized verb. Without
  GermaNet, the sentence pairs that contain such a sentence are excluded; the
  index is `None` only if no valid pair remains.

English lexical-semantic indices use WordNet instead and do not require the
GermaNet mount.

### Syllable counts

Syllable-based descriptive and readability indices use the language-specific
dictionaries bundled with `pyphen==0.17.2`. Differences between Pyphen
hyphenation and human syllabification are treated as a known external-resource
limitation.

### textstat

The readability formulas (`RDFRE`, `RDFKGL` and the project-only `RD*_ext`
indices) use the textstat fork
[felixvoll/textstat](https://github.com/felixvoll/textstat/tree/cohmetrix-de),
tag `v0.7.12-post2`. Apart from a German easy-word list and a UTF-8 fix for
loading it, it is identical to textstat 0.7.12. textstat gets the raw text and
does its own sentence and word splitting, so its counts can differ from the
other indices. Its language-specific behaviour belongs to the library and is
not part of the labels (`TEXTSTAT` in every language):

| Index                           | Language-specific part                                                                                      |
|---------------------------------|-------------------------------------------------------------------------------------------------------------|
| RDFRE                           | German uses Amstad's constants (180, 1, 58.5) instead of Flesch's                                           |
| RDFOG_ext, RDDCRS_ext, RDSPACHE_ext | "Difficult words" from a per-language easy-word list; the German list (2,579 words, the fork's addition) is a translation and manual adaptation of the English Dale-Chall/Spache list |
| RDFKGL, RDSMOG_ext, RDLW_ext, RDWSTF1–4_ext | Only syllable counting: Pyphen for German, the CMU Pronouncing Dictionary with Pyphen as fallback for English |
| RDARI_ext, RDCLI_ext            | None (characters, words, sentences)                                                                         |

### Text Easability reference corpus

The Text Easability components (`PC*`, indices 12-27) are a first test. The
original components (Graesser, McNamara & Kulikowich, 2011) come from a PCA
on the non-public TASA corpus. Here the English reference corpus is the
[CommonLit Ease of Readability (CLEAR) corpus](https://github.com/scrosseye/CLEAR-Corpus)
(Crossley et al., 2022): 4,724 excerpts, PCA with Varimax rotation on the
available published input measures (`easability/pca_input_measures.csv`),
components named and oriented (higher = easier) by marker measures
(`easability/components.csv`). The z score is relative to CLEAR (mean 0,
SD 1); the percentile is the share of CLEAR excerpts with a lower z. An
input value that is also undefined in some CLEAR excerpts (e.g. causal
cohesion without causal verbs) counts as the CLEAR mean; if any other input is
undefined (empty document, single sentence), the components are `None`. The
model is `src/main/resources/easability/text_easability_en.json`; the scripts
and open questions are in `easability/`.

The German reference corpus is CLEAR translated into German
(`translation_eval/translate_clear.py`; first version Kolibri-1 Q4_K_M, 4,669
of 4,724 excerpts without truncated translations and paragraph mismatches),
with its own PCA and model (`text_easability_de.json`, internal label groups
`_REFIT_CLEARtr`). The German inputs include GermaNet indices (polysemy, verb
overlap), so without GermaNet the German components are `None`.
`easability/compare_models.py` compares the English and German components
(congruence of the loadings, correlation of the scores of the same excerpts).

The corpus file is not part of this repository: place `CLEAR_corpus_final.csv`
from the CLEAR repository in the corpus directory (`CLEAR_DIR` in
`ClearCorpusCohMetrixTest`, `/storage/projects/CORE/projects2/coh_metrix/clear_corpus`;
`$CLEAR` below) and run `ClearCorpusCohMetrixTest`
(CLEAR reader -> spaCy -> Coh-Metrix -> XMI) to recreate the processed corpus.
The PCA scripts in `easability/` use the dependencies in `requirements-dev.txt`
(Python 3.12):

```bash
CLEAR=/storage/projects/CORE/projects2/coh_metrix/clear_corpus
python easability/extract_indices.py $CLEAR/xmi_en $CLEAR/indices_en.csv
python easability/pca.py $CLEAR/indices_en.csv easability/results/clear_en_pca \
    --model src/main/resources/easability/text_easability_en.json --language en \
    --corpus "CLEAR corpus (Crossley et al. 2022), CLEAR_corpus_final.csv, 4724 excerpts" \
    --label-suffix _REFIT_CLEAR
```

`pca.py` writes the model and, to `easability/results/clear_en_pca/`, a
readable summary (`report.md`: comparison with the published components,
loadings, marker scores, validation against the CLEAR difficulty ratings) and
the underlying tables, including the component scores of every CLEAR excerpt
(`scores.csv`, by excerpt ID).

German: place the translation in `$CLEAR` (default
`translation_kolibri-1-q4km/`, other versions with `-Dclear.translation=...`)
and give the GermaNet XML directory (it must be readable by the Docker daemon;
copy it from a network mount to a local directory):

```bash
mvn test -Dtest='ClearCorpusCohMetrixTest#runPipelineGermanTranslation' \
    -Dcohmetrix.germanet.path=/path/to/GN_V180_XML
python easability/extract_indices.py $CLEAR/xmi_de_kolibri-1-q4km \
    $CLEAR/indices_de_kolibri-1-q4km.csv
python easability/pca.py $CLEAR/indices_de_kolibri-1-q4km.csv \
    easability/results/clear_de_kolibri-1-q4km_pca \
    --model src/main/resources/easability/text_easability_de.json --language de \
    --corpus "CLEAR corpus (Crossley et al. 2022), CLEAR_corpus_final.csv, translated into German with Kolibri-1 Q4_K_M (translation_kolibri-1-q4km), 4669 of 4724 excerpts (without truncated translations and paragraph mismatches)" \
    --label-suffix _REFIT_CLEARtr
python easability/compare_models.py easability/results/clear_en_pca \
    easability/results/clear_de_kolibri-1-q4km_pca \
    easability/results/clear_en_vs_de_kolibri-1-q4km --name-a EN --name-b DE
```

`ClearCorpusTextEasabilityTest` processes every 100th excerpt again and checks
that the component's z scores equal the PCA scores, and that selecting only
some PC indices (`include_indices`) gives the same values (German: with
`-Dcohmetrix.germanet.path`).

## Result semantics

`0.0` is emitted only when an actual calculation was possible and its result
is zero. If an index lacks required words, sentences, paragraphs, annotation
information, vectors, a valid comparison pair, a non-zero denominator, or a
required lexical resource, the final Python value is `None`; the Lua
communication layer transfers it to the CAS as `Double.NaN`.

Some resource-backed indices use available-case evaluation: uncovered items
are excluded and a value is calculated from the remaining valid observations.
If too few valid observations remain, the result is `None`/`Double.NaN`.

The Wiener Sachtextformel indices `RDWSTF1_ext` through
`RDWSTF4_ext` are defined for German only. For English and other
languages, these four indices are not applicable and therefore return
`None`/`Double.NaN` instead of a numeric value. This behavior is already
visible in the completed English validation reports; no separate label or
single-purpose WSTF testcase is required.

## Index labels

Every index record carries three labels. Follow the rules below when adding or
changing an index. The background and the decisions behind the scheme are in
[`docs/label-scheme.md`](docs/label-scheme.md).

| Field (Python / UIMA)        | Meaning                                                                                                                                                         | Unique per output |
|------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|
| `label_v3` / `labelV3`       | **Public label.** The Coh-Metrix 3.0 label from `docs/coh-metrix-30-indices.md`, or `<NAME>_ext` for indices Coh-Metrix 3.0 does not have. Identical for all languages; the UIMA type. | Yes |
| `label_ttlab` / `labelTTLab` | **Internal label.** The public label plus groups naming the method and every resource the value depends on for this document, e.g. `WRDPOLc_GN`. May differ between languages and documents. | Yes |
| `label_v2` / `labelV2`       | Coh-Metrix 2.0 label from the same table, as a reference; `n/a` where the table has none, `None` for project-only indices.                                       | No                |

`INDEX_CATALOG` in `duui.py` lists every index with its public label, the
template of its internal label and its comparability (below); the `Index(...)`
calls take the internal label from it (`ttlab_labels`).

### Rules

1. **Public label.** The V3 label exactly as in Coh-Metrix 3.0. The only
   suffix is `_ext` (lowercase), for project-only indices: `index` 1001 and
   above, `label_v2` `None`. Example: `RDFOG_ext`.
2. **Internal label.** Always the public label plus zero or more groups:
   `<public>[_<GROUP>...]`. Indices without any resource have no groups
   (`DESWC`).
3. **Groups** are written like Coh-Metrix labels: an uppercase core with an
   optional lowercase variant (`SPACYlg`, `CLEARtr`, `LLMlist`), without
   underscores or hyphens. The method comes first (`REFIT`, `EMB`), then the
   data the value depends on, in input order (`SMCAUSvp_WN_LLMlist`: verbs,
   then particles).
4. **Name every resource**, including the original one (English `WN`,
   `MRC`): every lexicon, list, corpus, model or library the value depends
   on. Preprocessing (sentence boundaries, tokens, lemmas, POS, morphology,
   dependencies, noun chunks, Pyphen syllable counts) is not named; it is
   documented in the CAS metadata of the preprocessing components.
5. **No language codes.** Where the resource differs by language, its name
   says so (`WN`/`GN`, `CLEAR`/`CLEARtr`, `MRC`/`MRCtr`). Resources of the same
   kind (`LLMlist`, `WIKI`, `TEXTSTAT`) get the same name in every language;
   the document language is in the CAS. How a library handles each language
   is part of the library (see [textstat](#textstat)).
6. **No model names, no resource versions.** For spaCy vectors only the model
   size goes into the label (`SPACYsm`, `SPACYmd`, `SPACYlg`, `SPACYtrf`). A
   new version of a resource (Wikipedia sample, LLM list, MRC translation)
   bumps the index `version`, not the label. Exception: the name of the token
   vectors is not chosen by this component but by the tool or user that wrote
   them (`WordEmbedding.shortName`, see [Vectors](#vectors)).
7. **What was used at runtime.** If a fallback is used, the internal label
   names it. Example: without GermaNet, the German causal/intentional verb
   inventories use seed lists: `SMCAUSv_SEEDlist` instead of `SMCAUSv_GN`.
8. **Derived indices** (RDL2, PC*) name the resources of their inputs
   (`RDL2_WIKI`).
9. **No placeholders.** If the original resource is unavailable, emit only the
   substitute, under the original V3 label. Never emit a record that is always
   `None` just to occupy the original label. Example: there is no CELEX
   `WRDFRQc`; `WRDFRQc` is the Wikipedia variant (`WRDFRQc_WIKI`).
10. **UIMA type.** The CAS type is
    `org.texttechnologylab.uima.type.cohmetrix.<label_v3>`. A new index needs
    a matching type in `TypeSystem.xml`.

### Groups

| Group      | Kind   | Meaning                                                                                          |
|------------|--------|--------------------------------------------------------------------------------------------------|
| `REFIT`    | method | Model fitted again on another reference corpus (Text Easability PCA)                             |
| `EMB`      | method | Vector space instead of the TASA LSA space                                                       |
| `CLEAR`    | data   | CLEAR corpus, reference corpus of the English Text Easability model                              |
| `CLEARtr`  | data   | CLEAR translated into German, reference corpus of the German model                               |
| `SPACYxx`  | data   | spaCy token vectors from a model of size `xx` (`sm`, `md`, `lg`, `trf`); `SPACY` if unknown; other token vectors: their `shortName` (see [Vectors](#vectors)) |
| `LLMlist`  | data   | Word list generated with an LLM                                                                  |
| `WN`       | data   | Princeton WordNet                                                                                |
| `GN`       | data   | GermaNet                                                                                         |
| `SEEDlist` | data   | Hand-written seed list (fallback without GermaNet)                                               |
| `MRC`      | data   | MRC Psycholinguistic Database                                                                    |
| `MRCtr`    | data   | MRC translated into German                                                                       |
| `WIKI`     | data   | Word frequencies from a Wikipedia sample instead of CELEX                                        |
| `TEXTSTAT` | data   | Computed with the textstat library                                                               |

### Labels and comparability

The internal label names what was used, but not whether the value is
comparable to Coh-Metrix 3.0. That is the comparability column, per language
(also returned by `GET /v1/indices`):

| Value         | Meaning                                                                               |
|---------------|---------------------------------------------------------------------------------------|
| `original`    | Same method and resource as Coh-Metrix 3.0; differences come only from preprocessing |
| `counterpart` | The language's counterpart of the original resource or formula (GermaNet, Amstad)    |
| `translated`  | The original resource, translated                                                     |
| `corpus`      | Another corpus than the original (Wikipedia instead of CELEX)                         |
| `llmlist`     | A word list generated with an LLM instead of the original list                        |
| `emb`         | Another vector space than the TASA LSA space                                          |
| `refit`       | The model fitted again on another reference corpus                                    |
| `seedlist`    | A hand-written fallback list                                                          |
| `ext`         | Not in Coh-Metrix 3.0                                                                 |

German values are never directly comparable to English norms, also where the
column says `original`.

| Indices                                                                            | Public      | Internal EN           | Internal DE                                    | Comparable EN | Comparable DE          |
|------------------------------------------------------------------------------------|-------------|-----------------------|------------------------------------------------|---------------|------------------------|
| DES*, CRF*, LD*, SYN*, DR* except DRNEG, WRDNOUN … WRDPRP3p                        | V3          | same as public        | same as public                                 | original      | original               |
| PC* (16)                                                                           | `PCNARz`    | `PCNARz_REFIT_CLEAR`  | `PCNARz_REFIT_CLEARtr`                         | refit         | refit                  |
| LSA* (8), SMCAUSlsa                                                                | `LSASS1`    | `LSASS1_EMB_SPACYlg`  | `LSASS1_EMB_SPACYlg`                           | emb           | emb                    |
| CNC* (9), SMTEMP, DRNEG                                                            | `CNCAll`    | `CNCAll_LLMlist`      | `CNCAll_LLMlist`                               | llmlist       | llmlist                |
| SMCAUSv, SMINTEp                                                                   | `SMCAUSv`   | `SMCAUSv_WN`          | `SMCAUSv_GN` / `SMCAUSv_SEEDlist`              | original      | counterpart / seedlist |
| SMCAUSvp, SMCAUSr, SMINTEr                                                         | `SMCAUSvp`  | `SMCAUSvp_WN_LLMlist` | `SMCAUSvp_GN_LLMlist` / `SMCAUSvp_SEEDlist_LLMlist` | llmlist  | counterpart / seedlist, llmlist |
| SMCAUSwn, WRDPOLc, WRDHYPn/v/nv                                                    | `WRDPOLc`   | `WRDPOLc_WN`          | `WRDPOLc_GN`                                   | original      | counterpart            |
| WRDAOAc, WRDFAMc, WRDCNCc, WRDIMGc, WRDMEAc                                        | `WRDAOAc`   | `WRDAOAc_MRC`         | `WRDAOAc_MRCtr`                                | original      | translated             |
| WRDFRQc, WRDFRQa, WRDFRQmc                                                         | `WRDFRQc`   | `WRDFRQc_WIKI`        | `WRDFRQc_WIKI`                                 | corpus        | corpus                 |
| RDFRE                                                                              | `RDFRE`     | `RDFRE_TEXTSTAT`      | `RDFRE_TEXTSTAT`                               | original      | counterpart            |
| RDFKGL                                                                             | `RDFKGL`    | `RDFKGL_TEXTSTAT`     | `RDFKGL_TEXTSTAT`                              | original      | original               |
| RDFOG, RDSMOG, RDARI, RDCLI, RDLW, RDDCRS, RDSPACHE, RDWSTF1–4 (all `_ext`)        | `RDFOG_ext` | `RDFOG_ext_TEXTSTAT`  | `RDFOG_ext_TEXTSTAT`                           | ext           | ext                    |
| RDL2 (with SYNSTRUTa)                                                              | `RDL2`      | `RDL2_WIKI`           | `RDL2_WIKI`                                    | corpus        | corpus                 |
| RDL2 with SYNSTRUTt instead of SYNSTRUTa                                           | `RDL2t_ext` | `RDL2t_ext_WIKI`      | `RDL2t_ext_WIKI`                               | ext           | ext                    |

Rows with several indices show the first one; the others follow the same
pattern (`LSAGNd_EMB_SPACYlg`, `DRNEG_LLMlist`, `WRDHYPv_GN`, ...). Other
languages than German use the English resources. RDL2 (Crossley, Greenfield &
McNamara 2008) only says "syntactic similarity"; the variant with the
adjacent-sentence measure SYNSTRUTa matches its other adjacent-sentence inputs
and keeps `RDL2`, the variant with SYNSTRUTt is the project-only `RDL2t_ext`.

Adding a language changes only internal labels: a French wordnet keeps the
public label `WRDPOLc` and gets a new group, e.g. `WRDPOLc_WOLF`.

### Index version

Every index sets `version` explicitly in its `Index(...)` call, starting at
`"1.0"`. It versions the computation of that one index, independent of the
image version in the annotation metadata, so results can be compared index by
index across releases. Bump it in the same change that alters the value for
the same input:

- minor (`1.0` -> `1.1`): values change, the meaning stays, e.g. a bug fix,
  changed `None`/edge-case handling, an updated word list or a new resource
  version (Wikipedia sample, MRC translation, spaCy model);
- major (`1.x` -> `2.0`): the meaning changes, e.g. a different formula,
  unit or resource type, usually together with a label change.

Label renames and refactorings or speedups with identical results do not
change the version.

## Validation

The Docker component is regression-tested through
`CohMetrixDockerValidationTest`. The Java test loads 322 frozen bilingual CAS
snapshots, removes previous Coh-Metrix output, processes each CAS with the
Docker image, and checks 3,496 curated index expectations. The frozen upstream
annotations make changes in Coh-Metrix, its Lua communication layer, and the
container service reproducible without rerunning spaCy or Syntok.

The final online and Maven-offline runs on 2026-09-17 used the same image
digest and produced:

```text
Tests run: 3499, Failures: 0, Errors: 0, Skipped: 9
BUILD SUCCESS
```

The nine skipped assertions are documented external limitations: two Pyphen
syllabification differences, three frozen spaCy annotation differences for
German `SMTEMP`, and four GermaNet expectations for which the licensed
resource is intentionally absent from the static test image.

Setup, execution commands, expectation semantics, and the remaining optional
test hardening are documented in
[`JAVA_VALIDATION_TESTS.md`](JAVA_VALIDATION_TESTS.md).

## How to use

For using duui-coh-metrix as a DUUI image it is necessary to use the
[Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

### Start the Docker container

```
docker run --rm -p 1000:9714 docker.texttechnologylab.org/v2/duui-coh-metrix:latest
```

Find all available image tags at
<https://docker.texttechnologylab.org/v2/duui-coh-metrix/tags/list>.

To enable GermaNet-backed German indices:

```
docker run --rm -p 1000:9714 \
  -v /path/to/germanet:/usr/src/app/src/main/resources/germanet \
  docker.texttechnologylab.org/v2/duui-coh-metrix:latest
```

### Run within DUUI

```java
DUUIComposer composer = new DUUIComposer()
    .withDebugLevel(DUUIComposer.DebugLevel.INFO)
    .withComponentLogging(true)
    .withDebugSeverity(true)
    .withDebugSource(true);

composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/v2/duui-coh-metrix:latest")
);
```

### Selecting indices

By default all indices are computed. Two optional parameters select a subset;
indices that are not selected are neither computed nor output:

```java
new DUUIDockerDriver.Component("docker.texttechnologylab.org/v2/duui-coh-metrix:latest")
    .withParameter("include_indices", "Descriptive,LSA")
    .withParameter("exclude_indices", "DESPC,LSAGN,LSAGNd")
```

Both take a comma-separated list of names, matched case-insensitively, from
least to most specific:

1. `*`: all indices;
2. an index group (the `type_name`, e.g. `Descriptive`, `LSA`, `Word Information`);
3. a prefix of the internal label, ending in `*`: `RDL2*` (RDL2 and
   `RDL2t_ext`), `PCNAR*`, `LSAGN*` (LSAGN and LSAGNd), `WRDPOLc_G*`;
4. a public label (`LSAGN`, `RDFOG_ext`);
5. an internal label (`WRDPOLc_GN`, `LSASS1_EMB_SPACYlg`).

A name without `*` is exact: `LSAGN` does not select `LSAGNd`. Prefixes are
matched against the internal label, which starts with the public label, so
`WRDPOLc*` selects polysemy in every language, while `WRDPOLc_G*` or
`WRDPOLc_GN` only match German documents. Public labels and groups select the
same indices in a mixed-language corpus. `GET /v1/indices` lists all indices
with their group, public label, internal label per language (vectors as
`SPACY*`) and comparability.

For each index, the most specific matching name decides; among prefixes, the
longer one. If the most specific match is in both lists, the index is
excluded. An empty `include_indices` means `*`. Examples:

| `include_indices` | `exclude_indices`      | Result                                  |
|-------------------|------------------------|-----------------------------------------|
|                   |                        | all indices                             |
| `Descriptive`     | `DESPC,DESSC`          | Descriptive without DESPC and DESSC     |
|                   | `LSA`                  | all indices except LSA                  |
| `DESWC,LSAGN`     |                        | only DESWC and LSAGN                    |
| `*,DESWC`         | `Descriptive`          | all except Descriptive, but with DESWC  |
| `DESWC`           | `Descriptive`          | only DESWC                              |
| `RD*`             | `RDL2*`                | Readability without RDL2 and RDL2t_ext  |
| `Readability`     | `*_ext`                | all Readability: `*` only works at the end, so `*_ext` is an unknown name (warning) |

Indices computed from other indices (`RDL2`, the Text Easability components)
compute their inputs as well, but the inputs are only output if they are
selected themselves. Unknown names are logged as a warning and ignored; for a
label of the previous scheme (e.g. `LSASS1_spacy`) the warning names the
public label to use. A name that only exists in another language (e.g.
`WRDPOLc_GN` in an English document) selects nothing, without a warning. If
`include_indices` contains only unknown names, nothing is computed.

Skipping indices also skips their shared preprocessing, e.g. the LSA vectors,
connective counts or hypernymy lookups.

### Vectors

The LSA indices and `SMCAUSlsa` use token vectors. They are read from the
`WordEmbedding` annotations (`org.texttechnologylab.uima.type.WordEmbedding`,
one per token, same span as the token), e.g. written by `duui-spacy` (since
0.8.0) or any other tool. Each `WordEmbedding` has a `shortName`, the name of
its vector space, chosen by the tool or user that wrote it (`duui-spacy`:
`SPACY` + model size, e.g. `SPACYlg`, parameter `vectors_short_name`). The
optional parameter `vectors` selects them by this name:

```java
new DUUIDockerDriver.Component("docker.texttechnologylab.org/v2/duui-coh-metrix:latest")
    .withParameter("vectors", "SPACYlg")
```

- The name goes unchanged into the internal label (`LSASS1_EMB_SPACYlg`), so it
  must be an uppercase name with an optional lowercase variant
  (`[A-Z][A-Z0-9]*[a-z0-9]*`). This component needs no change for vectors of a
  new tool or model.
- Without the parameter, the only `shortName` in the CAS is used. If the CAS
  has vectors of several names, the LSA indices and `SMCAUSlsa` are `None`
  with an error that lists them; set `vectors` then.
- A name that is not in the CAS gives the error `No token vectors (<name>)`.

CASes without `WordEmbedding` annotations (e.g. from `duui-spacy` before
0.8.0, or with `vectors_in_spacytoken`) use the vectors of the `SpacyToken`
annotations, as before:

- `SPACY` (default): the model size goes into the internal label
  (`LSASS1_EMB_SPACYlg`) and is read from the CAS: the Lua layer sends the
  `modelName` of the `SpacyAnnotatorMetaData` that references the first
  `SpacyToken` (the layout of the `duui-spacy` component), and its last part
  (`sm`, `md`, `lg`, `trf`) is the size. Without that metadata the group is
  `SPACY` and a warning is logged. A size in the parameter (`SPACYlg`) is only
  used when the CAS has none; if the CAS says otherwise, the CAS wins and a
  warning is logged.
- Other names are not supported for `SpacyToken` vectors: the LSA indices are
  `None` with an error.

The size matters: a spaCy `sm` model has no static word vectors, so spaCy
returns rows of the contextual `doc.tensor` instead. A `trf` model returns no
token vectors at all. If no token of a document has a vector, the component
logs a warning and the LSA indices and `SMCAUSlsa` are `None` with the error
`No token vectors (<group>)`. Single tokens without a vector (words outside
the model's vocabulary) are normal and are skipped.

### Component logging

The Python service uses the official
[DUUIlogger](https://github.com/texttechnologylab/DUUIlogger). If component
logging is enabled in a logging-capable DUUI Core, request logs are returned
with the `/v1/process` response and are printed directly in the Java runner's
console. `INFO` shows request start/completion messages; `WARN` limits the
output to warnings and errors. `withDebugColorful(false)` is recommended for
CI logs and redirected files.

The transport is request/response based: Java receives the records after the
component response has arrived. Startup messages and logs from a process that
hangs or exits before sending its response remain available through Docker:

```bash
docker logs --timestamps --follow duui-coh-metrix-pipeline
```

## Cite

If you want to use the DUUI image please quote this as follows:

Alexander Leonhardt, Giuseppe Abrami, Daniel Baumartz and Alexander Mehler.
(2023). "Unlocking the Heterogeneous Landscape of Big Data NLP with DUUI."
Findings of the Association for Computational Linguistics: EMNLP 2023,
385–399. [[LINK](https://aclanthology.org/2023.findings-emnlp.29)]
[[PDF](https://aclanthology.org/2023.findings-emnlp.29.pdf)]

### BibTeX

```
@inproceedings{Leonhardt:et:al:2023,
  title     = {Unlocking the Heterogeneous Landscape of Big Data {NLP} with {DUUI}},
  author    = {Leonhardt, Alexander and Abrami, Giuseppe and Baumartz, Daniel and Mehler, Alexander},
  editor    = {Bouamor, Houda and Pino, Juan and Bali, Kalika},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2023},
  year      = {2023},
  address   = {Singapore},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2023.findings-emnlp.29},
  pages     = {385--399},
  pdf       = {https://aclanthology.org/2023.findings-emnlp.29.pdf},
  abstract  = {Automatic analysis of large corpora is a complex task, especially
               in terms of time efficiency. This complexity is increased by the
               fact that flexible, extensible text analysis requires the continuous
               integration of ever new tools. Since there are no adequate frameworks
               for these purposes in the field of NLP, and especially in the
               context of UIMA, that are not outdated or unusable for security
               reasons, we present a new approach to address the latter task:
               Docker Unified UIMA Interface (DUUI), a scalable, flexible, lightweight,
               and feature-rich framework for automatic distributed analysis
               of text corpora that leverages Big Data experience and virtualization
               with Docker. We evaluate DUUI{'}s communication approach against
               a state-of-the-art approach and demonstrate its outstanding behavior
               in terms of time efficiency, enabling the analysis of big text
               data.}
}

@misc{duui-coh-metrix,
  author         = {Baumartz, Daniel},
  title          = {Coh-Metrix Text Cohesion Analysis as {DUUI} component},
  year           = {2026},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-coh-metrix}
}

```
