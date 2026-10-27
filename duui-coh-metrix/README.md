# Coh-Metrix Text Cohesion Analysis [WIP]

DUUI implementation of Coh-Metrix 3.0 for English and German. The component
computes up to 121 index records over pre-annotated UIMA documents, including
descriptive, referential-cohesion, LSA, lexical-diversity, connective,
situation-model, syntactic-complexity, word-information, and readability
measures. The output also includes explicitly named TTLab alternatives where
the available data source or implementation differs from Coh-Metrix 3.0.

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

Token vectors are required for the `LSA*` indices and `SMCAUSlsa`. If no usable
vector or comparison can be constructed, the affected result is `None`.

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
Coh-Metrix implementation. Consequently, the corresponding TTLab indices use
the `_wiki` label suffix (`_enwiki10000` / `_dewiki10000` internally, see
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
with its own PCA and model (`text_easability_de.json`, internal label suffix
`_cleartranslate`). The German inputs include GermaNet indices (polysemy, verb
overlap), so without GermaNet the German components are `None`.
`easability/compare_models.py` compares the English and German components
(congruence of the loadings, correlation of the scores of the same excerpts).

The corpus file is not part of this repository: place `CLEAR_corpus_final.csv`
from the CLEAR repository in `clear_corpus/` and run `ClearCorpusCohMetrixTest`
(CLEAR reader -> spaCy -> Coh-Metrix -> XMI) to recreate the processed corpus.
The PCA scripts in `easability/` use the dependencies in `requirements-dev.txt`
(Python 3.12):

```bash
python easability/extract_indices.py clear_corpus/xmi_en clear_corpus/indices_en.csv
python easability/pca.py clear_corpus/indices_en.csv easability/results/clear_en_pca \
    --model src/main/resources/easability/text_easability_en.json --language en \
    --corpus "CLEAR corpus (Crossley et al. 2022), CLEAR_corpus_final.csv, 4724 excerpts" \
    --label-suffix _clear
```

`pca.py` writes the model and, to `easability/results/clear_en_pca/`, a
readable summary (`report.md`: comparison with the published components,
loadings, marker scores, validation against the CLEAR difficulty ratings) and
the underlying tables, including the component scores of every CLEAR excerpt
(`scores.csv`, by excerpt ID).

German: place the translation in `clear_corpus/` (default
`translation_kolibri-1-q4km/`, other versions with `-Dclear.translation=...`)
and give the GermaNet XML directory (it must be readable by the Docker daemon;
copy it from a network mount to a local directory):

```bash
mvn test -Dtest='ClearCorpusCohMetrixTest#runPipelineGermanTranslation' \
    -Dcohmetrix.germanet.path=/path/to/GN_V180_XML
python easability/extract_indices.py clear_corpus/xmi_de_kolibri-1-q4km \
    clear_corpus/indices_de_kolibri-1-q4km.csv
python easability/pca.py clear_corpus/indices_de_kolibri-1-q4km.csv \
    easability/results/clear_de_kolibri-1-q4km_pca \
    --model src/main/resources/easability/text_easability_de.json --language de \
    --corpus "CLEAR corpus (Crossley et al. 2022), CLEAR_corpus_final.csv, translated into German with Kolibri-1 Q4_K_M (translation_kolibri-1-q4km), 4669 of 4724 excerpts (without truncated translations and paragraph mismatches)" \
    --label-suffix _cleartranslate
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

The Wiener Sachtextformel indices `RDWSTF1_textstat` through
`RDWSTF4_textstat` are defined for German only. For English and other
languages, these four indices are not applicable and therefore return
`None`/`Double.NaN` instead of a numeric value. This behavior is already
visible in the completed English validation reports; no separate label or
single-purpose WSTF testcase is required.

## Index labels

Every index record carries four labels. Follow the rules below when adding or
changing an index. `label_ttlab` and `label_internal` are required and set
explicitly in every `Index(...)` call.

| Field (Python / UIMA)              | Meaning                                                                                                                                                       | Unique per output |
|------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|
| `label_v3` / `labelV3`             | Original Coh-Metrix 3.0 label from `docs/coh-metrix-30-indices.md`. Identifies *what* is measured. `None` for project-only indices.                           | No, see rule 5    |
| `label_v2` / `labelV2`             | Coh-Metrix 2.0 label from the same table; `n/a` where the table has none. `None` for project-only indices.                                                   | No, see rule 5    |
| `label_ttlab` / `labelTTLab`       | Public TTLab label for use in analyses: the V3 label, plus a suffix only where the English implementation deviates from the original. Identical for all languages. | Yes               |
| `label_internal` / `labelInternal` | Internal TTLab label: V3 label plus the concrete resource suffix for this language; no suffix where the original resource is used. May differ between languages. Always set, also where it equals `label_ttlab`, so either label can be used on its own. | Yes               |

### Rules

1. **When an implementation deviates.** Ask: *if our NLP preprocessing were
   perfect, would we reproduce the original Coh-Metrix value?*
   - No, because we use a different corpus, lexicon, word list, embedding
     model, formula or library: it deviates and gets a suffix.
   - Yes, and the only differences come from preprocessing (sentence
     boundaries, tokens, lemmas, POS, morphology, dependencies, noun chunks,
     Pyphen syllable counts): no suffix.

   For `label_ttlab` the test is applied to the English implementation
   (rule 2); for `label_internal` it is applied to each language's own
   implementation.
2. **The public label follows the English implementation.** Coh-Metrix is
   defined on English, so `label_ttlab` gets a suffix only if the English
   implementation deviates from the original. Another language that uses its
   counterpart of the English resource measures the same concept and keeps
   the same public label; its concrete resource appears only in
   `label_internal`. This includes translated resources, because they are
   built on the same data. `label_ttlab` is therefore identical for all
   languages. Examples:
   - `WRDPOLc` (public): internally `WRDPOLc` for English (Princeton WordNet,
     as in the original) and `WRDPOLc_germanet` for German.
   - `WRDAOAc` (public): internally `WRDAOAc` for English (original MRC
     database) and `WRDAOAc_mrctranslate` for German (MRC translated with
     DeepL).
   - `WRDFRQc_wiki` (public): English deviates (Wikipedia instead of CELEX);
     internally `WRDFRQc_enwiki10000` for English and `WRDFRQc_dewiki10000`
     for German.
3. **Derived indices inherit deviations.** An index computed from other
   indices carries the suffixes of its inputs. Example: `RDL2` uses
   `WRDFRQmc`, so it carries the Wikipedia suffix.
4. **The internal label names what was actually used at runtime.** If a
   resource is unavailable and a fallback is used, `label_internal` names the
   fallback, not the intended resource; `label_ttlab` stays aligned. Example:
   without GermaNet, the German causal/intentional verb inventories use seed
   lists, so `label_internal` uses `_seedlist` instead of `_germanet`.
5. **V3 and V2 labels repeat only for genuine variants.** If the original
   definition is ambiguous, all variants keep the V3/V2 label and `index`
   number and are told apart by a variant suffix in `label_ttlab` and
   `label_internal`. Example: the RDL2 formula only says `SYNSTRUT`, so both
   `RDL2_synstruta_*` and `RDL2_synstrutt_*` have V3 `RDL2`.
6. **No placeholders.** If the original resource is unavailable, emit only the
   substitute, under the original V3 label. Never emit a record that is always
   `None` just to occupy the original label. Example: there is no CELEX
   `WRDFRQc`, only `WRDFRQc_wiki`.
7. **Project-only indices** (no Coh-Metrix equivalent) have `label_v3` and
   `label_v2` set to `None`, an `index` of 1001 or above, and always a suffix.
   Example: `RDFOG_textstat`.
8. **Suffix order:** `<V3 label>_<variant>_<resource>`, for example
   `RDL2_synstruta_wiki`.
9. **UIMA type.** The CAS type is `org.texttechnologylab.uima.type.cohmetrix.<label_v3>`,
   or `<label_ttlab>` for project-only indices. Language variants and rule-5
   variants share one type and are distinguished by `labelTTLab`. A new
   project-only index needs a matching type in `TypeSystem.xml`.
10. **Suffixes name the kind of resource, not the tool or vendor.** Examples:
    `_llmlist` for a word list generated with an LLM (not the model name),
    `_mrctranslate` for a translated resource (not the translation service),
    `_textstat` for the library (not the language-specific formula it uses
    internally, such as Amstad's Flesch variant for German).

### Suffixes

Public suffixes describe how the English implementation deviates from the
original and are the same for all languages; internal suffixes name the
concrete resource per language.

| Public (`label_ttlab`) | English implementation (original)              | Internal EN      | Internal DE          |
|------------------------|------------------------------------------------|------------------|----------------------|
| *(none)*               | Princeton WordNet (same)                       | *(none)*         | `_germanet`          |
| *(none)*               | MRC database (same)                            | *(none)*         | `_mrctranslate`      |
| `_wiki`                | Wikipedia sample (CELEX)                       | `_enwiki10000`   | `_dewiki10000`       |
| `_clear`               | CLEAR corpus (TASA)                            | `_clear`         | `_cleartranslate`    |
| `_spacy`               | spaCy vectors (TASA LSA space)                 | `_spacy`         | `_spacy`             |
| `_textstat`            | textstat library (Coh-Metrix formulas)         | `_textstat`      | `_textstat`          |
| `_llmlist`             | LLM-generated word lists (Coh-Metrix lists)    | `_llmlist`       | `_llmlist`           |

Adding a language changes only internal labels, as long as its resources are
counterparts of the English ones. Example: a French wordnet keeps the public
label `WRDPOLc` and gets the internal label `WRDPOLc_<resource name>`. A new
public suffix is only needed when the English implementation changes.

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

Both take a comma-separated list of names, matched case-insensitively:

- `*`: all indices;
- an index group (the `type_name`, e.g. `Descriptive`, `LSA`, `Word Information`);
- a single index, by its public label `label_ttlab` (`LSAGN_spacy`) or its V3
  label (`LSAGN`). The V3 label `RDL2` selects both RDL2 variants.

Only these language-independent names are accepted, so the same parameters
select the same indices in a mixed-language corpus: `exclude_indices=WRDPOLc`
removes polysemy for English (WordNet) and German (GermaNet). Internal labels
such as `WRDPOLc_germanet` are not accepted. `GET /v1/indices` lists all
indices with their group, public and V3 label.

For each index, the most specific matching name decides: a single index beats
its group, a group beats `*`. If an index is named in both lists on the same
level, it is excluded. An empty `include_indices` means `*`. Examples:

| `include_indices` | `exclude_indices`      | Result                                  |
|-------------------|------------------------|-----------------------------------------|
|                   |                        | all indices                             |
| `Descriptive`     | `DESPC,DESSC`          | Descriptive without DESPC and DESSC     |
|                   | `LSA`                  | all indices except LSA                  |
| `DESWC,LSAGN`     |                        | only DESWC and LSAGN                    |
| `*,DESWC`         | `Descriptive`          | all except Descriptive, but with DESWC  |
| `DESWC`           | `Descriptive`          | only DESWC                              |

Indices computed from other indices (`RDL2`, the Text Easability components)
compute their inputs as well, but the inputs are only output if they are
selected themselves. Unknown names are logged as a warning and ignored; for an
internal label the warning names the public label to use. If
`include_indices` contains only unknown names, nothing is computed.

Skipping indices also skips their shared preprocessing, e.g. the LSA vectors,
connective counts or hypernymy lookups.

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
