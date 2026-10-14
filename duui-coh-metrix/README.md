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
the `_wiki10000` label suffix and their absolute values are not directly
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
- German `SMTEMP` remains computable from unambiguous spaCy tense/aspect
  annotations, but returns `None` when a construction requires GermaNet to
  determine the relevant lexical verb.

English lexical-semantic indices use WordNet instead and do not require the
GermaNet mount.

### Syllable counts

Syllable-based descriptive and readability indices use the language-specific
dictionaries bundled with `pyphen==0.17.2`. Differences between Pyphen
hyphenation and human syllabification are treated as a known external-resource
limitation.

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

Every output index has a `label_ttlab` value. If an implementation uses the
same operationalization as the Coh-Metrix 3.0 index, `label_ttlab` equals
`label_v3`. A suffix identifies an alternative data source or implementation,
for example `_spacy`, `_textstat`, `_wiki10000`, or `_mrctranslate`.

For lexical-semantic indices, the label identifies the language-specific
resource:

- German: `_germanet`
- English: `_wordnet`

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
