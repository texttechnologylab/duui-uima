# Full NLP Pipeline with spaCy

DUUI component for multi-task NLP annotation using [spaCy](https://spacy.io/).

Versions: spaCy 3.8.18 with the models 3.8.0 (`*_core_*_sm`, `*_core_*_lg`, `xx_ent_wiki_sm`; parameter
`model_variant`: `efficiency` (default, sm) or `large` (lg)), Python 3.11. The transformer models (`accuracy`) are not
included. All packages are pinned in `requirements.txt`.

# How To Use

For using duui-spacy as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm -p 1000:9714 docker.texttechnologylab.org/v2/duui-spacy:latest
```

Find all available image tags here: https://docker.texttechnologylab.org/v2/duui-spacy/tags/list

## Run within DUUI

```java
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/v2/duui-spacy:latest")
);
```

### Parameters

| Name | Description | Default |
|---|---|---|
| `use_existing_tokens` | If `"true"`, read existing `Token` annotations (exactly this type, not its subtypes such as `SpacyToken`) as input instead of raw text | `false` |
| `use_existing_sentences` | If `"true"` (requires `use_existing_tokens`), also read existing `Sentence` annotations to set sentence boundaries; spaCy then writes no sentences | `false` |
| `use_benepar` | If `"true"`, add the Berkeley Neural Parser (benepar) to the spaCy pipeline (languages `de`, `en`), see below | `false` |

### Constituency parsing with benepar

With `use_benepar`, the [Berkeley Neural Parser](https://github.com/nikitakit/self-attentive-parser) (benepar 0.2.0)
runs after the spaCy model, with `benepar_en3` (English, based on T5-small) or `benepar_de2` (German, based on XLM-R).
Papers: Nikita Kitaev and Dan Klein (2018), "Constituency Parsing with a Self-Attentive Encoder", ACL 2018,
https://aclanthology.org/P18-1249; Nikita Kitaev, Steven Cao and Dan Klein (2019), "Multilingual Constituency Parsing
with Self-Attention and Pre-Training", ACL 2019, https://aclanthology.org/P19-1340.

- Output: `SpacyToken.beneparLabels`, the labels of the constituent that consists of exactly this token (e.g. `NP` for
  a one-word noun phrase); not set if there is none. The German model has no one-word constituents, so German tokens
  get no labels. Constituents over several tokens are not written yet.
- benepar's POS tagger replaces spaCy's fine-grained tags (`tag_`) of the parsed sentences (benepar's default).
- benepar cannot parse a sentence of more than 512 sub-word tokens of its transformer model. Such a sentence is skipped
  (logged as a warning): its tokens get no labels and keep spaCy's tags; all other sentences are parsed.
- Memory: loading `de_core_news_lg` with `benepar_de2` needs up to ~6 GB, and takes about a minute on the first
  request; use a DUUI timeout above that (`withTimeout`, in seconds).
- Only in images built with `TEXTIMAGER_SPACY_WITH_BENEPAR=true` in `docker_build.sh` (default `false`; the image name
  then gets `-benepar`, e.g. `duui-spacy-benepar-en-de`). With `use_benepar`, other images log the error "benepar is
  not installed" and return no annotations for the document. Dependencies
  are pinned in `requirements-benepar.txt` (benepar 0.2.0 needs transformers 4.30.x, so Python 3.11).
- Test: `SpaCyBeneparTest` (service on `localhost:9714`).

### Pretokenized input

With `use_existing_tokens`, spaCy processes the given tokens unchanged and writes a **new `SpacyToken` for each of them**
(same begin/end) with all features (lemma, POS, morphology, vector, flags such as `is_alpha`, `is_punct`), as without
pretokenization. All other annotations (dependencies, named entities, noun chunks, sentences) begin and end at token
boundaries of the given tokens. The given `Token`s are not changed: annotations that existed before are not merged back
from a DUUI segment, so changes to them would be lost with a segmentation strategy. Afterwards the CAS contains both the
`Token` and the `SpacyToken` for every token; remove the plain tokens if needed (e.g. with `duui-remove-annotations`).

With `use_existing_sentences`, the parser keeps the given sentence boundaries (checked with spaCy 3.8.18,
`de_core_news_lg` and `en_core_web_lg` 3.8.0).

# Cite

If you want to use the DUUI image please quote this as follows:

Alexander Leonhardt, Giuseppe Abrami, Daniel Baumartz and Alexander Mehler. (2023). "Unlocking the Heterogeneous Landscape of Big Data NLP with DUUI." Findings of the Association for Computational Linguistics: EMNLP 2023, 385–399. [[LINK](https://aclanthology.org/2023.findings-emnlp.29)] [[PDF](https://aclanthology.org/2023.findings-emnlp.29.pdf)]

## BibTeX

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

@misc{duui-spacy,
  author         = {Baumartz, Daniel},
  title          = {Full {NLP} Pipeline via {spaCy} as {DUUI} component},
  year           = {2025},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-spacy}
}

```
