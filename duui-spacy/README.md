# Full NLP Pipeline with spaCy

DUUI component for multi-task NLP annotation using [spaCy](https://spacy.io/).

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

### Pretokenized input

With `use_existing_tokens`, spaCy processes the given tokens unchanged and writes a **new `SpacyToken` for each of them**
(same begin/end) with all features (lemma, POS, morphology, vector, flags such as `is_alpha`, `is_punct`), as without
pretokenization. All other annotations (dependencies, named entities, noun chunks, sentences) begin and end at token
boundaries of the given tokens. The given `Token`s are not changed: annotations that existed before are not merged back
from a DUUI segment, so changes to them would be lost with a segmentation strategy. Afterwards the CAS contains both the
`Token` and the `SpacyToken` for every token; remove the plain tokens if needed (e.g. with `duui-remove-annotations`).

With `use_existing_sentences`, the parser keeps the given sentence boundaries (checked with spaCy 3.7.2,
`de_core_news_lg` and `en_core_web_lg`).

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
