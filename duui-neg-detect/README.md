# Negation Detection with neg-detect

DUUI component for token-level negation detection. Accepts pre-tokenized, pre-sentence-split text via `Token` and `Sentence` annotations in the CAS, identifies negation cues and their associated scopes, foci, and events, and writes `CompleteNegation` annotations back into the CAS.

# How To Use

For using duui-neg-detect as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm --gpus all -p 1000:9714 docker.texttechnologylab.org/v2/duui-neg-detect:latest
```

Find all available image tags here: https://docker.texttechnologylab.org/v2/duui-neg-detect/tags/list

## Run within DUUI

```java
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/v2/duui-neg-detect:latest")
);
```

### Input types

The CAS must contain sentence and token segmentation before running this component:

| Type | Description |
|---|---|
| `de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence` | Sentence boundaries used to group tokens for the negation pipeline |
| `de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token` | Individual tokens covered by each sentence; surface text is passed to the BERT models |

### Output types

| Type | Description |
|---|---|
| `org.texttechnologylab.annotation.negation.CompleteNegation` | One annotation per detected negation instance. Holds a reference to the negation cue `Token` and `FSArray` fields for the scope, focus, and event token sets. |

Each `CompleteNegation` carries:

| Field | Type | Description |
|---|---|---|
| `cue` | `Token` | The negation trigger word |
| `scope` | `FSArray<Token>` | Tokens within the syntactic scope of the negation |
| `focus` | `FSArray<Token>` | Tokens that are the focal point of the negation |
| `event` | `FSArray<Token>` | Event tokens associated with the negation |

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

@misc{duui-neg-detect,
  author         = {Hammerla, Leon},
  title          = {Negation Detection via neg-detect as {DUUI} component},
  year           = {2025},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-neg-detect}
}

```
