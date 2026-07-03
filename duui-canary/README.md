# Automatic Speech Recognition (ASR)

DUUI implementation for automatic speech recognition using NVIDIA NeMo Canary.

## Included Models

| Name                     | Version  | URL                                                                 | Languages |
|--------------------------|----------|---------------------------------------------------------------------|-----------|
| nvidia/canary-1b-flash   | r2.3.0   | https://huggingface.co/nvidia/canary-1b-flash                       | en, de    |

# How To Use

For using duui-canary as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm --gpus all -p 1000:9714 docker.texttechnologylab.org/v2/duui-canary:latest
```

Find all available image tags here: https://docker.texttechnologylab.org/v2/duui-canary/tags/list

## Run within DUUI

```
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/v2/duui-canary:latest")
        .withParameter("language", "en")
        .withParameter("model", "nvidia/canary-1b-flash")
);
```

### Parameters

| Name       | Description                                                                 |
|------------|-----------------------------------------------------------------------------|
| `language` | Language of the input audio. Supported values: `en`, `de`                  |
| `model`    | Model to use for transcription. Supported values: `nvidia/canary-1b-flash` |

### Outputs

The annotator produces the following UIMA annotation types:

| Type                                                    | Description                                              |
|---------------------------------------------------------|----------------------------------------------------------|
| `org.texttechnologylab.annotation.type.AudioToken`      | Word-level token with text and audio timestamp (start/end in seconds) |
| `org.texttechnologylab.annotation.type.AudioSentence`   | Sentence/segment span with audio timestamp (start/end in seconds)     |

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

@misc{duui-canary,
  author         = {Baumartz, Daniel},
  title          = {Automatic Speech Recognition as {DUUI} component},
  year           = {2025},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-canary}
}

```
