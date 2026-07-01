# Speech Recognition with CrisperWhisper [WIP]

DUUI component for automatic speech recognition (ASR) using [CrisperWhisper](https://github.com/nyrahealth/CrisperWhisper), a fine-tuned Whisper variant optimised for precise word-level timestamps.
Accepts audio stored as base64 in the CAS sofa, transcribes it word by word, and writes `AudioToken` annotations.

# How To Use

For using duui-crisper-whisper as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm --gpus all -p 1000:9714 docker.texttechnologylab.org/v2/duui-crisper-whisper:latest
```

Find all available image tags here: https://docker.texttechnologylab.org/v2/duui-crisper-whisper/tags/list

## Run within DUUI

The CAS must carry the audio as a base64-encoded WAV string in its sofa data before this component is added to the pipeline.

```java
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/v2/duui-crisper-whisper:latest")
);
```

### Output types

| Type | Description |
|---|---|
| `org.texttechnologylab.annotation.type.AudioToken` | One annotation per recognised word. Carries `begin`/`end` character offsets in the reconstructed transcript, `timeStart`/`timeEnd` in seconds, and the word text via `value`. |
| `org.texttechnologylab.annotation.AnnotatorMetaData` | Created for each `AudioToken`. Records the annotator name, version, model name, and model revision. |
| `org.texttechnologylab.annotation.DocumentModification` | Single annotation recording the annotator name, processing timestamp, and a version comment. |

The CAS sofa data string is also updated in-place with the full space-joined transcript as plain text.

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

@misc{duui-crisper-whisper,
  author         = {Baumartz, Daniel},
  title          = {Speech Recognition via {CrisperWhisper} as {DUUI} component},
  year           = {2025},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-crisper-whisper}
}

```
