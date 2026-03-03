[![Version](https://img.shields.io/static/v1?label=Python&message=3.10&color=green)]()
[![Version](https://img.shields.io/static/v1?label=Transformers&message=5.2.0&color=yellow)]()
[![Version](https://img.shields.io/static/v1?label=Torch&message=2.2.0&color=red)]()

# Ancient Greek Part-of-Speech Tagger

DUUI implementation for Ancient Greek Part-of-Speech (POS) tagging. This component utilizes a fine-tuned `xlm-roberta-base` model trained on the Universal Dependencies [Ancient Greek Perseus treebank](https://github.com/UniversalDependencies/UD_Ancient_Greek-Perseus), achieving a 91.38% test accuracy for 17 Universal POS tags.

## 1. Annotations

The following is a list of Annotations that are needed as Input for the Docker-Image and are returned as Output by the Docker-Image:
- ### Input (Optional):
  - `de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence` (If sentences are provided, tagging is performed per sentence. Otherwise, the whole document text is processed).
- ### Output:
  - `de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS`

## 2. Included Models

| Name          | Source                                                                         | Revision                                 | Languages |
|---------------|--------------------------------------------------------------------------------|------------------------------------------|-----------|
| ancient-greek-pos-xlmr | https://huggingface.co/qbnguyen/ancient-greek-pos-xlmr             | a297f1e9bffaa7831ce6f2f58d8f6f3a22948952 | Ancient Greek |


# How To Use

For using duui-pos-ancient-greek as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```bash
docker run --rm -p 9714:9714 duui-pos-ancient-greek:latest
```

*(Note: If deployed to the TTLab registry, replace `duui-pos-ancient-greek:latest` with `docker.texttechnologylab.org/duui-pos-ancient-greek:latest`)*

## Run within DUUI

```java
composer.add(
    new DUUIDockerDriver.Component("duui-pos-ancient-greek:latest")
        .withScale(iWorkers)
        .withImageFetching()
        // Optional: specify a different HF model ID or local path
        // .withParameter("model_name", "qbnguyen/ancient-greek-pos-xlmr")
);
```

### Parameters

| Name         | Description                        |
|--------------|------------------------------------|
| `model_name` | Model to use. Default is `qbnguyen/ancient-greek-pos-xlmr` |


# Cite

If you want to use the DUUI image please quote this as follows:

Alexander Leonhardt, Giuseppe Abrami, Daniel Baumartz and Alexander Mehler. (2023). "Unlocking the Heterogeneous Landscape of Big Data NLP with DUUI." Findings of the Association for Computational Linguistics: EMNLP 2023, 385–399. [[LINK](https://aclanthology.org/2023.findings-emnlp.29)] [[PDF](https://aclanthology.org/2023.findings-emnlp.29.pdf)] 

## BibTeX

```bibtex
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

@misc{Nguyen:2026,
  author         = {Nguyen, Quoc-Bao},
  title          = {Ancient Greek POS Tagger as {DUUI} component},
  year           = {2026},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-pos-ancient-greek}
}
```