[![Version](https://img.shields.io/static/v1?label=duui-genre&message=0.1.0&color=blue)](https://docker.texttechnologylab.org/v2/duui-transformers-topic/tags/list)
[![Version](https://img.shields.io/static/v1?label=Python&message=3.12&color=green)]()
[![Version](https://img.shields.io/static/v1?label=Transformers&message=5.9.0&color=yellow)]()
[![Version](https://img.shields.io/static/v1?label=Torch&message=2.11.0&color=red)]()

# Transformers Genre

DUUI implementation for selected Hugging-Face-based transformer [Genre tools](https://huggingface.co/models?sort=trending&search=genre) models.
## Included Models

| Name                                                                          |                                                                                                  | Revision                       | Languages    |
|-------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|--------------------------------|--------------|
| turkunlp-genre-multi                                                    | https://huggingface.co/TurkuNLP/web-register-classification-multilingual | a22ad8b652f6825ec1505dab779979e0f255d7ae | Multilingual |
| turkunlp-genre-en                                      | https://huggingface.co/TurkuNLP/web-register-classification-en | 93969151434144dc8505865d31823c79bd385167 | EN           |
| turkunlp-genre-finerweb                                              |https://huggingface.co/TurkuNLP/finerweb-quality-classifier| 93d1635105c974a675e3be8c636d7a5cac6f7b11 | EN           |
| ssharoff-genre                                                |https://huggingface.co/ssharoff/genres| 93d1635105c974a675e3be8c636d7a5cac6f7b11| EN           |
| x-genre-classifier                               |https://huggingface.co/classla/xlm-roberta-base-multilingual-text-genre-classifier| ebe54ca322f6fd4dc95700705b99f23e3437c8d0 | Multingual   |
 
# How To Use

For using duui-genre as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm -p 9714:9714 docker.texttechnologylab.org/duui-genre-[modelname]:latest

```

Find all available image tags here: [https://docker.texttechnologylab.org/v2/duui-genre-[modelname]/tags/list](https://docker.texttechnologylab.org/v2/duui-transformers-topic-[modelname]/tags/list)

## Run within DUUI

```
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-genre-[modelname]:latest")
        .withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
);
```

### Parameters

| Name | Description |
| ---- | ----------- |
| `selection`  | Use `text` to process the full document text or any selectable UIMA type class name |

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

@misc{Bagci:2024,
  author         = {Bagci, Mevlüt},
  title          = {Hugging-Face-based genre models as {DUUI} component},
  year           = {2024},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-Genre}
}

```
