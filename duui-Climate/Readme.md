[![Version](https://img.shields.io/static/v1?label=duui-climate&message=0.1.0&color=blue)](https://docker.texttechnologylab.org/v2/duui-transformers-topic/tags/list)
[![Version](https://img.shields.io/static/v1?label=Python&message=3.12&color=green)]()
[![Version](https://img.shields.io/static/v1?label=Transformers&message=5.9.0&color=yellow)]()
[![Version](https://img.shields.io/static/v1?label=Torch&message=2.11.0&color=red)]()

# Transformers Climate

DUUI implementation for selected Hugging-Face-based transformer [Climate tools](https://huggingface.co/models?sort=trending&search=climatebert) models.
## Included Models

| Name                                                                          |                                                                                                 | Revision                       | Languages |
|-------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|--------------------------------|----------|
| distilroberta-base-climate-sentiment                                                    | https://huggingface.co/climatebert/distilroberta-base-climate-sentiment | e9f9a94ee4263f5ad5cfc97b8539a497fc88aa7d | EN       |
| distilroberta-base-climate-tcfd                                                    | https://huggingface.co/climatebert/distilroberta-base-climate-tcfd | 970630beedc21db81a84156448ad2e3ac860153d | EN       |
| distilroberta-base-climate-commitment                                                    | https://huggingface.co/climatebert/distilroberta-base-climate-commitment | 17337c3292df16a8fe93b1505dfe4122d50a4c91 | EN       |
| distilroberta-base-climate-sentiment                                                    | https://huggingface.co/climatebert/distilroberta-base-climate-sentiment | e9f9a94ee4263f5ad5cfc97b8539a497fc88aa7d | EN       |
| distilroberta-base-climate-specificity                                                    | https://huggingface.co/climatebert/distilroberta-base-climate-specificity | 4ada96ed4bf5c3a7a711282e41f1ab9b29f0ddea | EN       |
 
# How To Use

For using duui-climate as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm -p 9714:9714 docker.texttechnologylab.org/duui-climate-[modelname]:latest

```

Find all available image tags here: [https://docker.texttechnologylab.org/v2/duui-climate-[modelname]/tags/list](https://docker.texttechnologylab.org/v2/duui-transformers-topic-[modelname]/tags/list)

## Run within DUUI

```
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-climate-[modelname]:latest")
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
  title          = {Hugging-Face-based climate models as {DUUI} component},
  year           = {2026},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-Climate}
}

```
