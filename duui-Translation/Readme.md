[![Version](https://img.shields.io/static/v1?label=duui-translation&message=0.0.1&color=blue)](https://docker.texttechnologylab.org/v2/duui-translation/tags/list)
[![Version](https://img.shields.io/static/v1?label=Python&message=3.10&color=green)]()
[![Version](https://img.shields.io/static/v1?label=Transformers&message=4.38.2color=yellow)]()
[![Version](https://img.shields.io/static/v1?label=Torch&message=2.2.0&color=red)]()

# Language Translation Tools

DUUI implementation for selected language translation Tools.
## Included Models

| Name                                           | Revision                                  | URL                                                   | Language             |
|------------------------------------------------|-------------------------------------------|----------------------------------------------------------|----------------------|
| MBART                                          | e30b6cb8eb0d43a0b73cab73c7676b9863223a30  | https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt| Multilingual         |
| NLLB                                           | f8d333a098d19b4fd9a8b18f94170487ad3f821d  | https://huggingface.co/facebook/nllb-200-distilled-600M                           | Multilingual         |
| Whisper                                        | ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab  | https://github.com/openai/whisper  | Multilingual2English |
| FlanT5Base                                      | 7bcac572ce56db69c1ea7c8af255c5d7c9672fc2  | https://huggingface.co/google/flan-t5-base | Multilingual         |
# How To Use

For using duui-translation as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm -p 1000:9714 docker.texttechnologylab.org/v2/duui-translation:latest
```

Find all available image tags here: https://docker.texttechnologylab.org/v2/duui-translation/tags/list

## Run within DUUI

```
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/v2/duui-translation:latest")
        .withParameter("model_name", 'NLLB')
        .withParameter("translation", "de,tr")
);
```

### Parameters

| Name          | Description                                    |
|---------------|------------------------------------------------|
| `model_name`  | Model to use, see table above                  |
| `translation` | It's feasible to translate initially into one language and subsequently into another if the model operates more effectively this way. Simply delineate the languages with a comma  |

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
  title          = {Language Translation tools as {DUUI} component},
  year           = {2024},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-Translation}
}

```
