[![Version](https://img.shields.io/static/v1?label=duui-stance-mlburnham&message=0.0.1&color=blue)](https://docker.texttechnologylab.org/v2/duui-stance-mlburnham/tags/list)
[![Version](https://img.shields.io/static/v1?label=Python&message=3.10&color=green)]()
[![Version](https://img.shields.io/static/v1?label=Transformers&message=4.38.2color=yellow)]()
[![Version](https://img.shields.io/static/v1?label=Torch&message=2.2.0&color=red)]()

# Language Stance detection Tools

DUUI implementation for selected stance detection tools.
## Included Models

| Name      | Revision                                 | URL                                                                           | Language     | Notice       |
|-----------|------------------------------------------|-------------------------------------------------------------------------------|--------------|--------------|
| mlburnham | 4538315b9903f9821063023bebcf441cb8c53cdc | https://huggingface.co/mlburnham/deberta-v3-base-polistance-affect-v1.0                                 | EN           |              |
| kornosk   | 36311a4ad7200ac54d3e3aff37daee69d6472888 | https://huggingface.co/kornosk/bert-election2020-twitter-stance-trump | EN           |              |
| gpt3.5    | gpt3.5-turbo                             | https://platform.openai.com/                                     | Multilingual | gpt3.5-turbo |
| gpt4      | gpt-4                                    | https://platform.openai.com/                                    | Multilingual | gpt4         |
# How To Use

For using duui-stance as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm -p 1000:9714 docker.texttechnologylab.org/v2/duui-stance-[model_name]:latest
```

Find all available image tags here: https://docker.texttechnologylab.org/v2/duui-stance-[model_name]/tags/list

Example for gpt3.5: https://docker.texttechnologylab.org/v2/duui-stance-gpt3.5/tags/list

## Run within DUUI

```
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/v2/duui-stance-[model_name]:latest")
        .withParameter("chatgpt_key", "")
);
```

### Parameters

| Name          | Description |
|---------------|--|
| `chatgpt_key` | ChatGPT API key, note only needed by using chatgpt |

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
  title          = {Language Stance tools as {DUUI} component},
  year           = {2024},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-Stance}
}

```
