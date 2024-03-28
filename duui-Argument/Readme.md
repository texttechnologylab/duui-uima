[![Version](https://img.shields.io/static/v1?label=duui-argument&message=0.0.1&color=blue)](https://docker.texttechnologylab.org/v2/duui-argument/tags/list)
[![Version](https://img.shields.io/static/v1?label=Python&message=3.10&color=green)]()
[![Version](https://img.shields.io/static/v1?label=Transformers&message=4.38.2color=yellow)]()
[![Version](https://img.shields.io/static/v1?label=Torch&message=2.2.0&color=red)]()

# Language Argument Tools

DUUI implementation for selected argument Tools, weather the argument support, opposes or neutral to the topic.
## Included Models

| Name     | Revision                                 | URL                                                                           | Language     | Notice       |
|----------|------------------------------------------|-------------------------------------------------------------------------------|--------------|--------------|
| CHKLA    | 7c0e6b88c91828ba07dfc473d2d11628e3b734fc | https://huggingface.co/chkla/roberta-argument                                 | EN           |              |
| UKP      | 72f643b06a06b9ba82a25df2c134664fc26f84f3  | https://github.com/UKPLab/acl2019-BERT-argument-classification-and-clustering | EN           |              |
| UKPLARGE | 72f643b06a06b9ba82a25df2c134664fc26f84f3 | https://github.com/UKPLab/acl2019-BERT-argument-classification-and-clustering | EN           |              |
| Gpt3.5   | 63336788349e400fdbcf08c66e98b1e5b5209736 | https://github.com/openai/openai-cookboob                                     | Multilingual | gpt3.5-turbo |
| Gpt4     | 63336788349e400fdbcf08c66e98b1e5b5209736 | https://github.com/openai/openai-cookboob                                     | Multilingual | gpt4         |
# How To Use

For using duui-argument as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm -p 1000:9714 docker.texttechnologylab.org/v2/duui-argument:latest
```

Find all available image tags here: https://docker.texttechnologylab.org/v2/duui-argument/tags/list

## Run within DUUI

```
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/v2/duui-argument:latest")
        .withParameter("model_name", 'UKP')
        .withParameter("topic", "Zoo")
        .withParameter("chatgpt_key", "")
);
```

### Parameters

| Name          | Description |
|---------------|--|
| `model_name`  | Model to use, see table above |
| `topic`       | topic which should be match with the arguments |
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
  title          = {Language Argument tools as {DUUI} component},
  year           = {2024},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-Argument}
}

```
