[![Version](https://img.shields.io/static/v1?label=duui-transformers-emotion&message=0.1.0&color=blue)](https://docker.texttechnologylab.org/v2/duui-transformers-emotion/tags/list)
[![Version](https://img.shields.io/static/v1?label=Python&message=3.8&color=green)]()
[![Version](https://img.shields.io/static/v1?label=Transformers&message=4.27.4&color=yellow)]()
[![Version](https://img.shields.io/static/v1?label=Torch&message=1.13.1&color=red)]()

# Transformers Emotion

DUUI implementation for selected Hugging-Face-based transformer [Emotion tools](https://huggingface.co/models?sort=trending&search=emotion) models and [pol_emo_mDeBERTa](https://github.com/tweedmann/pol_emo_mDeBERTa2).
## Included Models

| Name                                                  | Revision                                 | Languages    |
|-------------------------------------------------------|------------------------------------------|--------------|
| 02shanky/finetuned-twitter-xlm-roberta-base-emotion   | 28e6d080e9f73171b574dd88ac768da9e6622c36 | Multilingual |
| DReAMy-lib/xlm-roberta-large-DreamBank-emotion-presence | b3487623ec2dd4b9bd0644d8266291afb9956e9f | Multilingual |
| pol_emo_mDeBERTa                           | 523da7dc2523631787ef0712bad53bfe2ac46840 | Multilingual |
| MilaNLProc/xlm-emo-t                                       | a6ee7c9fad08d60204e7ae437d41d392381496f0 | Multilingual |
 | j-hartmann/emotion-english-distilroberta-base | 0e1cd914e3d46199ed785853e12b57304e04178b | EN           |
 | michellejieli/emotion_text_classifier | dc4df5597fcda82589511c3900fedbe1c0ffec82 | EN           |
 | cardiffnlp/twitter-roberta-base-emotion | 2848306ad936b7cd47c76c2c4e14d694a41e0f54 | EN           |
 | finiteautomata/bertweet-base-emotion-analysis | c482c9e1750a29dcc393234816bcf468ff77cd2d | EN           |
# How To Use

For using duui-transformers-emotion as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm -p 1000:9714 docker.texttechnologylab.org/duui-transformers-emotion:latest
```

Find all available image tags here: https://docker.texttechnologylab.org/v2/duui-transformers-emotion/tags/list

## Run within DUUI

```
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-transformers-emotion:latest")
        .withScale(iWorkers)
        .withImageFetching()
);
```

### Parameters

| Name | Description |
| ---- | ----------- |
| `model_name` | Model to use, see table above |
| `selection`  | Use `text` to process the full document text or any selectable UIMA type class name |

## Building
Before build download [pol_emo_mDeBERTa2.zip](https://github.com/tweedmann/pol_emo_mDeBERTa2/releases/download/v.1.0.0/pol_emo_mDeBERTa2.zip) and save the folder pol_emo_DeBERTa under the python directory.

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
  title          = {Hugging-Face-based emotion models as {DUUI} component},
  year           = {2023},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-transformers-Emotion}
}

```
