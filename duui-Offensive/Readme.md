[![Version](https://img.shields.io/static/v1?label=Python&message=3.12&color=green)]()
[![Version](https://img.shields.io/static/v1?label=Transformers&message=4.52.4&color=yellow)]()
[![Version](https://img.shields.io/static/v1?label=Torch&message=2.6.0&color=red)]()

# Transformers Offensive Type Classification

DUUI implementation for selected Offensive type classification tools: [Offensive](https://huggingface.co/models?search=offensive).
## Included Models

| Name                            | Source                                                                        | Revision                            | Languages          |
|---------------------------------|-------------------------------------------------------------------------------|-------------------------------------|--------------------|
| cnerg-hatexplain                | https://huggingface.co/Hate-speech-CNERG/bert-base-uncased-hatexplain         | e487c81b768c7532bf474bd5e486dedea4cf3848 | EN                 |
| cnerg-hatexplain-rationale      | https://huggingface.co/Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two | 7b1a724a178c639a4b3446c0ff8f13d19be4f471 | EN                 |
| naija-xlm-t-base-hate           | https://huggingface.co/worldbank/naija-xlm-twitter-base-hate                  | 49fe8d380c290260b73e16ea005454ee28b27e5f | EN,HA,YO,IG,PIDGIN |
| hatebert-abuseval               | https://osf.io/tbd58/files/osfstorage?view_only=d90e681c672a494bb555de99fc7ae780 | d90e681c672a494bb555de99fc7ae780    | EN                 |
| hatebert-offenseval             | https://osf.io/tbd58/files/osfstorage?view_only=d90e681c672a494bb555de99fc7ae780 | d90e681c672a494bb555de99fc7ae780    | EN                 |
| bertweet-hate-speech            | https://huggingface.co/pysentimiento/bertweet-hate-speech                     | d9925de199f48face0d7026f07c3b492c423bbc0 | EN                 |
| robertuito-hate-speech          | https://huggingface.co/pysentimiento/robertuito-hate-speech                   | db125ee7be2ad74457b900ae49a7e0f14f7a496c | ES                 |
| bertabaporu-hate-speech         | https://huggingface.co/pysentimiento/bertabaporu-pt-hate-speech               | 9d50687a13df38c7d2fdf4b2227eb28c006214de | PT                 |
| bert-it-hate-speech             | https://huggingface.co/pysentimiento/bert-it-hate-speech                      | 627bbee98534e5bfbbc771fc6c7ecb35ffbfe90a | IT                 |
| imsypp-social-media             | https://huggingface.co/IMSyPP/hate_speech_multilingual                        | 2045782c975894635c4221a1d44aa23b24f0103e | MULTI              |
| imsypp-social-media-en          | https://huggingface.co/IMSyPP/hate_speech_en                                  | 6dc7c7d81577a178a48d484f72cca334f44c7f69 | EN                 |
| imsypp-social-media-it          | https://huggingface.co/IMSyPP/hate_speech_it                                  | 46e36cd04dce8d3517b8014ce782ecc5306e2106 | IT                 |
| imsypp-social-media-nl          | https://huggingface.co/IMSyPP/hate_speech_nl                                  | 571af0e4558288a3f1c249b5bfd1da8149a584a7 | NL                 |
| imsypp-social-media-slo         | https://huggingface.co/IMSyPP/hate_speech_slo                                 | 910059d15a0b554deb5591edc166015bd78848be | SLO                |
| cardiffnlp-hate-multiclass      | https://huggingface.co/cardiffnlp/twitter-roberta-base-hate-multiclass-latest                                  | b9a303f920f8527ac4151e65953c04505fdf0587 | EN                 |
| cardiffnlp-sensitive-multilabel | https://huggingface.co/cardiffnlp/twitter-roberta-large-sensitive-multilabel                                      | e362dc65d7042ac79d5893d250ba60be7d73ef39 | EN                 |



# How To Use

For using duui-offensive as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm -p 1000:9714 docker.texttechnologylab.org/duui-offensive-[modelname]:latest
```

Find all available image tags here: https://docker.texttechnologylab.org/v2/duui-offensive-[modelname]/tags/list

## Run within DUUI

```
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-offensive-[modelname]:latest")
        .withScale(iWorkers)
        .withImageFetching()
);
```

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
  title          = {Offensive classification tools as {DUUI} component},
  year           = {2025},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-Offensive}
}

```
