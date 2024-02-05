[![Version](https://img.shields.io/static/v1?label=duui-transformers-toxic&message=0.1&color=blue)](https://docker.texttechnologylab.org/v2/textimager-duui-transformers-toxic/tags/list)
[![Version](https://img.shields.io/static/v1?label=Python&message=3.8&color=green)]()
[![Version](https://img.shields.io/static/v1?label=Transformers&message=4.22.1&color=yellow)]()
[![Version](https://img.shields.io/static/v1?label=Torch&message=2.1.1&color=red)]()

# Transformers Sentiment

DUUI implementation for selected Hugging-Face-based transformer [Toxic tools](https://huggingface.co/models?sort=trending&search=sentiment).
and for [Detoxify](https://github.com/unitaryai/detoxify) model.
## Included Models

| Name                                                    | Revision                                 | Languages                              |
|---------------------------------------------------------|------------------------------------------|----------------------------------------|
| Detoxify                                                | 773203c10bcf0e8d801b4be8c93cfd97ffe5c2e0 | EN, FR, ES, IT, PT, TR, RU             |
| EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus      | 0126552291025f2fc854f5acdbe45b2212eabf4a | Multilingual                           |
| FredZhang7/one-for-all-toxicity-v3         | a2996bd4495269071eaf5daf73512234c33cb3d2 | Multilingual                           |
| citizenlab/distilbert-base-multilingual-cased-toxicity               | b4532a8b095d1886a7b5dff818331ecc88a855ae | EN, FR, NL, PT, IT, SP, DE, PL, DA, AF |
| tomh/toxigen_hatebert                   | c260d78a7867bb9a9748184afaf454d6ccf28129 | EN                                     |
| GroNLP/hateBERT         | 1d439ddf8a588fc8c44c4169ff9e102f3e839cca | EN                                     |
| pysentimiento/bertweet-hate-speech    | d9925de199f48face0d7026f07c3b492c423bbc0 | EN                                     |
| Hate-speech-CNERG/bert-base-uncased-hatexplain                 | e487c81b768c7532bf474bd5e486dedea4cf3848 | EN                                     |
| cardiffnlp/twitter-roberta-base-hate-latest                    | c74b0534df96af8232f6a3ffdb90d9a72223d7b7 | EN                                     |
| Hate-speech-CNERG/dehatebert-mono-german       | 53a24df030e8e20e7880a161494fb5922ce34617 | DE                |
| deepset/bert-base-german-cased-hatespeech-GermEval18Coarse                        | 70e4821931a8a685d83bc0e8bd8877157bdb3883 | DE                                     |
| martin-ha/toxic-comment-model | 9842c08b35a4687e7b211187d676986c8c96256d | EN                 |
| nicholasKluge/ToxicityModel                | d40cd71847981a0868aa3554c96c0aaf8c189753 | EN                                     |
| EIStakovskii/german_toxicity_classifier_plus_v2                | 1bcb7d11ffc9267111c7be1dad0d7ca2fbf73928          | EN                                     |
| nicholasKluge/ToxicityModel                | d40cd71847981a0868aa3554c96c0aaf8c189753 | EN                                     |
# How To Use

For using duui-transformers-toxic as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm -p 1000:9714 docker.texttechnologylab.org/textimager-duui-transformers-toxic:latest
```

Find all available image tags here: https://docker.texttechnologylab.org/v2/textimager-duui-transformers-toxic/tags/list

## Run within DUUI

```
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/textimager-duui-transformers-toxic:latest")
        .withScale(iWorkers)
        .withImageFetching()
);
```

### Parameters

| Name | Description |
| ---- | ----------- |
| `model_name` | Model to use, see table above |
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

@misc{Baumartz:2022,
  author         = {Bagci, Mevlüt},
  title          = {Hugging-Face-based toxic models  and detoxify models as {DUUI} component},
  year           = {2024},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-transformers-toxic}
}

```