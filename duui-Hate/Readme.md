[![Version](https://img.shields.io/static/v1?label=Python&message=3.8&color=green)]()
[![Version](https://img.shields.io/static/v1?label=Transformers&message=4.38.1&color=yellow)]()
[![Version](https://img.shields.io/static/v1?label=Torch&message=2.2.0&color=red)]()

# Transformers Hate Classification

DUUI implementation for selected hate classification tools: [Hate](https://huggingface.co/models?search=hate).
## Included Models

| Name | Source                                                                   | Revision                          | Languages |
|----|--------------------------------------------------------------------------|-----------------------------------|-----------|
| andrazp | https://huggingface.co/Andrazp/multilingual-hate-speech-robacofi         | c2b98c47f5e13c326a7af48ba544fff4d93fbc70 | Multi     |
| alexandrainst | https://huggingface.co/alexandrainst/da-hatespeech-detection-base        | 6ec0fe1587f6038765b0d7f59525dd4162c4acb2 | Multi     |
| l3cube | https://huggingface.co/l3cube-pune/me-hate-roberta                       | 63890c746c153af20a6cd9832ccbeda03e0d960b | Multi     |
| gronlp | https://huggingface.co/GroNLP/hateBERT                                   | 1d439ddf8a588fc8c44c4169ff9e102f3e839cca | EN        |
| cardiffnlp | https://huggingface.co/cardiffnlp/twitter-roberta-base-hate-latest       | c74b0534df96af8232f6a3ffdb90d9a72223d7b7 | EN        |
| cnerg | https://huggingface.co/Hate-speech-CNERG/dehatebert-mono-english         | 25d0e4d9122d2a5c283e07405a325e3dfd4a73b3 | EN        |
| deepset | https://huggingface.co/deepset/bert-base-german-cased-hatespeech-GermEval18Coarse | 70e4821931a8a685d83bc0e8bd8877157bdb3883 | DE        |
| cnergde | https://huggingface.co/Hate-speech-CNERG/dehatebert-mono-german          | 53a24df030e8e20e7880a161494fb5922ce34617 | DE        |
| cnerges | https://huggingface.co/Hate-speech-CNERG/dehatebert-mono-spanish         | 2b9664ac59ee7f0b054fc0b1433cbedff3c2bdba | ES        |
| cnergpl | https://huggingface.co/Hate-speech-CNERG/dehatebert-mono-polish          | ec586b2e2e6140879c6f533ccd5208d1c2692715 | PL        |
| cnergpt | https://huggingface.co/Hate-speech-CNERG/dehatebert-mono-portugese       | a212b2dd7e8e3d953787a49d92c469b30c6da6ba | PT        |
| cnergit | https://huggingface.co/Hate-speech-CNERG/dehatebert-mono-italian         | aeb70b454d5fc3046aa2a062c525d1ac60f2f01b | IT        |
| cnergar | https://huggingface.co/Hate-speech-CNERG/dehatebert-mono-arabic          | e592a5ee3b913ec33286ee90fb27c7f7f1a8b996 | AR        |
| cnergfr | https://huggingface.co/Hate-speech-CNERG/dehatebert-mono-french          | 7c0e8c45e9176581e57d4ae7e52327258116f969 | FR        |
| cnergid | https://huggingface.co/Hate-speech-CNERG/dehatebert-mono-indonesian      | 08693d6cc64f7e7b3019b2a3abe3b1a9c8ca74c2 | ID        |
| lftw-facebook | https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target  | 391c99ab8b3f65beb77746a2cf6ddf1ddf9817e6 | EN        |
| metahatebert | https://huggingface.co/irlab-udc/MetaHateBERT                            | 60fa7df002300a3fdf56dbfb5c0fbe2a45ee43fa | EN        |
| hatebert-hateval | https://osf.io/tbd58/files/osfstorage?view_only=d90e681c672a494bb555de99fc7ae780 | d90e681c672a494bb555de99fc7ae780 | EN        |
| hate-check-eziisk | https://huggingface.co/EZiisk/EZ_finetune_Vidgen_model_RHS_Best          | acad8f3dfadfa4a86695398c01953bc324efe03b | EN        |
| indo-bert-tweet | https://huggingface.co/Exqrch/IndoBERTweet-HateSpeech                    | 3a4ea9e295cdca78f87581bbf39729b767e9521c | EN, IN    |
| hate-ita-large | https://huggingface.co/MilaNLProc/hate-ita-xlm-r-large                   | f7d96c3ee937fd8f01f98ce9b7783dfea0f5085d | IT        |
| hate-ita-base | https://huggingface.co/MilaNLProc/hate-ita-xlm-r-base                    | 723fa1158c76c684312c43692afb9780810be099 | IT        |
| hate-ita | https://huggingface.co/MilaNLProc/hate-ita-xlm-r                         | 00a79c2221d16f04fcca2c9202dec85c6a815ba7 | IT        |
| mehate-bert | https://huggingface.co/l3cube-pune/me-hate-bert                         | 407f19357c3b2166db6cbc2107807fc07a17b8f5 | MULTI     |
| hatemoji | https://huggingface.co/HannahRoseKirk/Hatemoji                         | f2f98581ab15fb3ccf8b8a5465d7ca70c2958902 | EN        |
| codemix-hate   | https://huggingface.co/debajyotimaz/codemix_hate                          | b07d73f1a05dd04c0adbb941b5446064b14feb10 | EN, HI    |

# How To Use

For using duui-Hate as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm -p 1000:9714 docker.texttechnologylab.org/duui-hate-[modelname]:latest
```

Find all available image tags here: https://docker.texttechnologylab.org/v2/duui-hate-[modelname]/tags/list

## Run within DUUI

```
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-hate-[modelname]:latest")
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
  title          = {Hate classification tools as {DUUI} component},
  year           = {2024},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-Hate}
}

```
