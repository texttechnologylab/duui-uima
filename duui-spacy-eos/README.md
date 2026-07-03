# Sentence Splitting with spaCy (EOS)

DUUI component for end-of-sentence (EOS) detection using [spaCy](https://spacy.io/).

# How To Use

For using duui-spacy-eos as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm --gpus all -p 1000:9714 docker.texttechnologylab.org/duui-spacy-eos:latest
```

Find all available image tags here: https://docker.texttechnologylab.org/v2/duui-spacy-eos/tags/list

## Run within DUUI

The component reads the language automatically from the CAS document language. Use `language_override` to force a specific language regardless of the CAS value.

```java
// Auto-detect language from CAS
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-spacy-eos:latest")
);

// Force a specific language
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-spacy-eos:latest")
        .withParameter("language_override", "de")
);
```

### Parameters

| Name | Description | Default |
|---|---|---|
| `language_override` | Force a specific language code (e.g. `"de"`, `"en"`) regardless of the CAS document language. Useful when the CAS language is not set or incorrect. | CAS document language |

## Supported Languages

| Language | Code | spaCy Model |
|---|---|---|
| Catalan | `ca` | `ca_core_news_sm` |
| Chinese | `zh` | `zh_core_web_sm` |
| Croatian | `hr` | `hr_core_news_sm` |
| Danish | `da` | `da_core_news_sm` |
| Dutch | `nl` | `nl_core_news_sm` |
| English | `en` | `en_core_web_sm` |
| Finnish | `fi` | `fi_core_news_sm` |
| French | `fr` | `fr_core_news_sm` |
| German | `de` | `de_core_news_sm` |
| Greek | `el` | `el_core_news_sm` |
| Italian | `it` | `it_core_news_sm` |
| Japanese | `ja` | `ja_core_news_sm` |
| Korean | `ko` | `ko_core_news_sm` |
| Lithuanian | `lt` | `lt_core_news_sm` |
| Macedonian | `mk` | `mk_core_news_sm` |
| Norwegian Bokmål | `nb` | `nb_core_news_sm` |
| Polish | `pl` | `pl_core_news_sm` |
| Portuguese | `pt` | `pt_core_news_sm` |
| Romanian | `ro` | `ro_core_news_sm` |
| Russian | `ru` | `ru_core_news_sm` |
| Slovenian | `sl` | `sl_core_news_sm` |
| Spanish | `es` | `es_core_news_sm` |
| Swedish | `sv` | `sv_core_news_sm` |
| Ukrainian | `uk` | `uk_core_news_sm` |
| Multilingual / Unknown | `xx` / `x-unspecified` | `xx_sent_ud_sm` |

### Output types

| Type | Description |
|---|---|
| `de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence` | One annotation per detected sentence with begin/end character offsets from spaCy's `sent.start_char` / `sent.end_char`. |
| `org.texttechnologylab.annotation.SpacyAnnotatorMetaData` | Single annotation referencing all created `Sentence` annotations via `FSArray`. Carries the component name and version, spaCy library version, model name, model version, model language, model's spaCy version, and model's spaCy git version. |

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

@misc{duui-spacy-eos,
  author         = {Schaaf, Manuel},
  title          = {Sentence Splitting via {spaCy} as {DUUI} component},
  year           = {2025},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-spacy-eos}
}

```
