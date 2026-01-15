[![Version](https://img.shields.io/static/v1?label=taxonerd&message=0.1&color=blue)]()
[![Version](https://img.shields.io/pypi/v/taxonerd)]()

# taxoNERD
A DUUI pipeline for the use of [taxoNERD](https://github.com/nleguillarme/taxonerd).

# HowToUse
For using taxoNERD as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Use as Stand-Alone-Image
```
docker run docker.texttechnologylab.org/taxonerd:1.0
```

## Run with a specific port
```
docker run -p 1000:9714 docker.texttechnologylab.org/taxonerd:1.0
```

### Parameters

| Name        | Description                                                                                     | Default                                                                           |
|-------------|-------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| `model`     | TaxoNERD model (en_ner_eco_md, en_ner_eco_biobert, en_ner_eco_md_weak, en_ner_eco_biobert_weak) | en_ner_eco_md                                                                     |
| `linking`  | Linking-Source. Avialiable sources: 'gbif_backbone', 'taxref', ncbi_taxonomy'     | gbif_backbone                                                                     |
| `exclude`   | List of excluding pipeline steps                                                                | {'tagger', 'parser', 'taxo_abbrev_detector', 'taxon_linker', 'pysbd_sentencizer'} |
| `threshold` | Similarity threshold for entity linking                                                         | 0.7                                                                               |

## Run within DUUI
```
composer.add(new DUUIDockerDriver.
    Component("docker.texttechnologylab.org/taxonerd:1.0")
    .withScale(iWorkers)
    .withImageFetching());
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

@misc{Abrami:2022,
  author         = {Abrami, Giuseppe},
  title          = {taxoNerd as DUUI-Komponent},
  year           = {2022},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-taxoNERD}
}

```

