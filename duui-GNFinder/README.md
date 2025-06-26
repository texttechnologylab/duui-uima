# GNFinder
A DUUI pipeline for the use of [GNfinder](https://github.com/gnames/gnfinder).

[![Version](https://img.shields.io/static/v1?label=ttlabdocker_version&message=latest&color=blue)]()



# HoToUse
For using GNfinder as a DUUI image it is necessary to use the Docker Unified UIMA Interface.

## Use as Stand-Alone-Image
```bash
docker run docker.texttechnologylab.org/duui-gnfinder:latest
```

## Run with a specific port
```bash
docker run -p 1000:9714 docker.texttechnologylab.org/duui-gnfinder:latest
```

## Run within DUUI
```java
composer.add(new DUUIDockerDriver.
    Component("docker.texttechnologylab.org/duui-gnfinder:latest")
    .withScale(iWorkers)
    .withImageFetching());
```

## Existing Parameters

| Parameter | Description | Datatype | Default | Example |
| --- | --- | --- | --- | --- |
| adjustOdds | Adjust Bayes odds using density of found names. | Boolean | False |  |
| allMatches | Verification returns all found matches. | Boolean | False |  |
| ambiguousUninomials | Preserve uninomials that are also common words. | Boolean | False |  |
| lang | Text's language or 'detect' for automatic detection. | String | detect | "eng", "de" |
| noBayes | Do not run Bayes algorithms. | Boolean | False |  |
| sources | IDs of important data-sources to verify against. If sources are set and there are matches to their data, such matches are returned in "preferred-result" results. To find IDs refer to "https://resolver.globalnames.org/data_sources". | String |  | "1,11" |
| uniqueNames | Return unique names list. | Boolean | False | |
| verify | Verify found name-strings. | Boolean | True |  |



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
  title          = {GNfinder as DUUI-Komponent},
  year           = {2022},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/edit/main/duui-GNFinder}
}

```
