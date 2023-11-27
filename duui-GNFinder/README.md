# GNFinder
A DUUI pipeline for the use of [GNfinder](https://github.com/gnames/gnfinder).

[![Version](https://img.shields.io/static/v1?label=ttlabdocker_version&message=latest&color=blue)]()



# HoToUse
For using GNfinder as a DUUI image it is necessary to use the Docker Unified UIMA Interface.

## Use as Stand-Alone-Image
```bash
docker run docker.texttechnologylab.org/gnfinder:latest
```

## Run with a specific port
```bash
docker run -p 1000:9714 docker.texttechnologylab.org/gnfinder:latest
```

## Run within DUUI
```java
composer.add(new DUUIDockerDriver.
    Component("docker.texttechnologylab.org/gnfinder:latest")
    .withScale(iWorkers)
    .withImageFetching());
```


# Cite
If you want to use the DUUI image please quote this as follows:

A. Leonhardt, G. Abrami, D. Baumartz, and A. Mehler, “Unlocking the Heterogeneous Landscape of Big Data NLP with DUUI,” in Findings of the Association for Computational Linguistics: EMNLP 2023, 2023, pp. 1-15 


# BibTeX
```

@inproceedings{Leonhardt:et:al:2023,
  title = {Unlocking the Heterogeneous Landscape of Big Data {NLP} with {DUUI}},
  author = {Leonhardt, Alexander and Abrami, Giuseppe and Baumartz, Daniel and Mehler, Alexander},
  year = {2023},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2023},
  publisher = {Association for Computational Linguistics},
  pages = {1--15},
  note = {accepted}
}

@misc{Abrami:2022,
  author         = {Abrami, Giuseppe},
  title          = {GNfinder as DUUI-Komponent},
  year           = {2022},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/edit/main/duui-GNFinder}
}

```
