[![Version](https://img.shields.io/static/v1?label=taxonerd&message=0.1&color=blue)]()
[![Version](https://img.shields.io/pypi/v/taxonerd)]()

# taxoNERD
A DUUI pipeline for the use of [taxoNERD](https://github.com/nleguillarme/taxonerd).

# HowToUse
For using taxoNERD as a DUUI image it is necessary to use the Docker Unified UIMA Interface (DUUI).

## Use as Stand-Alone-Image
```
docker run docker.texttechnologylab.org/taxonerd_md:0.1
```

## Run with a specific port
```
docker run -p 1000:9714 docker.texttechnologylab.org/taxonerd_md:0.1
```

## Run within DUUI
```
composer.add(new DUUIDockerDriver.
    Component("docker.texttechnologylab.org/taxonerd_md:0.1")
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
  title          = {taxoNerd as DUUI-Komponent},
  year           = {2022},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-taxoNERD}
}

```

