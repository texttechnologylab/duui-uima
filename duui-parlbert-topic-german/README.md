# parlbert-topic-german

The parser uses the following model:
https://huggingface.co/chkla/parlbert-topic-german

# HowToUse
For using parlbert-topic-german as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

To use this parser, you can add the docker image like this as a DUUI Component.
```java
composer.add(new DUUIDockerDriver.Component("docker.texttechnologylab.org/parlbert-topic-german:latest")
        .withScale(iWorkers)
        .build());
```

The parser annotates the JCas with CategoryCoveredTagged annotations for the whole text and each Sentence annotation.
If a sentence or the whole text is too large, the parser skips the annotation for this sentence or the annotations for the whole text.

You can find a complete example in src/test/java/ParlbertTopicGermanTest.java


## Use as Stand-Alone-Image
```
docker run docker.texttechnologylab.org/parlbert-topic-german:latest
```

## Run with a specific port
```
docker run -p 1000:9714 docker.texttechnologylab.org/parlbert-topic-german:latest
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

@misc{Klebe:2022,
  author         = {Klebe, Max},
  title          = {DUUI-Component for parlbert-topic-german},
  year           = {2022},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/edit/main/duui-parlbert-topic-german}
}
```

