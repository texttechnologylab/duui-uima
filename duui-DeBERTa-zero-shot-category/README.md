# DeBerta-v3 Zero Shot Category Parser

The parser uses the following model:
https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli

To use this parser, you can add the docker image like this as a DUUI Component.
```java
composer.add(new DUUIDockerDriver.Component("docker.texttechnologylab.org/deberta-zero-shot-category:latest")
        .withScale(iWorkers)
        .withParameter("labels", labels)
        .withParameter("selection", Sentence.class.getName().toString())
        .build());
```
The parameter "labels" is **required** and should be a string containing a comma-separated list of your labels.
```java
public static final String labels = "Tiere,Pflanzen,Technologie,Geschichte,Kunst,Musik,Politik,Bildung,Sport,Gesundheit,Reisen,Essen und Trinken,Filme,Literatur,Umwelt,Wissenschaft,Mode,Philosophie,Psychologie,Wirtschaft";
```
The parameter "multiLabel" ("true" or "false") can be used to disable multi_label (true by default).

The parameter "selection" (Type) can be used create annotations for each individual text section covered by the specified type.

The parameter "clearGpuCacheAfter" (int) can be used to specify after how many annotated text sections (see selection parameter above) the GPU cache should be cleared. 500 by default.

The JCas will be annotated with a CategoryCoveredTagged Annotation for each label, where the value represents the label's value and the score represents the score.
```java
for(CategoryCoveredTagged categoryCoveredTagged: JCasUtil.select(jCas, CategoryCoveredTagged.class)){
    System.out.println(categoryCoveredTagged.getValue() + ": " + categoryCoveredTagged.getScore());
}
```
You can find a complete example in src/test/java/ZeroShotTest.java.

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

```
