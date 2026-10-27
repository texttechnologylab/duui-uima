# Text Readability Analysis

DUUI component for computing standard readability metrics using [py-readability-metrics](https://github.com/cdimascio/py-readability-metrics). Takes the raw document text directly from the CAS with no prior annotations required, computes nine classical readability formulas in a single pass, and writes one `CategoryCoveredTagged` annotation per metric spanning the full document. Each annotation carries the numeric score and a JSON `tags` field with formula-specific detail.

## Included Tools

| Name                   | Source                                              | Revision | Languages |
|------------------------|-----------------------------------------------------|----------|-----------|
| py-readability-metrics | https://github.com/cdimascio/py-readability-metrics | 1.4.5    | EN        |

## Metrics

Every request computes all nine metrics. The metric name is stored in `value`, the result in `score`, and `tags` holds a JSON string with the score plus the formula-specific fields below.

| `value`          | Metric                       | `score`                                  | Additional fields in `tags`                                        |
|------------------|------------------------------|------------------------------------------|--------------------------------------------------------------------|
| `flesch_kincaid` | Flesch-Kincaid Grade Level   | US school grade                          | `grade_level`                                                      |
| `flesch`         | Flesch Reading Ease          | Usually 0 to 100, higher is easier       | `ease` (`very_easy`, `easy`, `fairly_easy`, `standard`, `fairly_difficult`, `difficult`, `very_confusing`), `grade_levels` |
| `gunning_fog`    | Gunning Fog Index            | US school grade                          | `grade_level` (`na` below grade 6, `college`, `college_graduate`)  |
| `coleman_liau`   | Coleman-Liau Index           | US school grade                          | `grade_level`                                                      |
| `dale_chall`     | Dale-Chall Readability Score | Score based on a list of familiar words  | `grade_levels`                                                     |
| `ari`            | Automated Readability Index  | US school grade                          | `grade_levels`, `ages` (reader age range)                          |
| `linsear_write`  | Linsear Write Formula        | US school grade                          | `grade_level`                                                      |
| `smog`           | SMOG Index                   | US school grade                          | `grade_level`                                                      |
| `spache`         | Spache Readability Formula   | US school grade, meant for young readers | `grade_level`                                                      |

### Requirements and limitations

- The text needs **at least 100 words**, otherwise every metric fails.
- SMOG additionally needs **at least 30 sentences**.
- The formulas, syllable counting and word lists (Dale-Chall, Spache) are designed for English. The document language is sent to the component, but it does not change how the metrics are computed, so values for other languages are not calibrated.
- A metric that cannot be computed does not stop the others. Its `tags` contain `{"error": "<reason>"}` instead of the detail fields, and its `score` is `null` in the REST response. In the CAS, `score` is a primitive `double`, so a failed metric shows up as `0.0`: check `tags` for an `error` key before using a score.

# How To Use

For using duui-readability as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm -p 9714:9714 docker.texttechnologylab.org/duui-readability:latest
```

Find all available image tags here: https://docker.texttechnologylab.org/v2/duui-readability/tags/list

## Run within DUUI

```java
DUUIComposer composer = new DUUIComposer()
        .withSkipVerification(true)
        .withLuaContext(new DUUILuaContext().withJsonLibrary());

composer.addDriver(new DUUIDockerDriver());

composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-readability:latest")
        .withScale(iWorkers)
        .withImageFetching()
);

JCas jCas = JCasFactory.createJCas();
jCas.setDocumentLanguage("en");
jCas.setDocumentText(text);  // at least 100 words

composer.run(jCas);

for (CategoryCoveredTagged metric : JCasUtil.select(jCas, CategoryCoveredTagged.class)) {
    System.out.println(metric.getValue() + " = " + metric.getScore() + " " + metric.getTags());
}
```

### Output types

| Type | Description |
|---|---|
| `org.hucompute.textimager.uima.type.category.CategoryCoveredTagged` | One annotation per readability metric, spanning the full document. `value` holds the metric name, `score` holds the numeric result, and `tags` holds a JSON string with formula-specific detail fields (see [Metrics](#metrics)). |
| `org.texttechnologylab.annotation.AnnotatorMetaData` | Created for each `CategoryCoveredTagged` annotation. Records the annotator name, version, model name (`py-readability-metrics`), and model version. |
| `org.texttechnologylab.annotation.DocumentModification` | Single annotation recording the annotator name, processing timestamp, and a version comment. |

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

@misc{duui-readability,
  author         = {Baumartz, Daniel},
  title          = {Text Readability Analysis as {DUUI} component},
  year           = {2025},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-readability}
}

```
