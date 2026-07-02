# Trankit

This is the Trankit CUDA-Container (it can be used with CPU and GPU): Trankit is a pretrained-transformer based annotation pipeline similar to spacy or stanza. 

## 1. Annotations
  The following is a list of Annotations, that are needed as Input for the Docker-Image and are returned as Output by the Docker-Image (if sentences are already annotated in the cas, they are used for furhter annotations):
  - ### Input:
    - Optional: de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence
  - ### Output:
    - Optional: de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence
    - de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency
    - de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity
    - de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token
    - de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.morph.MorphologicalFeatures
    - de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS
    - de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma

## 2. Docker-Image-Versions
  There is only one Docker-Image for all languages. because of the multilingual nature of trankit itself. The default multilingual Transformer-Model is: xlm-roberta-base
  
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
