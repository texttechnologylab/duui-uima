# Paraphrasing

This is the Paraphraser CUDA-Container (it can be used with CPU and GPU): A paraphrase is an alternative version of a given Input-Sentence i.e. a different sentence (different sentence means different words, structure etc.) with the same meaning.

## 1. Annotations
  The following is a list of Annotations, that are needed as Input for the Docker-Image and are returned as     Output by the Docker-Image:
  - ### Input:
    - "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
  - ### Output:
    - "org.texttechnologylab.annotation.Paraphrase"

## 2. Docker-Image-Versions
  There is a Dockerfile for a german and an english Image. So build (with the respective .sh script) the desired Image. Every Dockerfile/Image has a list of models (model_ids), that can be used with it:
  - ### English Models:
    - "tuner007/pegasus_paraphrase" : PegasusBase
    - "humarin/chatgpt_paraphraser_on_T5_base" : T5Base
    - "eugenesiow/bart-paraphrase": BartBase
    - "prithivida/parrot_paraphraser_on_T5": ParrotBase
  - ### German Models:
    - "Lelon/t5-german-paraphraser-small" : T5Base
    - "Lelon/t5-german-paraphraser-large" : T5Base

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
  author         = {Hammerla, Leon},
  title          = {Paraphrasing tool as DUUI component},
  year           = {2023},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-Paraphrase}
}

```
