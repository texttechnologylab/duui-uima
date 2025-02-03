[![Version](https://img.shields.io/static/v1?label=duui-transformers-sentiment&message=0.1.2&color=blue)](https://docker.texttechnologylab.org/v2/duui-pdf-extract-ocr/tags/list)
[![Version](https://img.shields.io/static/v1?label=Python&message=3.12&color=green)]()

# DUUI PDF Extract & OCR

DUUI implementation of PDF extraction and OCR of [GerParCor](https://github.com/texttechnologylab/GerParCor) using textract and pytesseract.

## Supported Languages

| Language | Tesseract |
|----------|----------|
| de       | deu      |
| eu       | eng      |


# How To Use

For using duui-pdf-extract-ocr as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm -p 9714:9714 docker.texttechnologylab.org/duui-pdf-extract-ocr:latest
```

Find all available image tags here: https://docker.texttechnologylab.org/v2/duui-pdf-extract-ocr/tags/list

## Run within DUUI

```
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-pdf-extract-ocr:latest")
        //.withParameter("min_chars", String.valueOf(100))
        //.withParameter("ocr_dpi", String.valueOf(200))
        //.withParameter("ocr_preprocess", String.valueOf(false))
        .withTargetView("text")
        .withScale(iWorkers)
        .withImageFetching()
);
```

### Parameters

| Name | Description                                                                                  |
| ---- |----------------------------------------------------------------------------------------------|
| `min_chars` | Minimal amount of characters to assume successful text extraction from PDF, default is `100` |
| `ocr_dpi`  | DPI used for converting PDF into image for OCR, default is `200`                             |
| `ocr_preprocess` | Preprocess image to improve OCR, default is `False`                                           |

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

@misc{Baumartz:2022,
  author         = {Baumartz, Daniel},
  title          = {GerParCor PDF Extract & OCR using textract and pytesseract as DUUI component},
  year           = {2025},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-pdf-extract-ocr}
}

@InProceedings{Abrami:Bagci:Hammerla:Mehler:2022,
  author         = {Abrami, Giuseppe and Bagci, Mevl\"{u}t and Hammerla, Leon and Mehler, Alexander},
  title          = {German Parliamentary Corpus (GerParCor)},
  booktitle      = {Proceedings of the Language Resources and Evaluation Conference},
  month          = {June},
  year           = {2022},
  address        = {Marseille, France},
  publisher      = {European Language Resources Association},
  pages          = {1900--1906},
  url            = {https://aclanthology.org/2022.lrec-1.202}
}

@inproceedings{Abrami:et:al:2024,
    address   = {Torino, Italy},
    author    = {Abrami, Giuseppe and Bagci, Mevl{\"u}t and Mehler, Alexander},
    booktitle = {Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
    editor    = {Calzolari, Nicoletta and Kan, Min-Yen and Hoste, Veronique and Lenci, Alessandro and Sakti, Sakriani and Xue, Nianwen},
    month     = {may},
    pages     = {7707--7716},
    publisher = {ELRA and ICCL},
    title     = {{G}erman Parliamentary Corpus ({G}er{P}ar{C}or) Reloaded},
    url       = {https://aclanthology.org/2024.lrec-main.681},
    year      = {2024}
}

```

