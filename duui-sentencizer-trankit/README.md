# Trankit Sentencizer

This is the Trankit CUDA-Container (it can be used with CPU and GPU): Trankit is a pretrained-transformer based annotation pipeline similar to spacy or stanza. 

## 1. Annotations
  The following is a list of Annotations, that are needed as Input for the Docker-Image and are returned as Output by the Docker-Image:
  - ### Input:
    - Raw text
  - ### Output:
    - de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence

## 2. Docker-Image-Versions
  There is only one Docker-Image for all languages. because of the multilingual nature of trankit itself. The default multilingual Transformer-Model is: xlm-roberta-base
  

