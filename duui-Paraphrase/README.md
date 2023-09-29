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

