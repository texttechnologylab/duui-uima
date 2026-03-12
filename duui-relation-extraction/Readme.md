# Relation Extraction

DUUI implementation for selected Hugging-Face-based transformer [relation-extraction tools](https://huggingface.co/models?other=relation-extraction) models.

## Included Models

| Name                                   | link                                                                             | Revision                                  | Languages         |
|----------------------------------------|----------------------------------------------------------------------------------|-------------------------------------------|-------------------|
| Babelscape/rebel-large  | https://huggingface.co/Babelscape/rebel-large       | 44eb6cb4585df284ce6c4d6a7013f83fe473c052  | Multilingual      |
| ibm-research/knowgl-large | https://huggingface.co/ibm-research/knowgl-large | 94596fd9f697498f7ee7363dbf4cc66f08d499e8 | Multilingual |

## Execution

1. Im Ordner 'models' das gewünschte Modell runterladen, da der Container nicht mit dem Internet verbunden ist.
2. Docker bauen: docker build -t [IMAGE_NAME] .
3. Docker starten: docker run -p 8000:8000 -e MODEL_NAME=[MODEL_NAME] [IMAGE_NAME]