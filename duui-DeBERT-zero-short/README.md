# DeBert-v3 Zero Short Parser

The parser uses the following model:
https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli

To use this parser you can add the docker image like this as a DUUI Component.
```java
composer.add(new DUUIDockerDriver.Component("docker.texttechnologylab.org/german-sentiment-bert:latest")
                    .withScale(workers)
                    .build());
```
You can find a full example under src/test/java/GermanSentimentTest.java.

Note that the document you want to analyse **needs** to be annotated with the **de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence Annotation** Type.

