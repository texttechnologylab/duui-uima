# German Sentiment Bert DUUI

The parser uses the following model: 
https://huggingface.co/oliverguhr/german-sentiment-bert

To use this parser you can add the docker image like this as a DUUI Component.
```java
composer.add(new DUUIDockerDriver.Component("docker.texttechnologylab.org/german-sentiment-bert:latest")
                    .withScale(workers)
                    .build());
```
You can find a full example under src/test/java/GermanSentimentTest.java.

Note that the document you want to analyse **needs** to be annotated with the **de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence Annotation** Type.

