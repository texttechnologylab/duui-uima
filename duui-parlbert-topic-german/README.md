# parlbert-topic-german

The parser uses the following model:
https://huggingface.co/chkla/parlbert-topic-german

To use this parser, you can add the docker image like this as a DUUI Component.
```java
composer.add(new DUUIDockerDriver.Component("docker.texttechnologylab.org/parlbert-topic-german:latest")
        .withScale(iWorkers)
        .build());
```

The parser annotates the JCas with CategoryCoveredTagged annotations for the whole text and each Sentence annotation.
If a sentence or the whole text is too large, the parser skips the annotation for this sentence or the annotations for the whole text.

You can find a complete example in src/test/java/ParlbertTopicGermanTest.java
