# xlm-roberta-base-language-detection

The parser uses the following modell:
https://huggingface.co/papluca/xlm-roberta-base-language-detection

To use this parser, you can add the docker image like this as a DUUI Component.
```java
composer.add(new DUUIDockerDriver.Component("docker.texttechnologylab.org/xlm-roberta-base-language-detection:latest")
        .withParameter("annotationClassPath", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
        .withParameter("top_k", "10")
        .withScale(iWorkers)
        .build());
```

The parser annotates the JCas with 'org.texttechnologylab.annotation.Language' annotations for each annotation of the type defined in the 'annotationClassPath' parameter.
If the text covered by this annotation is too large, the parser skips this annotation.
If you leave the 'withParameter' empty or don't declare it at all, it annotates the languages for the entire text.

If you want to limit the outputted language annotations to the top k languages with the highest score, you can use the top_k parameter.

You can find a complete example in src/test/java/LanguageDetectionTest.java
