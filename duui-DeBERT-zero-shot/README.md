# DeBert-v3 Zero Shot Parser

The parser uses the following model:
https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli

To use this parser, you can add the docker image like this as a DUUI Component.
```java
composer.add(new DUUIDockerDriver.Component("docker.texttechnologylab.org/debert-zero-shot:latest")
        .withScale(iWorkers)
        .withParameter("labels", labels)
        .build());
```
The parameter "labels" is **required** and should be a string containing a comma-separated list of your labels.
```java
public static final String labels = "Tiere,Pflanzen,Technologie,Geschichte,Kunst,Musik,Politik,Bildung,Sport,Gesundheit,Reisen,Essen und Trinken,Filme,Literatur,Umwelt,Wissenschaft,Mode,Philosophie,Psychologie,Wirtschaft";
```
The JCas will be annotated with a CategoryCoveredTagged Annotation for each label, where the value represents the label's value and the score represents the score.
```java
for(CategoryCoveredTagged categoryCoveredTagged: JCasUtil.select(jCas, CategoryCoveredTagged.class)){
    System.out.println(categoryCoveredTagged.getValue() + ": " + categoryCoveredTagged.getScore());
}
```
You can find a complete example in src/test/java/ZeroShotTest.java.
