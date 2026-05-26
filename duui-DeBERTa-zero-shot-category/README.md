# DeBerta-v3 Zero Shot Category Parser

The parser uses the following model:
https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli

To use this parser, you can add the docker image like this as a DUUI Component.
```java
composer.add(new DUUIDockerDriver.Component("docker.texttechnologylab.org/deberta-zero-shot-category:latest")
        .withScale(iWorkers)
        .withParameter("labels", labels)
        .withParameter("selection", Sentence.class.getName().toString())
        .build());
```
The parameter "labels" is **required** and should be a string containing a comma-separated list of your labels.
```java
public static final String labels = "Tiere,Pflanzen,Technologie,Geschichte,Kunst,Musik,Politik,Bildung,Sport,Gesundheit,Reisen,Essen und Trinken,Filme,Literatur,Umwelt,Wissenschaft,Mode,Philosophie,Psychologie,Wirtschaft";
```
The parameter "multiLabel" ("true" or "false") can be used to disable multi_label (true by default).

The parameter "selection" (Type) can be used create annotations for each individual text section covered by the specified type.

The parameter "clearGpuCacheAfter" (int) can be used to specify after how many annotated text sections (see selection parameter above) the GPU cache should be cleared. 500 by default.

The JCas will be annotated with a CategoryCoveredTagged Annotation for each label, where the value represents the label's value and the score represents the score.
```java
for(CategoryCoveredTagged categoryCoveredTagged: JCasUtil.select(jCas, CategoryCoveredTagged.class)){
    System.out.println(categoryCoveredTagged.getValue() + ": " + categoryCoveredTagged.getScore());
}
```
You can find a complete example in src/test/java/ZeroShotTest.java.
