# Remove annotations

DUUI component that removes all annotations of given types from the CAS. By default only annotations of exactly the
given type are removed, not those of its subtypes: e.g. removing `de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token`
keeps the `SpacyToken`s (a subtype of `Token`), so that a later tool that selects `Token` gets only the `SpacyToken`s.

The removal is done in the Lua communication script; the service only checks the parameters. Every run adds a
`DocumentModification` with the number of removed annotations per type.

**Do not use with a DUUI segmentation strategy:** only new annotations are merged back from a segment, so removals
inside a segment are lost.

# How To Use

For using duui-remove-annotations as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm -p 1000:9714 docker.texttechnologylab.org/duui-remove-annotations:latest
```

## Run within DUUI

```java
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-remove-annotations:latest")
        .withParameter("types", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token")
        .withParameter("exact", "true")
);
```

### Parameters

| Name | Description | Default |
|---|---|---|
| `types` | Comma-separated full type names to remove (required); an unknown type is an error | |
| `exact` | `true`: only annotations of exactly these types; `false`: also of their subtypes | `true` |

### Input / output types

| | Types |
|---|---|
| Input | any (the types given in `types`) |
| Output | `org.texttechnologylab.annotation.DocumentModification` |

Note: annotations that reference a removed annotation (e.g. `AnnotatorMetaData`) are not removed.

## Tests

Start the service (e.g. `docker run --rm -p 9802:9714 docker.texttechnologylab.org/duui-remove-annotations:1.0.0`),
then `mvn test -Dremove.url=http://localhost:9802`.
