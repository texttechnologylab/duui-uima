# Universal Dependency Parser
A DUUI pipeline intergration.

[![Version](https://img.shields.io/static/v1?label=ttlabdocker_version&message=latest&color=blue)]()



# HoToUse
For using udepParser as a DUUI image it is necessary to use the Docker Unified UIMA Interface.

## Pre-processing requirements
Provided Cas Document must be pre-tokenized and split into sentences. 

## Use as Stand-Alone-Image
```sh
docker run docker.texttechnologylab.org/udepparser_cuda_1024:latest
```

## Run with a specific port (e.g. 1000)
```sh
docker run -p 1000:9714 docker.texttechnologylab.org/udepparser_cuda_1024:latest
```

## Run within DUUI using previously started docker container
```java
DUUIComposer composer = new DUUIComposer()
        .withLuaContext(
                new DUUILuaContext()
                        .withJsonLibrary()
        ).withSkipVerification(true);
DUUIRemoteDriver remote_driver = new DUUIRemoteDriver(10000);
composer.addDriver(remote_driver);
composer.add(
        new DUUIRemoteDriver.Component("http://127.0.0.1:1000")
);

composer.run(cas);
```
## Run within DUUI from repository
```java
DUUIComposer composer = new DUUIComposer()
        .withLuaContext(
                new DUUILuaContext()
                        .withJsonLibrary()
        ).withSkipVerification(true);

DUUIDockerDriver docker_driver = new DUUIDockerDriver();
composer.addDriver(docker_driver);
composer.add(new DUUIDockerDriver.
                Component("docker.texttechnologylab.org/udepparser_cuda_1024:latest")
                .withImageFetching()
);

composer.run(cas);
```

## Required UIMA input
```java

de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token
```
## UIMA output
```java
de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency
org.texttechnologylab.annotation.AnnotationPerspective
```

# Cite
If you want to use the DUUI image please quote this as follows:
TODO
