# GNFinder
A DUUI pipeline for the use of [GNfinder](https://github.com/gnames/gnfinder).

[![Version](https://img.shields.io/static/v1?label=ttlabdocker_version&message=latest&color=blue)]()



# HoToUse
For using GNfinder as a DUUI image it is necessary to use the Docker Unified UIMA Interface.

## Use as Stand-Alone-Image
```bash
docker run docker.texttechnologylab.org/gnfinder:latest
```

## Run with a specific port
```bash
docker run -p 1000:9714 docker.texttechnologylab.org/gnfinder:latest
```

## Run within DUUI
```java
composer.add(new DUUIDockerDriver.
    Component("docker.texttechnologylab.org/gnfinder:latest")
    .withScale(iWorkers)
    .withImageFetching());
```


# Cite
If you want to use the DUUI image please quote this as follows:

TODO

# BibTeX
```
@InProceedings{Leonhardt:Abrami:Mehler:2022,
  author         = {Leonhardt, Alexander and Abrami, Giuseppe and Mehler, Alexander},
  title          = {TODO},
  booktitle      = {},
  year           = {2022}
}

```
