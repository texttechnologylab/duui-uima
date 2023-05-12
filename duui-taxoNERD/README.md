[![Version](https://img.shields.io/static/v1?label=taxonerd&message=0.1&color=blue)]()
[![Version](https://img.shields.io/pypi/v/taxonerd)]()

# taxoNERD
A DUUI pipeline for the use of [taxoNERD](https://github.com/nleguillarme/taxonerd).

# HoToUse
For using taxoNERD as a DUUI image it is necessary to use the Docker Unified UIMA Interface.

## Use as Stand-Alone-Image
```
docker run docker.texttechnologylab.org/taxonerd_md:0.1
```

## Run with a specific port
```
docker run -p 1000:9714 docker.texttechnologylab.org/taxonerd_md:0.1
```

## Run within DUUI
```
composer.add(new DUUIDockerDriver.
    Component("docker.texttechnologylab.org/taxonerd_md:0.1")
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

