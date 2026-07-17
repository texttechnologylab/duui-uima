# Face-Anon-Simple
---
DUUI implementation for simple face anonymization, based on
https://github.com/hanweikung/face_anon_simple.
It supports:
- single face anonymization 
- multiple faces anonymization
- face redaction
  - blur
  - blackout
  - pixelate

## Parameters 


| required? | **Parameter Name**  | Default                            | Possible Values                                                | Explanation                                                         |
| --------- | ------------------- | ---------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------- |
| yes       | **anon_type**       | single_align                       | single_align<br>multiple_align<br>swap<br>redact               | choice between the different anonymization types                    |
| yes       | **hf_token**        | -                                  | personal hugging-face token                                    | needed for using the models                                         |
| no        | **anon_degree**     | 1.25                               | float                                                          | degree of anonymization <br>*(for the `<type>_align` `anon_type` )* |
| no        | **redact_type**     | blur                               | blur<br>black<br>pixel                                         | type of redaction<br>*(for redaction `anon_type)`*                  |
| no        | **blur**            | 51                                 | any **uneven** integer - higher values amount to stronger blur | amount of blur                                                      |
| no        | **pixel**           | 16                                 | any integer - higher values amount to smaller pixels           | degree of pixelation                                                |
| no        | **diffusion_model** | sd2-community/stable-diffusion-2-1 | hugging face link to any diffusion model, with UNet format     |                                                                     |
| no        | **clip_model**      | openai/clip-vit-large-patch14      | hugging face link to any clip model                            |                                                                     |
| no        | **seed**            | 1                                  | integer                                                        | seed for recreating anonymization                                   |
| no        | **guidance**        | 4.0                                | float                                                          | amount of guidance                                                  |
| no        | **inference_steps** | 25                                 | int                                                            | number of inference steps                                           |
| no        | **height**          | the passed images height           |                                                                | for resizing output                                                 |
| no        | **width**           | the passed images width            |                                                                | for resizing output                                                 |
| no        | **vis_input**       | False                              | Boolean                                                        | Displays input and output next to another in one image              |

## How To Use
Requires the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

### Start Docker Container

```bash
docker run -p 9714:9714 docker.texttechnologylab.org/duui-face-anon-simple
```
Find available image tags: [Docker Registry](https://docker.texttechnologylab.org/v2/duui-mutlimodality-transformer/tags/list)


## Use within DUUI
Examplary usage, see more examples in the `src/test/java/AnonTest.java` file.
There are also two test Images provided:
`multiple_people.jpg`: https://images.pexels.com/photos/10351367/pexels-photo-10351367.jpeg
`single_person.jpg:` https://images.pexels.com/photos/31430969/pexels-photo-31430969.jpeg
```java

    @Test
    public void testSingleFaceSimple() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component("http://127.0.0.1:8001")
                        .withParameter("anon_type", "single_align")
                        .withParameter("hf_token", hf_token) // the anonymization WILL fail, if no hugging face token is provided!
                        .withTargetView("output") // to easily iterate through the output images - save them in a seperate view
                        .build().withTimeout(1000)

        );

        createCas();
        composer.run(cas);
        readImagesInCas("single face");

    }
```

## BibTex
```bibtex
@inproceedings{Leonhardt:et:al:2023,
  title     = {Unlocking the Heterogeneous Landscape of Big Data {NLP} with {DUUI}},
  author    = {Leonhardt, Alexander and Abrami, Giuseppe and Baumartz, Daniel and Mehler, Alexander},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2023},
  year      = {2023},
  address   = {Singapore},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2023.findings-emnlp.29},
  pages     = {385--399},
  pdf       = {https://aclanthology.org/2023.findings-emnlp.29.pdf}
}

@misc{sittardt:2025,
  author         = {Sittardt, Coco},
  title          = {Simple Face Anonymization as {DUUI} Component},
  year           = {2026},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-face-anon}
}
@InProceedings{Kung_2025_WACV,
    author    = {Kung, Han-Wei and Varanka, Tuomas and Saha, Sanjay and Sim, Terence and Sebe, Nicu},
    title     = {Face Anonymization Made Simple},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {1040-1050}
}
```


