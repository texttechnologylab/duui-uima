
[![Version](https://img.shields.io/static/v1?label=duui-multimodal\&message=0.1.0\&color=blue)](https://docker.texttechnologylab.org/v2/duui-multimodal/tags/list)
[![Python](https://img.shields.io/static/v1?label=Python\&message=3.12\&color=green)]()
[![Transformers](https://img.shields.io/static/v1?label=Transformers\&message=4.48.2\&color=yellow)]()
[![Torch](https://img.shields.io/static/v1?label=Torch\&message=2.6.0\&color=red)]()

# DUUI Open-WebUI 

DUUI implementation for **multimodal Hugging Face models** that support combinations of:

* Text
* Image



---


##  Supported Models services

| Model Name | Source                                                                                           | Dockerimage | Mode       | Lang                            | Version |
|------------|--------------------------------------------------------------------------------------------------|-------------|------------|---------------------------------|---------|
| ollama     |  https://docs.ollama.com/api/openai-compatibility                        | NA          | image/text | multi                           | 0.0.1   |

---

## Supported Modes

| Mode    | Description                                                         |
|---------|---------------------------------------------------------------------|
| `text`  | Process raw text prompts                                            |
| `image` | Process images and prompt combinations                              |

---

## How To Use

Requires the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

### Start Docker Container

```bash
docker run -p 9714:9714 docker.texttechnologylab.org/duui-open-webui
```

Find available image tags: [Docker Registry](https://docker.texttechnologylab.org/v2/duui-mutlimodality-transformer/tags/list)

---

## Use within DUUI

### ollama setup
```java
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("model_name", "qwen2.5vl:3b")
                        .withParameter("mode", "image")
                        .withParameter("language", "en")
                        .withParameter("ollama_host", "localhost") // https:/llm.example
//                        .withParameter("ollama_port", "8080")
                        .withParameter("ollama_auth_token", "")
                        .withParameter("system_prompt", "")
                        .build().withTimeout(1000)
        );
```
### Transformer Models

```java
// Code before as it is.. 

List<String> prompts = Arrays.asList(
        "Who is the current president of the USA?",
        "Is Frankfurt the capital of EU finance?"
);

createCas("en", prompts);
        composer.run(cas);

verifyNoImages();

// Print results
        for (Result result : JCasUtil.select(cas, Result.class)) {
        System.out.println(result.getMeta());
        }
//
// Helper method to create CAS with prompts
public void createCas(String language, List<String> prompts) throws UIMAException {
    cas.setDocumentLanguage(language);
    StringBuilder sb = new StringBuilder();

    for (String messageText : prompts) {
        Prompt prompt = new Prompt(cas);
        prompt.setArgs("{}");

        Message message = new Message(cas);
        message.setRole("user");
        message.setContent(messageText);
        message.addToIndexes();

        FSArray messages = new FSArray(cas, 1);
        messages.set(0, message);
        prompt.setMessages(messages);
        prompt.addToIndexes();

        sb.append(messageText).append(" ");
    }

    inputView.setDocumentText(sb.toString().trim());
//        cas.setDocumentText(sb.toString().trim());
}
```



---

## Parameters

| Name         | Description                                            |
| ------------ |--------------------------------------------------------|
| `model_name` | Name of the multimodal model to use (inside ollama)    |
| `mode`       | Processing mode: text, image                           |
| `ollama_host`     | ollama host url                                        |
| `ollama_port`     | ollama port, default 8080                              |
| `ollama_auth_token`| ollama auth token if exists, default empty             |
| `system_prompt`    | System prompt for all prompts if needed, default empty |

---

## Cite

If you want to use the DUUI image, please cite the following:

**Leonhardt et al. (2023)**
*"Unlocking the Heterogeneous Landscape of Big Data NLP with DUUI."*
Findings of the Association for Computational Linguistics: EMNLP 2023, 385–399.
\[[LINK](https://aclanthology.org/2023.findings-emnlp.29)] \[[PDF](https://aclanthology.org/2023.findings-emnlp.29.pdf)]

**Abusaleh (2026)**
*"OpenWebUI wrapper as {DUUI} Component"*
\[[LINK](https://github.com/texttechnologylab/duui-uima/tree/main/duui-open-webui])]
---

## BibTeX

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

@misc{abusaleh:duui:openwebui:2026,
  author         = {Abusaleh, Ali},
  title          = {OpenWebUI Ollama wrapper as {DUUI} Component},
  year           = {2026},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-open-webui}
}


