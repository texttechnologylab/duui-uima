# LLM Inference via Ollama

DUUI component for generic LLM inference using [Ollama](https://ollama.ai/) as the backend. Accepts pre-built `Prompt` annotations from the CAS, sends their messages to an Ollama-hosted model, and writes responses back into `FillableMessage` content and `Result` annotations. The model is specified at runtime — any model available on the connected Ollama server can be used.

Note: This is a generalized component based on [duui-core-llm-rating](https://github.com/texttechnologylab/duui-uima/tree/main/duui-core-llm-rating).

## Prerequisites

This component does **not** run the LLM itself. It requires an external [Ollama](https://ollama.ai/) server with the desired model already pulled and running. Configure the server address and model name via the `llm_args` parameter.

## Supported Models

Any model served by the connected Ollama instance is supported. Examples:

| Model | Description |
|---|---|
| `llama3.2` | Meta Llama 3.2 (general purpose) |
| `gemma2:27b` | Google Gemma 2 27B |
| `mistral` | Mistral 7B |
| Custom / fine-tuned | Any model registered in the Ollama registry |

# How To Use

For using duui-ollama-langchain as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm -p 1000:9714 docker.texttechnologylab.org/v2/duui-ollama-langchain:latest
```

Find all available image tags here: https://docker.texttechnologylab.org/v2/duui-ollama-langchain/tags/list

## Run within DUUI

```java
JSONObject llmArgs = new JSONObject();
llmArgs.put("base_url", "localhost:11434");
llmArgs.put("model", "llama3.2");
llmArgs.put("temperature", 0);
llmArgs.put("num_ctx", 2048);

composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/v2/duui-ollama-langchain:latest")
        .withParameter("llm_args", llmArgs.toString())
);
```

### Parameters

| Name | Description |
|---|---|
| `llm_args` | JSON string passed directly to `ChatOllama`. Required key: `model`. See table below for all supported keys. |

### `llm_args` keys

| Key | Description | Default |
|---|---|---|
| `model` | Ollama model name, e.g. `"llama3.2"` or `"gemma2:27b"` | — (required) |
| `base_url` | Ollama server URL, e.g. `"localhost:11434"` | Ollama default |
| `temperature` | Sampling temperature (0 = deterministic) | `0.8` |
| `num_ctx` | Context window size in tokens | Model default |
| `num_predict` | Maximum tokens to generate (`-2` = fill context) | `-1` |
| `seed` | Random seed for reproducible outputs | random |
| `keep_alive` | Seconds to keep the model loaded in memory | `300` |

Any additional key accepted by [LangChain's `ChatOllama`](https://python.langchain.com/docs/integrations/chat/ollama/) can be passed here.

### Input types

The CAS must contain at least one `Prompt` annotation before running this component:

| Type | Description |
|---|---|
| `org.texttechnologylab.type.llm.prompt.Prompt` | Contains an ordered list of messages and a JSON args string for template placeholders |
| `org.texttechnologylab.type.llm.prompt.Message` | A single chat message with `role` (system/user/assistant) and `content` |
| `org.texttechnologylab.type.llm.prompt.FillableMessage` | Subtype of `Message` with empty content — the LLM fills this slot. Optional `contextName` makes the LLM response available as a template variable in subsequent messages. |

### Output types

| Type | Description |
|---|---|
| `org.texttechnologylab.type.llm.prompt.Result` | Created for each filled `FillableMessage`. Stores JSON metadata (timing, model args) and references the originating `Prompt` and `Message`. |

The `FillableMessage.content` field is also updated in-place with the LLM's response (JSON-encoded string).

# Cite

If you want to use the DUUI image please quote this as follows:

Alexander Leonhardt, Giuseppe Abrami, Daniel Baumartz and Alexander Mehler. (2023). "Unlocking the Heterogeneous Landscape of Big Data NLP with DUUI." Findings of the Association for Computational Linguistics: EMNLP 2023, 385–399. [[LINK](https://aclanthology.org/2023.findings-emnlp.29)] [[PDF](https://aclanthology.org/2023.findings-emnlp.29.pdf)]

## BibTeX

```
@inproceedings{Leonhardt:et:al:2023,
  title     = {Unlocking the Heterogeneous Landscape of Big Data {NLP} with {DUUI}},
  author    = {Leonhardt, Alexander and Abrami, Giuseppe and Baumartz, Daniel and Mehler, Alexander},
  editor    = {Bouamor, Houda and Pino, Juan and Bali, Kalika},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2023},
  year      = {2023},
  address   = {Singapore},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2023.findings-emnlp.29},
  pages     = {385--399},
  pdf       = {https://aclanthology.org/2023.findings-emnlp.29.pdf},
  abstract  = {Automatic analysis of large corpora is a complex task, especially
               in terms of time efficiency. This complexity is increased by the
               fact that flexible, extensible text analysis requires the continuous
               integration of ever new tools. Since there are no adequate frameworks
               for these purposes in the field of NLP, and especially in the
               context of UIMA, that are not outdated or unusable for security
               reasons, we present a new approach to address the latter task:
               Docker Unified UIMA Interface (DUUI), a scalable, flexible, lightweight,
               and feature-rich framework for automatic distributed analysis
               of text corpora that leverages Big Data experience and virtualization
               with Docker. We evaluate DUUI{'}s communication approach against
               a state-of-the-art approach and demonstrate its outstanding behavior
               in terms of time efficiency, enabling the analysis of big text
               data.}
}

@misc{duui-ollama-langchain,
  author         = {Baumartz, Daniel},
  title          = {LLM Inference via Ollama as {DUUI} component},
  year           = {2025},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-ollama-langchain}
}

```
