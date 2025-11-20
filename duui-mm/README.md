
[![Version](https://img.shields.io/static/v1?label=duui-multimodal\&message=0.1.0\&color=blue)](https://docker.texttechnologylab.org/v2/duui-multimodal/tags/list)
[![Python](https://img.shields.io/static/v1?label=Python\&message=3.12\&color=green)]()
[![Transformers](https://img.shields.io/static/v1?label=Transformers\&message=4.48.2\&color=yellow)]()
[![Torch](https://img.shields.io/static/v1?label=Torch\&message=2.6.0\&color=red)]()

# DUUI Multimodal Component

DUUI implementation for **multimodal Hugging Face models** that support combinations of:

* Text
* Image
* Audio
* Video (via uniform frame sampling and audio extraction using `ffmpeg`)

Supported models include variants like `microsoft/Phi-4-multimodal-instruct`.

---


##  Supported Models

| Model Name                                 | Source                                                                                           | Dockerimage                               | Mode  | Lang                            | Version     |
|--------------------------------------------|--------------------------------------------------------------------------------------------------|-------------------------------------------|-------|---------------------------------|-------------|
| `vllm/microsoft/Phi-4-multimodal-instruct` | 🤗 [Phi-4](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)                           | `duui-mutlimodality-vllm:latest`          | VLLM  | multi                           | `0af439b...` |
| `microsoft/Phi-4-multimodal-instruct`      | 🤗 [Phi-4](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)                           | `duui-mutlimodality:latest`               | VLLM  | multi                           | `0af439b...` |
| `vllm/Qwen/Qwen2.5-VL-7B-Instruct`         | 🤗 [Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)                           | `duui-mutlimodality-qwen-7b-vllm:latest`  | VLLM  | multi                           | `cc59489...` |
| `vllm/Qwen/Qwen2.5-VL-3B-Instruct`         | 🤗 [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)                           | `duui-mutlimodality-qwen-3b-vllm:latest`  | VLLM  | multi                           | `cc59489...` |
| `vllm/Qwen/Qwen2.5-VL-32B-Instruct`        | 🤗 [Qwen2.5-VL-32B](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct)                         | `duui-mutlimodality-qwen-32b-vllm:latest` | VLLM  | multi                           | `cc59489...` |
| `vllm/Qwen/Qwen2.5-VL-72B-Instruct`        | 🤗 [Qwen2.5-VL-72B](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct)                         | `duui-mutlimodality-qwen-72b-vllm:latest` | VLLM  | multi                           | `cc59489...` |
| `Qwen/Qwen2.5-VL-7B-Instruct`              | 🤗  [Qwen/Qwen2.5-VL-7B-Instruct] (https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)           | `duui-mutlimodality-transformer:latest`|       | Transformer                     | multi       | `cc59489...` |
| `Qwen/Qwen2.5-VL-7B-Instruct-AWQ`          | 🤗  [Qwen/Qwen2.5-VL-7B-Instruct-AWQ] (https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-AWQ)   | `duui-mutlimodality-transformer:latest`|              | Transformer                     | multi       | `536a357...` |
| `Qwen/Qwen2.5-VL-3B-Instruct`              | 🤗  [Qwen/Qwen2.5-VL-3B-Instruct] (https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)           | `duui-mutlimodality-transformer:latest`|      | Transformer                     | multi       | `6628554...` |
| `Qwen/Qwen2.5-VL-3B-Instruct-AWQ`          | 🤗   [Qwen/Qwen2.5-VL-3B-Instruct-AWQ] (https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-AWQ)  | `duui-mutlimodality-transformer:latest`|                 | Transformer                     | multi       | `e7b6239...` |
| `Qwen/Qwen2.5-VL-32B-Instruct`             | 🤗  [Qwen/Qwen2.5-VL-32B-Instruct] (https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct)         | `duui-mutlimodality-transformer:latest`|              | Transformer                     | multi       | `7cfb30d...` |
| `Qwen/Qwen2.5-VL-32B-Instruct-AWQ`         | 🤗   [Qwen/Qwen2.5-VL-32B-Instruct-AWQ] (https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct-AWQ) | `duui-mutlimodality-transformer:latest`|                   | Transformer                     | multi       | `66c370b...` |
| `Qwen/Qwen2.5-VL-72B-Instruct`             | 🤗    [Qwen/Qwen2.5-VL72B-Instruct] (https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct)        | `duui-mutlimodality-transformer:latest`|        | Transformer                     | multi       | `cd3b627...` |
| `Qwen/Qwen2.5-VL-72B-Instruct-AWQ`         | 🤗   [Qwen/Qwen2.5-VL-72B-Instruct-AWQ] (https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ) | `duui-mutlimodality-transformer:latest`|           | Transformer                     | multi       | `c8b87d4...` |
| `Qwen/Qwen2.5-Omni-3B`                     | 🤗    [Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B)                             | `duui-mutlimodality-transformer:latest`|                 | Transformer                     | multi       | `latest`     |
| `Qwen/Qwen3-32B`                           | 🤗 [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)                                            | `duui-mutlimodality-transformer:latest`|                 | Transformer  **with reasoning** | multi       | `latest`     |
| `Qwen/Qwen3-14B`                           | 🤗  [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B)                                           | `duui-mutlimodality-transformer:latest`|                 | Transformer  **with reasoning**                   | multi       | `latest`     |
| `Qwen/Qwen3-8B`                            | 🤗 [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)                                              | `duui-mutlimodality-transformer:latest`|                 | Transformer  **with reasoning**                   | multi       | `latest`     |
| `Qwen/Qwen3-4B`                            | 🤗 [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)                                              | `duui-mutlimodality-transformer:latest`|                 | Transformer  **with reasoning**                   | multi       | `latest`     |
| `Qwen/Qwen3-0.6B`                          | 🤗 [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)                                          | `duui-mutlimodality-transformer:latest`|                 | Transformer  **with reasoning**                   | multi       | `latest`     |

---

## Supported Modes

| Mode    | Description                                                         |
|---------|---------------------------------------------------------------------|
| `text`  | Process raw text prompts                                            |
| `image` | Process images and prompt combinations                              |
| `frames` | Process sequences of image frames with a shared prompt              |
| `audio` | Process audio files with accompanying text prompts                  |
| `video` | Process video input: extracts frames (every 5th), audio, and prompt |
| `frames_and_audio` | process **separate** frames and audio (provide them explicitly)     |


---

## How To Use

Requires the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

### Start Docker Container

```bash
docker run -p 9714:9714 docker.texttechnologylab.org/duui-mutlimodality-transformer
```

Find available image tags: [Docker Registry](https://docker.texttechnologylab.org/v2/duui-mutlimodality-transformer/tags/list)

---

## Use within DUUI

### VLLM models
```java
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-mutlimodality-vllm:latest")
        .withParameter("model_name", "microsoft/Phi-4-multimodal-instruct")
        .withParameter("mode", "video")  // Can be: text_only, image_only, audio, frames_only, video
);
```
### Transformer Models

```java


public void testText() throws Exception {
    composer.add(
            new DUUIRemoteDriver.Component("docker.texttechnologylab.org/duui-mutlimodality-transformer:latest")
                    .withParameter("model_name", "Qwen/Qwen3-1.7B")
                    .withParameter("mode", "text") // Can be: text_only, image_only, audio, frames_only, video
                    .build().withTimeout(1000) 

    );
            // writing you data in xmi format
            composer.add(new DUUIUIMADriver.Component(createEngineDescription(XmiWriter.class,
                    XmiWriter.PARAM_TARGET_LOCATION, sOutputPath,
                    XmiWriter.PARAM_PRETTY_PRINT, true,
                    XmiWriter.PARAM_OVERWRITE, true,
                    XmiWriter.PARAM_VERSION, "1.1"
            )).build());

        List<String> prompts = Arrays.asList(
                "Who is the current president of the USA?",
                "Is Frankfurt the capital of EU finance?"
        );
    
        createCas("en", prompts);


        composer.run(cas);
        
        // extract your results in Result class
        
        for(Result  result : JCasUtil.select(cas, Result.class)){
            String meta = result.getMeta();
            try{
                JSONObject obj = new JSONObject(meta);
                String response = obj.getString("response");
                String thinking_content = obj.getString("thinking_content"); // qwen3 only support this
                System.out.println(response);
                System.out.println(thinking_content);
            }
            catch(Exception e){
                System.out.println(e.getMessage());
            }
    
        }

}

```

---

## Parameters

| Name         | Description                                    |
| ------------ | ---------------------------------------------- |
| `model_name` | Name of the multimodal model to use            |
| `mode`       | Processing mode: text\_only, image\_only, etc. |
| `prompt`     | Prompt passed alongside media inputs           |

---

## Cite

If you want to use the DUUI image, please cite the following:

**Leonhardt et al. (2023)**
*"Unlocking the Heterogeneous Landscape of Big Data NLP with DUUI."*
Findings of the Association for Computational Linguistics: EMNLP 2023, 385–399.
\[[LINK](https://aclanthology.org/2023.findings-emnlp.29)] \[[PDF](https://aclanthology.org/2023.findings-emnlp.29.pdf)]

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

@misc{abusaleh:2025,
  author         = {Abusaleh, Ali},
  title          = {Multimodal Inference as {DUUI} Component},
  year           = {2025},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-mm}
}


