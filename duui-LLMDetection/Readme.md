[![Version](https://img.shields.io/static/v1?label=Python&message=3.12&color=green)]()
[![Version](https://img.shields.io/static/v1?label=Transformers&message=4.53.0&color=yellow)]()
[![Version](https://img.shields.io/static/v1?label=Torch&message=2.7.1&color=red)]()

# LLM Detection Models and Metrics as DUUI Components

DUUI implementation for selected tools to detect LLM-generated text and to compute metrics for LLM detection.
## Included Tools

| Name       | Model-Name                 | Source                                                                            | Revision                                                                        | Languages |
|------------|----------------------------|-----------------------------------------------------------------------------------|---------------------------------------------------------------------------------|-----------|
| Radar      | radar                      | https://huggingface.co/TrustSafeAI/RADAR-Vicuna-7B                                | 4ff1f23a69a36aa1df47b0933be6279f1b896c9b                                        | MULTI     |
| HelloSimpleAI | hello-simple-ai            | https://huggingface.co/Hello-SimpleAI/chatgpt-detector-roberta                    | d2b342c61775d5dd0221808a79983ed3b86ffd86                                        | EN        |
| E5LoRA     | e5-lora                    | https://huggingface.co/MayZhou/e5-small-lora-ai-generated-detector                | 2c7b0f1d8e4a3f6c9b8d3c4e2f1a6b7c8d9e0f1                                         | EN        |
| Binocular  | binocular-falcon3-1b       | https://huggingface.co/tiiuae/Falcon3-1B-Base , https://huggingface.co/tiiuae/Falcon3-1B-Instrcut | cb37ef3559b157b5c9d9226296ba01a5162da1f7,28ba2251970a01dd1edc7ba7dad2eb71216ccfdf | Multi     |
| DetectLLM-LRR | detectllm-lrr-gpt2         | https://huggingface.co/openai-community/gpt2                                      | 607a30d783dfa663caf39e06633721c8d4cfcd7e                                        | Multi     |
| Fast-DetectGPT | fast-detectgpt-gpt2        | https://huggingface.co/openai-community/gpt2                                      | 607a30d783dfa663caf39e06633721c8d4cfcd7e                                        | Multi     |
| Fast-DetectGPTwithScoring | fast-detectgpt-dif-gpt-neo | https://huggingface.co/EleutherAI/gpt-neo-1.3B,https://huggingface.co/EleutherAI/gpt-neo-125m | dbe59a7f4a88d01d1ba9798d78dbe3fe038792c8,21def0189f5705e2521767faed922f1f15e7d7db | Multi     |
| MachineTextDetector | machine-text-detector      | https://huggingface.co/GeorgeDrayson/modernbert-ai-detection                      | 08f218f1d05791ad99c26ede421f69c781a50360                                        | EN        |
| AIGCDetectorEn | aigc-detector-en           | https://huggingface.co/yuchuantian/AIGC_detector_env2                             | d67ec874221b33c7abb2c9a78019cb08f10a1da1                                        | EN        |
| AIGCDetectorZh | aigc-detector-zh           | https://huggingface.co/yuchuantian/AIGC_detector_zhv2                             | 547d42d7ed6423edce77296b4b06d375ae1a7e0f                                        | ZH        |
| SuperAnnotate | superannotate-ai           | https://huggingface.co/SuperAnnotate/ai-detector                                  | 74b2b8580915c202607c09f64f8170eaa87a6a14                                        | EN        |
| FakeSpotAI | fakespotai                 | https://huggingface.co/fakespot-ai/roberta-base-ai-text-detection-v1 | f9cdb14d1f8b105f597d80fa7b56f20c6ea0e9db                                        | EN        |
| Desklib | desklib                    | https://huggingface.co/desklib/ai-text-detector-v1.01 | 2bf0bfc06f980531bc49aa70fa06034febc85d5b                                        | EN        |
| Mage | mage                       | https://huggingface.co/yaful/MAGE | 0d82ca0fdf6ebef5babb813cc11bd8eb2552c846                                        | EN        |
| PHDScore | phdscore-xlm-roberta       | https://huggingface.co/FacebookAI/xlm-roberta-base | e73636d4f797dec63c3081bb6ed5c7b0bb3f2089                                        | Multi     |
| HC3AIDetect | hc3ai-detect               | https://huggingface.co/VSAsteroid/ai-text-detector-hc3 | 8c0676c4299a3d8a95fcfc8dcc3fda48be363944                                        | EN        |
| ArguGPTSentence | argugpt-sentence           | https://huggingface.co/SJTU-CL/RoBERTa-large-ArguGPT-sent | 08bb22175be075e8afce8a3d1693c59623afbd89                                        | EN        |
| ArguGPTDocument | argugpt-document           | https://huggingface.co/SJTU-CL/RoBERTa-large-ArguGPT | 257b89e4036c26ad128a108a20ce5960f68f4f20                                        | EN        |
| DetectAIve | detectaive                 | https://huggingface.co/raj-tomar001/LLM-DetectAIve_deberta-base | 04292cbfb741f015029813ab9003c5f154b38fb1                                        | EN        |
| AIDetectModel | aidetect-model             | https://huggingface.co/wangkevin02/AI_Detect_Model | dc66ed399a3b34ba2ac625504c799142d5cca333                                        | EN        |
| LogRank | logrank-gpt2-medium        | https://huggingface.co/openai-community/gpt2-medium | 6dcaa7a952f72f9298047fd5137cd6e4f05f41da                                        | MULTI     |
| Wild | wild-longformer            | https://huggingface.co/nealcly/detection-longformer | 61d3aab2e0e5afcce6cb00b92661b0f9ee18ac2b | EN |
| T5Sentinel | t5-sentinel                | https://github.com/MarkChenYutian/T5-Sentinel-public | cd600a0c577fb73592eacedd7cd5aae2e0950e5e | EN |
| OpenAIDetector | openai-detector            | https://huggingface.co/openai-community/roberta-large-openai-detector | 38f3e0ccf205e9c4e0b38ae9d75ec948141bf832 | EN |
| PirateXXAIDetector | piratexx-ai-detector | https://huggingface.co/PirateXX/AI-Content-Detector | 0d5450c5baf7c6ccab387a120bcc9c2a4d2d3d9c | EN |


# How To Use

For using duui-llmdetection as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm -p 1000:9714 docker.texttechnologylab.org/duui-llmdetection-[model-name]:latest
```

Find all available image tags here: https://docker.texttechnologylab.org/v2/duui-llmdetection-[model-name]/tags/list

## Run within DUUI

```
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-llmdetection-[toolname]:latest")
        .withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
        .withScale(iWorkers)
        .withImageFetching()
);
```

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

@misc{Bagci:2024,
  author         = {Bagci, Mevlüt},
  title          = {Text LLM Detection Metrics and Models as {DUUI} component},
  year           = {2025},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-LLMDetection},
}

```
