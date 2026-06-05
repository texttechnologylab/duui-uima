# DUUI Neer Match

A DUUI pipeline for neural entity matching using deep learning models. This component provides functionality for
matching entities (such as Named Entities or Tokens) across multiple documents using configurable similarity matchers
and neural network models.

## Overview

DUUI Neer Match enables matching of entities across documents using deep learning models trained with various similarity
metrics (e.g., Levenshtein distance, Jaro-Winkler, Hamming distance). The system supports custom entity properties and
flexible model configurations.

## How To Use

For using DUUI Neer Match as a DUUI image, it is necessary to use the Docker Unified UIMA Interface.

For information on how to train new models, refer to [TRAINING.md](TRAINING.md).

### Use as Stand-Alone-Image

**Note:** Currently there exists no docker image.

Probable future usage as a stand-alone image:

```bash
docker run docker.texttechnologylab.org/duui-neer-match:latest
```

### Run with a specific port

**Note:** Currently there exists no docker image.

Probable future usage with a specific port:

```bash
docker run -p 1000:9714 docker.texttechnologylab.org/duui-neer-match:latest
```

### Run within DUUI

**Note:** Currently there exists no docker image.

Probable future usage within DUUI:

```java
composer.add(new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-neer-match:latest")
    .withScale(iWorkers)
    .withImageFetching());
```

## Pipeline Architecture

The DUUI Neer Match pipeline works in two phases:

### Phase 1: Process (`/v1/process`)

During the process phase, entities are extracted from documents and stored in a pipeline-specific storage. Each
document's entities are indexed for later retrieval.

**Request Body:**

```json
{
  "pipeline_id": "unique-pipeline-id",
  "selection": "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token",
  "entities": [
    {
      "text": "example",
      "entity_id": 0,
      "properties": {
        "lemma": "example",
        "pos": "NN"
      }
    }
  ]
}
```

**Response:**

```json
{
  "pipeline_id": "unique-pipeline-id",
  "stored_index": 0,
  "stored_count": 5
}
```

### Phase 2: Finalize (`/v1/finalize`)

The finalize phase performs the actual entity matching across all stored documents using the selected neural network
model.

**Note:** The finalize route is currently **not supported by DUUI**. Direct HTTP requests must be used for finalization.

**Request Body:**

```json
{
  "pipeline_id": "unique-pipeline-id",
  "clear_storage": true,
  "properties": {
    "model": "token_test1",
    "threshold": 0.5,
    "batch_size": 16
  }
}
```

## Finalize Route Output

The finalize endpoint returns matching results for all document pairs in the pipeline:

```json
{
  "pipeline_id": "unique-pipeline-id",
  "results": [
    {
      "document_1_index": 0,
      "document_2_index": 1,
      "predictions": [
        {
          "document_1_entity": {
            "text": "first",
            "entity_id": 0,
            "properties": {
              "lemma": "first",
              "pos": "JJ"
            }
          },
          "document_2_entity": {
            "text": "first",
            "entity_id": 1,
            "properties": {
              "lemma": "first",
              "pos": "JJ"
            }
          },
          "score": 0.95
        }
      ]
    }
  ]
}
```

### Output Structure

- **pipeline_id**: The unique identifier for the pipeline that was finalized
- **results**: Array of MatchResult objects, one for each document pair
    - **document_1_index**: Index of the first document (0-based)
    - **document_2_index**: Index of the second document (0-based)
    - **predictions**: Array of MatchPrediction objects containing matched entity pairs
        - **document_1_entity**: Entity from the first document with text and properties
        - **document_2_entity**: Entity from the second document with text and properties
        - **score**: Similarity score between 0.0 and 1.0 indicating match confidence

## Supported Selections

The component supports matching for different annotation types with additional properties. If the selected annotation
type is not explicitly supported, it will only compare the covered text of the entities.

| Selection                                                   | Description                         |
|-------------------------------------------------------------|-------------------------------------|
| `de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token` | Matches tokens in documents         |
| `de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity`    | Matches named entities in documents |

## Existing Parameters

| Parameter | Description                                                                                                    | Datatype | Default | Example                                                     |
|-----------|----------------------------------------------------------------------------------------------------------------|----------|---------|-------------------------------------------------------------|
| selection | The type of annotation to match (e.g., Token, NamedEntity). Determines which properties are used for matching. | String   | None    | "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token" |

## Existing Finalization Parameters

| Parameter     | Description                                                                                                          | Datatype | Default | Example                   |
|---------------|----------------------------------------------------------------------------------------------------------------------|----------|---------|---------------------------|
| model         | The name of the model to use for matching. Models are loaded from the models directory.                              | String   | None    | "token_test1", "ne_test1" |
| threshold     | Minimum similarity score (0-1) for a match to be included in results. Matches below this threshold are filtered out. | Float    | None    | 0.5, 0.75                 |
| batch_size    | Number of entity pairs to process in one batch for improved performance.                                             | Integer  | 16      | 8, 32, 64                 |
| clear_storage | Whether to clear the pipeline storage after finalization. Set to false to reuse results.                             | Boolean  | True    | True, False               |

## Model Configuration

Each model requires a `config.json` file in its directory (typically generated during training) that specifies the
model's properties and similarity matchers. An example configuration for a token matching model is shown below:

```json
{
  "name": "token_test1",
  "type": "DL",
  "entity_properties": {
    "text": {
      "property_type": "text",
      "similarity_matchers": [
        "levenshtein",
        "jaro_winkler",
        "hamming"
      ]
    },
    "lemma": {
      "property_type": "text",
      "similarity_matchers": [
        "levenshtein"
      ]
    }
  },
  "model_file": "model.weights.h5"
}
```

## Model Storage

Models should be stored in the `src/main/resources/models/` directory with the following structure:

```
models/
├── example1/
│   ├── config.json
│   └── model.weights.h5
├── example2/
│   ├── config.json
│   └── model.weights.h5
└── example3/
    ├── config.json
    └── model.weights.h5
```

