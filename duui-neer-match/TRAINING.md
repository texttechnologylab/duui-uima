# Training New Models

This document provides a practical guide to training new models for entity matching using the provided trainer. It
covers the overall process, configuration options, and tips for getting started.
The goal is to keep training flexible without making the setup overly complicated.

## Quick Overview

Training works like this:

1. Prepare a training dataset.
2. Describe the model and training settings in `model_config.json`.
3. Run the training script.
4. Use the exported model folder in `src/main/resources/models/`.

The trainer reads the configuration, builds the model, trains it, optionally evaluates it on test data, and writes the
results to the configured export directory.

## Two Training Data Options

You can train from either of these sources:

### 1. Single Dataset

Use one dataset containing the original values. The trainer creates slightly modified variants automatically and learns
from those pairs.

This is the easiest option when you already have a list of tokens, names, or other entity values.

#### Configuration

In this case, the `training_data` section of `model_config.json` should look like this:

```json
"training_data": {
"type": "single_dataset", # indicates that we are using a single dataset
"file": "./tokens.csv", # the path to the dataset file (can be .csv or .txt)
"sample_size": 400, # the number of original samples to use for training (or no value for all)
"min_length": 3, # minimum length of the text values to consider for training (or no value for no minimum)
"samples_per_original": 3, # how many modified samples to create from each original value
"property_mutations": {
# how to create modified samples for each property
"text": {
"type": "text", # indicates that this is a text property
"min_noise": 0.1, # minimum noise level (e.g., 10% of the characters)
"max_noise": 0.2  # maximum noise level (e.g., 20% of the characters)
},
"size": {
"type": "numeric", # indicates that this is a numeric property
"max_noise": 2  # maximum absolute value to add or subtract from the original value (e.g., +/- 2)
},
"whatever": {
"type": "unchanged"  # indicates that this property should be included in the training pairs but not modified (i.e., it stays the same as the original value)
}
}
}
```

**Note:** Make sure that the property_mutations section includes all and only the properties defined in the
`entity_properties` section of the configuration. Each property must have a corresponding mutation strategy defined.
While the mutation strategy is not directly connected to the property type, certain strategies are only supported for
specific property types.

### 2. Provided Split

Use three files:

* one file with entities
* one file with targets
* one file with the known matches between them

This is useful when you already have curated training pairs.

#### Configuration

In this case, the `training_data` section of `model_config.json` should look like this:

```json
"training_data": {
"type": "provided_split", # indicates that we are using a provided split of datasets
"entitie_list_file": "./entities.csv", # the path to the entities dataset file (either .csv or .txt)
"target_list_file": "./targets.csv", # the path to the targets dataset file (either .csv or .txt)
"matches_file": "./matches.csv"  # the path to the file containing known matches between entities and targets (should have columns "entity_id" and "target_id", only .csv format is supported for the matches file)
}
```

The entities and targets may include "id" columns that are then used instead of the index when processing the matches
file. The existence of an "id" column in either file does not require its presence in the other file.

**Note from Author:** The names "entity" and "target" are the way I called them originally, so I just decided to keep
it, however which is the entity and which is the target does not matter.

## What Goes Into `model_config.json`

The configuration file defines three main things:

* the model name and export folder
* the entity properties that should be compared
* the training settings and training data source

At a minimum, the `text` property should be defined and should use at least one similarity matcher.

Example:

```json
{
  "export_path": "path/to/exported/model/my_model",
  "name": "my_model",
  "type": "DL",
  "entity_properties": {
    "text": {
      "property_type": "text",
      "similarity_matchers": [
        "levenshtein",
        "jaro_winkler"
      ]
    }
  },
  "training_settings": {
    "learning_rate": 0.0001,
    "optimizer": "adam",
    "loss_function": "binary_crossentropy",
    "batch_size": 32,
    "epochs": 15,
    "verbose": true
  },
  "training_data": {
    "type": "single_dataset",
    "file": "./tokens.csv",
    "sample_size": 400,
    "min_length": 3,
    "samples_per_original": 3,
    "property_mutations": {
      "text": {
        "type": "text",
        "min_noise": 0.1,
        "max_noise": 0.2
      }
    }
  }
}
```

## Supported Similarity Matchers

The trainer currently supports these similarity matchers:

* `basic_ratio`
* `damerau_levenshtein`
* `discrete`
* `euclidean`
* `gaussian`
* `hamming`
* `indel`
* `jaro`
* `jaro_winkler`
* `lcsseq`
* `levenshtein`
* `osa`
* `partial_ratio`
* `partial_ratio_alignment`
* `partial_token_ratio`
* `partial_token_set_ratio`
* `partial_token_sort_ratio`
* `postfix`
* `prefix`
* `token_ratio`
* `token_set_ratio`
* `token_sort_ratio`

If you want a safe starting point, use a small set of text-oriented matchers such as `levenshtein`, `jaro_winkler`, and
`hamming`.

## Running Training

Place `model_config.json` in the working directory and run the training script from there. The trainer will read the
configuration, prepare the data, build the model, and start training.

After training finishes, the model is written to the configured export folder together with a `config.json` file and the
model weights.

## Practical Tips

* Keep the first version small. A simple model with one or two properties is easier to test.
* Use `test_data` when you want a quick quality check before exporting.
* Start with a modest sample size and increase it only if the model clearly needs more data.
* Keep the exported model folder name aligned with the model name (this is actually required for the runtime as a sanity
  check).
* In some of the first epochs of training some of the evaluation metrics may be NaN, which is normal. If they stay NaN
  for many epochs, check the training data and settings.
* Keep the way neer-match works in mind: It does not feed the text in the model directly, but rather the similarity
  scores from the configured matchers. This means just having many entries that just give similar scores for all pairs
  will not help the model learn to distinguish between them. The training data should include a good variety of
  similarity scores across the different matchers to allow the model to learn meaningful patterns.

## Typical Output

After training, you should end up with a folder like this:

```text
path/to/exported/model/my_model/
├── config.json
└── model.weights.h5
```

That folder is what the runtime uses when matching entities later.

## Important Notes

### Requirements for the Training Data

#### Dataset Format

The datasets should either be a .csv file with a header row, or alternatively (only applicable if there only exists a "
text" column) a .txt file with the entity values listed line by line.

#### Required columns

All datasets are required to have a "text" column and - as the name suggests - it needs to accept text (string) values.
The "text" column is always mapped to the covered text of the selected UIMA annotations.

<footer>
<hr>
<h3>LLM Usage Disclosure</h3>

This document was partially created with the help of a large language model (LLM) to assist with writing and structuring
the content.
All final content was reviewed and edited by a human to ensure accuracy and clarity.
</footer>
