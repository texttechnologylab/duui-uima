import logging
import os.path
import sys
from typing import Dict, List, Literal, Annotated, Tuple, Union, Optional
import pandas as pd
import tensorflow as tf
import random
from neer_match.matching_model import DLMatchingModel, NSMatchingModel
from neer_match.similarity_map import SimilarityMap
from pydantic import BaseModel, Field


class EntityPropertyConfig(BaseModel):
    # the type of the property
    property_type: Literal["text", "numeric"]
    # the similarity matchers to use for this property
    similarity_matchers: List[str]


class ExportModelConfig(BaseModel):
    # the name of the model
    name: str
    # the type of the model, either "DL"
    type: Literal["DL"]
    # the properties of the entities
    entity_properties: Dict[str, EntityPropertyConfig]
    # the path to the model file
    nn_model_file: str = Field(default="model.weights.h5", alias="model_file")


class TrainingSettings(BaseModel):
    learning_rate: float
    optimizer: Literal["adam", "sgd", "rmsprop"] = "adam"
    loss_function: str = "binary_crossentropy"
    batch_size: int
    epochs: int
    verbose: bool = False


class ProvidedSplitTrainingDataConfig(BaseModel):
    type: Literal["provided_split"]
    # the path to the entity list file (.txt with one value per line or .csv with a column "value" + optional "id")
    entity_list_file: str
    # the path to the target list file (.txt with one value per line or .csv with a column "value" + optional "id")
    target_list_file: str
    # the path to the matches file (.csv with columns "entity", "target")
    matches_file: str


class UnchangedPropertyMutationConfig(BaseModel):
    type: Literal["unchanged"]


class TextPropertyMutationConfig(BaseModel):
    type: Literal["text"]
    # the minimum noise to introduce (between 0 and 1)
    min_noise: float = 0.01
    # the maximum noise to introduce (between 0 and 1)
    max_noise: float = 0.2


class NumericPropertyMutationConfig(BaseModel):
    type: Literal["numeric"]
    # the maximum absolute noise to introduce (at most +/- this value)
    max_noise: float = 0.2


class SingleDatasetTrainingDataConfig(BaseModel):
    type: Literal["single_dataset"]
    # the path to the wordlist or dataset file (.txt with one value per line)
    file: str

    # the minimum length of words to include in the training data
    min_length: Optional[int] = None

    # the mutation configs for each property (key is the property name, e.g. "text")
    property_mutations: Dict[
        str,
        Annotated[
            Union[
                UnchangedPropertyMutationConfig,
                TextPropertyMutationConfig,
                NumericPropertyMutationConfig,
            ],
            Field(discriminator="type"),
        ],
    ] = Field(default_factory=dict)

    # the number of modified samples to generate per original sample
    samples_per_original: int = 5

    # the number of words to use from the wordlist
    sample_size: Optional[int] = None


class TestDataConfig(BaseModel):
    # the sample size for testing the model (number of entities and targets to use)
    sample_size: int = 100


class ModelConfig(ExportModelConfig):
    # the path to store the model file
    export_path: str
    # the training settings
    training_settings: TrainingSettings
    # the training data configuration
    training_data: Union[
        ProvidedSplitTrainingDataConfig, SingleDatasetTrainingDataConfig
    ] = Field(discriminator="type")
    # the test data configuration
    test_data: Optional[TestDataConfig] = None


type Dataset = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


# noinspection PyUnhashable
def load_training_data_provided_split(
    config: ProvidedSplitTrainingDataConfig,
    test_config: Optional[TestDataConfig] = None,
) -> Tuple[Dataset, Optional[Dataset]]:
    # load entity list
    if config.entity_list_file.endswith(".txt"):
        entities = pd.read_csv(config.entity_list_file, header=None, names=["text"])
    else:
        entities = pd.read_csv(config.entity_list_file)
        if "text" not in entities.columns:
            raise ValueError("Entity list file must contain a 'text' column")
    # load target list
    if config.target_list_file.endswith(".txt"):
        targets = pd.read_csv(config.target_list_file, header=None, names=["text"])
    else:
        targets = pd.read_csv(config.target_list_file)
        if "text" not in targets.columns:
            raise ValueError("Target list file must contain a 'text' column")
    # load matches
    matches = pd.read_csv(config.matches_file)
    if not all(col in matches.columns for col in ["entity", "target"]):
        raise ValueError("Matches file must contain 'entity' and 'target' columns")
    # map entity ids to indices if available
    if "id" in entities.columns:
        entity_id_to_index = {idd: index for index, idd in entities["id"].items()}
        matches["entity"] = matches["entity"].map(entity_id_to_index)
    # map target ids to indices if available
    if "id" in targets.columns:
        target_id_to_index = {idd: index for index, idd in targets["id"].items()}
        matches["target"] = matches["target"].map(target_id_to_index)
    # remove id columns if present
    if "id" in entities.columns:
        entities = entities.drop(columns=["id"])
    if "id" in targets.columns:
        targets = targets.drop(columns=["id"])
    if (
        (not isinstance(entities, pd.DataFrame))
        or (not isinstance(targets, pd.DataFrame))
        or (not isinstance(matches, pd.DataFrame))
    ):
        raise ValueError("Entities, targets, and matches must be pandas DataFrames")
    # rename matches columns to "left" and "right"
    matches = matches.rename(columns={"entity": "left", "target": "right"})
    if (test_config is not None) and (test_config.sample_size is not None):
        # sample test data
        test_matches = matches.sample(n=test_config.sample_size)
        test_entities = entities.iloc[test_matches["left"]].reset_index(drop=True)
        test_targets = targets.iloc[test_matches["right"]].reset_index(drop=True)
        return (entities, targets, matches), (test_entities, test_targets, test_matches)
    return (entities, targets, matches), None


def text_introduce_noise(text: str, modifications: int) -> str:
    if modifications == 0:
        return text
    text_list = list(text)
    for _ in range(modifications):
        operation = random.choice(["insert", "delete", "substitute"])
        index = random.randint(0, len(text_list) - 1)
        if operation == "insert":
            char = random.choice("abcdefghijklmnopqrstuvwxyz")
            text_list.insert(index, char)
        elif operation == "delete" and len(text_list) > 1:
            text_list.pop(index)
        elif operation == "substitute":
            char = random.choice("abcdefghijklmnopqrstuvwxyz")
            text_list[index] = char
    return "".join(text_list)


def numeric_introduce_noise(value: float, max_noise: float) -> float:
    noise = random.uniform(-max_noise, max_noise)
    return value + noise


def build_dataset_from_words(
    words: List[Dict[str, Union[str, float]]],
    samples_per_original: int,
    mutation_configs: Dict[
        str,
        Union[
            UnchangedPropertyMutationConfig,
            TextPropertyMutationConfig,
            NumericPropertyMutationConfig,
        ],
    ],
) -> Dataset:
    samples = []
    matches = []
    for index, word in enumerate(words):
        samples.append(word)
        matches.append((index, len(samples) - 1))
        for _ in range(samples_per_original):
            modified_entry = {}
            for property_name, mutation_config in mutation_configs.items():
                if mutation_config.type == "unchanged":
                    modified_entry[property_name] = word[property_name]
                elif mutation_config.type == "text":
                    modified_entry[property_name] = text_introduce_noise(
                        word[property_name],
                        int(len(word[property_name]) * mutation_config.max_noise),
                    )
                elif mutation_config.type == "numeric":
                    modified_entry[property_name] = numeric_introduce_noise(
                        word[property_name], mutation_config.max_noise
                    )
                else:
                    raise ValueError(
                        f"Unsupported mutation type: {mutation_config.type}"
                    )
            samples.append(modified_entry)
            matches.append((index, len(samples) - 1))
    entities_df = pd.DataFrame(
        {k: [entry[k] for entry in words] for k in mutation_configs.keys()}
    )
    targets_df = pd.DataFrame(
        {k: [entry[k] for entry in samples] for k in mutation_configs.keys()}
    )
    matches_df = pd.DataFrame(matches, columns=["left", "right"])
    return entities_df, targets_df, matches_df


def load_training_data_single_dataset(
    properties: Dict[str, EntityPropertyConfig],
    config: SingleDatasetTrainingDataConfig,
    test_config: Optional[TestDataConfig] = None,
) -> Tuple[Dataset, Optional[Dataset]]:
    if any(prop not in properties for prop in config.property_mutations.keys()) or len(
        config.property_mutations
    ) != len(properties):
        raise ValueError(
            "Mutation configs must be provided for all properties defined in the model config"
        )
    if "text" not in properties:
        raise ValueError("The 'text' property must be defined in the model config")
    all_words: List[Dict[str, Union[str, float]]]
    if config.file.endswith(".txt"):
        all_words = pd.read_csv(config.file, header=None, names=["text"]).to_dict(
            orient="records"
        )
    elif config.file.endswith(".csv"):
        df = pd.read_csv(config.file)
        if "text" not in df.columns:
            raise ValueError("Dataset file must contain a 'text' column")
        all_words = df.to_dict(orient="records")
    else:
        raise ValueError(
            "Unsupported file format for dataset. Only .txt and .csv are supported"
        )

    def convert_entity(
        entry: Dict[str, Union[str, float]],
    ) -> Dict[str, Union[str, float]]:
        converted_entry = {}
        for property_name, property_config in properties.items():
            if property_name not in entry:
                raise ValueError(f"Missing property '{property_name}' in dataset entry")
            value = entry[property_name]
            if property_config.property_type == "text":
                converted_entry[property_name] = str(value)
            elif property_config.property_type == "numeric":
                converted_entry[property_name] = float(value)
            else:
                raise ValueError(
                    f"Unsupported property type: {property_config.property_type}"
                )
        return converted_entry

    all_words = [convert_entity(entry) for entry in all_words]
    if config.min_length is not None:
        all_words = [
            entry for entry in all_words if len(entry["text"]) >= config.min_length
        ]
    if config.sample_size is not None:
        words = random.sample(all_words, min(len(all_words), config.sample_size))
    else:
        words = all_words
    entities_df, targets_df, matches_df = build_dataset_from_words(
        words, config.samples_per_original, config.property_mutations
    )
    if test_config is not None:
        test_words = random.sample(
            all_words, min(len(all_words), test_config.sample_size)
        )
        test_entities_df, test_targets_df, test_matches_df = build_dataset_from_words(
            test_words, config.samples_per_original, config.property_mutations
        )
        return (entities_df, targets_df, matches_df), (
            test_entities_df,
            test_targets_df,
            test_matches_df,
        )
    return (entities_df, targets_df, matches_df), None


def load_training_data(
    properties: Dict[str, EntityPropertyConfig],
    config: Union[ProvidedSplitTrainingDataConfig, SingleDatasetTrainingDataConfig],
    test_config: Optional[TestDataConfig] = None,
) -> Tuple[Dataset, Optional[Dataset]]:
    if isinstance(config, ProvidedSplitTrainingDataConfig):
        return load_training_data_provided_split(config, test_config)
    elif isinstance(config, SingleDatasetTrainingDataConfig):
        return load_training_data_single_dataset(properties, config, test_config)
    else:
        raise ValueError(f"Unsupported training data type: {config.type}")


def create_model(config: ModelConfig) -> DLMatchingModel:
    similarity_map = SimilarityMap(
        {k: v.similarity_matchers for k, v in config.entity_properties.items()}
    )
    model = DLMatchingModel(similarity_map=similarity_map)
    if config.training_settings.optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.training_settings.learning_rate
        )
    elif config.training_settings.optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=config.training_settings.learning_rate
        )
    elif config.training_settings.optimizer == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=config.training_settings.learning_rate
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.training_settings.optimizer}")
    if config.training_settings.loss_function == "binary_crossentropy":
        loss_function = tf.keras.losses.BinaryCrossentropy()
    elif config.training_settings.loss_function == "mean_squared_error":
        loss_function = tf.keras.losses.MeanSquaredError()
    else:
        raise ValueError(
            f"Unsupported loss function: {config.training_settings.loss_function}"
        )
    model.compile(optimizer=optimizer, loss=loss_function)
    return model


def train_model(config: ModelConfig, dataset: Dataset) -> DLMatchingModel:
    left, right, matches = dataset
    model = create_model(config)
    model.fit(
        left,
        right,
        matches,
        epochs=config.training_settings.epochs,
        verbose=config.training_settings.verbose,
        batch_size=config.training_settings.batch_size,
    )
    return model


def export_model(model: DLMatchingModel, config: ModelConfig):
    export_path = config.export_path
    if not os.path.exists(export_path):
        os.makedirs(export_path)
    # save model weights
    model_file_path = f"{export_path}/{config.nn_model_file}"
    model.save_weights(model_file_path)
    # save model config
    export_config = ExportModelConfig(
        name=config.name,
        type=config.type,
        entity_properties=config.entity_properties,
        nn_model_file=config.nn_model_file,
    )
    config_file_path = f"{export_path}/config.json"
    with open(config_file_path, "w") as f:
        f.write(export_config.model_dump_json(indent=4))


def test_model(model: DLMatchingModel, test_data: Dataset, batch_size: int):
    left, right, matches = test_data
    evaluation = model.evaluate(left, right, matches, verbose=1, batch_size=batch_size)
    print(f"Test evaluation: {evaluation}")


def main():
    # load model config
    config_path = "model_config.json"
    if not os.path.exists(config_path):
        print(f"Error: Model config file '{config_path}' not found\n", file=sys.stderr)
        raise ValueError(f"Model config file '{config_path}' not found")
    with open(config_path, "r") as f:
        config = ModelConfig.model_validate_json(f.read())
    if (
        "text" not in config.entity_properties
        or len(config.entity_properties["text"].similarity_matchers) == 0
    ):
        print(
            "Error: Missing similarity matchers for 'text' property in model config\n",
            file=sys.stderr,
        )
        raise ValueError(
            "Model config must specify similarity matchers for 'text' property"
        )
    if os.path.exists(f"{config.export_path}/config.json"):
        print(
            f"Model '{config.name}' already exists at '{config.export_path}'. Overwrite? (y/n)"
        )
        while True:
            response = input().strip().lower()
            if response == "y" or response == "yes":
                break
            if response == "j" or response == "ja":
                print(
                    "Note to self: User seems to be unable to understand english"
                )  # Small joke
                break
            if response == "n" or response == "no":
                print("Aborting")
                return
    training_data, test_data = load_training_data(
        config.entity_properties, config.training_data, config.test_data
    )
    # train model
    model = train_model(config, training_data)
    # test model if test data is provided
    if test_data is not None:
        test_model(model, test_data, config.training_settings.batch_size)
    # export model
    export_model(model, config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
