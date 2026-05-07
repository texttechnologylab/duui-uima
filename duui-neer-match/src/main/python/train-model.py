import logging
import os.path
import sys
from typing import List, Literal, Annotated, Tuple, Union, Optional
import pandas as pd
import tensorflow as tf
import random
from neer_match.matching_model import DLMatchingModel, NSMatchingModel
from neer_match.similarity_map import SimilarityMap
from pydantic import BaseModel, Field


class ExportModelConfig(BaseModel):
    # the name of the model
    name: str
    # the type of the model, either "DL"
    type: Literal["DL"]
    # value similarity matchers
    similarity_matchers: List[str]
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


class WordlistTrainingDataConfig(BaseModel):
    type: Literal["wordlist"]
    # the path to the wordlist file (.txt with one value per line)
    file: str

    # the minimum length of words to include in the training data
    min_length: Optional[int] = None

    # min noise to introduce (between 0 and 1)
    min_noise: float = 0.01
    # max noise to introduce (between 0 and 1)
    max_noise: float = 0.2
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
    training_data: Union[ProvidedSplitTrainingDataConfig, WordlistTrainingDataConfig] = Field(discriminator="type")
    # the test data configuration
    test_data: Optional[TestDataConfig] = None


type Dataset = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


# noinspection PyUnhashable
def load_training_data_provided_split(config: ProvidedSplitTrainingDataConfig,
                                      test_config: Optional[TestDataConfig] = None) \
        -> Tuple[Dataset, Optional[Dataset]]:
    # load entity list
    if config.entity_list_file.endswith(".txt"):
        entities = pd.read_csv(config.entity_list_file, header=None, names=["value"])
    else:
        entities = pd.read_csv(config.entity_list_file)
        if "value" not in entities.columns:
            raise ValueError("Entity list file must contain a 'value' column")
    # load target list
    if config.target_list_file.endswith(".txt"):
        targets = pd.read_csv(config.target_list_file, header=None, names=["value"])
    else:
        targets = pd.read_csv(config.target_list_file)
        if "value" not in targets.columns:
            raise ValueError("Target list file must contain a 'value' column")
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
    if (not isinstance(entities, pd.DataFrame)) or (not isinstance(targets, pd.DataFrame)) or (
            not isinstance(matches, pd.DataFrame)):
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


def introduce_noise(word: str, modifications: int) -> str:
    if modifications == 0:
        return word
    word_list = list(word)
    for _ in range(modifications):
        operation = random.choice(["insert", "delete", "substitute"])
        index = random.randint(0, len(word_list) - 1)
        if operation == "insert":
            char = random.choice("abcdefghijklmnopqrstuvwxyz")
            word_list.insert(index, char)
        elif operation == "delete" and len(word_list) > 1:
            word_list.pop(index)
        elif operation == "substitute":
            char = random.choice("abcdefghijklmnopqrstuvwxyz")
            word_list[index] = char
    return "".join(word_list)


def build_dataset_from_words(words: List[str], samples_per_original: int, min_noise: float,
                             max_noise: float) -> Dataset:
    samples = []
    matches = []
    for index, word in enumerate(words):
        samples.append(word)
        matches.append((index, len(samples) - 1))
        for _ in range(samples_per_original):
            noise_level = random.uniform(min_noise, max_noise)
            modifications = int(len(word) * noise_level)
            modified_word = introduce_noise(word, modifications)
            samples.append(modified_word)
            matches.append((index, len(samples) - 1))
    entities_df = pd.DataFrame({"value": words})
    targets_df = pd.DataFrame({"value": samples})
    matches_df = pd.DataFrame(matches, columns=["left", "right"])
    return entities_df, targets_df, matches_df


def load_training_data_wordlist(config: WordlistTrainingDataConfig, test_config: Optional[TestDataConfig] = None) \
        -> Tuple[Dataset, Optional[Dataset]]:
    all_words: List[str] = []
    with open(config.file, "r") as f:
        for line in f:
            word = line.strip()
            if word:
                all_words.append(word)
    if config.min_length is not None:
        all_words = [word for word in all_words if len(word) >= config.min_length]
    if config.sample_size is not None:
        words = random.sample(all_words, min(len(all_words), config.sample_size))
    else:
        words = all_words
    (entities_df, targets_df, matches_df) = build_dataset_from_words(
        words, config.samples_per_original, config.min_noise, config.max_noise
    )
    if test_config is not None:
        test_words = random.sample(all_words, min(len(all_words), test_config.sample_size))
        test_entities_df, test_targets_df, test_matches_df = build_dataset_from_words(
            test_words, config.samples_per_original, config.min_noise, config.max_noise
        )
        return (entities_df, targets_df, matches_df), (test_entities_df, test_targets_df, test_matches_df)
    return (entities_df, targets_df, matches_df), None


def load_training_data(config: Union[ProvidedSplitTrainingDataConfig, WordlistTrainingDataConfig],
                       test_config: Optional[TestDataConfig] = None) \
        -> Tuple[Dataset, Optional[Dataset]]:
    if isinstance(config, ProvidedSplitTrainingDataConfig):
        return load_training_data_provided_split(config, test_config)
    elif isinstance(config, WordlistTrainingDataConfig):
        return load_training_data_wordlist(config, test_config)
    else:
        raise ValueError(f"Unsupported training data type: {config.type}")


def create_model(config: ModelConfig) -> DLMatchingModel:
    similarity_map = SimilarityMap({"value": config.similarity_matchers})
    model = DLMatchingModel(similarity_map=similarity_map)
    if config.training_settings.optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.training_settings.learning_rate)
    elif config.training_settings.optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=config.training_settings.learning_rate)
    elif config.training_settings.optimizer == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=config.training_settings.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {config.training_settings.optimizer}")
    if config.training_settings.loss_function == "binary_crossentropy":
        loss_function = tf.keras.losses.BinaryCrossentropy()
    elif config.training_settings.loss_function == "mean_squared_error":
        loss_function = tf.keras.losses.MeanSquaredError()
    else:
        raise ValueError(f"Unsupported loss function: {config.training_settings.loss_function}")
    model.compile(optimizer=optimizer, loss=loss_function)
    return model


def train_model(config: ModelConfig, dataset: Dataset) -> DLMatchingModel:
    left, right, matches = dataset
    model = create_model(config)
    model.fit(
        left, right, matches,
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
        similarity_matchers=config.similarity_matchers,
        nn_model_file=config.nn_model_file
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
    if len(config.similarity_matchers) == 0:
        print("Error: No similarity matchers specified in model config\nAborting.", file=sys.stderr)
        raise ValueError("No similarity matchers specified in model config")
    if len(config.similarity_matchers) == 1:
        print("Warning: Only one similarity matcher specified. This will render the deep learning useless, as the best solution would be an identity function.")
        print("Do you want to continue? (y/n)")
        while True:
            response = input().strip().lower()
            if response == "y" or response == "yes":
                break
            if response == "n" or response == "no":
                print("Aborting")
                return
    if os.path.exists(f"{config.export_path}/config.json"):
        print(f"Model '{config.name}' already exists at '{config.export_path}'. Overwrite? (y/n)")
        while True:
            response = input().strip().lower()
            if response == "y" or response == "yes":
                break
            if response == "j" or response == "ja":
                print("Note to self: User seems to be unable to understand english")  # Small joke
                break
            if response == "n" or response == "no":
                print("Aborting")
                return
    (training_data, test_data) = load_training_data(config.training_data, config.test_data)
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
