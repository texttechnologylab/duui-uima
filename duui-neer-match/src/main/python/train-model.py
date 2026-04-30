import logging
import os.path
from typing import List, Optional, Literal, Annotated, Tuple
import pandas as pd
import tensorflow as tf
import pydantic
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
    model_file: str = "model.ckpt"

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

type TrainingDataConfig = Annotated[ProvidedSplitTrainingDataConfig, Field(discriminator="type")]

class ModelConfig(ExportModelConfig):
    # the path to store the model file
    export_path: str
    # the training settings
    training_settings: TrainingSettings
    # the training data configuration
    training_data: TrainingDataConfig


# noinspection PyUnhashable
def load_training_data(config: ProvidedSplitTrainingDataConfig)-> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    # rename matches columns to "left" and "right"
    matches = matches.rename(columns={"entity": "left", "target": "right"})
    return entities, targets, matches

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
    model.compile(optimizer=optimizer, loss=config.training_settings.loss_function)
    return model

def train_model(config: ModelConfig) -> DLMatchingModel:
    left, right, matches = load_training_data(config.training_data)
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
    model_file_path = f"{export_path}/{config.model_file}"
    model.save_weights(model_file_path)
    # save model config
    export_config = ExportModelConfig(
        name=config.name,
        type=config.type,
        similarity_matchers=config.similarity_matchers,
        model_file=config.model_file
    )
    config_file_path = f"{export_path}/config.json"
    with open(config_file_path, "w") as f:
        f.write(export_config.model_dump_json(indent=4))

def main():
    # load model config
    config_path = "model_config.json"
    if not os.path.exists(config_path):
        raise ValueError(f"Model config file '{config_path}' not found")
    with open(config_path, "r") as f:
        config = ModelConfig.model_validate_json(f.read())
    # train model
    model = train_model(config)
    # export model
    export_model(model, config)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
