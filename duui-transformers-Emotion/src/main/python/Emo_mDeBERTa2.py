# torch libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# pytorch libraries
from scipy.special import softmax
import numpy as np

# transformers libraries
import pytorch_lightning as pl
from torchmetrics import F1Score
from torchmetrics.functional import accuracy, auroc
from transformers import AutoTokenizer, DebertaV2Model, AdamW, get_linear_schedule_with_warmup
from typing import List

import tqdm

import pandas as pd

LABEL_COLUMNS = ['anger_v2', 'fear_v2', 'disgust_v2', 'sadness_v2', 'joy_v2', 'enthusiasm_v2', 'pride_v2', 'hope_v2']
map_labels = {
    "anger_v2": "anger",
    "fear_v2": "fear",
    "disgust_v2": "disgust",
    "sadness_v2": "sadness",
    "joy_v2": "joy",
    "enthusiasm_v2": "enthusiasm",
    "pride_v2": "pride",
    "hope_v2": "hope"
}
BASE_MODEL_NAME = "microsoft/mdeberta-v3-base"

batch_size = 8
device = "cuda" if torch.cuda.is_available() else "cpu"


class CrowdCodedTagger(pl.LightningModule):
    def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.n_classes = n_classes
        self.bert = DebertaV2Model.from_pretrained(BASE_MODEL_NAME, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCELoss()

    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.last_hidden_state[:, 0])
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outputs):

        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        for i, name in enumerate(LABEL_COLUMNS):
            class_roc_auc = auroc(predictions[:, i], labels[:, i])
            self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)

    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=2e-5) #DEFINING LEARNING RATE

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps
        )

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )


class DebertaEmoCheck:
    def __init__(self, path_model: str, device='cuda:0'):
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        model = CrowdCodedTagger(n_classes=8)
        model.load_state_dict(torch.load(path_model), strict=False)
        model.to(device)
        model.eval()
        self.model = model
        self.device = device


    def emotion_prediction(self, texts: List[str]):
        with torch.no_grad():
            score_list = []
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            # del inputs.data
            outputs = self.model(**inputs.to(device))
            tensor_values = outputs[1].tolist()
            decimal_numbers = [[num for num in sublist] for sublist in tensor_values]
            for dec in decimal_numbers:
                score_dict_i = {}
                for i in range(len(dec)):
                    score_dict_i[map_labels[LABEL_COLUMNS[i]]] = float(dec[i])
                score_dict_i = dict(sorted(score_dict_i.items(), key=lambda item: item[1], reverse=True))
                score_list.append(score_dict_i)
        return score_list


if __name__ == '__main__':
    device_i = "cuda:0" if torch.cuda.is_available() else "cpu"
    texts_text = [
        "I am very happy today!", "I am very sad today!", "I am very angry today!", "I am very surprised today!", "I am very disgusted today!", "I am very fearful today!"
    ]
    texts_de = [
        "Ich bin heute sehr glücklich!", "Ich bin heute sehr traurig!", "Ich bin heute sehr wütend!", "Ich bin heute sehr überrascht!", "Ich bin heute sehr angewidert!",
        "Ich bin heute sehr ängstlich!"
    ]
    texts_tr = [
        "Bugün çok mutluyum!", "Bugün çok üzgünüm!", "Bugün çok kızgınım!", "Bugün çok şaşırdım!", "Bugün çok iğrendim!", "Bugün çok korktum!"
    ]
    path_model = f"/home/bagci/Downloads/pol_emo_mDeBERTa2/pol_emo_mDeBERTa/model/pytorch_model.pt"
    EmotionCheck = DebertaEmoCheck(path_model, device_i)
    print(EmotionCheck.emotion_prediction(texts_text))
    print(EmotionCheck.emotion_prediction(texts_de))
    print(EmotionCheck.emotion_prediction(texts_tr))