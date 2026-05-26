import torch
import math
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline, AutoModelForCausalLM
from scipy.special import softmax
import numpy as np
from typing import List
from detoxify import Detoxify
# from googleapiclient import discovery
import json
import time


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


map_toxic = {
    "LABEL_0": "non toxic",
    "LABEL_1": "toxic",
}

toxic_models = {
    "Aira-ToxicityModel": "nicholasKluge/ToxicityModel",
    "textdetox": "textdetox/xlmr-large-toxicity-classifier",
    "citizenlab": "citizenlab/distilbert-base-multilingual-cased-toxicity",
    "EIStakovskii": "EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus",
    "FredZhang7": "FredZhang7/one-for-all-toxicity-v3",
    "Detoxifying": "Detoxifying",
}

models_list = {
    "garak-llm/roberta_toxicity_classifier":
        {
            0 : "non toxic",
            1 : "toxic"
        },
    "s-nlp/russian_toxicity_classifier":
        {
            0: "non toxic",
            1: "toxic"
        },
    "malexandersalazar/xlm-roberta-large-binary-cls-toxicity": {
            0: "non toxic",
            1: "toxic"
        },
    "sismetanin/rubert-toxic-pikabu-2ch": {
            0: "non toxic",
            1: "toxic"
        },
        "textdetox/glot500-toxicity-classifier": {
            0: "non toxic",
            1: "toxic"
        },
    "textdetox/xlmr-large-toxicity-classifier": {
            0: "non toxic",
            1: "toxic"
        },
    "textdetox/bert-multilingual-toxicity-classifier": {
            0: "non toxic",
            1: "toxic"
        },
    "dardem/xlm-roberta-large-uk-toxicity": {
            0: "non toxic",
            1: "toxic"
        },
    "Xuhui/ToxDect-roberta-large": {
            0: "non toxic",
            1: "toxic"
        },
    }


class ToxicCheck:
    def __init__(self, model_name: str, device='cuda:0'):
        self.device = device
        self.model_name = model_name
        if model_name == "malexandersalazar/xlm-roberta-large-binary-cls-toxicity":
            self.tokenizer = AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-large')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.class_mapping = self.model.config.id2label
        if model_name in toxic_models:
            self.labels = list(map_toxic.values())
        else:
            self.labels = list(self.class_mapping.values())
            for i in range(len(self.labels)):
                if self.labels[i] == "neutral" or self.labels[i] == "not toxic" or self.labels[i] == "not_toxic" or self.labels[i] == "non-toxic":
                    self.labels[i] = "non toxic"

    def toxic_prediction(self, texts: List[str]):
        with torch.no_grad():
            score_list = []
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
                self.device)
            outputs = self.model(**inputs)
            scores = outputs[0].detach().cpu().numpy()
            if self.model_name == "nicholasKluge/ToxicityModel":
                for score in scores:
                    score_dict_i = {"non toxic": float(score[0]), "toxic": float(1 - score[0])}
                    score_list.append(score_dict_i)
            else:
                for score in scores:
                    score_dict_i = {}
                    score_i = softmax(score)
                    ranking = np.argsort(score_i)
                    ranking = ranking[::-1]
                    # if "HATE" in self.labels:
                    #     score_dict_i["toxic"] = float(score_i[ranking["HATE"]])
                    # if "NOT HATE" in self.labels:
                    #     score_dict_i["non toxic"] = float(score_i[ranking["NOT HATE"]])
                    # if "NOT_HATE" in self.labels:
                    #     score_dict_i["non toxic"] = float(score_i[ranking["NOT_HATE"]])
                    for i in range(score.shape[0]):
                        if self.labels[ranking[i]] in map_toxic:
                            score_dict_i[map_toxic[self.labels[ranking[i]]]] = float(score_i[ranking[i]])
                        elif self.labels[ranking[i]] == "":
                            score_dict_i[self.labels[ranking[i]].replace("-", " ")] = float(score_i[ranking[i]])
                        else:
                            score_dict_i[self.labels[ranking[i]]] = float(score_i[ranking[i]])
                    score_list.append(score_dict_i)
        return score_list


class ToxicAiraCheck:
    def __init__(self, model_name: str, device='cuda:0'):
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            device)
        self.model.eval()

    def toxic_prediction(self, texts: List[str]):
        prompt = "Create a toxic tweet: "
        with torch.no_grad():
            score_list = []
            for text in texts:
                inputs = self.tokenizer(prompt, text, return_tensors="pt", padding=True, truncation=True,
                                        max_length=512).to(self.device)
                outputs = self.model(**inputs)[0].item()
                softmax_score = softmax([outputs, 1 - outputs])
                score_dict_i = {"non toxic": float(softmax_score[0]), "toxic": float(softmax_score[1])}
                # sort after the scores
                score_dict_i = dict(sorted(score_dict_i.items(), key=lambda item: item[1], reverse=True))
                score_list.append(score_dict_i)
        return score_list


class Detoxifying:
    def __init__(self, device="cuda:0", mode="all", lang="multilingual"):
        self.device = device
        self.model = Detoxify(lang, device=device)
        self.mode = mode

    def toxic_prediction(self, texts: List[str]):
        prediction = self.model.predict(texts)
        pred_out = []
        for pred_i in prediction["toxicity"]:
            pred = "toxic"
            pos = pred_i
            non_pred = "non toxic"
            non_pos = 1 - pred_i
            preds = {
                pred: pos,
                non_pred: non_pos
            }
            pred_out.append(preds)
        return pred_out

    def toxic_prediction_all(self, texts: List[str]):
        prediction = self.model.predict(texts)
        pred_out = prediction
        return pred_out


