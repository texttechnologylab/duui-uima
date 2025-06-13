import torch
import math
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
import numpy as np
from typing import List


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


map_hate = {
    "Andrazp/multilingual-hate-speech-robacofi": {
        0: "NOT HATE",
        1: "HATE"
    },
    "alexandrainst/da-hatespeech-detection-base": {
        0: "NOT HATE",
        1: "HATE"
    },
    "l3cube-pune/me-hate-roberta": {
        0: "NOT HATE",
        1: "HATE"
    },
    "GroNLP/hateBERT": {
        0: "NOT HATE",
        1: "HATE"
    },
    "Hate-speech-CNERG/bert-base-uncased-hatexplain": {
        1: "NOT HATE",
        2: "OFFENSIVE",
        0: "HATE",
    },
    "cardiffnlp/twitter-roberta-base-hate-latest": {
        0: "NOT HATE",
        1: "HATE"
    },
    "deepset/bert-base-german-cased-hatespeech-GermEval18Coarse": {
        0: "NOT HATE",
        1: "HATE"
    },
    "Hate-speech-CNERG/dehatebert-mono-english": {
        0: "NOT HATE",
        1: "HATE"
    },
    "Hate-speech-CNERG/dehatebert-mono-german": {
        0: "NOT HATE",
        1: "HATE"
    },
    "Hate-speech-CNERG/dehatebert-mono-spanish": {
        0: "NOT HATE",
        1: "HATE"
    },
    "Hate-speech-CNERG/dehatebert-mono-polish": {
        0: "NOT HATE",
        1: "HATE"
    },
    "Hate-speech-CNERG/dehatebert-mono-portugese": {
        0: "NOT HATE",
        1: "HATE"
    },
    "Hate-speech-CNERG/dehatebert-mono-italian": {
        0: "NOT HATE",
        1: "HATE"
    },
    "Hate-speech-CNERG/dehatebert-mono-arabic": {
        0: "NOT HATE",
        1: "HATE"
    },
    "Hate-speech-CNERG/dehatebert-mono-french": {
        0: "NOT HATE",
        1: "HATE"
    },
    "Hate-speech-CNERG/dehatebert-mono-indonesian": {
        0: "NOT HATE",
        1: "HATE"
    },
    "facebook/roberta-hate-speech-dynabench-r4-target":{
        0: "NOT HATE",
        1: "HATE"
    },
    "irlab-udc/MetaHateBERT": {
        0: "NOT HATE",
        1: "HATE"
    },
    "HateBERT_hateval": {
        0: "NOT HATE",
        1: "HATE"
    },
    "Exqrch/IndoBERTweet-HateSpeech": {
        0: "NOT HATE",
        1: "HATE"
    },
    "MilaNLProc/hate-ita-xlm-r-large": {
        0: "NOT HATE",
        1: "HATE"
    },
}


class HateCheck:
    def __init__(self, model_name: str, device='cuda:0'):
        self.device = device
        if model_name == "Exqrch/IndoBERTweet-HateSpeech":
            self.tokenizer = AutoTokenizer.from_pretrained("indolem/indobertweet-base-uncased")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.class_mapping = self.model.config.id2label
        self.labels = list(map_hate[model_name].values())


    def hate_prediction(self, texts: List[str]):
        with torch.no_grad():
            score_list = []
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            outputs = self.model(**inputs)
            scores = outputs[0].detach().cpu().numpy()
            for score in scores:
                score_dict_i = []
                score_i = softmax(score)
                ranking = np.argsort(score_i)
                ranking = ranking[::-1]
                for i in range(score.shape[0]):
                    score_dict_i.append({"label": self.labels[ranking[i]], "score": float(score_i[ranking[i]])})
                score_list.append(score_dict_i)
        return score_list


class HateCheckEziisk:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("EZiisk/EZ_finetune_Vidgen_model_RHS_Best_Tokenizer")
        self.model = AutoModelForSequenceClassification.from_pretrained("EZiisk/EZ_finetune_Vidgen_model_RHS_Best").to(device)
        self.class_mapping = self.model.config.id2label
        self.labels = ["NOT HATE", "HATE"]

    def hate_prediction(self, texts: List[str]):
        score_list = []
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
            self.device)
        outputs = self.model(**inputs)
        scores = outputs[0].detach().cpu().numpy()
        for score in scores:
            score_dict_i = []
            score_i = softmax(score)
            ranking = np.argsort(score_i)
            ranking = ranking[::-1]
            for i in range(score.shape[0]):
                score_dict_i.append({"label": self.labels[ranking[i]], "score": float(score_i[ranking[i]])})
            score_list.append(score_dict_i)
        return score_list