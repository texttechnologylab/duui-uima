import torch
import math
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
import numpy as np
from typing import List

model_name_map = {
    "climatebert/distilroberta-base-climate-detector": "ClimateDetector",
    "climatebert/distilroberta-base-climate-tcfd": "ClimateTCFD",
    "climatebert/distilroberta-base-climate-commitment": "ClimateCommitment",
    "climatebert/distilroberta-base-climate-sentiment": "ClimateSentiment",
    "climatebert/distilroberta-base-climate-specificity": "ClimateSpecificity",
}

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class ClimateBert:
    def __init__(self, model_name: str, device='cuda:0'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.class_mapping = self.model.config.id2label
        self.labels = list(self.class_mapping.values())

    def prediction(self, texts: List[str]):
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            outputs = self.model(**inputs)
            logits = outputs[0].float()          # convert bfloat16 -> float32
            probs = torch.softmax(logits, dim=-1)

            score_list = []

            for prob in probs.cpu():
                ranking = torch.argsort(prob, descending=True)

                score_dict_i = {
                    self.labels[i]: float(prob[i])
                    for i in ranking
                }

                score_list.append(score_dict_i)
        return score_list

