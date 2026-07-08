import torch
import math
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
import numpy as np
from typing import List

ssharoff_genres = {
    0: "argum",
    1: "fictive",
    2: "instruct",
    3: "reporting",
    4: "legal",
    5: "personal",
    6: "commercial",
    7: "academic",
    8: "info",
    9: "reviews",
}


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class GenreCheck:
    def __init__(self, model_name: str, device='cuda:0'):
        self.device = device
        if model_name == "TurkuNLP/web-register-classification-en" or model_name=="TurkuNLP/web-register-classification-multilingual":
            self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # if "manifesto-project" in model_name:
        #     self.model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True).to(device)
        # elif "WebOrganizer/TopicClassifier" in model_name:
        #     self.model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True, use_memory_efficient_attention=False).to(device)
        # else:
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        if "ssharoff" in model_name:
            self.class_mapping = ssharoff_genres
        else:
            self.class_mapping = self.model.config.id2label
        self.labels = list(self.class_mapping.values())

    def genre_prediction(self, texts: List[str]):
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

