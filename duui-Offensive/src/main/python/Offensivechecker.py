import torch
import math
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
import numpy as np
from typing import List

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

map_hate = {
    "Hate-speech-CNERG/bert-base-uncased-hatexplain": {
        0: "Hate",
        1: "Not Hate",
        2: "Offensive",
    },
    "Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two": {
        0: "Normal",
        1: "Abusive",
    },
    "worldbank/naija-xlm-twitter-base-hate": {
        0: "Neutral",
        1: "Offensive",
        2: "Hate"
    },
    "HateBERT_offenseval": {
        0: "Not Offensive",
        1: "Offensive",
    },
    "HateBERT_abuseval": {
        0: "Not Abusive",
        1: "Abusive",
    },
    "pysentimiento/bertweet-hate-speech":{
        0: "Hateful",
        1: "Targeted",
        2: "Aggressive",
    },
    "pysentimiento/robertuito-hate-speech": {
        0: "Hateful",
        1: "Targeted",
        2: "Aggressive",
    },
    "pysentimiento/bertabaporu-pt-hate-speech": {
        0: "Sexism",
        1: "Body",
        2: "Racism",
        3: "Ideology",
        4: "Homophobia"
    },
    "pysentimiento/bert-it-hate-speech": {
        0: "hateful",
        1: "stereotype"
    },
    "IMSyPP/hate_speech_multilingual": {
        0: "Acceptable",
        1: "Inappropriate",
        2: "Offensive",
        3: "Violent"
    },
    "IMSyPP/hate_speech_en": {
        0: "Acceptable",
        1: "Inappropriate",
        2: "Offensive",
        3: "Violent"
    },
    "IMSyPP/hate_speech_it": {
        0: "Acceptable",
        1: "Inappropriate",
        2: "Offensive",
        3: "Violent"
    },
    "IMSyPP/hate_speech_slo":{
        0: "Acceptable",
        1: "Inappropriate",
        2: "Offensive",
        3: "Violent"
    },
    "IMSyPP/hate_speech_nl":{
        0: "Acceptable",
        1: "Inappropriate",
        2: "Offensive",
        3: "Violent"
    },
    "cardiffnlp/twitter-roberta-base-hate-multiclass-latest": {
        0: "Sexism",
        1: "Racism",
        2: "Disability",
        3: "Sexual Orientation",
        4: "Religion",
        5: "Other",
        6: "Not Hate",
    },
    "cardiffnlp/twitter-roberta-large-sensitive-multilabel": {
        0: "Conflictual",
        1: "Profanity",
        2: "Sex",
        3: "Drugs",
        4: "Selfharm",
        5: "Spam",
        6: "Not Sensitive"
    }
}

# access_token = read_json("data/token.json")["token"]
access_token = ""


class OffensiveCheck:
    def __init__(self, model_name: str, device='cuda:0'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.class_mapping = self.model.config.id2label
        self.labels = list(map_hate[model_name].values())

    def offensive_prediction(self, texts: List[str]):
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


if __name__ == '__main__':
    device_i = "cuda:0" if torch.cuda.is_available() else "cpu"
    off = OffensiveCheck("cardiffnlp/twitter-roberta-base-hate-multiclass-latest", device_i)
    print(off.offensive_prediction(["I hate you! I will destroy this place!",
                "I very happy to be here. I love this place."]))
