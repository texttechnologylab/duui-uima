import torch
import math
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
import numpy as np
from typing import List
from detoxify import Detoxify
# from googleapiclient import discovery
# import json
# import time


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


map_toxic = {
    "LABEL_0": "non toxic",
    "LABEL_1": "toxic",
}


class ToxicCheck:
    def __init__(self, model_name: str, device='cuda:0'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.class_mapping = self.model.config.id2label
        self.labels = list(self.class_mapping.values())
        for i in range(len(self.labels)):
            if self.labels[i] == "neutral" or self.labels[i] == "not toxic" or self.labels[i] == "not_toxic":
                self.labels[i] = "non toxic"

    def toxic_prediction(self, texts: List[str]):
        with torch.no_grad():
            score_list = []
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            outputs = self.model(**inputs)
            scores = outputs[0].detach().cpu().numpy()
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


# class Perspective:
#     def __init__(self, api_key: str):
#         self.api_key = api_key
#         self.service = discovery.build("commentanalyzer", "v1alpha1", developerKey=api_key, discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1", static_discovery=False,)
#
#     def toxic_prediction(self, texts: List[str]):
#         results = []
#         for text in texts:
#             time_start = time.time()
#             analyze_request = {
#                 'comment': {'text': text},
#                 'requestedAttributes': {'TOXICITY': {},
#                                         # "SEVERE_TOXICITY": {}, "IDENTITY_ATTACK": {}, "INSULT": {}, "THREAT": {}, "PROFANITY": {}, "SEXUALLY_EXPLICIT": {},
#                                         # "FLIRTATION": {}
#                                         }
#             }
#             response = self.service.comments().analyze(body=analyze_request).execute()
#             out_dict = {}
#             for key in response["attributeScores"]:
#                 out_dict[key] = response["attributeScores"][key]["summaryScore"]["value"]
#             results.append(out_dict)
#             time_end = time.time()
#             # if less than 1 second, wait for 1 second
#             if time_end - time_start < 1:
#                 time.sleep(1 - (time_end - time_start))
#         return results
