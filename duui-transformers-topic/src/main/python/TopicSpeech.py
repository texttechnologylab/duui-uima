import torch
import math
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
import numpy as np
from typing import List
from setfit import SetFitModel


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


map_political = {
    0: 'Macroeconomics',
    1: 'Civil Rights',
    2: 'Health',
    3: 'Agriculture',
    4: 'Labor',
    5: 'Education',
    6: 'Environment',
    7: 'Energy',
    8: 'Immigration',
    9: 'Transportation',
    10: 'Law and Crime',
    11: 'Social Welfare',
    12: 'Housing',
    13: 'Domestic Commerce',
    14: 'Defense',
    15: 'Technology',
    16: 'Foreign Trade',
    17: 'Internal Affairs',
    18: 'Government Operations',
    19: 'Public Lands',
    20: 'Culture',
    21: 'Others'
}

class TopicCheck:
    def __init__(self, model_name: str, device='cuda:0'):
        self.device = device
        if "manifesto-project" in model_name or "poltextlab" in model_name:
            self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if "manifesto-project" in model_name:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True).to(device)
        elif "WebOrganizer/TopicClassifier" in model_name:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True, use_memory_efficient_attention=False).to(device)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        if "poltextlab" in model_name:
            self.class_mapping = map_political
        else:
            self.class_mapping = self.model.config.id2label
        self.labels = list(self.class_mapping.values())

    def topic_prediction(self, texts: List[str]):
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
                for i in range(score.shape[0]):
                    score_dict_i[self.labels[ranking[i]]] = float(score_i[ranking[i]])
                score_list.append(score_dict_i)
        return score_list


class TopicCheckSetFit:
    def __init__(self, model_name):
        self.model = SetFitModel.from_pretrained(model_name)

    def topic_prediction(self, texts):
        out_list = self.model(texts).tolist()
        out_list_dict = []
        for i, out_i in enumerate(out_list):
            out_list_dict.append({out_i: 1.0})
        return out_list_dict


# class MudesHateCheck:
#     def __init__(self, device='cuda:0'):
#         self.device = device
#         self.model = MUDESApp("multilingual-base", use_cuda=True)
#
#     def hate_prediction(self, texts: List[str]):
#         pred = self.model.p
#         return self.model.predict(texts)


if __name__ == '__main__':
    device_i = "cuda:0" if torch.cuda.is_available() else "cpu"

    models = {
        # "KnutJaegersberg/topic-classification-IPTC-subject-labels": "KnutJaegersberg",
        "poltextlab/xlm-roberta-large-manifesto-cap": "poltextlab",
        "manifesto-project/manifestoberta-xlm-roberta-56policy-topics-context-2023-1-1": "manifesto-project",
    }
    models_en = {
        "cardiffnlp/roberta-large-tweet-topic-single-all": "cardiffnlp",
        "nickmuchi/finbert-tone-finetuned-finance-topic-classification": "nickmuchi",
    }
    models_de = {
        "chkla/parlbert-topic-german": "chkla",
    }
    texts = [
        "I hate you", "I love you", "I hate you, but I love you"
    ]
    text_de = [
        "Ich hasse dich", "Ich liebe dich", "Ich hasse dich, aber ich liebe dich"
    ]
    text_tr = [
        "Seni seviyorum", "Seni sevmiyorum", "Seni seviyorum ama seni sevmiyorum"
    ]
    # topic_check = TopicCheckSetFit("KnutJaegersberg/topic-classification-IPTC-subject-labels")
    # print(topic_check.predict(texts))
    # for model_i in models:
    #     topic_check = TopicCheck(model_i, device_i)
    #     print(topic_check.topic_prediction(texts))
    # print(topic_check.topic_prediction(text_de))
    # print(topic_check.topic_prediction(text_tr))

    texts = [
        # "🚀 Science is our gateway to understanding the universe! 🌌 Take our survey and share your thoughts on the future of scientific discovery. Your input can shape tomorrow's innovations! 🧬🔬 #ScienceSurvey #FutureOfScience"
        "“America, ride or die” — Elon Musk"
    ]
    test_modesl = {
        "cardiffnlp/roberta-large-tweet-topic-single-all": "cardiffnlp",
        # "nickmuchi/finbert-tone-finetuned-finance-topic-classification": "nickmuchi",
        "poltextlab/xlm-roberta-large-manifesto-cap": "poltextlab",
        "manifesto-project/manifestoberta-xlm-roberta-56policy-topics-context-2023-1-1": "manifesto-project",
    }

