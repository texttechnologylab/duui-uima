import torch
import math
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
import numpy as np
from typing import List
from setfit import SetFitModel
from germansentiment import SentimentModel
from pysentimiento import create_analyzer


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


model_mapping = {
    "cardiffnlp/twitter-xlm-roberta-base-sentiment": {
        0: "negative",
        1: "neutral",
        2: "positive"
    },
    "citizenlab/twitter-xlm-roberta-base-sentiment-finetunned": {
        0: "negative",
        1: "neutral",
        2: "positive"
    },
    "lxyuan/distilbert-base-multilingual-cased-sentiments-student": {
        0: "positive",
        1: "neutral",
        2: "negative"
    },
    "philschmid/distilbert-base-multilingual-cased-sentiment": {
        0: "negative",
        1: "neutral",
        2: "positive"
    },
    "cardiffnlp/twitter-roberta-base-sentiment-latest": {
        0: "negative",
        1: "neutral",
        2: "positive"
    },
    "j-hartmann/sentiment-roberta-large-english-3-classes": {
        0: "negative",
        1: "neutral",
        2: "positive"
    },
    "bardsai/finance-sentiment-de-base":{
        0: "positive",
        1: "neutral",
        2: "negative"
    }
}


class SentimentCheck:
    def __init__(self, model_name: str, device='cuda:0'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, ).to(device)
        self.class_mapping = model_mapping[model_name]
        self.labels = list(self.class_mapping.values())

    def prediction(self, texts: List[str]):
        with torch.no_grad():
            score_list = []
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
                self.device)
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


class SentimentCheckSetFit:
    def __init__(self, model_name: str, device='cuda:0'):
        self.device = device
        self.model = SetFitModel.from_pretrained(model_name, device=device)
        self.map = {
            0: "negative",
            1: "neutral",
            2: "positive",
            3: "None"
        }

    def prediction(self, texts: List[str]):
        out_list = self.model(texts).tolist()
        out_list_all = []
        for i in out_list:
            out_list_all.append({self.map[i]: "1.0"})
        return out_list_all


class SentimentCheckGerman:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.model = SentimentModel()

    def prediction(self, texts: List[str]):
        out_list = self.model.predict_sentiment(texts, output_probabilities=True)
        out_list_all = []
        for i in out_list[1]:
            out_list_i = {}
            for sent_i in i:
                out_list_i[sent_i[0]] = sent_i[1]
            out_list_all.append(out_list_i)
        return out_list_all


class PySentimientoCheck:
    def __init__(self, lang="en"):
        self.analyzer = create_analyzer(task="sentiment", lang=lang)

    def prediction(self, texts: List[str]):
        output = []
        for text in texts:
            prediction_i = self.analyzer.predict(text)
            out_i = {
                "output": prediction_i.output,
                "probabilities": prediction_i.probas
            }
            output.append(out_i)
        return output


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
    models_text = {
        "cardiffnlp/twitter-xlm-roberta-base-sentiment": "cardiffnlp",
        "nlptown/bert-base-multilingual-uncased-sentiment": "nlptown",
        "lxyuan/distilbert-base-multilingual-cased-sentiments-student": "lxyuan",
        "clampert/multilingual-sentiment-covid19": "clampert",
        "philschmid/distilbert-base-multilingual-cased-sentiment": "philschmid",
        "HasinMDG/XLM_Roberta_Sentiment_Toward_Topics_Baseline": "HasinMDG",
        "citizenlab/twitter-xlm-roberta-base-sentiment-finetunned": "citizenlab",
        # new models
        # "pysentimiento": "pysentimiento",
        # "siebert/sentiment-roberta-large-english": "SiEBERT-English-Language-Sentiment-Classification",
        # "FFZG-cleopatra/M2SA-text-only": "M2SA",
    }

    models_m2 = {
        "cardiffnlp/twitter-xlm-roberta-base-sentiment": "cardiffnlp",
        "nlptown/bert-base-multilingual-uncased-sentiment": "nlptown",
        "lxyuan/distilbert-base-multilingual-cased-sentiments-student": "lxyuan",
        "clampert/multilingual-sentiment-covid19": "clampert",
        "philschmid/distilbert-base-multilingual-cased-sentiment": "philschmid",
        "HasinMDG/XLM_Roberta_Sentiment_Toward_Topics_Baseline": "HasinMDG",
        "citizenlab/twitter-xlm-roberta-base-sentiment-finetunned": "citizenlab",
        "FFZG-cleopatra/M2SA-text-only": "FFZG-cleopatra",
    }
    models = {
        "cardiffnlp/twitter-xlm-roberta-base-sentiment": "cardiffnlp",
        "nlptown/bert-base-multilingual-uncased-sentiment": "nlptown",
        "lxyuan/distilbert-base-multilingual-cased-sentiments-student": "lxyuan",
        "clampert/multilingual-sentiment-covid19": "clampert",
        "philschmid/distilbert-base-multilingual-cased-sentiment": "philschmid",
        "HasinMDG/XLM_Roberta_Sentiment_Toward_Topics_Baseline": "HasinMDG",
        "citizenlab/twitter-xlm-roberta-base-sentiment-finetunned": "citizenlab",
    }
    # english_models = {
    #     "cardiffnlp/twitter-roberta-base-sentiment-latest": "ENcardiffnlp",
    #     "siebert/sentiment-roberta-large-english": "ENsiebert",
    #     "j-hartmann/sentiment-roberta-large-english-3-classes": "ENj-hartmann",
    # }
    # german_models = {
    #     "oliverguhr/german-sentiment-bert": "DEoliverguhr",
    #     "bardsai/finance-sentiment-de-base": "DEbardsai",
    #     "mdraw/german-news-sentiment-bert": "DEmdraw",
    # }
