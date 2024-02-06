import torch
import math
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
import numpy as np
from typing import List
from detoxify import Detoxify


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


map_toxic = {
    "LABEL_0": "non toxic",
    "LABEL_1": "toxic",
}


class ToxicCheck:
    def __init__(self, model_name: str, device='cuda:0'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=f"/storage/nlp/huggingface/models")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=f"/storage/nlp/huggingface/models").to(device)
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
                score_dict_i = []
                score_i = softmax(score)
                ranking = np.argsort(score_i)
                ranking = ranking[::-1]
                for i in range(score.shape[0]):
                    if self.labels[ranking[i]] in map_toxic:
                        score_dict_i.append({"label": map_toxic[self.labels[ranking[i]]], "score": float(score_i[ranking[i]])})
                    else:
                        score_dict_i.append({"label": self.labels[ranking[i].replace("-", " ")], "score": float(score_i[ranking[i]])})
                        # score_dict_i[self.labels[ranking[i]].replace("-", " ")] = float(score_i[ranking[i]])
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
            preds = [
                {"label": pred, "score": pos}, {"label": non_pred, "score": non_pos}
            ]
            pred_out.append(preds)
        return pred_out

    def toxic_prediction_all(self, texts: List[str]):
        prediction = self.model.predict(texts)
        pred_out = prediction
        return pred_out


if __name__ == '__main__':
    device_i = "cuda:2" if torch.cuda.is_available() else "cpu"
    models = [
        "EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus", "FredZhang7/one-for-all-toxicity-v3",
        "citizenlab/distilbert-base-multilingual-cased-toxicity"
    ]
    # for model in models:
    #     toxic_check = ToxicCheck(model, device_i)
    #     print(toxic_check.toxic_prediction(["I hate you", "I love you"]))
    models_en = {
        "tomh/toxigen_hatebert": "tomh",
        "GroNLP/hateBERT": "GroNLP",
        "pysentimiento/bertweet-hate-speech": "pysentimiento",
        "Hate-speech-CNERG/bert-base-uncased-hatexplain": "Hate-speech-CNERG",
        "cardiffnlp/twitter-roberta-base-hate-latest": "cardiffnlp",
    }
    for model in models_en:
        toxic_check = ToxicCheck(model, device_i)
        print(toxic_check.toxic_prediction(["I hate you", "I love you"]))
    models_de = {
        "Hate-speech-CNERG/dehatebert-mono-german": "Hate-speech-CNERG",
        "deepset/bert-base-german-cased-hatespeech-GermEval18Coarse": "deepset",
    }
    # for model in models_de:
    #     toxic_check = ToxicCheck(model, device_i)
    #     print(toxic_check.toxic_prediction(["I hate you", "I love you"]))
    texts = [
        "Why did you sleep last night without letting me know?", "i am thankful for sunshine. #thankful #positive",
        "@user lets fight against #love #peace"
    ]
    text_de = [
        "Warum hast du letzte Nacht geschlafen, ohne mir Bescheid zu sagen?",
        "Ich bin dankbar für Sonnenschein. #dankbar #positiv", "@user lasst uns gegen #liebe #frieden kämpfen"
    ]
    text_tr = [
        "Dün gece neden haber vermeden uyudun?", "Güneş ışığı için minnettarım. #minnettar #pozitif",
        "@user #aşk #barışa karşı savaşalım"
    ]
    models_en = {
        "Detoxifying": "Detoxifying",
        "martin-ha/toxic-comment-model": "martin-ha",
        "nicholasKluge/ToxicityModel": "nicholasKluge",
        # "s-nlp/roberta_first_toxicity_classifier": "s-nlp",
        # "tillschwoerer/roberta-base-finetuned-toxic-comment-detection": "tillschwoerer",
    }
    # for model in models_en:
    #     toxic_check = Detoxifying(device_i)
    #     print(toxic_check.toxic_prediction(["I hate you", "I love you"]))
    model_de = {
        "EIStakovskii/german_toxicity_classifier_plus_v2": "EIStakovskii",
        # "ankekat1000/toxic-bert-german": "ankekat1000",
        # "airKlizz/gbert-base-germeval21-toxic-with-data-augmentation": "airKlizz",
    }
    # for model in model_de:
    #     toxic_check = Detoxifying(device_i)
    #     print(toxic_check.toxic_prediction(["I hate you", "I love you"]))

