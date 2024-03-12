import torch
import math
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
import numpy as np
from typing import List


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


map_dreamToNormal = {
    "AN": "anger",
    "SD": "sadness",
    "AP": "fear",
    "CO": "fear",
    "HA": "joy",

}

map_dreambank = {
    "AN": "anger",
    "SD": "sadness",
    "AP": "apprehension",
    "co": "confusion",
    "ha": "happiness",
}


class EmotionCheck:
    def __init__(self, model_name: str, device='cuda:0'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.class_mapping = self.model.config.id2label
        self.labels = list(self.class_mapping.values())

    def emotion_prediction(self, texts: List[str]):
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
                    if self.labels[ranking[i]] in map_dreamToNormal:
                        score_dict_i[map_dreamToNormal[self.labels[ranking[i]]]] = float(score_i[ranking[i]])
                    else:
                        score_dict_i[self.labels[ranking[i]]] = float(score_i[ranking[i]])
                score_list.append(score_dict_i)
        return score_list


if __name__ == '__main__':
    device_i = "cuda:0" if torch.cuda.is_available() else "cpu"
    models = [
        "02shanky/finetuned-twitter-xlm-roberta-base-emotion",
        "DReAMy-lib/xlm-roberta-large-DreamBank-emotion-presence",
        "MilaNLProc/xlm-emo-t"
    ]
    texts_text = [
        "I am very happy today!", "I am very sad today!", "I am very angry today!", "I am very surprised today!", "I am very disgusted today!", "I am very fearful today!"
    ]
    texts_de = [
        "Ich bin heute sehr glücklich!", "Ich bin heute sehr traurig!", "Ich bin heute sehr wütend!", "Ich bin heute sehr überrascht!", "Ich bin heute sehr angewidert!", "Ich bin heute sehr ängstlich!"
    ]
    texts_tr = [
        "Bugün çok mutluyum!", "Bugün çok üzgünüm!", "Bugün çok kızgınım!", "Bugün çok şaşırdım!", "Bugün çok iğrendim!", "Bugün çok korktum!"
    ]
    EmotionCheck = EmotionCheck(models[2], device_i)
    print(EmotionCheck.emotion_prediction(texts_text))
    print(EmotionCheck.emotion_prediction(texts_de))
    print(EmotionCheck.emotion_prediction(texts_tr))

