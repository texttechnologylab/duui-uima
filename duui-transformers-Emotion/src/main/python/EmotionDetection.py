import torch
import math
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
import numpy as np
from pysentimiento import create_analyzer
from typing import List
import nltk
from emoatlas import EmoScores

# nltk.download('all', download_dir="nltk_data")
nltk.data.path.append("nltk_data")

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


models_emotion = {
    "DReAMy": "DReAMy-lib/xlm-roberta-large-DreamBank-emotion-presence",
    "02shanky": "02shanky/finetuned-twitter-xlm-roberta-base-emotion",
    "MilaNLProc": "MilaNLProc/xlm-emo-t",
    "CardiffNLP": "cardiffnlp/twitter-roberta-base-emotion-latest",
    "pysentimiento": "pysentimiento",
    "j-hartmann": "j-hartmann/emotion-english-distilroberta-base",
    "ActivationAI": "ActivationAI/distilbert-base-uncased-finetuned-emotion",
    "SamLowe": "SamLowe/roberta-base-go_emotions",
    "michellejieli": "michellejieli/emotion_text_classifier",
    "EmoAtlas": "EmoAtlas",
    "MRM8488": "mrm8488/t5-base-finetuned-emotion"
}
map_emotion = {
    "DReAMy-lib/xlm-roberta-large-DreamBank-emotion-presence": {
        0: "Anger",
        1: "Apprehension",
        2: "Sadness",
        3: "Confusion",
        4: "Happiness",
    },
    "02shanky/finetuned-twitter-xlm-roberta-base-emotion": {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise"
    },
    "MilaNLProc/xlm-emo-t": {
        0: "anger",
        1: "fear",
        2: "joy",
        3: "sadness",
    },
    "cardiffnlp/twitter-roberta-base-emotion-latest": {
        0: "anger",
        1: "anticipation",
        2: "disgust",
        3: "fear",
        4: "joy",
        5: "love",
        6: "optimism",
        7: "pessimism",
        8: "sadness",
        9: "surprise",
        10: "trust"
    },
    "j-hartmann/emotion-english-distilroberta-base": {
        0: "anger",
        1: "disgust",
        2: "fear",
        3: "joy",
        4: "neutral",
        5: "sadness",
        6: "surprise"
    },
    "ActivationAI/distilbert-base-uncased-finetuned-emotion": {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise"
    },
    "SamLowe/roberta-base-go_emotions": {
        0: "admiration",
        1: "amusement",
        2: "anger",
        3: "annoyance",
        4: "approval",
        5: "caring",
        6: "confusion",
        7: "curiosity",
        8: "desire",
        9: "disappointment",
        10: "disapproval",
        11: "disgust",
        12: "embarrassment",
        13: "excitement",
        14: "fear",
        15: "gratitude",
        16: "grief",
        17: "joy",
        18: "love",
        19: "nervousness",
        20: "optimism",
        21: "pride",
        22: "realization",
        23: "relief",
        24: "remorse",
        25: "sadness",
        26: "surprise",
        27: "neutral"
    },
    "michellejieli/emotion_text_classifier": {
        0: "anger",
        1: "disgust",
        2: "fear",
        3: "joy",
        4: "neutral",
        5: "sadness",
        6: "surprise"
    },
    "mrm8488/t5-base-finetuned-emotion": {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise"
    },
    "finiteautomata/bertweet-base-emotion-analysis": {
         0: "others",
         1: "joy",
         2: "sadness",
         3: "anger",
         4: "surprise",
         5: "disgust",
         6: "fear"
    }
}




class EmotionCheck:
    def __init__(self, model_name: str, device='cuda:0'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/storage/nlp/huggingface/models")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir="/storage/nlp/huggingface/models").to(device)
        self.class_mapping = self.model.config.id2label
        self.labels = list(map_emotion[model_name].values())

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
                    score_dict_i[self.labels[ranking[i]]] = float(score_i[ranking[i]])
                score_list.append(score_dict_i)
        return score_list


class PySentimientoCheck:
    def __init__(self, lang="en"):
        self.analyzer = create_analyzer(task="emotion", lang=lang)

    def emotion_prediction(self, texts: List[str]):
        output = []
        for text in texts:
            prediction_i = self.analyzer.predict(text)
            out_i = prediction_i.probas
            output.append(out_i)
        return output


class EmoAtlas:
    def __init__(self, language="english"):
        # nltk.download('wordnet')
        self.emo_atlas = EmoScores(language)

    def emotion_prediction(self, texts: List[str]):
        output = []
        for text in texts:
            prediction_i = self.emo_atlas.emotions(text, normalization_strategy="emotion_words")
            out_i = prediction_i
            output.append(out_i)
        return output


if __name__ == '__main__':
    emo_model = EmoAtlas()
    texts = [ "I am very happy today!", "I am very sad today!", "I am very angry today!", "I am very surprised today!", "I am very disgusted today!", "I am very fearful today!"]
    print(emo_model.emotion_prediction(texts))
    device_i = "cuda:0" if torch.cuda.is_available() else "cpu"
    pysentimiento_model = PySentimientoCheck("en")
    print(pysentimiento_model.emotion_prediction(texts))
    for model_i in models_emotion:
        print(model_i)
        EmotionCheck_i = EmotionCheck(models_emotion[model_i], device_i)
        print(EmotionCheck_i.emotion_prediction(texts))
    # models = [
    #     "02shanky/finetuned-twitter-xlm-roberta-base-emotion",
    #     "DReAMy-lib/xlm-roberta-large-DreamBank-emotion-presence",
    #     "MilaNLProc/xlm-emo-t"
    # ]
    # texts_text = [
    #     "I am very happy today!", "I am very sad today!", "I am very angry today!", "I am very surprised today!", "I am very disgusted today!", "I am very fearful today!"
    # ]
    # texts_de = [
    #     "Ich bin heute sehr glücklich!", "Ich bin heute sehr traurig!", "Ich bin heute sehr wütend!", "Ich bin heute sehr überrascht!", "Ich bin heute sehr angewidert!", "Ich bin heute sehr ängstlich!"
    # ]
    # texts_tr = [
    #     "Bugün çok mutluyum!", "Bugün çok üzgünüm!", "Bugün çok kızgınım!", "Bugün çok şaşırdım!", "Bugün çok iğrendim!", "Bugün çok korktum!"
    # ]
    # EmotionCheck = EmotionCheck(models[2], device_i)
    # print(EmotionCheck.emotion_prediction(texts_text))
    # print(EmotionCheck.emotion_prediction(texts_de))
    # print(EmotionCheck.emotion_prediction(texts_tr))

    # models = {
    #     "DReAMy": "DReAMy-lib/xlm-roberta-large-DreamBank-emotion-presence",
    #     "02shanky": "02shanky/finetuned-twitter-xlm-roberta-base-emotion",
    #     "MilaNLProc": "MilaNLProc/xlm-emo-t",
    #     "CardiffNLP": "cardiffnlp/twitter-roberta-base-emotion-latest",
    #     "pysentimiento": "pysentimiento",
    #     "j-hartmann": "j-hartmann/emotion-english-distilroberta-base"
    # }
    # texts = [
    #     # "🚀 Science is our gateway to understanding the universe! 🌌 Take our survey and share your thoughts on the future of scientific discovery. Your input can shape tomorrow's innovations! 🧬🔬 #ScienceSurvey #FutureOfScience"
    #     "“America, ride or die” — Elon Musk"
    # ]
    # output_save = {}
    # for model_i in models:
    #     print(model_i)
    #     if model_i == "pysentimiento":
    #         test_sentiment = PySentimientoCheck("en")
    #         output_save[model_i] = test_sentiment.emotion_prediction(texts)
    #     else:
    #         EmotionCheck_i = EmotionCheck(models[model_i], device_i)
    #         output_save[model_i] = EmotionCheck_i.emotion_prediction(texts)
    # save_json(output_save, "/storage/projects/bagci/data/Musk_EmotionCheck.json", True)
