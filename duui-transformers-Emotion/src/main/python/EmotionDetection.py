import torch
import math
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
import numpy as np
from pysentimiento import create_analyzer
from typing import List
import nltk
from emoatlas import EmoScores

from collections import namedtuple
from script import BertForSequenceClassification
from pytorch_transformers import (
    BertTokenizer,
    BertModel,
    BertConfig,
)

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
    },
    "pranaydeeps/EXALT-Baseline": {
        0: "love",
        1: "joy",
        2: "anger",
        3: "fear",
        4: "sadness",
        5: "neutral"
    },
    "boltuix/bert-emotion": {
        0: "sadness",
        1: "anger",
        2: "love",
        3: "surprise",
        4: "fear",
        5: "happiness",
        6: "neutral",
        7: "disgust",
        8: "shame",
        9: "guilt",
        10: "confusion",
        11: "desire",
        12: "sarcasm"
    },
    "MilaNLProc/feel-it-italian-emotion": {
        0: "anger",
        1: "fear",
        2: "joy",
        3: "sadness"
    },
    "cardiffnlp/twitter-roberta-base-emotion-multilabel-latest": {
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
    "finiteautomata/beto-emotion-analysis": {
        0: "others",
        1: "joy",
        2: "sadness",
        3: "anger",
        4: "surprise",
        5: "disgust",
        6: "fear"
    },
    "poltextlab/xlm-roberta-large-pooled-emotions6": {
        0: "anger",
        1: "fear",
        2: "disgust",
        3: "sadness",
        4: "joy",
        5: "neutral"
    },
    "cardiffnlp/twitter-roberta-base-emotion":{
        0: "joy",
        1: "optimism",
        2: "anger",
        3: "sadness"
    },
    "daveni/twitter-xlm-roberta-emotion-es": {
        0: "sadness",
        1: "joy",
        2: "anger",
        3: "surprise",
        4: "disgust",
        5: "fear",
        6: "others"
    },
    "ChrisLalk/German-Emotions": {
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
    "msgfrom96/xlm_emo_multi": {
        0: "anger",
        1: "anticipation",
        2: "disgust",
        3: "fear",
        4: "joy",
        5: "sadness",
        6: "surprise",
        7: "trust",
        8: "love",
        9: "optimism",
        10: "pessimism"
    },
    "cointegrated/rubert-tiny2-cedr-emotion-detection": {
        0: "neutral",
        1: "joy",
        2: "sadness",
        3: "surprise",
        4: "fear",
        5: "anger"
    },
    "Aniemore/rubert-tiny2-russian-emotion-detection": {
        0: "neutral",
        1: "happiness",
        2: "sadness",
        3: "enthusiasm",
        4: "fear",
        5: "anger",
        6: "disgust"
    },
    "AnasAlokla/multilingual_go_emotions": {
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
    "Johnson8187/Chinese-Emotion-Small": {
        0: "neutral",
        1: "concern",
        2: "happiness",
        3: "anger",
        4: "sadness",
        5: "questioning",
        6: "surprise",
        7: "disgust",
    },
    "Johnson8187/Chinese-Emotion": {
        0: "neutral",
        1: "concern",
        2: "happiness",
        3: "anger",
        4: "sadness",
        5: "questioning",
        6: "surprise",
        7: "disgust",
    },
    "Zoopa/emotion-classification-model": {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise"
    },
    "esuriddick/distilbert-base-uncased-finetuned-emotion": {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise"
    },
    "Panda0116/emotion-classification-model": {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise"
    },
    "lordtt13/emo-mobilebert": {
        0: "others",
        1: "happy",
        2: "sad",
        3: "angry"
    },
    "alex-shvets/roberta-large-emopillars-contextual-emocontext": {
        0: 'admiration',
        1: 'amusement',
        2: 'disapproval',
        3: 'disgust',
        4: 'embarrassment',
        5: 'excitement',
        6: 'fear',
        7: 'gratitude',
        8: 'grief',
        9: 'joy',
        10: 'love',
        11: 'nervousness',
        12: 'anger',
        13: 'optimism',
        14: 'pride',
        15: 'realization',
        16: 'relief',
        17: 'remorse',
        18: 'sadness',
        19: 'surprise',
        20: 'neutral',
        21: 'annoyance',
        22: 'approval',
        23: 'caring',
        24: 'confusion',
        25: 'curiosity',
        26: 'desire',
        27: 'disappointment'
    },
    "AdapterHub/bert-base-uncased-pf-emo": {
        0: "others",
        1: "happy",
        2: "sad",
        3: "angry"
    }

}

Sample = namedtuple("Sample", [
    "input_ids",
    "input_mask",
    "segment_ids",
    "label_id",
    "lang"
])

def to_sample(text, label, lang, tokenizer, max_seq_length):
    # Tokenize and convert to ids
    input_tokens = tokenizer.tokenize(text)

    # Account for [CLS] and [SEP] with "- 2"
    if len(input_tokens) > max_seq_length - 2:
        input_tokens = input_tokens[:(max_seq_length - 2)]

    # Add CLS and SEP tokens
    input_tokens = [tokenizer.cls_token] + input_tokens + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    segment_ids = [0 for i in range(len(input_ids))]

    # Initialize padding mask
    input_mask = [1] * len(input_ids)

    # Pad to max_seq_length
    padding_length = max_seq_length - len(input_ids)
    input_ids = input_ids + ([0] * padding_length)
    input_mask = input_mask + ([0] * padding_length)
    segment_ids = segment_ids + ([0] * padding_length)

    return Sample(
        torch.tensor(input_ids),
        torch.tensor(input_mask),
        torch.tensor(segment_ids),
        torch.tensor(label, dtype=torch.float),
        lang
    )


class EmotionClassification:
    def __init__(self, model_name, device, max_seq_length=256):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased",
                                                       do_lower_case=False)
        self.bert_config = BertConfig.from_pretrained("bert-base-multilingual-cased")
        self.bert_model = BertModel.from_pretrained("bert-base-multilingual-cased")

        self.model = BertForSequenceClassification(self.bert_model, self.bert_config, num_labels=5)

        self.model = self.model.to(device)
        # self.model.load_state_dict(torch.load(model_name))
        self.max_seq_length = max_seq_length
        self.emotions_labels = {
            0: "anger",
            1: "anticipation",
            2: "fear",
            3: "joy",
            4: "sadness"
        }
        self.device = device


    def emotion_prediction(self, texts):
        dummy_labels = [np.zeros(5) for _ in texts]
        samples = [to_sample(text, label, "nan", self.tokenizer, self.max_seq_length)
                   for text, label in zip(texts, dummy_labels)]

        results = []

        for sample in samples:
            inputs = {
                'input_ids': sample[0].unsqueeze(0).to(self.device),
                'attention_mask': sample[1].unsqueeze(0).to(self.device),
                'token_type_ids': sample[2].unsqueeze(0).to(self.device),
                'labels': sample[3].unsqueeze(0).to(self.device)
            }
            with torch.no_grad():
                logits = self.model(**inputs)

            probs = logits[0].detach().cpu().numpy()
            probs = softmax(probs).tolist()
            # probs = torch.sigmoid(logits).cpu().numpy().flatten()
            # probs to float
            emotion_scores = dict(zip(list(self.emotions_labels.values()), probs))
            results.append(emotion_scores)
        return results




class EmotionCheck:
    def __init__(self, model_name: str, device='cuda:0'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
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

class PolyTextLabEmotionModel:
    def __init__(self, model_name: str, device='cuda:0', token_reader="default"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        self.model = AutoModelForSequenceClassification.from_pretrained("poltextlab/xlm-roberta-large-pooled-MORES", token=token_reader).to(device)
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
                    score_dict_i[self.labels[ranking[i]]] = float(score_i[ranking[i]])
                score_list.append(score_dict_i)
        return score_list


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