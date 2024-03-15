from huggingface_hub import hf_hub_download
import fasttext
from typing import List
from iso639 import languages
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import gcld3
import langdetect
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from scipy.special import softmax
import numpy as np


def get_lang_detector(nlp, name):
    return LanguageDetector()


class LanguageDetection:
    def __init__(self):
        self.model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
        self.model = fasttext.load_model(self.model_path)

    def language_prediction(self, text: List[str]):
        lang_text = self.model.predict(text, k=5)
        lang_out = []
        for c, lang in enumerate(lang_text[0]):
            land_dict = {}
            for c2, lang_i in enumerate(lang):
                lang_pred = lang_i.split("__label__")[1].split("_")[0]
                if lang_pred in languages.part3:
                    lang_part1 = languages.part3[lang_pred].part1
                    if lang_part1 != "":
                        lang_pred = lang_part1
                land_dict[lang_pred] = float(lang_text[1][c][c2])
            # land_dict = {k: v for k, v in sorted(land_dict.items(), key=lambda item: item[1], reverse=True)}
            lang_out.append(land_dict)
        return lang_out


class LanguageCheck:
    def __init__(self, model_name: str, device='cuda:0'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/storage/nlp/huggingface/models")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                        cache_dir="/storage/nlp/huggingface/models").to(
            device)
        self.class_mapping = self.model.config.id2label
        self.labels = list(self.class_mapping.values())
        for c, label in enumerate(self.labels):
            if "-" in label:
                self.labels[c] = label.split("-")[0]

    def language_prediction(self, texts: List[str]):
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


class LanguageIdentification:
    def __init__(self, model_name:str):
        self.model_glcd3 = gcld3.NNetLanguageIdentifier(min_num_bytes=1, max_num_bytes=200000)
        self.model_fasttext = fasttext.load_model("lid.176.bin")
        self.model_spacy = spacy.load("en_core_web_sm")
        Language.factory("language_detector", func=get_lang_detector)
        self.model_spacy.add_pipe('language_detector', last=True)
        self.model_google = langdetect
        self.model_name = model_name

    def glc3d_identification(self, text):
        result = self.model_glcd3.FindTopNMostFreqLangs(text, 5)
        res_back = {}
        for c, res_i in enumerate(result):
            res_back[res_i.language] = res_i.probability
        res_back = {k: v for k, v in sorted(res_back.items(), key=lambda item: item[1], reverse=True)}
        return res_back

    def fastText_identification(self, text):
        result = self.model_fasttext.predict(text, k=5)
        res_back = {}
        for c, lang in enumerate(result[0]):
            res_back[lang.split("__label__")[1].split("_")[0]] = float(result[1][c])
        res_back = {k: v for k, v in sorted(res_back.items(), key=lambda item: item[1], reverse=True)}
        return res_back

    def spacy_identification(self, text):
        result = self.model_spacy(text)
        res_back = {result._.language["language"]: result._.language["score"]}
        return res_back

    def google_identification(self, text):
        result = self.model_google.detect_langs(text)
        res_back = {}
        for res_i in result:
            res_back[res_i.lang] = res_i.prob
        res_back = {k: v for k, v in sorted(res_back.items(), key=lambda item: item[1], reverse=True)}
        return res_back

    def lang_prediction(self, texts):
        lang_out = []
        for text in texts:
            match self.model_name:
                case "glc3d":
                    lang_out.append(self.glc3d_identification(text))
                case "fasttext":
                    lang_out.append(self.fastText_identification(text))
                case "spacy":
                    lang_out.append(self.spacy_identification(text))
                case "google":
                    lang_out.append(self.google_identification(text))
        return lang_out
