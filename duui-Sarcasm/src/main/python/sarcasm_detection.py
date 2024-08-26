from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import string
from typing import List
import torch
import numpy as np
from scipy.special import softmax

models_sarcasm = {
    "Multilingual_Sarcasm_Detector": "helinivan/multilingual-sarcasm-detector",
    # "T5-base_fine-tuned_for_Sarcasm_Detection": "mrm8488/t5-base-finetuned-sarcasm-twitter"
}

map_sarcasm = {
    "helinivan/multilingual-sarcasm-detector": {
        0: "NOT SARCASM",
        1: "SARCASM"
    },
    # "mrm8488/t5-base-finetuned-sarcasm-twitter": {
    #     0: "NOT SARCASM",
    #     1: "SARCASM"
    # }
}


class SarcasmCheck:
    def __init__(self, model_name: str, device='cuda:0', cache_dir="/storage/nlp/huggingface/models"):
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir).to(device)
        self.class_mapping = self.model.config.id2label
        self.labels = list(map_sarcasm[model_name].values())

    def preprocess_data(self, texts: List[str]):
        texts = [text.lower().translate(str.maketrans('', '', string.punctuation)) for text in texts]
        return texts

    def sarcasm_prediction(self, texts):
        if self.model_name == "helinivan/multilingual-sarcasm-detector":
            texts = self.preprocess_data(texts)
        with torch.no_grad():
            score_list = []
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
                self.device)
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
    device_i = "cuda:1" if torch.cuda.is_available() else "cpu"
    sarcasm = SarcasmCheck(models_sarcasm["T5-base_fine-tuned_for_Sarcasm_Detection"], device_i)
    texts = ["I love the weather today", "I hate the weather today", "Great, it broke the first day..."]
    print(sarcasm.sarcasm_prediction(texts))
