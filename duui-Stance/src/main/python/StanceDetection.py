from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.special import softmax
import numpy as np
from transformers import pipeline
import json
import time
from openai import OpenAI


class TransformerStance:
    def __init__(self, model_name, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.id2label = {0: 'AGAINST', 1: 'FAVOR', 2: 'NONE'}

    def predict(self, texts, hypothesis_template):
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
                score_dict_i[self.id2label[ranking[i]]] = float(score_i[ranking[i]])
            score_list.append(score_dict_i)
        return score_list


class zeroshotClassification:
    def __init__(self, model_name, device):
        self.classifier = pipeline("zero-shot-classification", model=model_name, device=device, batch_size=32)
        self.device = device

    def predict(self, texts, hypothesis_template):
        score_list = []
        # classify the documents
        labels = ['supports', 'opposes', 'is neutral towards']
        res = self.classifier(texts, labels, hypothesis_template=hypothesis_template, multi_label=True)
        for res_i in res:
            score_dict_i = {}
            labels_i = res_i['labels']
            scores_i = res_i['scores']
            for i in range(len(labels_i)):
                score_dict_i[labels_i[i]] = float(scores_i[i])
            # sort the dictionary
            score_dict_i = dict(sorted(score_dict_i.items(), key=lambda item: item[1], reverse=True))
            score_list.append(score_dict_i)
        return score_list


class ChatGPT:
    def __init__(self, model_name, key_chatgpt, temperature=0.7, max_tokens=100):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=key_chatgpt)

    def predict(self, texts, hypothesis):
        output = []
        content = "As a Stance classification expert, evaluate if Text A aligns with Hypothesis B."
        for c, text in enumerate(texts):
            out_i = {}
            try:
                user_input = f"Text A:'{text}' Hypothesis B:'{hypothesis}'"
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": content},
                        {"role": "user", "content": user_input}
                    ],
                    temperature=self.temperature,
                    tools=tools,
                )
                time.sleep(0.5)
                if response.choices[0].finish_reason == 'tool_calls':
                    function_call = response.choices[0].message.tool_calls[0].function
                    if function_call.name == 'stance_classification':
                        out_i = json.loads(function_call.arguments)
                        out_i["error"] = False
                output.append(out_i)
            except Exception as ex:
                print(ex)
                output.append({"error": True})
        return output


tools = [
    {
        "type": "function",
        "function": {
            "name": "stance_classification",
            "description": "Evaluate if Text A aligns with Hypothesis B. Provide 'label' (0 for neutral, 1 for support, 2 for oppose), 'confidence' (0-100), and 'reason' (up to 20 words).",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "integer",
                        "enum": [0, 1, 2],
                        "description": "0: neutral, 1: support, 2: oppose",
                    },
                    "confidence": {
                        "type": "string",
                        "description": "The confidence value between 0-100",
                    },
                    "reason": {
                        "type": "string",
                        "description": "The reason for the classification in a maximum of 20 words",
                    }
                },
                "required": ["label", "confidence", "reason"],
            },
        }
    },
]