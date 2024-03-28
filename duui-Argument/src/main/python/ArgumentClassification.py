from transformers import BertForSequenceClassification, BertTokenizer, AutoTokenizer, AutoModelForSequenceClassification
import torch
from ukp_classes import InputExample, convert_examples_to_features
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from awq import AutoAWQForCausalLM
import numpy as np
from scipy.special import softmax
import json
from openai import OpenAI
import os
import time


num_labels = 3
label_list = ["NoArgument", "Argument_against", "Argument_for"]
max_seq_length = 64
eval_batch_size = 8


class TransformerArgument:
    def __init__(self, model_name: str, device: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()
        self.class_mapping = self.model.config.id2label
        self.labels = list(self.class_mapping.values())
        self.labels = [str(label).lower() for label in self.labels]

    def predict(self, texts, topic):
        input_text = []
        for text in texts:
            text = f"Topic={topic}, Argument={text}"
            input_text.append(text)
        with torch.no_grad():
            score_list = []
            inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
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


class UkpArgument:
    def __init__(self, model_name: str, device: str):
        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
        self.device = device
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)
        self.model.eval()

    def predict(self, texts, topic):
        input_examples = self.text_topic_to_input_examples(texts, topic)
        eval_features = convert_examples_to_features(input_examples, label_list, max_seq_length, self.tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)
        # predicted_labels = []
        score_list = []
        with torch.no_grad():
            for input_ids, input_mask, segment_ids in eval_dataloader:
                input_ids = input_ids.to(self.device)
                input_mask = input_mask.to(self.device)
                segment_ids = segment_ids.to(self.device)
                logits = self.model(input_ids, segment_ids, input_mask)
                logits = logits.logits.detach().cpu().numpy()
                score_i = softmax(logits, axis=1)
                ranking = np.argsort(score_i)[::-1]
                # for prediction in np.argmax(logits, axis=1):
                #     predicted_labels.append(label_list[prediction])
                for i in range(logits.shape[0]):
                    score_dict_i = {}
                    for j in ranking[i]:
                        score_dict_i[label_list[j]] = float(score_i[i][j])
                    #sorted score
                    score_dict_i = {k: v for k, v in sorted(score_dict_i.items(), key=lambda item: item[1], reverse=True)}
                    score_list.append(score_dict_i)
        return score_list

    def text_topic_to_input_examples(self, texts, topic):
        input_examples = []
        for c, text in enumerate(texts):
            input_examples.append(InputExample(text_a=topic, text_b=text, label='NoArgument'))
        return input_examples


class BlokeArgument:
    def __init__(self, model_name: str, device: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, fuse_layers=True, trust_remote_code=False, safetensors=True, cache_dir="/storage/nlp/huggingface/models")
        self.device = device
        self.model = AutoAWQForCausalLM.from_pretrained(model_name, trust_remote_code=False, cache_dir="/storage/nlp/huggingface/models").to(device)
        self.model.eval()

    def predict(self, prompts):
        predicted_labels = []
        with torch.no_grad():
            for prompt in prompts:
                prompt_template = f'''<s>[INST] {prompt}
                [/INST]'''
                tokens = self.tokenizer(
                    prompt_template,
                    return_tensors='pt'
                ).input_ids.to(self.device)
                generation_output = self.model.generate(
                    tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    top_k=40,
                    max_new_tokens=512
                )
                predicted_labels.append(self.tokenizer.decode(generation_output[0]))
        return predicted_labels


class ChatGPT:
    def __init__(self, model_name, key_chatgpt, temperature=0.7, max_tokens=100):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=key_chatgpt)

    def predict(self, texts, topic):
        output = []
        content = "You're a skilled argument classifier. Determine if Argument A supports, opposes, or is neutral regarding Topic B."
        for c, text in enumerate(texts):
            out_i = {}
            try:
                user_input = f"Argument A:'{text}'\nTopic B:'{topic}'"
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
                    if function_call.name == 'argument_classification':
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
            "name": "argument_classification",
            "description": "Determine if Argument A supports, opposes, or is neutral regarding Topic B, under label. Explain your answer in a maximum of 20 words under 'reason'. Also state how confident you are under 'confidence' between 0-100.",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "integer",
                        "enum": [0, 1, 2],
                        "description": "0: supports, 1: opposes, 2: neutral",
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