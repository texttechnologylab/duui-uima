from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time
import json
from openai import OpenAI


class EntailmentCheck:
    def __init__(self, model_name, device):
        self.device = device
        if model_name=="soumyasanyal/entailment-verifier-xxl":
            self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl", cache_dir="/storage/nlp/huggingface", truncation=True, padding=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/storage/nlp/huggingface", truncation=True, padding=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir="/storage/nlp/huggingface")
        self.model.to(device)
        self.model.eval()

    def entailment_check(self, premise, hypothesis):
        prompts = []
        for i in range(len(premise)):
            prompt = f"Premise: {premise[i]}\nHypothesis: {hypothesis[i]}\nGiven the premise, is the hypothesis correct?\nAnswer:"
            prompts.append(prompt)
        input_ids = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).input_ids
        scores = self.get_score(input_ids)
        entailment = []
        for i in range(len(premise)):
            entailment.append({
                "entailment": scores[i].item(),
                "non_entailment": 1 - scores[i].item()
            })
        # entailment = {
        #     "entailment": scores,
        #     "non_entailment": 1 - scores
        # }
        return entailment

    def get_score(self, input_ids):
        pos_ids = self.tokenizer('Yes').input_ids
        neg_ids = self.tokenizer('No').input_ids
        pos_id = pos_ids[0]
        neg_id = neg_ids[0]

        zeros = torch.zeros((input_ids.size(0), 1), dtype=torch.long)
        logits = self.model(input_ids, decoder_input_ids=torch.zeros((input_ids.size(0), 1), dtype=torch.long)).logits
        pos_logits = logits[:, 0, pos_id]
        neg_logits = logits[:, 0, neg_id]
        posneg_logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits.unsqueeze(-1)], dim=1)
        scores = torch.nn.functional.softmax(posneg_logits, dim=1)[:, 0]
        return scores


class ChatGPT:
    def __init__(self, model_name, key_chatgpt, temperature=1.0, max_tokens=100):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=key_chatgpt)

    def entailment_check(self, premises, hypothesis):
        output = []
        content = "You are now a tool for entailment detection. Classify whether sentence B entails sentence A."
        for c, premise in enumerate(premises):
            hypotheses = hypothesis[c]
            out_i = {}
            try:
                user_input = f"Text A:'{premise}' Hypothesis B:'{hypotheses}'"
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": content},
                        {"role": "user", "content": user_input}
                    ],
                    temperature=self.temperature,
                    tools=tools,
                )
                if response.choices[0].finish_reason == 'tool_calls':
                    function_call = response.choices[0].message.tool_calls[0].function
                    if function_call.name == 'entailment_check':
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
            "name": "entailment_check",
            "description": "Classify whether sentence B entails sentence A as '1' for entailment, '0' for non-entailment. Save confidence ranging from 0 to 100 under 'confidence'. Provide a brief reason in max. 20 words under 'reason'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "integer",
                        "enum": [0, 1],
                        "description": "0: Non-entailment, 1: Entailment",
                    },
                    "confidence": {
                        "type": "string",
                        "description": "The confidence value between 0-100",
                    },
                    "reason": {
                        "type": "string",
                        "description": "The reason for your decision in a maximum of 20 words",
                    }
                },
                "required": ["label", "confidence", "reason"],
            },
        }
    },
]


if __name__ == '__main__':
    premise = ["A fossil fuel is a kind of natural resource. Coal is a kind of fossil fuel.", "A fossil fuel is a kind of natural resource. Coal is a kind of fossil fuel."]
    hypothesis = ["Coal is not a kind of natural resource.", "Coal is a kind of natural resource."]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # model_name = "soumyasanyal/entailment-verifier-xxl"
    model_name = "google/flan-t5-base"
    # entailment_i = EntailmentCheck(model_name, device)
    # entailment_out = entailment_i.entailment_check(premise, hypothesis)

    with open("data/config_chatgpt.json", "r") as f:
        key_chatgpt = json.load(f)["key"]
    model = "gpt-3.5-turbo-16k"
    chatgpt = ChatGPT(model, key_chatgpt)
    chatgpt_out = chatgpt.entailment_check(premise, hypothesis)
    print(chatgpt_out)
