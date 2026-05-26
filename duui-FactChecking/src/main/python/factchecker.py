from transformers import GPT2LMHeadModel, GPT2Tokenizer
# from fact_checking import FactChecker
from utils import convert_to_json
from evaluator import get_evaluator
from nubia_score import Nubia
import torch
from typing import List, Dict
from transformers import BertForSequenceClassification, BertTokenizer
from scipy.special import softmax
import numpy as np
from minicheck import MiniCheck


# class FactCheck:
#     def __init__(self):
#         self.model = GPT2LMHeadModel.from_pretrained('gpt2')
#         self.tokenizer = GPT2Tokenizer.from_pretrained('fractalego/fact-checking')
#         self.fact_checker = FactChecker(self.model, self.tokenizer)
#
#     def check(self, claim, evidence):
#         return self.fact_checker.validate_with_replicas(evidence, claim)
#
#     def all_check(self, claim, evidence):
#         val_data = {
#             "validate": self.fact_checker.validate(evidence, claim),
#             "validate_replica": self.fact_checker.validate_with_replicas(evidence, claim)
#         }
#         return val_data


class NubiaFactCheck:
    def __init__(self):
        print("Loading Nubia...")
        self.nubia = Nubia()
        print("Nubia loaded.")

    def check_i(self, claim: str, evidence: str, six_dim=False, aggregator="agg_two"):
        scores = self.nubia.score(claim, evidence, get_features=True, six_dim=six_dim, aggregator=aggregator)
        labels = {k: v for k, v in scores["features"].items()}
        return {"consistency": scores["nubia_score"], "Labels": labels}

    def check(self, claims: List[str], evidences: List[str], six_dim=False, aggregator="agg_two"):
        out_i = []
        for c, claim_i in enumerate(claims):
            fact_i = evidences[c]
            out_i.append(self.check_i(claim_i, fact_i, six_dim, aggregator))
        return out_i


def nubia_factcheck(src_list, output_list):
    nubia = Nubia()
    scores = nubia.score(src_list, output_list, get_features=True)
    labels = {k: v for k, v in scores["features"].items()}
    return {"nubia_score": scores["nubia_score"], "Labels": labels}


class UniEvalFactCheck:

    def __init__(self, device="cpu"):
        self.device = device
        self.evaluator = get_evaluator("fact", device=self.device)

    def check(self, claim: List[str], evidence: List[str]):
        data = convert_to_json(output_list=claim, src_list=evidence)
        eval_scores = self.evaluator.evaluate(data, print_result=True)
        return eval_scores



class FactMiniCheck:
    def __init__(self, model_name='flan-t5-large', device='cuda:0', cache_dir=None) -> None:
        # model_name can be one of ['roberta-large', 'deberta-v3-large', 'flan-t5-large']
        assert model_name in ['roberta-large', 'deberta-v3-large', 'flan-t5-large']
        self.model = MiniCheck(model_name=model_name, device=device, cache_dir=cache_dir)

    def check(self, docs: List[str], claims: List[str], chunk_size=None) -> List[Dict[str, float]]:
        out_score = []
        pred_label, raw_prob, _, _ = self.model.score(docs=docs, claims=claims, chunk_size=chunk_size)
        for i in range(len(pred_label)):
            label = "Fact"
            score = raw_prob[i]
            # if pred_label[i] == 0:
            #     score = 1 - score
            out_score.append({"consistency": score, "Not Fact": 1 - score})
        return out_score

class TransformerFactCheck:
    def __init__(self, model_name, device) -> None:
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name).to(device)
        self.device = device
        self.class_mapping = {"0": "consistency", "1": "Not Fact"}
        self.labels = list(self.class_mapping.values())
        self.model.eval()

    def check(self, docs: List[str], claims: List[str]) -> List[Dict[str, float]]:
        with torch.no_grad():
            score_list = []
            inputs = self.tokenizer(docs, claims, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
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


# def unieval_factcheck(src_list, output_list, task):
#     data = convert_to_json(output_list=output_list, src_list=src_list)
#     # Initialize evaluator for a specific task
#     evaluator = get_evaluator(task, device="cpu")
#     # Get factual consistency scores
#     eval_scores = evaluator.evaluate(data, print_result=True)
#     return eval_scores


if __name__ == '__main__':
    _evidence = """
    Jane writes code for Huggingface.
    """

    _claim = 'Jane is an engineer.'
    device_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
    evidence1 = """
    Justine Tanya Bateman (born February 19, 1966) is an American writer, producer, and actress . She is best known for her regular role as Mallory Keaton on the sitcom Family Ties (1982 -- 1989). Until recently, Bateman ran a production and consulting company, SECTION 5 . In the fall of 2012, she started studying computer science at UCLA.
    """
    claim1 = 'Justine Bateman is a producer.'
    # factchecking = FactCheck()
    # print(factchecking.all_check(claim1, evidence1))
    # print(factchecking.all_check(_claim, _evidence))
    unicheck = UniEvalFactCheck(device=device_1)
    print(unicheck.check([evidence1], [claim1]))
    nubiacheck = NubiaFactCheck()
    print(nubiacheck.check_i(claim1, evidence1))
