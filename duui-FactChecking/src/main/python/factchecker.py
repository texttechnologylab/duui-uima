from transformers import GPT2LMHeadModel, GPT2Tokenizer
# from fact_checking import FactChecker
from utils import convert_to_json
from evaluator import get_evaluator
from nubia_score import Nubia
import torch
from typing import List


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
