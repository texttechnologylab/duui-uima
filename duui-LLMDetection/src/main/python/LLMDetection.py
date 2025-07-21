import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from abc import ABC, abstractmethod
from typing import TypedDict
import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import numpy as np
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer


accuracy = evaluate.load("accuracy")

class PredictionResults(TypedDict):
    prediction: list[float]


class DetectorABC(ABC):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
            device: str | torch.device = ("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.tokenizer = tokenizer

    @abstractmethod
    def tokenize(self, texts: list[str]) -> BatchEncoding: ...

    @abstractmethod
    def process(self, inputs: dict) -> PredictionResults: ...


class Radar(DetectorABC):
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(
            AutoTokenizer.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B"),
            device=device,
        )
        self.device = torch.device(device)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "TrustSafeAI/RADAR-Vicuna-7B",
        )
        self.model.eval()
        self.model.to(self.device)

    def tokenize(self, texts: list[str]) -> BatchEncoding:
        return self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=512,
            return_length=True,
        )

    @torch.inference_mode()
    def predict(self, inputs: dict) -> list[float]:
        encoding = self.tokenizer.pad(inputs, return_tensors="pt").to(self.device)
        outputs = self.model(**encoding)
        output_probs = F.log_softmax(outputs.logits, -1)[:, 1].exp().tolist()
        return output_probs

    def process(self, inputs: dict) -> dict[str, list[float]]:
        return {
            "prediction": self.predict(
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                }
            )
        }

    @torch.inference_mode()
    def process_texts(self, texts: list[str]) -> list[dict[str, float]]:
        with torch.no_grad():
            encoding = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            outputs = self.model(**encoding)
            output_probs = F.log_softmax(outputs.logits, -1)[:, 0].exp().tolist()
            output_prob_list = [{"LLM": prob, "Human": 1 - prob} for prob in output_probs]
        return output_prob_list

def compute_metrics_acc(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)


class RoBERTaClassifier(DetectorABC):
    def __init__(
            self,
            model_name="Hello-SimpleAI/chatgpt-detector-roberta",
            device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(
            AutoTokenizer.from_pretrained(model_name),
            device=device,
        )
        self.device = torch.device(device)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

    def tokenize(self, texts: list[str]) -> BatchEncoding:
        return self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=512,
            return_length=True,
        )

    @torch.inference_mode()
    def predict(self, inputs: dict) -> list[float]:
        encoding = self.tokenizer.pad(inputs, return_tensors="pt").to(self.device)
        outputs = self.model(**encoding)
        probs = outputs.logits
        return probs[:, 1].detach().cpu().flatten().tolist()

    def process(self, inputs: dict) -> dict[str, list[float]]:
        return {
            "prediction": self.predict(
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                }
            )
        }

    @torch.inference_mode()
    def process_texts(self, texts: list[str]) -> list[dict[str, float]]:
        encoding = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        outputs = self.model(**encoding)
        probs = F.log_softmax(outputs.logits, -1)[:, 1].exp().tolist()
        prob_list = [{"LLM": prob, "Human": 1 - prob} for prob in probs]
        return prob_list


ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
softmax_fn = torch.nn.Softmax(dim=-1)


def perplexity(
        encoding: BatchEncoding,
        logits: torch.Tensor,
        median: bool = False,
        temperature: float = 1.0,
):
    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

    if median:
        ce_nan = ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels).masked_fill(
            ~shifted_attention_mask.bool(), float("nan")
        )
        ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)

    else:
        ppl = (
                      ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels)
                      * shifted_attention_mask
              ).sum(1) / shifted_attention_mask.sum(1)
        ppl = ppl.to("cpu").float().numpy()

    return ppl


def entropy(
        p_logits: torch.Tensor,
        q_logits: torch.Tensor,
        encoding: BatchEncoding,
        pad_token_id: int,
        median: bool = False,
        sample_p: bool = False,
        temperature: float = 1.0,
):
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]

    if not temperature:
        p_scores, q_scores = p_logits, q_logits
    else:
        p_scores, q_scores = p_logits / temperature, q_logits / temperature

    p_proba = softmax_fn(p_scores).view(-1, vocab_size)

    if sample_p:
        p_proba = torch.multinomial(
            p_proba.view(-1, vocab_size), replacement=True, num_samples=1
        ).view(-1)

    q_scores = q_scores.view(-1, vocab_size)

    ce = ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

    if median:
        ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
        agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        agg_ce = (
            ((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy()
        )

    return agg_ce

GLOBAL_BINOCULARS_THRESHOLD = (
    0.9015310749276843  # selected using Falcon-7B and Falcon-7B-Instruct at bfloat16
)
DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1

class Binoculars(DetectorABC):
    def __init__(
            self,
            observer_name_or_path: str = "tiiuae/falcon-7b",
            performer_name_or_path: str = "tiiuae/falcon-7b-instruct",
            use_bfloat16: bool = True,
            max_token_observed: int = 512,
    ) -> None:
        super().__init__(AutoTokenizer.from_pretrained(observer_name_or_path))
        self.observer_model = AutoModelForCausalLM.from_pretrained(
            observer_name_or_path,
            device_map={"": DEVICE_1},
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
        )
        self.performer_model = AutoModelForCausalLM.from_pretrained(
            performer_name_or_path,
            device_map={"": DEVICE_2},
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
        )

        self.observer_model.eval()
        self.performer_model.eval()

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_token_observed = max_token_observed

        tokenizer = self.tokenizer

        def _tokenize(texts: list[str]) -> BatchEncoding:
            return tokenizer(
                texts,
                padding=False,
                truncation=True,
                max_length=max_token_observed,
                return_length=True,
                return_token_type_ids=False,
            )

        self.tokenize = _tokenize

    def tokenize(self):
        pass

    @torch.inference_mode()
    def _get_logits(
            self, encodings: BatchEncoding
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observer_logits = self.observer_model(
            **encodings.to(self.observer_model.device)
        ).logits
        performer_logits = self.performer_model(
            **encodings.to(self.performer_model.device)
        ).logits
        if DEVICE_1 != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    @torch.inference_mode()
    def predict(self, inputs: dict) -> list[float]:
        encodings = self.tokenizer.pad(inputs, return_tensors="pt")
        observer_logits, performer_logits = self._get_logits(encodings)
        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(
            observer_logits.to(DEVICE_1),
            performer_logits.to(DEVICE_1),
            encodings.to(DEVICE_1),
            self.tokenizer.pad_token_id,  # type: ignore
        )
        binoculars_scores = ppl / x_ppl
        return binoculars_scores.tolist()

    def process(self, inputs: dict) -> dict[str, list[float]]:
        return {
            "prediction": self.predict(
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                }
            )
        }

    @torch.inference_mode()
    def process_texts(self, texts: list[str]) -> list[float]:
        encodings = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False,
        ).to(self.device)
        observer_logits, performer_logits = self._get_logits(encodings)
        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(
            observer_logits.to(DEVICE_1),
            performer_logits.to(DEVICE_1),
            encodings.to(DEVICE_1),
            self.tokenizer.pad_token_id,  # type: ignore
        )
        binoculars_scores = ppl / x_ppl
        return binoculars_scores.tolist()




if __name__ == '__main__':
    texts = [
        "This is a test sentence.",
        "Another example of a sentence to check.",
        "Yet another text for evaluation."
        "This text is generated with an AI model.",
        "I am not a chatbot.",
        "Japan's ruling coalition has lost its majority in the country's upper house, but Prime Minister Shigeru Ishiba has said he has no plans to quit."
    ]
    # detector = Radar()
    # results = detector.process_texts(texts)
    # for text, score in zip(texts, results):
    #     print(f"Text: {text}\nRadar Score: {score:.4f}\n")
    #
    # detectorRoberta = RoBERTaClassifier()
    # resultsRoberta = detectorRoberta.process_texts(texts)
    # for text, score in zip(texts, resultsRoberta):
    #     print(f"Text: {text}\nRoBERTa Score: {score}\n")


    binoculars_detector = Binoculars()
    binoculars_results = binoculars_detector.process_texts(texts)
    for text, score in zip(texts, binoculars_results):
        print(f"Text: {text}\nBinoculars Score: {score:.4f}\n")


