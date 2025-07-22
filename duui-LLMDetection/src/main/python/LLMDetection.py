import torch.nn.functional as F
from transformers.tokenization_utils_base import BatchEncoding
from abc import ABC, abstractmethod
from typing import TypedDict
import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import numpy as np
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, PreTrainedModel
import inspect


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
            observer_name_or_path: str = "tiiuae/Falcon3-1B-Base",
            performer_name_or_path: str = "tiiuae/Falcon3-1B-Instruct",
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
    def process_texts(self, texts: list[str]) -> list[dict[str, float]]:
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
        binoculars_scores = binoculars_scores.tolist()
        binoculars_scores_list = [{"Binocular-Score": score_i} for score_i in binoculars_scores]
        return binoculars_scores_list


class E5Lora(DetectorABC):
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(
            AutoTokenizer.from_pretrained(
                "MayZhou/e5-small-lora-ai-generated-detector"
            ),
            device=device,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "MayZhou/e5-small-lora-ai-generated-detector"
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
        encoding = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        outputs = self.model(**encoding)
        output_probs = F.log_softmax(outputs.logits, -1)[:, 1].exp().tolist()
        prob_list = [{"LLM": prob, "Human": 1 - prob} for prob in output_probs]
        return prob_list



class MetricsCalculator(DetectorABC):
    def __init__(
            self,
            model: str,
            batch_size: int = 128,
            max_length: int = 512,
            device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(AutoTokenizer.from_pretrained(model), device=device)
        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.model.eval()
        self.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self.tokenizer.pad_token_id = self.pad_token_id
        self.batch_size = batch_size
        self.max_length = max_length

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: PreTrainedModel):
        self._model = model
        self._requires_position_ids = "position_ids" in set(
            inspect.signature(self.model.forward).parameters.keys()
        )
        self._model.to(self.device)

    def to(self, device: str | torch.device):
        self.device = torch.device(device)
        self._model.to(self.device)
        return self

    def tokenize(self, texts: list[str]) -> BatchEncoding:
        # We can just pad to the right (not left), because we do not need to generate anything.
        # Padding left would work too (given correct attention mask and position IDs),
        # but slicing the outputs is a little bit more complicated.
        return self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
        )

    def process(self, inputs: dict):
        return {
            "prediction": self.predict(
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                }
            )
        }

    def predict(self, inputs: dict):
        """
        Calculate metrics for the given pre-processed dataset.

        Args:
            dataset (list[Sample]): A sequence of pre-processed documents to be processed.
            pad_token_id (int): The token ID to use for padding.

        Returns:
            list[Metrics]: A list of calculated metrics.
        """
        encoding = self.tokenizer.pad(inputs, return_tensors="pt").to(self.device)
        return self._process_batch(
            encoding.input_ids, encoding.attention_mask, self.pad_token_id
        )


    def _process_batch(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            pad_token_id: int,
    ) -> list[float]:
        """
        Process the a batch of input sequences and calculate transition scores.
        Runs a forward pass on the model and extracts the top k probabilities.

        Args:
            input_ids (torch.Tensor): A list of input sequences, each represented as a list of token IDs.
            attention_mask (torch.Tensor): A list of attention masks for each input sequence.
            pad_token_id (int): The token ID that has been used for padding.

        Returns:
            list[TransitionScores]: A list output probability tuples.
        """
        (
            batch_probabilities,
            batch_log_probabilities,
        ) = self._forward(input_ids, attention_mask)

        results = []
        for (
                target_ids,
                probabilities,
                log_probabilities,
        ) in zip(
            input_ids.to(self.device),
            batch_probabilities,
            batch_log_probabilities,
        ):
            # Truncate the sequence to the last non-pad token
            labels = target_ids[1:].view(-1, 1)
            labels = labels[: labels.ne(pad_token_id).sum()]
            labels = labels.to(log_probabilities.device)

            probabilities: torch.Tensor = probabilities[: labels.size(0)]
            log_probabilities: torch.Tensor = log_probabilities[: labels.size(0)]

            log_likelihood = log_probabilities.gather(-1, labels).squeeze(-1)

            # Get target probabilities and ranks
            _, sorted_indices = torch.sort(probabilities, descending=True)
            _, target_ranks = torch.where(sorted_indices.eq(labels))

            score = self._calculate_score(
                probabilities, log_probabilities, log_likelihood, target_ranks
            )

            results.append(score)

        return results

    @torch.no_grad()
    def _forward(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Create `position_ids` on the fly, if required
        # Source: https://github.com/huggingface/transformers/blob/v4.48.1/src/transformers/generation/utils.py#L414
        position_ids = None
        if self._requires_position_ids:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.to(self.device)

        outputs = self._model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            position_ids=position_ids,
        )

        probabilities: torch.Tensor = outputs.logits.softmax(-1)
        log_probabilities: torch.Tensor = outputs.logits.log_softmax(-1)

        return (
            probabilities,
            log_probabilities,
        )

    @abstractmethod
    def _calculate_score(
            self,
            probabilities: torch.Tensor,
            log_probabilities: torch.Tensor,
            log_likelihoods: torch.Tensor,
            target_ranks: torch.Tensor,
            device: torch.device | None = None,
    ) -> float: ...

    @torch.inference_mode()
    def process_texts(self, texts: list[str]):
        encoding = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        output = self._process_batch(
            encoding.input_ids,
            encoding.attention_mask,
            self.pad_token_id,
        )
        return output


class DetectLLM_LRR(MetricsCalculator):
    @torch.inference_mode()
    def _calculate_score(
            self,
            _probabilities: torch.Tensor,
            _log_probabilities: torch.Tensor,
            log_likelihoods: torch.Tensor,
            target_ranks: torch.Tensor,
            device: torch.device | None = None,
    ) -> float:
        """Implements the DetectLLM Log-Likelihood Log-Rank Ratio.

        Args:
            log_likelihoods (torch.Tensor): A tensor of log probabilities for each target token.
            target_ranks (torch.Tensor): A tensor of ranks for each target token.
            device (torch.device, optional): Device to run the calculations on. Defaults to None.

        Returns:
            float: The calculated log-likelihood log-rank ratio.

        Source:
            - Paper: https://aclanthology.org/2023.findings-emnlp.827.pdf
            - GitHub: https://github.com/mbzuai-nlp/DetectLLM
            - Implementation:
                - https://github.com/mbzuai-nlp/DetectLLM/blob/main/baselines/all_baselines.py#L35:L42
                - https://github.com/mbzuai-nlp/DetectLLM/blob/main/baselines/all_baselines.py#L94:L100
        """
        device = device or self.device
        return (
            -torch.div(
                log_likelihoods.to(device).sum(),
                target_ranks.to(device).log1p().sum(),
            )
            .cpu()
            .item()
        )

    @torch.inference_mode()
    def process_texts(self, texts: list[str]):
        encoding = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        output = self._process_batch(
            encoding.input_ids,
            encoding.attention_mask,
            self.pad_token_id,
        )
        output_list = [{"DetectLLM-LRR": score_i} for score_i in output]
        return output_list


class FastDetectGPT(MetricsCalculator):
    """Fast-DetectGPT analytic criterion using the **same model** for both reference and scoring."""

    @torch.inference_mode()
    def _calculate_score(
            self,
            probabilities: torch.Tensor,
            log_probabilities: torch.Tensor,
            log_likelihoods: torch.Tensor,
            _target_ranks: torch.Tensor,
            device: torch.device | None = None,
    ) -> float:
        """Implements the Fast-DetectGPT analytic criterion.
        Here, we use the notation from the paper, instead of the implementation (where variables are named `probs_ref`, `mean_ref`, `var_ref`, etc.).

        Source:
            - Paper: https://arxiv.org/abs/2310.05130
            - GitHub: https://github.com/baoguangsheng/fast-detect-gpt
            - Implementation: https://github.com/baoguangsheng/fast-detect-gpt/blob/main/scripts/fast_detect_gpt.py#L52:L70
        """
        device = device or self.device

        expectation = (probabilities.to(device) * log_probabilities.to(device)).sum(-1)
        variance = (
                           probabilities.to(device) * log_probabilities.to(device).square()
                   ).sum(-1) - expectation.square()

        fast_detect_gpt = (
                                  log_likelihoods.to(device).sum(-1) - expectation.sum(-1)
                          ) / variance.sum(-1).sqrt()

        return fast_detect_gpt.cpu().item()

    @torch.inference_mode()
    def process_texts(self, texts: list[str]):
        encoding = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        output = self._process_batch(
            encoding.input_ids,
            encoding.attention_mask,
            self.pad_token_id,
        )
        output_list = [{"Fast-DetectGPT": score_i} for score_i in output]
        return output_list


class FastDetectGPTwithScoring(FastDetectGPT):
    """Fast-DetectGPT analytic criterion using **different models** for reference and scoring."""

    def __init__(
            self,
            reference_model: str,
            scoring_model: str,
            batch_size: int = 128,
            max_length: int = 512,
            device_1: str | torch.device = DEVICE_1,
            device_2: str | torch.device = DEVICE_2,
    ):
        vocab_s = set(AutoTokenizer.from_pretrained(scoring_model).get_vocab().keys())
        vocab_r = set(AutoTokenizer.from_pretrained(reference_model).get_vocab().keys())
        # Check if:
        assert (
            # both vocabularies are equal or
                vocab_r == vocab_s
                # either vocabulary is a superset of the other (-> empty difference)
                or not vocab_r.difference(vocab_s)
                or not vocab_s.difference(vocab_r)
        ), (
            "The tokenizer vocabularies of the reference model and scoring model must match!"
        )
        super().__init__(reference_model, batch_size, max_length, device_1)
        self.device_2 = torch.device(device_2 or self.device)
        self.scoring_model = AutoModelForCausalLM.from_pretrained(scoring_model)
        self.scoring_model.eval()
        self.vocab_size = min(len(vocab_s), len(vocab_r))

    @property
    def scoring_model(self):
        return self._scoring_model

    @scoring_model.setter
    def scoring_model(self, model: PreTrainedModel):
        self._scoring_model = model
        self._scoring_requires_position_ids = "position_ids" in set(
            inspect.signature(self.model.forward).parameters.keys()
        )
        self._scoring_model.to(self.device_2)

    def to(
            self, device: str | torch.device, device_2: str | torch.device | None = None
    ):
        device_2 = device_2 or device
        self.device = torch.device(device)
        self.device_2 = torch.device(device_2)
        self.model.to(self.device)
        self.scoring_model.to(self.device_2)
        return self

    @torch.inference_mode()
    def _forward(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Create `position_ids` on the fly, if required
        # Source: https://github.com/huggingface/transformers/blob/v4.48.1/src/transformers/generation/utils.py#L414
        position_ids = None
        if self._requires_position_ids or self._scoring_requires_position_ids:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids

        reference_probs: torch.Tensor = (
            self._model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                position_ids=position_ids is not None and position_ids.to(self.device),
            )
            .logits[:, :, : self.vocab_size]
            .softmax(-1)
            .to(self.device_2)
        )

        scoring_log_probs: torch.Tensor = (
            self._scoring_model(
                input_ids=input_ids.to(self.device_2),
                attention_mask=attention_mask.to(self.device_2),
                position_ids=position_ids is not None and position_ids.to(self.device_2),
            )
            .logits[:, :, : self.vocab_size]
            .log_softmax(-1)
            .to(self.device_2)
        )

        return (
            reference_probs,
            scoring_log_probs,
        )

    @torch.inference_mode()
    def process_texts(self, texts: list[str]):
        encoding = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        output = self._process_batch(
            encoding.input_ids,
            encoding.attention_mask,
            self.pad_token_id,
        )
        output_list = [{"Fast-DetectGPTwithScoring": score_i} for score_i in output]
        return output_list





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


