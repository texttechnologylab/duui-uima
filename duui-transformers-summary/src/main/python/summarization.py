from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re
from mdmls import Summarizer

class Summarization:
    def __init__(self, model_name, device='cuda:0'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.prefix = "summarize: "

    def summarize(self, text, sum_len=96):
        with torch.no_grad():
            inputs = self.tokenizer(self.prefix + text, max_length=512, truncation=True, padding="max_length", return_tensors='pt').to(self.device)
            preds = self.model.generate(**inputs, max_length=sum_len, min_length=30)
            decoded_predictions = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            return decoded_predictions[0]


class MDMLSummarization:
    def __init__(self, device=0):
        self.device = device
        self.summarizer = Summarizer(device=device)

    def summarize(self, text, sum_len):
        if sum_len < 200:
            min_len = sum_len-30
            if min_len < 0:
                min_len = 0
        else:
            min_len = 200
        output = self.summarizer(text, max_length=sum_len, min_length=min_len)
        output = output.split("\n")
        if len(output) > 1:
            output = output[1]
        else:
            output = output[0]
        return output


class MT5Summarization:
    def __init__(self, model_name, device='cuda:0'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    def summarize(self, text, sum_len=84):
        WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
        input_ids = self.tokenizer(
            [WHITESPACE_HANDLER(text)],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )["input_ids"]
        output_ids = self.model.generate(
            input_ids=input_ids,
            max_length=sum_len,
            no_repeat_ngram_size=2,
            num_beams=4
        )[0]
        summary = self.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        return summary


if __name__ == '__main__':
    text = """The Apollo program, also known as Project Apollo, was the third United States human spaceflight program carried out by the National Aeronautics and Space Administration (NASA), which accomplished landing the first humans on the Moon from 1969 to 1972."""
    model_i = "csebuetnlp/mT5_multilingual_XLSum"
    device_i = "cuda:0" if torch.cuda.is_available() else "cpu"
    # summarizer = Summarization(model_i, device_i)
    # print(summarizer.summarize(text))
    # mdml_summarizer = MDMLSummarization(-1)
    # print(mdml_summarizer.summarize(text))
    mt5_summarizer = MT5Summarization(model_i, device_i)
    print(mt5_summarizer.summarize(text))