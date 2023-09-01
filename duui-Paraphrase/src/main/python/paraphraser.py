from abc import ABC, abstractmethod
from typing import List, Callable, Optional
import torch
from transformers import (
    PegasusTokenizer,
    PegasusForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BartForConditionalGeneration,
    BartTokenizer
)
from parrot import Parrot


def override(func: Callable) -> Callable: return func


class Paraphraser(ABC):
    def __init__(self, cuda: bool = False, gpu_id: int = 0):
        """
        Specifying device.
        :param cuda:
        :param gpu_id:
        """
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() and cuda else "cpu")

    @abstractmethod
    def generate(self, input_sequence: str, **kwargs) -> List[str]:
        """
        Abstract Function for creating a response using gpu or cpu.
        :param input_sequence:
        :return:
        """

    def __call__(self, input_sequence: str, **kwargs) -> List[str]:
        return self.generate(input_sequence, **kwargs)


class PegasusBase(Paraphraser):
    def __init__(self,
                 model_name: str,
                 tokenizer_name: str,
                 cuda: bool = False,
                 gpu_id: int = 0,
                 **kwargs):
        """
        Initiating Pegasus-Model.
        :param model_name:
        :param tokenizer_name:
        :param cuda:
        """
        super().__init__(cuda, gpu_id)
        self.tokenizer = PegasusTokenizer.from_pretrained(tokenizer_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @override
    def generate(self,
                 input_sequence: str,
                 num_return_sequences: int = 1,
                 num_beams: int = 10,
                 **kwargs) -> List[str]:
        """
        Implementation of the generate method. Number of output sequences and search beams can be specified.
        :param num_beams:
        :param num_return_sequences:
        :param input_sequence:
        :return:
        """
        batch = self.tokenizer([input_sequence],
                               truncation=True,
                               padding='longest',
                               max_length=512,
                               return_tensors="pt").to(self.device)
        with torch.no_grad():
            translated = self.model.generate(**batch,
                                             max_length=512,
                                             num_beams=num_beams,
                                             num_return_sequences=num_return_sequences,
                                             temperature=1.5)
        tgt_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text


class T5Base(Paraphraser):
    def __init__(self,
                 model_name: str,
                 tokenizer_name: str,
                 cuda: bool = False,
                 gpu_id: int = 0,
                 **kwargs):
        """
        Initiating T5-Model.
        :param model_name:
        :param tokenizer_name:
        :param cuda:
        """
        super().__init__(cuda, gpu_id)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @override
    def generate(self,
                 input_sequence: str,
                 num_return_sequences: int = 1,
                 num_beams: int = 5,
                 num_beam_groups: int = 5,
                 repetition_penalty: float = 10.0,
                 diversity_penalty: float = 3.0,
                 no_repeat_ngram_size: int = 2,
                 temperature: float = 0.7,
                 max_length: int = 128,
                 **kwargs
                 ) -> List[str]:
        """
        Implementation of the generate method. Number of output sequences and search beams can be specified +
        other params.
        :param input_sequence:
        :param num_return_sequences:
        :param num_beams:
        :param num_beam_groups:
        :param repetition_penalty:
        :param diversity_penalty:
        :param no_repeat_ngram_size:
        :param temperature:
        :param max_length:
        :return:
        """

        batch = self.tokenizer(f'paraphrase: {input_sequence}',
                               return_tensors="pt",
                               padding="longest",
                               max_length=max_length,
                               truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**batch,
                                          temperature=temperature,
                                          repetition_penalty=repetition_penalty,
                                          num_return_sequences=num_return_sequences,
                                          no_repeat_ngram_size=no_repeat_ngram_size,
                                          num_beams=num_beams,
                                          num_beam_groups=num_beam_groups,
                                          max_length=max_length,
                                          diversity_penalty=diversity_penalty)

        res = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return res


class T5BaseCustom(Paraphraser):
    def __init__(self,
                 model_name: str,
                 tokenizer_name: str,
                 cuda: bool = False,
                 gpu_id: int = 0,
                 **kwargs):
        """
        Initiating T5-Model.
        :param model_name:
        :param tokenizer_name:
        :param cuda:
        """
        super().__init__(cuda, gpu_id)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @override
    def generate(self,
                 input_sequence: str,
                 num_return_sequences: int = 1,
                 num_beams: int = 5,
                 num_beam_groups: int = 5,
                 repetition_penalty: float = 10.0,
                 diversity_penalty: float = 3.0,
                 no_repeat_ngram_size: int = 2,
                 temperature: float = 0.7,
                 max_length: int = 128,
                 **kwargs
                 ) -> List[str]:
        """
        Implementation of the generate method. Number of output sequences and search beams can be specified +
        other params.
        :param input_sequence:
        :param num_return_sequences:
        :param num_beams:
        :param num_beam_groups:
        :param repetition_penalty:
        :param diversity_penalty:
        :param no_repeat_ngram_size:
        :param temperature:
        :param max_length:
        :return:
        """

        batch = self.tokenizer(input_sequence,
                               return_tensors="pt",
                               padding="longest",
                               max_length=max_length,
                               truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**batch,
                                          temperature=temperature,
                                          repetition_penalty=repetition_penalty,
                                          num_return_sequences=num_return_sequences,
                                          no_repeat_ngram_size=no_repeat_ngram_size,
                                          num_beams=num_beams,
                                          num_beam_groups=num_beam_groups,
                                          max_length=max_length,
                                          diversity_penalty=diversity_penalty)

        res = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return res


class BartBase(Paraphraser):
    def __init__(self,
                 model_name: str,
                 tokenizer_name: str,
                 cuda: bool = False,
                 gpu_id: int = 0,
                 **kwargs):
        """
        Initiating Bart-Model.
        :param model_name:
        :param tokenizer_name:
        :param cuda:
        """
        super().__init__(cuda, gpu_id)
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @override
    def generate(self,
                 input_sequence: str,
                 **kwargs) -> List[str]:
        """
        Implementation of the generate method..
        :param num_beams:
        :param num_return_sequences:
        :param input_sequence:
        :return:
        """
        batch = self.tokenizer(input_sequence, return_tensors='pt').to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(batch['input_ids'])
        generated_sentence = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_sentence


class ParrotBase(Paraphraser):
    def __init__(self,
                 model_name: str,
                 cuda: bool = False,
                 gpu_id: int = 0,
                 **kwargs):
        """
        Initiating Parrot(t5-NLU)-Model.
        :param model_name:
        :param cuda:
        """
        super().__init__(cuda, gpu_id)
        self.cuda = cuda
        self.model = Parrot(model_tag=model_name, use_gpu=self.cuda)

    @override
    def generate(self,
                 input_sequence: str,
                 num_return_sequences: int = 1,
                 **kwargs) -> List[str]:
        """
        Parrot paraphraser response wrapper. Sometimes not num_return_sequences get generated, but a lower number.
        :param input_sequence:
        :param num_return_sequences:
        :return:
        """
        para_seqs = self.model.augment(input_phrase=input_sequence,
                                       use_gpu=self.cuda,
                                       max_length=128,
                                       max_return_phrases=10,
                                       adequacy_threshold=0.9,
                                       fluency_threshold=0.9)
        para_seqs.sort(key=lambda x: x[1], reverse=True)
        return [para_seq[0] for para_seq in para_seqs][:num_return_sequences]


class AutoParaphraser(ABC):
    def __new__(cls,
                model_name: str,
                tokenizer_name: Optional[str] = None,
                cuda: bool = False,
                gpu_id: int = 0):
        """
        Initializing a Paraphraser based on its model name.
        :param model_name:
        :param tokenizer_name:
        :param cuda:
        :param gpu_id:
        """
        if tokenizer_name is None:
            tokenizer_name = model_name
        inp = locals()
        linked_models = {"tuner007/pegasus_paraphrase": PegasusBase,
                         "humarin/chatgpt_paraphraser_on_T5_base": T5Base,
                         "eugenesiow/bart-paraphrase": BartBase,
                         "prithivida/parrot_paraphraser_on_T5": ParrotBase,
                         "Lelon/t5-german-paraphraser-small": T5BaseCustom,
                         "Lelon/t5-german-paraphraser-large": T5BaseCustom}
        return linked_models[model_name](**inp)

    @abstractmethod
    def generate(self, input_sequence: str, **kwargs) -> List[str]:
        """
        Abstract Function for creating a response using gpu or cpu.
        :param input_sequence:
        :return:
        """

    def __call__(self, input_sequence: str, **kwargs) -> List[str]:
        return self.generate(input_sequence, **kwargs)


if __name__ == "__main__":
    """
    # PEGASUS
    name = 'tuner007/pegasus_paraphrase'
    pegasus = PegasusBase(name, name, True)

    print(pegasus("This is an easy test sentence, change it as you like!", **{"num_return_sequences":1, "num_beams": 10}))
   
    # T5
    t5 = T5Base("humarin/chatgpt_paraphraser_on_T5_base", "humarin/chatgpt_paraphraser_on_T5_base", True)
    text = 'What are the best places to see in New York?'
    print(t5(text))
    
    # BART
    bart = BartBase('eugenesiow/bart-paraphrase', 'eugenesiow/bart-paraphrase', True)
    print(bart("They were there to enjoy us and they were there to pray for us."))
    """

    # PARROT
    parr = ParrotBase("prithivida/parrot_paraphraser_on_T5", cuda=True)
    print(parr.generate("They were there to enjoy us and they were there to pray for us.", 5))

    # Auto
    parr2 = AutoParaphraser("prithivida/parrot_paraphraser_on_T5", cuda=True)
    print(parr.generate("They were there to enjoy us and they were there to pray for us.", 5))
